"""Kraken `german_print`: `recognize` reads Paddle's crops (condition 3), `segment`
runs `blla` on the whole page (condition 2).

`german_print` was trained on Reichsanzeiger-GT.
"""
import os
import time
import uuid
from pathlib import Path

import torch
from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
from kraken.containers import BBoxLine, Segmentation
from kraken.ketos.util import to_ptl_device
from kraken.lib.ctc_decoder import greedy_decoder
from kraken.lib.util import open_image
from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel

from providers.base import Provider
from providers.config import TOP_K
from providers.schema import Char, Line, Result

REC_DOI = "10.5281/zenodo.10519596"   # german_print, has ſ in its codec
SEG_DOI = "10.5281/zenodo.14602569"   # blla, baseline segmentation

DEVICE = os.environ.get("KRAKEN_DEVICE", "cuda:0")
BATCH = int(os.environ.get("KRAKEN_BATCH", "16"))
WORKERS = int(os.environ.get("KRAKEN_WORKERS", "4"))


def _model_path(doi, filename):
    """htrmopo stores each model under `uuid5(NAMESPACE_DNS, doi)`, so the path
    is the same on any machine after `kraken get <doi>`."""
    path = (Path.home() / ".local/share/htrmopo"
            / str(uuid.uuid5(uuid.NAMESPACE_DNS, doi)) / filename)
    if not path.exists():
        raise FileNotFoundError(f"missing model: {path}; run `kraken get {doi}`")
    return path


def _span(cut):
    """A character's x range. `cuts` comes as an (x0, x1) pair in bbox mode and
    as a 4-point polygon in baseline mode."""
    if cut is None or len(cut) == 0:
        return None, None
    xs = [p[0] for p in cut] if hasattr(cut[0], "__len__") else [cut[0], cut[-1]]
    return float(min(xs)), float(max(xs))


def _frames(probs, codec):
    """`(first_frame, end_frame, candidates)` per character, re-decoded from the
    `(C, W)` softmax; candidates are the per-class maxima over the span, as in
    Calamari's `find_alternatives`."""
    if probs is None:
        return []
    matrix = torch.as_tensor(probs)
    charset = codec.l2c_single           # label -> char; blank is not in there
    out = []
    for label, start, end, _ in greedy_decoder(matrix)[0]:
        # greedy_decoder's end is inclusive; the contract is exclusive
        window = matrix[:, start:end + 1].max(dim=1).values
        ranked = torch.argsort(window, descending=True)[:TOP_K + 2].tolist()
        out.append((int(start), int(end) + 1,
                    [[charset[i], round(float(window[i]), 4)]
                     for i in ranked if i != label and i in charset][:TOP_K]))
    return out


def _line(rec, probs=None, codec=None, image_path=None):
    confidences = [float(c) for c in rec.confidences]
    cuts = list(rec.cuts)
    frames = _frames(probs, codec) if codec is not None else []
    chars = []
    for i, (text, prob) in enumerate(zip(rec.prediction, confidences)):
        x0, x1 = _span(cuts[i] if i < len(cuts) else None)
        t0, t1, alternatives = frames[i] if i < len(frames) else (None, None, [])
        chars.append(Char(text=text, prob=round(prob, 4), x0=x0, x1=x1,
                          t0=t0, t1=t1, alternatives=alternatives))
    return Line(
        text=rec.prediction,
        confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        chars=chars,
        image_path=image_path,
    )


class KrakenProvider(Provider):
    name = "kraken"

    def load(self):
        self.recognizer = RecognitionTaskModel.load_model(
            _model_path(REC_DOI, "german_print.mlmodel"))
        self._segmenter = None
        accelerator, device = to_ptl_device(DEVICE)
        # return_logits fills the record only on the baseline path (see _recognize)
        self.rec_config = RecognitionInferenceConfig(
            accelerator=accelerator, device=device, return_logits=bool(TOP_K),
            batch_size=BATCH, num_line_workers=WORKERS)
        self._accel = (accelerator, device)

    @property
    def segmenter(self):
        """Lazy: a process doing only condition 3 never allocates `blla`."""
        if self._segmenter is None:
            accelerator, device = self._accel
            self._segmenter = SegmentationTaskModel.load_model(
                _model_path(SEG_DOI, "blla.mlmodel"))
            self.seg_config = SegmentationInferenceConfig(
                accelerator=accelerator, device=device, batch_size=BATCH)
        return self._segmenter

    def predict(self, request):
        """`{mode: "recognize", line_paths}` or `{mode: "segment", page_path}`."""
        start = time.perf_counter()
        if request.get("mode", "recognize") == "segment":
            result = self._segment(request["page_path"])
        else:
            result = Result(provider=self.name,
                            lines=[self._recognize(p)
                                   for p in request["line_paths"]])
        result.elapsed = time.perf_counter() - start
        return result

    def _recognize(self, line_path):
        """One crop per call: batching zero-pads lines (lib/vgsl/rpred.py) and
        changes the decoded text, even between widths 12 px apart."""
        image = open_image(line_path)
        bounds = Segmentation(
            type="bbox",
            text_direction=self.rec_config.text_direction,
            imagename=str(line_path),
            script_detection=False,
            lines=[BBoxLine(id=f"_{uuid.uuid4()}",
                            bbox=(0, 0, image.width, image.height))],
        )
        records = list(self.recognizer.predict(
            im=image, segmentation=bounds, config=self.rec_config))
        if not records:
            return Line(text="", image_path=line_path)
        # batch of one, so net.outputs is this line's softmax
        probs = getattr(self.recognizer.net, "outputs", None)
        return _line(records[0], probs=None if probs is None else probs[0],
                     codec=self.recognizer.net.codec, image_path=line_path)

    def _segment(self, page_path):
        image = open_image(page_path)
        bounds = self.segmenter.predict(im=image, config=self.seg_config)
        lines = []
        for i, rec in enumerate(self.recognizer.predict(
                im=image, segmentation=bounds, config=self.rec_config)):
            line = _line(rec, probs=rec.logits, codec=self.recognizer.net.codec)
            source = bounds.lines[i] if i < len(bounds.lines) else None
            line.box = list(source.baseline) if source and source.baseline else None
            lines.append(line)
        return Result(provider=self.name, page_path=str(page_path), lines=lines)

"""Calamari `deep3_antiqua-hist` over Paddle's line crops.

5-fold ensemble with confidence voting; each fold becomes a `variant` of the line.
"""
import os
import time
from pathlib import Path

import numpy as np
from calamari_ocr.ocr.predict.predictor import MultiPredictor, PredictorParams
from PIL import Image

from providers.base import Provider
from providers.config import TOP_K
from providers.schema import Char, Line, Result

MODEL_DIR = Path(__file__).resolve().parent / "models" / "calamari_models_experimental"
MODEL = os.environ.get("CALAMARI_MODEL", "deep3_antiqua-hist")

BATCH = int(os.environ.get("CALAMARI_BATCH", "16"))
PROCS = int(os.environ.get("CALAMARI_PROCS", "4"))
# crops decoded into memory at once; keep it a multiple of BATCH
CHUNK = int(os.environ.get("CALAMARI_CHUNK", str(BATCH * 16)))
GPUS = os.environ.get("CALAMARI_GPU", "0")
# DefaultCTCDecoder replaces 0 with 1e-4 (default_ctc_decoder.py:11); any positive value avoids it
MIN_P = float(os.environ.get("CALAMARI_MIN_P", "1e-12"))


def _checkpoints():
    paths = sorted(str(p) for p in (MODEL_DIR / MODEL).glob("*.ckpt.json"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint in {MODEL_DIR / MODEL}")
    return paths


def _relax_decoder(predictor):
    """Lower the CTC decoder threshold on every fold's post-processor."""
    datas = list(getattr(predictor, "datas", [])) + [predictor.data]
    touched = 0
    for data in datas:
        for processor in getattr(data.params.post_proc, "processors", []):
            params = getattr(processor, "ctc_decoder_params", None)
            if params is not None:
                params.min_p_threshold = MIN_P
                touched += 1
    if not touched:
        raise RuntimeError("no CTCDecoder found in the pipeline")


def _chars(prediction):
    chars = []
    for pos in prediction.positions:
        if not pos.chars:
            continue
        best, rest = pos.chars[0], pos.chars[1:TOP_K + 1]
        chars.append(Char(
            text=best.char,
            prob=round(float(best.probability), 4),
            # global_* is rounded by Calamari (predict/params.py:99); local_* is the frame index
            x0=pos.global_start,
            x1=pos.global_end,
            t0=pos.local_start,
            # local_end is inclusive; the contract is exclusive
            t1=pos.local_end + 1,
            alternatives=[[c.char, round(float(c.probability), 4)] for c in rest],
        ))
    return chars


def _line(prediction, image_path):
    return Line(
        text=prediction.sentence,
        confidence=round(float(prediction.avg_char_probability), 4),
        chars=_chars(prediction),
        image_path=image_path,
    )


class CalamariProvider(Provider):
    name = "calamari"

    def load(self):
        params = PredictorParams(silent=True, progress_bar=False)
        params.pipeline.batch_size = BATCH
        params.pipeline.num_processes = PROCS
        if GPUS:
            params.device.gpus = [int(g) for g in GPUS.split(",")]
        self.predictor = MultiPredictor.from_paths(
            checkpoints=_checkpoints(), predictor_params=params
        )
        _relax_decoder(self.predictor)

    def predict(self, request):
        """`{line_paths: [...]}` -> one `Line` per crop, in the same order."""
        line_paths = request["line_paths"]
        start = time.perf_counter()
        lines = []
        for i in range(0, len(line_paths), CHUNK):
            lines.extend(self._read(line_paths[i:i + CHUNK]))
        return Result(provider=self.name, lines=lines,
                      elapsed=time.perf_counter() - start)

    def _read(self, paths):
        """Decode in chunks to bound memory; the chunk is a multiple of the batch."""
        images = [np.asarray(Image.open(p).convert("L")) for p in paths]
        lines = []
        for sample in self.predictor.predict_raw(images, size=len(images)):
            folds, voted = sample.outputs
            path = paths[len(lines)]
            line = _line(voted, path)
            line.variants = [_line(f.prediction, path) for f in folds]
            lines.append(line)
        return lines

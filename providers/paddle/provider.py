"""PP-OCRv6: `detect` cuts the line crops every engine reads, `recognize` reads
a list of crops in batches. Each model is loaded on first use.
"""
import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
from paddlex import create_model
from paddlex.inference.models.text_recognition.predictor import TextRecRunnerPredictor

from providers.base import Provider
from providers.config import TOP_K
from providers.schema import Char, Line, Result

DET_MODEL = "PP-OCRv6_medium_det"
REC_MODEL = "PP-OCRv6_medium_rec"

# inference directory of a fine-tuned model; empty = official weights
REC_DIR = os.environ.get("PADDLE_REC_DIR") or None
DET_LIMIT = int(os.environ.get("PADDLE_DET_LIMIT", "6000"))
REC_BATCH = int(os.environ.get("PADDLE_REC_BATCH", "128"))
DET_BATCH = int(os.environ.get("PADDLE_DET_BATCH", "1"))
# DB mask expansion factor (paddlex default 2.0), recorded in the output
UNCLIP = float(os.environ.get("PADDLE_UNCLIP", "2.0"))

CTC_BLANK = 0


def _rectify(image, poly):
    """Crop the line polygon and unwarp it into a rectangle. Nothing is
    dropped; the 1 px floor only avoids `warpPerspective` with a zero side."""
    src = np.asarray(poly, dtype=np.float32)
    if src.shape[0] != 4:  # polygon with more vertices: fall back to its bbox
        (x0, y0), (x1, y1) = src.min(0), src.max(0)
        src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    width = max(1, int(max(np.linalg.norm(src[0] - src[1]),
                           np.linalg.norm(src[2] - src[3]))))
    height = max(1, int(max(np.linalg.norm(src[0] - src[3]),
                            np.linalg.norm(src[1] - src[2]))))

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    line = cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(src, dst), (width, height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    return np.ascontiguousarray(line)


def _field(result, key):
    """Read a result field as a list. `or []` is not usable here: several of
    them come back as ndarray, whose truth value raises ValueError."""
    value = result[key] if key in result else None
    return [] if value is None else list(value)


def _top(window, label):
    """The TOP_K most likely candidates, excluding the winner (argpartition, not a full sort)."""
    k = min(TOP_K + 1, window.shape[0])
    top = np.argpartition(window, -k)[-k:]
    top = top[np.argsort(window[top])[::-1]]
    return [i for i in top if i != label][:TOP_K]


def _decode(probs, charset, px_per_frame):
    """Replay Paddle's CTC rule (collapse repeats, drop blank), keeping position
    and the per-class maximum over each character's frame span, as Calamari and
    the Kraken re-decode do."""
    best = probs.argmax(axis=-1)
    chars = []
    start = 0
    for end in range(1, len(best) + 1):
        if end < len(best) and best[end] == best[start]:
            continue
        label = best[start]
        if label != CTC_BLANK:
            window = probs[start:end].max(axis=0)
            chars.append(Char(
                text=charset[label],
                prob=round(float(window[label]), 4),
                x0=start * px_per_frame,
                x1=end * px_per_frame,
                t0=int(start),
                t1=int(end),
                alternatives=[[charset[i], round(float(window[i]), 4)]
                              for i in _top(window, label)],
            ))
        start = end
    return chars


def _key(text, score):
    return text if score is None else (text, round(float(score), 6))


class _Tap:
    """Capture the recognizer softmax, which the public API throws away.

    The predictor only reports one score per line; the CTC frames live in its
    `post_op`. Rows are matched back by `(text, score)` rather than by order,
    because the predictor sorts its batch by aspect ratio internally.
    """

    def __init__(self, predictor):
        self.charset = predictor.post_op.character
        self.by_key = {}
        self._heights = []
        self._shape = None
        self._decoded = []

        read, to_batch, post_op, process = (
            predictor.pre_tfs["Read"], predictor.pre_tfs["ToBatch"],
            predictor.post_op, predictor.process)

        def tap_read(imgs):
            # heights from Read: instances are paths standalone, arrays in the pipeline
            out = read(imgs=imgs)
            self._heights = [np.asarray(im).shape[0] for im in out]
            return out

        def tap_batch(imgs):
            out = to_batch(imgs)
            self._shape = out[0].shape              # (B, C, H, W)
            return out

        def tap_post(pred, **kwargs):
            # decode here and drop the (B, T, C) tensor, over 1 GB at batch 128
            probs = np.asarray(pred[0])
            _, _, height, width = self._shape
            stride = width / probs.shape[1]         # input px per frame
            self._decoded = [
                # scale back to the crop: original height / network height
                _decode(probs[i], self.charset, stride * h / height)
                for i, h in enumerate(self._heights[:probs.shape[0]])
            ]
            del probs
            return post_op(pred, **kwargs)

        def tap_process(batch_data, **kwargs):
            out = process(batch_data, **kwargs)
            for key, chars in zip(map(_key, out["rec_text"], out["rec_score"]),
                                  self._decoded):
                self.by_key.setdefault(key, []).append(chars)
            self._decoded = []
            return out

        predictor.pre_tfs["Read"] = tap_read
        predictor.pre_tfs["ToBatch"] = tap_batch
        predictor.post_op = tap_post
        predictor.process = tap_process

    def reset(self):
        self.by_key.clear()

    def take(self, text, score):
        pending = self.by_key.get(_key(text, score))
        return pending.pop(0) if pending else []


class PaddleProvider(Provider):
    name = "paddle"

    def load(self):
        # both models are lazy
        self._detector = None
        self._recognizer = None
        self.tap = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = create_model(
                DET_MODEL,
                limit_side_len=DET_LIMIT,
                limit_type="max",
                unclip_ratio=UNCLIP,
                # paddlex silently clamps limit_side_len to max_side_limit (default 4000)
                max_side_limit=max(DET_LIMIT, 4000),
            )
        return self._detector

    @property
    def recognizer(self):
        if self._recognizer is None:
            self._recognizer = create_model(REC_MODEL, model_dir=REC_DIR)
            if TOP_K:
                # sub-models exist only after the first prediction
                list(self._recognizer(
                    [np.full((48, 320, 3), 255, dtype=np.uint8)], batch_size=1))
                found = [o for o in gc.get_objects()
                         if isinstance(o, TextRecRunnerPredictor)]
                if len(found) != 1:
                    raise RuntimeError(
                        f"expected 1 TextRecRunnerPredictor, found {len(found)}")
                self.tap = _Tap(found[0])
        return self._recognizer

    def predict(self, request):
        mode = request.get("mode", "detect")
        if mode == "detect":
            return self._detect(request)
        if mode == "recognize":
            return self._recognize(request)
        raise ValueError(f"unknown mode: {mode!r}")

    def _detect(self, request):
        """`{page_path, lines_dir, stem}` -> geometry plus the line crops on
        disk. No recognition here."""
        page_path = request["page_path"]
        lines_dir = request.get("lines_dir")
        stem = request.get("stem") or Path(page_path).stem

        start = time.perf_counter()
        results = list(self.detector([page_path], batch_size=1))
        if not results:
            return Result(provider=self.name, page_path=page_path,
                          elapsed=time.perf_counter() - start,
                          warnings={"unclip": UNCLIP})

        polys = _field(results[0], "dt_polys")
        scores = _field(results[0], "dt_scores")
        scores = (scores + [None] * len(polys))[:len(polys)]

        # reading order (y, x), so the suffix __lNNN is the line index
        boxes = [self._bbox(p) for p in polys]
        order = sorted(range(len(polys)), key=lambda i: (boxes[i][1], boxes[i][0]))
        paths = self._write(page_path, [polys[i] for i in order], lines_dir, stem)

        lines = [
            Line(
                text="",
                confidence=None if scores[i] is None else round(float(scores[i]), 4),
                image_path=paths[j],
                box=boxes[i],
            )
            for j, i in enumerate(order)
        ]
        return Result(provider=self.name, page_path=page_path, lines=lines,
                      elapsed=time.perf_counter() - start,
                      warnings={"unclip": UNCLIP})

    def _recognize(self, request):
        """`{line_paths: [...]}` -> one `Line` per crop, in the same order."""
        line_paths = request["line_paths"]
        start = time.perf_counter()
        if self.tap:
            self.tap.reset()

        lines = []
        for path, result in zip(line_paths,
                                self.recognizer(line_paths, batch_size=REC_BATCH)):
            text = result["rec_text"]
            score = result["rec_score"]
            lines.append(Line(
                text=text,
                confidence=None if score is None else round(float(score), 4),
                chars=self.tap.take(text, score) if self.tap else [],
                image_path=path,
            ))
        return Result(provider=self.name, lines=lines,
                      elapsed=time.perf_counter() - start)

    @staticmethod
    def _bbox(poly):
        p = np.asarray(poly, dtype=float)
        return [float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max())]

    @staticmethod
    def _write(page_path, polys, lines_dir, stem):
        """One PNG per line; the page is decoded once."""
        if not lines_dir:
            return [None] * len(polys)
        Path(lines_dir).mkdir(parents=True, exist_ok=True)
        page = cv2.imread(page_path)
        paths = []
        for i, poly in enumerate(polys):
            path = f"{lines_dir}/{stem}__l{i:03d}.png"
            cv2.imwrite(path, _rectify(page, poly))
            paths.append(path)
        return paths

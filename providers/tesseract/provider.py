"""Tesseract 5 with the language models it is given: `recognize` reads line crops
(`--psm 7`), `segment` reads whole pages with its own layout analysis (`--psm 1`).

No per-character probabilities, alternatives or frame indices. Spaces are
reinserted into `chars` so that `''.join(c.text for c in chars) == text`.
"""
import os
import time

import pytesseract
from PIL import Image

from providers.base import Provider
from providers.schema import Char, Line, Result

# set TESSERACT_LANG per corpus; eng on Portuguese print gives plausible garbage
LANG = os.environ.get("TESSERACT_LANG", "eng")
# 7 = single text line
PSM = os.environ.get("TESSERACT_PSM", "7")
# 1 = automatic segmentation with OSD (needs the osd data); 3 is the same without it
PSM_PAGE = os.environ.get("TESSERACT_PSM_PAGE", "1")
OEM = os.environ.get("TESSERACT_OEM", "3")


class TesseractProvider(Provider):
    name = "tesseract"

    def load(self):
        """Check the language packs at start-up; the binary loads its model per call."""
        self.config = f"--oem {OEM} --psm {PSM}"
        self.config_page = f"--oem {OEM} --psm {PSM_PAGE}"
        version = pytesseract.get_tesseract_version()
        available = set(pytesseract.get_languages(config=""))
        missing = [l for l in LANG.split("+") if l not in available]
        if missing:
            raise RuntimeError(
                f"tesseract {version} has no language pack for {missing}: "
                f"install it (`apt install tesseract-ocr-{missing[0]}`) or set "
                f"TESSERACT_LANG. Available: {sorted(available)}")

    def predict(self, request):
        """`{mode: "recognize", line_paths}` or `{mode: "segment", page_path}`."""
        start = time.perf_counter()
        if request.get("mode") == "segment":
            result = self._segment(request["page_path"])
        else:
            result = Result(provider=self.name,
                            lines=[self._read(p) for p in request["line_paths"]])
        result.elapsed = time.perf_counter() - start
        return result

    def _segment(self, page_path):
        """Whole page with Tesseract's own layout; one `Line` per output line, no `chars`."""
        with Image.open(page_path) as im:
            text = pytesseract.image_to_string(
                im.convert("RGB"), lang=LANG, config=self.config_page)
        lines = [Line(text=t) for t in text.splitlines() if t.strip()]
        return Result(provider=self.name, page_path=str(page_path), lines=lines)

    def _read(self, path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size
            text = pytesseract.image_to_string(
                im, lang=LANG, config=self.config)
            boxes = pytesseract.image_to_boxes(
                im, lang=LANG, config=self.config)
            data = pytesseract.image_to_data(
                im, lang=LANG, config=self.config,
                output_type=pytesseract.Output.DICT)

        text = text.replace("\n", " ").strip()
        return Line(text=text, confidence=_confidence(data),
                    chars=_chars(text, boxes, height), image_path=str(path))


def _confidence(data):
    """Mean of the per-word confidences, on 0..1. Tesseract marks non-text
    blocks with -1, which would drag the mean down if counted."""
    confs = [float(c) for c in data.get("conf", []) if float(c) >= 0]
    return round(sum(confs) / len(confs) / 100.0, 4) if confs else None


def _chars(text, boxes, height):
    """One `Char` per character of `text`; spaces and unmatched symbols get no
    coordinates. Box origin is bottom left; only x is read."""
    parsed = []
    for row in boxes.splitlines():
        parts = row.split(" ")
        if len(parts) >= 5:
            parsed.append((parts[0], float(parts[1]), float(parts[3])))

    chars, k = [], 0
    for ch in text:
        if ch.isspace():
            chars.append(Char(text=ch))
            continue
        if k < len(parsed) and parsed[k][0] == ch:
            _, x0, x1 = parsed[k]
            chars.append(Char(text=ch, x0=x0, x1=x1))
            k += 1
        else:
            chars.append(Char(text=ch))
    return chars

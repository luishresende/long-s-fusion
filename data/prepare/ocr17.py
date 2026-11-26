#!/usr/bin/env python3
"""OCR17+ (e-ditiones, Gabay/Clérice, 17th-century French). PAGE-XML + PNG
already in the cloned repo (ocr17_raw/Data/<book>/{png,pageXmlTranskribusCorrected}),
same schema as Reichsanzeiger-GT.

Usage: ../../providers/paddle/.venv/bin/python ocr17.py
"""
import csv
import glob
import os
import re
import xml.etree.ElementTree as ET
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fetch import ensure_clone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "ocr17_raw", "Data")

BASE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments")
OUT_DIR = os.path.join(BASE, "datasets", "ocr17")


NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def parse_gold(xml_path):
    root = ET.parse(xml_path).getroot()
    order = [r.get("regionRef") for r in root.findall(".//p:ReadingOrder//p:RegionRefIndexed", NS)]
    regions = {}
    for tr in root.findall(".//p:TextRegion", NS):
        lines = []
        for tl in tr.findall("p:TextLine", NS):
            # text in TextLine/TextEquiv or Word/TextEquiv, varying by page
            words = [u.text for u in tl.findall(".//p:TextEquiv/p:Unicode", NS) if u.text]
            if words:
                lines.append(" ".join(words))
        regions[tr.get("id")] = "\n".join(lines)
    parts = [regions[r] for r in order if r in regions] if order else list(regions.values())
    return "\n".join(p for p in parts if p.strip())


OCR17PLUS_URL = "https://github.com/Heresta/OCR17plus.git"


def acquire():
    """OCR17+ is a different repository from `e-ditiones/OCR17`: the latter is
    the training set, this one is the evaluation ground truth used here."""
    ensure_clone(os.path.join(HERE, "ocr17_raw"), OCR17PLUS_URL)


def main():
    acquire()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for book in sorted(os.listdir(DATA_DIR)):
        png_dir = os.path.join(DATA_DIR, book, "png")
        corrected = glob.glob(os.path.join(DATA_DIR, book, "pageXmlTranskribusCor*ected"))
        if not (os.path.isdir(png_dir) and corrected):
            continue
        xml_dir = corrected[0]
        pngs = {os.path.splitext(f)[0]: f for f in os.listdir(png_dir) if f.endswith(".png")}
        xmls = {}
        for f in os.listdir(xml_dir):
            if not f.endswith(".xml"):
                continue
            stem = os.path.splitext(f)[0]
            if stem not in pngs:
                stem = next((p for p in pngs if stem.endswith(p)), stem)
            xmls[stem] = f
        m = re.match(r"[A-Za-z]+(\d{4})", book)
        date = int(m.group(1)) if m else None
        for stem in sorted(set(pngs) & set(xmls)):
            gold = parse_gold(os.path.join(xml_dir, xmls[stem]))
            if not gold.strip():
                continue
            rows.append({
                "id": stem, "book": book, "date": date,
                "image_path": os.path.relpath(os.path.join(png_dir, pngs[stem]), BASE),
                "gold": gold,
            })
        print(f"  {book}: {len(set(pngs) & set(xmls))} pages")

    if not rows:
        raise SystemExit(f"no page assembled: check the structure of {DATA_DIR}")

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "book", "date", "image_path", "gold"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} pages -> {os.path.join(OUT_DIR, 'manifest.csv')}")


if __name__ == "__main__":
    main()

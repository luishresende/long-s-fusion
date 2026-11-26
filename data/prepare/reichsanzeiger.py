#!/usr/bin/env python3
"""Reichsanzeiger-GT (UB-Mannheim, CC0). Clones the PAGE-XML (Transkribus),
downloads the scans from the UB Mannheim server and builds the manifest,
converting to PNG.

Usage: ../../providers/paddle/.venv/bin/python reichsanzeiger.py
"""
import csv
import os
import xml.etree.ElementTree as ET
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fetch import ensure_clone, ensure_download

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "reichsanzeiger_raw", "data")
XML_DIR = os.path.join(RAW_DIR, "reichsanzeiger-1820-1939", "GT-PAGE")
IMAGES_DIR = os.path.join(RAW_DIR, "images")

BASE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments")
OUT_DIR = os.path.join(BASE, "datasets", "reichsanzeiger")
PAGES_DIR = os.path.join(OUT_DIR, "pages")


NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def parse_gold(xml_path):
    root = ET.parse(xml_path).getroot()
    page = root.find("p:Page", NS)
    order = [r.get("regionRef") for r in root.findall(".//p:ReadingOrder//p:RegionRefIndexed", NS)]
    regions = {}
    for tr in root.findall(".//p:TextRegion", NS):
        lines = []
        for tl in tr.findall("p:TextLine", NS):
            uni = tl.find("p:TextEquiv/p:Unicode", NS)
            if uni is not None and uni.text:
                lines.append(uni.text)
        regions[tr.get("id")] = "\n".join(lines)
    parts = [regions[r] for r in order if r in regions] if order else list(regions.values())
    return page.get("imageFilename"), "\n".join(p for p in parts if p.strip())


REICHSANZEIGER_URL = "https://github.com/UB-Mannheim/reichsanzeiger-gt.git"
IMAGES_URL = ("https://digi.bib.uni-mannheim.de/reichsanzeiger.fcgi"
               "?FIF=/reichsanzeiger/film/")


def acquire():
    ensure_clone(os.path.join(HERE, "reichsanzeiger_raw"), REICHSANZEIGER_URL)
    with open(os.path.join(RAW_DIR, "imageurls.list"), encoding="utf-8") as fh:
        entries = [ln.split(None, 1) for ln in fh.read().splitlines() if ln.strip()]
    os.makedirs(IMAGES_DIR, exist_ok=True)
    missing = [(rel, name.strip()) for rel, name in entries
              if not os.path.exists(os.path.join(IMAGES_DIR, name.strip()))]
    if missing:
        print(f"[fetch] downloading {len(missing)} images of {len(entries)}", flush=True)
    for i, (rel, name) in enumerate(missing, 1):
        ensure_download(os.path.join(IMAGES_DIR, name), IMAGES_URL + rel)
        print(f"[fetch]   {i}/{len(missing)} {name}", flush=True)


def main():
    acquire()
    os.makedirs(PAGES_DIR, exist_ok=True)
    xml_files = sorted(f for f in os.listdir(XML_DIR) if f.endswith(".xml"))
    print(f"{len(xml_files)} XML pages")

    rows = []
    for i, fn in enumerate(xml_files, 1):
        img_filename, gold = parse_gold(os.path.join(XML_DIR, fn))
        if not gold.strip():
            continue
        stem = os.path.splitext(fn)[0]
        png_path = os.path.join(PAGES_DIR, stem + ".png")
        if not os.path.exists(png_path):
            src = os.path.join(IMAGES_DIR, img_filename)
            if not os.path.exists(src):
                print(f"  missing image: {img_filename}")
                continue
            Image.open(src).convert("RGB").save(png_path)
        date = int(stem.split("_")[0])
        rows.append({"id": stem, "date": date, "image_path": os.path.relpath(png_path, BASE), "gold": gold})

    if not rows:
        raise SystemExit("no page assembled: check whether the images were downloaded")

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "date", "image_path", "gold"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} pages -> {os.path.join(OUT_DIR, 'manifest.csv')}")


if __name__ == "__main__":
    main()

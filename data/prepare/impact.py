#!/usr/bin/env python3
"""IMPACT (KB Lab, Netherlands), subset of books from 1630-1796. XML already
extracted into impact_raw/xml/xml/*.xml (PAGE-XML 2010-03-19); images in
impact_raw/tif/ (extracted from the zip, .tif). Converts to PNG and builds
the manifest.

Usage: ../../providers/paddle/.venv/bin/python impact.py
"""
import csv
import os
import xml.etree.ElementTree as ET
import zipfile
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fetch import ensure_download, ensure_unzip

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "impact_raw")
XML_DIR = os.path.join(RAW_DIR, "xml", "xml")
TIF_ZIP = os.path.join(RAW_DIR, "impact_boeken.tif.zip")

BASE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments")
OUT_DIR = os.path.join(BASE, "datasets", "impact")
PAGES_DIR = os.path.join(OUT_DIR, "pages")


NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}


def parse_gold(xml_path):
    root = ET.parse(xml_path).getroot()
    page = root.find("p:Page", NS)
    order = [r.get("regionRef") for r in root.findall(".//p:RegionRefIndexed", NS)]
    regions = {}
    for tr in root.findall(".//p:TextRegion", NS):
        uni = tr.find("p:TextEquiv/p:Unicode", NS)
        regions[tr.get("id")] = (uni.text or "") if uni is not None else ""
    if order:
        parts = [regions[r] for r in order if r in regions]
    else:
        parts = list(regions.values())
    return page.get("imageFilename"), "\n".join(p for p in parts if p.strip())


# https://lab.kb.nl/dataset/ground-truth-impact-project, public domain; TIFF zip 15.1 GB, XML zip 9 MB
IMPACT_TIF_URL = "https://kbresearch.nl/static/data/impact_boeken.tif.zip"
IMPACT_XML_URL = "https://kbresearch.nl/static/data/impact_boeken.xml.zip"


def acquire():
    ensure_download(TIF_ZIP, IMPACT_TIF_URL)
    xml_zip = os.path.join(RAW_DIR, "impact_boeken.xml.zip")
    ensure_download(xml_zip, IMPACT_XML_URL)
    ensure_unzip(xml_zip, os.path.join(RAW_DIR, "xml"), marker=XML_DIR)


def main():
    acquire()
    os.makedirs(PAGES_DIR, exist_ok=True)
    xml_files = sorted(f for f in os.listdir(XML_DIR) if f.endswith(".xml"))
    print(f"{len(xml_files)} XML pages")

    zf = zipfile.ZipFile(TIF_ZIP)
    names_by_base = {os.path.basename(n): n for n in zf.namelist()}

    rows = []
    for i, fn in enumerate(xml_files, 1):
        img_filename, gold = parse_gold(os.path.join(XML_DIR, fn))
        if not gold.strip():
            continue
        stem = os.path.splitext(fn)[0]
        png_path = os.path.join(PAGES_DIR, stem + ".png")
        if not os.path.exists(png_path):
            zname = names_by_base.get(img_filename)
            if zname is None:
                continue
            with zf.open(zname) as fh:
                Image.open(fh).convert("RGB").save(png_path)
        rows.append({"id": stem, "image_path": os.path.relpath(png_path, BASE), "gold": gold})
        if i % 200 == 0:
            print(f"  {i}/{len(xml_files)}")

    if not rows:
        raise SystemExit("no page assembled: check the XML and the TIFF zip")

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "image_path", "gold"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} pages -> {os.path.join(OUT_DIR, 'manifest.csv')}")


if __name__ == "__main__":
    main()

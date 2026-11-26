#!/usr/bin/env python3
"""PORTO (LIACC/FEUP, HuggingFace `LIACC/PORTO`, CC-BY-4.0). The whole
dataset, with no subsampling, including complex_layout (Paddle's detection
already handles it).

Usage: ../../providers/paddle/.venv/bin/python porto.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

BASE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments")
OUT_DIR = os.path.join(BASE, "datasets", "porto")
PAGES_DIR = os.path.join(OUT_DIR, "pages")


def main():
    from datasets import load_dataset

    os.makedirs(PAGES_DIR, exist_ok=True)
    ds = load_dataset("LIACC/PORTO")["train"]
    print(f"{len(ds)} pages in PORTO")

    rows = []
    for i, r in enumerate(ds):
        stem = f"{r['book']}_{r['date']}_{i}"
        img_path = os.path.join(PAGES_DIR, stem + ".png")
        if not os.path.exists(img_path):
            r["img"].convert("RGB").save(img_path)
        rows.append({
            "id": stem, "book": r["book"], "date": r["date"], "corpus": r["corpus"],
            "complex_layout": r["complex_layout"],
            "image_path": os.path.relpath(img_path, BASE),
            "gold": r["transcription"],
            "aadi": r["aadi"], "tesseract": r["tesseract"],
            "transkribus": r["transkribus"], "paddleocr_porto": r["paddleocr"],
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ds)}")

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {os.path.join(OUT_DIR, 'manifest.csv')}")


if __name__ == "__main__":
    main()

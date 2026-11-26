#!/usr/bin/env python3
"""Prepares the Vesalainen et al. (2026) English test set (Zenodo 18597235,
CC-BY-4.0) as a line dataset shaped like `goldseg`; the training split is unused.

Usage: python data/prepare/vesalainen.py [--language english]
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'prepare' / 'vesalainen_raw' / 'test'
EXP = ROOT / 'experiments'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--language', default='english',
                    choices=['english', 'french'])
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    name = a.name or f'vesalainen_{a.language[:2]}'

    csv_path = RAW / f'full_{a.language}_dataset_relative.csv'
    if not csv_path.exists():
        raise SystemExit(
            f'not found: {csv_path}\n'
            f'Zenodo 18597235 carries one file, OCR_18th_century.zip (7.4 GB), '
            f'which contains two nested zips. Only the test one is needed:\n'
            f'  cd {RAW.parent}\n'
            f'  curl -L -o OCR_18th_century.zip '
            f'"https://zenodo.org/records/18597235/files/OCR_18th_century.zip?download=1"\n'
            f'  unzip OCR_18th_century.zip OCR_Testing.zip   # skips the 6.4 GB training half\n'
            f'  unzip OCR_Testing.zip -d {RAW.name}')

    lines = []
    # tab-separated despite the .csv extension, with Windows paths
    for r in csv.DictReader(open(csv_path, encoding='utf-8'), delimiter='\t'):
        rel = r['file_name'].replace('\\', '/')
        img = RAW / rel
        if not img.exists():
            continue
        lines.append({
            'id': Path(rel).stem,
            'book': rel.split('/')[0],       # Hume / Generic
            'corpus': name,
            'image_path': str(img.relative_to(ROOT)),
            'gold': r['text'],
        })
    if not lines:
        raise SystemExit('no image found; incomplete extraction?')

    dest_ds = EXP / 'datasets' / name
    dest_ds.mkdir(parents=True, exist_ok=True)
    fields = ['id', 'book', 'corpus', 'image_path', 'gold']
    with open(dest_ds / 'manifest.csv', 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(lines)

    dest_res = EXP / 'results' / name / 'raw'
    dest_res.mkdir(parents=True, exist_ok=True)
    with open(dest_res / 'detect.jsonl', 'w', encoding='utf-8') as fh:
        for r in lines:
            fh.write(json.dumps({
                'id': r['id'],
                'provider': 'ground truth',
                'lines': [{'image_path': r['image_path'],
                           'box': None, 'index': 0}],
                'elapsed': 0.0,
            }, ensure_ascii=False) + '\n')

    LONG_S = 'ſʃ∫'
    ch = sum(len(r['gold']) for r in lines)
    lg = sum(1 for r in lines for c in r['gold'] if c in LONG_S)
    import collections
    print(f'dataset `{name}`: {len(lines)} crops, '
          f'{len({r["book"] for r in lines})} books')
    for k, n in collections.Counter(r['book'] for r in lines).items():
        print(f'    {k:12} {n:6}')
    print(f'  {ch} characters, {lg} ſ ({1000 * lg / ch:.1f} per thousand)')
    print(f'\nnext: python inference_pipeline/recognize.py --dataset {name} '
          f'--engine paddle   (and calamari, kraken)')


if __name__ == '__main__':
    main()

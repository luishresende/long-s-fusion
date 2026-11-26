#!/usr/bin/env python3
"""The corpus table and the training split, regenerated from the data.

Reports the evaluation sets, with units and crops, and the train/validation/test
split of the fine-tuning data with its composition per corpus.

Usage: python analyses/corpus_stats.py
       python analyses/corpus_stats.py --datasets goldseg,ocr17,porto
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
csv.field_size_limit(10 ** 7)

DEFAULT = 'goldseg,ocr17,reichsanzeiger,porto,impact,vesalainen_en'


def crops_of(res):
    det = res / 'raw' / 'detect.jsonl'
    if not det.exists():
        return None
    n = 0
    for line in open(det, encoding='utf-8'):
        try:
            n += len([l for l in json.loads(line).get('lines', [])
                      if l.get('image_path')])
        except json.JSONDecodeError:
            continue
    return n


def evaluation_sets(datasets):
    print('EVALUATION SETS')
    print(f'  {"corpus":18s} {"units":>9s} {"crops":>10s}   note')
    for ds in datasets:
        res = HERE / 'results' / ds
        pages_csv = res / 'pages.csv'
        if not pages_csv.exists():
            print(f'  {ds:18s} {"—":>9s} {"—":>10s}   no pages.csv')
            continue
        pages = list(csv.DictReader(open(pages_csv, encoding='utf-8')))
        crops = crops_of(res)
        note = ''
        if pages and 'corpus' in pages[0]:
            by = collections.Counter(p['corpus'] for p in pages)
            note = ' + '.join(f'{k} {v:,}' for k, v in sorted(by.items()))
        print(f'  {ds:18s} {len(pages):9,d} '
              f'{crops if crops is not None else "—":>10}   {note}')


def training_split():
    base = HERE / 'finetune' / 'train_data'
    if not base.exists():
        print('\nTRAINING SPLIT\n  train_data/ not present on this machine')
        return
    print('\nTRAINING SPLIT')
    total = 0
    seen = {}
    for name in ('train_list.txt', 'val_list.txt', 'test_list.txt'):
        p = base / name
        if not p.exists():
            print(f'  {name:18s} missing')
            continue
        rows = [l for l in open(p, encoding='utf-8') if l.strip()]
        total += len(rows)
        by = collections.Counter()
        for r in rows:
            path = r.split('\t')[0]
            by['ocr17' if 'ocr17' in path else
               'reichsanzeiger' if 'reichsanzeiger' in path else 'other'] += 1
        seen[name] = {r.split('\t')[0] for r in rows}
        comp = ' + '.join(f'{k} {v:,}' for k, v in sorted(by.items()))
        print(f'  {name.replace("_list.txt", ""):18s} {len(rows):9,d}   {comp}')
    print(f'  {"TOTAL":18s} {total:9,d}')

    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = len(seen[a] & seen[b])
            flag = 'OK' if overlap == 0 else f'*** {overlap:,} SHARED ***'
            print(f'  overlap {a.replace("_list.txt","")}/'
                  f'{b.replace("_list.txt","")}: {flag}')
    print('  The test split is the goldseg of the corpus table.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=DEFAULT)
    a = ap.parse_args()
    evaluation_sets([d.strip() for d in a.datasets.split(',')])
    training_split()


if __name__ == '__main__':
    main()

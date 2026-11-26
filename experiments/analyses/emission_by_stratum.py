#!/usr/bin/env python3
"""Glyph emission per gold stratum of `ſ / (ſ + s)`: diplomatic at 0.30 and above,
modernized below 0.02; `--by-date` splits the modernized stratum at 1850.

Usage: python analyses/emission_by_stratum.py --dataset porto --label paddle_ft3
       python analyses/emission_by_stratum.py --dataset porto --label paddle_ft3 --by-date
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon

csv.field_size_limit(10 ** 7)

DIPLOMATIC = 0.30
MODERNIZED = 0.02
MIN_S = 10          # below this the page cannot be classified


def text_by_crop(raw, label):
    out = {}
    paths = list(Path(raw).glob(f'{label}.jsonl')) + \
        sorted(Path(raw).glob(f'{label}.*of*.jsonl'))
    for p in paths:
        for line in open(p, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for ln in rec.get('lines') or []:
                if ln.get('image_path'):
                    out[ln['image_path']] = ln.get('text') or ''
    return out


def stratum(gold, min_s=MIN_S):
    """The paper's three strata, or None when there are too few esses."""
    n_long = sum(1 for c in gold if c == 'ſ')
    n_round = sum(1 for c in gold if c == 's')
    if n_long + n_round < min_s:
        return None, n_long
    frac = n_long / (n_long + n_round)
    if frac >= DIPLOMATIC:
        return 'diplomatic', n_long
    return ('modernized' if frac < MODERNIZED else 'intermediate'), n_long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='porto')
    ap.add_argument('--label', required=True, help='a label in raw/')
    ap.add_argument('--by-date', action='store_true',
                    help='split the modernized stratum at 1850')
    ap.add_argument('--min-s', type=int, default=MIN_S,
                    help='pages with fewer esses are left unclassified; 1 gives the strata of Section 3.1')
    a = ap.parse_args()

    res = HERE.parent / 'results' / a.dataset
    pages = list(csv.DictReader(open(res / 'pages.csv', encoding='utf-8')))
    det = {}
    for line in open(res / 'raw' / 'detect.jsonl', encoding='utf-8'):
        rec = json.loads(line)
        det[rec['id']] = [ln['image_path'] for ln in rec.get('lines', [])
                          if ln.get('image_path')]
    texts = text_by_crop(res / 'raw', a.label)
    if not texts:
        sys.exit(f'label {a.label!r} not found in {res / "raw"}')

    acc = {}
    for p in pages:
        gold = canon.canonicalize(p['gold'], space='page')
        name, n_long = stratum(gold, a.min_s)
        if name is None:
            name = 'unclassifiable'
        elif name == 'modernized' and a.by_date:
            year = re.search(r'(1[6-9]\d\d)', p.get('date') or '')
            year = int(year.group(1)) if year else 0
            name = ('modernized, to 1850' if year and year <= 1850 else
                    'modernized, post-1850' if year else 'modernized, undated')
        hyp = canon.canonicalize(
            '\n'.join(texts.get(c, '') for c in det.get(p['id'], [])),
            space='page')
        emitted = sum(1 for c in hyp if c == 'ſ')
        row = acc.setdefault(name, [0, 0, 0, 0])
        row[0] += 1
        row[1] += n_long
        row[2] += emitted
        row[3] += sum(1 for c in hyp if c == 's')

    print(f'{a.dataset} / {a.label}')
    print(f'{"stratum":24s} {"pages":>8s} {"ſ gold":>9s} {"ſ emitted":>11s} {"ratio":>8s} {"fraction":>9s}')
    order = ['diplomatic', 'intermediate', 'modernized', 'modernized, to 1850',
             'modernized, post-1850', 'modernized, undated', 'unclassifiable']
    tot_gold = tot_emit = tot_pages = 0
    for name in order:
        if name not in acc:
            continue
        n, gold, emitted, round_s = acc[name]
        tot_pages += n
        tot_gold += gold
        tot_emit += emitted
        ratio = f'{emitted / gold:.2f}x' if gold else '—'
        fraction = emitted / max(emitted + round_s, 1)
        print(f'{name:24s} {n:8d} {gold:9d} {emitted:11d} {ratio:>8s} {fraction:9.3f}')
    print(f'{"TOTAL":24s} {tot_pages:8d} {tot_gold:9d} {tot_emit:11d} '
          f'{tot_emit / max(tot_gold, 1):7.2f}x')


if __name__ == '__main__':
    main()

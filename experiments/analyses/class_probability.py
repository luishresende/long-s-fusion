#!/usr/bin/env python3
"""Probability and rank of the `ſ` row over CTC frames, through a tap on `post_op`;
`--gold-only` just counts the `ſ` in the gold.

Usage: python analyses/class_probability.py --gold-only --datasets goldseg,porto,impact
       python analyses/class_probability.py --dataset goldseg --limit 2000
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from fusion import canon
from fusion.paths import to_path

csv.field_size_limit(10 ** 7)

LONG_S_ROW = 315          # dictionary index 314 plus the CTC blank at 0


def count_opportunities(datasets):
    """`ſ` in the gold, which is what an emission would have to match."""
    total = 0
    print(f'{"corpus":18s} {"units":>8s} {"ſ in the gold":>14s}')
    for ds in datasets:
        pages = list(csv.DictReader(
            open(HERE / 'results' / ds / 'pages.csv', encoding='utf-8')))
        n = sum(sum(1 for c in canon.canonicalize(p['gold'], space='page')
                    if c == 'ſ') for p in pages)
        total += n
        print(f'{ds:18s} {len(pages):8,d} {n:14,d}')
    print(f'{"TOTAL":18s} {"":8s} {total:14,d}')
    return total


def probe(dataset, limit, batch):
    """Median probability and rank of the glyph row, over CTC frames."""
    from paddlex import create_model
    from paddlex.inference.models.text_recognition.predictor import (
        TextRecRunnerPredictor)
    import gc
    import os

    model = os.environ.get('PADDLE_REC_MODEL', 'PP-OCRv6_medium_rec')
    rec = create_model(model, model_dir=os.environ.get('PADDLE_REC_DIR'))

    res = HERE / 'results' / dataset
    crops = []
    for line in open(res / 'raw' / 'detect.jsonl', encoding='utf-8'):
        rec_ = json.loads(line)
        crops += [str(to_path(l['image_path'])) for l in rec_.get('lines', [])
                  if l.get('image_path')]
        if limit and len(crops) >= limit:
            break
    crops = crops[:limit] if limit else crops
    print(f'{dataset}: probing {len(crops):,} crops, batch {batch}')

    mass, ranks = [], []

    # sub-models exist only after the first call
    list(rec([np.full((48, 320, 3), 255, dtype=np.uint8)], batch_size=1))

    # tap the inner TextRecRunnerPredictor; setting rec.post_op on the wrapper binds nothing
    found = [o for o in gc.get_objects()
             if isinstance(o, TextRecRunnerPredictor)]
    if len(found) != 1:
        raise RuntimeError(
            f'expected 1 TextRecRunnerPredictor, found {len(found)}')
    predictor = found[0]
    original = predictor.post_op

    def tap(pred, **kwargs):
        probs = np.asarray(pred[0])                 # (B, T, C)
        col = probs[:, :, LONG_S_ROW]
        mass.append(col.reshape(-1).astype(np.float64))
        # rank 1 = the most likely class at that frame
        greater = (probs > col[:, :, None]).sum(axis=2) + 1
        ranks.append(greater.reshape(-1).astype(np.int32))
        del probs
        return original(pred, **kwargs)

    predictor.post_op = tap
    try:
        list(rec(crops, batch_size=batch))
    finally:
        predictor.post_op = original

    if not mass:
        raise RuntimeError('the tap captured no CTC frame')
    m = np.concatenate(mass)
    r = np.concatenate(ranks)
    print(f'\nCTC frames seen: {m.size:,}')
    print(f'  P(ſ) median      {np.median(m):.3e}')
    print(f'  P(ſ) mean        {m.mean():.3e}')
    print(f'  P(ſ) p99         {np.percentile(m, 99):.3e}')
    print(f'  P(ſ) max         {m.max():.3e}')
    print(f'  rank median      {int(np.median(r)):,}  of 18,710')
    print(f'  rank p1 / p99    {int(np.percentile(r, 1)):,} / '
          f'{int(np.percentile(r, 99)):,}')
    print(f'  frames where ſ wins  {(r == 1).sum():,}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold-only', action='store_true',
                    help='count opportunities and stop; no model, no GPU')
    ap.add_argument('--datasets', default='goldseg,porto,impact',
                    help='for --gold-only')
    ap.add_argument('--dataset', default='goldseg', help='for the probe')
    ap.add_argument('--limit', type=int, default=2000,
                    help='crops to probe; 0 for all')
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()

    if a.gold_only:
        count_opportunities([d.strip() for d in a.datasets.split(',')])
        return
    probe(a.dataset, a.limit, a.batch)


if __name__ == '__main__':
    main()

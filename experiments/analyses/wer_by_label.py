#!/usr/bin/env python3
"""WER for labels that `glyph.py` does not report, with the same joining and
canonicalization, micro-averaged over the corpus.

Usage: python analyses/wer_by_label.py --dataset goldseg --corpus ocr17 \
                                       --labels trocr,qwen
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon

csv.field_size_limit(10 ** 7)


def text_by_crop(raw, label):
    """`results/<ds>/raw/<label>.jsonl` -> {crop path: text}."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--corpus', default=None,
                    help='inside goldseg, restrict to ocr17 or reichsanzeiger')
    ap.add_argument('--space', default=None, choices=['line', 'page'])
    ap.add_argument('--labels', required=True, help='comma separated')
    a = ap.parse_args()

    space = a.space or ('page' if a.dataset in ('porto', 'impact', 'ocr17',
                                                'reichsanzeiger') else 'line')
    res = HERE.parent / 'results' / a.dataset
    pages = list(csv.DictReader(open(res / 'pages.csv', encoding='utf-8')))
    if a.corpus:
        pages = [p for p in pages if p.get('corpus') == a.corpus]
        if not pages:
            sys.exit(f'no page with corpus={a.corpus!r}')

    det = {}
    for line in open(res / 'raw' / 'detect.jsonl', encoding='utf-8'):
        rec = json.loads(line)
        det[rec['id']] = [ln['image_path'] for ln in rec.get('lines', [])
                          if ln.get('image_path')]

    print(f'{a.dataset}{"/" + a.corpus if a.corpus else ""}, '
          f'{len(pages)} units, space `{space}`')
    for label in a.labels.split(','):
        label = label.strip()
        texts = text_by_crop(res / 'raw', label)
        if not texts:
            print(f'  {label:14s} MISSING')
            continue
        num = den = 0
        for p in pages:
            hyp = '\n'.join(texts.get(c, '') for c in det.get(p['id'], []))
            gold = canon.canonicalize(p['gold'], space=space).split()
            hyp = canon.canonicalize(hyp, space=space).split()
            num += Levenshtein.distance(gold, hyp)
            den += len(gold)
        print(f'  {label:14s} WER {num / max(den, 1):.4f}')


if __name__ == '__main__':
    main()

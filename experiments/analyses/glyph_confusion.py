#!/usr/bin/env python3
"""What a system writes where the glyph belongs, and what belongs where it writes it.

Both sides go through `canon` as in `glyph.py`.

Usage: python analyses/glyph_confusion.py --dataset goldseg --label paddle
       python analyses/glyph_confusion.py --dataset goldseg --label 1class
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon

csv.field_size_limit(10 ** 7)
LONG_S = 'ſ'


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


def profile(gold, hyp):
    """Both directions in one pass over the Levenshtein opcodes."""
    wrote = collections.Counter()      # what the system put where gold has ſ
    carried = collections.Counter()    # what gold has where the system wrote ſ
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(gold, hyp):
        if tag == 'equal':
            n = i2 - i1
            for k in range(n):
                if gold[i1 + k] == LONG_S:
                    wrote[LONG_S] += 1
                    carried[LONG_S] += 1
        elif tag == 'replace':
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                g, h = gold[i1 + k], hyp[j1 + k]
                if g == LONG_S:
                    wrote[h] += 1
                if h == LONG_S:
                    carried[g] += 1
            for g in gold[i1 + n:i2]:          # gold left over: deleted
                if g == LONG_S:
                    wrote['«deleted»'] += 1
            for h in hyp[j1 + n:j2]:           # hypothesis left over: inserted
                if h == LONG_S:
                    carried['«nothing»'] += 1
        elif tag == 'delete':
            for g in gold[i1:i2]:
                if g == LONG_S:
                    wrote['«deleted»'] += 1
        elif tag == 'insert':
            for h in hyp[j1:j2]:
                if h == LONG_S:
                    carried['«nothing»'] += 1
    return wrote, carried


def show(title, counter, note):
    total = sum(counter.values())
    print(f'\n{title}  (n = {total:,})')
    print(f'  {note}')
    for ch, n in counter.most_common(6):
        name = ch if ch.startswith('«') else repr(ch)
        print(f'    {name:12s} {n:8,d}  {n / max(total, 1):6.1%}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--label', required=True)
    ap.add_argument('--space', default=None, choices=['line', 'page'])
    a = ap.parse_args()

    space = a.space or ('page' if a.dataset in ('porto', 'impact', 'ocr17',
                                                'reichsanzeiger') else 'line')
    res = HERE.parent / 'results' / a.dataset
    pages = list(csv.DictReader(open(res / 'pages.csv', encoding='utf-8')))
    if a.corpus:
        pages = [p for p in pages if p.get('corpus') == a.corpus]
    det = {}
    for line in open(res / 'raw' / 'detect.jsonl', encoding='utf-8'):
        rec = json.loads(line)
        det[rec['id']] = [ln['image_path'] for ln in rec.get('lines', [])
                          if ln.get('image_path')]
    texts = text_by_crop(res / 'raw', a.label)
    if not texts:
        sys.exit(f'label {a.label!r} not found in {res / "raw"}')

    wrote, carried = collections.Counter(), collections.Counter()
    for p in pages:
        gold = canon.canonicalize(p['gold'], space=space)
        hyp = canon.canonicalize(
            '\n'.join(texts.get(c, '') for c in det.get(p['id'], [])),
            space=space)
        w, c = profile(gold, hyp)
        wrote += w
        carried += c

    print(f'{a.dataset}{"/" + a.corpus if a.corpus else ""} / {a.label}, '
          f'{len(pages)} units, space `{space}`')
    show('WHERE THE GOLD CARRIES ſ, THE SYSTEM WROTE', wrote,
         'the `f` row is the confusion this work is about')
    show('WHERE THE SYSTEM WROTE ſ, THE GOLD CARRIES', carried,
         '«nothing» is spurious insertion: the glyph went where no character was')


if __name__ == '__main__':
    main()

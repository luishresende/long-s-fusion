#!/usr/bin/env python3
"""Glyph measures for every system on one corpus, from the texts in `results/<ds>/`:

  emission   `ſ` emitted / `ſ` in the ground truth, no alignment
  recall     hits / gold
  precision  hits / emissions
  F1         harmonic mean of the two
  net        (hits - damages) / gold
  CER        strict and folded

Usage: python analyses/glyph.py --dataset goldseg [--corpus ocr17]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon                                       # noqa: E402
from fusion.common import DEFAULT, Acc, merge_line              # noqa: E402

csv.field_size_limit(10 ** 7)
LONG_S = 'ſ'


class FSPair:
    """Folded reading: positions where the base wrote `f` and the gold has an s;
    a swap to `s` or `ſ` is a hit either way."""

    def __init__(self):
        self.budget = self.hit = self.damage = 0

    def add(self, g, base, h):
        """`g` ground truth, `base` generalist text, `h` system text, all canonicalized."""
        gf = g.replace('ſ', 's')
        bf = base.replace('ſ', 's')
        hf = h.replace('ſ', 's')
        # recoverable positions: the base put `f`, the ground truth has `s`
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(gf, bf):
            if tag == 'replace':
                for a_, b_ in zip(gf[i1:i2], bf[j1:j2]):
                    if a_ == 's' and b_ == 'f':
                        self.budget += 1
        # how much the system recovered: `s` aligned where the base was wrong
        right_h = right_b = 0
        for tag, i1, i2, _, _ in Levenshtein.opcodes(gf, hf):
            if tag == 'equal':
                right_h += gf[i1:i2].count('s')
        for tag, i1, i2, _, _ in Levenshtein.opcodes(gf, bf):
            if tag == 'equal':
                right_b += gf[i1:i2].count('s')
        d = right_h - right_b
        if d >= 0:
            self.hit += d
        else:
            self.damage += -d

    @property
    def recall(self):
        return self.hit / self.budget if self.budget else 0.0

    @property
    def net(self):
        return ((self.hit - self.damage) / self.budget
                if self.budget else 0.0)


class Glyph:
    """Hits, emissions and damage of a system, over the canonicalized text."""

    def __init__(self):
        self.gold = self.emitted = self.hit = 0

    def add(self, g, h):
        self.gold += g.count(LONG_S)
        self.emitted += h.count(LONG_S)
        for tag, i1, i2, _, _ in Levenshtein.opcodes(g, h):
            if tag == 'equal':
                self.hit += g[i1:i2].count(LONG_S)

    @property
    def damage(self):
        return max(0, self.emitted - self.hit)

    @property
    def recall(self):
        return self.hit / self.gold if self.gold else 0.0

    @property
    def precision(self):
        return self.hit / self.emitted if self.emitted else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def net(self):
        """Net fraction of the glyph budget; negative when damage exceeds hits."""
        return (self.hit - self.damage) / self.gold if self.gold else 0.0


def text_per_crop(raw, label):
    """`results/<ds>/raw/<label>.jsonl` -> {crop path: text}."""
    paths = list(Path(raw).glob(f'{label}.jsonl')) + \
        sorted(Path(raw).glob(f'{label}.*of*.jsonl'))
    if not paths:
        return None
    out = {}
    for p in paths:
        for ln in open(p, encoding='utf-8'):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            for l in r.get('lines', []):
                if l.get('image_path'):
                    out[l['image_path']] = l.get('text', '')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--corpus', default=None,
                    help='in goldseg, restricts to ocr17 or reichsanzeiger')
    ap.add_argument('--space', default=None, choices=['line', 'page'])
    ap.add_argument('--systems', default='paddle,calamari,kraken,trocr,qwen,'
                                         'paddle_ft3',
                    help='labels in raw/; the fusions are computed here')
    a = ap.parse_args()

    res = HERE.parent / 'results' / a.dataset
    space = a.space or ('page' if a.dataset in ('porto', 'impact', 'ocr17',
                                                'reichsanzeiger') else 'line')
    pages = list(csv.DictReader(open(res / 'pages.csv', encoding='utf-8')))
    if a.corpus:
        pages = [p for p in pages if p.get('corpus') == a.corpus]
        if not pages:
            sys.exit(f'no page with corpus={a.corpus!r}')

    # page -> crops map, in order
    det = {}
    for ln in open(res / 'raw' / 'detect.jsonl', encoding='utf-8'):
        r = json.loads(ln)
        det[r['id']] = [l['image_path'] for l in r.get('lines', [])
                        if l.get('image_path')]

    available, missing = {}, []
    for label in a.systems.split(','):
        label = label.strip()
        if not label:
            continue
        t = text_per_crop(res / 'raw', label)
        if t:
            available[label] = t
        else:
            missing.append(label)
    if 'paddle' not in available:
        sys.exit('no paddle: there is no base text for the fusion')

    # fusions: the full rule, over the same base text
    for spec in ('calamari', 'kraken'):
        if spec in available:
            available[f'fusion+{spec}'] = {
                c: merge_line(available['paddle'].get(c, ''),
                              available[spec].get(c, ''), rules=DEFAULT)
                for c in available['paddle']}

    print(f'{a.dataset}{"/" + a.corpus if a.corpus else ""}, '
          f'{len(pages)} units, space `{space}`\n')
    print(f'{"system":18s} {"emission":>8s} {"recall":>7s} {"precision":>9s} '
          f'{"F1":>6s} {"net":>7s} | {"CER":>7s} {"ΔCER":>8s} {"CERfold":>7s}'
          f' | {"rec.fold":>8s} {"net.fold":>9s}')
    print('-' * 112)

    base_cer = None
    for label, crop_text in available.items():
        gl = Glyph()
        pair = FSPair()
        strict = Acc(space=space)
        folded = Acc(space=space, fold_s=True)
        for p in pages:
            crops = det.get(p['id'], [])
            h = '\n'.join(crop_text.get(c, '') for c in crops)
            g = p['gold']
            strict.add(g, h)
            folded.add(g, h)
            gc = canon.canonicalize(g, space=space)
            hc = canon.canonicalize(h, space=space)
            gl.add(gc, hc)
            bc = canon.canonicalize(
                '\n'.join(available['paddle'].get(c, '') for c in crops),
                space=space)
            pair.add(gc, bc, hc)
        if label == 'paddle':
            base_cer = strict.cer
        d = ((base_cer - strict.cer) / base_cer * 100) if base_cer else 0.0
        print(f'{label:18s} {gl.emitted / max(gl.gold,1):7.2f}x {gl.recall:7.1%} '
              f'{gl.precision:9.1%} {gl.f1:6.1%} {gl.net:+7.1%} | '
              f'{strict.cer:7.4f} {d:+7.2f}% {folded.cer:7.4f}'
              f' | {pair.recall:8.1%} {pair.net:+9.1%}')
    # systems not run yet print as dashes
    for label in missing:
        print(f'{label:18s} {"—":>8s} {"—":>7s} {"—":>9s} {"—":>6s} {"—":>7s} | '
              f'{"—":>7s} {"—":>8s} {"—":>7s} | {"—":>8s} {"—":>9s}')
    print(f'\n`ſ` in the ground truth: {gl.gold}')


if __name__ == '__main__':
    main()

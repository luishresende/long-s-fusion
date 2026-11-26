#!/usr/bin/env python3
"""Would an arithmetic combination of the two engines restore the glyph?

Rule 1 is symbolic: where the generalist wrote `f` and the specialist read a long
s at the aligned position, the output is `ſ`, and no probability is consulted.
This script measures the alternative on the same positions. For each of them it
combines the two per-character distributions (the emitted character plus the
alternatives each engine reports) by mean, geometric mean and maximum, and
counts which character the combination would choose.

The alignment is the one of the fusion, `Levenshtein.opcodes`, taken over the
characters each engine reports for the line.

When more than one character has the top score the combination does not decide, and the position is counted as `tie`.

Usage: python analyses/rule1_combiners.py --dataset porto --specialist calamari
"""
import argparse
import collections
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion.common import LONG_S

MODES = ('mean', 'product', 'max')


def dist(rec):
    d = {rec['text']: float(rec.get('prob') or 0)}
    for alt, p in rec.get('alternatives') or []:
        d.setdefault(alt, float(p))
    return d


def lines_of(raw, label, keep):
    """crop name -> (string of emitted characters, {index: distribution})."""
    out = {}
    paths = list(raw.glob(f'{label}.jsonl')) + sorted(raw.glob(f'{label}.*of*.jsonl'))
    for p in paths:
        for ln in open(p, encoding='utf-8'):
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            for line in rec.get('lines', []):
                chars = [c for c in line.get('chars') or [] if len(c.get('text', '')) == 1]
                if not chars or not line.get('image_path'):
                    continue
                name = line['image_path'].rsplit('/', 1)[-1]
                out[name] = (''.join(c['text'] for c in chars),
                             {i: dist(c) for i, c in enumerate(chars) if c['text'] in keep})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='porto')
    ap.add_argument('--specialist', default='calamari')
    a = ap.parse_args()

    raw = HERE.parent / 'results' / a.dataset / 'raw'
    base = lines_of(raw, 'paddle', {'f'})
    spec = lines_of(raw, a.specialist, set(LONG_S))
    if not base or not spec:
        sys.exit('missing raw/paddle or raw/' + a.specialist)

    chosen = {m: collections.Counter() for m in MODES}
    n = 0
    for name, (btext, bdist) in base.items():
        if name not in spec:
            continue
        stext, sdist = spec[name]
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(btext, stext):
            if tag != 'replace':
                continue
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if btext[i] != 'f' or stext[j] not in LONG_S:
                    continue
                n += 1
                dp, dc = bdist[i], sdist[j]
                for m in MODES:
                    score = {}
                    for x in set(dp) | set(dc):
                        p, q = dp.get(x, 0.0), dc.get(x, 0.0)
                        score[x] = ((p + q) / 2 if m == 'mean' else
                                    (p * q) ** 0.5 if m == 'product' else max(p, q))
                    top = max(score.values())
                    best = [x for x, v in score.items() if v == top]
                    chosen[m]['tie' if len(best) > 1 else
                              'long s' if best[0] in LONG_S else
                              's' if best[0] == 's' else
                              'f' if best[0] == 'f' else 'other'] += 1

    print(f'{a.dataset} / {a.specialist}: {n:,} positions where rule 1 applies\n')
    for m in MODES:
        row = '  '.join(f'{k} {100 * v / n:5.1f}%' for k, v in chosen[m].most_common())
        print(f'  {m:8} would choose  {row}')


if __name__ == '__main__':
    main()

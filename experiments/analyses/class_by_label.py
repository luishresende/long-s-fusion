#!/usr/bin/env python3
"""Edit operations by error class, for any label in `raw/`.

`class_impact.py` compares the generalist with the fusion. This script applies
the same classes to the output of any system, which is what the paragraph on the
full fine-tune needs: how many confusions among `s`, `f` and the long s remain,
and how many extra characters the system writes.

Usage: python analyses/class_by_label.py --dataset goldseg --labels paddle,paddle_ft3
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from class_impact import count_families                         # noqa: E402
from fusion import canon                                        # noqa: E402
from wer_by_label import text_by_crop                           # noqa: E402

csv.field_size_limit(10 ** 7)
PAGE_SETS = ('porto', 'impact', 'ocr17', 'reichsanzeiger')
GLYPH = ('f↔long s', 'ſ↔s (spelling)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--corpus', default=None,
                    help='inside goldseg, restrict to ocr17 or reichsanzeiger')
    ap.add_argument('--labels', required=True, help='comma separated')
    a = ap.parse_args()

    space = 'page' if a.dataset in PAGE_SETS else 'line'
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

    labels = [x.strip() for x in a.labels.split(',')]
    counts = {}
    for label in labels:
        texts = text_by_crop(res / 'raw', label)
        if not texts:
            sys.exit(f'label {label!r} not found in {res / "raw"}')
        fam = collections.Counter()
        for p in pages:
            gold = canon.canonicalize(p['gold'], space=space)
            hyp = canon.canonicalize(
                '\n'.join(texts.get(c, '') for c in det.get(p['id'], [])), space=space)
            count_families(gold, hyp, fam)
        counts[label] = fam

    print(f'{a.dataset}{"/" + a.corpus if a.corpus else ""}, {len(pages)} units, space `{space}`')
    print(f'{"error class":26s}' + ''.join(f'{x:>14s}' for x in labels))
    classes = sorted({k for fam in counts.values() for k in fam})
    for k in classes:
        print(f'{k:26s}' + ''.join(f'{counts[x][k]:14,d}' for x in labels))
    print(f'{"s, f and long s together":26s}'
          + ''.join(f'{sum(counts[x][k] for k in GLYPH):14,d}' for x in labels))
    print(f'{"total edits":26s}' + ''.join(f'{sum(counts[x].values()):14,d}' for x in labels))


if __name__ == '__main__':
    main()

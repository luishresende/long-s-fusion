#!/usr/bin/env python3
"""CER, WER and glyph emission of any engine output, next to the generalist.

Used for the comparison engines that enter no fusion, such as Tesseract. Each
label in `raw/` is read in the form it was written:

  crops   records carry `image_path` per line. In the line-level sets the crop
          is the unit. In the page-level sets the page is rebuilt by joining
          its crops in the order of `raw/detect.jsonl`, matched by file name, so
          that outputs produced under another path prefix still line up.
  page    records carry the page `id` and the lines the engine found itself.

The ground truth and the PP-OCRv6 column come from `pages.csv`, and every
comparison goes through `fusion.canon`, strict reading, micro-averaged, with the
`space` of the protocol: `line` for the line-level sets, `page` otherwise.

Usage: python analyses/engine_baseline.py --dataset porto \\
           --labels tesseract_best,tesseract_best_page
       python analyses/engine_baseline.py --dataset goldseg --corpus ocr17 \\
           --labels tesseract_best
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon
from fusion.common import LONG_S, cer_wer

csv.field_size_limit(10 ** 9)
LINE_SETS = ('goldseg', 'vesalainen_en')


def records(raw, label):
    paths = list(raw.glob(f'{label}.jsonl')) + sorted(raw.glob(f'{label}.*of*.jsonl'))
    for p in paths:
        for ln in open(p, encoding='utf-8'):
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def read_label(raw, label, line_level):
    """label -> ({unit id: text}, form), form being 'crops' or 'page'."""
    by_page, by_crop = {}, {}
    for rec in records(raw, label):
        lines = rec.get('lines') or []
        if 'id' in rec and not any(l.get('image_path') for l in lines):
            by_page[rec['id']] = '\n'.join(l.get('text') or '' for l in lines)
            continue
        for l in lines:
            if l.get('image_path'):
                by_crop[Path(l['image_path']).name] = l.get('text') or ''
    if not by_page and not by_crop:
        return None, None
    if not by_crop:
        return by_page, 'page'
    if line_level:
        return {Path(name).stem: t for name, t in by_crop.items()}, 'crops'
    out = {}
    for rec in records(raw, 'detect'):
        names = [Path(l['image_path']).name for l in rec.get('lines', [])
                 if l.get('image_path')]
        out[rec['id']] = '\n'.join(by_crop.get(n, '') for n in names)
    return out, 'crops'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--labels', required=True, help='comma-separated labels in raw/')
    ap.add_argument('--corpus', default=None,
                    help='in goldseg, restricts to ocr17 or reichsanzeiger')
    a = ap.parse_args()

    res = HERE.parent / 'results' / a.dataset
    line_level = a.dataset in LINE_SETS
    space = 'line' if line_level else 'page'
    pages = list(csv.DictReader(open(res / 'pages.csv', encoding='utf-8')))
    if a.corpus:
        pages = [p for p in pages if p.get('corpus') == a.corpus]
    if not pages:
        sys.exit('no unit to evaluate')

    systems = {'paddle (pages.csv)': ({p['id']: p.get('paddle') or '' for p in pages}, 'column')}
    for label in a.labels.split(','):
        text, form = read_label(res / 'raw', label.strip(), line_level)
        if text is None:
            print(f'! no raw/{label}*.jsonl')
            continue
        systems[label.strip()] = (text, form)

    gold_glyphs = sum(canon.canonicalize(p['gold'], space=space).count('ſ') for p in pages)
    scope = f'{a.dataset}' + (f' / {a.corpus}' if a.corpus else '')
    print(f'{scope}: {len(pages):,} units, space `{space}`, '
          f'{gold_glyphs:,} long s in the ground truth\n')
    print(f'{"system":28} {"form":7} {"CER":>7} {"WER":>7} {"missing":>8} {"long s emitted":>15}')
    for name, (text, form) in systems.items():
        ce = cn = we = wn = missing = emitted = 0
        for p in pages:
            hyp = text.get(p['id'])
            if hyp is None:
                missing += 1
                hyp = ''
            emitted += sum(hyp.count(c) for c in LONG_S)
            a1, a2, a3, a4 = cer_wer(p['gold'], hyp, space=space)
            ce, cn, we, wn = ce + a1, cn + a2, we + a3, wn + a4
        print(f'{name:28} {form:7} {ce / cn:7.4f} {we / wn:7.4f} {missing:8,} {emitted:15,}')


if __name__ == '__main__':
    main()

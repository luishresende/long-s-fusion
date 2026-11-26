#!/usr/bin/env python3
"""Compares all systems on `goldseg`: gold segmentation, the same provider for
every model, line by line with no post-processing.

The fine-tuned model is in domain here (the 64 works are in train and test).

Usage: python eval_goldseg.py [--output reports/line_results.md]
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fusion import canon                                       # noqa: E402
from fusion.common import Acc, BagAcc, Rules, merge_line       # noqa: E402

csv.field_size_limit(10 ** 7)
RAW = None      # set in main(), depends on --dataset


def per_crop(name):
    """crop path -> text, read from a raw/<name>.jsonl."""
    out = {}
    for path in list(RAW.glob(f'{name}.jsonl')) + sorted(RAW.glob(f'{name}.*of*.jsonl')):
        for line in open(path, encoding='utf-8'):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for ln in rec.get('lines', []):
                if ln.get('image_path'):
                    out[Path(ln['image_path']).name] = ln.get('text', '')
    return out


class LongS:
    """Emission ratio (no alignment) and aligned recall of `ſ`, which is a floor."""

    def __init__(self):
        self.gold = self.hyp = self.right = 0

    def add(self, g, h):
        self.gold += g.count('ſ')
        self.hyp += h.count('ſ')
        for tag, i1, i2, _, _ in Levenshtein.opcodes(g, h):
            if tag == 'equal':
                self.right += g[i1:i2].count('ſ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='goldseg',
                    help='any line dataset, that is, one crop per "page". It '
                         'serves `goldseg` and the external Vesalainen corpus, '
                         'which has the same shape.')
    ap.add_argument('--output', default='reports/line_results.md')
    ap.add_argument('--rules', default='')
    ap.add_argument('--specialist', default='',
                    help='label in raw/ of a third specialist, read by crop file '
                         'name, e.g. tesseract_frak. Adds its row and its fusion.')
    ap.add_argument('--corpus', default='',
                    help='restricts to one goldseg corpus (ocr17, reichsanzeiger)')
    a = ap.parse_args()

    global RAW
    RAW = HERE / 'results' / a.dataset / 'raw'
    rules = Rules.from_text(a.rules)

    pages = list(csv.DictReader(
        open(HERE / 'results' / a.dataset / 'pages.csv', encoding='utf-8')))
    if a.corpus:
        pages = [r for r in pages if r['corpus'] == a.corpus]
        if not pages:
            raise SystemExit(f'no page of corpus {a.corpus!r} '
                             f'in {a.dataset}')
    gold = {r['id']: r['gold'] for r in pages}
    lines = [r for r in csv.DictReader(
        open(HERE / 'results' / a.dataset / 'lines.csv', encoding='utf-8'))
        if r['page_id'] in gold]
    ft = per_crop('paddle_ft3')
    if not ft:
        print('! no raw/paddle_ft3.jsonl; the fine-tuned model is left out')

    systems = ['paddle', 'paddle_ft3', 'calamari', 'kraken',
               'fusion_calamari', 'fusion_kraken']
    if not ft:
        systems.remove('paddle_ft3')
    specs = ['calamari', 'kraken']
    third = per_crop(a.specialist) if a.specialist else {}
    if a.specialist and not third:
        raise SystemExit(f'no raw/{a.specialist}*.jsonl')
    if third:
        specs.append(a.specialist)
        systems += [a.specialist, f'fusion_{a.specialist}']
    acc = {s: Acc(space='line') for s in systems}
    accd = {s: Acc(space='line', fold_s=True) for s in systems}
    bow = {s: BagAcc(space='line') for s in systems}
    ls = {s: LongS() for s in systems}
    stats = {f'fusion_{x}': {} for x in specs}
    empty = collections.Counter()
    n = 0

    for r in lines:
        gold_text = gold.get(r['page_id'])
        if gold_text is None:
            continue
        n += 1
        name = Path(r['line_path']).name
        texts = {
            'paddle': r['paddle'],
            'paddle_ft3': ft.get(name, ''),
            'calamari': r['calamari'],
            'kraken': r['kraken'],
        }
        if third:
            texts[a.specialist] = third.get(name, '')
        for spec in specs:
            t = texts[spec]
            texts[f'fusion_{spec}'] = (
                merge_line(r['paddle'], t, stats[f'fusion_{spec}'], rules=rules)
                if t else r['paddle'])
        g = canon.canonicalize(gold_text, space='line')
        for s in systems:
            h = texts[s]
            if not h.strip():
                empty[s] += 1
            acc[s].add(gold_text, h)
            accd[s].add(gold_text, h)
            bow[s].add(gold_text, h)
            ls[s].add(g, canon.canonicalize(h, space='line'))

    label = {'paddle': 'Paddle base',
             'paddle_ft3': 'PP-OCRv6 fine-tuned (trained on these works)',
             'calamari': 'Calamari alone', 'kraken': 'Kraken alone',
             'fusion_calamari': '**Paddle + Calamari**',
             'fusion_kraken': '**Paddle + Kraken**'}
    if third:
        label[a.specialist] = f'{a.specialist} alone'
        label[f'fusion_{a.specialist}'] = f'**Paddle + {a.specialist}**'
    base = acc['paddle']

    def d(v, b):
        return f'{(b - v) / b:+.2%}' if b else '—'

    L = ['<!-- GENERATED by experiments/eval_goldseg.py. Manual edits are lost on the next run: change the script, not the .md. -->\n',
         f'# `{a.dataset}`: all systems on the same ground\n',
         f'{n} line crops of the `test` set (OCR17 + Reichsanzeiger), '
         f'**ground truth** segmentation (detection is not involved), compared '
         f'**line by line** with **zero post-processing**.\n',
         'All systems go through the **same inference path** (the provider), '
         'including the fine-tuned one. Without that the comparison would be '
         'between normalizations and not between systems.\n',
         f'Fusion rules: `{rules}`\n',
         '| system | CER | ΔCER | WER | ΔWER | BoW-F1 | folded CER | empty |',
         '|---|--:|--:|--:|--:|--:|--:|--:|']
    for s in systems:
        if not acc[s].length:
            continue
        L.append(f'| {label[s]} | {acc[s].cer:.4f} | {d(acc[s].cer, base.cer)} | '
                 f'{acc[s].wer:.4f} | {d(acc[s].wer, base.wer)} | '
                 f'{bow[s].f1:.4f} | {accd[s].cer:.4f} | {empty[s]} |')

    L.append('\n## Recovery of `ſ`\n')
    L.append('| system | `ſ` in ground truth | emitted | ratio | aligned recall |')
    L.append('|---|--:|--:|--:|--:|')
    for s in systems:
        v = ls[s]
        if not v.gold:
            continue
        L.append(f'| {label[s]} | {v.gold} | {v.hyp} | {v.hyp / v.gold:.2f}× | '
                 f'{v.right / v.gold:.3f} |')

    L.append('\n## Fusion edits\n')
    for s in specs:
        st = stats[f'fusion_{s}']
        L.append(f'- **{s}**: {st.get("swapped_f_for_s", 0)} `f`→`ſ` swaps, '
                 f'{st.get("injected_space", 0)} injected spaces')

    out = HERE / a.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L[4:]))
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()

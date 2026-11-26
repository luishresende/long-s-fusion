#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for the CER comparisons of the paper.

The engines are deterministic, so the only uncertainty left is sampling: would
the ordering of two systems hold on other pages of the same collections? Each
comparison is resampled paired. The units are drawn with replacement, and in
every replicate the micro-averaged CER of both systems is recomputed over the
same draw, so what varies is the set of pages, never the system.

  unit     the page in the page-level sets, the line in `goldseg` and
           `vesalainen_en`. With `--unit book` the units are grouped by work
           first, and the works are drawn instead: a sensitivity analysis for
           the dependence between pages of one book. Only where the ground
           truth carries a work identifier (`ocr17`, `porto`, `goldseg`) and
           the comparison spans at least `MIN_WORKS` works.
  replicate  sum of errors over sum of characters, for A and for B, on the same
           draw. Reported: the difference B - A and the relative gain
           (A - B) / A, the same gain the tables print.
  interval  percentile, 95%.

Each hypothesis is built the way the script behind its table builds it, with
the functions of the package: `merge_line` for the fusions, `alex_lexicon` for
the lexicon, `glyph.text_per_crop` for the outputs read from `raw/`. The errors
and characters of every unit are computed once, so the bootstrap is a sum of
vectors. Needs `numpy`, which the Paddle environment carries.

Usage: python analyses/bootstrap_ci.py                          # every comparison
       python analyses/bootstrap_ci.py --comparison lexicon_ocr17 --unit book
       python analyses/bootstrap_ci.py --list
"""
import argparse
import collections
import csv
import functools
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon                                          # noqa: E402
from fusion.common import DEFAULT, cer_wer, merge_line           # noqa: E402
from alex_lexicon import apply_lexicon, build_lexicon, stratum    # noqa: E402
from glyph import text_per_crop                                   # noqa: E402

csv.field_size_limit(10 ** 9)
RESULTS = HERE.parent / 'results'
CHUNK = 200
# the diplomatic stratum of porto holds 2 works, too few clusters to resample
MIN_WORKS = 10


@dataclass(frozen=True)
class Unit:
    book: str
    err_a: int
    err_b: int
    n: int


def errors(gold, hyp, space):
    e, n, _, _ = cer_wer(gold, hyp, space=space)
    return e, n


@functools.lru_cache(maxsize=None)
def pages(ds):
    return list(csv.DictReader(open(RESULTS / ds / 'pages.csv', encoding='utf-8')))


@functools.lru_cache(maxsize=None)
def lines(ds):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(RESULTS / ds / 'lines.csv', encoding='utf-8')):
        out[r['page_id']].append(r)
    for rows in out.values():
        rows.sort(key=lambda r: int(r['line_idx']))
    return out


@functools.lru_cache(maxsize=None)
def raw_text(ds, label):
    t = text_per_crop(RESULTS / ds / 'raw', label)
    if t is None:
        sys.exit(f'no raw/{label}*.jsonl in {ds}')
    return t


def fused(rows, spec, rules=DEFAULT):
    return '\n'.join(merge_line(r['paddle'], r[spec], rules=rules) if r[spec]
                     else r['paddle'] for r in rows)


def page_fusion(ds, spec, against):
    """`eval_page.py`: pages with no line are skipped; A is the base or Kraken
    with its own segmenter, B the fusion."""
    out = []
    for p in pages(ds):
        rows = lines(ds).get(p['id'])
        if not rows:
            continue
        a = p['paddle'] if against == 'paddle' else (p.get('kraken_standalone') or '')
        ea, n = errors(p['gold'], a, 'page')
        eb, _ = errors(p['gold'], fused(rows, spec), 'page')
        out.append(Unit(p.get('book', ''), ea, eb, n))
    return out


def line_fusion(ds, corpus, spec):
    """`eval_goldseg.py`: one unit per crop of `lines.csv`."""
    meta = {p['id']: p for p in pages(ds) if not corpus or p['corpus'] == corpus}
    out = []
    for pid, rows in lines(ds).items():
        p = meta.get(pid)
        if p is None:
            continue
        for r in rows:
            ea, n = errors(p['gold'], r['paddle'], 'line')
            eb, _ = errors(p['gold'], fused([r], spec), 'line')
            out.append(Unit(p.get('book', ''), ea, eb, n))
    return out


def raw_pair(ds, corpus, label_a, label_b, space):
    """`glyph.py`: texts from `raw/`, crops in the order of `detect.jsonl`. The
    label `fusion+<spec>` is the full rule over `paddle`, as there."""
    det = {}
    for ln in open(RESULTS / ds / 'raw' / 'detect.jsonl', encoding='utf-8'):
        r = json.loads(ln)
        det[r['id']] = [l['image_path'] for l in r.get('lines', [])
                        if l.get('image_path')]

    def text(label):
        if label.startswith('fusion+'):
            base, spec = raw_text(ds, 'paddle'), raw_text(ds, label[len('fusion+'):])
            return {c: merge_line(base.get(c, ''), spec.get(c, ''), rules=DEFAULT)
                    for c in base}
        return raw_text(ds, label)

    ta, tb = text(label_a), text(label_b)
    out = []
    for p in pages(ds):
        if corpus and p.get('corpus') != corpus:
            continue
        crops = det.get(p['id'], [])
        ea, n = errors(p['gold'], '\n'.join(ta.get(c, '') for c in crops), space)
        eb, _ = errors(p['gold'], '\n'.join(tb.get(c, '') for c in crops), space)
        out.append(Unit(p.get('book', ''), ea, eb, n))
    return out


def lexicon_units(ds, corpus):
    """`alex_lexicon.py --mode rest --target diplomatic`: unit id -> (book,
    gold, corrected text), each work corrected with the lexicon of the others."""
    rows = [r for r in pages(ds) if not corpus or r.get('corpus') == corpus]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['book']].append(r)
    eligible = {k for k, rs in groups.items()
                if len({w for r in rs for w in r['gold'].split() if 'ſ' in w}) >= 50}
    out = {}
    for ref in sorted(groups):
        if ref not in eligible and not any(k in eligible for k in groups if k != ref):
            continue
        lex, _ = build_lexicon([r['gold'] for k, rs in groups.items() if k != ref
                                for r in rs])
        for r in groups[ref]:
            out[r['id']] = (ref, r['gold'], apply_lexicon(r.get('paddle') or '', lex)[0])
    return out


def lexicon_vs_fusion(ds, corpus, spec):
    space = 'line' if ds == 'goldseg' else 'page'
    lex = lexicon_units(ds, corpus)
    out = []
    for pid, (book, gold, corrected) in lex.items():
        rows = lines(ds).get(pid)
        if not rows:
            continue
        ea, n = errors(gold, corrected, space)
        eb, _ = errors(gold, fused(rows, spec), space)
        out.append(Unit(book, ea, eb, n))
    return out


def diplomatic(ds):
    """`finetune/out_of_domain.py`, diplomatic stratum: three tuned columns (A)
    against rule 1 alone over the base, with Calamari (B)."""
    tuned = raw_text(ds, '3classes')
    rule1 = replace(DEFAULT, space=False)
    out = []
    for p in pages(ds):
        rows = lines(ds).get(p['id'])
        if not rows:
            continue
        if stratum(canon.canonicalize(p['gold'], space='page')) != 'diplomatic':
            continue
        a = '\n'.join(tuned.get(r['line_path'], '') for r in rows)
        ea, n = errors(p['gold'], a, 'page')
        eb, _ = errors(p['gold'], fused(rows, 'calamari', rule1), 'page')
        out.append(Unit(p.get('book', ''), ea, eb, n))
    return out


# name -> (table, A, B, unit, works?, builder)
COMPARISONS = {
    'fusion_ocr17_line': ('T2', 'PP-OCRv6', 'fusion +Kraken', 'line', True,
                          lambda: line_fusion('goldseg', 'ocr17', 'kraken')),
    'fusion_reichsanzeiger_line': ('T2', 'PP-OCRv6', 'fusion +Calamari', 'line', True,
                                   lambda: line_fusion('goldseg', 'reichsanzeiger', 'calamari')),
    'fusion_porto': ('T2', 'PP-OCRv6', 'fusion +Calamari', 'page', True,
                     lambda: page_fusion('porto', 'calamari', 'paddle')),
    'fusion_impact': ('T2', 'PP-OCRv6', 'fusion +Calamari', 'page', False,
                      lambda: page_fusion('impact', 'calamari', 'paddle')),
    'fusion_vesalainen_en': ('T2/T3', 'PP-OCRv6', 'fusion +Calamari', 'line', False,
                             lambda: raw_pair('vesalainen_en', None, 'paddle',
                                              'fusion+calamari', 'line')),
    'diplomatic_porto': ('T7', 'three columns', 'fusion, glyph rule', 'page', True,
                         lambda: diplomatic('porto')),
    'diplomatic_impact': ('T7', 'three columns', 'fusion, glyph rule', 'page', False,
                          lambda: diplomatic('impact')),
    'lexicon_ocr17': ('T8', 'lexicon', 'fusion +Kraken', 'line', True,
                      lambda: lexicon_vs_fusion('goldseg', 'ocr17', 'kraken')),
    'lexicon_reichsanzeiger': ('T8', 'lexicon', 'fusion +Calamari', 'line', True,
                               lambda: lexicon_vs_fusion('goldseg', 'reichsanzeiger',
                                                         'calamari')),
    'lexicon_porto': ('T8', 'lexicon', 'fusion +Calamari', 'page', True,
                      lambda: lexicon_vs_fusion('porto', None, 'calamari')),
    'kraken_impact': ('5.3', 'Kraken, own segmenter', 'fusion +Calamari', 'page', False,
                      lambda: page_fusion('impact', 'calamari', 'kraken')),
    'kraken_ocr17': ('5.3', 'Kraken, own segmenter', 'fusion +Kraken', 'page', True,
                     lambda: page_fusion('ocr17', 'kraken', 'kraken')),
    'kraken_porto': ('5.3', 'Kraken, own segmenter', 'fusion +Calamari', 'page', True,
                     lambda: page_fusion('porto', 'calamari', 'kraken')),
    'columns3_goldseg': ('T6', 'PP-OCRv6', 'three columns', 'line', True,
                         lambda: raw_pair('goldseg', None, 'paddle', '3classes', 'line')),
    'column1_goldseg': ('T6', 'PP-OCRv6', 'one column', 'line', True,
                        lambda: raw_pair('goldseg', None, 'paddle', '1class', 'line')),
    'columns3_vesalainen_en': ('T3', 'PP-OCRv6', 'three columns', 'line', False,
                               lambda: raw_pair('vesalainen_en', None, 'paddle',
                                                '3classes', 'line')),
}


def by_book(units):
    acc = collections.defaultdict(lambda: [0, 0, 0])
    for u in units:
        s = acc[u.book]
        s[0] += u.err_a
        s[1] += u.err_b
        s[2] += u.n
    return [Unit(k, *v) for k, v in acc.items()]


def bootstrap(units, replicates, seed):
    """(CER A, CER B, [diff B - A], [gain (A - B) / A]) over `replicates` draws."""
    m = np.array([(u.err_a, u.err_b, u.n) for u in units], dtype=np.int64)
    rng = np.random.default_rng(seed)
    diff, gain = np.empty(replicates), np.empty(replicates)
    for start in range(0, replicates, CHUNK):
        k = min(CHUNK, replicates - start)
        idx = rng.integers(0, len(m), size=(k, len(m)))
        a, b, n = (m[:, j][idx].sum(axis=1) for j in range(3))
        ca, cb = a / n, b / n
        diff[start:start + k] = cb - ca
        gain[start:start + k] = (ca - cb) / ca
    ea, eb, n = m.sum(axis=0)
    return ea / n, eb / n, diff, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comparison', default='all',
                    help=f'one of {", ".join(COMPARISONS)}, or `all`')
    ap.add_argument('--unit', default=None, choices=['page', 'line', 'book'],
                    help='default: the native unit of each comparison, then '
                         '`book` where the set has works')
    ap.add_argument('--replicates', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261123)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    if a.list:
        for name, (table, la, lb, unit, works, _) in COMPARISONS.items():
            print(f'{name:28} {table:6} {unit:5} {"works" if works else "":6} {la} -> {lb}')
        return
    names = list(COMPARISONS) if a.comparison == 'all' else [a.comparison]
    for name in names:
        if name not in COMPARISONS:
            sys.exit(f'unknown comparison {name!r}; see --list')

    started = time.perf_counter()
    built = {}
    for name in names:
        table, la, lb, unit, works, build = COMPARISONS[name]
        if a.unit in ('page', 'line') and a.unit != unit:
            print(f'! {name}: the unit is the {unit}, not the {a.unit}')
            continue
        if a.unit == 'book' and not works:
            print(f'! {name}: no work identifier')
            continue
        built[name] = build()
    t_build = time.perf_counter() - started

    print(f'\npaired bootstrap, B = {a.replicates:,}, seed {a.seed}, 95% percentile '
          f'interval. diff = CER B - CER A (negative: B lower); gain = (A - B) / A\n')
    head = (f'{"comparison":28} {"table":5} {"unit":5} {"n":>6} {"CER A":>7} {"CER B":>7} '
            f'{"diff [95% CI]":>28} {"gain [95% CI]":>27} {"crosses 0":>9}')
    passes = [a.unit] if a.unit else ['native', 'book']
    t_boot = time.perf_counter()
    for unit_pass in passes:
        rows = []
        for name, units in built.items():
            table, la, lb, unit, works, _ = COMPARISONS[name]
            if unit_pass == 'book':
                if not works:
                    continue
                units, unit = by_book(units), 'book'
                if len(units) < MIN_WORKS:
                    rows.append(f'{name:28} {table:5} {unit:5} {len(units):6,}  '
                                f'skipped: fewer than {MIN_WORKS} works')
                    continue
            ca, cb, diff, gain = bootstrap(units, a.replicates, a.seed)
            dl, dh = np.percentile(diff, [2.5, 97.5])
            gl, gh = np.percentile(gain, [2.5, 97.5])
            rows.append(f'{name:28} {table:5} {unit:5} {len(units):6,} {ca:7.4f} {cb:7.4f} '
                        f'{cb - ca:+8.4f} [{dl:+.4f}, {dh:+.4f}] '
                        f'{(ca - cb) / ca:+7.2%} [{gl:+7.2%}, {gh:+7.2%}] '
                        f'{"yes" if dl <= 0 <= dh else "no":>9}')
        if rows:
            print(head)
            print('-' * len(head))
            print('\n'.join(rows) + '\n')
    t_boot = time.perf_counter() - t_boot

    print('A and B of each comparison:')
    for name in built:
        table, la, lb, *_ = COMPARISONS[name]
        print(f'  {name:28} A = {la}, B = {lb}')
    print(f'\nper-unit errors {t_build:.0f} s, bootstrap {t_boot:.0f} s')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Lexical baseline of Alex, Grover, Klein & Tobin (2012, §5.2): misspellings
generated from a reference lexicon and resolved by frequency, leave-one-work-out.

  --mode one / rest              one reference work, or all but the tested one
  --target modernized / diplomatic   f -> s with folded reading, or f -> ſ strict

Usage: python analyses/alex_lexicon.py --dataset ocr17 --target modernized
"""
import argparse
import collections
import csv
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon                                       # noqa: E402
from fusion.common import Acc                                   # noqa: E402

csv.field_size_limit(10 ** 7)
LONG_S = 'ſ'
MAX_S = 6          # words with more `ſ` than this: only the full swap is generated


def stratum(g):
    """As in `out_of_domain.py`: the fraction of long s among the s of the page's gold."""
    s, long_s = g.count('s'), g.count(LONG_S)
    if s + long_s == 0:
        return None
    f = long_s / (s + long_s)
    return 'diplomatic' if f >= 0.30 else (
        'modernized' if f < 0.02 else 'intermediate')


def variants(word, glyph=LONG_S):
    """Every spelling the glyph->f confusion would produce. Subsets of the
    positions, as in "all the possible misspellings". Under the modernized target
    the glyph is `s`, which is their case."""
    pos = [i for i, c in enumerate(word) if c == glyph]
    if not pos:
        return []
    if len(pos) > MAX_S:
        return [word.replace(LONG_S, 'f')]
    out = []
    for k in range(1, len(pos) + 1):
        for combo in itertools.combinations(pos, k):
            ch = list(word)
            for i in combo:
                ch[i] = 'f'
            out.append(''.join(ch))
    return out


def build_lexicon(texts, glyph=LONG_S):
    """Unigram frequencies and a map misspelling -> form with the glyph, kept only
    when the form is more frequent than the misspelling."""
    freq = collections.Counter()
    for t in texts:
        freq.update(t.split())
    lex = {}
    for word, n in freq.items():
        if glyph not in word:
            continue
        for wrong in variants(word, glyph):
            if n <= freq.get(wrong, 0):
                continue                      # `feed` does not become `seed`
            previous = lex.get(wrong)
            if previous is None or n > freq[previous]:
                lex[wrong] = word
    return lex, freq


def apply_lexicon(text, lex):
    out, swaps = [], 0
    for p in text.split():
        target = lex.get(p)
        if target:
            out.append(target)
            swaps += 1
        else:
            out.append(p)
    return ' '.join(out), swaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ocr17')
    ap.add_argument('--key', default=None,
                    help='column that identifies the work (default: book, or '
                         'date in the reichsanzeiger)')
    ap.add_argument('--mode', default='rest', choices=['rest', 'one'],
                    help='`rest` (default) is their scenario: the paper\'s '
                         'reference corpus is nine Gutenberg books, not one. '
                         '`one` is a sensitivity analysis at minimal resource, '
                         'and must not be reported as a replication.')
    ap.add_argument('--engine', default='paddle',
                    help='column of the base text')
    ap.add_argument('--min-types', type=int, default=50,
                    help='minimum ſ types for a group to serve as reference')
    ap.add_argument('--corpus', default=None,
                    help='in `goldseg`, restricts to ocr17 or reichsanzeiger')
    ap.add_argument('--target', default='diplomatic',
                    choices=['diplomatic', 'modernized'],
                    help='`modernized` is their task (f->s, modern reference, '
                         'folded reading); `diplomatic` is ours (f->ſ, period '
                         'reference, strict reading).')
    ap.add_argument('--space', default=None, choices=['line', 'page'],
                    help='default: `line` in goldseg (per-line ground truth), '
                         '`page` in the others. It is the same choice as in '
                         'eval_*.')
    a = ap.parse_args()
    space = a.space or ('line' if a.dataset == 'goldseg' else 'page')
    modern = a.target == 'modernized'
    glyph = 's' if modern else LONG_S
    # modernized target: fold ſ->s in the reference, as in a modern edition
    prep = (lambda t: t.replace(LONG_S, 's')) if modern else (lambda t: t)

    base = HERE.parent / 'results' / a.dataset / 'pages.csv'
    rows = list(csv.DictReader(open(base, encoding='utf-8')))
    if a.corpus:
        rows = [r for r in rows if r.get('corpus') == a.corpus]
        if not rows:
            sys.exit(f'no row with corpus={a.corpus!r}')
    key = a.key or ('date' if a.dataset == 'reichsanzeiger' else 'book')
    if key not in rows[0]:
        sys.exit(f'{a.dataset}: no `{key}` column, so this corpus has no work '
                 f'identifier and the cross-validation does not apply.')

    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)

    eligible = {}
    for k, rs in groups.items():
        types = {p for r in rs for p in prep(r['gold']).split() if glyph in p}
        if len(types) >= a.min_types:
            eligible[k] = len(types)

    print(f'{a.dataset}{"/" + a.corpus if a.corpus else ""}: '
          f'{len(groups)} groups by `{key}`, target `{a.target}`, '
          f'space `{space}`, '
          f'{"folded" if modern else "strict"} reading, '
          f'{len(eligible)} can be a reference (>= {a.min_types} types with {glyph!r})')
    if not eligible:
        sys.exit('no group has enough ſ: the method is not applicable here.')

    refs = sorted(eligible) if a.mode == 'one' else sorted(groups)
    print(f'mode `{a.mode}`: {len(refs)} folds\n')
    print(f'{"reference":22s} {"ref tokens":>10s} {"CER base":>9s} '
          f'{"CER lexicon":>11s} {"Δ":>8s} {"swaps":>8s} {"hit:damage":>12s}')

    tot = collections.Counter()
    # the pooled CER is leave-one-out only in `rest` mode, where each work is tested once
    pool_base, pool_lex = Acc(space=space, fold_s=modern), Acc(space=space, fold_s=modern)
    # by gold stratum: under the modernized target every swap there is a false positive
    strat_base = collections.defaultdict(lambda: Acc(space=space, fold_s=modern))
    strat_lex = collections.defaultdict(lambda: Acc(space=space, fold_s=modern))
    strat_hd = collections.defaultdict(collections.Counter)
    gains = []
    for ref in refs:
        if a.mode == 'one':
            ref_texts = [r['gold'] for r in groups[ref]]
            test = [r for k, rs in groups.items() if k != ref for r in rs]
        else:
            if ref not in eligible and not any(k in eligible for k in groups if k != ref):
                continue
            ref_texts = [r['gold'] for k, rs in groups.items() if k != ref
                         for r in rs]
            test = groups[ref]
        lex, freq = build_lexicon([prep(t) for t in ref_texts], glyph)
        n_ref = sum(len(t.split()) for t in ref_texts)

        ac_base = Acc(space=space, fold_s=modern)
        ac_lex = Acc(space=space, fold_s=modern)
        hit = damage = swaps = 0
        for r in test:
            g, h = r['gold'], (r.get(a.engine) or '')
            new, n = apply_lexicon(h, lex)
            swaps += n
            ac_base.add(g, h)
            ac_lex.add(g, new)
            pool_base.add(g, h)
            pool_lex.add(g, new)
            e = stratum(g)
            if e:
                strat_base[e].add(g, h)
                strat_lex[e].add(g, new)
            # hit:damage per swapped position, under the strict reading
            gs = canon.canonicalize(g, space=space, fold_s=modern)
            for before, after in zip(h.split(), new.split()):
                if before == after:
                    continue
                if after in gs.split():
                    hit += 1
                    if e:
                        strat_hd[e]['hit'] += 1
                else:
                    damage += 1
                    if e:
                        strat_hd[e]['damage'] += 1
        ratio = f'{hit / damage:.1f}:1' if damage else ('inf:1' if hit else '—')
        d = (ac_base.cer - ac_lex.cer) / ac_base.cer * 100 if ac_base.cer else 0
        print(f'{ref[:22]:22s} {n_ref:10d} {ac_base.cer:9.4f} {ac_lex.cer:11.4f} '
              f'{d:+7.2f}% {swaps:8d} {ratio:>12s}')
        gains.append(d)
        tot['hit'] += hit
        tot['damage'] += damage
        tot['swaps'] += swaps
    r = f"{tot['hit'] / tot['damage']:.1f}:1" if tot['damage'] else '—'
    print(f"\nswaps {tot['swaps']}, hits {tot['hit']}, "
          f"damages {tot['damage']}, ratio {r}")
    gains.sort()
    med = gains[len(gains) // 2] if gains else 0
    print(f'gain per fold: min {gains[0]:+.2f}%  median {med:+.2f}%  '
          f'max {gains[-1]:+.2f}%')
    if a.mode == 'rest':
        dp = (pool_base.cer - pool_lex.cer) / pool_base.cer * 100
        print(f'leave-one-out CER: base {pool_base.cer:.4f} -> '
              f'lexicon {pool_lex.cer:.4f}  ({dp:+.2f}%)')
    print(f'\n{"stratum":16s} {"CER base":>9s} {"CER lexicon":>11s} {"Δ":>8s} '
          f'{"hit:damage":>12s}')
    for e in ('diplomatic', 'intermediate', 'modernized'):
        if e not in strat_base:
            continue
        b, l = strat_base[e].cer, strat_lex[e].cer
        d = (b - l) / b * 100 if b else 0
        hd = strat_hd[e]
        r = (f"{hd['hit'] / hd['damage']:.1f}:1" if hd['damage']
             else ('inf:1' if hd['hit'] else '—'))
        print(f'{e:16s} {b:9.4f} {l:11.4f} {d:+7.2f}% {r:>12s}')


if __name__ == '__main__':
    main()

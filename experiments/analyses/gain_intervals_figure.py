#!/usr/bin/env python3
"""Figure of the fusion gain in the five corpora, with its bootstrap intervals.

One row per corpus: the relative CER reduction of the fusion over PP-OCRv6 and
its 95% interval, resampling pages or lines, and resampling works where the
ground truth identifies them. Every number comes from `bootstrap_ci.py`, with
its default replicates and seed, so the figure and the intervals quoted in the
text are the same computation.

Needs `matplotlib`, besides what `bootstrap_ci.py` needs.

Usage: python analyses/gain_intervals_figure.py --output gain_intervals.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_ci as bc
from figure_style import save

ROWS = [('fusion_ocr17_line', 'ocr17 (fr)'),
        ('fusion_reichsanzeiger_line', 'reichs. (de)'),
        ('fusion_porto', 'porto (pt)'),
        ('fusion_impact', 'impact (nl)'),
        ('fusion_vesalainen_en', 'vesalainen_en (en)')]
DARK, LIGHT = '#1f4e79', '#c0692b'
TEXT = {'x': 'relative CER reduction over the uncorrected PP-OCRv6 (%)',
        'units': 'resampling of pages or lines',
        'works': 'resampling of works'}


def interval(units, replicates, seed):
    ca, cb, _, gain = bc.bootstrap(units, replicates, seed)
    lo, hi = np.percentile(gain, [2.5, 97.5])
    return 100 * (ca - cb) / ca, 100 * lo, 100 * hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='gain_intervals.pdf')
    ap.add_argument('--replicates', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261123)
    a = ap.parse_args()

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for y, (name, label) in enumerate(ROWS):
        _, _, _, _, works, build = bc.COMPARISONS[name]
        units = build()
        g, lo, hi = interval(units, a.replicates, a.seed)
        ax.errorbar(g, y - 0.14, xerr=[[g - lo], [hi - g]], fmt='o', ms=4, lw=1.4,
                    capsize=3, color=DARK)
        print(f'{label:20} page/line {g:6.2f} [{lo:6.2f}, {hi:6.2f}]')
        books = bc.by_book(units) if works else []
        if len(books) >= bc.MIN_WORKS:
            g2, lo, hi = interval(books, a.replicates, a.seed)
            ax.errorbar(g2, y + 0.14, xerr=[[g2 - lo], [hi - g2]], fmt='s', ms=3.5,
                        lw=1.4, capsize=3, color=LIGHT)
            print(f'{"":20} work      {g2:6.2f} [{lo:6.2f}, {hi:6.2f}]')
    ax.axvline(0, color='#888', lw=0.7)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([l for _, l in ROWS], style='italic')
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_xlabel(TEXT['x'])
    ax.grid(axis='x', color='#ddd', lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.errorbar([], [], xerr=[], fmt='o', ms=4, color=DARK, label=TEXT['units'])
    ax.errorbar([], [], xerr=[], fmt='s', ms=3.5, color=LIGHT, label=TEXT['works'])
    ax.legend(frameon=False, loc='center right', fontsize=7)
    fig.tight_layout()
    save(fig, a.output)


if __name__ == '__main__':
    main()

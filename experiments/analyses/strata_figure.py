#!/usr/bin/env python3
"""Figure of the glyph record of each page, ground truth against Calamari.

One point per page of `porto` and `impact`. The horizontal axis is the fraction
of long s among the s of the page's ground truth, the one that defines the three
strata of Section 3.1. The vertical axis is the same fraction in the reading of
Calamari over the crops of that page. Pages on the diagonal have a ground truth
that records what is printed. Pages at the left edge and high on the vertical
axis are printed with the glyph and transcribed without it.

Reads `pages.csv` and `lines.csv`. Needs `matplotlib`.

Usage: python analyses/strata_figure.py --output strata.pdf
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from fusion import canon
sys.path.insert(0, str(HERE))
from alex_lexicon import stratum
from figure_style import save

csv.field_size_limit(10 ** 9)
RESULTS = HERE.parent / 'results'
PANELS = [('porto', 'porto (pt)'), ('impact', 'impact (nl)')]
COLORS = {'diplomatic': '#1f4e79', 'intermediate': '#7a7a7a', 'modernized': '#c0692b'}
TEXT = {'x': 'fraction of ſ in the page ground truth',
        'y': 'fraction of ſ in the Calamari reading'}


def fraction(text):
    t = canon.canonicalize(text, space='page')
    s, long_s = t.count('s'), t.count('ſ')
    return None if s + long_s == 0 else long_s / (s + long_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='strata.pdf')
    a = ap.parse_args()

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                         'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, (ds, title) in zip(axs, PANELS):
        read = collections.defaultdict(list)
        for r in csv.DictReader(open(RESULTS / ds / 'lines.csv', encoding='utf-8')):
            read[r['page_id']].append(r['calamari'] or '')
        pts = collections.defaultdict(list)
        for p in csv.DictReader(open(RESULTS / ds / 'pages.csv', encoding='utf-8')):
            g, c = fraction(p['gold']), fraction('\n'.join(read.get(p['id'], [])))
            if g is not None and c is not None:
                pts[stratum(canon.canonicalize(p['gold'], space='page'))].append((g, c))
        for key in ('modernized', 'intermediate', 'diplomatic'):
            xy = pts[key]
            ax.scatter([x for x, _ in xy], [y for _, y in xy], s=5, alpha=0.35,
                       color=COLORS[key], linewidths=0,
                       label=key)
            print(f'{ds:8} {key:13} {len(xy):5} pages')
        ax.plot([0, 1], [0, 1], color='#888', lw=0.6, ls='--')
        for x in (0.02, 0.30):  # the limits of `alex_lexicon.stratum`
            ax.axvline(x, color='#bbb', lw=0.6)
        ax.set_title(title, style='italic', fontsize=8, loc='left')
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(-0.03, 1.0)
        ax.set_xlabel(TEXT['x'])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        leg = ax.legend(frameon=False, loc='lower right', fontsize=7, markerscale=2.5)
        for h in leg.legend_handles:
            h.set_alpha(1)
    axs[0].set_ylabel(TEXT['y'])
    fig.tight_layout()
    save(fig, a.output)


if __name__ == '__main__':
    main()

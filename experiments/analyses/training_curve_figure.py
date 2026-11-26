#!/usr/bin/env python3
"""Figure of the column-tuning curves: validation CER and `ſ` emission per step,
from `training.log` and `projection.log` (evaluations every 50 steps, `--eval-every 50`).

Usage: python analyses/training_curve_figure.py --output training_curve.pdf
       python analyses/training_curve_figure.py --logs /path/to/training_logs
"""
import argparse
import collections
import io
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from training_curve import ARM, HEADER, POINT                    # noqa: E402
from figure_style import decimal_axis, save    # noqa: E402

LOGS = ('training.log', 'projection.log')
BASELINE = re.compile(r'(?:antes do treino|before training): CER ([\d.]+)')
LABEL_SHIFT = {('cer', 'single_class'): -6}  # keeps the label off the baseline line
ARMS = ['three_columns', 'single_class', 'projection_full']
STYLE = {'three_columns': dict(color='#1f4e79', ls='-', lw=1.6),
         'single_class': dict(color='#c0692b', ls='--', lw=1.6),
         'projection_full': dict(color='#7a7a7a', ls=':', lw=1.8)}
TEXT = {'three_columns': 'ſ, f, s',
        'single_class': 'ſ only',
        'projection_full': 'full projection',
        'step': 'training step',
        'cer': 'validation CER',
        'emission': 'emission ratio of ſ',
        'baseline': 'baseline, {:.4f}',
        'gold': 'ground truth'}


def read(paths):
    """arm -> [(step, cer, emission)], and the baseline CER of the logs."""
    curves, baselines = collections.defaultdict(list), set()
    for path in paths:
        arm = None
        for line in io.open(path, encoding='utf-8', errors='replace'):
            head = HEADER.match(line)
            if head:
                arm = ARM[head.group(1).lower()]
                continue
            base = BASELINE.search(line)
            if base:
                baselines.add(float(base.group(1)))
                continue
            point = POINT.search(line)
            if point and arm:
                _, step, _, _, cer, emission, _, _ = point.groups()
                curves[arm].append((int(step), float(cer), float(emission)))
    if len(baselines) != 1:
        sys.exit(f'expected one baseline CER across the logs, found {sorted(baselines)}')
    return curves, baselines.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=str(HERE.parent.parent / 'training_logs'),
                    help='directory holding training.log and projection.log')
    ap.add_argument('--output', default='training_curve.pdf')
    a = ap.parse_args()

    paths = [Path(a.logs) / name for name in LOGS]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit(f'missing: {", ".join(missing)}')
    curves, baseline = read(paths)
    for arm in ARMS:
        pts = curves.get(arm)
        if not pts:
            sys.exit(f'no evaluation point for {arm} in the logs')
        best = min(pts, key=lambda p: p[1])
        print(f'{arm:16} {len(pts):3} points, steps {pts[0][0]}-{pts[-1][0]}, '
              f'best CER {best[1]:.4f} at step {best[0]}, last emission {pts[-1][2]:.3f}')
    print(f'baseline CER {baseline:.4f}')

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(6.95, 2.63))
    last_step = max(p[0] for pts in curves.values() for p in pts)
    for ax, column, label in ((axs[0], 1, 'cer'), (axs[1], 2, 'emission')):
        for arm in ARMS:
            pts = curves[arm]
            ax.plot([p[0] for p in pts], [p[column] for p in pts], **STYLE[arm],
                    label=TEXT[arm])
            ax.annotate(TEXT[arm], (pts[-1][0], pts[-1][column]), xytext=(4, LABEL_SHIFT.get((label, arm), 0)),
                        textcoords='offset points', va='center', fontsize=7,
                        color=STYLE[arm]['color'])
        ax.set_xlim(0, last_step * 1.28)
        ax.set_xticks(range(0, int(last_step * 1.28) + 1, 500))
        ax.set_xlabel(TEXT['step'])
        ax.set_ylabel(TEXT[label])
        ax.grid(axis='y', color='#e4e4e4', lw=0.5)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    axs[0].yaxis.set_major_locator(MultipleLocator(0.02))
    axs[0].axhline(baseline, color='#555', lw=0.8)
    axs[0].text(last_step * 0.92, baseline, TEXT['baseline'].format(baseline),
                ha='right', va='bottom', fontsize=7, color='#555')
    axs[1].axhline(1.0, color='#555', lw=0.8)
    axs[1].text(60, 1.0, TEXT['gold'], ha='left', va='bottom', fontsize=7, color='#555')
    axs[1].set_ylim(0, 1.12)
    decimal_axis(axs[0].yaxis, '{:.2f}')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               handlelength=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, a.output)


if __name__ == '__main__':
    main()

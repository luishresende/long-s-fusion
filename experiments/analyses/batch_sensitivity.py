#!/usr/bin/env python3
"""How much the recognizer's output moves when only the batch changes: share of
crops with different text, and ΔCER. PaddleX pads each batch to its widest crop.

Usage: python analyses/batch_sensitivity.py --dataset goldseg --corpus ocr17 \
              --batches 8,128
       python analyses/batch_sensitivity.py --stage detect --dataset ocr17 \
              --limits 3000,6000
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from fusion import canon

csv.field_size_limit(10 ** 7)


def read_jsonl(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for ln in rec.get('lines') or []:
            if ln.get('image_path'):
                out[ln['image_path']] = ln.get('text') or ''
    return out


def recognize(dataset, batch, out_path):
    """One pass of stage 2 with the batch size forced."""
    env = dict(os.environ, PADDLE_REC_BATCH=str(batch))
    cmd = [sys.executable, '-u', 'inference_pipeline/recognize.py',
           '--dataset', dataset, '--engine', 'paddle',
           '--label', out_path.stem, '--chunk', str(batch)]
    print(f'  batch {batch}: {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def compare(dataset, corpus, a_texts, b_texts, space):
    res = HERE / 'results' / dataset
    pages = list(csv.DictReader(open(res / 'pages.csv', encoding='utf-8')))
    if corpus:
        pages = [p for p in pages if p.get('corpus') == corpus]
    det = {}
    for line in open(res / 'raw' / 'detect.jsonl', encoding='utf-8'):
        rec = json.loads(line)
        det[rec['id']] = [l['image_path'] for l in rec.get('lines', [])
                          if l.get('image_path')]

    shared = [c for p in pages for c in det.get(p['id'], [])
              if c in a_texts and c in b_texts]
    differ = sum(1 for c in shared if a_texts[c] != b_texts[c])

    def cer(texts):
        num = den = 0
        for p in pages:
            hyp = '\n'.join(texts.get(c, '') for c in det.get(p['id'], []))
            g = canon.canonicalize(p['gold'], space=space)
            h = canon.canonicalize(hyp, space=space)
            num += Levenshtein.distance(g, h)
            den += len(g)
        return num / max(den, 1)

    ca, cb = cer(a_texts), cer(b_texts)
    print(f'\ncrops compared          {len(shared):,}')
    print(f'crops with different text {differ:,}  '
          f'({differ / max(len(shared), 1):.1%})')
    print(f'CER run A               {ca:.4f}')
    print(f'CER run B               {cb:.4f}')
    print(f'ΔCER                    {abs(ca - cb):.4f}  '
          f'({abs(ca - cb) / max(ca, 1e-9):.2%} relative)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='recognize',
                    choices=['recognize', 'detect'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--space', default=None, choices=['line', 'page'])
    ap.add_argument('--batches', default='8,128')
    ap.add_argument('--limits', default='3000,6000', help='for --stage detect')
    ap.add_argument('--reuse', default=None,
                    help='two label names already in raw/, to skip the runs')
    a = ap.parse_args()

    space = a.space or ('page' if a.dataset in ('porto', 'impact', 'ocr17',
                                                'reichsanzeiger') else 'line')
    if a.stage == 'detect':
        print('detector sensitivity is a pipeline re-run: set '
              'PADDLE_DET_LIMIT to each of', a.limits,
              'run inference_pipeline/detect.py, and compare the line counts '
              'in raw/detect.jsonl. It overwrites the corpus detection.')
        return

    res = HERE / 'results' / a.dataset / 'raw'
    if a.reuse:
        la, lb = [s.strip() for s in a.reuse.split(',')]
        ta, tb = read_jsonl(res / f'{la}.jsonl'), read_jsonl(res / f'{lb}.jsonl')
    else:
        ba, bb = [int(s) for s in a.batches.split(',')]
        pa, pb = res / f'_batch{ba}.jsonl', res / f'_batch{bb}.jsonl'
        recognize(a.dataset, ba, pa)
        recognize(a.dataset, bb, pb)
        ta, tb = read_jsonl(pa), read_jsonl(pb)
    compare(a.dataset, a.corpus, ta, tb, space)


if __name__ == '__main__':
    main()

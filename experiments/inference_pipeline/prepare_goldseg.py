#!/usr/bin/env python3
"""Builds `goldseg`: the ground-truth line crops of `train_data` as one-line pages,
with a synthetic `detect.jsonl`; `book` is the `group` of `records.csv`.

Usage: python inference_pipeline/prepare_goldseg.py [--splits test]
       python inference_pipeline/prepare_goldseg.py --splits test,val --name goldseg_tv
"""
import argparse
import csv
import json
from pathlib import Path

from stages import ROOT, to_key

csv.field_size_limit(10 ** 7)
TRAIN_DATA = ROOT / 'finetune' / 'train_data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='test',
                    help='train_data splits to include (test,val)')
    ap.add_argument('--name', default='goldseg',
                    help='name of the generated dataset')
    a = ap.parse_args()

    meta = {r['path']: r for r in
            csv.DictReader(open(TRAIN_DATA / 'records.csv', encoding='utf-8'))}

    lines = []
    for split in a.splits.split(','):
        for l in open(TRAIN_DATA / f'{split}_list.txt', encoding='utf-8'):
            rel, _, txt = l.rstrip('\n').partition('\t')
            m = meta.get(rel, {})
            lines.append({
                'id': Path(rel).stem,        # stable md5, unique per crop
                'book': m.get('group', ''),
                'corpus': m.get('corpus',
                                'ocr17' if '/ocr17_' in rel else 'reichsanzeiger'),
                'split': split,
                'image_path': str((TRAIN_DATA / rel).relative_to(ROOT)),
                'gold': txt,
            })

    missing = [r for r in lines if not (ROOT / r['image_path']).exists()]
    if missing:
        raise SystemExit(f'{len(missing)} crops do not exist on disk, '
                         f'e.g.: {missing[0]["image_path"]}')

    dest_ds = ROOT / 'datasets' / a.name
    dest_ds.mkdir(parents=True, exist_ok=True)
    fields = ['id', 'book', 'corpus', 'split', 'image_path', 'gold']
    with open(dest_ds / 'manifest.csv', 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(lines)

    # synthetic detect.jsonl, one crop per page; elapsed is 0 (no detection)
    dest_res = ROOT / 'results' / a.name / 'raw'
    dest_res.mkdir(parents=True, exist_ok=True)
    with open(dest_res / 'detect.jsonl', 'w', encoding='utf-8') as fh:
        for r in lines:
            fh.write(json.dumps({
                'id': r['id'],
                'provider': 'ground truth',
                'lines': [{'image_path': to_key(ROOT / r['image_path']),
                           'box': None, 'index': 0}],
                'elapsed': 0.0,
            }, ensure_ascii=False) + '\n')

    import collections
    per = collections.Counter((r['corpus'], r['split']) for r in lines)
    print(f'dataset `{a.name}`: {len(lines)} crops '
          f'({len({r["book"] for r in lines})} books)')
    for (c, f), n in sorted(per.items()):
        print(f'    {c:16} {f:5} {n:6}')
    print(f'  manifest: {dest_ds / "manifest.csv"}')
    print(f'  detect  : {dest_res / "detect.jsonl"}')
    print(f'\nnext: python inference_pipeline/recognize.py --dataset {a.name} '
          f'--engine paddle   (and calamari, kraken)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""The evaluation points of the column-tuning logs as a CSV, one arm per run header.

Scored on the training validation split without `canon`, so not comparable with the tables.

Usage: python analyses/training_curve.py LOG [LOG ...] > curve.csv
"""
import csv
import io
import re
import sys

# matches the Portuguese archived logs and the English ones of train_class.py
POINT = re.compile(
    r'(?:ép|ep)\s+(\d+)\s+(?:passo|step)\s+(\d+)\s+(?:perda|loss)\s+([\d.]+)'
    r'\s+‖col‖\s+([\d.]+)\s+CER\s+([\d.]+)\s+(?:emissão|emission)\s+([\d.]+)'
    r'\s+(\d+):(\d+)')
HEADER = re.compile(r'^===.*?(TRES CLASSES|CLASSE UNICA|PROJECAO INTEIRA)', re.I)
ARM = {'tres classes': 'three_columns',
       'classe unica': 'single_class',
       'projecao inteira': 'projection_full'}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    out = csv.writer(sys.stdout)
    out.writerow(['arm', 'epoch', 'step', 'loss', 'col_norm', 'cer',
                  'emission', 'hits', 'damage'])
    for path in sys.argv[1:]:
        arm = None
        for line in io.open(path, encoding='utf-8', errors='replace'):
            head = HEADER.match(line)
            if head:
                arm = ARM[head.group(1).lower()]
                continue
            point = POINT.search(line)
            if point and arm:
                out.writerow([arm, *point.groups()])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Builds the fine-tune set from the ground-truth PAGE/ALTO XML, cropping each
`TextLine` by its polygon, split by percentage stratified by the long s.

Only `pageXmlTranskribusCor*ected` is ground truth; `pageXmlTranskribus` is raw HTR.

Usage, from experiments/:
     ../providers/paddle/.venv/bin/python finetune/extract_lines.py \
         --output finetune/train_data [--height 64] [--val 0.1] [--test 0.1]
"""
import argparse
import csv
import glob
import hashlib
import os
import random
import unicodedata
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
RAW = 'data/prepare'
DATA = 'experiments/datasets'
LONG_S = 'ſʃ∫' + chr(0xEBA6) + chr(0xEBA2)

CORPORA = {
    'ocr17': {
        'xml': f'{RAW}/ocr17_raw/Data/*/pageXmlTranskribusCor*ected/*.xml',
        'images': [f'{RAW}/ocr17_raw/Data/{{book}}/png/{{stem}}.png',
                   f'{DATA}/ocr17/pages/{{stem}}.png'],
        # eScriptorium ALTO4, only for the four books with no corrected PAGE
        'alto': f'{RAW}/ocr17_raw/Data/*/alto4eScriptorium/*.xml',
        'alto_images': [f'{RAW}/ocr17_raw/Data/{{book}}/img/{{stem}}.jpg',
                        f'{RAW}/ocr17_raw/Data/{{book}}/jpg/{{stem}}.jpg',
                        f'{RAW}/ocr17_raw/Data/{{book}}/png/{{stem}}.png'],
        'book': lambda f: f.split('/Data/')[1].split('/')[0],
    },
    'reichsanzeiger': {
        'xml': f'{RAW}/reichsanzeiger_raw/data/reichsanzeiger-1820-1939/GT-PAGE/*.xml',
        'images': [f'{DATA}/reichsanzeiger/pages/{{stem}}.png',
                   f'{RAW}/reichsanzeiger_raw/data/images/{{stem}}.jpg'],
        'book': lambda f: os.path.basename(f)[:4],      # year
    },
}


def line_text(ln, ns):
    """Line-level TextEquiv first, then the Words (OCR17+ has text only per word)."""
    for te in ln.findall(f'{{{ns}}}TextEquiv'):
        u = te.find(f'{{{ns}}}Unicode')
        if u is not None and u.text and u.text.strip():
            return u.text.strip()
        t = ''.join(te.itertext()).strip()
        if t:
            return t
    words = []
    for w in ln.findall(f'{{{ns}}}Word'):
        te = w.find(f'{{{ns}}}TextEquiv')
        if te is None:
            continue
        u = te.find(f'{{{ns}}}Unicode')
        t = (u.text if u is not None and u.text else ''.join(te.itertext())).strip()
        if t:
            words.append(t)
    return ' '.join(words)


def polygon(ln, ns):
    co = ln.find(f'{{{ns}}}Coords')
    if co is None:
        return None
    pts = co.get('points')
    if pts:
        return [tuple(int(float(v)) for v in p.split(',')) for p in pts.split()]
    out = [(int(p.get('x')), int(p.get('y'))) for p in co.findall(f'{{{ns}}}Point')]
    return out or None


def alto_text(ln, ns):
    """ALTO4 keeps the text in String/@CONTENT, one or more per line."""
    parts = [(s.get('CONTENT') or '').strip() for s in ln.findall(f'{{{ns}}}String')]
    return ' '.join(p for p in parts if p)


def alto_polygon(ln, ns):
    """Shape/Polygon/@POINTS as `x y x y ...` (PAGE uses commas); falls back to
    the HPOS/VPOS/WIDTH/HEIGHT box."""
    shape = ln.find(f'{{{ns}}}Shape')
    if shape is not None:
        pg = shape.find(f'{{{ns}}}Polygon')
        if pg is not None and pg.get('POINTS'):
            v = [int(float(x)) for x in pg.get('POINTS').split()]
            if len(v) >= 6:
                return list(zip(v[0::2], v[1::2]))
    try:
        x, y = float(ln.get('HPOS')), float(ln.get('VPOS'))
        w, h = float(ln.get('WIDTH')), float(ln.get('HEIGHT'))
    except (TypeError, ValueError):
        return None
    return [(int(x), int(y)), (int(x + w), int(y)),
            (int(x + w), int(y + h)), (int(x), int(y + h))]


def crop_line(img, poly, height, margin=2):
    """Crop by the polygon, with everything outside it in white. Returns None if
    the region is degenerate."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = max(0, min(xs) - margin), min(img.width, max(xs) + margin)
    y0, y1 = max(0, min(ys) - margin), min(img.height, max(ys) + margin)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = img.crop((x0, y0, x1, y1)).convert('RGB')
    mask = Image.new('L', crop.size, 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in poly], fill=255)
    background = Image.new('RGB', crop.size, (255, 255, 255))
    background.paste(crop, mask=mask)
    width = max(8, round(background.width * height / background.height))
    return background.resize((width, height), Image.LANCZOS)


def image_path(cfg, xml_file, stem, fmt='page'):
    book = cfg['book'](xml_file)
    patterns = (cfg['alto_images'] if fmt == 'alto' and 'alto_images' in cfg
                else cfg['images'])
    # Pradon1697: page_correctedNAME_0009.xml against NAME_0009.png
    stems = [stem]
    if stem.startswith('page_corrected'):
        stems.append(stem[len('page_corrected'):])
    for s in stems:
        for pattern in patterns:
            p = ROOT / pattern.format(book=book, stem=s)
            if p.exists():
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--val', type=float, default=0.10)
    ap.add_argument('--test', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=20260803)
    ap.add_argument('--corpora', default='ocr17,reichsanzeiger')
    ap.add_argument('--limit', type=int, default=0,
                    help='pilot: N lines per corpus')
    a = ap.parse_args()

    out = Path(a.output)
    (out / 'lines').mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    records = []
    no_image = degenerate = empty = reused = 0

    for name in a.corpora.split(','):
        cfg = CORPORA[name]
        files = [(f, 'page')
                 for f in sorted(glob.glob(str(ROOT / cfg["xml"])))]
        if cfg.get('alto'):
            with_page = {cfg['book'](f) for f, _ in files}
            extra = [(f, 'alto') for f in sorted(glob.glob(str(ROOT / cfg['alto'])))
                     if cfg['book'](f) not in with_page]
            if extra:
                books = sorted({cfg['book'](f) for f, _ in extra})
                print(f'{name}: {len(extra)} XML in ALTO4, from {len(books)} books '
                      f'with no PAGE: {", ".join(l.split("_")[0] for l in books)}')
            files += extra
        print(f'{name}: {len(files)} ground truth XML')
        n = 0
        for f, fmt in files:
            if a.limit and n >= a.limit:
                break
            try:
                root = ET.parse(f).getroot()
            except ET.ParseError:
                continue
            ns = root.tag.split('}')[0].strip('{')
            stem = os.path.splitext(os.path.basename(f))[0]
            img_path = image_path(cfg, f, stem, fmt)
            if img_path is None:
                no_image += 1
                continue
            img = Image.open(img_path)
            for i, ln in enumerate(root.iter(f'{{{ns}}}TextLine')):
                if a.limit and n >= a.limit:
                    break
                raw = (alto_text(ln, ns) if fmt == 'alto'
                       else line_text(ln, ns))
                txt = unicodedata.normalize('NFC', raw)
                if not txt:
                    empty += 1
                    continue
                poly = (alto_polygon if fmt == 'alto' else polygon)(ln, ns)
                if not poly or len(poly) < 3:
                    degenerate += 1
                    continue
                # stable crop name; ALTO stems repeat across books, so the book enters the key
                key_seed = (f'{name}/{cfg["book"](f)}/{stem}/{i}'
                            if fmt == 'alto' else f'{name}/{stem}/{i}')
                key = hashlib.md5(key_seed.encode()).hexdigest()[:16]
                rel = f'lines/{name}_{key}.jpg'
                if (out / rel).exists():
                    reused += 1
                else:
                    crop = crop_line(img, poly, a.height)
                    if crop is None:
                        degenerate += 1
                        continue
                    crop.save(out / rel, 'JPEG', quality=92)
                records.append({'path': rel, 'text': txt, 'corpus': name,
                                'group': cfg['book'](f),
                                'has_long_s': any(c in txt for c in LONG_S)})
                n += 1
        print(f'  -> {n} lines')

    # split by percentage, stratified by the presence of a long s
    splits = {'train': [], 'val': [], 'test': []}
    for key in (True, False):
        group = [r for r in records if r['has_long_s'] is key]
        rng.shuffle(group)
        n_val = int(len(group) * a.val)
        n_test = int(len(group) * a.test)
        splits['val'] += group[:n_val]
        splits['test'] += group[n_val:n_val + n_test]
        splits['train'] += group[n_val + n_test:]

    for name, items in splits.items():
        rng.shuffle(items)
        with open(out / f'{name}_list.txt', 'w', encoding='utf-8') as fh:
            for r in items:
                fh.write(f"{r['path']}\t{r['text']}\n")
    with open(out / 'records.csv', 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['path', 'text', 'corpus', 'group',
                                           'has_long_s'])
        w.writeheader()
        for name, items in splits.items():
            for r in items:
                w.writerow(r)

    print(f'\ndiscarded: {no_image} pages with no image, '
          f'{degenerate} lines with a degenerate polygon, {empty} with no text')
    print(f'crops reused from a previous run: {reused}')
    print(f"\n{'split':10}{'lines':>9}{'with long s':>14}{'%':>8}")
    print('-' * 42)
    for name in ('train', 'val', 'test'):
        items = splits[name]
        c = sum(1 for r in items if r['has_long_s'])
        print(f'{name:10}{len(items):>9}{c:>14}{100 * c / len(items) if items else 0:>7.1f}%')
    print(f"\nlists in {out}/; point the config here:")
    print(f"  Train.dataset.data_dir: {out}/")
    print(f"  Train.dataset.label_file_list: [{out}/train_list.txt]")
    print(f"  Eval.dataset.label_file_list:  [{out}/val_list.txt]")


if __name__ == '__main__':
    main()

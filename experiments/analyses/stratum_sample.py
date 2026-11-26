"""Blind sample of pages for the manual check of the ground-truth strata.

Draws the same number of pages from each stratum of `porto` and `impact` with a
fixed seed and writes one HTML sheet with line crops of each page. The sheet
does not show the stratum. The annotator marks whether the print has the long
s, and `--score` compares the exported answers with the strata.

    python3 stratum_sample.py --pages-root DIR --out DIR
    python3 stratum_sample.py --out DIR --score answers.csv
"""
import argparse
import base64
import csv
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from alex_lexicon import stratum

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
CORPORA = ('porto', 'impact')
STRATA = ('diplomatic', 'intermediate', 'modernized')
SEED = 20261123
PER_STRATUM = 10
CROPS = 5
CROP_HEIGHT = 64

csv.field_size_limit(sys.maxsize)

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Stratum sample</title>
<style>
body{font:15px/1.4 sans-serif;max-width:1100px;margin:16px auto;padding:0 16px;
background:#fafafa;color:#111}
.p{background:#fff;border:1px solid #ccc;border-radius:6px;padding:12px;margin:14px 0}
.p.done{border-color:#2a7}
.p img{display:block;max-width:100%;margin:4px 0;border:1px solid #eee}
label{margin-right:18px;cursor:pointer}
#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ccc}
button{font-size:15px;padding:4px 12px}
</style>
<div id="bar"><b id="n"></b> &nbsp; <button onclick="save()">Export answers (CSV)</button></div>
<p>For each page, mark whether the print uses the long s. The crops are lines
of that page. The link opens the whole page.</p>
__ITEMS__
<script>
const K='stratum-sample-__SEED__';
let a={};try{a=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
function paint(){let d=0;document.querySelectorAll('.p').forEach(p=>{
const v=a[p.dataset.id];if(v){d++;p.classList.add('done');
p.querySelector('input[value="'+v+'"]').checked=true}});
document.getElementById('n').textContent=d+' of '+document.querySelectorAll('.p').length}
document.addEventListener('change',e=>{const p=e.target.closest('.p');if(!p)return;
a[p.dataset.id]=e.target.value;try{localStorage.setItem(K,JSON.stringify(a))}catch(e){}paint()});
function save(){let s='item,answer\\n';document.querySelectorAll('.p').forEach(p=>{
s+=p.dataset.id+','+(a[p.dataset.id]||'')+'\\n'});
const l=document.createElement('a');l.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));
l.download='answers.csv';l.click()}
paint();
</script></html>
"""


def pages_by_stratum(corpus):
    out = defaultdict(list)
    with open(RESULTS / corpus / 'pages.csv', encoding='utf-8', newline='') as h:
        for r in csv.DictReader(h):
            st = stratum(r['gold'])
            if st:
                out[st].append(r['id'])
    return out


def draw():
    rng = random.Random(SEED)
    sample = []
    for corpus in CORPORA:
        by = pages_by_stratum(corpus)
        for st in STRATA:
            for page in rng.sample(sorted(by[st]), min(PER_STRATUM, len(by[st]))):
                sample.append({'corpus': corpus, 'page': page, 'stratum': st})
    rng.shuffle(sample)
    for i, s in enumerate(sample, 1):
        s['item'] = f'{i:03d}'
    return sample


def boxes(corpus, wanted):
    out = defaultdict(list)
    with open(RESULTS / corpus / 'raw' / 'detect.jsonl', encoding='utf-8') as h:
        for ln in h:
            for l in json.loads(ln).get('lines', []):
                page = Path(l['image_path']).name.rsplit('__l', 1)[0]
                if page in wanted:
                    out[page].append(l['box'])
    return out


def crops(image, page_boxes):
    from PIL import Image
    img = Image.open(image).convert('L')
    widest = sorted(page_boxes, key=lambda b: b[0] - b[2])[:3 * CROPS]
    chosen = sorted(widest, key=lambda b: b[1])[::3][:CROPS]
    out = []
    for x0, y0, x1, y1 in chosen:
        c = img.crop((int(x0), int(y0), int(x1), int(y1)))
        c = c.resize((max(1, c.width * CROP_HEIGHT // max(1, c.height)), CROP_HEIGHT))
        buf = io.BytesIO()
        c.save(buf, 'JPEG', quality=80)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def build(pages_root, out):
    sample = draw()
    items = []
    for corpus in CORPORA:
        chosen = [s for s in sample if s['corpus'] == corpus]
        bx = boxes(corpus, {s['page'] for s in chosen})
        for s in chosen:
            image = next((pages_root / corpus / 'pages').glob(s['page'] + '.*'))
            s['image'], s['crops'] = image, crops(image, bx[s['page']])
    for s in sorted(sample, key=lambda s: s['item']):
        imgs = ''.join(f'<img src="data:image/jpeg;base64,{c}">' for c in s['crops'])
        radios = ''.join(
            f'<label><input type="radio" name="r{s["item"]}" value="{v}"> {t}</label>'
            for v, t in (('yes', 'has long s'), ('no', 'no long s'),
                         ('unclear', 'cannot tell')))
        items.append(f'<div class="p" data-id="{s["item"]}"><b>{s["item"]}</b> '
                     f'<a href="{s["image"].as_uri()}" target="_blank">whole page</a>'
                     f'{imgs}<div>{radios}</div></div>')
    out.mkdir(parents=True, exist_ok=True)
    html = PAGE.replace('__ITEMS__', '\n'.join(items)).replace('__SEED__', str(SEED))
    (out / 'sheet.html').write_text(html, encoding='utf-8')
    with open(out / 'key.csv', 'w', encoding='utf-8', newline='') as h:
        w = csv.writer(h)
        w.writerow(['item', 'corpus', 'page', 'stratum'])
        for s in sorted(sample, key=lambda s: s['item']):
            w.writerow([s['item'], s['corpus'], s['page'], s['stratum']])
    print(f'{len(sample)} pages -> {out / "sheet.html"}')


def score(out, answers):
    key = {r['item']: r for r in csv.DictReader(open(out / 'key.csv', encoding='utf-8'))}
    tab = defaultdict(Counter)
    for r in csv.DictReader(open(answers, encoding='utf-8')):
        k = key[r['item'].zfill(3)]
        tab[(k['corpus'], k['stratum'])][r['answer'] or 'blank'] += 1
    print(f'{"corpus":8}{"stratum":14}{"yes":>5}{"no":>5}{"unclear":>9}{"blank":>7}')
    for corpus in CORPORA:
        for st in STRATA:
            c = tab[(corpus, st)]
            print(f'{corpus:8}{st:14}{c["yes"]:5}{c["no"]:5}{c["unclear"]:9}{c["blank"]:7}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pages-root', type=Path,
                    help='directory with <corpus>/pages/<page id>.<ext>')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--score', type=Path, help='answers.csv exported by the sheet')
    a = ap.parse_args()
    if a.score:
        score(a.out, a.score)
    else:
        build(a.pages_root, a.out)


if __name__ == '__main__':
    main()

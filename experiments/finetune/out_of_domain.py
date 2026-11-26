#!/usr/bin/env python3
"""Transfer of the column fine-tune to unseen languages, on the pipeline's own
crops, stratified by the fraction of `ſ` in each page's gold.

Usage:
    python finetune/out_of_domain.py --corpus porto --label paddle       # Table 7
    python finetune/out_of_domain.py --corpus porto --label 3classes
    python finetune/out_of_domain.py --corpus porto --label paddle --fusion calamari
    python finetune/out_of_domain.py --corpus porto --column ep2 --bias 2
    python finetune/out_of_domain.py --corpus porto           # base, reference
"""
import argparse
import collections
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # .../experiments

csv.field_size_limit(10 ** 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'PP-OCRv6_medium_rec_hist.yml'))
    ap.add_argument('--weights', default=str(HERE / 'PP-OCRv6_medium_rec_pretrained.pdparams'))
    ap.add_argument('--output', default=str(HERE / 'out_class'))
    ap.add_argument('--corpus', default='porto',
                    help='any dataset with `results/<ds>/lines.csv`. Besides '
                         '`porto` and `impact`, it serves the external '
                         'Vesalainen corpus, where each "page" is one line, so '
                         'reassembling by `line_idx` is the identity there.')
    ap.add_argument('--column', default=None,
                    help='checkpoint name; empty = base')
    ap.add_argument('--bias', type=float, default=0.0)
    ap.add_argument('--fusion', default=None, metavar='SPECIALIST',
                    help='instead of the fine-tuned model, applies rule 1 of the '
                         'fusion over the base text, consulting this column of '
                         'lines.csv (calamari or kraken), so that both methods '
                         'share base text, alignment, canon and stratum.')
    ap.add_argument('--provider', nargs='?', const='base', default=None,
                    metavar='DIR',
                    help='uses the same inference path as the pipeline '
                         '(paddlex.create_model, padding by the batch width) '
                         'instead of the local loop, which resizes to a fixed '
                         '48x960. With no argument, official weights; with a '
                         'directory, the exported model. The two paths diverge '
                         'on 28.4% of the crops.')
    ap.add_argument('--label', default=None,
                    help='reads the text from `results/<corpus>/raw/<label>.jsonl` '
                         'instead of running the recognizer. This is the path of '
                         'the paper: the same outputs every other table uses, and '
                         'no engine, image or weight is needed. `paddle` is the '
                         'baseline, `3classes` and `1class` the tuned models.')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--pages', type=int, default=0, help='0 = all of them')
    ap.add_argument('--class-index', dest='class_index', type=int, default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT.parent))
    from rapidfuzz.distance import Levenshtein
    from fusion import canon

    model = post = None
    if a.label:
        label = f'deposit({a.label})'
    else:
        import paddlex
        repo = (Path(paddlex.__file__).parent / 'repo_manager' / 'repos' / 'PaddleOCR')
        if not (repo / 'ppocr').is_dir():
            sys.exit(f'PaddleOCR not found at {repo}')
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(ROOT))

        import cv2
        import numpy as np
        import paddle
        import yaml
        from ppocr.utils.save_load import load_model
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from ppocr.data.imaug.rec_img_aug import RecResizeImg

        cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
        dic = canon.charset_list()
        fc_target = (a.class_index if a.class_index is not None else dic.index('ſ')) + 1

        d = cfg['Global'].get('character_dict_path')
        if d and not Path(d).is_absolute() and not Path(d).exists():
            cfg['Global']['character_dict_path'] = str(repo / d)
        # provider mode: PaddleX recognizes, no ppocr model is built
        label = f'provider({a.provider})' if a.provider else 'base'
        if not a.provider:
            post = build_post_process(cfg['PostProcess'], cfg['Global'])
            n_chars = len(getattr(post, 'character'))
            cfg['Architecture']['Head']['out_channels_list'] = {
                'CTCLabelDecode': n_chars, 'NRTRLabelDecode': n_chars + 3}
            model = build_model(cfg['Architecture'])
            if not Path(a.weights).exists():
                sys.exit(f'weights not found: {a.weights}')
            cfg['Global']['pretrained_model'] = a.weights[:-len('.pdparams')] \
                if a.weights.endswith('.pdparams') else a.weights
            load_model(cfg, model, model_type=cfg['Architecture']['model_type'])
            if a.column:
                model.set_state_dict(
                    paddle.load(str(Path(a.output) / f'{a.column}.pdparams')))
                label = a.column
            model.eval()
    if a.fusion:
        label += f' + fusion rule 1 ({a.fusion})'
    print(f'== {a.corpus} | {label}'
          + ('' if a.fusion else f' | bias {a.bias:+g}'), flush=True)

    # the lines the pipeline cut
    dres = ROOT / 'results' / a.corpus
    meta = {r['id']: r for r in
            csv.DictReader(open(dres / 'pages.csv', encoding='utf-8'))}
    pages = {k: r['gold'] for k, r in meta.items()}
    lines = collections.defaultdict(list)
    spec = {}                                # (page, line) -> specialist
    for r in csv.DictReader(open(dres / 'lines.csv', encoding='utf-8')):
        lines[r['page_id']].append((int(r['line_idx']), r['line_path']))
        if a.fusion:
            if a.fusion not in r:
                sys.exit(f'column {a.fusion!r} does not exist in lines.csv; '
                         f'there is {[c for c in r]}')
            spec[(r['page_id'], int(r['line_idx']))] = r[a.fusion] or ''
    ids = [p for p in pages if p in lines]
    if a.pages:
        ids = ids[:a.pages]
    for k in lines:
        lines[k].sort()
    plan = [(p, i, c) for p in ids for i, c in lines[p]]
    print(f'{len(ids)} pages, {len(plan)} lines', flush=True)

    op = None if a.provider or a.label else RecResizeImg(image_shape=[3, 48, 960])

    # cv2.imread releases the GIL, so threads parallelize the decoding
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=6)

    from fusion.paths import to_path

    def load(c):
        img = cv2.imread(str(to_path(c)))
        return None if img is None else op({'image': img})['image']

    def recognize(paths):
        batch = list(pool.map(load, paths))
        good = [i for i, x in enumerate(batch) if x is not None]
        out = [''] * len(batch)
        if not good:
            return out
        x = paddle.to_tensor(np.stack([batch[i] for i in good]))
        with paddle.no_grad():
            pred_out = model(x)
            pred = pred_out['ctc'] if isinstance(pred_out, dict) else pred_out
            if a.bias:
                # softmax output: multiplying by exp(v) adds v to the logit
                pred = pred.clone()
                pred[:, :, fc_target] *= math.exp(a.bias)
            txt = post(pred, None)
        for i, (t, _) in zip(good, txt):
            out[i] = t
        return out

    if a.label:
        sys.path.insert(0, str(ROOT / 'analyses'))
        from glyph import text_per_crop
        by_crop = text_per_crop(dres / 'raw', a.label)
        if by_crop is None:
            sys.exit(f'no raw/{a.label}*.jsonl in {dres}')
        text = {(p_, idx): by_crop.get(c, '') for p_, idx, c in plan}
    elif a.provider:
        from paddlex import create_model
        md = None if a.provider == 'base' else a.provider
        rec = create_model('PP-OCRv6_medium_rec') if md is None \
            else create_model('PP-OCRv6_medium_rec', model_dir=md)
        text = {}
        paths = [c for _, _, c in plan]
        for (p_, idx, _), out in zip(plan, rec.predict(paths)):
            text[(p_, idx)] = out['rec_text']
        del rec
    else:
        text = {}
        for i in range(0, len(plan), a.batch):
            block = plan[i:i + a.batch]
            for (p, idx, _), t in zip(block, recognize([c for _, _, c in block])):
                text[(p, idx)] = t
    if a.fusion:
        from dataclasses import replace
        from fusion.common import DEFAULT, merge_line
        # rule 1 only, over the same base text: the column fine-tune changes only the glyph
        rules = replace(DEFAULT, space=False)
        st = {}
        for key, base in list(text.items()):
            e = spec.get(key, '')
            if e:
                text[key] = merge_line(base, e, st, rules=rules)
        print(f'rule 1 fired {st.get("swapped_f_for_s", 0)} times; '
              f'rule 2 injected {st.get("injected_space", 0)} spaces',
              flush=True)

    # strata by the fraction of ſ among the s of the page's own gold (the PORTO gold is bimodal)
    def stratum(g):
        s, long_s = g.count('s'), g.count('ſ')
        if s + long_s == 0:
            return None                    # page with no s: it says nothing
        f = long_s / (s + long_s)
        return 'diplomatic' if f >= 0.30 else (
            'modernized' if f < 0.02 else 'intermediate')

    def period(p):
        """50-year bin, when the corpus carries a date. `ſ` falls out of use over
        the 19th century, so mixing 1700 with 1900 mixes two typographies."""
        d = (meta[p].get('date') or '').strip()[:4]
        return f'{int(d) // 50 * 50}s' if d.isdigit() else None

    T = collections.defaultdict(lambda: dict(dist=0, n_char=0, gold=0, emitted=0,
                                             ok=0, bad=0, pages=0))
    E = collections.defaultdict(lambda: dict(dist=0, n_char=0, gold=0, emitted=0,
                                             ok=0, bad=0, pages=0))
    # confusion profile, diplomatic stratum only: goes = gold ſ -> hypothesis, comes = hypothesis ſ -> gold
    goes, comes = collections.Counter(), collections.Counter()
    for p in ids:
        g = canon.canonicalize(pages[p], space='page')
        h = canon.canonicalize('\n'.join(text.get((p, i), '') for i, _ in lines[p]),
                               space='page')
        e = stratum(g)
        if e is None:
            continue
        ep = period(p)
        targets = [T[e], T['TOTAL']]
        if e == 'diplomatic' and ep:
            targets.append(E[ep])
        for t in targets:
            t['pages'] += 1
            t['dist'] += Levenshtein.distance(g, h)
            t['n_char'] += len(g)
            t['gold'] += g.count('ſ')
            t['emitted'] += h.count('ſ')
        ops = Levenshtein.opcodes(g, h)
        for tag, i1, i2, j1, j2 in ops:
            if tag == 'equal':
                n = g[i1:i2].count('ſ')
                for t in targets:
                    t['ok'] += n
            elif tag in ('replace', 'insert'):
                n = h[j1:j2].count('ſ')
                for t in targets:
                    t['bad'] += n
        if e != 'diplomatic':
            continue
        for tag, i1, i2, j1, j2 in ops:
            if tag == 'equal':
                n = g[i1:i2].count('ſ')
                goes['ſ (right)'] += n
                comes['ſ (right)'] += n
            elif tag == 'replace':
                m = min(i2 - i1, j2 - j1)
                for k in range(m):
                    x, y = g[i1 + k], h[j1 + k]
                    if x == 'ſ':
                        goes[y] += 1
                    if y == 'ſ':
                        comes[x] += 1
                goes['(deleted)'] += g[i1 + m:i2].count('ſ')
                comes['(inserted)'] += h[j1 + m:j2].count('ſ')
            elif tag == 'delete':
                goes['(deleted)'] += g[i1:i2].count('ſ')
            elif tag == 'insert':
                comes['(inserted)'] += h[j1:j2].count('ſ')

    print(f'\n{"stratum":16}{"pages":>6}{"ſ gold":>9}{"CER":>9}{"emission":>10}'
          f'{"recall":>8}{"precision":>11}{"ratio":>9}')
    print('-' * 78)
    for key in ('diplomatic', 'intermediate', 'modernized', 'TOTAL'):
        t = T.get(key)
        if not t or not t['n_char']:
            continue
        rec = t['ok'] / t['gold'] if t['gold'] else 0.0
        prec = t['ok'] / (t['ok'] + t['bad']) if t['ok'] + t['bad'] else 0.0
        ratio = t['ok'] / t['bad'] if t['bad'] else float('inf')
        print(f'{key:16}{t["pages"]:6}{t["gold"]:9}{t["dist"] / t["n_char"]:9.4f}'
              f'{t["emitted"] / t["gold"] if t["gold"] else 0:10.3f}'
              f'{rec:8.3f}{prec:11.3f}{ratio:8.1f}:1')

    if E:
        print('\nby period, within the diplomatic stratum:')
        print(f'{"period":16}{"pages":>6}{"ſ gold":>9}{"CER":>9}{"emission":>10}'
              f'{"recall":>8}{"precision":>11}{"ratio":>9}')
        print('-' * 78)
        for key in sorted(E):
            t = E[key]
            if not t['n_char']:
                continue
            rec = t['ok'] / t['gold'] if t['gold'] else 0.0
            prec = t['ok'] / (t['ok'] + t['bad']) if t['ok'] + t['bad'] else 0.0
            ratio = t['ok'] / t['bad'] if t['bad'] else float('inf')
            print(f'{key:16}{t["pages"]:6}{t["gold"]:9}{t["dist"] / t["n_char"]:9.4f}'
                  f'{t["emitted"] / t["gold"] if t["gold"] else 0:10.3f}'
                  f'{rec:8.3f}{prec:11.3f}{ratio:8.1f}:1')

    gold_d = T['diplomatic']['gold'] or 1
    emitted_d = sum(n for n in comes.values()) or 1
    print('\nin the diplomatic stratum, the ground truth has ſ and the hypothesis wrote:')
    for c, n in goes.most_common(8):
        print(f'  {c!r:14} {n:7}  {n / gold_d * 100:5.1f}%')
    print('\nin the diplomatic stratum, the hypothesis wrote ſ and the ground truth had:')
    for c, n in comes.most_common(8):
        print(f'  {c!r:14} {n:7}  {n / emitted_d * 100:5.1f}%')


if __name__ == '__main__':
    main()

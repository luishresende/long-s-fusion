#!/usr/bin/env python3
"""Evaluation of a column checkpoint: CER, emission ratio and the `ſ` confusions
in both directions, raw output against the training label (no `canon`).

Usage:
    python finetune/eval_class.py --column finetune/out_class/ep1.pdparams
    python finetune/eval_class.py            # the base, for reference
"""
import argparse
import collections
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'PP-OCRv6_medium_rec_hist.yml'))
    ap.add_argument('--weights', default=str(HERE / 'PP-OCRv6_medium_rec_pretrained.pdparams'))
    ap.add_argument('--column', default=None,
                    help='checkpoint with ctc_head.fc.{weight,bias}; '
                         'absent = base model')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0,
                    help='0 = the whole set')
    ap.add_argument('--bias', type=float, default=0.0,
                    help='adds this constant to the ſ logit before the argmax. '
                         'Negative gives precision back in exchange for recall, '
                         'and sweeps the whole curve with one model, without '
                         'retraining.')
    ap.add_argument('--class-index', dest='class_index', type=int, default=None)
    a = ap.parse_args()

    import paddlex
    repo = (Path(paddlex.__file__).parent / 'repo_manager' / 'repos' / 'PaddleOCR')
    if not (repo / 'ppocr').is_dir():
        sys.exit(f'PaddleOCR not found at {repo}')
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(HERE.parent.parent))
    from fusion import canon

    import paddle
    import yaml
    from rapidfuzz.distance import Levenshtein
    from ppocr.utils.save_load import load_model
    from ppocr.modeling.architectures import build_model
    from ppocr.data import build_dataloader
    from ppocr.postprocess import build_post_process
    from ppocr.utils.logging import get_logger

    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    log = get_logger()

    dic = canon.charset_list()
    fc_target = (a.class_index if a.class_index is not None else dic.index('ſ')) + 1

    d = cfg['Global'].get('character_dict_path')
    if d and not Path(d).is_absolute() and not Path(d).exists():
        cfg['Global']['character_dict_path'] = str(repo / d)
    for split in ('Train', 'Eval'):
        if split in cfg:
            ds = cfg[split]['dataset']
            ds['data_dir'] = str(HERE / Path(ds['data_dir']).name)
            ds['label_file_list'] = [str(HERE / 'train_data' / Path(x).name)
                                     for x in ds['label_file_list']]

    post = build_post_process(cfg['PostProcess'], cfg['Global'])
    n_chars = len(getattr(post, 'character'))
    cfg['Architecture']['Head']['out_channels_list'] = {
        'CTCLabelDecode': n_chars, 'NRTRLabelDecode': n_chars + 3}
    model = build_model(cfg['Architecture'])
    cfg['Global']['pretrained_model'] = a.weights[:-len('.pdparams')] \
        if a.weights.endswith('.pdparams') else a.weights
    load_model(cfg, model, model_type=cfg['Architecture']['model_type'])

    label = 'BASE (official weights)'
    if a.column:
        col = paddle.load(a.column)
        model.set_state_dict(col)
        label = Path(a.column).stem
    if a.bias:
        label += f' + bias {a.bias:+g}'
    print(f'== {label}')

    cfg['Eval']['loader']['num_workers'] = 4
    cfg['Eval']['loader']['batch_size_per_card'] = a.batch
    eval_loader = build_dataloader(cfg, 'Eval', paddle.device.get_device(), log)

    dist = n_char = 0
    gold_s = emitted = ok_s = 0
    goes_to = collections.Counter()      # ground truth has ſ -> hypothesis wrote
    comes_from = collections.Counter()   # hypothesis has ſ -> ground truth had
    model.eval()
    with paddle.no_grad():
        for i, batch in enumerate(eval_loader):
            if a.limit and i >= a.limit:
                break
            out = model(batch[0])
            pred = out['ctc'] if isinstance(out, dict) else out
            if a.bias:
                # softmax output: multiplying by exp(bias) adds bias to the logit
                pred = pred.clone()
                pred[:, :, fc_target] *= math.exp(a.bias)
            txt = post(pred, None)
            labels = [''.join(post.character[k] for k in row if k > 0)
                      for row in batch[1].numpy()]
            for (h, _), g in zip(txt, labels):
                dist += Levenshtein.distance(g, h)
                n_char += len(g)
                gold_s += g.count('ſ')
                emitted += h.count('ſ')
                for tag, i1, i2, j1, j2 in Levenshtein.opcodes(g, h):
                    if tag == 'equal':
                        n = g[i1:i2].count('ſ')
                        ok_s += n
                        goes_to['ſ (right)'] += n
                        comes_from['ſ (right)'] += n
                    elif tag == 'replace':
                        m = min(i2 - i1, j2 - j1)
                        for k in range(m):
                            x, y = g[i1 + k], h[j1 + k]
                            if x == 'ſ':
                                goes_to[y] += 1
                            if y == 'ſ':
                                comes_from[x] += 1
                        for x in g[i1 + m:i2]:
                            if x == 'ſ':
                                goes_to['(deleted)'] += 1
                        for y in h[j1 + m:j2]:
                            if y == 'ſ':
                                comes_from['(inserted)'] += 1
                    elif tag == 'delete':
                        goes_to['(deleted)'] += g[i1:i2].count('ſ')
                    elif tag == 'insert':
                        comes_from['(inserted)'] += h[j1:j2].count('ſ')

    print(f'CER            {dist / n_char:.4f}  ({dist} edits / {n_char} characters)')
    print(f'ſ in the gold  {gold_s}')
    print(f'ſ emitted      {emitted}   emission ratio {emitted / gold_s:.3f}')
    print(f'aligned recall {ok_s / gold_s:.4f}')

    print('\nwhen the ground truth has ſ, the hypothesis wrote:')
    for c, n in goes_to.most_common(8):
        print(f'  {c!r:14} {n:7}  {n / gold_s * 100:5.1f}%')

    # the damage is everything that became ſ without being ſ
    damage = sum(n for c, n in comes_from.items() if c != 'ſ (right)')
    hits = comes_from['ſ (right)']
    print(f'\nwhen the hypothesis has ſ, the ground truth had:')
    for c, n in comes_from.most_common(8):
        print(f'  {c!r:14} {n:7}  {n / max(emitted, 1) * 100:5.1f}%')
    print(f'\nhit {hits} : damage {damage} = '
          f'{hits / damage if damage else float("inf"):.1f}:1   '
          f'precision {hits / (hits + damage) * 100 if hits + damage else 0:.2f}%')


if __name__ == '__main__':
    main()

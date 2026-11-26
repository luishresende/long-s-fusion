#!/usr/bin/env python3
"""Fine-tune of selected columns of the CTC projection (`ſ` by default, 193
parameters per class); everything else is restored after every step.

Usage:
    python finetune/train_class.py --epochs 3 --batch 8
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'PP-OCRv6_medium_rec_hist.yml'))
    ap.add_argument('--weights', default=str(HERE / 'PP-OCRv6_medium_rec_pretrained.pdparams'))
    ap.add_argument('--output', default=str(HERE / 'out_class'))
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='larger than in the full fine-tune (5e-05) on purpose: '
                         'these are 193 parameters, not a whole model')
    ap.add_argument('--accum', type=int, default=1,
                    help='micro-batches summed before one optimizer step. '
                         'Effective batch = --batch x --accum. It is how the '
                         'large batch of multi-GPU training is obtained on a '
                         'single GPU.')
    ap.add_argument('--eval-every', dest='eval_every', type=int, default=100,
                    help='optimizer steps between evaluations')
    ap.add_argument('--steps', type=int, default=3000,
                    help='step ceiling; 200 already yielded recall 0.55, so the '
                         '27 thousand of a whole epoch are waste')
    ap.add_argument('--eval-batch', dest='eval_batch', type=int, default=None,
                    help='evaluation batch; by default it follows --batch. Fix '
                         'it when comparing runs: the PP-OCRv6 output depends on '
                         'the composition of the batch, so changing the '
                         'evaluation batch changes the CER of the same model.')
    ap.add_argument('--lr-neck', dest='lr_neck', type=float, default=None,
                    help='rate for the neck only; by default it uses --lr. The '
                         'two parts need very different scales: the `fc` works '
                         'well at 5e-4 and the neck already degrades at 1e-5. A '
                         'single rate serves one and destroys the other.')
    ap.add_argument('--neck', action='store_true',
                    help='unfreezes the CTC neck (`head.ctc_encoder`, 1,188,672 '
                         'parameters). It is where shape becomes contextual '
                         'representation; the `fc` only labels the 192-dim '
                         'vector it delivers. Fine-tuning `fc` columns is a '
                         'linear probe over frozen features, and it saturates '
                         'at ~61% recall out of domain.')
    ap.add_argument('--init-column', dest='init_column', default=None,
                    help='`fc` checkpoint to load before training, to start from '
                         'the already fine-tuned model instead of the official '
                         'weights.')
    ap.add_argument('--classes', default='ſ',
                    help='characters to unfreeze, comma-separated. `ſ` alone is '
                         'the minimal experiment; `ſ,f,s` lets the optimizer '
                         'adjust the boundary from both sides instead of only '
                         'inflating ſ against everything else. `all` releases '
                         'the whole projection, which is the control condition: '
                         'it measures what is lost by not restricting.')
    a = ap.parse_args()

    # ppocr is imported directly: the PaddleX trainer forces MSTextRecDataset
    import paddlex
    repo = (Path(paddlex.__file__).parent / 'repo_manager' / 'repos' / 'PaddleOCR')
    if not (repo / 'ppocr').is_dir():
        sys.exit(f'PaddleOCR not found at {repo}')
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(HERE.parent.parent))
    from fusion import canon

    import paddle
    from ppocr.utils.save_load import load_model
    from ppocr.modeling.architectures import build_model
    from ppocr.data import build_dataloader
    from ppocr.losses import build_loss
    from ppocr.postprocess import build_post_process
    from ppocr.utils.utility import set_seed
    from ppocr.utils.logging import get_logger
    import yaml

    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    log = get_logger()
    set_seed(42)

    dic = canon.charset_list()
    # the CTC inserts a blank at position 0, so the row in the matrix is index + 1
    all_classes = a.classes.strip().lower() == 'all'
    if all_classes:
        # control: the whole projection trains, so leak_check() measures drift
        fc_targets = list(range(len(dic) + 1))
        print(f'whole projection released: {len(fc_targets)} columns, '
              f'{192 * len(fc_targets) + len(fc_targets):,} weights')
    else:
        fc_targets = []
        for c in a.classes.split(','):
            c = c.strip()
            if not c:
                continue
            if c not in dic:
                sys.exit(f'character {c!r} is not in the dictionary')
            fc_targets.append(dic.index(c) + 1)
            print(f'target class: {c!r} = U+{ord(c):04X} -> index {dic.index(c)} '
                  f'in the dictionary, row {fc_targets[-1]} of the projection')
    # the metrics always follow ſ, even when the `fc` is frozen
    fc_target = dic.index('ſ') + 1

    # the YAML points at the dictionary by a path relative to the PaddleOCR repo
    d = cfg['Global'].get('character_dict_path')
    if d and not Path(d).is_absolute() and not Path(d).exists():
        cfg['Global']['character_dict_path'] = str(repo / d)
    # the data paths are relative to this directory
    for split in ('Train', 'Eval'):
        if split in cfg:
            ds = cfg[split]['dataset']
            ds['data_dir'] = str(HERE / Path(ds['data_dir']).name)
            ds['label_file_list'] = [str(HERE / 'train_data' / Path(x).name)
                                     for x in ds['label_file_list']]

    post = build_post_process(cfg['PostProcess'], cfg['Global'])
    # NRTR needs 3 more classes than CTC (blank, start and end of sequence)
    n_chars = len(getattr(post, 'character'))
    cfg['Architecture']['Head']['out_channels_list'] = {
        'CTCLabelDecode': n_chars, 'NRTRLabelDecode': n_chars + 3}
    print(f'classes: {n_chars} (CTC) / {n_chars + 3} (NRTR)')
    model = build_model(cfg['Architecture'])
    # without pretrained_model, load_model starts from random weights
    if not Path(a.weights).exists():
        sys.exit(f'weights not found: {a.weights}')
    cfg['Global']['pretrained_model'] = a.weights[:-len('.pdparams')] \
        if a.weights.endswith('.pdparams') else a.weights
    load_model(cfg, model, model_type=cfg['Architecture']['model_type'])
    if a.init_column:
        model.set_state_dict(paddle.load(a.init_column))
        print(f'initial column loaded from {a.init_column}')

    fc = None
    NECK = 'ctc_encoder'
    for name, p in model.named_parameters():
        is_fc = name.endswith('ctc_head.fc.weight') or name.endswith('ctc_head.fc.bias')
        is_neck = NECK in name
        if name.endswith('ctc_head.fc.weight'):
            fc = p                       # always referenced, to measure ‖col‖
        p.stop_gradient = not ((is_fc and fc_targets) or (is_neck and a.neck))
    n_fc = sum(p.numel() for n, p in model.named_parameters()
               if not p.stop_gradient and NECK not in n)
    n_neck = sum(p.numel() for n, p in model.named_parameters()
                 if not p.stop_gradient and NECK in n)
    print(f'with gradient: fc {n_fc:,} (masked down to {len(fc_targets) * 193}) '
          f'| neck {n_neck:,}')
    if not (n_fc or n_neck):
        sys.exit('nothing to train: pass --classes and/or --neck')

    # the fc is restored outside the target columns after each optimizer step
    orig = {n: p.clone().detach() for n, p in model.named_parameters()
            if not p.stop_gradient and NECK not in n}

    # 1-D class mask, broadcast over the (192, 18710) weight
    MASK = {}
    if orig:
        n_out = orig[[n for n in orig if n.endswith('bias')][0]].shape[0]
        _m = paddle.zeros([n_out], dtype='float32')
        for i in fc_targets:
            _m[i] = 1.0
        MASK = {2: _m.reshape([1, n_out]), 1: _m}

    def restore():
        with paddle.no_grad():
            for n, p in model.named_parameters():
                if p.stop_gradient or n not in orig:
                    continue          # the neck is not restored
                m = MASK[len(p.shape)]
                p.set_value(p * m + orig[n] * (1 - m))

    def leak_check():
        """Largest deviation from the official projection outside the target columns; must be 0."""
        worst = 0.0
        with paddle.no_grad():
            for n, p in model.named_parameters():
                if p.stop_gradient or n not in orig:
                    continue
                m = MASK[len(p.shape)]
                worst = max(worst, float((paddle.abs(p - orig[n]) * (1 - m)).max()))
        return worst

    if all_classes:
        print(f'no column restriction: the {len(fc_targets)} fc columns train'
              + ('  (neck free)' if a.neck else ''))
    else:
        print(f'restore installed on {len(orig)} fc tensors: only columns '
              f'{fc_targets} survive each step'
              + ('  (neck free)' if a.neck else ''))
    print(f'  -> {len(fc_targets) * 193} effectively trainable parameters '
          f'({len(fc_targets)} classes x 193)')

    def freeze_bn():
        """BatchNorm statistics are not frozen by `stop_gradient`; call after every `model.train()`."""
        n = 0
        for l in model.sublayers():
            if hasattr(l, '_mean') and hasattr(l, '_variance'):
                l.eval()
                n += 1
        return n

    print(f'BatchNorm frozen: {freeze_bn()} layers in inference mode')

    # SimpleDataSet at a fixed scale: the backbone is frozen
    cfg['Train']['dataset']['name'] = 'SimpleDataSet'
    cfg['Train']['dataset']['transforms'] = cfg['Eval']['dataset']['transforms']
    cfg['Train']['loader']['batch_size_per_card'] = a.batch
    # no MultiScaleSampler: SimpleDataSet has no ds_width
    cfg['Train'].pop('sampler', None)
    cfg['Train']['loader']['shuffle'] = True
    cfg['Train']['loader']['drop_last'] = True
    cfg['Train']['loader']['num_workers'] = 2
    # no eval workers: each evaluation's new iterator leaves worker processes behind
    cfg['Eval']['loader']['num_workers'] = 0
    cfg['Eval']['loader']['batch_size_per_card'] = a.eval_batch or a.batch
    train_loader = build_dataloader(cfg, 'Train', paddle.device.get_device(), log)
    eval_loader = build_dataloader(cfg, 'Eval', paddle.device.get_device(), log)
    print(f'batches: train {len(train_loader)}  eval {len(eval_loader)}')

    loss_fn = build_loss(cfg['Loss'])
    # one optimizer per part: per-group learning_rate differs across Paddle versions
    p_fc = [p for n, p in model.named_parameters()
            if not p.stop_gradient and NECK not in n]
    p_neck = [p for n, p in model.named_parameters()
              if not p.stop_gradient and NECK in n]
    lr_neck = a.lr if a.lr_neck is None else a.lr_neck
    optims = []
    if p_fc:
        optims.append(paddle.optimizer.Adam(learning_rate=a.lr, parameters=p_fc,
                                            beta1=0.9, beta2=0.999))
    if p_neck:
        optims.append(paddle.optimizer.Adam(learning_rate=lr_neck,
                                            parameters=p_neck,
                                            beta1=0.9, beta2=0.999))
    print(f'rates: fc {a.lr:.1e}' + (f' | neck {lr_neck:.1e}' if p_neck else ''))

    col = lambda: float(paddle.linalg.norm(fc[:, fc_target]).numpy())
    print(f'effective batch: {a.batch} x {a.accum} = {a.batch * a.accum} '
          f'| rate {a.lr:.1e} | {len(train_loader) // a.accum} steps per epoch')
    print(f'initial norm of column {fc_target}: {col():.4f}\n')

    def evaluate(limit=0):
        """CER, emission ratio (no alignment) and hit:damage over the eval set."""
        from rapidfuzz.distance import Levenshtein
        model.eval()
        dist = n_char = gold_s = emitted = hits = damage = 0
        with paddle.no_grad():
            for i, batch in enumerate(eval_loader):
                if limit and i >= limit:
                    break
                out = model(batch[0])
                pred = out['ctc'] if isinstance(out, dict) else out
                txt = post(pred, None)
                # labels are class indices (0 = blank = padding)
                labels = [''.join(post.character[i] for i in row if i > 0)
                          for row in batch[1].numpy()]
                for (h, _), g in zip(txt, labels):
                    dist += Levenshtein.distance(g, h)
                    n_char += len(g)
                    gold_s += g.count('ſ')
                    emitted += h.count('ſ')
                    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(g, h):
                        if tag == 'equal':
                            hits += g[i1:i2].count('ſ')
                        elif tag == 'replace':
                            m = min(i2 - i1, j2 - j1)
                            damage += h[j1:j1 + m].count('ſ')
                            damage += h[j1 + m:j2].count('ſ')
                        elif tag == 'insert':
                            damage += h[j1:j2].count('ſ')
        model.train()
        freeze_bn()
        return (dist / n_char if n_char else 0.0,
                emitted / gold_s if gold_s else 0.0,
                hits, damage, gold_s)

    cer, emission, hit, dmg, ns = evaluate()
    print(f'before training: CER {cer:.4f} | emission {emission:.3f} | '
          f'hit {hit} damage {dmg} | {ns} ſ in the validation set\n', flush=True)

    def save(name):
        """Save only `head.ctc_head.fc.{weight,bias}` (and the neck with --neck)."""
        d = {n: p for n, p in model.state_dict().items()
             if n.endswith('ctc_head.fc.weight')
             or n.endswith('ctc_head.fc.bias')
             or (a.neck and NECK in n)}
        paddle.save(d, f'{a.output}/{name}.pdparams')

    step = micro = 0
    Path(a.output).mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        if step >= a.steps:
            break
        t0 = time.time()
        for batch in train_loader:
            if step >= a.steps:
                break
            # the loss gets the full batch: MultiLoss slices it itself
            out = model(batch[0], data=batch[1:])
            p = loss_fn(out, batch)
            # gradient accumulation: effective batch = batch x accum
            (p['loss'] / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            for o in optims:
                o.step()
            for o in optims:
                o.clear_grad()
            restore()        # undoes the step everywhere but in the target column
            step += 1
            if step % a.eval_every == 0:
                cer, emission, hit, dmg, _ = evaluate()
                save(f'p{step}')
                print(f'  ep {ep} step {step:5}  loss {float(p["loss"]):.2f}  '
                      f'‖col‖ {col():.4f}  CER {cer:.4f}  emission {emission:.3f}  '
                      f'{hit}:{dmg} = {hit / dmg if dmg else float("inf"):.1f}:1  '
                      f'leak {leak_check():.1e}  '
                      f'({(time.time()-t0)/60:.1f} min)', flush=True)
        cer, emission, hit, dmg, _ = evaluate()
        print(f'EPOCH {ep}: CER {cer:.4f} | emission {emission:.3f} | '
              f'{hit}:{dmg} = {hit / dmg if dmg else float("inf"):.1f}:1 | '
              f'‖col‖ {col():.4f}', flush=True)
        save(f'ep{ep}')


if __name__ == '__main__':
    main()

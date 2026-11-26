#!/usr/bin/env python3
"""Exports a column checkpoint as a PaddleX inference directory, for `PADDLE_REC_DIR`.

Usage:
    python finetune/export.py --column finetune/out_3classes/ep0.pdparams \
                              --output finetune/inference_3classes_ep0
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'PP-OCRv6_medium_rec_hist.yml'))
    ap.add_argument('--weights', default=str(HERE / 'PP-OCRv6_medium_rec_pretrained.pdparams'))
    ap.add_argument('--column', required=True,
                    help='checkpoint with the fine-tuned columns')
    ap.add_argument('--output', required=True,
                    help='inference directory to create')
    ap.add_argument('--official', default=str(Path.home() /
                    '.paddlex/official_models/PP-OCRv6_medium_rec'),
                    help='official model, where inference.yml is copied from')
    a = ap.parse_args()

    import paddlex
    repo = (Path(paddlex.__file__).parent / 'repo_manager' / 'repos' / 'PaddleOCR')
    if not (repo / 'ppocr').is_dir():
        sys.exit(f'PaddleOCR not found at {repo}')
    sys.path.insert(0, str(repo))

    import paddle
    import yaml
    from ppocr.utils.save_load import load_model
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    d = cfg['Global'].get('character_dict_path')
    if d and not Path(d).is_absolute() and not Path(d).exists():
        cfg['Global']['character_dict_path'] = str(repo / d)
    post = build_post_process(cfg['PostProcess'], cfg['Global'])
    n_chars = len(getattr(post, 'character'))
    cfg['Architecture']['Head']['out_channels_list'] = {
        'CTCLabelDecode': n_chars, 'NRTRLabelDecode': n_chars + 3}

    model = build_model(cfg['Architecture'])
    cfg['Global']['pretrained_model'] = a.weights[:-len('.pdparams')] \
        if a.weights.endswith('.pdparams') else a.weights
    load_model(cfg, model, model_type=cfg['Architecture']['model_type'])
    col = paddle.load(a.column)
    model.set_state_dict(col)
    print(f'columns loaded from {a.column}: {list(col)}')
    model.eval()

    # export the CTC branch only; the NRTR head is training-only
    if hasattr(model.head, 'return_all_feats'):
        model.head.return_all_feats = False

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    shape = [None, 3, 48, None]        # free width, so PaddleX pads per batch
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    paddle.jit.save(model, str(out / 'inference'))
    print(f'program exported to {out}')

    # inference.yml from the official model: same alphabet and pre/post-processing
    official = Path(a.official) / 'inference.yml'
    if not official.exists():
        sys.exit(f'official inference.yml not found at {official}')
    shutil.copy(official, out / 'inference.yml')
    print(f'inference.yml copied from {official}')
    print('\ncontents:')
    for f in sorted(out.iterdir()):
        print(f'  {f.name:24} {f.stat().st_size:>12,} bytes')


if __name__ == '__main__':
    main()

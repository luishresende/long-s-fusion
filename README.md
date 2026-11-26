# long-s-fusion

Code for the paper *Long-s recovery in historical print by asymmetric fusion of
OCR engines* (submitted to the Journal of Imaging).

PP-OCRv6 reads historical print well but never writes the long s (ſ, U+017F). It
writes `f` in its place, although the character is in its vocabulary. This
repository holds two ways of restoring the glyph while keeping PP-OCRv6 as the
author of the text and of the segmentation.

- **Asymmetric fusion.** A specialist engine (Calamari or Kraken) reads the same
  line crops, and it is allowed to change the generalist's output in two ways
  only: turn an `f` into `ſ`, and insert a missing space. Nothing is trained.
- **Column tuning.** Three columns of the output projection of PP-OCRv6 (the
  classes `ſ`, `f` and `s`) are trained, 579 of 3,611,030 parameters, and every
  other weight is kept identical to the official ones.

## Layout

```
fusion/        the fusion rules, the canonical form used for comparison, the metrics
providers/     one environment per OCR engine, behind a common JSON-lines contract
data/prepare/  one acquisition script per corpus (nothing is redistributed)
experiments/
  inference_pipeline/   detection once, then every engine over the same crops
  eval_page.py          page protocol: all systems on detector crops
  eval_goldseg.py       line protocol: all systems on ground-truth crops
  ablation.py           one rule removed at a time, and the per-work filter
  class_impact.py       error classes before and after the fusion
  finetune/             column tuning, export, evaluation outside the training domain
  analyses/             one script per remaining figure of the paper
```

## Using the fusion

The fusion itself has no dependency beyond the standard library and `rapidfuzz`:

```python
from fusion.common import merge_line

merge_line("Obfervações deftas", "Obſervações deſtas")   # -> "Obſervações deſtas"
```

`base` is the generalist's line and `specialist` the specialist's reading of the
same crop. The production rule set is `fusion.common.DEFAULT`.

## Reproducing the paper

`REPLICATION.md` has the full procedure, from an empty machine to each table.
`experiments/analyses/README.md` maps every table and figure to its script.

The engine outputs on the six evaluation sets, the tuned columns, the exported
models, the training lists and the training logs are not in git. They are
archived on Zenodo (https://doi.org/10.5281/zenodo.24547462). With that archive
unpacked, every table can be recomputed on a CPU without running any engine.

The corpora and the third-party models are not redistributed. Each is fetched
from its original source by the scripts in `data/prepare/` and the commands in
`REPLICATION.md`.

## Licence

The code is released under the MIT licence (see `LICENSE`). The corpora and the
third-party models keep their own licences.

## Citation

See `CITATION.cff`. The reference to the article will be added once it is
published.

## Funding

This work was funded by Fundação para a Ciência e a Tecnologia (FCT) within the
project "CompLiTT: Computatio Litterarum limitis: Calculations of literary
culture on the border", grant 2023.11359.PEX
(https://doi.org/10.54499/2023.11359.PEX).

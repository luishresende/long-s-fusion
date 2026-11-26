# Analyses

One script per question. All of them read `../results/<dataset>/`: `pages.csv`
and `lines.csv` for text, `raw/*.jsonl` for per-character candidates. Run from
`experiments/`:

```
../providers/paddle/.venv/bin/python analyses/<script>.py [--dataset X]
```

Only `rapidfuzz` is needed for the scripts below, except where noted.

## Where each number of the paper comes from

| in the paper | script |
|---|---|
| Table 1, corpus counts and the training split | `corpus_stats.py` |
| Section 3.1, long-s fraction per stratum | `emission_by_stratum.py --dataset porto --label calamari --by-date --min-s 1`, `emission_by_stratum.py --dataset impact --label calamari --min-s 1`, `strata_figure.py` (needs `matplotlib`) |
| Section 3.2, Tesseract as a second generalist | `engine_baseline.py --dataset <set> --labels tesseract,tesseract_best,tesseract_page,tesseract_best_page` |
| Section 3.3, probability and rank of the glyph class at every CTC step | `class_probability.py` (needs the Paddle environment and a GPU) |
| Section 3.3, what the generalist writes where the ground truth has `ſ` | `glyph_confusion.py` |
| Section 3.4.1, arithmetic combinations against the symbolic rule | `rule1_combiners.py` |
| Section 3.4.1, the gap criterion for the space rule | `rule2_gap.py` |
| Section 3.5, sensitivity to the recognition batch | `batch_sensitivity.py` (needs the Paddle environment) |
| Figure of the glyph record per page (Section 3.1) | `strata_figure.py` (needs `matplotlib`) |
| Figure of the fusion gain with its intervals (Section 4.1) | `gain_intervals_figure.py` (needs `matplotlib`) |
| Table 2, fusion on the five corpora | `../eval_page.py`, `../eval_goldseg.py` |
| Table 3, English corpus | `glyph.py --dataset vesalainen_en`; the one-column row at 5e-4 with `--systems paddle,1class,1class_lr5e-4,3classes` |
| Table 4, ablation and the per-work filter | `../ablation.py` |
| Table 5, line protocol | `../eval_goldseg.py`, `glyph.py --dataset goldseg --corpus <c>` |
| Table 5, WER of the zero-shot rows | `wer_by_label.py --labels trocr,qwen` |
| Table 6, column tuning in domain | `glyph.py --dataset goldseg --systems paddle,1class,1class_lr5e-4,3classes_lr1e-4,3classes,full_projection_ep0,full_projection_lr5e-4_ep0` |
| Paragraphs after Table 6, what each tuned model writes at the glyph positions | `glyph_confusion.py --dataset goldseg --label <label of Table 6>` |
| Figure of the training curves (Section 4.5) | `training_curve_figure.py` (needs `matplotlib`), which reads the training logs of the Zenodo deposit with the parser of `training_curve.py`; `training_curve.py` alone writes the points as CSV |
| Table 7, outside the training domain | `../finetune/out_of_domain.py --corpus <c> --label <label>`, `emission_by_stratum.py` |
| Manual check of the ground-truth strata (blind sample of 60 pages) | `stratum_sample.py --pages-root DIR --out DIR`, then `--score answers.csv` (the sheet needs `Pillow`) |
| Table 8, lexical baseline of Alex et al. | `alex_lexicon.py` |
| Table 9, strict and folded readings | `rule_precision.py`, `rule_precision_strict.py` |
| Section 5.2, cost of the space rule | `rule2_by_work.py`, `rule2_lexicon.py` (needs `pyspellchecker`, module `spellchecker`) |
| error classes before and after the fusion | `../class_impact.py` |
| error classes of the full fine-tune, paragraph after Table 5 | `class_by_label.py --dataset goldseg --labels paddle,paddle_ft3` |
| confidence intervals of the CER differences | `bootstrap_ci.py` |

`glyph.py` is the single ruler for the glyph: emission, recall, precision and
net gain, strict and folded, for every system on every corpus.

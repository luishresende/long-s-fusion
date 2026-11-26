# Acquiring the corpora

One script per corpus. Each reads the raw distribution and writes
`experiments/datasets/<name>/manifest.csv` with `id, image_path, gold` (plus
`date`/`book` where the corpus provides them), and, where images have to be
converted, `experiments/datasets/<name>/pages/*.png`.

`image_path` is written relative to `experiments/`, which is where
`inference_pipeline/stages.py` resolves it from. Writing it relative to anything
else means the pipeline cannot find a single page.

None of the corpora is redistributed here, and all four are acquired
automatically. Each script downloads or clones what it needs on first run and
skips the step if the raw distribution is already on disk.

| corpus | licence | source |
|---|---|---|
| PORTO | CC-BY-4.0 | HuggingFace `LIACC/PORTO` |
| Reichsanzeiger-GT | CC0 | `github.com/UB-Mannheim/reichsanzeiger-gt` |
| OCR17+ | see repository | `github.com/Heresta/OCR17plus` |
| IMPACT | public domain | `lab.kb.nl/dataset/ground-truth-impact-project` |

OCR17+ is `Heresta/OCR17plus`, which holds the evaluation ground truth used
here; `e-ditiones/OCR17` is the training set and gives a different corpus.

IMPACT takes 15.1 GB. The TIFF archive is `impact_boeken.tif.zip`; the
PAGE-XML is a separate 9 MB file. Budget disk for the zip, the extracted TIFFs
and the converted PNGs at the same time.

Use the Paddle virtualenv, because `porto.py` needs `datasets` (HuggingFace) and
the other scripts need `Pillow`, and only that environment declares both.

```bash
V=../../providers/paddle/.venv/bin/python
$V porto.py           # downloads from HuggingFace
$V reichsanzeiger.py  # clones UB-Mannheim/reichsanzeiger-gt, then the scans
$V ocr17.py           # clones Heresta/OCR17plus
$V impact.py          # downloads 15.1 GB from KB Lab, then extracts
```

Re-running any of them is safe. Acquisition is skipped when the raw data is
already there, and a partial download is written to `.part` so it is never
mistaken for a complete file.

The scripts write PNG, converting from TIFF where needed, because the rest of
the pipeline assumes PNG.

Reichsanzeiger comes in two pieces. The repository carries the PAGE-XML
only; the scans live on the UB Mannheim image server and are listed in
`data/imageurls.list`. `reichsanzeiger.py` downloads the 101 of them itself,
which is what `data/download_images.sh` does in the repository.

## After this

The `pages.csv` and `lines.csv` under `experiments/results/` are generated,
not distributed. Build them with the inference pipeline:

```bash
cd ../../experiments
$V -u inference_pipeline/run.py --dataset porto     # and the others
```

That runs the detector once and every engine over the same crops, which is the
protocol the paper depends on.

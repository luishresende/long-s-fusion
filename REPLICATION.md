# Replication package

How to reproduce every table of *Long-s recovery in historical print by
asymmetric fusion of OCR engines*, starting from an empty machine.

Two routes lead to the tables. The full route runs the engines over the corpora
and takes days of CPU. The short route downloads the engine outputs and the
tuned columns from the Zenodo deposit (see `README.md`), unpacks them under
`experiments/results/` and `experiments/finetune/`, and goes straight to §4,
where everything except training and inference runs on a CPU in minutes.

```bash
# the short route, from the root of the repository
for f in engine-outputs tuned-columns full-finetune-baseline training-splits-and-logs \
         tesseract-baseline learning-rate-control ten-epoch-control; do
    tar xzf /path/to/$f.tar.gz
done
```

The code and the deposit go together. The scripts expect the file names, the
column names and the relative keys of that deposit (for instance
`experiments/finetune/train_data/records.csv`), so files from an earlier run of
the pipeline cannot be mixed with it.

The archives use the directory names of this guide. `training_logs/` lands at
the root and is read by `analyses/training_curve.py` and
`analyses/training_curve_figure.py`.

---

## 0. Starting from nothing

On a machine with none of this, in order:

```bash
# 1. the repository, and the Calamari model repository it depends on
git clone <this repository> long-s-fusion
cd long-s-fusion
git submodule update --init --depth 1 \
    providers/calamari/models/calamari_models_experimental

# 2. uv, which every environment below is built with. It installs to
#    ~/.local/bin, which is not always on PATH in a non-login shell.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

**Disk.** Budget 150 GB. Around 35 GB of that is download: the wheels for four
environments, the model weights, and the corpora, of which OCR17+ alone is 11 GB
and the English test set 7.4 GB. The rest is what the pipeline writes, mostly the
line crops under `experiments/results/*/lines/` and the training data.

**GPU.** Needed for training and convenient for recognition. §2 has the figures.

Each environment pins its own Python, and `uv` fetches whichever the system
lacks. Two things must be fetched by hand: the English test corpus of §4 needs a
manual Zenodo download, and the PP-OCRv6 training weights need a manual `curl`.
Both URLs are in §4. The four corpora
of §3 do fetch themselves, and so do the PP-OCRv6 inference weights.

---

## 1. What is and is not included

| item | included | note |
|---|---|---|
| source code | yes | this repository |
| environment lock | yes | `uv.lock` per engine |
| tuned column checkpoints | Zenodo | not in git; download them, or rebuild them with `train_class.py` (§4) before `export.py` |
| engine outputs (`raw/*.jsonl`, `pages.csv`, `lines.csv`) | Zenodo | not in git; download them, or regenerate them with the pipeline of §3 |
| PP-OCRv6 inference weights | no | downloaded on first use, into `~/.paddlex/official_models/` |
| PP-OCRv6 training weights | no | `PP-OCRv6_medium_rec_pretrained.pdparams`, 222 MB, fetched by hand (nothing here downloads it); URL in §4 |
| Calamari / Kraken models | no | third-party, see §3 |
| corpus acquisition scripts | yes | `data/prepare/`, one per corpus |
| the four corpora | no | third-party, all four acquired automatically, see §3 |

The checkpoints store only `head.ctc_head.fc.{weight,bias}`. Reconstructing a
tuned model means loading the official weights and overwriting those two tensors.
That is by construction. `train_class.py` restores every other column after each
optimizer step and prints the largest deviation as `leak`.

---

## 2. Hardware and environment

Measured on an NVIDIA RTX 5090 (32 GB). Training one class configuration takes
about 55 minutes for three epochs at batch 128.

The recognition path does not need a GPU of that size; batch 32 fits in 6 GB.
The GPU matters for training. The CPU bounds the pipeline. On 8 cores, a full
pass over PORTO takes about 12 hours, of which 5h27 is Calamari and 18 minutes is
PP-OCRv6. Both engines allocate the GPU and leave it idle, since the work is in
image decoding and, for Kraken, baseline post-processing.

```bash
# one virtualenv per engine, because their dependencies conflict
cd providers/paddle    && uv sync    # paddle 3.3.0 + CUDA
cd ../calamari         && uv sync
cd ../kraken           && uv sync
cd ../hf               && uv sync    # TrOCR and Qwen2.5-VL, the zero-shot rows
```

Each `pyproject.toml` pins its own Python (3.12, 3.10, 3.11 and 3.12), and `uv`
fetches whichever the system lacks. There are four, which is the count §0
budgets disk for. `providers/hf` is not an OCR engine of the fusion, but the two
systems it serves are rows of Tables 3 and 5, and nothing else builds it.

The training code imports `ppocr` from the PaddleOCR repository, which lives
inside the PaddleX package but does not ship with it. Installing it in a
clean environment needs a workaround, because `paddlex --install` calls
`remove()` before cloning and that `chdir`s into a `repos/` that does not exist
yet, failing with `FileNotFoundError`:

```bash
cd providers/paddle
mkdir -p .venv/lib/python3.12/site-packages/paddlex/repo_manager/repos
./.venv/bin/paddlex --install PaddleOCR -y
```

Without this, everything in `finetune/` (Tables 6 and 7) fails to import.

The order matters. `paddlex --install` goes
through `pip`, so the packages it adds are not in `uv.lock`, and a later
`uv sync` prunes them. Run `uv sync` first, `paddlex --install` after, and
repeat the install whenever a dependency changes and the environment is
synced again.

The Paddle environment ends up with two packages that provide `cv2`:
`opencv-contrib-python` 4.10.0.84, required by `paddlex[ocr]`, and
`opencv-python-headless` 5.0.0.93, required by `albumentations`. Which one
answers `import cv2` depends on the installation order, not on the lock. The
deposited outputs were produced with 5.0.0.93. With 4.10 the detected boxes are
the same but the resampled crops differ by a few grey levels, and Kraken reads
92.1% of a sample of crops identically instead of 100%. Reinstall the headless
package as the last step, after `uv sync` and after `paddlex --install`:

```bash
uv pip install --python .venv/bin/python --reinstall --no-deps \
    opencv-python-headless==5.0.0.93
```

A later `uv sync` keeps it. A later `paddlex --install` undoes it, and the
command has to be run again.

---

## 3. Data

None of the five corpora is redistributed here, but acquisition is scripted.
`data/prepare/` holds one preparation script per corpus; see its README for the
full commands.

| corpus | language | licence | acquisition |
|---|---|---|---|
| PORTO | Portuguese, 1676–1929 | CC BY 4.0 | HuggingFace `LIACC/PORTO` |
| Reichsanzeiger-GT | German, 1819–1939 | CC0 1.0 in the repository, CC BY 4.0 in its Zenodo record | `github.com/UB-Mannheim/reichsanzeiger-gt` |
| OCR17+ | French, 1624–1697 | CC BY (version not stated), images excluded | `github.com/Heresta/OCR17plus` |
| IMPACT | Dutch, 1630–1796 | public domain | `lab.kb.nl`, 15.1 GB |
| OCR 18th Century | English, 18th century | CC BY 4.0 | Zenodo 18597235, manual download (§4) |

The first four download or clone themselves on first run. OCR17+ is
`Heresta/OCR17plus`, not `e-ditiones/OCR17`, which is the training set.

Each script writes `experiments/datasets/<name>/manifest.csv`, and the ones that
convert images also write `experiments/datasets/<name>/pages/`. OCR17+ is the
exception. Its PNGs are already usable in the clone, so the manifest points at
them where they are.

```bash
V=../../providers/paddle/.venv/bin/python
cd data/prepare && $V porto.py    # and reichsanzeiger.py, ocr17.py, impact.py
```

Re-running a script is cheap. The clone, the download and the PNG conversion are
all skipped when the output is already there, so only the manifest is rebuilt.

`experiments/results/<corpus>/{pages,lines}.csv` are produced by the inference
pipeline from the corpora above (or come from the Zenodo deposit):

```bash
cd experiments
$V -u inference_pipeline/run.py --dataset porto     # and the others
```

That step runs the detector once and every engine over the same crops. It is the
expensive part (about 10 minutes per corpus on the recognition side), but it is
reproducible from public data for three of the four corpora.

Expected layout once both steps have run:

```
experiments/
  datasets/
    <corpus>/       pages/  manifest.csv
  results/
    <corpus>/       pages.csv  lines.csv  lines/
```

`pages.csv` carries the page-level gold; `lines.csv` carries one row per detected
line, with `line_path` and one column per engine.
The engines also write their standard error to
`experiments/results/<corpus>/logs/*.stderr.log`. Those logs carry absolute
paths of the machine, in warnings and tracebacks, and are not part of what the
deposit holds.
Every page and crop path that the pipeline writes into `raw/*.jsonl` and
`lines.csv` is a key relative to the repository root, resolved only when an image
is opened, so a rerun writes the same keys as the deposit.

Engine models are not all automatic. Only PP-OCRv6 downloads itself, from
the PaddleX model zoo, on first use. The other two have to be fetched:

```bash
# Calamari: deep3_antiqua-hist, a git submodule of the model repository
git submodule update --init --depth 1 providers/calamari/models/calamari_models_experimental

# Kraken: german_print for recognition, blla for baseline segmentation
providers/kraken/.venv/bin/kraken get 10.5281/zenodo.10519596
providers/kraken/.venv/bin/kraken get 10.5281/zenodo.14602569
```

Without the submodule the Calamari provider dies at load with `no checkpoint in
providers/calamari/models/...`; without the Kraken models it dies with
`missing model: ~/.local/share/htrmopo/...`.

---

## 4. Reproducing each table

Run from `experiments/`. `$V` is the Paddle virtualenv,
`../providers/paddle/.venv/bin/python`.

The analysis scripts need only `rapidfuzz`, except `analyses/rule2_lexicon.py`
(Section 5.2), which also needs `pyspellchecker` (module `spellchecker`). Both
are declared in `experiments/pyproject.toml`; in the Paddle environment the second
one is added with `uv pip install --python ../providers/paddle/.venv/bin/python pyspellchecker`.

### Building the intermediate CSVs

Everything else reads these. The detector runs once, and every engine reads
the same crops.

```bash
$V -u inference_pipeline/run.py --dataset porto      # and impact, ocr17, reichsanzeiger
```

Two environment variables affect the numbers. Both are the defaults in
`providers/paddle/provider.py`, and the published tables were produced with them.

| variable | value | what it does |
|---|---|---|
| `PADDLE_DET_LIMIT` | 6000 | the detector's `limit_side_len` |
| `PADDLE_REC_BATCH` | 128 | crops per recognition batch |

With a detector limit of 3000, about 20% fewer lines are found on OCR17+; no
engine sees that text, so every CER shifts. With a batch of 8,
22.4% of the recognized lines come out different, and the base CER of OCR17+
moves from 0.1306 to 0.1858. The batch matters because padding width sets the
number of CTC steps (§5), and 128 divides the 256-crop chunk into two full
batches; 64 and 256 both reproduce worse.

Changing either means re-running every corpus. A run that mixes two values is
not comparable with itself.

For the line-level protocol, the crops come from the gold polygons instead of the
detector, so there is nothing to detect. The crops themselves are cut by
`extract_lines.py`, which also builds the fine-tuning set. It has to run first,
and it reads the OCR17+ and Reichsanzeiger clones under `data/prepare/`:

```bash
$V -u finetune/extract_lines.py --output finetune/train_data
$V -u inference_pipeline/prepare_goldseg.py --splits test
for e in paddle calamari kraken; do
  $V -u inference_pipeline/recognize.py --dataset goldseg --engine $e --chunk 256
done
$V -u inference_pipeline/run.py --dataset goldseg --csv-only
```

The recognition step is not optional. `--csv-only` only reassembles what is
already in `raw/`, so skipping it leaves every engine column empty. `ablation.py`
reads this dataset too and fails with `FileNotFoundError` if it is missing.

This is only possible for OCR17 and reichsanzeiger, which ship line-level PAGE
XML. PORTO and IMPACT provide page-level gold only.

### The fusion across languages  (Table 2)

```bash
$V eval_page.py --datasets porto,impact,ocr17,reichsanzeiger
$V eval_goldseg.py
```

`eval_page.py` rewrites `reports/page_results.md` whole, with one section per
dataset named on the command line and nothing else. Passing the shorter
`--datasets porto,impact` does not add to an existing report. It drops the
`ocr17` and `reichsanzeiger` sections. The page-level CER of those
two is a published figure (0.1306 and 0.7893), so they belong in the same run.

Both take `--rules` to toggle individual rules and `--output` to name the
Markdown report they write. `eval_goldseg.py` also takes `--corpus` to report
OCR17 and Reichsanzeiger separately, which is how Table 5 presents them.

### All systems under gold segmentation  (Table 5)

`eval_goldseg.py` writes this table as well; it reports every system on the
14,437 gold-polygon crops in one pass.

The two rows of the Fraktur model of Tesseract in the German panel come from a
third specialist, read from `raw/` by crop file name. Its outputs are in
`tesseract-baseline.tar.gz` and cover the German lines only:

```bash
$V eval_goldseg.py --corpus reichsanzeiger --specialist tesseract_frak
```

This gives 0.0814 for `tesseract_frak` alone and 0.0647 for its fusion with
PP-OCRv6, next to the unchanged rows of Calamari and Kraken.

### The full fine-tune  (the row that collapses, in Tables 3 and 5)

The configuration is `finetune/PP-OCRv6_medium_rec_hist_test3.yml`, run with the
stock PaddleOCR `tools/train.py` on the same training split as the column runs.
Its best checkpoint was exported with the stock PaddleOCR export tool and read
the test lines through the same path as every other system, under the label
`paddle_ft3`:

```bash
PADDLE_REC_DIR=$PWD/finetune/inf_full_finetune $V -u inference_pipeline/recognize.py \
   --dataset goldseg --engine paddle --label paddle_ft3
```

The checkpoint, the configuration, the training log and the exported model are
in the Zenodo deposit.

The paragraph that follows Table 5 counts the edits of this model by error class
and gives its CER outside the training domain:

```bash
$V analyses/class_by_label.py --dataset goldseg --labels paddle,paddle_ft3
$V finetune/out_of_domain.py --corpus porto  --label paddle_ft3    # the TOTAL line
$V finetune/out_of_domain.py --corpus impact --label paddle_ft3
```

### Column tuning, in domain  (Table 6 and the training-curve figure)

Three conditions, same data, same batch, three epochs of 902 steps each:

```bash
W=~/.paddleocr/models/PP-OCRv6_medium_rec_pretrained.pdparams
COMMON="--weights $W --batch 128 --accum 1 --epochs 3 --eval-every 50"

$V -u finetune/train_class.py $COMMON --classes "ſ"     --lr 1e-4 --output finetune/out_1class
$V -u finetune/train_class.py $COMMON --classes "ſ,f,s" --lr 5e-4 --output finetune/out_3classes
$V -u finetune/train_class.py $COMMON --classes all     --lr 1e-4 --output finetune/out_full_projection
```

Each evaluation line prints `leak`, the largest deviation of the projection
outside the target columns. It must be `0.0e+00` on every line of the first
two runs. Anything else means the isolation leaked and the run is void. The
third run releases every column on purpose and is the control condition.

The checkpoint is selected by validation CER, which picks `ep2` for one column,
`ep1` for three columns and `ep0` for the full projection. Selecting it by
out-of-domain performance would be fitting the test set.

Table 6 is measured on the test split, through the same inference path as every
other table. Each selected checkpoint is exported and run over the 14,437 lines
of `goldseg` under its own label. The labels below are the ones the deposit
uses, and reusing a label for another model silently overwrites its output:

```bash
run() {  # run <checkpoint> <export dir> <label>
  $V -u finetune/export.py --weights $W --column $1 --output $2
  PADDLE_REC_DIR=$PWD/$2 $V -u inference_pipeline/recognize.py \
     --dataset goldseg --engine paddle --label $3
}
run finetune/out_1class/ep2.pdparams          finetune/inf_1class_ep2          1class
run finetune/out_3classes/ep1.pdparams        finetune/inf_3classes_ep1        3classes
run finetune/out_full_projection/ep0.pdparams finetune/inf_full_projection_ep0 full_projection_ep0

$V analyses/glyph.py --dataset goldseg \
   --systems paddle,1class,3classes,full_projection_ep0,paddle_ft3
```

`paddle_ft3` is the full fine-tune of the previous subsection. The deposit also
carries the outputs and the exported models of the other two epochs of the full
projection, as `full_projection_ep1` and `full_projection_ep2`. The export and
recognition commands are reconstructed from the code and from the run logs in
`training_logs/`; the original shell lines were not kept.

The paragraph that follows the table, on what each tuned model writes where the
ground truth has the glyph, comes from the same outputs:

```bash
for l in 1class 3classes; do
  $V analyses/glyph_confusion.py --dataset goldseg --label $l
done
```

### The second learning rate  (the two-rate rows of Table 6)

Each configuration is reported at two learning rates, so that the number of
unfrozen parameters is not confounded with the rate. The three runs above are
one half of the grid. The other half changes `--lr` and nothing else:

```bash
$V -u finetune/train_class.py $COMMON --classes "ſ"     --lr 5e-4 --output finetune/out_1class_lr5e-4
$V -u finetune/train_class.py $COMMON --classes "ſ,f,s" --lr 1e-4 --output finetune/out_3classes_lr1e-4
$V -u finetune/train_class.py $COMMON --classes all     --lr 5e-4 --output finetune/out_full_projection_lr5e-4

run finetune/out_1class_lr5e-4/ep2.pdparams          finetune/inf_1class_lr5e-4_ep2          1class_lr5e-4
run finetune/out_3classes_lr1e-4/ep2.pdparams        finetune/inf_3classes_lr1e-4_ep2        3classes_lr1e-4
run finetune/out_full_projection_lr5e-4/ep0.pdparams finetune/inf_full_projection_lr5e-4_ep0 full_projection_lr5e-4_ep0

$V analyses/glyph.py --dataset goldseg --systems \
   paddle,1class,1class_lr5e-4,3classes_lr1e-4,3classes,full_projection_ep0,full_projection_lr5e-4_ep0
```

Validation CER at the end of each epoch, which is what selects the checkpoint:

| configuration | rate | ep0 | ep1 | ep2 | selected |
|---|---|--:|--:|--:|---|
| one column | 1e-4 | 0.0786 | 0.0782 | 0.0778 | ep2 |
| one column | 5e-4 | 0.0780 | 0.0773 | 0.0769 | ep2 |
| three columns | 1e-4 | 0.0722 | 0.0679 | 0.0663 | ep2 |
| three columns | 5e-4 | 0.0653 | 0.0650 | 0.0675 | ep1 |
| full projection | 1e-4 | 0.1003 | 0.1155 | 0.1305 | ep0 |
| full projection | 5e-4 | 0.1528 | 0.1936 | 0.2228 | ep0 |

On the test split the CER is 0.0770 and 0.0761 for one column, 0.0650 and 0.0644
for three columns, and 0.1072 and 0.1591 for the full projection, against 0.0786
of the baseline. These three runs are in `learning-rate-control.tar.gz` of the deposit,
with their training logs and command lines in `training_logs/lr_control/`. The
one-column model does not converge in three epochs at either rate. Its glyph
emission on validation is still rising at the end of the third.

A control run with ten epochs settles that. At 1e-4 the emission stops rising
at 0.24 in the eighth epoch and the validation CER does not go below 0.0765; at
5e-4 the emission stays between 0.27 and 0.31. On the test split, the evaluation
point with the lowest validation CER gives a CER of 0.0759 and a recall of 23.5%
at 1e-4 (step 7900), and 0.0756 and 25.5% at 5e-4 (step 1750). The two logs and
the two outputs, labels `1class_10ep_lr1e-4` and `1class_10ep_lr5e-4`, are in
`ten-epoch-control.tar.gz`.

```bash
$V -u finetune/train_class.py ${COMMON/--epochs 3/--epochs 10} --classes "ſ" --lr 1e-4 --output finetune/out_1class_10ep
python analyses/glyph.py --dataset goldseg --systems paddle,1class,1class_10ep_lr1e-4,3classes
```

### What to expect from a rerun

The short route recomputes every table from the deposited outputs and gives the
same digits. The long route does not. Training on a GPU is not bit-reproducible,
and neither is recognition across machines. In a rerun from scratch on another
machine, with the same data:

- a second training of the column runs gave a CER of 0.0640 for three columns,
  against the 0.0644 of the deposited checkpoint, and 0.0762 to 0.0766 for one
  column, against 0.0770;
- the PP-OCRv6 text changed in 19% of the pages of `porto` and 34% of the pages
  of `impact`, and one page of `porto` had a different number of detected lines;
- the page-level table of `porto` and `impact` still came out identical to the
  fourth decimal of the CER.

### Transfer to unseen languages  (Table 7)

The table is computed from the deposited outputs, like every other one. `--label`
names the output in `raw/` whose text is read, and `--fusion` applies rule 1
alone over that text, so that the two methods change the same class:

```bash
for ds in porto impact; do
  $V finetune/out_of_domain.py --corpus $ds --label paddle                     # baseline
  $V finetune/out_of_domain.py --corpus $ds --label 1class               # one column, 1e-4
  $V finetune/out_of_domain.py --corpus $ds --label 1class_lr5e-4        # one column, 5e-4
  $V finetune/out_of_domain.py --corpus $ds --label 3classes                   # three columns
  $V finetune/out_of_domain.py --corpus $ds --label paddle --fusion calamari   # rule 1
done
```

The row of the table is the `diplomatic` line of each run. Only `rapidfuzz` is
needed in this mode. The `1class_lr5e-4` outputs of `porto`, `impact` and
`vesalainen_en` are in `learning-rate-control.tar.gz`. The English row of the
same model, in Table 3, comes from
`analyses/glyph.py --dataset vesalainen_en --systems paddle,1class,1class_lr5e-4,3classes`.

Producing the `3classes` and `1class` outputs again takes the export of
each tuned model and a recognition pass over the crops of the two corpora:

```bash
$V -u finetune/export.py \
   --weights ~/.paddleocr/models/PP-OCRv6_medium_rec_pretrained.pdparams \
   --column finetune/out_3classes/ep1.pdparams \
   --output finetune/inf_3classes_ep1
PADDLE_REC_DIR=$PWD/finetune/inf_3classes_ep1 $V -u inference_pipeline/recognize.py \
   --dataset porto --engine paddle --label 3classes
```

The script also has `--provider [DIR]`, which recognizes the crops on the spot
instead of reading `raw/`. It runs the recognizer over the whole list of crops
in one call, so the batches are composed differently from those of the pipeline
and the text is not the deposited one. On `porto` the baseline of the diplomatic
stratum comes out at 0.1168 instead of 0.1149. An earlier version of the table
was measured that way. The paper uses the deposited outputs.

The training weights `PP-OCRv6_medium_rec_pretrained.pdparams` are not the
ones PaddleX downloads for inference, and nothing in this repository fetches them.
PaddleX pulls `inference.pdiparams` plus `inference.json` on first use, an
inference bundle rather than a training checkpoint, so `train_class.py` ends in
`weights not found: ...` with exit 1, after the class resolution has already
printed.

They are public, and the link is in the PaddleOCR documentation that
`paddlex --install PaddleOCR` (§2) clones into the venv, at
`docs/version3.x/module_usage/text_recognition.md`, where each model lists an
"Inference Model" and a "Training Model" side by side. Fetch it before §4:

```bash
mkdir -p ~/.paddleocr/models && cd ~/.paddleocr/models
curl -L -o PP-OCRv6_medium_rec_pretrained.pdparams \
  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams"
```

222 MB, about 40 seconds. Everything that needs a tuned column depends on this
one file: `train_class.py` (Table 6), `export.py` and the adjusted arms of
`analyses/glyph.py --dataset goldseg`. The base engines, both
fusions, the lexicon and the ablation do not need it.

### The external corpus in English  (Table 3)

The one corpus not assembled here. Their training split, 195k lines, is never
used, because spending it would make English a seen language and cost the only
clean external test there is.

Unlike the four corpora of §3, this one does not fetch itself. Zenodo 18597235 is
a single 7.4 GB archive holding two nested zips, and only the test half is
needed; the 6.4 GB training half is not used:

```bash
cd data/prepare/vesalainen_raw
curl -L -o OCR_18th_century.zip \
    "https://zenodo.org/records/18597235/files/OCR_18th_century.zip?download=1"
unzip OCR_18th_century.zip OCR_Testing.zip     # 7 min at 17 MB/s, then 1.0 GB
unzip OCR_Testing.zip -d test                  # test/full_english_dataset_relative.csv
```

Then, from `experiments/`:

```bash
$V ../data/prepare/vesalainen.py
$V -u inference_pipeline/recognize.py --dataset vesalainen_en --engine paddle
# then calamari, kraken, and the tuned columns, as above
$V analyses/glyph.py --dataset vesalainen_en
```

`vesalainen.py` reports what it built: 9,981 crops over 2 books, 526,118
characters, 14,706 `ſ` at 28.0 per thousand.

The two zero-shot systems of Tables 3 and 5 come from the fourth environment, and are
driven by `HF_MODEL` with an explicit `--label`, because the engine is `hf` for
both and the label is what names the column in `raw/`. They want a smaller chunk
than the OCR engines:

```bash
HF_MODEL=microsoft/trocr-large-printed $V -u inference_pipeline/recognize.py \
    --dataset vesalainen_en --engine hf --label trocr --chunk 64
HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct  $V -u inference_pipeline/recognize.py \
    --dataset vesalainen_en --engine hf --label qwen  --chunk 64
$V -u inference_pipeline/run.py --dataset vesalainen_en --csv-only
```

`trocr` and `qwen` are two of the labels `analyses/glyph.py --systems` expects by
default, which is why it prints them as dashes until these have run. The same two
commands, with `--dataset goldseg|porto|impact`, are what fills their rows there.

The fusion reaches its largest gain here. With 14,706 glyphs in about 525k
characters and a base CER of 0.0435, the glyph accounts for 64% of every edit
separating the generalist from the gold.

### The lexical baseline  (Table 8)

Their method decides by unigram frequency over a reference corpus, and their
reference was nine Gutenberg books, so the replication uses more than one work.
It is leave-one-out by work, each volume corrected with the lexicon of the others.

Run one command per corpus and always pass the dataset. Table 8 puts `ocr17` and
`reichsanzeiger` on the line ruler (`goldseg`) and `porto` on the page ruler, and
`alex_lexicon.py` defaults to neither. Its default is `ocr17` at page level, a
fourth combination that appears in no table:

```bash
$V analyses/alex_lexicon.py --dataset goldseg --corpus ocr17 \
       --mode rest --target diplomatic        # 0.0731 -> 0.0605, +17.24%
$V analyses/alex_lexicon.py --dataset goldseg --corpus reichsanzeiger \
       --mode rest --target diplomatic        # 0.0801 -> 0.0755,  +5.72%
$V analyses/alex_lexicon.py --dataset porto \
       --mode rest --target diplomatic        # 0.0508 -> 0.0492,  +3.00%

# the two-target contrast, same three rulers
$V analyses/alex_lexicon.py --dataset goldseg --corpus ocr17 \
       --mode rest --target modernized
```

Run with only `--mode rest --target diplomatic`, it returns
`0.1306 -> 0.1150, +11.94%`. That is `ocr17` measured by page, which is not the
ruler of the table.

The lexicon beats the fusion in French. What separates the three corpora is the
reference resource available, not the language.

### The effect of the metric  (Table 9)

CPU only, reads the CSVs directly.

```bash
$V analyses/rule_precision.py --dataset porto          # folded reading
$V analyses/rule_precision_strict.py --dataset porto  # strict, stratified
```

### Contribution of each component  (Table 4)

```bash
$V ablation.py
```

### The second generalist, Tesseract  (Section 3.2)

Tesseract enters no fusion and no table. It is the comparison behind two
sentences of Section 3.2: a second generalist does not emit the glyph either,
and its CER is above that of PP-OCRv6. Its outputs are in
`tesseract-baseline.tar.gz` of the Zenodo deposit, and one command per set
recomputes every figure from them:

```bash
$V analyses/engine_baseline.py --dataset porto  --labels tesseract,tesseract_best,tesseract_page,tesseract_best_page
$V analyses/engine_baseline.py --dataset impact --labels tesseract,tesseract_best,tesseract_page,tesseract_best_page
$V analyses/engine_baseline.py --dataset ocr17  --labels tesseract,tesseract_best,tesseract_page,tesseract_best_page
$V analyses/engine_baseline.py --dataset reichsanzeiger \
    --labels tesseract_page,tesseract_best_page,tesseract_frak_page,tesseract_best_latf_page
$V analyses/engine_baseline.py --dataset vesalainen_en --labels tesseract,tesseract_best
$V analyses/engine_baseline.py --dataset goldseg --labels tesseract,tesseract_best
$V analyses/engine_baseline.py --dataset goldseg --corpus reichsanzeiger \
    --labels tesseract,tesseract_best,tesseract_frak,tesseract_best_latf
```

| label | models | input |
|---|---|---|
| `tesseract` | language packs of the distribution | the line crops of `raw/detect.jsonl`, `--psm 7` |
| `tesseract_best` | `tessdata_best` | the same crops |
| `tesseract_page` | language packs of the distribution | the whole page, `--psm 1`, Tesseract's own layout analysis |
| `tesseract_best_page` | `tessdata_best` | the whole page |
| `tesseract_frak`, `tesseract_frak_page` | `frk` of the distribution | German only |
| `tesseract_best_latf`, `tesseract_best_latf_page` | `deu_latf` of `tessdata_best` | German only |

Strict CER, micro-averaged, as printed by the commands above:

| set | PP-OCRv6 | `tesseract` | `tesseract_best` | `tesseract_page` | `tesseract_best_page` |
|---|--:|--:|--:|--:|--:|
| `porto` | 0.0508 | 0.0740 | 0.0734 | 0.0834 | 0.0811 |
| `impact` | 0.1015 | 0.1887 | 0.1847 | 0.1385 | 0.1355 |
| `ocr17` | 0.1306 | 0.5028 | 0.4964 | 0.1441 | 0.1352 |
| `reichsanzeiger` | 0.7893 | | | 0.5706 | 0.5725 |
| `goldseg` | 0.0786 | 0.1257 | 0.1669 | | |
| `vesalainen_en` | 0.0435 | 0.0808 | 0.0775 | | |

The models of the five modern languages (`por`, `nld`, `fra`, `deu`, `eng`) emit
no long s in any set. The page protocol is not usable for `reichsanzeiger`
(Section 3.5 of the paper), which is where Tesseract has the lower CER.

`frk` and `deu_latf` are models of the historical script, not generalists. They
were run on the German material only. `frk` enters the paper as a third
specialist of the fusion, in the German panel of Table 5 (see the subsection of
that table above), and `deu_latf` is kept for reference. On the 11,955 German lines of `goldseg` (`--corpus reichsanzeiger`),
PP-OCRv6 has CER 0.0801, `tesseract_frak` 0.0814 with 6,158 glyphs emitted and
`tesseract_best_latf` 0.1338 with 6,043. On the German pages they reach 0.5636
and 0.5612. Evaluate these two labels with `--corpus reichsanzeiger` only,
because over the whole of `goldseg` the 2,482 French lines count as empty text,
and the CER
printed for them, 1.0000, reflects missing output.

Notes on these numbers:

- The crops Tesseract read are the ones of a second detection pass, on another
  machine. They coincide with `raw/detect.jsonl` of the deposit in every page
  but one of `porto`, whose missing crops count as empty text. Read against its
  own detection, `tesseract_best` on `porto` has 0.0733 instead of 0.0734.
- The keys of these records are relative to the repository root, like those of
  the rest of the deposit, so no path needs to exist.
- In `goldseg` each label has 14 shards. Shards 0 to 6 are the French lines and
  7 to 13 the German ones, because the language model is chosen per corpus.

#### Running Tesseract again

A fifth environment, needed for this subsection only:

```bash
# Tesseract 5.5.1 itself came from a PPA build; the language packs from the distribution
sudo apt install tesseract-ocr \
    tesseract-ocr-por tesseract-ocr-nld tesseract-ocr-fra tesseract-ocr-deu \
    tesseract-ocr-eng tesseract-ocr-frk
cd providers/tesseract && uv sync
mkdir -p ~/tessdata_best && cd ~/tessdata_best
for l in por nld fra deu eng deu_latf osd; do
    curl -LO https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata
done
```

The runs used Tesseract 5.5.1 with Leptonica 1.82.0 (`5.5.1-1ppa1~noble1`). The
language packs of the distribution are the Debian snapshot `1:4.1.0-2`, which
does not match the current `tessdata` repository, so the checksums are what
identifies the models:

| model | `tessdata_best` (main, 22 November 2026) | distribution, `/usr/share/tesseract-ocr/5/tessdata` |
|---|---|---|
| `por` | `711de9dbb8052067bd42f16b9119967f30bada80d57e2ef24f65d09f531adb04` | `c4932b937207a9514b7514d518b931a99938c02a28a5a5a553f8599ed58b7deb` |
| `nld` | `92e7a1ad4bf8082e268de57c7823316ec024935702c6ed2a1e473b3a071aa733` | `ced0e5e046a84c908a6aa7accbef9a232c4a5d9a8276691b81c6ee64d02963f6` |
| `fra` | `907743d98915c91a3906dfbf6e48b97598346698fe53aaa797e1a064ffcac913` | `ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca` |
| `deu` | `8407331d6aa0229dc927685c01a7938fc5a641d1a9524f74838cdac599f0d06e` | `19d219bbb6672c869d20a9636c6816a81eb9a71796cb93ebe0cb1530e2cdb22d` |
| `eng` | `8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba` | `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` |
| `deu_latf` / `frk` | `429b8c33d37e1a4823a2bb082a644ff13e252f735b7945e08a518d22ddcc459e` | `7cd1b541e9d3884b9546a7d292c4f349cdb45ff646b4b43166dfc5099a8ad1a1` |
| `osd` | `9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff` | the same file |

The model directory is chosen by `TESSDATA_PREFIX` and the language by
`TESSERACT_LANG`. `--oem 3` everywhere. Tesseract has no GPU path, so the work
is split in 7 processes of one thread each. From `experiments/`:

```bash
BEST=$HOME/tessdata_best
# crops of the generalist, here porto with tessdata_best
for i in 0 1 2 3 4 5 6; do
    TESSDATA_PREFIX=$BEST TESSERACT_LANG=por TESSERACT_PSM=7 OMP_THREAD_LIMIT=1 \
        $V -u inference_pipeline/recognize.py --dataset porto --engine tesseract \
        --label tesseract_best --shard $i/7 &
done; wait
# whole pages, Tesseract's own layout analysis
for i in 0 1 2 3 4 5 6; do
    TESSDATA_PREFIX=$BEST TESSERACT_LANG=por TESSERACT_PSM_PAGE=1 OMP_THREAD_LIMIT=1 \
        $V -u inference_pipeline/tesseract_page.py --dataset porto \
        --label tesseract_best_page --shard $i/7 &
done; wait
```

The other sets change `--dataset` and the language: `nld` for `impact`, `fra`
for `ocr17`, `deu` for `reichsanzeiger`, `eng` for `vesalainen_en`. The labels
without `best` drop `TESSDATA_PREFIX`, and the Fraktur labels use
`TESSERACT_LANG=frk` or `deu_latf`. The runs used `--chunk 256` on `porto` and
`impact`, 128 on `ocr17` and 64 on the line-level sets.

`goldseg` mixes two languages, so its `detect.jsonl` is first split by corpus,
each half is read with its language, and the shards are renamed into one label:

```bash
$V - <<'PY'
import csv, json
csv.field_size_limit(10 ** 9)
res = 'results/goldseg/'
corpus = {r['id']: r['corpus'] for r in csv.DictReader(open(res + 'pages.csv', encoding='utf-8'))}
out = {'ocr17': open(res + 'raw/detect_fra.jsonl', 'w', encoding='utf-8'),
       'reichsanzeiger': open(res + 'raw/detect_deu.jsonl', 'w', encoding='utf-8')}
for line in open(res + 'raw/detect.jsonl', encoding='utf-8'):
    out[corpus[json.loads(line)['id']]].write(line)
PY
for l in fra deu; do for i in 0 1 2 3 4 5 6; do
    TESSDATA_PREFIX=$BEST TESSERACT_LANG=$l TESSERACT_PSM=7 OMP_THREAD_LIMIT=1 \
        $V -u inference_pipeline/recognize.py --dataset goldseg --engine tesseract \
        --detect detect_$l.jsonl --label tesseract_best_$l --chunk 64 --shard $i/7 &
done; wait; done
cd results/goldseg/raw
for i in 0 1 2 3 4 5 6; do
    mv tesseract_best_fra.${i}of7.jsonl tesseract_best.${i}of14.jsonl
    mv tesseract_best_deu.${i}of7.jsonl tesseract_best.$((i + 7))of14.jsonl
done
```

### Figures

The three data figures are drawn by scripts that need `matplotlib`, which none
of the environments of §2 carries and which no `pyproject.toml` lists. Run them
with any Python that has `rapidfuzz`, `numpy` and `matplotlib`, called `$F`
below. The published figures were drawn with Python 3.12.9, `matplotlib` 3.9.2,
`numpy` 2.2.6 and `rapidfuzz` 3.14.5:

```bash
F=python3     # any Python with rapidfuzz, numpy and matplotlib
$F analyses/strata_figure.py          --output strata.pdf          # glyph record per page, Section 3.1
$F analyses/gain_intervals_figure.py  --output gain_intervals.pdf  # fusion gain with its intervals, Section 4.1
$F analyses/training_curve_figure.py  --output training_curve.pdf  # column-tuning curves, Section 4.5
```

Each script writes a 600 dpi raster when `--output` ends in `.png`. The shared
pieces are in `analyses/figure_style.py`.

| figure | script | reads | time |
|---|---|---|--:|
| glyph record per page, Section 3.1 | `strata_figure.py` | `pages.csv` and `lines.csv` of `porto` and `impact` | 3 s |
| fusion gain with its intervals, Section 4.1 | `gain_intervals_figure.py` | what `bootstrap_ci.py` reads, with its replicates and seed | 17 s |
| column-tuning curves, Section 4.5 | `training_curve_figure.py` | `training_logs/training.log` and `projection.log` | 1 s |

The curves are the evaluations `train_class.py` printed every 50 steps, parsed
with the patterns of `training_curve.py`, and the horizontal line is the
`before training` CER of the same logs, 0.0796. `--logs` points at another
directory when the deposit is not unpacked at the root.

The other two figures have no script:

- **The method diagram** (Section 3) is TikZ, in the LaTeX source of the paper.
- **The example line** (Introduction) was cut by hand. The line is
  `OMD_1707_515__l034` of `porto`, but its crop comes from an earlier detection
  run, not from `raw/detect.jsonl` of the deposit, and the boxes around the two
  words were placed by hand on the ink of each word. Reproducing it needs the
  page image of `porto`, which the deposit does not carry, and those manual
  coordinates.

### Confidence intervals of the CER differences

The engines are deterministic, so running them again gives no spread. The
uncertainty left in a comparison is sampling, namely whether the ordering of two
systems would hold on other pages of the same collections.
`analyses/bootstrap_ci.py` measures it with a paired bootstrap. The units
are drawn with replacement, and in each replicate the micro-averaged CER of both
systems is recomputed over the same draw, so the pages vary and the systems do
not.

The unit is the page in the page-level sets and the line in `goldseg` and
`vesalainen_en`. Pages of one book are not independent, so where the ground
truth names the work (`ocr17`, `porto`, `goldseg`) the same comparisons are
resampled again with the work as the unit, as a sensitivity analysis. A
comparison that spans fewer than 10 works is not resampled by work. The
diplomatic stratum of `porto` holds 2.

```bash
$V analyses/bootstrap_ci.py                                         # every comparison, by page or line, then by work
$V analyses/bootstrap_ci.py --comparison lexicon_ocr17 --unit book  # one comparison, one unit
$V analyses/bootstrap_ci.py --list                                  # names, tables, systems A and B
```

10,000 replicates (`--replicates`), seed 20261123 (`--seed`), 95% percentile
intervals. The script reads `pages.csv`, `lines.csv` and the `detect`, `paddle`,
`calamari`, `3classes` and `1class` outputs of `raw/`, and needs `numpy`,
which the Paddle environment carries. It takes about a minute on one CPU core,
53 s to compute the errors of every unit once and 5 s for the 26 bootstraps.

Each hypothesis is built the way the script of its table builds it, and every
CER below matches the printed one to the fourth decimal. In `kraken_ocr17` the fusion is
the one with Kraken, as in the French row of Table 2. The other two Section 5.3
rows use Calamari.

The difference is CER B − CER A, so a negative value means B is lower. The gain
is (A − B) / A, the relative gain the tables print.

| comparison | table | A | B | unit | n | CER A | CER B | B − A [95% CI] | gain [95% CI] |
|---|---|---|---|---|--:|--:|--:|--:|--:|
| `fusion_ocr17_line` | 2 | PP-OCRv6 | fusion + Kraken | line | 2,482 | 0.0731 | 0.0617 | −0.0114 [−0.0123, −0.0106] | +15.65% [+14.47, +16.86] |
| `fusion_reichsanzeiger_line` | 2 | PP-OCRv6 | fusion + Calamari | line | 11,955 | 0.0801 | 0.0733 | −0.0068 [−0.0072, −0.0064] | +8.46% [+7.95, +8.98] |
| `fusion_porto` | 2 | PP-OCRv6 | fusion + Calamari | page | 3,782 | 0.0508 | 0.0464 | −0.0043 [−0.0047, −0.0040] | +8.56% [+7.89, +9.23] |
| `fusion_impact` | 2 | PP-OCRv6 | fusion + Calamari | page | 2,016 | 0.1015 | 0.0945 | −0.0071 [−0.0074, −0.0066] | +6.95% [+6.50, +7.40] |
| `fusion_vesalainen_en` | 2, 3 | PP-OCRv6 | fusion + Calamari | line | 9,981 | 0.0435 | 0.0214 | −0.0222 [−0.0227, −0.0217] | +50.93% [+49.86, +52.02] |
| `diplomatic_porto` | 7 | three columns | fusion, glyph rule | page | 555 | 0.0971 | 0.0861 | −0.0110 [−0.0114, −0.0105] | +11.30% [+10.64, +11.95] |
| `diplomatic_impact` | 7 | three columns | fusion, glyph rule | page | 783 | 0.1271 | 0.1246 | −0.0025 [−0.0027, −0.0022] | +1.96% [+1.72, +2.20] |
| `lexicon_ocr17` | 8 | lexicon | fusion + Kraken | line | 2,482 | 0.0605 | 0.0617 | +0.0012 [+0.0001, +0.0022] | −1.92% [−3.73, −0.16] |
| `lexicon_reichsanzeiger` | 8 | lexicon | fusion + Calamari | line | 11,955 | 0.0755 | 0.0733 | −0.0022 [−0.0026, −0.0018] | +2.91% [+2.33, +3.48] |
| `lexicon_porto` | 8 | lexicon | fusion + Calamari | page | 3,782 | 0.0492 | 0.0464 | −0.0028 [−0.0031, −0.0026] | +5.73% [+5.27, +6.20] |
| `kraken_impact` | 5.3 | Kraken, own segmenter | fusion + Calamari | page | 2,016 | 0.0904 | 0.0945 | +0.0041 [+0.0020, +0.0062] | −4.54% [−6.84, −2.26] |
| `kraken_ocr17` | 5.3 | Kraken, own segmenter | fusion + Kraken | page | 719 | 0.0756 | 0.1283 | +0.0527 [+0.0463, +0.0593] | −69.75% [−78.95, −60.88] |
| `kraken_porto` | 5.3 | Kraken, own segmenter | fusion + Calamari | page | 3,782 | 0.0736 | 0.0464 | −0.0272 [−0.0287, −0.0256] | +36.95% [+34.96, +38.65] |
| `columns3_goldseg` | 6 | PP-OCRv6 | three columns | line | 14,437 | 0.0786 | 0.0644 | −0.0142 [−0.0148, −0.0136] | +18.09% [+17.35, +18.83] |
| `column1_goldseg` | 6 | PP-OCRv6 | one column | line | 14,437 | 0.0786 | 0.0770 | −0.0016 [−0.0018, −0.0015] | +2.08% [+1.87, +2.29] |
| `columns3_vesalainen_en` | 3 | PP-OCRv6 | three columns | line | 9,981 | 0.0435 | 0.0323 | −0.0112 [−0.0116, −0.0109] | +25.78% [+24.93, +26.64] |

With the work as the unit, n is the number of works:

| comparison | n | B − A [95% CI] | gain [95% CI] |
|---|--:|--:|--:|
| `fusion_ocr17_line` | 21 | −0.0114 [−0.0138, −0.0094] | +15.65% [+12.87, +18.76] |
| `fusion_reichsanzeiger_line` | 43 | −0.0068 [−0.0080, −0.0056] | +8.46% [+6.91, +10.12] |
| `fusion_porto` | 14 | −0.0043 [−0.0098, +0.0006] | +8.56% [−2.06, +13.29] |
| `lexicon_ocr17` | 21 | +0.0012 [−0.0018, +0.0035] | −1.92% [−6.15, +2.75] |
| `lexicon_reichsanzeiger` | 43 | −0.0022 [−0.0033, −0.0011] | +2.91% [+1.50, +4.33] |
| `lexicon_porto` | 14 | −0.0028 [−0.0064, +0.0006] | +5.73% [−2.02, +9.26] |
| `kraken_ocr17` | 18 | +0.0527 [+0.0277, +0.0863] | −69.75% [−113.26, −36.98] |
| `kraken_porto` | 14 | −0.0272 [−0.0346, −0.0202] | +36.95% [+31.95, +46.21] |
| `columns3_goldseg` | 64 | −0.0142 [−0.0158, −0.0126] | +18.09% [+15.49, +21.04] |
| `column1_goldseg` | 64 | −0.0016 [−0.0020, −0.0012] | +2.08% [+1.59, +2.53] |

By page or line, no interval of the difference contains zero. By work, three
do: the fusion against the generalist on `porto`, and the lexicon against the
fusion on `porto` and on `ocr17`.

### Corpus counts  (Table 1)

```bash
$V analyses/corpus_stats.py        # units per corpus and the training split
```

`corpus_stats.py` prints a dash in every cell and exits 0 when the results are
missing, so read its output instead of trusting its exit code.
`experiments/analyses/README.md` maps the remaining figures of the paper to the
script that produces each one.

# `inference_pipeline/`

Runs the engines and writes raw JSON lines. No metric is computed here, so
fusion rules, normalization and evaluation can change without running OCR again.

| file | role |
|---|---|
| `run.py` | driver: calls the stages in order and assembles `pages.csv` and `lines.csv` |
| `detect.py` | stage 1: the PP-OCRv6 detector cuts the line crops |
| `recognize.py` | stage 2: batched recognition, `--engine paddle\|calamari\|kraken\|hf` |
| `kraken_page.py` | Kraken with its own segmenter (`blla`), the full-system comparison |
| `prepare_goldseg.py` | builds the line-protocol dataset from ground-truth polygons |
| `stages.py`, `workers.py` | manifest, result directories, checkpointing, worker client |

The crops of stage 1 are the shared input: PP-OCRv6, Calamari and Kraken read
the same images, which is what the fusion requires. Each engine still applies
its own preprocessing, the one it was trained with.

`kraken_page.py` shares nothing with the other engines. Kraken segments the page
itself and recognizes its own lines. Calamari has no segmenter, so it always
reads the crops of the generalist.

Kraken cannot be batched, because padding leaks into the decoded text. It runs
one crop per call and is parallelized by shards, with byte-identical output.

## Layout

```
experiments/
  inference_pipeline/
  datasets/<dataset>/   manifest.csv, pages/        (written by data/prepare/)
  results/<dataset>/    raw/*.jsonl, pages.csv, lines.csv, lines/, logs/
providers/              one environment per engine, each with its worker.py
```

## Usage

```bash
cd experiments
python inference_pipeline/run.py --dataset porto              # every stage, in order
python inference_pipeline/run.py --dataset porto --csv-only   # reassemble from raw/

# or stage by stage
python inference_pipeline/detect.py    --dataset porto
python inference_pipeline/recognize.py --dataset porto --engine calamari
python inference_pipeline/recognize.py --dataset porto --engine kraken --shard 0/8 &
python inference_pipeline/kraken_page.py --dataset porto
```

Then evaluate with `python eval_page.py --datasets porto` from `experiments/`.

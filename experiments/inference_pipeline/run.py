#!/usr/bin/env python3
"""Driver: runs the stages in order and assembles the CSVs for the evaluators.

The stages (`detect.py`, `recognize.py`, `kraken_page.py`) also run on their own.
No metric is computed here.

Usage: python inference_pipeline/run.py --dataset porto
       python inference_pipeline/run.py --dataset porto --limit 10
       python inference_pipeline/run.py --dataset porto --csv-only     # reassemble from raw/

Output in results/<dataset>/:
  raw/detect.jsonl    geometry and crop paths, per page
  raw/<engine>.jsonl  recognition, per chunk of crops
  raw/kraken_page.jsonl  condition 2, per page
  pages.csv, lines.csv   plain text for the evaluators
  lines/, logs/
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from stages import load_manifest, read_jsonl, read_shards, results_dir

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 7)

ENGINES = ("paddle", "calamari", "kraken")


def stage(script, *args):
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"\n=== {' '.join([script, *args])}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"{script} failed with code {result.returncode}")


def sharded_stage(script, args, shards):
    """Launch the shards together and wait; shards take whole chunks, so the
    output does not depend on the shard count."""
    if shards <= 1:
        return stage(script, *args)
    print(f"\n=== {' '.join([script, *args])}  x{shards} shards", flush=True)
    procs = [subprocess.Popen([sys.executable, str(HERE / script), *args,
                               "--shard", f"{i}/{shards}"])
             for i in range(shards)]
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"{script}: shards failed with {codes}")


def by_crop(raw_dir, engine):
    """Recognition is stored per chunk, possibly split across shards; flatten
    it to crop path -> line."""
    out = {}
    for record in read_shards(raw_dir, engine, key="chunk").values():
        for line in record.get("lines", []):
            if line.get("image_path"):
                out[line["image_path"]] = line
    return out


def build_csv(rows, out_dir):
    detect = read_jsonl(out_dir / "raw" / "detect.jsonl")
    standalone = read_shards(out_dir / "raw", "kraken_page")
    recognized = {e: by_crop(out_dir / "raw", e) for e in ENGINES}

    extras = [k for k in rows[0] if k not in ("id", "gold", "image_path")]
    with open(out_dir / "pages.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", *extras, "gold", "n_lines", "t_detect_s",
                    "paddle", "kraken_standalone"])
        for r in rows:
            page = detect.get(r["id"])
            if not page:
                continue
            crops = [l["image_path"] for l in page.get("lines", [])
                     if l.get("image_path")]
            paddle_text = "\n".join(
                recognized["paddle"].get(c, {}).get("text", "") for c in crops)
            w.writerow([r["id"], *[r.get(k, "") for k in extras], r["gold"],
                        len(crops), round(page.get("elapsed", 0), 3),
                        paddle_text,
                        "\n".join(l["text"] for l in
                                  standalone.get(r["id"], {}).get("lines", []))])

    folds = [f"calamari_fold_{i}" for i in range(5)]
    with open(out_dir / "lines.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_id", "line_idx", "line_path", "paddle", "calamari",
                    "kraken", *folds])
        for r in rows:
            page = detect.get(r["id"])
            if not page:
                continue
            for i, line in enumerate(page.get("lines", [])):
                crop = line.get("image_path")
                if not crop:
                    continue
                cal = recognized["calamari"].get(crop, {})
                variants = cal.get("variants", [])
                w.writerow([
                    r["id"], i, crop,
                    recognized["paddle"].get(crop, {}).get("text", ""),
                    cal.get("text", ""),
                    recognized["kraken"].get(crop, {}).get("text", ""),
                    *[(variants[j]["text"] if j < len(variants) else "")
                      for j in range(5)],
                ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--shards", type=int, default=1,
                    help="concurrent processes per engine; the only speed-up "
                         "available to kraken, which cannot batch")
    ap.add_argument("--kraken-shards", type=int, default=None,
                    help="override --shards for kraken alone")
    ap.add_argument("--skip-kraken-page", action="store_true",
                    help="skip condition 2, the only per-page stage left")
    ap.add_argument("--csv-only", action="store_true",
                    help="reassemble the CSVs from raw/, without running OCR")
    a = ap.parse_args()

    rows = load_manifest(a.dataset, a.limit)
    out_dir = results_dir(a.dataset)

    if not a.csv_only:
        limit = ["--limit", str(a.limit)] if a.limit else []
        stage("detect.py", "--dataset", a.dataset, *limit)
        for engine in ENGINES:
            shards = (a.kraken_shards if engine == "kraken"
                      and a.kraken_shards else a.shards)
            sharded_stage("recognize.py",
                          ["--dataset", a.dataset, "--engine", engine,
                           "--chunk", str(a.chunk)], shards)
        if not a.skip_kraken_page:
            shards = a.kraken_shards or a.shards
            sharded_stage("kraken_page.py",
                          ["--dataset", a.dataset, *limit], shards)

    build_csv(rows, out_dir)
    print(f"\n-> {out_dir}/  (pages.csv, lines.csv)")
    print(f"analysis: python eval_page.py --datasets {a.dataset}  (run from experiments/)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tesseract over whole pages, with its own layout analysis.

One call per page under `--psm 1`: Tesseract finds the blocks, the columns and
the reading order itself, and returns the text already ordered. Nothing from the
crop stage is used here, so this is the engine as a complete system.

Shards like the other page stages: page N goes to shard `N % n`.

Usage: TESSERACT_LANG=por python inference_pipeline/tesseract_page.py --dataset porto
       TESSERACT_LANG=por python inference_pipeline/tesseract_page.py --dataset porto --shard 0/7 &
       TESSDATA_PREFIX=~/tessdata_best TESSERACT_LANG=deu_latf python \
         inference_pipeline/tesseract_page.py --dataset reichsanzeiger \
         --label tesseract_best_latf_page
"""
import argparse
import sys
import time

from stages import (Sink, guarded, load_manifest, parse_shard, results_dir,
                    shard_path)
from workers import Worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="process only pages where index %% n == i, into a "
                         "shard file, so N processes can run concurrently")
    ap.add_argument("--label", default="tesseract_page",
                    help="name of the file in raw/. It exists to run the same "
                         "engine with other models without overwriting: "
                         "TESSDATA_PREFIX=<tessdata_best> --label "
                         "tesseract_best_page")
    a = ap.parse_args()

    rows = load_manifest(a.dataset, a.limit)
    shard = parse_shard(a.shard)
    mine = [r for i, r in enumerate(rows)
            if shard is None or i % shard[1] == shard[0]]
    out_dir = results_dir(a.dataset)
    label = a.label + (f" shard {a.shard}" if shard else "")
    print(f"{len(rows)} pages ({len(mine)} for this shard), "
          f"{a.dataset} / {label}")

    with Sink(shard_path(out_dir / "raw", a.label, shard), a.restart,
              a.retry_failed) as sink:
        pending = [r for r in mine if not sink.has(r["id"])]
        if len(pending) < len(mine):
            print(f"resuming: {len(mine) - len(pending)}/{len(mine)} pages done",
                  file=sys.stderr)
        if not pending:
            return

        with Worker("tesseract", log_dir=out_dir / "logs") as worker:
            started = time.perf_counter()
            for n, r in enumerate(pending, 1):
                call = guarded(worker, sink, r["id"])
                record = call({"mode": "segment", "page_path": r["image_path"]})
                record["id"] = r["id"]
                sink.write(record)
                if n % 10 == 0 or n == len(pending):
                    per_page = (time.perf_counter() - started) / n
                    print(f"  {n}/{len(pending)}  {per_page:.1f}s/page  "
                          f"~{(len(pending) - n) * per_page / 3600:.1f}h left",
                          file=sys.stderr)
            timing = worker.timing()

    failed = sum(1 for r in sink.done.values() if "error" in r)
    print(f"\ntesseract page: {timing['t_total']:.1f}s, "
          f"{timing['n_restarts']} restarts"
          + (f", {failed} pages failed" if failed else ""))
    print(f"-> {shard_path(out_dir / 'raw', a.label, shard)}")


if __name__ == "__main__":
    main()

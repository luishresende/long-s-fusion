#!/usr/bin/env python3
"""Stage 1: Paddle's detector cuts the line crops that every engine reads.

Usage: python inference_pipeline/detect.py --dataset porto
       PADDLE_UNCLIP=1.5 python inference_pipeline/detect.py --dataset porto --lines-dir lines_u15

Output in results/<dataset>/:
  raw/detect.jsonl   one record per page: crop paths, boxes, unclip, timing
  <lines-dir>/       the crops
"""
import argparse
import json
import sys
import time

from stages import Sink, guarded, load_manifest, results_dir, safe_stem
from workers import Worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lines-dir", default="lines",
                    help="crop subfolder; change it for an unclip sweep")
    ap.add_argument("--output", default="detect.jsonl")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()

    rows = load_manifest(a.dataset, a.limit)
    out_dir = results_dir(a.dataset)
    lines_dir = out_dir / a.lines_dir
    lines_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} pages, {a.dataset} / detect -> {a.lines_dir}/")

    with Sink(out_dir / "raw" / a.output, a.restart, a.retry_failed) as sink:
        pending = [r for r in rows if not sink.has(r["id"])]
        if len(pending) < len(rows):
            print(f"resuming: {len(rows) - len(pending)}/{len(rows)} pages done",
                  file=sys.stderr)
        if not pending:
            return

        with Worker("paddle", log_dir=out_dir / "logs") as worker:
            started = time.perf_counter()
            for n, r in enumerate(pending, 1):
                call = guarded(worker, sink, r["id"])
                record = call({"mode": "detect", "page_path": r["image_path"],
                               "lines_dir": str(lines_dir),
                               "stem": safe_stem(r["id"])})
                record["id"] = r["id"]
                sink.write(record)
                if n % 25 == 0 or n == len(pending):
                    per_page = (time.perf_counter() - started) / n
                    print(f"  {n}/{len(pending)}  {per_page:.2f}s/page  "
                          f"~{(len(pending) - n) * per_page / 3600:.1f}h left",
                          file=sys.stderr)
            timing = worker.timing()

    total = sum(len(r.get("lines", [])) for r in sink.done.values())
    failed = sum(1 for r in sink.done.values() if "error" in r)
    print(f"\n{total} crops from {len(sink.done)} pages"
          + (f"  ({failed} pages failed)" if failed else ""))
    print(f"detector: {timing['t_total']:.1f}s, load {timing['t_load']:.1f}s, "
          f"{timing['n_restarts']} restarts")
    print(f"-> {out_dir}/{a.lines_dir}/  and  raw/{a.output}")
    print(f"next: python inference_pipeline/recognize.py --dataset {a.dataset} --engine paddle")


if __name__ == "__main__":
    main()

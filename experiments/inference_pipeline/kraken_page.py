#!/usr/bin/env python3
"""Condition 2: Kraken segments (`blla`) and recognizes whole pages itself.

Page N goes to shard `N % n`.

Usage: python inference_pipeline/kraken_page.py --dataset porto
       python inference_pipeline/kraken_page.py --dataset porto --shard 0/4 &
       python inference_pipeline/kraken_page.py --dataset porto --shard 1/4 &
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
    a = ap.parse_args()

    rows = load_manifest(a.dataset, a.limit)
    shard = parse_shard(a.shard)
    mine = [r for i, r in enumerate(rows)
            if shard is None or i % shard[1] == shard[0]]
    out_dir = results_dir(a.dataset)
    label = "kraken standalone" + (f" shard {a.shard}" if shard else "")
    print(f"{len(rows)} pages ({len(mine)} for this shard), "
          f"{a.dataset} / {label}")

    with Sink(shard_path(out_dir / "raw", "kraken_page", shard), a.restart,
              a.retry_failed) as sink:
        pending = [r for r in mine if not sink.has(r["id"])]
        if len(pending) < len(mine):
            print(f"resuming: {len(mine) - len(pending)}/{len(mine)} pages done",
                  file=sys.stderr)
        if not pending:
            return

        with Worker("kraken", log_dir=out_dir / "logs") as worker:
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
    print(f"\nkraken standalone: {timing['t_total']:.1f}s, "
          f"{timing['n_restarts']} restarts"
          + (f", {failed} pages failed" if failed else ""))
    print(f"-> {shard_path(out_dir / 'raw', 'kraken_page', shard)}")


if __name__ == "__main__":
    main()

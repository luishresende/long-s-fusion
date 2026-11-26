#!/usr/bin/env python3
"""Stage 2: one engine reads stage 1's crops in large chunks, batched across pages.

Kraken reads one crop per call (see `KrakenProvider._recognize`), so shard it.

Usage: python inference_pipeline/recognize.py --dataset porto --engine calamari
       python inference_pipeline/recognize.py --dataset porto --engine kraken --shard 0/4 &
       python inference_pipeline/recognize.py --dataset porto --engine kraken --shard 1/4 &
"""
import argparse
import sys
import time

from stages import (Sink, guarded, parse_shard, read_jsonl, results_dir,
                    shard_path)
from workers import Worker

ENGINES = ("paddle", "calamari", "kraken", "hf", "tesseract")


def crops_of(detect_path):
    """Crop paths in page order, each tagged with the page that produced it."""
    out = []
    for page_id, record in read_jsonl(detect_path).items():
        for line in record.get("lines", []):
            if line.get("image_path"):
                out.append((page_id, line["image_path"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--chunk", type=int, default=256,
                    help="crops per worker call; the engine batches inside it")
    ap.add_argument("--detect", default="detect.jsonl")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="process only chunks where index %% n == i, into a "
                         "shard file, so N processes can run concurrently")
    ap.add_argument("--label", default=None,
                    help="name of the file in raw/ (default: the engine). It "
                         "exists to run the same engine with different weights "
                         "without overwriting: PADDLE_REC_DIR=<fine-tuned> "
                         "--engine paddle --label paddle_ft")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()

    out_label = a.label or a.engine
    shard = parse_shard(a.shard)
    out_dir = results_dir(a.dataset)
    crops = crops_of(out_dir / "raw" / a.detect)
    if not crops:
        sys.exit(f"no crops in raw/{a.detect}; run inference_pipeline/detect.py first")

    chunks = [crops[i:i + a.chunk] for i in range(0, len(crops), a.chunk)]
    mine = [i for i in range(len(chunks))
            if shard is None or i % shard[1] == shard[0]]
    shown = out_label + (f" shard {a.shard}" if shard else "")
    print(f"{len(crops)} crops in {len(chunks)} chunks "
          f"({len(mine)} for this shard), {a.dataset} / {shown}")

    sink = Sink(shard_path(out_dir / "raw", out_label, shard), a.restart,
                a.retry_failed, key="chunk")
    with sink:
        # a chunk written with another --chunk covers different crops
        stale = [i for i in mine if sink.has(i)
                 and sink.done[i].get("chunk_size") != a.chunk]
        if stale:
            print(f"! {len(stale)} chunks were written with another --chunk "
                  f"size and will be redone", file=sys.stderr)
        pending = [i for i in mine if not sink.has(i) or i in set(stale)]
        if len(pending) < len(mine):
            print(f"resuming: {len(mine) - len(pending)}/{len(mine)} chunks done",
                  file=sys.stderr)
        if not pending:
            return

        with Worker(a.engine, log_dir=out_dir / "logs") as worker:
            started = time.perf_counter()
            for n, index in enumerate(pending, 1):
                chunk = chunks[index]
                call = guarded(worker, sink, index, label=f"chunk {index}")
                record = call({"mode": "recognize",
                               "line_paths": [p for _, p in chunk]})
                record["chunk"] = index
                record["chunk_size"] = a.chunk
                record["page_ids"] = [pid for pid, _ in chunk]
                sink.write(record)
                done = n * a.chunk
                rate = done / max(1e-9, time.perf_counter() - started)
                print(f"  chunk {n}/{len(pending)}  {rate:.1f} crops/s  "
                      f"~{(len(pending) - n) * a.chunk / rate / 3600:.1f}h left",
                      file=sys.stderr)
            timing = worker.timing()

    failed = sum(1 for r in sink.done.values() if "error" in r)
    print(f"\n{shown}: {timing['t_total']:.1f}s, load {timing['t_load']:.1f}s, "
          f"{timing['n_restarts']} restarts"
          + (f", {failed} chunks failed" if failed else ""))
    print(f"-> {shard_path(out_dir / 'raw', out_label, shard)}")


if __name__ == "__main__":
    main()

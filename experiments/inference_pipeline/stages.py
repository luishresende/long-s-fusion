"""Shared pieces of the collection stages: manifest, checkpointing, jsonl."""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # experiments/inference_pipeline
# results/ and datasets/ live in experiments/, shared with the analyses
ROOT = HERE.parent                          # experiments/
sys.path.insert(0, str(ROOT.parent))
from fusion.paths import OutsideRepository, to_key, to_path    # noqa: E402,F401

csv.field_size_limit(10 ** 7)

# only the on-disk crop stem is sanitized; pages.csv keeps the original id
UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def safe_stem(page_id):
    return UNSAFE.sub("_", page_id)


def results_dir(dataset):
    d = ROOT / "results" / dataset
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def load_manifest(dataset, limit=None):
    """`image_path` in the manifest is relative to `experiments/`; it comes back as a key."""
    with open(ROOT / "datasets" / dataset / "manifest.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        p = Path(r["image_path"])
        r["image_path"] = to_key(p if p.is_absolute() else ROOT / p)
    return rows[:limit] if limit else rows


def read_jsonl(path, key="id"):
    """Ignores a truncated last line (an interrupted write)."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[record[key]] = record
    return out


def parse_shard(value):
    """`i/n` -> (i, n). Each shard takes the units where `index % n == i`."""
    if not value:
        return None
    index, total = (int(x) for x in value.split("/"))
    if not 0 <= index < total:
        raise ValueError(f"invalid shard {value!r}")
    return index, total


def shard_path(raw_dir, engine, shard=None):
    """One file per shard: records are far larger than an atomic append."""
    if shard is None:
        return Path(raw_dir) / f"{engine}.jsonl"
    index, total = shard
    return Path(raw_dir) / f"{engine}.{index}of{total}.jsonl"


def read_shards(raw_dir, engine, key="id"):
    """Merge every shard of an engine, whether it ran sharded or not."""
    merged = {}
    for path in sorted(Path(raw_dir).glob(f"{engine}.jsonl")) + \
            sorted(Path(raw_dir).glob(f"{engine}.*of*.jsonl")):
        merged.update(read_jsonl(path, key))
    return merged


class Sink:
    """Append-only jsonl flushed per record, so a stage resumes per unit of work."""

    def __init__(self, path, restart=False, retry_failed=False, key="id"):
        self.path = Path(path)
        self.key = key
        if restart:
            self.path.unlink(missing_ok=True)
        self.done = read_jsonl(self.path, key)
        self.retry_failed = retry_failed
        self.fh = self.path.open("a", encoding="utf-8")

    def has(self, unit):
        record = self.done.get(unit)
        if record is None:
            return False
        # failures count as done unless retry_failed
        return not (self.retry_failed and "error" in record)

    def write(self, record):
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.done[record[self.key]] = record

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def guarded(worker, sink, unit, label=""):
    """Wraps a worker call so a failure restarts the engine and retries once,
    then records the unit as failed and lets the run continue."""
    def run(request):
        error = None
        for attempt in (1, 2):
            try:
                return worker.call(request)
            except OutsideRepository:
                raise
            except Exception as exc:      # WorkerError, WorkerTimeout
                error = exc
                print(f"  ! {worker.name} failed on {label or unit} "
                      f"(attempt {attempt}): {type(exc).__name__}", flush=True)
                if attempt == 1:
                    worker.restart()
        return {"lines": [], "error": f"{type(error).__name__}: {error}"[:600]}
    return run

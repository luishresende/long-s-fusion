"""Client for the OCR workers.

Each engine runs as a persistent process in its own venv and speaks one JSON
object per line (`providers/worker.py`). Its stderr goes to a file, since an
unread pipe fills and blocks the worker, and every response has a timeout.
"""
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

from stages import OutsideRepository, to_key, to_path

# .../experiments/inference_pipeline -> .../experiments -> repo root
PROVIDERS = Path(__file__).resolve().parent.parent.parent / "providers"

TIMEOUT = 900.0   # generous: a dense page in Calamari goes past 60 s
PATH_FIELDS = ("page_path", "image_path", "lines_dir", "line_paths")


def absolute_paths(obj):
    """Path fields that are still absolute, at any depth."""
    if isinstance(obj, list):
        return [p for x in obj for p in absolute_paths(x)]
    if not isinstance(obj, dict):
        return []
    found = []
    for k, v in obj.items():
        values = v if isinstance(v, list) else [v]
        if k in PATH_FIELDS:
            found += [x for x in values if isinstance(x, str) and Path(x).is_absolute()]
        else:
            found += absolute_paths(v)
    return found


def convert_paths(obj, convert):
    """Applies `convert` to every path field, at any depth: keys go to the engine as paths, and come back as keys."""
    if isinstance(obj, list):
        return [convert_paths(x, convert) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k in PATH_FIELDS and isinstance(v, str):
            out[k] = convert(v)
        elif k in PATH_FIELDS and isinstance(v, list):
            out[k] = [convert(x) if isinstance(x, str) else x for x in v]
        else:
            out[k] = convert_paths(v, convert)
    return out


class WorkerError(RuntimeError):
    pass


class WorkerTimeout(WorkerError):
    pass


class Worker:
    def __init__(self, name, env=None, log_dir=None, timeout=TIMEOUT):
        self.name = name
        self.env = env
        self.timeout = timeout
        self.elapsed = 0.0
        self.calls = 0
        self.restarts = 0
        self.log_path = Path(log_dir or ".") / f"{name}.stderr.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._start()

    def _start(self):
        self.log = self.log_path.open("ab")
        started = time.perf_counter()
        self.proc = subprocess.Popen(
            ["uv", "run", "--directory", str(PROVIDERS / self.name),
             "python", "worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=self.env,
        )
        self.inbox = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        if self._receive(self.timeout) is None:
            raise WorkerError(f"{self.name} died while loading:\n{self.tail()}")
        self.load_time = time.perf_counter() - started

    def _pump(self, proc):
        """Dedicated thread: keeps stdout drained even while the main thread is
        doing something else."""
        for line in proc.stdout:
            self.inbox.put(line)
        self.inbox.put(None)      # EOF

    def _receive(self, timeout):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            raise WorkerTimeout(
                f"{self.name} did not answer in {timeout:.0f}s\n{self.tail()}")

    def tail(self, n=4000):
        try:
            self.log.flush()
            with self.log_path.open("rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - n))
                return fh.read().decode("utf-8", "replace")
        except Exception:
            return f"<see {self.log_path}>"

    def call(self, request):
        request = convert_paths(request, lambda k: str(to_path(k)))
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            raise WorkerError(f"{self.name} died:\n{self.tail()}")

        started = time.perf_counter()
        line = self._receive(self.timeout)
        if line is None:
            raise WorkerError(f"{self.name} died:\n{self.tail()}")
        self.elapsed += time.perf_counter() - started
        self.calls += 1
        result = json.loads(line)
        if "error" in result:
            raise WorkerError(
                f"{self.name}: {result['error']}\n{result.get('traceback', '')}")
        result = convert_paths(result, to_key)
        outside = absolute_paths(result)
        if outside:
            raise OutsideRepository(
                f"{self.name} returned {len(outside)} path(s) outside the repository, e.g. {outside[0]}. "
                "Outputs store paths relative to the repository root, so the data must live under it "
                "(a symbolic link is fine).")
        return result

    def restart(self):
        self.close()
        self.restarts += 1
        self._start()

    def close(self):
        """Does not block: CUDA teardown is sometimes slow and by this point the
        results are already in hand."""
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
        try:
            self.log.close()
        except Exception:
            pass

    def timing(self, since=None):
        """`since` = an earlier snapshot, to measure just the interval."""
        base = since or {"t_total": 0.0, "n_calls": 0}
        return {"t_total": round(self.elapsed - base["t_total"], 3),
                "t_load": round(self.load_time, 3),
                "n_calls": self.calls - base["n_calls"],
                "n_restarts": self.restarts}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

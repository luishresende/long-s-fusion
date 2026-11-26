"""Service loop: one persistent process per engine, one JSON object per line.

Protocol (stdin -> stdout, one JSON line each way):

    <- {"ready": "<name>"}          after the model is loaded
    -> {...}                        request, shape defined by the provider
    <- {...}                        response, or {"error": ..., "traceback": ...}

Terminates on EOF or an empty line.

Loading and inference run with stdout redirected to stderr: the OCR libraries
print banners, version warnings and progress bars without asking, and any stray
byte on stdout would corrupt the JSON stream.
"""
import contextlib
import dataclasses
import json
import sys
import traceback


def serve(provider, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    def reply(payload):
        if dataclasses.is_dataclass(payload):
            payload = dataclasses.asdict(payload)
        stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdout.flush()

    with contextlib.redirect_stdout(sys.stderr):
        provider.load()
    reply({"ready": provider.name})

    for line in stdin:
        line = line.strip()
        if not line:
            break
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = provider.predict(request)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
        reply(result)

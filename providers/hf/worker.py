"""Worker entrypoint for the hf provider. Runs in this directory's venv:

    HF_MODEL=microsoft/trocr-large-printed \
      uv run --directory providers/hf python worker.py
"""
from provider import HFProvider

from providers import serve

if __name__ == "__main__":
    serve(HFProvider())

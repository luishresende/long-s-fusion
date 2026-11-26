"""Worker entrypoint for the paddle provider. Runs in this directory's venv:

    uv run --directory providers/paddle python worker.py
"""
from provider import PaddleProvider

from providers import serve

if __name__ == "__main__":
    serve(PaddleProvider())

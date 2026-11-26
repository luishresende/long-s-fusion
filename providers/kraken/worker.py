"""Worker entrypoint for the kraken provider. Runs in this directory's venv:

    uv run --directory providers/kraken python worker.py
"""
from provider import KrakenProvider

from providers import serve

if __name__ == "__main__":
    serve(KrakenProvider())

"""Worker entrypoint for the calamari provider. Runs in this directory's venv:

    uv run --directory providers/calamari python worker.py
"""
from provider import CalamariProvider

from providers import serve

if __name__ == "__main__":
    serve(CalamariProvider())

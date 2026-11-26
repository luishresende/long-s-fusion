"""Worker entrypoint for the tesseract provider. Runs in this directory's venv:

    uv run --directory providers/tesseract python worker.py
"""
from provider import TesseractProvider

from providers import serve

if __name__ == "__main__":
    serve(TesseractProvider())

"""Keys of pages and crops: paths relative to the repository root, the same on every machine.

The outputs store keys, so the same crop matches across engines and across machines; the
real file is opened through `to_path` only when an image is read. Neither function calls
`resolve()`: `experiments/results` may be a symbolic link to a folder outside the repository.
"""
import os
from pathlib import Path

REPO = Path(os.path.abspath(Path(__file__).resolve().parent.parent))


class OutsideRepository(ValueError):
    """A path that has no key: the data must live under the repository root (a symbolic link is fine)."""


def to_key(path):
    """Absolute path inside the repository -> key; a key, or a path outside the repository, is returned as given."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return Path(os.path.normpath(p)).relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def to_path(key):
    """Key -> absolute path to open; an absolute path is returned as given."""
    p = Path(key)
    return p if p.is_absolute() else REPO / p

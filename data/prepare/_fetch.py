#!/usr/bin/env python3
"""Acquisition helpers shared by the corpus preparation scripts.

Each `<corpus>.py` calls `ensure_*` before converting. If the raw distribution
is already on disk, nothing is downloaded. This is what makes the replication
package runnable from an empty directory.

Sources, all public:

    PORTO           HuggingFace `LIACC/PORTO`, CC-BY-4.0 (handled in porto.py)
    Reichsanzeiger  github.com/UB-Mannheim/reichsanzeiger-gt, CC0
    OCR17+          github.com/Heresta/OCR17plus
    IMPACT          lab.kb.nl/dataset/ground-truth-impact-project, public domain
"""
import os
import subprocess
import sys
import urllib.request

_opener = urllib.request.build_opener()
_opener.addheaders = [('User-Agent', 'Mozilla/5.0 (long-s-fusion replication)')]
urllib.request.install_opener(_opener)


def _say(msg):
    print(f'[fetch] {msg}', flush=True)


def ensure_clone(dest, url, depth=1):
    """Clone `url` into `dest` unless it is already there.

    Shallow by default: these repositories carry the images, and the history is
    not needed to reproduce anything.
    """
    if os.path.isdir(dest) and os.listdir(dest):
        _say(f'{os.path.basename(dest)} already present, skipping clone')
        return dest
    _say(f'cloning {url} -> {dest}')
    cmd = ['git', 'clone', '--quiet']
    if depth:
        cmd += ['--depth', str(depth)]
    cmd += [url, dest]
    subprocess.run(cmd, check=True)
    return dest


def ensure_download(dest_path, url):
    """Download `url` to `dest_path` unless the file is already there.

    Writes to a `.part` file first, so an interrupted download is never mistaken
    for a complete one on the next run.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        _say(f'{os.path.basename(dest_path)} already present, skipping download')
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    partial = dest_path + '.part'
    _say(f'downloading {url}')

    last = [-1]

    def progress(blocks, block_size, total):
        if total <= 0:
            return
        done = blocks * block_size
        pct = min(100, done * 100 // total)
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            sys.stdout.write(f'\r[fetch]   {pct:3d}%  '
                             f'{done / 2**30:.1f} / {total / 2**30:.1f} GB')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, partial, reporthook=progress)
    print()
    os.replace(partial, dest_path)
    return dest_path


def ensure_unzip(zip_path, dest_dir, marker=None):
    """Extract `zip_path` into `dest_dir` unless `marker` already exists."""
    import zipfile
    if marker and os.path.exists(marker):
        _say(f'{os.path.basename(dest_dir)} already extracted, skipping')
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    _say(f'extracting {os.path.basename(zip_path)} -> {dest_dir}')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    return dest_dir

#!/usr/bin/env python3
"""Canonical form for comparing ground truth and hypothesis: unifies encoding
(same printed glyph, different codepoints), never spelling.

It also applies the map the PaddleOCR encoder applied to the training labels.
"""
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / 'experiments'

# copy of PaddleOCR's ppocrv6_dict.txt (18,708 entries, md5 68c581dfdd791747b70eb3ac2b35233d)
DEFAULT_DICT = HERE / 'ppocrv6_dict.txt'

# one printed glyph, several letters
LIGATURES = {
    # MUFI private-area codepoints, written as chr()
    chr(0xEADA): 'st',  chr(0xEBA2): 'ſi', chr(0xEBA3): 'ſl',
    chr(0xEBA6): 'ſſ',  chr(0xEBA7): 'ſſi', chr(0xEEC5): 'ct',
    'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
    'ﬅ': 'ſt', 'ﬆ': 'st',
    # ĳ also appears as ij on other pages; expanding both sides scores them equally
    'ĳ': 'ij', 'Ĳ': 'IJ',
}

# one printed glyph, different codepoints (1:1)
EQUIVALENTS = {
    # ʃ and ∫ are the same type as ſ; PORTO uses all three
    'ʃ': 'ſ', '∫': 'ſ',
    # hyphens outside the charset; ¬ is in the charset and stays
    '‐': '-', '‑': '-', '⸗': '-', '­': '-', '‧': '-',
    '„': '"',                       # German low quotes (6,189)
    '⁄': '/',                       # fraction slash
    'ꝛ': 'r',                       # r rotunda; training learned `r`
    'ℳ': 'M', 'ℛ': 'R',        # training learned `M`/`R`
    # Cyrillic homoglyphs typed by transcribers inside Latin text
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X',
}

# U+FFFD, e-ditiones' unreadable marker, is dropped
MARKERS = {'�'}

# only with fold_s=True; ʃ and ∫ are already ſ by then
FOLD_S = str.maketrans({'ſ': 's'})

_charset_cache = {}


def charset_path(path=DEFAULT_DICT):
    """Where the charset file actually is."""
    path = Path(path)
    if not path.exists():
        # last resort: the copy inside PaddleOCR, if it happens to be installed
        try:
            import paddlex
            alt = (Path(paddlex.__file__).parent / 'repo_manager' / 'repos'
                   / 'PaddleOCR' / 'ppocr' / 'utils' / 'dict' / 'ppocrv6_dict.txt')
            if alt.exists():
                path = alt
        except ImportError:
            pass
        if not path.exists():
            raise FileNotFoundError(
                f'charset not found at {path}. It ships as '
                f'fusion/ppocrv6_dict.txt; if that file is missing, copy '
                f'ppocr/utils/dict/ppocrv6_dict.txt from PaddleOCR.')
    return path


def charset_list(path=DEFAULT_DICT):
    """The classes in order; the CTC projection row is index + 1 (blank at 0)."""
    return [l.rstrip('\n') for l in
            open(charset_path(path), encoding='utf-8')]


def charset(path=DEFAULT_DICT):
    """Classes the model can emit. Anything outside is unreachable."""
    path = charset_path(path)
    if path not in _charset_cache:
        _charset_cache[path] = set(charset_list(path))
    return _charset_cache[path]


def _resolve(ch, chars):
    """Fallback outside the charset: strip combining accents, then NFKC, else
    drop, as the encoder did with the training labels."""
    base = ''.join(c for c in unicodedata.normalize('NFD', ch)
                   if not unicodedata.combining(c))
    if base and all(c in chars for c in base):
        return base
    nk = ''.join(EQUIVALENTS.get(c, c)
                 for c in unicodedata.normalize('NFKC', ch))
    if nk and all(c in chars or c == ' ' for c in nk):
        return nk
    return ''


def canonicalize(txt, *, fold_s=False, space='line', chars=None, dropped=None):
    """Canonical form of `txt`; apply it to both sides of a comparison.

    `fold_s`  collapse `ſ`≡`s`
    `space`   'line': no collapsing (per-line gold); 'page': collapse whitespace
              runs (paragraph gold against detected lines)
    `chars`   emittable classes; None uses the PP-OCRv6 charset, False disables
              the fallback
    `dropped` optional Counter that receives what was discarded
    """
    if space not in ('line', 'page'):
        raise ValueError(
            f"space={space!r} is not valid; use 'line' or 'page'")
    if chars is None:
        chars = charset()

    txt = unicodedata.normalize('NFC', txt)
    for k, v in LIGATURES.items():
        if k in txt:
            txt = txt.replace(k, v)

    out = []
    for ch in txt:
        if ch.isspace():
            out.append(' ')
            continue
        if ch in MARKERS:
            if dropped is not None:
                dropped[ch] += 1
            continue
        ch = EQUIVALENTS.get(ch, ch)
        if chars is not False and ch not in chars:
            fixed = _resolve(ch, chars)
            if dropped is not None and not fixed:
                dropped[ch] += 1
            out.append(fixed)
            continue
        out.append(ch)

    t = ''.join(out)
    if fold_s:
        t = t.translate(FOLD_S)
    if space == 'page':
        squeezed, prev = [], None
        for c in t:
            if c == ' ' and prev == ' ':
                continue
            squeezed.append(c)
            prev = c
        t = ''.join(squeezed)
    return t.strip()


def canonicalize_marked(txt, marks):
    """`canonicalize` that carries one mark per character through to the output."""
    out, kept = [], []
    for ch, m in zip(txt, marks):
        piece = ' ' if ch.isspace() else canonicalize(ch)
        out.append(piece)
        kept.extend(([m] + [None] * (len(piece) - 1)) if piece else [])
    t = ''.join(out)
    lead = len(t) - len(t.lstrip())
    h = t.strip()
    assert h == canonicalize(txt)
    return h, kept[lead:lead + len(h)]


def _selftest():
    """Check fixed cases, then report unreachable codepoints in the gold sets on disk."""
    import csv
    csv.field_size_limit(10 ** 7)
    F = EXPERIMENTS / 'finetune'
    chars = charset()

    cases = [
        ('sijn',            chr(0xEBA2) + 'jn'),
        ('Vlissingen,',     'Vli' + chr(0xEBA6) + 'ingen,'),
        ('geslacht',        'ge' + chr(0xEBA3) + 'acht'),
        ('Vorst',           'Vor' + chr(0xEADA)),
        ('tracteert',       'tra' + chr(0xEEC5) + 'eert'),
        ('permissien',      'permi' + chr(0xEBA7) + 'en'),
        ('Untersuchungs-',  'Unterſuchungs⸗'),
        ('af-slaen',        'af‑' + chr(0xEBA3) + 'aen'),
        ('schrijven',       'ſchrĳven'),
    ]
    print('--- cases (fold_s=True: check the letter, not the spelling) ---')
    ok = True
    for expected, given in cases:
        got = canonicalize(given, fold_s=True, chars=chars)
        ok &= got == expected
        print(f'  {"ok " if got == expected else "FAIL"} {given!r:34} -> '
              f'{got!r:22} (expected {expected!r})')

    # strict: the three long-s spellings must converge to ſ
    print('\n--- cases (fold_s=False: unify encoding, keep spelling) ---')
    strict = [
        ('caſa', 'caſa'), ('caſa', 'caʃa'), ('caſa', 'ca∫a'),
        ('Vliſſingen', 'Vli' + chr(0xEBA6) + 'ingen'),
        ('ſijn', chr(0xEBA2) + 'jn'),
    ]
    for expected, given in strict:
        got = canonicalize(given, chars=chars)
        ok &= got == expected
        print(f'  {"ok " if got == expected else "FAIL"} {given!r:34} -> '
              f'{got!r:22} (expected {expected!r})')

    # coverage needs the corpora; skipped on a fresh checkout
    texts = {}
    for split in ('test', 'val'):
        lst = F / 'train_data' / f'{split}_list.txt'
        if lst.exists():
            texts[split] = [l.rstrip('\n').partition('\t')[2]
                            for l in open(lst, encoding='utf-8')]
    for ds in ('porto', 'impact'):
        p = F / 'transfer' / ds / 'pages.csv'
        if p.exists():
            texts[ds] = [r['gold'] for r in csv.DictReader(open(p, encoding='utf-8'))]

    if not texts:
        print('\n--- coverage: no corpora on disk, section skipped ---')
        print('\nself-test', 'PASSED' if ok else 'FAILED', '(cases only)')
        return ok

    print('\n--- coverage over the gold sets ---')
    for name, lines in texts.items():
        d = Counter()
        before = after = reachable = 0
        for t in lines:
            before += sum(1 for c in t if not c.isspace())
            c = canonicalize(t, space='page' if name in ('porto', 'impact')
                             else 'line', chars=chars, dropped=d)
            after += sum(1 for x in c if not x.isspace())
            reachable += sum(1 for x in c if x in chars)
        lost = sum(d.values())
        print(f'  {name:7} {before:>9} chars -> {after:>9} '
              f'| unreachable: {after - reachable} | dropped: {lost}'
              + (f' {dict(d.most_common(5))}' if lost else ''))
    print('\nself-test', 'PASSED' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)

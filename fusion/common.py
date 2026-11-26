"""Asymmetric fusion rules and metrics, shared across datasets and engines."""
import time
from collections import Counter
from dataclasses import dataclass, fields, replace
from contextlib import contextmanager

from rapidfuzz.distance import Levenshtein

from . import canon

# single-letter spellings of ſ; the MUFI ligature U+EBA6 (ſſ) would shift the alignment
LONG_S = frozenset("ſʃ∫")

BEFORE = set("?!;:,.»)]\"'")
AFTER = set("«([\"'")

# a space injected next to a hyphen is almost always damage (34 hits vs 189 on PORTO)
HYPHEN = set("-‐‑¬⸗")


@dataclass(frozen=True)
class Rules:
    """Rule and veto toggles; the defaults are the production configuration.

    `tilde` is off: 26.8:1 in Portuguese, 0.14:1 in Dutch, 0.00:1 in German.
    """
    glyph: bool = True             # rule 1: base `f` + specialist `ſ` -> `ſ`
    space: bool = True             # rule 2: space injection
    veto_punct: bool = True        # veto component of rule 2 (BEFORE/AFTER)
    veto_hyphen: bool = True       # veto component of rule 2
    veto_digit: bool = False       # measured, not adopted: unstable across corpora
    tilde: bool = False            # rule 3: not implemented by `merge_line`
    tilde_margin: float = 0.0      # minimum score margin for rule 3 to fire

    @classmethod
    def from_text(cls, spec):
        """`Rules.from_text('tilde,-veto_hyphen')`: comma-separated, `-` disables."""
        names = {f.name for f in fields(cls)}
        kw = {}
        for item in (spec or '').split(','):
            item = item.strip()
            if not item:
                continue
            on, name = (False, item[1:]) if item.startswith('-') else (True, item)
            if name not in names:
                raise ValueError(f'unknown rule: {name!r}; known: {sorted(names)}')
            kw[name] = on
        return replace(cls(), **kw)


DEFAULT = Rules()


def veto_space(prev, nxt, rules):
    """True if the space must not be injected between `prev` and `nxt`."""
    if rules.veto_punct and (nxt in BEFORE or prev in AFTER):
        return True
    if rules.veto_hyphen and (nxt in HYPHEN or prev in HYPHEN):
        return True
    if rules.veto_digit and (nxt.isdigit() or prev.isdigit()):
        return True
    return False


def merge_line(base, specialist, stats=None, glyph=None, space=None, rules=DEFAULT):
    """Asymmetric fusion.

    The specialist may only turn `f` into `ſ` and inject a space (subject to the
    vetoes) in `base`. `glyph`/`space` override `rules`; the per-book gate
    passes `glyph=False`.
    """
    if glyph is not None or space is not None:
        rules = replace(rules,
                        glyph=rules.glyph if glyph is None else glyph,
                        space=rules.space if space is None else space)
    if rules.tilde:
        raise ValueError('the `tilde` rule needs per-character distributions; '
                         '`merge_line` does not implement it')
    glyph, space = rules.glyph, rules.space
    stats = stats if stats is not None else {}
    out = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(base, specialist):
        if tag == "replace":
            for a, b in zip(base[i1:i2], specialist[j1:j2]):
                if glyph and a == "f" and b in LONG_S:
                    out.append("ſ")
                    stats["swapped_f_for_s"] = stats.get("swapped_f_for_s", 0) + 1
                else:
                    out.append(a)
            out.append(base[i1 + min(i2 - i1, j2 - j1):i2])
        elif tag in ("equal", "delete"):
            out.append(base[i1:i2])
        elif tag == "insert" and space:
            fragment = specialist[j1:j2]
            if fragment.strip() == "":
                nxt = base[i1] if i1 < len(base) else ""
                prev = out[-1][-1] if out and out[-1] else ""
                if veto_space(prev, nxt, rules):
                    continue
                out.append(fragment)
                stats["injected_space"] = stats.get("injected_space", 0) + 1
    return "".join(out)


def cer_wer(gold, hyp, *, fold_s=False, space="page"):
    g = canon.canonicalize(gold, fold_s=fold_s, space=space)
    h = canon.canonicalize(hyp, fold_s=fold_s, space=space)
    gw, hw = g.split(), h.split()
    return (Levenshtein.distance(g, h), len(g),
            Levenshtein.distance(gw, hw), len(gw))


class Acc:
    """Micro-averaged CER/WER over the canonical form.

    `space='line'` applies no post-processing (per-line gold); `'page'` collapses
    whitespace runs (paragraph gold against detected lines). `fold_s=True` folds
    `ſ`≡`s`.
    """

    def __init__(self, *, fold_s=False, space="page"):
        self.fold_s, self.space = fold_s, space
        self.ed = self.length = self.wed = self.wlen = 0

    def add(self, gold, hyp):
        ed, length, wed, wlen = cer_wer(gold, hyp, fold_s=self.fold_s,
                                        space=self.space)
        self.ed += ed
        self.length += length
        self.wed += wed
        self.wlen += wlen

    @property
    def cer(self):
        return self.ed / self.length if self.length else 0.0

    @property
    def wer(self):
        return self.wed / self.wlen if self.wlen else 0.0


class BagAcc:
    """Bag-of-words: compares the word multiset, ignoring order.

    After Pletschacher et al. (HIP 2015), for pages with no single reading order.
    """

    def __init__(self, *, fold_s=False, space="page"):
        self.fold_s, self.space = fold_s, space
        self.inter = self.n_gold = self.n_hyp = 0

    def add(self, gold, hyp):
        gw = Counter(canon.canonicalize(gold, fold_s=self.fold_s,
                                        space=self.space).split())
        hw = Counter(canon.canonicalize(hyp, fold_s=self.fold_s,
                                        space=self.space).split())
        self.inter += sum((gw & hw).values())
        self.n_gold += sum(gw.values())
        self.n_hyp += sum(hw.values())

    @property
    def precision(self):
        return self.inter / self.n_hyp if self.n_hyp else 0.0

    @property
    def recall(self):
        return self.inter / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def wer(self):
        return 1 - self.f1


@contextmanager
def timer():
    """`with timer() as t: ...` -> t.elapsed in seconds, after the block."""
    class T:
        elapsed = None
    t = T()
    started = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - started

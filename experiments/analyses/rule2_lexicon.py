#!/usr/bin/env python3
"""Can a modern dictionary tell whether a rule-2 space split a word? Per work:
coverage first, then precision on the decided cases against `rule2_gap.py`.

Usage: python analyses/rule2_lexicon.py --dataset porto
"""
import argparse
import bisect
import collections
import csv
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion.common import AFTER, BEFORE, HYPHEN
from fusion import canon

RES = Path(__file__).resolve().parent.parent / "results"
csv.field_size_limit(10 ** 7)
FOLD = str.maketrans({"ſ": "s", "ʃ": "s", "∫": "s"})


def marked_merge(base, spec, glyph=True, space=True):
    out, mark = [], []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(base, spec):
        if tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for a, b in zip(base[i1:i2], spec[j1:j2]):
                if glyph and a == "f" and b == "ſ":
                    out.append("ſ"); mark.append("g")
                else:
                    out.append(a); mark.append(None)
            rest = base[i1 + n:i2]
            out.append(rest); mark.extend([None] * len(rest))
        elif tag in ("equal", "delete"):
            out.append(base[i1:i2]); mark.extend([None] * (i2 - i1))
        elif tag == "insert" and space:
            frag = spec[j1:j2]
            if frag.strip() == "":
                nxt = base[i1] if i1 < len(base) else ""
                prev = "".join(out)[-1] if out and "".join(out) else ""
                if nxt in BEFORE or prev in AFTER:
                    continue
                if nxt in HYPHEN or prev in HYPHEN:
                    continue
                out.append(frag); mark.extend(["e"] * len(frag))
    return "".join(out), mark


def token(h, i):
    """The word rule 2 cut at index `i`: (left, right)."""
    a = i
    while a > 0 and h[a - 1].isalpha():
        a -= 1
    b = i + 1
    while b < len(h) and h[b].isalpha():
        b += 1
    return h[a:i], h[i + 1:b]


def norm(w):
    return w.translate(FOLD).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="porto")
    ap.add_argument("--specialist", default="calamari")
    ap.add_argument("--lang", default="pt")
    a = ap.parse_args()

    from spellchecker import SpellChecker
    sp = SpellChecker(language=a.lang)
    D = sp.word_frequency.dictionary

    def known(w):
        return len(w) > 0 and norm(w) in D

    with open(RES / a.dataset / "pages.csv", encoding="utf-8") as fh:
        pages = list(csv.DictReader(fh))
    lines = collections.defaultdict(list)
    with open(RES / a.dataset / "lines.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lines[r["page_id"]].append(r)
    for k in lines:
        lines[k].sort(key=lambda r: int(r["line_idx"]))

    key = "book" if pages and "book" in pages[0] else "_"
    S = collections.defaultdict(collections.Counter)
    cross = collections.Counter()

    for p in pages:
        rows = lines.get(p["id"], [])
        book = p.get(key) or "_"
        txt, mk = [], []
        for i, r in enumerate(rows):
            if i:
                txt.append("\n"); mk.append(None)
            if r[a.specialist]:
                t, m = marked_merge(r["paddle"], r[a.specialist])
            else:
                t, m = r["paddle"], [None] * len(r["paddle"])
            txt.append(t); mk.extend(m)
        h, mk = canon.canonicalize_marked("".join(txt), mk)
        g = canon.canonicalize(p["gold"])
        ns, nl = g.count("s"), g.count("ſ")
        if ns + nl == 0:
            continue

        pos_idx = {}
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(g, h):
            if tag == "equal":
                for k in range(i2 - i1):
                    pos_idx[j1 + k] = i1 + k
            elif tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    pos_idx[j1 + k] = i1 + k
        aligned = sorted(pos_idx)

        for i, m in enumerate(mk):
            if m != "e" or i >= len(h):
                continue
            # gold verdict, only to score the dictionary
            if i in pos_idx:
                truth = "hit" if g[pos_idx[i]].isspace() else "dmg"
            else:
                lo_k, hi_k = bisect.bisect_left(aligned, i), bisect.bisect_right(aligned, i)
                lo = aligned[lo_k - 1] if lo_k else None
                hi = aligned[hi_k] if hi_k < len(aligned) else None
                if lo is None or hi is None:
                    truth = "und"
                else:
                    b = g[pos_idx[lo] + 1:pos_idx[hi]]
                    truth = "hit" if (b != "" and b.strip() == "") else ("dmg" if b == "" else "und")

            left, right = token(h, i)
            joined = left + right
            if not joined.isalpha() or len(joined) < 2:
                verd = "no word"
            else:
                jk, lk, rk = known(joined), known(left), known(right)
                if jk and not (lk and rk):
                    verd = "join"            # the dictionary says: do not inject
                elif (lk and rk) and not jk:
                    verd = "split"           # the dictionary says: inject
                elif jk and lk and rk:
                    verd = "both"
                else:
                    verd = "neither"         # out of dictionary
            S[book][verd] += 1
            S[book]["fired"] += 1
            if truth in ("hit", "dmg"):
                cross[(verd, truth)] += 1

    print(f"{a.dataset.upper()}: rule 2 against the `{a.lang}` dictionary "
          f"({len(D):,} words)\n")
    print(f"{'work':9s} {'inj.':>6s} | {'join':>7s} {'split':>7s} {'both':>6s} "
          f"{'neither':>7s} {'no word':>7s} | {'decided':>7s}")
    print("-" * 72)
    for b, d in sorted(S.items(), key=lambda x: -x[1]["fired"]):
        n = d["fired"]
        dec = d["join"] + d["split"]
        print(f"{b:9s} {n:6,d} | {d['join']:7,d} {d['split']:7,d} {d['both']:6,d} "
              f"{d['neither']:7,d} {d['no word']:7,d} | {100 * dec / n:6.1f}%")

    print(f"\nThe dictionary against the ground truth\n{'=' * 72}")
    print(f"{'verdict':12s} {'hit':>8s} {'damage':>8s} {'total':>8s}  "
          f"{'verdict precision':>22s}")
    for verd in ("join", "split", "both", "neither", "no word"):
        hit, dmg = cross[(verd, "hit")], cross[(verd, "dmg")]
        n = hit + dmg
        if not n:
            continue
        # `join` is meant to predict damage; the others, a hit
        prec = (100 * dmg / n) if verd == "join" else (100 * hit / n)
        target = "damage" if verd == "join" else "hit"
        print(f"{verd:12s} {hit:8,d} {dmg:8,d} {n:8,d}  {prec:17.1f}% {target}")

    j_h, j_d = cross[("join", "hit")], cross[("join", "dmg")]
    tot_h = sum(cross[(v, "hit")] for v in
                ("join", "split", "both", "neither", "no word"))
    tot_d = sum(cross[(v, "dmg")] for v in
                ("join", "split", "both", "neither", "no word"))
    print(f"\nIf rule 2 were blocked where the dictionary says `join`:")
    print(f"  damage avoided: {j_d:,} of {tot_d:,}  ({100 * j_d / tot_d:.1f}%)")
    print(f"  hits lost     : {j_h:,} of {tot_h:,}  ({100 * j_h / tot_h:.1f}%)")
    if j_h:
        print(f"  avoided:lost ratio = {j_d / j_h:.1f}:1")
    print(f"  resulting hit rate: "
          f"{100 * (tot_h - j_h) / max(1, (tot_h - j_h) + (tot_d - j_d)):.2f}%  "
          f"(was {100 * tot_h / (tot_h + tot_d):.2f}%)")


if __name__ == "__main__":
    main()

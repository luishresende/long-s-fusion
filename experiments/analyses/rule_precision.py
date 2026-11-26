#!/usr/bin/env python3
"""Accuracy of the f->s swap and the space injection, with a 1:1 normalization
(ſ ʃ -> s, ʒ->z, u->v, j->i) so each marked output character stays aligned.

Usage: uv run python rule_precision.py --dataset porto
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for `import fusion`
from fusion.common import AFTER, BEFORE, HYPHEN

RES = Path(__file__).resolve().parent.parent / "results"
csv.field_size_limit(10 ** 7)
ONE_TO_ONE = str.maketrans({"ſ": "s", "ʃ": "s", "ʒ": "z", "u": "v", "j": "i"})


def marked_merge(base, spec, glyph=True, space=True):
    """Same as common.merge_line, but returns (text, marks)."""
    out, mark = [], []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(base, spec):
        if tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for a, b in zip(base[i1:i2], spec[j1:j2]):
                if glyph and a == "f" and b == "ſ":
                    out.append("s")
                    mark.append("g")
                else:
                    out.append(a)
                    mark.append(None)
            rest = base[i1 + n:i2]
            out.append(rest)
            mark.extend([None] * len(rest))
        elif tag in ("equal", "delete"):
            out.append(base[i1:i2])
            mark.extend([None] * (i2 - i1))
        elif tag == "insert" and space:
            frag = spec[j1:j2]
            if frag.strip() == "":
                nxt = base[i1] if i1 < len(base) else ""
                prev = "".join(out)[-1] if out and "".join(out) else ""
                if nxt in BEFORE or prev in AFTER:
                    continue
                if nxt in HYPHEN or prev in HYPHEN:   # mirrors common.py
                    continue
                out.append(frag)
                mark.extend(["e"] * len(frag))
    return "".join(out), mark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="porto")
    ap.add_argument("--specialist", default="calamari")
    ap.add_argument("--gate", type=float, default=0.30)
    a = ap.parse_args()

    with open(RES / a.dataset / "pages.csv", encoding="utf-8") as fh:
        pages = list(csv.DictReader(fh))
    lines = collections.defaultdict(list)
    with open(RES / a.dataset / "lines.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lines[r["page_id"]].append(r)
    for k in lines:
        lines[k].sort(key=lambda r: int(r["line_idx"]))

    key = "book" if pages and "book" in pages[0] else "_"
    c = collections.defaultdict(lambda: [0, 0])
    for p in pages:
        g = p.get(key) or "_"
        for r in lines.get(p["id"], []):
            c[g][0] += r[a.specialist].count("ſ")
            c[g][1] += r[a.specialist].count("f")
    blocked = {g for g, (s, f) in c.items() if (s / (s + f) if s + f else 0) < a.gate}

    R = {"g": collections.Counter(), "e": collections.Counter()}
    unmatched_g = unmatched_e = 0

    for p in pages:
        rows = lines.get(p["id"], [])
        glyph = a.gate <= 0 or (p.get(key) or "_") not in blocked
        txt, mk = [], []
        pad = []
        for i, r in enumerate(rows):
            if i:
                txt.append("\n"); mk.append(None); pad.append("\n")
            if r[a.specialist]:
                t, m = marked_merge(r["paddle"], r[a.specialist], glyph=glyph)
            else:
                t, m = r["paddle"], [None] * len(r["paddle"])
            txt.append(t); mk.extend(m); pad.append(r["paddle"])
        h = "".join(txt).translate(ONE_TO_ONE)
        g = p["gold"].translate(ONE_TO_ONE)
        pd = "".join(pad).translate(ONE_TO_ONE)

        pos_map = {}
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(g, h):
            if tag == "equal":
                for k in range(i2 - i1):
                    pos_map[j1 + k] = g[i1 + k]
            elif tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    pos_map[j1 + k] = g[i1 + k]
        for idx, m in enumerate(mk):
            if m is None:
                continue
            target = pos_map.get(idx)
            if target is None:
                if m == "g":
                    unmatched_g += 1
                    R["g"]["error"] += 1
                    R["g"]["became <unmatched>"] += 1
                else:
                    unmatched_e += 1
                continue
            expected = "s" if m == "g" else " "
            R[m]["hit" if target == expected else "error"] += 1
            if target != expected:
                R[m][f"became {target!r}"] += 1

        # recall of rule 1: gold 's' where Paddle wrote 'f' and it stayed 'f'
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(g, pd):
            if tag != "replace":
                continue
            for k in range(min(i2 - i1, j2 - j1)):
                if g[i1 + k] == "s" and pd[j1 + k] == "f":
                    R["g"]["opportunities"] += 1

    print(f"\n{a.dataset.upper()}, specialist={a.specialist}, gate={a.gate}\n")
    for m, name in (("g", "RULE 1: f->s swap"), ("e", "RULE 2: space injection")):
        d = R[m]
        fired = d["hit"] + d["error"]
        if not fired:
            continue
        print(f"{name}")
        print(f"  firings             : {fired}")
        print(f"  hits                : {d['hit']}  ({100*d['hit']/fired:.2f}%)")
        print(f"  damages             : {d['error']}  ({100*d['error']/fired:.2f}%)")
        print(f"  hit:damage ratio    : {d['hit']/d['error']:.1f}:1"
              if d["error"] else "  hit:damage ratio    : no damage")
        if m == "g" and d["opportunities"]:
            print(f"  opportunities (gold 's' x paddle 'f'): {d['opportunities']}"
                  f"  -> coverage {100*d['hit']/d['opportunities']:.1f}%")
        worst = [(k, v) for k, v in d.items() if k.startswith("became")]
        worst.sort(key=lambda x: -x[1])
        if worst:
            print("  where it goes wrong: " + ", ".join(f"{k}={v}" for k, v in worst[:6]))
        print()
    if unmatched_g or unmatched_e:
        print(f"(no correspondence in the alignment: {unmatched_g} from rule 1, "
              f"{unmatched_e} from rule 2; those of rule 1 are counted as damage)")


if __name__ == "__main__":
    main()

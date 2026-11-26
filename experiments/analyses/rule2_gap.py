#!/usr/bin/env python3
"""Rule 2 judged by what the gold holds between the aligned neighbours of each
injected space, with every damage classified by its context:

    whitespace there            -> hit
    nothing there (adjacent)    -> damage, the rule split a gold word
    other characters there      -> undetermined

Usage: python analyses/rule2_gap.py --dataset porto --examples 10
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


def context(h, i, w=14):
    return (h[max(0, i - w):i] + "[" + h[i] + "]" + h[i + 1:i + w]).replace("\n", "⏎")


def classify(h, i):
    """Why the injected space at `i` has no gold counterpart."""
    left, right = h[max(0, i - 3):i], h[i + 1:i + 4]
    if any(c in HYPHEN for c in left + right) or "\n" in left + right:
        return "line break or hyphenation"
    if left[-1:].isalpha() and right[:1].isalpha():
        return "inside a ground-truth word"
    if left[-1:].isdigit() or right[:1].isdigit():
        return "next to a digit"
    if not left[-1:].isalnum() or not right[:1].isalnum():
        return "next to punctuation or symbol"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="porto")
    ap.add_argument("--specialist", default="calamari")
    ap.add_argument("--gate", type=float, default=0.30)
    ap.add_argument("--examples", type=int, default=10)
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
    blocked = ({g for g, (s, f) in c.items()
                if (s / (s + f) if s + f else 0) < a.gate} if a.gate > 0 else set())

    verdict = collections.Counter()
    old_verdict = collections.Counter()
    reason = collections.Counter()
    bystratum = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)

    for p in pages:
        rows = lines.get(p["id"], [])
        glyph = (p.get(key) or "_") not in blocked
        txt, mk = [], []
        for i, r in enumerate(rows):
            if i:
                txt.append("\n"); mk.append(None)
            if r[a.specialist]:
                t, m = marked_merge(r["paddle"], r[a.specialist], glyph=glyph)
            else:
                t, m = r["paddle"], [None] * len(r["paddle"])
            txt.append(t); mk.extend(m)
        h, mk = canon.canonicalize_marked("".join(txt), mk)
        g = canon.canonicalize(p["gold"])
        ns, nl = g.count("s"), g.count("ſ")
        if ns + nl == 0:
            continue
        fr = nl / (ns + nl)
        stratum = ("diplomatic" if fr >= 0.30
                   else "modernized" if fr < 0.02 else "intermediate")

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
            # old criterion, for comparison
            old_verdict["dropped" if i not in pos_idx else
                        ("hit" if pos_idx.get(i) is not None
                         and g[pos_idx[i]] == " " else "damage")] += 1
            # gap criterion; an aligned injected space has a gold character under it
            if i in pos_idx:
                v = "hit" if g[pos_idx[i]].isspace() else "damage"
                lo = hi = i
            else:
                lo_k = bisect.bisect_left(aligned, i)
                hi_k = bisect.bisect_right(aligned, i)
                lo = aligned[lo_k - 1] if lo_k else None
                hi = aligned[hi_k] if hi_k < len(aligned) else None
                if lo is None or hi is None:
                    verdict["undetermined"] += 1
                    bystratum[stratum]["undetermined"] += 1
                    continue
                between = g[pos_idx[lo] + 1:pos_idx[hi]]
                if between != "" and between.strip() == "":
                    v = "hit"
                elif between == "":
                    v = "damage"
                else:
                    v = "undetermined"
            verdict[v] += 1
            bystratum[stratum][v] += 1
            if v == "damage":
                why = classify(h, i)
                reason[why] += 1
                if len(examples[why]) < a.examples:
                    examples[why].append((
                        p["id"], stratum, context(h, i),
                        g[max(0, pos_idx[lo] - 14):pos_idx[hi] + 14].replace("\n", "⏎")))

    tot = sum(verdict.values())
    print(f"{a.dataset.upper()}, specialist={a.specialist}, gate={a.gate}\n")
    print(f"RULE 2, {tot:,} firings\n{'=' * 62}")
    print("old criterion (gold character under the space):")
    o = sum(old_verdict.values())
    for k_ in ("hit", "damage", "dropped"):
        print(f"  {k_:14} {old_verdict[k_]:6,d}  ({100 * old_verdict[k_] / o:5.2f}%)")
    ac, da = old_verdict["hit"], old_verdict["damage"]
    print(f"  published rate: {100 * ac / (ac + da):.2f}% "
          f"(over {ac + da:,}, ignoring the dropped ones)")
    print(f"\ngap criterion (does the gold have a space between the neighbours?):")
    for k_ in ("hit", "damage", "undetermined"):
        print(f"  {k_:14} {verdict[k_]:6,d}  ({100 * verdict[k_] / tot:5.2f}%)")
    ac2, da2 = verdict["hit"], verdict["damage"]
    print(f"  corrected rate: {100 * ac2 / (ac2 + da2):.2f}%  (over {ac2 + da2:,})")

    print(f"\nby stratum, gap criterion\n{'=' * 62}")
    for s in ("diplomatic", "intermediate", "modernized"):
        d = bystratum[s]
        n = d["hit"] + d["damage"]
        if not n:
            continue
        print(f"  {s:14} hit {100 * d['hit'] / n:5.2f}%  "
              f"({d['hit']:,} of {n:,})")

    print(f"\nREASON FOR THE DAMAGE\n{'=' * 62}")
    nd = sum(reason.values())
    for why, v in reason.most_common():
        print(f"  {why:34} {v:6,d}  ({100 * v / nd:5.1f}%)")
    for why, _ in reason.most_common(3):
        print(f"\n  --- {why}")
        for pid, st, hyp, gold in examples[why][:4]:
            print(f"    {pid} [{st}]")
            print(f"      fusion: ...{hyp}...")
            print(f"      gold  : ...{gold}...")


if __name__ == "__main__":
    main()

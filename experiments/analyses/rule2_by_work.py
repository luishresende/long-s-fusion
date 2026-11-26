#!/usr/bin/env python3
"""Rule 2 per work: the injection rate and the specialist's ſ ratio as candidate
gates, against the hit rate of the gap criterion of `rule2_gap.py`.

Usage: python analyses/rule2_by_work.py --dataset porto
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="porto")
    ap.add_argument("--specialist", default="calamari")
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
    S = collections.defaultdict(collections.Counter)

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
        fr = nl / (ns + nl)
        S[book]["pages"] += 1
        S[book]["chars"] += len(h)
        S[book]["gold_s"] += ns
        S[book]["gold_long"] += nl
        if fr < 0.02:
            S[book]["pages_mod"] += 1
        for r in rows:
            S[book]["spec_long"] += r[a.specialist].count("ſ")
            S[book]["spec_f"] += r[a.specialist].count("f")

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
            S[book]["fired"] += 1
            if i in pos_idx:
                v = "hit" if g[pos_idx[i]].isspace() else "dmg"
            else:
                lo_k, hi_k = bisect.bisect_left(aligned, i), bisect.bisect_right(aligned, i)
                lo = aligned[lo_k - 1] if lo_k else None
                hi = aligned[hi_k] if hi_k < len(aligned) else None
                if lo is None or hi is None:
                    v = "und"
                else:
                    b = g[pos_idx[lo] + 1:pos_idx[hi]]
                    v = "hit" if (b != "" and b.strip() == "") else ("dmg" if b == "" else "und")
            S[book][v] += 1

    print(f"{a.dataset.upper()}, specialist {a.specialist}, rule 2 per work\n")
    print(f"{'work':9s} {'pgs':>5s} {'%mod':>6s} | {'ſ spec.':>7s} {'ſ gold':>7s} | "
          f"{'inj.':>6s} {'/1k ch':>7s} | {'hit':>7s}")
    print("-" * 74)
    rows_out = []
    for b, d in sorted(S.items(), key=lambda x: -x[1]["fired"]):
        hit, dmg = d["hit"], d["dmg"]
        acc = 100 * hit / (hit + dmg) if hit + dmg else float("nan")
        rate = 1000 * d["fired"] / d["chars"] if d["chars"] else 0
        sp = d["spec_long"] / (d["spec_long"] + d["spec_f"]) if d["spec_long"] + d["spec_f"] else 0
        gl = d["gold_long"] / (d["gold_long"] + d["gold_s"]) if d["gold_long"] + d["gold_s"] else 0
        pmod = 100 * d["pages_mod"] / d["pages"]
        rows_out.append((b, rate, acc, sp, gl))
        print(f"{b:9s} {d['pages']:5d} {pmod:5.1f}% | {sp:7.3f} {gl:7.3f} | "
              f"{d['fired']:6,d} {rate:7.2f} | {acc:6.2f}%")

    print(f"\nPearson correlation against the hit rate per work:")
    import statistics as st
    def pearson(xs, ys):
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return num / den if den else float("nan")
    ok = [r for r in rows_out if r[2] == r[2]]
    accs = [r[2] for r in ok]
    print(f"  injection rate per 1k characters : r = {pearson([r[1] for r in ok], accs):+.3f}")
    print(f"  ſ ratio of the specialist        : r = {pearson([r[3] for r in ok], accs):+.3f}")
    print(f"  ſ ratio of the ground truth      : r = {pearson([r[4] for r in ok], accs):+.3f}")


if __name__ == "__main__":
    main()

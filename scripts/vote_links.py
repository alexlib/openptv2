"""Per-link ensemble voting with agreement flags (prototype).

Takes several engine outputs (ptv_is.*) on identical inputs plus the
reference, and writes agreement sidecars agree.<frame> next to them:
each row -> flag, top choice, vote count, per-engine prev values.

Flags: 4 = unanimous, 3 = majority, 2 = split (2-2 or 2-1-1),
1 = all disagree, 0 = fewer than 2 engines have this row.

Also reports the no-engine-right set (union misses) with agreement
patterns for manual adjudication.

Run from repo root:
    uv run python -u scripts/vote_links.py --ref <refdir> --eng <name=dir>...
        --out <outdir> --first F --last L
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def read_prev(d: Path, f: int):
    lines = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return [int(l.split()[0]) for l in lines[1: n + 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--eng", action="append", required=True,
                    help="NAME=DIR, repeatable")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    args = ap.parse_args()

    engines = []
    for spec in args.eng:
        name, d = spec.split("=", 1)
        engines.append((name, Path(d)))
    names = [n for n, _ in engines]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    P = {name: {f: read_prev(d, f) for f in range(args.first, args.last + 1)}
         for name, d in engines}
    R = {f: read_prev(args.ref, f) for f in range(args.first, args.last + 1)}

    patterns: Counter = Counter()
    hard = []  # (frame, row, ref_prev, votes)
    for f in range(args.first, args.last + 1):
        lines = [f"{len(R[f])}"]
        n = min(len(R[f]), *(len(P[x][f]) for x in names))
        for i in range(n):
            votes = [P[x][f][i] for x in names]
            cnt = Counter(votes)
            top, c = cnt.most_common(1)[0]
            if len(cnt) == 1:
                flag = 4
            elif c >= 3:
                flag = 3
            elif c == 2:
                flag = 2
            else:
                flag = 1
            lines.append(f"{i:5d} {flag} {top:5d} {c} "
                         + " ".join(f"{v:5d}" for v in votes))
            if R[f][i] >= 0:
                if top == R[f][i] and flag >= 3:
                    patterns["majority-right"] += 1
                if not any(v == R[f][i] for v in votes):
                    hard.append((f, i, R[f][i], votes))
                    patterns[f"hard:flag{flag}"] += 1
        (out / f"agree.{f}").write_text("\n".join(lines) + "\n")

    print(f"engines: {names}", flush=True)
    print(f"union misses (no engine right): {len(hard)}", flush=True)
    print(f"patterns: {dict(patterns)}", flush=True)
    for f, i, r, v in hard:
        print(f"  frame {f} row {i:4d} ref_prev={r:4d} "
              + " ".join(f"{n}={x:4d}" for n, x in zip(names, v)),
              flush=True)


if __name__ == "__main__":
    main()

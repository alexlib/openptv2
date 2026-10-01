"""Summarise a two_phase option grid from the eval.json files of synth_bench.

    uv run python scripts/tune_summary.py q_seed q_young [--cases C1 C2 ...]

Prints, per case, velocity error (vErr) and true points kept (yield) for every
grid value, plus the mean change against plain two_phase over all cases.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

WORK = Path("/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29/runs")
CASES = [
    "L1_d1_c0_k1_n200",
    "L1_d1_c0_k1_n200_s0.08",
    "L1_d2_c0.3_k4_n150",
    "L1_d4_c0_k1_n150",
    "L1_d1_c0_k8_n150",
]


def load(case, label):
    p = WORK / case / label / "eval.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "vel_err_rel" not in d:
        return None
    return d["vel_err_rel"], d["yield"], d["ghost_point_frac"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs=2)
    ap.add_argument("--cases", nargs="*", default=CASES)
    a = ap.parse_args()
    k1, k2 = a.keys
    labels = {}
    for d in (WORK / a.cases[0]).iterdir():
        m = re.fullmatch(rf"two_phase\+{k1}=([\d.]+)\+{k2}=([\d.]+)", d.name)
        if m:
            labels[(float(m[1]), float(m[2]))] = d.name
    base = {c: load(c, "two_phase") for c in a.cases}
    print(
        f"{k1:>7} {k2:>7} | "
        + " | ".join(f"{c[3:20]:>17}" for c in a.cases)
        + " |  mean dvErr  mean dYield  mean dGhost"
    )
    for key in sorted(labels):
        row, dv, dy, dg = [], [], [], []
        for c in a.cases:
            r = load(c, labels[key])
            if r is None or base[c] is None:
                row.append(f"{'-':>17}")
                continue
            row.append(f"{r[0]:.4f}/{r[1]:.3f}".rjust(17))
            dv.append(r[0] - base[c][0])
            dy.append(r[1] - base[c][1])
            dg.append(r[2] - base[c][2])
        n = len(dv)
        print(
            f"{key[0]:7g} {key[1]:7g} | "
            + " | ".join(row)
            + f" | {np.mean(dv):+.4f} ({n}) {np.mean(dy):+.4f} {np.mean(dg):+.4f}"
            if n
            else f"{key[0]:7g} {key[1]:7g} | incomplete"
        )
    print(
        "baseline "
        + " | ".join(
            f"{base[c][0]:.4f}/{base[c][1]:.3f}" if base[c] else "-" for c in a.cases
        )
    )


if __name__ == "__main__":
    main()

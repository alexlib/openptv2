"""Plot the union-miss links for manual adjudication.

For every reference link that NO engine reproduced, draws the local 3D
neighborhood (X-Y and X-Z): frame N-1 particles, frame N candidates, the
reference assignment (green) and each engine's vote (parity red, default
orange, 4BE purple, two-phase brown; abstentions labeled). Prints a
judgment table with step/acc/crowding/history per choice.

Run from repo root:
    uv run python -u scripts/plot_hard_links.py --out scratch/hard_links
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
ENGS = {
    "parity": REPO / "scratch" / "_wp1_ab" / "res",
    "default": REPO / "scratch" / "_ens_lr1" / "res",
    "4be": REPO / "scratch" / "_ens_4be" / "res",
    "2p": REPO / "scratch" / "_ens_2p" / "res_2p",
}
ECOL = {"parity": "#c0392b", "default": "#e67e22", "4be": "#8e44ad",
        "2p": "#5d4037"}
FIRST, LAST = 100001, 100010


def read_prev(d: Path, f: int):
    lines = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return [int(l.split()[0]) for l in lines[1: n + 1]]


def read_xyz(f: int):
    lines = (REF / f"rt_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in l.split()[1:4]] for l in lines[1: n + 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "scratch" / "hard_links")
    ap.add_argument("--first", type=int, default=100003)
    ap.add_argument("--last", type=int, default=100010)
    ap.add_argument("--per-fig", type=int, default=6)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    names = list(ENGS)
    P = {n: {f: read_prev(d, f) for f in range(args.first, args.last + 1)}
         for n, d in ENGS.items()}
    R = {f: read_prev(REF, f) for f in range(args.first, args.last + 1)}
    XYZ = {f: read_xyz(f) for f in range(FIRST - 1, LAST + 1)
           if (REF / f"rt_is.{f}").exists()}

    # ref chains for history length
    RP = {f: read_prev(REF, f) for f in range(FIRST, LAST + 1)}
    RN = {}
    for f in range(FIRST, LAST + 1):
        lines = (REF / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        RN[f] = [int(l.split()[1]) for l in lines[1: n + 1]]
    histlen = {}
    for f in range(FIRST, LAST + 1):
        for i in range(len(RP[f])):
            if RP[f][i] < 0:
                ch = [(f, i)]
                cf, ci = f, i
                while cf <= LAST and ci < len(RN[cf]) and RN[cf][ci] >= 0:
                    ci = RN[cf][ci]
                    cf += 1
                    ch.append((cf, ci))
                for k, node in enumerate(ch):
                    histlen[node] = (len(ch), k)

    hard = []
    for f in range(args.first, args.last + 1):
        n = min(len(R[f]), *(len(P[x][f]) for x in names))
        for i in range(n):
            if R[f][i] < 0:
                continue
            votes = [P[x][f][i] for x in names]
            if not any(v == R[f][i] for v in votes):
                hard.append((f, i, R[f][i], votes))
    print(f"union misses: {len(hard)}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = []
    for idx0 in range(0, len(hard), args.per_fig):
        chunk = hard[idx0: idx0 + args.per_fig]
        fig, axs = plt.subplots(len(chunk), 2,
                                figsize=(13, 2.6 * len(chunk)))
        if len(chunk) == 1:
            axs = np.array([axs])
        for ax_row, (f, i, r, votes) in zip(axs, chunk):
            X1, X3 = XYZ[f - 1], XYZ[f]
            tgt = X3[i]
            box = (np.abs(X1 - tgt).max(axis=1) < 4.0)
            box3 = (np.abs(X3 - tgt).max(axis=1) < 4.0)
            src = X1[r] if 0 <= r < len(X1) else None
            ax_xy, ax_xz = ax_row
            for ax, dims, xl, yl in ((ax_xy, (0, 1), "x (mm)", "y (mm)"),
                                     (ax_xz, (0, 2), "x (mm)", "z (mm)")):
                ax.scatter(X1[box][:, dims[0]], X1[box][:, dims[1]],
                           s=8, color="#bbbbbb", label="frame N-1")
                ax.scatter(X3[box3][:, dims[0]], X3[box3][:, dims[1]],
                           s=14, color="#7fb3d5", label="frame N cands")
                ax.scatter([tgt[dims[0]]], [tgt[dims[1]]], s=90,
                           facecolor="none", edgecolor="black", lw=1.2)
                if src is not None:
                    ax.annotate("", (tgt[dims[0]], tgt[dims[1]]),
                                xytext=(src[dims[0]], src[dims[1]]),
                                arrowprops={"arrowstyle": "->", "color": "green",
                                            "lw": 2})
                    ax.scatter([src[dims[0]]], [src[dims[1]]], s=60,
                               marker="*", color="green")
                for ename, v in zip(names, votes):
                    if v is not None and v >= 0 and v < len(X1):
                        o = X1[v]
                        ax.annotate("", (tgt[dims[0]], tgt[dims[1]]),
                                    xytext=(o[dims[0]], o[dims[1]]),
                                    arrowprops={"arrowstyle": "->",
                                                "color": ECOL[ename],
                                                "lw": 1.2, "ls": "--"})
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.grid(alpha=0.3)
                ax.set_aspect("equal", adjustable="datalim")
            # feature row for the judgment table
            step = float(np.linalg.norm(tgt - src)) if src is not None else -1
            crowd = int(box3.sum()) - 1
            pp = RP[f - 1][r] if 0 <= r < len(RP[f - 1]) else -1
            if 0 <= pp and (f - 2) in XYZ and pp < len(XYZ[f - 2]):
                acc = float(np.linalg.norm(tgt - 2 * src + XYZ[f - 2][pp]))
            else:
                acc = float("nan")
            L, _ = histlen.get((f - 1, r), (1, 0))
            cnt = Counter(votes)
            agree = ",".join(f"{n}={v}" for n, v in zip(names, votes))
            rows.append((f, i, r, step, acc, crowd, L, agree))
            ax_xy.set_title(f"{f}:{i} ref {r}->{i} "
                            f"step={step:.2f} acc={acc:.2f} crowd={crowd} "
                            f"hist={L} | {agree}", fontsize=8, loc="left")
        handles = [Line2D([0], [0], color="green", lw=2, label="3dptv ref"),
                   Line2D([0], [0], color="#bbbbbb", marker="o", ls="",
                           label="frame N-1"),
                   Line2D([0], [0], color="#7fb3d5", marker="o", ls="",
                           label="frame N"),
                   *[Line2D([0], [0], color=c, lw=1.2, ls="--", label=n)
                     for n, c in ECOL.items()]]
        fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = out / f"hard_{idx0 // args.per_fig + 1}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"wrote {path}", flush=True)

    print(f"\n{'frame':>8}{'row':>6}{'ref':>6}{'step':>7}{'acc':>7}"
          f"{'crowd':>7}{'hist':>6}   votes(parity,default,4be,2p)",
          flush=True)
    import csv
    with open(out / "hard_links.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "row", "ref_prev", "step", "acc", "crowd",
                    "histlen", "parity", "default", "4be", "2p"])
        for (f, i, r, step, acc, crowd, L, agree) in rows:
            v = agree.split(",")
            w.writerow([f, i, r, f"{step:.3f}", f"{acc:.3f}", crowd, L,
                        *[x.split("=")[1] for x in v]])
            print(f"{f:>8}{i:>6}{r:>6}{step:>7.2f}{acc:>7.2f}"
                  f"{crowd:>7}{L:>6}   {agree}", flush=True)


if __name__ == "__main__":
    main()

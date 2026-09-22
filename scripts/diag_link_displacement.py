"""Overlay the per-link displacement distributions of two tracking runs.

Both runs must have been made from the SAME rt_is.*, so a link-by-link
comparison is exact: position data is identical, only the linking differs.

Read the shape, not just the counts:

  * a HARD CLIFF in one run at some |step| -> a search-volume clamp
    (dvxmin/dvxmax mapping, or the searchquader pixel projection)
  * a SOFT DEFICIT in the tail, no cliff  -> candidates are found but
    rejected downstream (dacc / dangle gating)
  * a deficit at ALL speeds               -> candidate search or particle
    addition differs, not a kinematic bound

Run from repo root:
    uv run python scripts/diag_link_displacement.py \
        --a <folder>/res_orig --b <folder>/res_retry0 \
        --rt <folder>/res --first 100002 --last 100010
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_links(res: Path, frame: int):
    p = res / f"ptv_is.{frame}"
    if not p.exists():
        return []
    lines = p.read_text().strip().splitlines()
    if not lines:
        return []
    n = int(lines[0])
    return [int(line.split()[0]) for line in lines[1 : n + 1]]


def read_xyz(res: Path, frame: int):
    p = res / f"rt_is.{frame}"
    lines = p.read_text().strip().splitlines()
    n = int(lines[0])
    return np.array(
        [[float(v) for v in line.split()[1:4]] for line in lines[1 : n + 1]]
    )


def steps(links_dir: Path, rt_dir: Path, first: int, last: int):
    out = []
    for f in range(first, last + 1):
        prev = read_links(links_dir, f)
        cur = read_xyz(rt_dir, f)
        old = read_xyz(rt_dir, f - 1)
        for i, p in enumerate(prev):
            if 0 <= p < len(old) and i < len(cur):
                out.append(float(np.linalg.norm(cur[i] - old[p])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="reference run links")
    ap.add_argument("--b", type=Path, required=True, help="run under test links")
    ap.add_argument("--rt", type=Path, help="rt_is source (default: --a)")
    ap.add_argument("--first", type=int, default=100002)
    ap.add_argument("--last", type=int, default=100010)
    ap.add_argument("--label-a", default="reference")
    ap.add_argument("--label-b", default="openptv2")
    ap.add_argument("--dvmax", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("scratch/link_displacement.png"))
    args = ap.parse_args()

    rt = args.rt or args.a
    sa = steps(args.a, rt, args.first, args.last)
    sb = steps(args.b, rt, args.first, args.last)

    print(f"{args.label_a}: {len(sa)} links   {args.label_b}: {len(sb)} links")
    print(f"\n{'quantile':>10}{args.label_a:>14}{args.label_b:>14}")
    for q in (50, 75, 90, 95, 99, 100):
        print(f"{q:>9}%{np.percentile(sa, q):>14.4f}{np.percentile(sb, q):>14.4f}")
    print(f"\n{'mean':>10}{sa.mean():>14.4f}{sb.mean():>14.4f}")

    # Where do the missing links live? Bin both and difference the counts.
    hi = max(np.percentile(sa, 99.9), np.percentile(sb, 99.9))
    edges = np.linspace(0, hi, 26)
    ca, _ = np.histogram(sa, bins=edges)
    cb, _ = np.histogram(sb, bins=edges)
    print(f"\n{'step range (mm)':>20}{args.label_a:>12}{args.label_b:>12}"
          f"{'missing':>10}{'%lost':>8}")
    for k in range(len(edges) - 1):
        if ca[k] == 0 and cb[k] == 0:
            continue
        miss = int(ca[k] - cb[k])
        pct = miss / ca[k] * 100 if ca[k] else 0.0
        print(f"{edges[k]:>9.3f}-{edges[k + 1]:<10.3f}{ca[k]:>12}{cb[k]:>12}"
              f"{miss:>10}{pct:>7.1f}%")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax.hist(sa, bins=edges, alpha=0.55, label=f"{args.label_a} ({len(sa)})",
            color="#2a6ebb")
    ax.hist(sb, bins=edges, alpha=0.55, label=f"{args.label_b} ({len(sb)})",
            color="#c0392b")
    if args.dvmax:
        ax.axvline(args.dvmax, color="k", ls="--", label=f"dvxmax = {args.dvmax}")
    ax.set_ylabel("links")
    ax.set_title("Per-link displacement: where the missing links live")
    ax.legend()
    ax.grid(alpha=0.3)

    mid = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(ca > 0, (ca - cb) / ca, np.nan)
    ax2.bar(mid, frac * 100, width=(edges[1] - edges[0]) * 0.9, color="#7f8c8d")
    ax2.axhline(0, color="k", lw=0.8)
    if args.dvmax:
        ax2.axvline(args.dvmax, color="k", ls="--")
    ax2.set_xlabel("link displacement (mm)")
    ax2.set_ylabel(f"% of {args.label_a} links\nmissing in {args.label_b}")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

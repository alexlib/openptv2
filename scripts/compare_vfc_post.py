"""Compare two run stores AFTER the post-processing of the pipeline.

The raw tracking statistics are misleading: a false link makes a trajectory
longer.  The pipeline (openptv-cloud) therefore post-processes the tracking:
reconnect the pieces across gaps, drop the spikes, smooth the velocities.  This
script applies the same steps (from openptv2.reconnect, openptv2.smoothness_filter
and openptv2.quality_post) to one or more run stores and prints the final
numbers.

Usage:
    uv run python scripts/compare_vfc_post.py LABEL=PATH [LABEL=PATH ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openptv2.quality_post import weighted_savgol  # noqa: E402
from openptv2.reconnect import reconnect_pieces  # noqa: E402
from openptv2.smoothness_filter import smoothness_keep  # noqa: E402
from openptv2.storage import RunStore  # noqa: E402


def load_trajectories(store_path: Path, fps: float = 5000.0):
    """The flat trajectory arrays of a sealed store (trajid, frame, pos)."""
    store = RunStore.open(store_path, mode="r")
    traj = store.root["traj"]
    length = np.asarray(traj["length"])
    first_row = np.asarray(traj["first_row"])
    n_rows = int(length.sum())
    pos = np.asarray(store.root["trajectories"]["pos"][:n_rows], dtype=np.float64)
    time = np.asarray(store.root["trajectories"]["time"][:n_rows], dtype=np.float64)
    trajid = np.asarray(traj["trajid"])
    # the frame of every point
    frame = np.empty(n_rows, dtype=np.int64)
    starts = np.zeros(len(length), dtype=np.int64)
    np.cumsum(length[:-1], out=starts[1:])
    row = np.arange(n_rows)
    which = np.searchsorted(starts, row, side="right") - 1
    frame[:] = time.astype(np.int64)
    return trajid[which], frame, pos


def stats(trajid, frame, pos, label, fps):
    length = np.bincount(trajid)
    length = length[length > 0]
    n = len(length)
    points = int(length.sum())
    # the acceleration over the consecutive frames
    order = np.lexsort((frame, trajid))
    t, f, p = trajid[order], frame[order], pos[order]
    same = np.diff(t) == 0
    idx = np.flatnonzero(same)
    consec = (f[idx + 1] - f[idx - 1]) == 2
    sel = idx[consec]
    a = np.zeros((0, 3))
    if len(sel):
        a = (p[sel + 1] - 2 * p[sel] + p[sel - 1]) / ((1.0 / fps) ** 2)
    a_flat = a.ravel()
    m2 = float(np.mean(a_flat**2)) if a_flat.size else float("nan")
    ka = float(np.mean(a_flat**4) / m2**2) if m2 > 0 else float("nan")
    jumps = int((np.abs(a_flat) > 5000.0).sum()) if a_flat.size else 0
    return {
        "label": label,
        "tracks": n,
        "points": points,
        "mean_len": float(length.mean()) if n else 0.0,
        "median_len": float(np.median(length)) if n else 0.0,
        "over_30": float(np.mean(length > 30)) if n else 0.0,
        "kurtosis": ka,
        "jumps": jumps,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="LABEL=PATH pairs")
    ap.add_argument("--fps", type=float, default=5000.0)
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--filter-k", type=float, default=6.0)
    ap.add_argument("--window", type=int, default=21)
    args = ap.parse_args(argv)

    rows = []
    for item in args.runs:
        if "=" not in item:
            print(f"Error: {item!r} is not LABEL=PATH")
            return 2
        label, path = item.split("=", 1)
        store = Path(path)
        if not store.exists():
            store = store / "res" / "run.zarr"
        trajid, frame, pos = load_trajectories(store, args.fps)
        rows.append(stats(trajid, frame, pos, f"{label} raw", args.fps))

        # the post-processing of the pipeline
        new_tid = reconnect_pieces(
            trajid, frame, pos, max_gap=args.max_gap, tol_sigmas=4.0
        )
        keep = smoothness_keep(
            new_tid, frame, pos, k=args.filter_k, window=args.window, deg=2
        )
        t2, f2, p2 = new_tid[keep], frame[keep], pos[keep]
        rows.append(stats(t2, f2, p2, f"{label} +reconnect", args.fps))
        rows.append(stats(t2, f2, p2, f"{label} +reconnect+filter", args.fps))

    head = (
        f"{'Run':<34} {'Tracks':>8} {'Points':>9} {'Mean len':>9} {'Median len':>11} "
        f"{'>30 frames':>11} {'K_a':>10} {'Jumps':>9}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        ka = f"{r['kurtosis']:.1f}" if np.isfinite(r["kurtosis"]) else "n/a"
        print(
            f"{r['label']:<34} {r['tracks']:8d} {r['points']:9d} "
            f"{r['mean_len']:9.2f} {r['median_len']:11.1f} {r['over_30']:10.1%} "
            f"{ka:>10} {r['jumps']:9d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

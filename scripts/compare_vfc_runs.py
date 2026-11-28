#!/usr/bin/env python
"""Compare tracking runs of the same data (for example VFC off and VFC on).

Prints one row per run: the number of trajectories, the number of points, the
track lengths and the number of long tracks.  Use it to check that a new
option does not break the trajectories.

Usage:
    uv run python scripts/compare_vfc_runs.py LABEL=PATH [LABEL=PATH ...]

PATH is a run store (res/run.zarr) or a folder that contains res/run.zarr.
The store must be sealed, that is it must contain the traj index.  Run the
tracking with `openptv2-batch --mode tracking` to get one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if (p / "res" / "run.zarr").exists():
        return p / "res" / "run.zarr"
    if p.name.endswith(".zarr") and p.exists():
        return p
    if (p / "run.zarr").exists():
        return p / "run.zarr"
    if p.is_dir():
        stores = sorted(p.glob("res/*.zarr")) or sorted(p.glob("*.zarr"))
        if len(stores) == 1:
            return stores[0]
        if stores:
            names = ", ".join(s.name for s in stores)
            raise FileNotFoundError(
                f"{path} holds several stores ({names}); give the store itself"
            )
    raise FileNotFoundError(f"No run store at {path}")


def _stats(store_path: Path, fps: float) -> dict:
    import zarr

    z = zarr.open(store_path, mode="r")
    if "traj" not in z:
        raise KeyError(
            f"{store_path} has no traj index. The run is not sealed; "
            "run the tracking with --mode tracking."
        )
    traj = z["traj"]
    length = np.asarray(traj["length"])
    first = np.asarray(traj["first"])
    last = np.asarray(traj["last"])
    n_points = int(length.sum())
    span = int(last.max() - first.min() + 1) if len(length) else 0

    ka, n_acc, a_rms, a_p999, n_jumps = _acceleration_kurtosis(z, fps)
    return {
        "tracks": int(len(length)),
        "points": n_points,
        "mean_len": float(length.mean()) if len(length) else 0.0,
        "median_len": float(np.median(length)) if len(length) else 0.0,
        "max_len": int(length.max()) if len(length) else 0,
        "long_tracks": int((length >= 5).sum()),
        "span": span,
        "points_per_frame": n_points / span if span else 0.0,
        "frac_over_10": float(np.mean(length > 10)) if len(length) else 0.0,
        "frac_over_30": float(np.mean(length > 30)) if len(length) else 0.0,
        "kurtosis": ka,
        "n_acc": n_acc,
        "a_rms": a_rms,
        "a_p999": a_p999,
        "n_jumps": n_jumps,
    }


def _acceleration_kurtosis(z, fps: float) -> tuple[float, int, float, float, int]:
    """K_a = <a^4>/<a^2>^2 over the consecutive frames of every trajectory.

    Same statistic as `openptv2.benchmarking.metrics.acceleration_kurtosis`
    (Gaussian noise gives 3, real turbulence 10-50, a false link pushes it
    much higher), computed here on the flat store arrays.
    """
    traj = z["traj"]
    length = np.asarray(traj["length"])
    first_row = np.asarray(traj["first_row"])
    if len(length) == 0:
        return float("nan"), 0, float("nan")
    n_rows = int(length.sum())
    pos = np.asarray(z["trajectories"]["pos"][:n_rows], dtype=np.float64)
    time = np.asarray(z["trajectories"]["time"][:n_rows], dtype=np.float64)

    # The index of the first point of every trajectory.
    starts = np.zeros(len(length), dtype=np.int64)
    np.cumsum(length[:-1], out=starts[1:])
    # A point belongs to a trajectory: the one whose start is the largest
    # start that is not above it.
    row = np.arange(n_rows)
    which = np.searchsorted(starts, row, side="right") - 1
    local = row - starts[which]  # the position inside the trajectory

    # The previous and the next point are inside the trajectory only.
    prev_ok = local >= 1
    next_ok = local <= (length[which] - 2)
    has_triple = prev_ok & next_ok

    p_mid = pos
    p_prev = pos[np.clip(row - 1, 0, n_rows - 1)]
    p_next = pos[np.clip(row + 1, 0, n_rows - 1)]
    # The time array holds the frame numbers.
    frames = np.round(time).astype(np.int64)
    f_prev = np.zeros(n_rows, dtype=np.int64)
    f_prev[1:] = frames[:-1]
    f_next = np.zeros(n_rows, dtype=np.int64)
    f_next[:-1] = frames[1:]
    consecutive = (f_prev - f_next) == -2
    sel = has_triple & consecutive

    if not np.any(sel):
        return float("nan"), 0, float("nan"), float("nan"), 0
    dt = 1.0 / fps
    a = (p_next[sel] - 2 * p_mid[sel] + p_prev[sel]) / (dt * dt)
    a = a.ravel()
    m2 = float(np.mean(a**2))
    if m2 <= 0:
        return float("nan"), int(a.size), float("nan"), float("nan"), 0
    m4 = float(np.mean(a**4))
    # The RMS acceleration in mm/s^2, the robust companion of the kurtosis.
    a_rms = float(np.sqrt(m2) * 1000.0)
    # The 99.9th percentile and the number of jumps: a false link is a jump
    # of a whole particle distance in one frame, far above the noise floor.
    a_abs = np.abs(a)
    p999 = float(np.percentile(a_abs, 99.9))
    n_jumps = int((a_abs > 5000.0).sum())
    return m4 / (m2 * m2), int(a.size), a_rms, p999, n_jumps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="LABEL=PATH pairs")
    ap.add_argument(
        "--long-track",
        type=int,
        default=5,
        help="length (frames) that counts as a long track",
    )
    ap.add_argument("--fps", type=float, default=5000.0, help="frame rate")
    args = ap.parse_args(argv)

    rows = []
    for item in args.runs:
        if "=" not in item:
            print(f"Error: {item!r} is not LABEL=PATH")
            return 2
        label, path = item.split("=", 1)
        try:
            store = _resolve(path)
            stats = _stats(store, args.fps)
        except (FileNotFoundError, KeyError) as exc:
            print(f"Error: {label}: {exc}")
            return 2
        stats["label"] = label
        stats["store"] = str(store)
        rows.append(stats)

    head = (
        f"{'Run':<16} {'Tracks':>8} {'Points':>9} {'Mean len':>9} {'Median len':>11} "
        f"{'Max len':>8} {'Long':>7} {'Points/frame':>13} {'>10 frames':>11} "
        f"{'>30 frames':>11} {'K_a':>10} {'a_rms':>10} {'a p99.9':>10} {'Jumps':>9}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        ka = f"{r['kurtosis']:.1f}" if np.isfinite(r["kurtosis"]) else "n/a"
        arms = (
            f"{r['a_rms']:.1f}" if np.isfinite(r.get("a_rms", float("nan"))) else "n/a"
        )
        p999 = (
            f"{r['a_p999']:.0f}"
            if np.isfinite(r.get("a_p999", float("nan")))
            else "n/a"
        )
        print(
            f"{r['label']:<16} {r['tracks']:8d} {r['points']:9d} "
            f"{r['mean_len']:9.2f} {r['median_len']:11.1f} {r['max_len']:8d} "
            f"{r['long_tracks']:7d} {r['points_per_frame']:13.1f} "
            f"{r['frac_over_10']:10.1%} {r['frac_over_30']:10.1%} {ka:>10} "
            f"{arms:>10} {p999:>10} {r['n_jumps']:9d}"
        )

    if len(rows) == 2:
        a, b = rows
        print()
        print(f"{a['label']} -> {b['label']}:")
        for key, name, better in (
            ("points", "points", None),
            ("mean_len", "mean track length", "up"),
            ("frac_over_30", "share of tracks over 30 frames", "up"),
            ("points_per_frame", "points per frame", None),
            ("kurtosis", "acceleration kurtosis K_a", "down"),
            ("a_rms", "acceleration RMS [mm/s2]", "down"),
            ("a_p999", "acceleration p99.9 [mm/s2]", "down"),
            ("n_jumps", "jumps > 5000 mm/s2", "down"),
        ):
            va, vb = a[key], b[key]
            if not np.isfinite(va) or not np.isfinite(vb) or va == 0:
                continue
            change = (vb - va) / va * 100.0
            mark = ""
            if better == "up" and change > 1:
                mark = "  (better)"
            elif better == "up" and change < -1:
                mark = "  (worse)"
            elif better == "down" and change < -1:
                mark = "  (better)"
            elif better == "down" and change > 1:
                mark = "  (worse)"
            print(f"  {name:<30} {va:10.2f} -> {vb:10.2f}  {change:+6.1f} %{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

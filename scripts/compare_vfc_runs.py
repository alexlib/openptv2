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


def _stats(store_path: Path) -> dict:
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
    return {
        "tracks": int(len(length)),
        "points": n_points,
        "mean_len": float(length.mean()) if len(length) else 0.0,
        "median_len": float(np.median(length)) if len(length) else 0.0,
        "max_len": int(length.max()) if len(length) else 0,
        "long_tracks": int((length >= 5).sum()),
        "span": span,
        "points_per_frame": n_points / span if span else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="LABEL=PATH pairs")
    ap.add_argument(
        "--long-track",
        type=int,
        default=5,
        help="length (frames) that counts as a long track",
    )
    args = ap.parse_args(argv)

    rows = []
    for item in args.runs:
        if "=" not in item:
            print(f"Error: {item!r} is not LABEL=PATH")
            return 2
        label, path = item.split("=", 1)
        try:
            store = _resolve(path)
            stats = _stats(store)
        except (FileNotFoundError, KeyError) as exc:
            print(f"Error: {label}: {exc}")
            return 2
        stats["label"] = label
        stats["store"] = str(store)
        rows.append(stats)

    head = (
        f"{'Run':<16} {'Tracks':>8} {'Points':>9} {'Mean len':>9} {'Median len':>11} "
        f"{'Max len':>8} {'Long':>7} {'Points/frame':>13}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['label']:<16} {r['tracks']:8d} {r['points']:9d} "
            f"{r['mean_len']:9.2f} {r['median_len']:11.1f} {r['max_len']:8d} "
            f"{r['long_tracks']:7d} {r['points_per_frame']:13.1f}"
        )

    if len(rows) == 2:
        a, b = rows
        print()
        print(f"{a['label']} -> {b['label']}:")
        for key, name, better in (
            ("points", "points", None),
            ("mean_len", "mean track length", "up"),
            ("long_tracks", "long tracks", "up"),
            ("points_per_frame", "points per frame", None),
        ):
            va, vb = a[key], b[key]
            if va == 0:
                continue
            change = (vb - va) / va * 100.0
            mark = ""
            if better == "up" and change > 1:
                mark = "  (better)"
            elif better == "up" and change < -1:
                mark = "  (worse)"
            print(f"  {name:<20} {va:10.1f} -> {vb:10.1f}  {change:+6.1f} %{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

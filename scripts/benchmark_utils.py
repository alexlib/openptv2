"""Shared helpers to run the synthetic_turbulent benchmark and collect tracks.

Used by:
  * scripts/benchmark_synthetic_turbulent.py  (CLI benchmark)
  * notebooks/tracking_dashboard.py           (interactive marimo dashboard)

Runs each tracker on an isolated copy of test_data/synthetic_turbulent with the
same tracking parameters and returns, per tracker:
  * predicted trajectories  {track_id: [(frame, x, y, z), ...]}
  * proPTV-style identity metrics
and the ground-truth trajectories.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import openptv2.benchmarking as bm

SRC = Path("test_data/synthetic_turbulent")
FIRST = 10001
N_FRAMES = 30
LAST = FIRST + N_FRAMES - 1
TRACKERS = ["fast_3d", "fast_3d_smooth", "myptv_3d_tracking", "proptv_tracking"]

BASE_OVERRIDES = dict(dvxmax=6.0, dvxmin=-6.0, dvymax=6.0, dvymin=-6.0,
                      dvzmax=6.0, dvzmin=-6.0, dacc=6.0)


def read_gt_frames() -> dict[int, list[tuple[int, float, float, float]]]:
    """Reconstruct per-frame ground truth from origin_*.txt (proPTV-style)."""
    frames: dict[int, list] = {}
    for fn in range(FIRST, LAST + 1):
        p = SRC / "res" / f"origin_{fn}.txt"
        if not p.exists():
            continue
        rows = []
        for line in p.read_text().strip().splitlines()[1:]:
            parts = line.split(",")
            pid = int(parts[0])
            rows.append((pid, float(parts[1]), float(parts[2]), float(parts[3])))
        frames[fn] = rows
    return frames


def build_true_tracks(
    frames: dict[int, list[tuple[int, float, float, float]]],
) -> dict[int, list[tuple[int, float, float, float]]]:
    """Ground-truth tracks {pid: [(frame,x,y,z)]} (frames 0-based)."""
    tt: dict[int, list] = {}
    for fn, rows in frames.items():
        for pid, x, y, z in rows:
            if pid < 0:
                continue
            tt.setdefault(pid, []).append((fn - FIRST, x, y, z))
    return {k: sorted(list(v)) for k, v in tt.items()}


def _isolate_run_dir() -> tuple[Path, Path]:
    run_dir = Path(tempfile.mkdtemp())
    for sub in ("cal", "res", "img"):
        shutil.copytree(SRC / sub, run_dir / sub)
    yaml_run = run_dir / "parameters_Run1.yaml"
    shutil.copy(SRC / "parameters_Run1.yaml", yaml_run)
    return run_dir, yaml_run


def run_single_tracker(
    tracker: str,
    track_overrides: dict | None = None,
) -> tuple[dict, float]:
    """Run one tracker, return ({track_id: [(frame,x,y,z)]} 0-based, time_s)."""
    _, yaml_run = _isolate_run_dir()
    t0 = time.perf_counter()
    pred = bm.run_tracker(yaml_run, tracker, track_overrides=track_overrides)
    dt = time.perf_counter() - t0
    pred0 = {k: [(f - FIRST, x, y, z) for (f, x, y, z) in v]
             for k, v in pred.items()}
    return pred0, dt


def run_all_trackers(
    trackers: list[str] | None = None,
    track_overrides: dict | None = None,
    silent: bool = True,
) -> dict[str, dict]:
    """Run all trackers; return {tracker: {...}}.

    Each entry has:
        tracks: {track_id: [(frame,x,y,z)]}
        metrics: IdentityMetrics
        time_s: float
    """
    trackers = trackers or TRACKERS
    overrides = track_overrides or BASE_OVERRIDES
    tt = build_true_tracks(read_gt_frames())
    results: dict[str, dict] = {}
    for tr in trackers:
        t0 = time.perf_counter()
        try:
            pred0, dt = run_single_tracker(tr, track_overrides=overrides)
            m = bm.compute_identity_metrics(tt, pred0, eps=1.0)
            results[tr] = {"tracks": pred0, "metrics": m, "time_s": dt}
        except Exception as e:  # surface error, keep going for other trackers
            results[tr] = {"tracks": {}, "metrics": None, "time_s": 0.0,
                           "error": str(e)}
        if not silent:
            m = results[tr].get("metrics")
            if m:
                print(f"{tr:<22} | pmt {m.pmt:5.1f}% | purity {m.purity:.2f} | {results[tr]['time_s']:.1f}s")
            else:
                print(f"{tr:<22} | ERROR {results[tr].get('error')}")
    return results


def remap_gt_to_tracker_space(tt, gt_ids_to_show=None):
    """Optional: select a subset of ground-truth ids."""
    if gt_ids_to_show is None:
        return tt
    return {k: v for k, v in tt.items() if k in gt_ids_to_show}


__all__ = [
    "SRC", "FIRST", "LAST", "N_FRAMES", "TRACKERS", "BASE_OVERRIDES",
    "read_gt_frames", "build_true_tracks", "run_single_tracker",
    "run_all_trackers", "remap_gt_to_tracker_space",
]

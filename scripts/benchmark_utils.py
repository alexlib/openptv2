"""Shared helpers to run tracker benchmarks and collect tracks.

Used by:
  * scripts/bench_trackers.py         (CLI benchmark; single entry point)
  * notebooks/tracking_dashboard.py   (interactive marimo dashboard)

Defaults to test_data/synthetic_turbulent (220 particles/frame), but every
entry point takes a `src` dataset dir + frame range so the same helpers drive
the density-sweep variants too. Runs each tracker on an isolated copy with
the same tracking parameters and returns, per tracker:
  * predicted trajectories  {track_id: [(frame, x, y, z), ...]}
  * proPTV-style identity metrics + link-level metrics (see combined_metrics)
and the ground-truth trajectories.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import openptv2.benchmarking as bm
from openptv2.tracking_metrics import calculate_tracking_metrics

SRC = Path("test_data/synthetic_turbulent")
FIRST = 10001
N_FRAMES = 30
LAST = FIRST + N_FRAMES - 1
TRACKERS = ["fast_3d", "fast_3d_smooth", "myptv_3d_tracking", "proptv_tracking"]

BASE_OVERRIDES = dict(dvxmax=6.0, dvxmin=-6.0, dvymax=6.0, dvymin=-6.0,
                      dvzmax=6.0, dvzmin=-6.0, dacc=6.0)


def read_gt_frames(
    src: Path = SRC, first: int = FIRST, n_frames: int = N_FRAMES,
) -> dict[int, list[tuple[int, float, float, float]]]:
    """Reconstruct per-frame ground truth from origin_*.txt (proPTV-style)."""
    frames: dict[int, list] = {}
    for fn in range(first, first + n_frames):
        p = src / "res" / f"origin_{fn}.txt"
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
    first: int = FIRST,
) -> dict[int, list[tuple[int, float, float, float]]]:
    """Ground-truth tracks {pid: [(frame,x,y,z)]} (frames 0-based)."""
    tt: dict[int, list] = {}
    for fn, rows in frames.items():
        for pid, x, y, z in rows:
            if pid < 0:
                continue
            tt.setdefault(pid, []).append((fn - first, x, y, z))
    return {k: sorted(list(v)) for k, v in tt.items()}


def build_ghost_frames(
    frames: dict[int, list[tuple[int, float, float, float]]],
    first: int = FIRST,
) -> dict[int, np.ndarray]:
    """Ghost (pid < 0) positions per frame (0-based), for
    ``compute_identity_metrics(..., ghost_pos_by_frame=...)``."""
    out: dict[int, np.ndarray] = {}
    for fn, rows in frames.items():
        ghosts = [(x, y, z) for pid, x, y, z in rows if pid < 0]
        if ghosts:
            out[fn - first] = np.array(ghosts)
    return out


def combined_metrics(
    tt: dict[int, list[tuple[int, float, float, float]]],
    pred0: dict[int, list[tuple[int, float, float, float]]],
    eps: float = 1.0,
    ghosts: dict[int, np.ndarray] | None = None,
) -> dict:
    """One flat row merging the two independent metric systems computed
    from the same run: proPTV-style identity metrics (F/C/purity/pmt/ghost
    capture, position-matched) and link-level metrics (yield/precision/FCR/
    gap-recovery, matched on both endpoints of a link). Field names do not
    collide between the two, so this is a plain dict merge -- no new metric
    is invented here.
    """
    identity = bm.compute_identity_metrics(tt, pred0, eps=eps, ghost_pos_by_frame=ghosts)
    link = calculate_tracking_metrics(tt, pred0, distance_tolerance=eps)
    return {**identity.to_dict(), **link.to_dict()}


def _isolate_run_dir(src: Path = SRC) -> tuple[Path, Path]:
    run_dir = Path(tempfile.mkdtemp())
    for sub in ("cal", "res", "img"):
        shutil.copytree(src / sub, run_dir / sub)
    yaml_run = run_dir / "parameters_Run1.yaml"
    shutil.copy(src / "parameters_Run1.yaml", yaml_run)
    return run_dir, yaml_run


def run_single_tracker(
    tracker: str,
    track_overrides: dict | None = None,
    src: Path = SRC,
    first: int = FIRST,
) -> tuple[dict, float]:
    """Run one tracker, return ({track_id: [(frame,x,y,z)]} 0-based, time_s)."""
    _, yaml_run = _isolate_run_dir(src)
    t0 = time.perf_counter()
    pred = bm.run_tracker(yaml_run, tracker, track_overrides=track_overrides)
    dt = time.perf_counter() - t0
    pred0 = {k: [(f - first, x, y, z) for (f, x, y, z) in v]
             for k, v in pred.items()}
    return pred0, dt


def run_all_trackers(
    trackers: list[str] | None = None,
    track_overrides: dict | None = None,
    silent: bool = True,
    src: Path = SRC,
    first: int = FIRST,
    n_frames: int = N_FRAMES,
) -> dict[str, dict]:
    """Run all trackers on the dataset at ``src``; return {tracker: {...}}.

    Each entry has:
        tracks: {track_id: [(frame,x,y,z)]}
        metrics: IdentityMetrics (F/C/purity/pmt/ghost-capture)
        row: dict merging `metrics` with link-level yield/precision/FCR/
            gap-recovery (see combined_metrics)
        time_s: float
    """
    trackers = trackers or TRACKERS
    overrides = track_overrides or BASE_OVERRIDES
    frames = read_gt_frames(src, first, n_frames)
    tt = build_true_tracks(frames, first)
    ghosts = build_ghost_frames(frames, first)
    results: dict[str, dict] = {}
    for tr in trackers:
        try:
            pred0, dt = run_single_tracker(tr, track_overrides=overrides, src=src, first=first)
            m = bm.compute_identity_metrics(tt, pred0, eps=1.0, ghost_pos_by_frame=ghosts)
            row = {**m.to_dict(),
                   **calculate_tracking_metrics(tt, pred0, distance_tolerance=1.0).to_dict()}
            results[tr] = {"tracks": pred0, "metrics": m, "row": row, "time_s": dt}
        except Exception as e:  # surface error, keep going for other trackers
            results[tr] = {"tracks": {}, "metrics": None, "row": None, "time_s": 0.0,
                           "error": str(e)}
        if not silent:
            m = results[tr].get("metrics")
            if m:
                r = results[tr]["row"]
                print(f"{tr:<22} | pmt {m.pmt:5.1f}% | purity {m.purity:.2f} | "
                      f"yield {r['yield_recall']:.2f} | precision {r['precision']:.2f} | "
                      f"ghost {m.ghost_capture_rate:.2%} | {results[tr]['time_s']:.1f}s")
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
    "read_gt_frames", "build_true_tracks", "build_ghost_frames",
    "combined_metrics", "run_single_tracker",
    "run_all_trackers", "remap_gt_to_tracker_space",
]

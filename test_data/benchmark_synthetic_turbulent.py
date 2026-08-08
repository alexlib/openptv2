"""Benchmark the saved synthetic_turbulent test case across trackers.

Runs each tracker on an isolated copy of test_data/synthetic_turbulent with
the same tracking parameters, then reports proPTV-style identity metrics.
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

import openptv2.benchmarking as bm

SRC = Path("test_data/synthetic_turbulent")
FIRST = 10001
TRACKERS = ["fast_3d", "hybrid_3d_corr", "myptv_3d_tracking", "proptv_tracking"]
OVERRIDES = dict(dvxmax=6.0, dvxmin=-6.0, dvymax=6.0, dvymin=-6.0,
                 dvzmax=6.0, dvzmin=-6.0, dacc=6.0)


def read_gt_frames():
    """Reconstruct per-frame ground-truth from origin_*.txt (proPTV-style)."""
    frames: dict[int, list] = {}
    for fn in range(FIRST, FIRST + 30):
        p = SRC / "res" / f"origin_{fn}.txt"
        if not p.exists():
            continue
        rows = []
        for line in p.read_text().strip().splitlines()[1:]:
            parts = line.split(",")
            pid = int(parts[0])
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            rows.append((pid, xyz[0], xyz[1], xyz[2]))
        frames[fn] = rows
    return frames


def build_true_tracks(frames):
    """Reconstruct ground-truth tracks {pid: [(frame,x,y,z)]} from origin files."""
    tt = {}
    for fn, rows in frames.items():
        for pid, x, y, z in rows:
            if pid < 0:
                continue
            tt.setdefault(pid, []).append((fn - FIRST, x, y, z))
    # some tracks share pid across frames; dedupe
    return {k: sorted(list(v)) for k, v in tt.items()}


def main():
    assert SRC.exists(), f"generate test_data first: run python test_data/create_synthetic_turbulent.py"

    frames = read_gt_frames()
    tt = build_true_tracks(frames)

    header = (f"{'tracker':<22} | {'#tr':>5} | {'F':>5} | {'C':>5} | "
              f"{'Cr':>5} | {'pmt':>6} | {'time_s':>7}")
    print(f"test_case: synthetic_turbulent (30 frames, {len(tt)} true trajs)")
    print(header)
    print("-" * len(header))

    for tr in TRACKERS:
        run_dir = Path(tempfile.mkdtemp())
        for sub in ("cal", "res", "img"):
            shutil.copytree(SRC / sub, run_dir / sub)
        shutil.copy(SRC / "parameters_Run1.yaml", run_dir / "parameters_Run1.yaml")
        yaml_run = run_dir / "parameters_Run1.yaml"
        try:
            t0 = time.perf_counter()
            pred = bm.run_tracker(yaml_run, tr, track_overrides=OVERRIDES)
            dt = time.perf_counter() - t0
            pred0 = {k: [(f - FIRST, x, y, z) for (f, x, y, z) in v]
                     for k, v in pred.items()}
            m = bm.compute_identity_metrics(tt, pred0, eps=1.0)
            print(f"{tr:<22} | {m.n_reconstructed:>5} | {m.fragmentation:>5.2f} | "
                  f"{m.completeness:>5.2f} | {m.purity:>5.2f} | {m.pmt:>5.1f}% | {dt:>6.2f}")
        except Exception as e:
            print(f"{tr:<22} | ERROR: {e}")


if __name__ == "__main__":
    main()

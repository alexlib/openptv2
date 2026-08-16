"""Which tracker gives the most CORRECT acceleration, in the Lagrangian sense?

Not "how many links" -- how faithful are the velocity and acceleration
statistics the tracker's trajectories imply, against the ground truth's own.

A wrong link injects a spurious jump, and the acceleration PDF is dominated
by its tails, so this is where a small wrong-link rate should hurt far more
than a large missing-link rate. That is the user's hypothesis: correctness
cannot be recovered by post-processing, length can.

Ground truth here is noise-free and Gaussian (K_a = 3.0), so any excess
kurtosis in a tracker's acceleration PDF is CONTAMINATION, not physics --
which makes this a clean contamination assay even though it cannot test
intermittency preservation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import benchmark_utils as bu  # noqa: E402

import openptv2.algorithms.track4be as t4be  # noqa: E402
from openptv2.benchmarking.runner import run_tracker  # noqa: E402

DATASETS = [
    ("220 p/frame", Path("test_data/synthetic_turbulent"), 10001, 30),
    ("970 p/frame", Path("test_data/synthetic_turbulent_1k"), 10001, 30),
]
RUNS = [
    ("3MA dacc=6", "priority_segment_3d", 6.0, 0, False),
    ("3MA dacc=6 +bridge", "priority_segment_3d", 6.0, 0, True),
    ("3MA dacc=3.6 +bridge", "priority_segment_3d", 3.6, 0, True),
    ("4BE paper", "4be", 6.0, 0, False),
    ("4BE greedy", "4be", 6.0, 1, False),
    ("4BE paper +bridge", "4be", 6.0, 0, True),
]


def kinematics(tracks):
    """Component-wise velocity and acceleration over contiguous runs."""
    vs, as_ = [], []
    for pts in tracks.values():
        pts = sorted(pts)
        f = np.array([p[0] for p in pts])
        x = np.array([[p[1], p[2], p[3]] for p in pts])
        for seg in np.split(x, np.where(np.diff(f) != 1)[0] + 1):
            if len(seg) >= 2:
                vs.append(np.diff(seg, axis=0))
            if len(seg) >= 3:
                as_.append(np.diff(seg, 2, axis=0))
    v = np.concatenate(vs).ravel() if vs else np.zeros(0)
    a = np.concatenate(as_).ravel() if as_ else np.zeros(0)
    return v, a


def stats(z):
    if z.size < 4:
        return float("nan"), float("nan")
    zc = z - z.mean()
    return float(z.std()), float(np.mean(zc**4) / (np.mean(zc**2) ** 2))


for label, src, first, n in DATASETS:
    tt = bu.build_true_tracks(bu.read_gt_frames(src, first, n), first)
    v_t, a_t = kinematics(tt)
    v_rms_t, v_k_t = stats(v_t)
    a_rms_t, a_k_t = stats(a_t)
    print(f"\n=== {label} ===")
    print(f"{'TRUTH':<22} {'v_rms':>7} {'a_rms':>7} {'K_a':>7} "
          f"{'a_rms err':>10} {'|a|>5sig':>9} {'prec':>7} {'yield':>7}")
    print(f"{'ground truth':<22} {v_rms_t:7.3f} {a_rms_t:7.3f} {a_k_t:7.2f} "
          f"{'--':>10} {100 * np.mean(np.abs(a_t - a_t.mean()) > 5 * a_rms_t):8.3f}% "
          f"{'--':>7} {'--':>7}")

    for name, tracker, dacc, greedy, pp in RUNS:
        t4be.GREEDY_CONFLICTS = greedy
        _, yaml_run = bu._isolate_run_dir(src)
        ov = {**bu.BASE_OVERRIDES, "dacc": dacc, "postprocess": pp}
        try:
            pred = run_tracker(yaml_run, tracker, track_overrides=ov)
        except Exception as e:
            print(f"{name:<22} ERROR {e}")
            continue
        pred0 = {k: [(f - first, x, y, z) for (f, x, y, z) in v]
                 for k, v in pred.items()}
        m = bu.combined_metrics(tt, pred0)
        v_p, a_p = kinematics(pred0)
        v_rms, _ = stats(v_p)
        a_rms, a_k = stats(a_p)
        # fraction of predicted accelerations beyond 5 sigma of the TRUE
        # distribution -- these are essentially all wrong-link artefacts
        outl = 100 * np.mean(np.abs(a_p - a_t.mean()) > 5 * a_rms_t) if a_p.size else float("nan")
        print(f"{name:<22} {v_rms:7.3f} {a_rms:7.3f} {a_k:7.2f} "
              f"{100 * (a_rms / a_rms_t - 1):+9.1f}% {outl:8.3f}% "
              f"{m['precision']:7.4f} {m['yield_recall']:7.4f}", flush=True)
    t4be.GREEDY_CONFLICTS = 0

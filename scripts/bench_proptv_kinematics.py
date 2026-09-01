"""Rank trackers by kinematic accuracy on the proPTV 500_30 case (real
turbulence intermittency -- see docs/plans/
2026-08-17-lagrangian-accuracy-program.md, Phase 2's proPTV note).

adapt_proptv_dataset.py (2026-08-18 rewrite) rescales proPTV's [0,1]-cube
ground truth into a mm-scale working volume (+-20mm, same order as
openptv2's other synthetic sets) via its own self-consistent pinhole rig --
so this dataset is mm-scale like every other one here, no special eps.

Each tracker runs at its OWN auto-recommended parameters
(benchmark_utils.per_tracker_overrides, dataset-scaled via
tracking_recommender). Link precision/yield can still saturate at 1.0 for a
correct tracker at this (currently noise-free) density -- the real
differentiator, and the whole point of this script, is acceleration
fidelity against ground truth: a false/mismatched link is not just noise, it
injects the WRONG kinematics into the recovered statistics (e.g. a
smoothness-favouring cost function will systematically suppress K_a below
the true value, not just add scatter around it) -- so K_a and a_rms error
against truth are the metrics that actually matter here, not yield/precision
alone.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import benchmark_utils as bu  # noqa: E402

SRC = Path("test_data/proptv_500_30")
FIRST, N = 10001, 30

# The 5-engine survivor set (see docs/plans/2026-08-17-lagrangian-accuracy-
# program.md): 3MA, 4BE, trackcorr, MyPTV, proPTV. nearest_hungarian_3d/
# predictive_gmm_3d are literal aliases of myptv_3d_tracking/proptv_tracking
# (see plugins/loader.py) -- not separate engines, not listed here.
TRACKERS = [
    "priority_segment_3d",
    "trackcorr",
    "4be",
    "myptv_3d_tracking",
    "proptv_tracking",
]


def kinematics(tracks):
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


def main():
    frames = bu.read_gt_frames(SRC, FIRST, N)
    tt = bu.build_true_tracks(frames, FIRST)
    _v_t, a_t = kinematics(tt)
    a_rms_t, a_k_t = stats(a_t)
    print(f"truth: a_rms {a_rms_t:.5f}  K_a {a_k_t:.2f}\n")

    overrides = bu.per_tracker_overrides(TRACKERS, src=SRC, first=FIRST, n_frames=N)

    print(
        f"{'tracker':<22} {'a_err':>8} {'K_a':>8} {'>5sig':>8} "
        f"{'meanlen':>8} {'prec':>7} {'yield':>7} {'time_s':>7}"
    )
    for tr in TRACKERS:
        ov = overrides[tr]
        try:
            pred0, dt = bu.run_single_tracker(
                tr, track_overrides=ov, src=SRC, first=FIRST
            )
        except Exception as e:
            print(f"{tr:<22} ERROR {e}")
            continue
        m = bu.combined_metrics(tt, pred0, eps=1.0)
        v_p, a_p = kinematics(pred0)
        a_rms, a_k = stats(a_p)
        lens = np.array([len(v) for v in pred0.values()]) if pred0 else np.zeros(1)
        outl = (
            100 * np.mean(np.abs(a_p - a_t.mean()) > 5 * a_rms_t)
            if a_p.size
            else float("nan")
        )
        print(
            f"{tr:<22} {100 * (a_rms / a_rms_t - 1):+7.1f}% {a_k:8.2f} {outl:7.3f}% "
            f"{lens.mean():8.2f} {m['precision']:7.4f} {m['yield_recall']:7.4f} "
            f"{dt:7.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

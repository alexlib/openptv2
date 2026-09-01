"""Univariate noise-source sweep on the proPTV realistic pipeline.

2026-08-18's realistic pipeline (adapt_proptv_dataset.convert_realistic)
showed severe trajectory fragmentation (meanlen 2.5-5.7 vs 30) at
"moderate" severity, but isolating each noise mechanism alone (dropout,
merge, calibration residual) gave 97.7-99.2% per-frame correspondence
match rates -- none alone explains it. So the fragmentation is not raw
correspondence-stage data loss; it's emergent from how the TRACKERS'
own gating (dacc/angle thresholds, conflict resolution) responds to
modest, compounding noise across a 30-frame sequence.

This sweep isolates which mechanism actually drives TRACKING-level damage
(not just correspondence-stage match rate): baseline = "mild" severity,
then one knob at a time bumped to a "high" value, run through the real
5-tracker + kinematics benchmark (bench_proptv_kinematics.py's own
functions, not a reimplementation).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import adapt_proptv_dataset as apd  # noqa: E402
import benchmark_utils as bu  # noqa: E402
from bench_proptv_kinematics import kinematics, stats  # noqa: E402

SRC = Path("test_data/proptv_500_30")
PROPTV_ROOT = Path(r"C:/Users/alex/Github/proPTV/data")
FIRST, N = 10001, 30
TRACKERS = [
    "priority_segment_3d",
    "trackcorr",
    "4be",
    "myptv_3d_tracking",
    "proptv_tracking",
]

BASELINE = dict(apd.SEVERITY_PRESETS["mild"])

SWEEP = {
    "noise_px": 0.3,
    "dropout_p": 0.06,
    "merge_radius_px": 4.0,
    # angle/pos/cc scaled together -- "calibration residual" as one axis.
    "calib_severity": dict(
        calib_angle_sigma_deg=0.04, calib_pos_sigma_mm=0.04, calib_cc_ppm=400.0
    ),
}


def run_one(label: str, params: dict) -> None:
    apd.convert_realistic(
        PROPTV_ROOT / "500_30",
        Path("test_data/synthetic_turbulent"),
        SRC,
        seed=0,
        **params,
    )

    frames = bu.read_gt_frames(SRC, FIRST, N)
    tt = bu.build_true_tracks(frames, FIRST)
    _v_t, a_t = kinematics(tt)
    a_rms_t, a_k_t = stats(a_t)

    overrides = bu.per_tracker_overrides(TRACKERS, src=SRC, first=FIRST, n_frames=N)
    print(f"\n=== {label} === (truth: a_rms {a_rms_t:.5f}  K_a {a_k_t:.2f})")
    print(
        f"{'tracker':<22} {'a_err':>8} {'K_a':>8} {'>5sig':>8} {'meanlen':>8} {'yield':>7}"
    )
    for tr in TRACKERS:
        try:
            pred0, _dt = bu.run_single_tracker(
                tr, track_overrides=overrides[tr], src=SRC, first=FIRST
            )
        except Exception as e:  # noqa: BLE001
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
            f"{lens.mean():8.2f} {m['yield_recall']:7.4f}",
            flush=True,
        )


def main():
    run_one("baseline (mild)", BASELINE)
    for knob, high_value in SWEEP.items():
        params = dict(BASELINE)
        if knob == "calib_severity":
            params.update(high_value)
        else:
            params[knob] = high_value
        run_one(f"{knob} -> high", params)


if __name__ == "__main__":
    main()

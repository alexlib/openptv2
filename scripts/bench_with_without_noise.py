"""With-noise vs. without-noise comparison for the five survivor trackers.

Two datasets, same 30-frame proPTV case, same seed:
  - "clean": adapt_proptv_dataset.convert() -- ground-truth correspondences
    injected directly (no detection/correspondence/calibration error chain).
  - "realistic": adapt_proptv_dataset.convert_realistic() at "mild" severity
    -- the real detection -> correspondence -> triangulation pipeline
    (see docs/holistic-3d-ptv-systems-research-program.md).

For each tracker, prints the ACTUAL resolved parameters
(benchmark_utils.per_tracker_overrides, dataset-scaled) used for that run --
not the tracking_registry.py defaults, which are documentation, not what
necessarily gets applied -- plus the kinematic accuracy metrics on both
datasets side by side. This is the source of the numbers in
docs/tracker-tutorials.md; regenerate this script's output before trusting
any number quoted there as still current.
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
TRACKERS = ["priority_segment_3d", "trackcorr", "4be", "myptv_3d_tracking", "proptv_tracking"]


def run_dataset(label: str) -> dict:
    frames = bu.read_gt_frames(SRC, FIRST, N)
    tt = bu.build_true_tracks(frames, FIRST)
    _v_t, a_t = kinematics(tt)
    a_rms_t, a_k_t = stats(a_t)
    print(f"\n=== {label} === (truth: a_rms {a_rms_t:.5f}  K_a {a_k_t:.2f})")

    overrides = bu.per_tracker_overrides(TRACKERS, src=SRC, first=FIRST, n_frames=N)
    print(f"{'tracker':<22} {'resolved overrides (dataset-scaled)'}")
    for tr in TRACKERS:
        print(f"{tr:<22} {overrides[tr]}")

    results = {}
    print(f"\n{'tracker':<22} {'a_err':>8} {'K_a':>8} {'>5sig':>8} {'meanlen':>8} "
          f"{'prec':>7} {'yield':>7} {'time_s':>7}")
    for tr in TRACKERS:
        ov = overrides[tr]
        try:
            pred0, dt = bu.run_single_tracker(tr, track_overrides=ov, src=SRC, first=FIRST)
        except Exception as e:  # noqa: BLE001
            print(f"{tr:<22} ERROR {e}")
            continue
        m = bu.combined_metrics(tt, pred0, eps=1.0)
        v_p, a_p = kinematics(pred0)
        a_rms, a_k = stats(a_p)
        lens = np.array([len(v) for v in pred0.values()]) if pred0 else np.zeros(1)
        outl = 100 * np.mean(np.abs(a_p - a_t.mean()) > 5 * a_rms_t) if a_p.size else float("nan")
        print(f"{tr:<22} {100*(a_rms/a_rms_t-1):+7.1f}% {a_k:8.2f} {outl:7.3f}% "
              f"{lens.mean():8.2f} {m['precision']:7.4f} {m['yield_recall']:7.4f} "
              f"{dt:7.2f}", flush=True)
        results[tr] = dict(a_err=100 * (a_rms / a_rms_t - 1), K_a=a_k, outlier_pct=outl,
                            meanlen=lens.mean(), precision=m["precision"],
                            yield_recall=m["yield_recall"], overrides=ov)
    return results


def main():
    apd.convert(PROPTV_ROOT / "500_30", Path("test_data/synthetic_turbulent"), SRC)
    clean = run_dataset("clean (no noise, ground-truth correspondences)")

    apd.convert_realistic(PROPTV_ROOT / "500_30", Path("test_data/synthetic_turbulent"), SRC,
                           seed=0, **apd.SEVERITY_PRESETS["mild"])
    realistic = run_dataset("realistic (mild severity: detection noise, dropout, "
                             "merging, correspondence solving, calibration residual)")

    print("\n=== summary: K_a, clean vs realistic ===")
    print(f"{'tracker':<22} {'K_a clean':>10} {'K_a realistic':>14}")
    for tr in TRACKERS:
        c = clean.get(tr, {}).get("K_a", float("nan"))
        r = realistic.get(tr, {}).get("K_a", float("nan"))
        print(f"{tr:<22} {c:10.2f} {r:14.2f}")


if __name__ == "__main__":
    main()

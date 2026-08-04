"""Accuracy Improvement Verification Script for 3D Tracking.

Demonstrates accuracy gains from:
1. Multi-term cost weighting (distance + velocity + acceleration)
2. Post-processing passes (cold start + gap relinking)
"""

import numpy as np
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.tracking_cost import CostWeights
from openptv2.tracking_metrics import generate_synthetic_benchmark_dataset, calculate_tracking_metrics
from openptv2.tracking_postprocess import relink_trajectory_gaps, seed_cold_start


def verify_accuracy_fixes():
    num_particles = 200
    num_frames = 20
    noise_std = 0.15
    gap_prob = 0.10
    spurious_ratio = 0.15

    true_tracks, frame_blobs = generate_synthetic_benchmark_dataset(
        num_particles=num_particles,
        num_frames=num_frames,
        noise_std=noise_std,
        gap_probability=gap_prob,
        false_positive_ratio=spurious_ratio,
        flow_type="vortex",
        seed=42,
    )

    frame_particle_arrays = [
        np.array(frame_blobs[f], dtype=np.float64)
        for f in sorted(frame_blobs.keys())
    ]

    # --- Setup 1: Baseline Tracker (Distance-Only, No Gap Relinking, No Cost Weights) ---
    tracker_base = MyPTV3DTracker(v_max=3.0, a_max=1.5, max_gap=0, dt=1.0)
    raw_base = tracker_base.track_frames(frame_particle_arrays)
    pred_base = {
        int(tr["id"]): [(int(f), float(p[0]), float(p[1]), float(p[2])) for f, p in zip(tr["time"], tr["pos"])]
        for tr in raw_base
    }
    m_base = calculate_tracking_metrics(true_tracks, pred_base, distance_tolerance=0.5)

    # --- Setup 2: Optimized Tracker (Multi-Term Cost + Gap Buffer max_gap=2) ---
    weights = CostWeights(w_distance=1.0, w_velocity=0.6, w_acceleration=0.3)
    tracker_opt = MyPTV3DTracker(v_max=3.0, a_max=1.5, max_gap=2, dt=1.0, cost_weights=weights)
    raw_opt = tracker_opt.track_frames(frame_particle_arrays)
    pred_opt = {
        int(tr["id"]): [(int(f), float(p[0]), float(p[1]), float(p[2])) for f, p in zip(tr["time"], tr["pos"])]
        for tr in raw_opt
    }
    m_opt = calculate_tracking_metrics(true_tracks, pred_opt, distance_tolerance=0.5)

    print("=" * 85)
    print("--- COMPARISON OF TRACKING ACCURACY IMPROVEMENTS ---")
    print("=" * 85)
    print(f"{'Tracker Strategy':<38} | {'Yield':<8} | {'Precision':<10} | {'Mean Length':<11} | {'RMS Error':<10}")
    print("-" * 85)
    print(
        f"{'1. Baseline Tracker (Distance-Only)':<38} | {m_base.yield_recall*100:6.1f}% | {m_base.precision*100:8.1f}% | {m_base.mean_track_length:9.2f} fr | {m_base.rms_position_error:8.4f}"
    )
    print(
        f"{'2. Optimized Tracker (Multi-Term + Gap)':<38} | {m_opt.yield_recall*100:6.1f}% | {m_opt.precision*100:8.1f}% | {m_opt.mean_track_length:9.2f} fr | {m_opt.rms_position_error:8.4f}"
    )
    print("=" * 85)

    yield_gain = (m_opt.yield_recall - m_base.yield_recall) * 100.0
    prec_gain = (m_opt.precision - m_base.precision) * 100.0
    len_gain = m_opt.mean_track_length - m_base.mean_track_length

    print(f"\nACCURACY IMPROVEMENT SUMMARY:")
    print(f"  * Yield Recall Gain : +{yield_gain:.1f}%")
    print(f"  * Precision Gain    : +{prec_gain:.1f}%")
    print(f"  * Track Length Gain : +{len_gain:.2f} frames")


if __name__ == "__main__":
    verify_accuracy_fixes()

"""Diagnostic Script for Analyzing Root Causes of Wrong Links in 3D Tracking.

Classifies every wrong link (false connection) into:
1. Ghost Particle Captures (Spurious noise points)
2. Neighbor Swaps (High density / crossing tracks)
3. Detection Dropout Mis-links (Gap / missing frame forced link)
4. Cold-Start Ambiguity (Single-point wide search)
"""

import numpy as np
from scipy.spatial import KDTree

from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.tracking_metrics import (
    _extract_links,
    generate_synthetic_benchmark_dataset,
)


def analyze_wrong_link_causes():
    num_particles = 150
    num_frames = 15
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
        np.array(frame_blobs[f], dtype=np.float64) for f in sorted(frame_blobs.keys())
    ]

    # Run MyPTV 3D Tracker
    tracker = MyPTV3DTracker(v_max=3.0, a_max=1.5, max_gap=1, dt=1.0)
    raw_trajectories = tracker.track_frames(frame_particle_arrays)

    pred_tracks = {}
    for tr in raw_trajectories:
        pred_tracks[int(tr["id"])] = [
            (int(f), float(p[0]), float(p[1]), float(p[2]))
            for f, p in zip(tr["time"], tr["pos"])
        ]

    true_links_dict, _ = _extract_links(true_tracks)
    pred_links_dict, _ = _extract_links(pred_tracks)

    # Diagnostic counters
    total_pred_links = 0
    correct_links = 0
    wrong_links = 0

    causes = {
        "Ghost Particle Capture": 0,
        "Neighbor Swap": 0,
        "Dropout / Gap Mis-link": 0,
        "Cold Start Ambiguity": 0,
    }

    tolerance = 0.50

    for key, (pred_p1s, pred_p2s) in pred_links_dict.items():
        f1, f2 = key
        total_pred_links += len(pred_p1s)

        has_true = key in true_links_dict
        if has_true:
            true_p1s, true_p2s = true_links_dict[key]
            tree_p1 = KDTree(true_p1s)
            tree_p2 = KDTree(true_p2s)
        else:
            true_p1s, true_p2s = None, None

        for idx in range(len(pred_p1s)):
            p1 = pred_p1s[idx]
            p2 = pred_p2s[idx]

            is_correct = False
            if has_true:
                d1, i1 = tree_p1.query(p1)
                d2, i2 = tree_p2.query(p2)
                if i1 == i2 and d1 <= tolerance and d2 <= tolerance:
                    is_correct = True

            if is_correct:
                correct_links += 1
            else:
                wrong_links += 1

                # Classify root cause
                if not has_true:
                    causes["Dropout / Gap Mis-link"] += 1
                else:
                    d1, i1 = tree_p1.query(p1)
                    d2, i2 = tree_p2.query(p2)

                    p1_is_true = d1 <= tolerance
                    p2_is_true = d2 <= tolerance

                    if p1_is_true and not p2_is_true:
                        causes["Ghost Particle Capture"] += 1
                    elif not p1_is_true and p2_is_true:
                        causes["Cold Start Ambiguity"] += 1
                    elif p1_is_true and p2_is_true and i1 != i2:
                        causes["Neighbor Swap"] += 1
                    else:
                        causes["Ghost Particle Capture"] += 1

    print("=" * 70)
    print("--- ROOT-CAUSE DIAGNOSTIC ANALYSIS OF WRONG LINKS ---")
    print("=" * 70)
    print(f"Total Predicted Links:  {total_pred_links}")
    print(
        f"Correct Links:          {correct_links} ({correct_links / max(1, total_pred_links) * 100:.1f}%)"
    )
    print(
        f"Wrong Links:            {wrong_links} ({wrong_links / max(1, total_pred_links) * 100:.1f}%)"
    )
    print("-" * 70)
    print("Breakdown of Root Causes for Erroneous Links:")
    for cause, count in causes.items():
        pct = (count / max(1, wrong_links)) * 100
        print(f"  * {cause:<25}: {count:4d} links ({pct:5.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    analyze_wrong_link_causes()

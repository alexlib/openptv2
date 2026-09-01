"""Benchmark every tracker exposed in the GUI's tracker-selection dropdown
(``tracking_presets.TRACKER_CHOICES``), on the same dataset, reporting time,
quality (precision/yield/purity/ghost-capture/pmt), and correctness against
ground truth.

For a tracker with a forward-only vs forward+backward option, run both and
report two lines (see tracking_presets.TRACKER_SUPPORTS_BACKWARD /
tracking_presets._DIRECTION_BACKWARD_PRESETS): "trackcorr"/"cython_epipolar_
tracking" default to forward-only; "full_multipass" is the SAME engine
(cython_epipolar_tracking plugin) with forward+backward forced via the
legacy preset-name direction lookup -- no direction override plumbing
needed, it's already wired.

Dataset: test_data/synthetic_turbulent (the project's own existing 220
particles/frame, 30-frame synthetic scene with ground truth -- reused as-is
via scripts/benchmark_utils.py's default SRC, not regenerated).

Usage:
    uv run python scripts/benchmark_all_gui_trackers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu  # noqa: E402

# (label shown in the GUI dropdown, tracker key run_tracker understands, direction note)
ROWS = [
    ("OpenPTV Fast 3D (Default - Cython)", "priority_segment_3d", "n/a"),
    (
        "OpenPTV Epipolar (Multi-Camera Cython)",
        "cython_epipolar_tracking",
        "forward only",
    ),
    ("OpenPTV Epipolar (Multi-Camera 2D+3D)", "trackcorr", "forward only"),
    ("OpenPTV Epipolar (Multi-Camera 2D+3D)", "full_multipass", "forward + backward"),
    ("MyPTV 3D (Nearest-Neighbor Hungarian)", "nearest_hungarian_3d", "n/a"),
    ("MyPTV 2D (Image-Space Assignment)", "myptv_2d_tracking", "n/a"),
    (
        "proPTV (Predictive GMM - Optional)",
        "predictive_gmm_3d",
        "n/a (no direction toggle found)",
    ),
]


def main():
    trackers = [key for _label, key, _note in ROWS]
    results = bu.run_all_trackers(trackers=trackers, silent=True)

    print(f"\nDataset: {bu.SRC}  frames {bu.FIRST}..{bu.FIRST + bu.N_FRAMES - 1}\n")
    header = (
        f"{'GUI label':<42} {'tracker key':<24} {'direction':<20} "
        f"{'time':>8} {'prec':>6} {'yield':>6} {'ghost':>6} {'pmt':>7} {'purity':>7}"
    )
    print(header)
    print("-" * len(header))
    for label, key, note in ROWS:
        r = results[key]
        if r.get("row") is None:
            print(f"{label:<42} {key:<24} {note:<20} ERROR: {r.get('error')}")
            continue
        row = r["row"]
        m = r["metrics"]
        print(
            f"{label:<42} {key:<24} {note:<20} "
            f"{r['time_s']:7.2f}s {row['precision']:6.3f} {row['yield_recall']:6.3f} "
            f"{m.ghost_capture_rate:6.3f} {m.pmt:6.1f}% {m.purity:7.3f}"
        )


if __name__ == "__main__":
    main()

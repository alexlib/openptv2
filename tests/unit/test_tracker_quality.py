"""Ground-truth quality regression floor for the default tracker.

Fixture: ``test_data/synthetic_turbulent_1k`` (~1000 particles/frame, 30
frames; see scripts/create_synthetic_turbulent.py --density 1000 and
docs/plans/master-plan.md Stage 0). Unlike
tests/unit/test_track3d.py, which pins a raw *link count* on test_cavity,
this pins link-level *correctness* against exact ground truth (see
src/openptv2/tracking_metrics.py's yield/precision, which requires both
endpoints of a predicted link to match the same true link).

Floors are set from the measured baseline (fast_3d, default BASE_OVERRIDES),
with margin for incidental rebuild/tie-break drift -- not aspirational
targets. Raised once already after Stage 1b (global cost-ordered claiming
within a level instead of particle-by-particle in index order) measurably
improved precision 0.718 -> 0.871 and recall 0.648 -> 0.812 at this density.
A future quality improvement is expected to raise these floors further; a
drop below them without an intentional algorithm change is a regression.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_utils as bu  # noqa: E402
from create_synthetic_turbulent import make_dataset  # noqa: E402

DATASET = Path(__file__).resolve().parents[2] / "test_data" / "synthetic_turbulent_1k"


@pytest.mark.slow
def test_fast_3d_quality_floor_at_1k_density():
    # res/img are gitignored repo-wide (test_data/**/res/, **/*_targets) --
    # only cal/ and parameters_Run1.yaml are checked in. Regenerate the rest
    # deterministically (fixed seed) on a fresh checkout.
    if not (DATASET / "res").exists():
        make_dataset(DATASET, num_particles=1000, num_frames=bu.N_FRAMES, seed=2026)

    results = bu.run_all_trackers(
        ["fast_3d"], silent=True, src=DATASET, first=bu.FIRST, n_frames=bu.N_FRAMES,
    )
    row = results["fast_3d"]["row"]
    assert row is not None, results["fast_3d"].get("error")

    # Measured baseline (post Stage 1b): precision 0.871, yield_recall
    # 0.812, ghost_capture 0.038 (see docstring). Floors keep ~8-9% margin
    # below the measurement.
    assert row["precision"] >= 0.80, row
    assert row["yield_recall"] >= 0.75, row
    assert row["ghost_capture_rate"] <= 0.10, row


def test_trajectory_shape_stats_length_and_smoothness():
    """No fixtures needed: pins the shape metrics used by
    scripts/compare_trackers_vs_liboptv.py's trajectory-shape table
    (length/gaps/smoothness comparison across trackers)."""
    tracks = {
        # Straight, constant-velocity: 0 deg smoothness.
        0: [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0), (2, 2.0, 0.0, 0.0), (3, 3.0, 0.0, 0.0)],
        # A 90-degree turn at frame 2.
        1: [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0), (2, 1.0, 1.0, 0.0)],
        # Short fragment (< 5 frames).
        2: [(0, 5.0, 5.0, 0.0), (1, 5.1, 5.0, 0.0)],
    }
    stats = bu.trajectory_shape_stats(tracks)

    assert stats["n_tracks"] == 3
    assert stats["max_length"] == 4
    assert stats["min_length"] == 2
    assert stats["frac_short_lived"] == 1.0  # all three are < 5 frames
    # Track 0 (4 points, 3 velocity vectors) contributes two 0-deg samples
    # (collinear steps); track 1 (3 points) contributes one 90-deg sample
    # (the direction break); track 2 is too short (< 3 points) to contribute.
    assert stats["n_smoothness_samples"] == 3
    assert 25.0 < stats["mean_smoothness_deg"] < 35.0  # (0 + 0 + 90) / 3 = 30


def test_trajectory_shape_stats_empty():
    stats = bu.trajectory_shape_stats({})
    assert stats["n_tracks"] == 0
    assert stats["mean_smoothness_deg"] != stats["mean_smoothness_deg"]  # NaN

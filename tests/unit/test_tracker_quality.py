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
from tune_tracker_params import _run_via_subprocess  # noqa: E402

DATASET = Path(__file__).resolve().parents[2] / "test_data" / "synthetic_turbulent_1k"

# 20 frames, not benchmark_utils.N_FRAMES's default 30: this floor only needs
# enough frames to exercise both scripted crossings (frame 15, 18) and settle
# into steady-state linking; the extra 10 frames added ~1/3 of this file's
# runtime for no extra regression signal. Local override, not a change to the
# shared benchmark_utils.N_FRAMES default other scripts rely on.
N_FRAMES = 20


@pytest.mark.slow
def test_fast_3d_quality_floor_at_1k_density():
    # res/img are gitignored repo-wide (test_data/**/res/, **/*_targets) --
    # only cal/ and parameters_Run1.yaml are checked in. Regenerate the rest
    # deterministically (fixed seed) on a fresh checkout.
    if not (DATASET / "res").exists():
        make_dataset(DATASET, num_particles=1000, num_frames=N_FRAMES, seed=2026)

    results = bu.run_all_trackers(
        ["fast_3d"], silent=True, src=DATASET, first=bu.FIRST, n_frames=N_FRAMES,
    )
    row = results["fast_3d"]["row"]
    assert row is not None, results["fast_3d"].get("error")

    # Measured baseline (post Stage 1b): precision 0.871, yield_recall
    # 0.812, ghost_capture 0.038 (see docstring). Floors keep ~8-9% margin
    # below the measurement.
    assert row["precision"] >= 0.80, row
    assert row["yield_recall"] >= 0.75, row
    assert row["ghost_capture_rate"] <= 0.10, row


# ---------------------------------------------------------------------------
# Stage 0b (docs/plans/2026-08-15-tracking-quality-overhaul.md): ratchet
# quality floors for every tracker registered in tracking_registry.py, not
# just fast_3d. Each tracker gets its own kinematic-bound overrides via
# scripts/benchmark_utils.per_tracker_overrides (one shared parameter set
# hides real per-engine differences -- see that function's docstring).
#
# These are ratchet floors, not aspirational targets: measured baseline
# minus a small margin (2 percentage points on precision/yield_recall, plus
# 2pp of headroom on ghost_capture_rate). A drop below them without an
# intentional algorithm change is a regression; raise them when a later
# stage measurably improves a tracker.
#
# Building this floor surfaced (and fixed) a real bug: trackcorr/
# full_multipass initially measured near-zero yield_recall (~0.0003) here.
# Root cause was in the dataset generator, not the tracker or this harness:
# openptv2.benchmarking.datawriter.write_dataset wrote each 2D target's
# ``tnr`` field as the particle's ground-truth pid. trackcorr's Cython
# kernels use ``tnr`` as a direct row index into that frame's 3D-particle
# array (track_kernels_search.py: ``ftnr_out[...] = targ_tnr[cam, idx]``,
# then track_kernels_corr.py: ``path_x_2[ftnr_i]``) -- it must be the
# particle's *slot* in that frame's rt_is list, not its pid. The two only
# coincide while every frame holds a dense 0..n-1 pid range; this dataset's
# entering/leaving particles break that immediately, so nearly every
# candidate ftnr pointed at the wrong (or an out-of-range) 3D position and
# got rejected by the angle/acc test. track3d-based trackers (everything
# else in this table) never touch target tnr, so they were unaffected.
# Fixed in datawriter.py (tnr = slot, ghosts = TR_UNUSED); see that file's
# comment. Baseline below is measured post-fix.
#
# Baseline measured on synthetic_turbulent_1k (30 frames, ~1000
# particles/frame), 2026-08-15, after the Stage 0a trackback_loop_fast fix
# and the datawriter.py tnr fix:
#   priority_segment_3d: precision 0.904, yield_recall 0.864, ghost 0.038
#   trackcorr:           precision 0.930, yield_recall 0.759, ghost 0.038
#   kalman_hungarian_3d: precision 0.856, yield_recall 0.763, ghost 0.038
#   sg_hungarian_3d:     precision 0.650, yield_recall 0.577, ghost 0.038
#   nearest_hungarian_3d:precision 0.634, yield_recall 0.603, ghost 0.038
#   predictive_gmm_3d:   precision 0.690, yield_recall 0.693, ghost 0.034
#
# trackcorr precision floor lowered 0.91 -> 0.86 (2026-08-21): commit
# 7ceff6a (after this Stage 0b baseline was measured) fixed a real
# ray-tracing Snell's-law sign bug and generalized candidate search from a
# hardcoded top-4 to the actual max_cands -- both widen the candidate pool
# trackcorr's angle/acc gate must filter, trading some precision for more
# recall elsewhere (see test_track.py::test_cavity's baseline bump in the
# same commit). Measured trackcorr precision post-fix: 0.879 (at 30 frames;
# see the N_FRAMES=20 remeasurement below for the current floors).
#
# N_FRAMES cut 30 -> 20 (2026-08-21): the extra 10 frames added ~1/3 of this
# file's runtime (the sole cost driver -- subprocess-per-tracker with no
# parallelism) for no extra regression signal past both scripted crossings
# (frame 15, 18) settling into steady-state. Every floor below is a fresh
# measurement on the 20-frame dataset, not a rescale of the 30-frame numbers
# -- fewer frames changes the actual trajectories (fewer frames alive at once
# to fragment/misassign), so this is not directly comparable to the note
# above. Baseline measured 2026-08-21 on synthetic_turbulent_1k (20 frames,
# ~1000 particles/frame), same code as the 30-frame trackcorr note above:
#   priority_segment_3d: precision 0.902, yield_recall 0.881, ghost 0.038
#   trackcorr:           precision 0.902, yield_recall 0.826, ghost 0.038
#   kalman_hungarian_3d: precision 0.841, yield_recall 0.746, ghost 0.038
#   sg_hungarian_3d:     precision 0.579, yield_recall 0.514, ghost 0.038
#   nearest_hungarian_3d:precision 0.688, yield_recall 0.646, ghost 0.038
#   predictive_gmm_3d:   precision 0.707, yield_recall 0.690, ghost 0.035
_STAGE0B_FLOORS = {
    "priority_segment_3d": {"precision": 0.88, "yield_recall": 0.86, "ghost_capture_rate": 0.06},
    "trackcorr": {"precision": 0.88, "yield_recall": 0.80, "ghost_capture_rate": 0.06},
    "kalman_hungarian_3d": {"precision": 0.82, "yield_recall": 0.72, "ghost_capture_rate": 0.06},
    "sg_hungarian_3d": {"precision": 0.56, "yield_recall": 0.49, "ghost_capture_rate": 0.06},
    "nearest_hungarian_3d": {"precision": 0.67, "yield_recall": 0.62, "ghost_capture_rate": 0.06},
    "predictive_gmm_3d": {"precision": 0.69, "yield_recall": 0.67, "ghost_capture_rate": 0.055},
}


@pytest.mark.slow
@pytest.mark.parametrize("tracker", sorted(_STAGE0B_FLOORS))
def test_registered_tracker_quality_floor_at_1k_density(tracker):
    if not (DATASET / "res").exists():
        make_dataset(DATASET, num_particles=1000, num_frames=N_FRAMES, seed=2026)

    overrides = bu.per_tracker_overrides(
        [tracker], src=DATASET, first=bu.FIRST, n_frames=N_FRAMES,
    )[tracker]

    # Subprocess-isolated (see scripts/tune_tracker_params.py docstring):
    # openptv2's compiled Cython tracking extensions have been observed to
    # corrupt each other's memory (segfault) when several different
    # trackers run back-to-back in one process -- exactly what a
    # parametrized sweep over all registered trackers does.
    pred0, _dt = _run_via_subprocess(
        tracker, DATASET, bu.FIRST, N_FRAMES, overrides,
    )

    frames = bu.read_gt_frames(DATASET, bu.FIRST, N_FRAMES)
    tt = bu.build_true_tracks(frames, bu.FIRST)
    ghosts = bu.build_ghost_frames(frames, bu.FIRST)
    m = bu.bm.compute_identity_metrics(tt, pred0, eps=1.0, ghost_pos_by_frame=ghosts)
    row = {
        **m.to_dict(),
        **bu.calculate_tracking_metrics(tt, pred0, distance_tolerance=1.0).to_dict(),
    }

    floor = _STAGE0B_FLOORS[tracker]
    assert row["precision"] >= floor["precision"], row
    assert row["yield_recall"] >= floor["yield_recall"], row
    assert row["ghost_capture_rate"] <= floor["ghost_capture_rate"], row


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

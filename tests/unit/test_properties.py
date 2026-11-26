"""Property-based invariant tests for tracking.

Uses manual parametrized strategies (no hypothesis dependency) to check
invariants that must hold across many synthetic scenes: track ID uniqueness,
track length bounds, reconstructed-point volume bounds, and monotonic
frame coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.benchmarking import ScenarioSpec, generate_scenario
from openptv2.benchmarking.metrics import compute_identity_metrics


def _make_spec(**overrides) -> ScenarioSpec:
    base = dict(
        num_particles=20,
        num_frames=15,
        velocity=1.0,
        velocity_jitter=0.2,
        noise_mm=0.1,
        ghost_ratio=0.1,
        gap_probability=0.1,
        seed=1,
    )
    base.update(overrides)
    return ScenarioSpec(**base)


def _scene(seed: int, **kw):
    spec = _make_spec(seed=seed, **kw)
    true_tracks, frame_gt = generate_scenario(spec)
    return true_tracks, frame_gt, spec


class TestTrackingInvariants:
    def test_true_track_ids_unique(self):
        true_tracks, _, _ = _scene(seed=1)
        ids = list(true_tracks.keys())
        assert len(ids) == len(set(ids)), "true track IDs must be unique"

    def test_frame_coverage_matches_num_frames(self):
        true_tracks, frame_gt, spec = _scene(seed=2)
        assert len(frame_gt) == spec.num_frames

    def test_gap_probability_creates_shorter_tracks(self):
        true_tracks_full, _, _ = _scene(seed=3, gap_probability=0.0)
        true_tracks_gaps, _, _ = _scene(seed=3, gap_probability=0.5)
        full_lengths = [len(v) for v in true_tracks_full.values()]
        gap_lengths = [len(v) for v in true_tracks_gaps.values()]
        assert min(gap_lengths) <= min(full_lengths)

    def test_ghost_ratio_adds_spurious_particles(self):
        _, frame_gt_clean, _ = _scene(seed=4, ghost_ratio=0.0)
        _, frame_gt_ghosts, _ = _scene(seed=4, ghost_ratio=0.5)
        clean_max = max(len(pts) for pts in frame_gt_clean.values())
        ghost_max = max(len(pts) for pts in frame_gt_ghosts.values())
        assert ghost_max >= clean_max

    def test_compute_identity_metrics_output_bounds(self):
        true_tracks, _, _ = _scene(seed=5)
        metrics = compute_identity_metrics(true_tracks, {}, eps=1.0)
        assert 0.0 <= metrics.fragmentation
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.purity <= 1.0
        assert 0.0 <= metrics.pmt <= 1.0

    def test_identity_metrics_perfect_match(self):
        true_tracks, _, _ = _scene(seed=6)
        metrics = compute_identity_metrics(true_tracks, dict(true_tracks), eps=0.5)
        assert metrics.completeness >= 0.9
        assert metrics.purity >= 0.9

    def test_entering_leaving_tracks_shorter(self):
        true_tracks, _, spec = _scene(
            seed=7, entering_particles=3, leaving_particles=3,
        )
        short = [len(v) for v in true_tracks.values() if len(v) < spec.num_frames]
        assert len(short) >= 3, "entering/leaving particles must have shorter tracks"

    @pytest.mark.parametrize("n_particles", [5, 20, 100])
    def test_particle_count_respected(self, n_particles):
        true_tracks, _, spec = _scene(seed=8, num_particles=n_particles)
        assert len(true_tracks) == n_particles

    @pytest.mark.parametrize("n_frames", [5, 20, 50])
    def test_frame_count_respected(self, n_frames):
        true_tracks, frame_gt, spec = _scene(seed=9, num_frames=n_frames)
        assert len(frame_gt) == n_frames

    def test_track_length_monotonic_with_frames(self):
        t5, _, s5 = _scene(seed=10, num_frames=5)
        t50, _, s50 = _scene(seed=10, num_frames=50)
        avg5 = sum(len(v) for v in t5.values()) / len(t5)
        avg50 = sum(len(v) for v in t50.values()) / len(t50)
        assert avg50 >= avg5, "more frames should yield longer average tracks"

    def test_reconstructed_positions_finite(self):
        true_tracks, _, _ = _scene(seed=11)
        for tid, pts in true_tracks.items():
            for _f, x, y, z in pts:
                assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)

    def test_metrics_ghost_capture_rate_bounds(self):
        true_tracks, _, _ = _scene(seed=12, ghost_ratio=0.3)
        metrics = compute_identity_metrics(true_tracks, {}, eps=1.0)
        assert 0.0 <= metrics.ghost_capture_rate <= 1.0

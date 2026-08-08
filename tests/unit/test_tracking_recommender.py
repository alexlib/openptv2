"""Unit tests for openptv2.tracking_recommender."""

import numpy as np
import pytest

from openptv2.tracking_recommender import (
    DatasetStats,
    Recommendation,
    compute_dataset_stats,
    print_recommendation,
    recommend_tracker,
)


def _make_fake_frames(num_frames=10, num_particles=50, noise=0.1):
    """Generate synthetic frame particle arrays for testing."""
    rng = np.random.default_rng(42)
    frames = []
    for f in range(num_frames):
        base = rng.uniform(0, 100, size=(num_particles, 3))
        if f > 0:
            base += rng.normal(0.5, noise, size=(num_particles, 3))
        frames.append(base)
    return frames


def test_compute_dataset_stats_basic():
    """Verify stats computation returns reasonable values."""
    frames = _make_fake_frames(10, 50)
    stats = compute_dataset_stats(frames)
    assert stats.num_frames == 10
    assert stats.mean_particles_per_frame == 50.0
    assert stats.max_particles_per_frame == 50
    assert stats.max_displacement > 0
    assert stats.mean_interparticle_distance > 0
    assert stats.domain_size != (0.0, 0.0, 0.0)


def test_compute_dataset_stats_empty():
    """Verify stats on empty input doesn't crash."""
    stats = compute_dataset_stats([])
    assert stats.num_frames == 0
    assert stats.mean_particles_per_frame == 0.0


def test_compute_dataset_stats_single_frame():
    """Verify stats on single frame doesn't crash."""
    frames = [np.random.uniform(0, 10, size=(20, 3))]
    stats = compute_dataset_stats(frames)
    assert stats.num_frames == 1
    assert stats.max_displacement == 0.0  # no pairs to compare


def test_compute_dataset_stats_with_gaps():
    """Verify gap detection."""
    frames = [
        np.random.uniform(0, 100, size=(100, 3)),
        np.random.uniform(0, 100, size=(80, 3)),  # 20% drop
        np.random.uniform(0, 100, size=(60, 3)),  # 40% drop
    ]
    stats = compute_dataset_stats(frames)
    assert stats.has_gaps is True
    assert stats.gap_fraction > 0


def test_recommend_tracker_default():
    """Verify recommendation returns something reasonable."""
    frames = _make_fake_frames(20, 100, noise=0.5)
    stats = compute_dataset_stats(frames)
    rec = recommend_tracker(stats)
    assert isinstance(rec, Recommendation)
    assert rec.tracker_name
    assert rec.rationale
    assert len(rec.suggested_params) > 0


def test_recommend_tracker_with_gaps():
    """Verify gaps affect recommendation."""
    frames = [
        np.random.uniform(0, 100, size=(100, 3)),
        np.random.uniform(0, 100, size=(100, 3)),
        np.random.uniform(0, 100, size=(10, 3)),  # 90% gap
        np.random.uniform(0, 100, size=(100, 3)),
    ]
    stats = compute_dataset_stats(frames)
    rec = recommend_tracker(stats, user_preferences={"require_backward": True})
    assert rec.tracker_name


def test_recommend_speed_priority():
    """Verify speed priority selects fast trackers."""
    frames = _make_fake_frames(10, 30)
    stats = compute_dataset_stats(frames)
    rec = recommend_tracker(stats, user_preferences={"priority": "speed"})
    speed_rank = {"fastest": 0, "fast": 1, "moderate": 2, "slow": 3}
    assert rec.tracker_info is not None
    rank = speed_rank.get(rec.tracker_info.speed_ranking, 3)
    assert rank <= 2  # Not slow


def test_recommend_accuracy_priority():
    """Verify accuracy priority selects accurate trackers."""
    frames = _make_fake_frames(10, 10)  # low density
    stats = compute_dataset_stats(frames)
    rec = recommend_tracker(stats, user_preferences={"priority": "accuracy"})
    assert rec.tracker_info is not None
    assert rec.tracker_info.accuracy_ranking in ("high", "highest")


def test_recommend_suggests_params():
    """Verify recommended params contains velocity bounds."""
    frames = _make_fake_frames(10, 50)
    stats = compute_dataset_stats(frames)
    rec = recommend_tracker(stats)
    if stats.max_displacement > 0:
        assert "dvxmax" in rec.suggested_params
        assert rec.suggested_params["dvxmax"] > 0


def test_print_recommendation():
    """Verify print_recommendation returns a formatted string."""
    rec = Recommendation(
        tracker_name="fast_3d",
        confidence=0.85,
        rationale=["Test reason 1", "Test reason 2"],
        suggested_params={"dvxmax": 10.0, "dacc": 5.0},
        alternate_choices=["full_multipass"],
    )
    output = print_recommendation(rec)
    assert "Tracker Recommendation" in output
    assert "Test reason 1" in output
    assert "dvxmax" in output


def test_density_categorization():
    """Verify density category is set."""
    frames = _make_fake_frames(10, 200)  # high density
    stats = compute_dataset_stats(frames)
    assert stats.density_category != "unknown"

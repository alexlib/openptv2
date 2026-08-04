"""Unit tests for openptv2.tracking_metrics."""

import pytest
from openptv2.tracking_metrics import (
    TrackingMetrics,
    calculate_tracking_metrics,
    generate_synthetic_benchmark_dataset,
)


def test_synthetic_dataset_generator():
    """Verify synthetic dataset generator creates tracks and noisy blobs."""
    true_tracks, frame_blobs = generate_synthetic_benchmark_dataset(
        num_particles=20,
        num_frames=10,
        noise_std=0.01,
        gap_probability=0.1,
        flow_type="vortex",
        seed=123,
    )

    assert len(true_tracks) == 20
    assert len(frame_blobs) == 10
    # Frame 0 should have 20 blobs
    assert len(frame_blobs[0]) == 20


def test_perfect_tracking_metrics():
    """Verify tracking metrics report 100% yield and precision for identical tracks."""
    true_tracks, _ = generate_synthetic_benchmark_dataset(
        num_particles=10,
        num_frames=5,
        noise_std=0.0,
        gap_probability=0.0,
        flow_type="linear",
        seed=42,
    )

    # Identical predicted tracks
    metrics = calculate_tracking_metrics(
        true_tracks, true_tracks, distance_tolerance=0.05
    )

    assert metrics.yield_recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.false_connection_rate == pytest.approx(0.0)
    assert metrics.total_correct_links == metrics.total_true_links
    assert metrics.mean_track_length == pytest.approx(5.0)
    assert metrics.max_track_length == 5


def test_noisy_tracking_metrics():
    """Verify tracking metrics correctly evaluate slightly noisy predictions."""
    true_tracks, _ = generate_synthetic_benchmark_dataset(
        num_particles=15,
        num_frames=8,
        noise_std=0.0,
        gap_probability=0.0,
        flow_type="burgers",
        seed=99,
    )

    # Add minor noise to predicted tracks
    predicted_tracks = {}
    for track_id, points in true_tracks.items():
        noisy_points = []
        for f, x, y, z in points:
            noisy_points.append((f, x + 0.005, y - 0.005, z + 0.002))
        predicted_tracks[track_id] = noisy_points

    metrics = calculate_tracking_metrics(
        true_tracks, predicted_tracks, distance_tolerance=0.05
    )

    assert metrics.yield_recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.rms_position_error > 0.0
    assert metrics.rms_position_error < 0.02


def test_tracking_metrics_to_dict():
    """Verify TrackingMetrics dataclass conversion to dict."""
    true_tracks, _ = generate_synthetic_benchmark_dataset(
        num_particles=5, num_frames=3, seed=1
    )
    metrics = calculate_tracking_metrics(true_tracks, true_tracks)
    d = metrics.to_dict()

    assert isinstance(d, dict)
    assert "yield_recall" in d
    assert "precision" in d
    assert "rms_position_error" in d

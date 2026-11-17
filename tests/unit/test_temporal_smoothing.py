"""Tests for temporal_smoothing module."""

import numpy as np

from openptv2.temporal_smoothing import (
    trajectory_acceleration_residual,
    smoothness_keep,
    irls_weights,
    smooth_trajectories,
    resolve_smoothing_config,
)


def test_residual_zero_for_smooth_line():
    pos = np.column_stack([np.arange(10, dtype=float), np.zeros(10), np.zeros(10)])
    assert trajectory_acceleration_residual(pos) == 0.0


def test_residual_high_for_noisy():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(10, 3))
    assert trajectory_acceleration_residual(pos) > 0.5


def test_residual_short():
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert trajectory_acceleration_residual(pos) == 0.0


def test_smoothness_keep():
    pos = np.column_stack([np.arange(10, dtype=float), np.zeros(10), np.zeros(10)])
    assert smoothness_keep(pos, 1e-9) is True
    noisy = np.random.default_rng(2).normal(size=(10, 3)) * 5
    assert smoothness_keep(noisy, 1e-9) is False


def test_irls_weights_shape_and_range():
    pos = np.column_stack([np.arange(10, dtype=float), np.zeros(10), np.zeros(10)])
    w = irls_weights(pos)
    assert w.shape == (10,)
    assert np.all(w >= 1e-3) and np.all(w <= 1.0)


def test_smooth_trajectories_keeps_smooth_drops_noisy():
    smooth = np.column_stack([np.arange(10, dtype=float), np.zeros(10), np.zeros(10)])
    noisy = np.random.default_rng(3).normal(size=(10, 3)) * 5
    flags = smooth_trajectories([smooth, noisy], threshold=1.0)
    assert flags[0] is True
    assert flags[1] is False


def test_resolve_smoothing_config():
    assert resolve_smoothing_config({"temporal_smoothing": True}, {}) is True
    assert resolve_smoothing_config({"temporal_smoothing": "false"}, {}) is False
    assert resolve_smoothing_config({"temporal_smoothing": "auto"}, {"slow_flow": True}) is True
    assert resolve_smoothing_config({"temporal_smoothing": "auto"}, {"jump_rate": 0.2}) is True
    assert resolve_smoothing_config({"temporal_smoothing": "auto"}, {}) is True
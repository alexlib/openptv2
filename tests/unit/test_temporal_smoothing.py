"""Tests for temporal_smoothing module."""

import numpy as np

from openptv2.temporal_smoothing import (
    irls_weights,
    resolve_smoothing_config,
    smooth_trajectories,
    smoothness_keep,
    trajectory_acceleration_residual,
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
    assert resolve_smoothing_config({"temporal_smoothing": "auto"}, {}) is False


def test_irls_predict_state_quadratic():
    from openptv2.temporal_smoothing import irls_predict_state

    t = np.arange(6, dtype=float)
    pos = np.column_stack([t, 0.5 * t ** 2, np.zeros_like(t)])
    p_pred, v_pred, resid = irls_predict_state(t, pos, 6.0)
    assert np.allclose(p_pred, [6.0, 18.0, 0.0], atol=1e-6)
    assert np.allclose(v_pred, [1.0, 6.0, 0.0], atol=1e-6)
    assert resid < 1e-6


def test_stitch_tracklets_irls_connects_broken_tracklet():
    from openptv2.temporal_smoothing import stitch_tracklets_irls

    # 6 frames: tracklet A (0,1,2) -> broken link -> tracklet B (3,4,5)
    frames = {}
    for f in range(6):
        prv = np.array([-1], dtype=np.int32)
        nxt = np.array([-1], dtype=np.int32)
        xyz = np.array([[float(f), 0.0, 0.0]], dtype=np.float64)
        frames[f] = (prv, nxt, xyz)
    for f in (0, 1, 3, 4):
        frames[f][1][0] = 0
        frames[f + 1][0][0] = 0

    n_stitched, dirty = stitch_tracklets_irls(frames, 0, 5, pos_tol=0.5, vel_tol=0.5)
    assert n_stitched == 1
    assert dirty == {2, 3}
    assert int(frames[2][1][0]) == 0
    assert int(frames[3][0][0]) == 0

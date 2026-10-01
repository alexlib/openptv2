"""A10 after tracking: end trimming, spike dropping, weighted smoothing."""

import numpy as np

from openptv2.quality_post import (
    drop_spikes,
    quality_weights,
    trim_doubtful_ends,
    weighted_savgol,
)


def test_trim_cuts_doubtful_ends_only():
    tid = np.zeros(8, int)
    fr = np.arange(8)
    g = np.array([0.9, 0.8, 0.1, 0.9, 0.1, 0.1, 0.7, 0.95])
    keep = trim_doubtful_ends(tid, fr, g, thr=0.5, max_trim=3)
    assert keep.tolist() == [False, False, True, True, True, True, False, False]
    keep_short = trim_doubtful_ends(tid, fr, g, thr=0.5, max_trim=3, max_frac=0.25)
    assert keep_short.tolist() == [False, False, True, True, True, True, False, False]
    # an 8-point track loses at most int(8 * 0.1) = 0 points per end
    assert trim_doubtful_ends(tid, fr, g, 0.5, 3, max_frac=0.1).all()
    keep1 = trim_doubtful_ends(tid, fr, g, thr=0.5, max_trim=1)
    assert keep1.tolist() == [False, True, True, True, True, True, True, False]


def test_drop_spikes_only_interior_jumps():
    tid = np.zeros(6, int)
    fr = np.arange(6)
    g = np.array([0.9, 0.1, 0.8, 0.1, 0.7, 0.8])  # row 2 is a spike
    keep = drop_spikes(tid, fr, g, thr=0.5, jump=0.4)
    assert keep.tolist() == [True, True, False, True, True, True]


def test_weighted_savgol_unit_weights_recovers_cubic_and_derivatives():
    n, fps = 40, 1000.0
    t = np.arange(n, dtype=float)
    x = np.c_[0.01 * t + 1e-4 * t**2 + 1e-6 * t**3, 0 * t, 0 * t]
    tid, fr = np.zeros(n, int), np.arange(n)
    _, _, p, v, a = weighted_savgol(tid, fr, x, None, fps, window=11, order=3)
    assert np.allclose(p, x, atol=1e-9)
    assert np.allclose(v[:, 0], (0.01 + 2e-4 * t + 3e-6 * t**2) * fps, rtol=1e-6)
    assert np.allclose(a[:, 0], (2e-4 + 6e-6 * t) * fps**2, rtol=1e-4, atol=1e-6)


def test_low_weight_point_is_ignored():
    n = 21
    t = np.arange(n, dtype=float)
    x = np.c_[0.1 * t, 0 * t, 0 * t]
    x_bad = x.copy()
    x_bad[10, 0] += 5.0  # outlier
    tid, fr = np.zeros(n, int), np.arange(n)
    w = np.ones(n)
    w[10] = 1e-9
    _, _, p, v, _ = weighted_savgol(tid, fr, x_bad, w, 1.0, window=11, order=3)
    assert abs(p[10, 0] - x[10, 0]) < 1e-3 and abs(v[10, 0] - 0.1) < 1e-3
    _, _, p0, _, _ = weighted_savgol(tid, fr, x_bad, None, 1.0, window=11, order=3)
    assert abs(p0[10, 0] - x[10, 0]) > 0.5


def test_quality_weights_floor_and_order():
    w = quality_weights(np.array([0.0, 0.5, 1.0]))
    assert w[0] == 1.0 and w[1] < w[0] and w[2] == 0.05


def test_short_trajectory_uses_all_its_points():
    n = 10  # window 21 asked, only 10 points: all 10 are fitted
    t = np.arange(n, dtype=float)
    x = np.c_[0.1 * t, 0 * t, 0 * t]
    _, _, p, v, _ = weighted_savgol(np.zeros(n, int), np.arange(n), x, None, 1.0, 21, 3)
    assert np.allclose(p, x, atol=1e-9) and np.allclose(v[:, 0], 0.1, atol=1e-9)

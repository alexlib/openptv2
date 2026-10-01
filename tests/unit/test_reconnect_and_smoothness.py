"""A5 reconnect and A9 smoothness filter (pred.npz-style arrays)."""

import numpy as np

from openptv2.reconnect import reconnect_pieces
from openptv2.smoothness_filter import leave_one_out_residual, smoothness_keep


def _line(tid, frames, x0, v, rng, noise=0.02, lane=None):
    f = np.asarray(frames)
    pos = np.stack(
        [
            x0 + v * f,
            np.full(len(f), (tid if lane is None else lane) * 10.0),
            np.zeros(len(f)),
        ],
        axis=1,
    )
    return np.full(len(f), tid), f, pos + rng.normal(0, noise, pos.shape)


def _stack(parts):
    return tuple(np.concatenate(x) for x in zip(*parts))


def test_reconnect_joins_a_broken_straight_track_across_a_gap():
    rng = np.random.default_rng(0)
    a = _line(0, range(0, 12), 0.0, 0.1, rng)
    b = _line(1, range(14, 26), 0.0, 0.1, rng, lane=0)  # same path, 2 frames missing
    tid, fr, pos = _stack([a, b])
    new = reconnect_pieces(tid, fr, pos, max_gap=4)
    assert len(np.unique(new)) == 1


def test_reconnect_does_not_join_different_particles_or_too_large_gaps():
    rng = np.random.default_rng(1)
    a = _line(0, range(0, 12), 0.0, 0.1, rng)
    far = _line(1, range(14, 26), 0.0, 0.1, rng)
    far = (far[0], far[1], far[2] + np.array([0.0, 3.0, 0.0]))  # 3 mm off the path
    tid, fr, pos = _stack([a, far])
    assert len(np.unique(reconnect_pieces(tid, fr, pos, max_gap=4))) == 2
    late = _line(2, range(30, 42), 0.0, 0.1, rng)  # same path but 19 frames later
    tid, fr, pos = _stack([a, late])
    assert len(np.unique(reconnect_pieces(tid, fr, pos, max_gap=4))) == 2


def test_reconnect_uses_each_end_and_start_once():
    rng = np.random.default_rng(2)
    a = _line(0, range(0, 10), 0.0, 0.1, rng)
    b1 = _line(1, range(12, 22), 0.0, 0.1, rng, lane=0)
    b2 = _line(2, range(12, 22), 0.0, 0.1, rng, lane=0)  # a duplicate candidate
    tid, fr, pos = _stack([a, b1, b2])
    assert len(np.unique(reconnect_pieces(tid, fr, pos, max_gap=4))) == 2


def test_smoothness_filter_drops_an_outlier_and_keeps_clean_points():
    rng = np.random.default_rng(3)
    t, f, p = _line(0, range(0, 40), 0.0, 0.1, rng)
    p[20] += np.array([0.0, 0.8, 0.0])
    res = leave_one_out_residual(t, f, p)
    assert np.nanargmax(res) == 20
    keep = smoothness_keep(t, f, p, k=6.0, deg=1)
    assert not keep[20] and keep.sum() >= 38


def test_smoothness_filter_does_not_cut_across_a_gap_and_spares_short_tracks():
    rng = np.random.default_rng(4)
    frames = list(range(0, 15)) + list(range(20, 35))  # a reconnected gap
    t, f, p = _line(0, frames, 0.0, 0.1, rng)
    assert smoothness_keep(t, f, p, k=6.0, deg=1).all()
    s = _line(1, range(0, 3), 0.0, 0.1, rng)  # shorter than min_len: never judged
    assert smoothness_keep(*s, k=6.0).all()

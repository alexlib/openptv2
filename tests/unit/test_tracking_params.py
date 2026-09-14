"""Tests for openptv2.tracking_params (data-driven trackcorr parameters)."""

import numpy as np
import pytest

from openptv2.tracking_params import (
    noise_sigma,
    recommend_from_store,
    recommend_trackcorr_params,
    track_statistics,
    tracks_from_store,
)


def _tracks(n_tracks=400, length=30, speed=0.1, sigma=(0.01, 0.02, 0.05), seed=0):
    rng = np.random.default_rng(seed)
    tid = np.repeat(np.arange(n_tracks), length)
    frame = np.tile(np.arange(length), n_tracks)
    vel = rng.normal(0.0, 1.0, (n_tracks, 3))
    vel = speed * vel / np.linalg.norm(vel, axis=1, keepdims=True)
    pos = np.repeat(vel, length, axis=0) * frame[:, None] + rng.normal(0.0, 1.0, (n_tracks * length, 3)) * sigma
    return tid, frame, pos


def test_noise_sigma_recovers_white_noise():
    sigma = np.array([0.01, 0.02, 0.05])
    est = noise_sigma(*_tracks(sigma=sigma))
    np.testing.assert_allclose(est, sigma, rtol=0.1)


def test_noise_sigma_ignores_a_wrong_link():
    tid, frame, pos = _tracks()
    pos = pos.copy()
    pos[(tid == 3) & (frame >= 15)] += [2.0, 0.0, 0.0]  # a jump to "another particle"
    np.testing.assert_allclose(noise_sigma(tid, frame, pos), [0.01, 0.02, 0.05], rtol=0.15)


def test_slow_noisy_motion_turns_the_angle_limit_off():
    stats = track_statistics(*_tracks(speed=0.1))
    params, reasons = recommend_trackcorr_params(stats)
    assert params["angle"] == 270.0
    assert any("angle limit off" in r for r in reasons)
    # dacc comes from the noise: above the per-axis noise acceleration, below a gross value
    assert np.sqrt(6) * 0.05 < params["dacc"] < 2.0
    # dv covers every measured step
    assert params["dvxmax"] >= stats["step_abs_p999"][0]
    assert params["dvzmin"] == -params["dvzmax"]


def test_fast_clean_motion_keeps_an_angle_limit():
    stats = track_statistics(*_tracks(speed=1.0, sigma=(0.001, 0.001, 0.001)))
    params, _ = recommend_trackcorr_params(stats)
    assert params["angle"] < 270.0


def test_track_statistics_needs_links():
    with pytest.raises(ValueError):
        track_statistics(np.arange(4), np.zeros(4), np.zeros((4, 3)))


def test_tracks_from_store_follows_prev_pointers(tmp_path):
    from openptv2.storage import RunStore

    store = RunStore(tmp_path / "run.zarr", mode="w")
    xyz = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    for f in range(1, 4):
        prev = np.array([-1, -1]) if f == 1 else np.array([1, 0])  # tracks swap row order
        nxt = np.array([1, 0]) if f < 3 else np.array([-2, -2])
        store.write_linkage(f, prev, nxt, xyz + f, name="ptv_is")

    tid, frame, pos = tracks_from_store(store, 1, 3)

    assert len(tid) == 6
    # each track id appears once per frame
    for t in np.unique(tid):
        assert sorted(frame[tid == t].tolist()) == [1, 2, 3]


def test_recommend_from_store_round_trip(tmp_path):
    from openptv2.storage import RunStore

    tid, frame, pos = _tracks(n_tracks=200, length=12)
    store = RunStore(tmp_path / "run.zarr", mode="w")
    for f in range(12):
        order = np.flatnonzero(frame == f)
        prev = np.full(len(order), -1) if f == 0 else np.arange(len(order))
        nxt = np.arange(len(order)) if f < 11 else np.full(len(order), -2)
        store.write_linkage(f + 1, prev, nxt, pos[order], name="ptv_is")

    params, stats, reasons = recommend_from_store(store, 1, 12)

    np.testing.assert_allclose(stats["noise_sigma"], [0.01, 0.02, 0.05], rtol=0.2)
    assert params["angle"] == 270.0 and len(reasons) == 3

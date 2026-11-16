"""Benchmarks for temporal smoothing ghost suppression."""

import numpy as np
import pytest

from openptv2.temporal_smoothing import (
    smooth_trajectories,
    trajectory_acceleration_residual,
)


def _traj(smooth: bool, n: int = 20, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
    if smooth:
        return base + rng.normal(scale=0.01, size=base.shape)
    # ghost: random walk / high acceleration
    return rng.normal(size=(n, 3)) * 5.0


@pytest.mark.parametrize("n_ghosts", [1, 5, 10])
def test_smoothing_removes_ghost_trajectories(n_ghosts):
    trajs = [_traj(True, seed=i) for i in range(10)] + [_traj(False, seed=100 + i) for i in range(n_ghosts)]
    flags = smooth_trajectories(trajs, threshold=1.0)
    # all smooth kept
    assert all(flags[:10])
    # ghosts dropped
    assert not any(flags[10:])


def test_smoothing_preserves_real_trajectories_high_noise():
    # Real but noisy (turbulent) trajectories should still survive a generous threshold.
    rng = np.random.default_rng(5)
    trajs = [np.column_stack([np.arange(20, dtype=float), rng.normal(scale=0.5, size=20), np.zeros(20)]) for _ in range(5)]
    flags = smooth_trajectories(trajs, threshold=5.0)
    assert all(flags)


def test_residual_separates_smooth_from_ghost():
    smooth = _traj(True)
    ghost = _traj(False)
    r_s = trajectory_acceleration_residual(smooth)
    r_g = trajectory_acceleration_residual(ghost)
    assert r_s < r_g
    assert r_s < 1.0
    assert r_g > 1.0
"""Unit tests for the vendored proPTV core routines.

Covers the self-contained numpy functions in openptv2.plugins.proptv
(initialisation and prediction) that power the proPTV tracker. These are the
integers-only pieces that are deterministic and don't need camera geometry, so
they get their own focused tests; the full ProPTVTracker plugin is exercised
end-to-end by the batch burger tests.
"""

import numpy as np
import pytest

from openptv2.plugins.proptv._config import ProPTVConfig
from openptv2.plugins.proptv.initialisation import (
    find_nn_points,
    init_acceleration_3d,
    init_position_3d,
    init_velocity_3d,
)
from openptv2.plugins.proptv.prediction import (
    GMM,
    Approximate,
    Predict,
)

# ── initialisation ──────────────────────────────────────────────────────────


def test_find_nn_points_returns_n_closest():
    p = np.array([0.0, 0.0, 0.0])
    P = np.array(
        [
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    nn = find_nn_points(p, P, 2)
    # The two nearest points are at distance 1 and 2.
    dists = sorted(np.linalg.norm(nn - p, axis=1))
    assert dists == pytest.approx([1.0, 2.0])


def test_find_nn_points_requires_n_lt_len():
    """argpartition needs kth < len(P); requesting len(P) or more fails."""
    p = np.array([0.0, 0.0, 0.0])
    P = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        find_nn_points(p, P, 3)


def test_init_position_smooths_linear_exactly():
    t = np.arange(10, dtype=float)
    track = np.stack([t, 2 * t, 3 * t], axis=1)
    pos = init_position_3d(track)
    np.testing.assert_allclose(pos, track, atol=1e-8)


def test_init_velocity_linear_is_constant():
    t = np.arange(10, dtype=float)
    track = np.stack([t, 2 * t, 3 * t], axis=1)
    vel = init_velocity_3d(track)
    # velocity of [1,2,3] per step
    np.testing.assert_allclose(vel, np.tile([1.0, 2.0, 3.0], (10, 1)), atol=1e-6)


def test_init_acceleration_linear_is_zero():
    t = np.arange(10, dtype=float)
    track = np.stack([t, 2 * t, 3 * t], axis=1)
    acc = init_acceleration_3d(track)
    np.testing.assert_allclose(acc, np.zeros_like(track), atol=1e-3)


def test_init_routines_preserve_shape_and_format():
    rng = np.random.default_rng(0)
    track = rng.standard_normal((8, 3))
    for fn in (init_position_3d, init_velocity_3d, init_acceleration_3d):
        out = fn(track)
        assert out.shape == (8, 3)
        assert np.isfinite(out).all()


# ── prediction (GMM) ────────────────────────────────────────────────────────


def test_gmm_recovers_linear_trajectory():
    """The GMM approximation must reproduce a linear trajectory."""
    n = 8
    t = np.arange(n, dtype=float)
    v = np.array([1.0, -0.5, 0.25])
    X = np.stack([t * v[d] for d in range(3)], axis=1)

    w, psi_X, psi_V, psi_A = GMM(t, X)
    approx_X, approx_V, approx_A = Approximate(t, w, psi_X, psi_V, psi_A)

    np.testing.assert_allclose(approx_X, X, atol=1e-4)
    # velocity stays close to the constant v
    np.testing.assert_allclose(approx_V, np.tile(v, (n, 1)), atol=1e-3)
    # constant-velocity trajectory has ~zero acceleration
    np.testing.assert_allclose(approx_A, np.zeros((n, 3)), atol=1e-3)


def test_predict_advances_linear_state():
    """Predict must extrapolate one step of linear motion."""
    n = 8
    t = np.arange(n, dtype=float)
    v = np.array([1.0, 2.0, 3.0])
    X = np.stack([t * v[d] for d in range(3)], axis=1)

    w, psi_X, psi_V, psi_A = GMM(t, X)
    approx_X, approx_V, approx_A = Approximate(t, w, psi_X, psi_V, psi_A)
    Xn, Vn, An = Predict(t, approx_X, approx_V, approx_A)

    # one step forward from the last position
    np.testing.assert_allclose(Xn, X[-1] + v, atol=1e-3)


def test_proptv_config_defaults_are_sane():
    cfg = ProPTVConfig()
    assert cfg.t_init == 4
    assert cfg.maxvel == 20.0
    assert cfg.angle == 30.0
    assert cfg.activeMatches_extend == 3
    assert cfg.dt == 1
    assert len(cfg.Vmin) == 3 and len(cfg.Vmax) == 3
    assert len(cfg.NN) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for ray_tracing module.

Each function is tested with explicit known inputs and expected outputs,
following the pattern of lib/tests/check_ray_tracing.c.
"""

import numpy as np
import pytest

from algorithms.calibration import Calibration
from algorithms.parameters import MultimediaPar
from algorithms.ray_tracing import ray_tracing

EPS = 1e-5


def test_ray_tracing_known_values():
    """Reproduce the check_ray_tracing.c test_ray_tracing values exactly."""
    # Input: image-plane position
    x = 100.0
    y = 100.0

    # Calibration matching C test
    cal = Calibration()
    cal.set_pos([0.0, 0.0, 100.0])
    # Set dm explicitly (same non-identity matrix as C test, not from rotation_matrix)
    cal.ext_par["dm"][0] = [1.0,  0.2, -0.3]
    cal.ext_par["dm"][1] = [0.2,  1.0,  0.0]
    cal.ext_par["dm"][2] = [-0.3, 0.0,  1.0]
    cal.set_primary_point(np.array([0.0, 0.0, 100.0]))  # xh, yh, cc
    cal.glass_par = np.array([0.0001, 0.00001, 1.0])

    # Multimedia: 3-media (air – glass – water), only first layer used
    mm = MultimediaPar(n1=1.0, n2=[1.49], d=[5.0], n3=1.33)

    X, a = ray_tracing(x, y, cal, mm)

    # Expected from check_ray_tracing.c
    np.testing.assert_allclose(
        X, [110.406944, 88.325788, 0.988076], atol=EPS,
        err_msg=f"Crossing point X wrong: got {X}"
    )
    np.testing.assert_allclose(
        a, [0.387960, 0.310405, -0.867834], atol=EPS,
        err_msg=f"Direction vector a wrong: got {a}"
    )


def test_ray_tracing_returns_two_arrays():
    """ray_tracing always returns (X, a) as two 3-element arrays."""
    cal = Calibration()
    cal.set_pos([0.0, 0.0, 100.0])
    cal.set_primary_point(np.array([0.0, 0.0, 10.0]))
    cal.glass_par = np.array([0.0, 0.0, 1.0])
    mm = MultimediaPar(n1=1.0, n2=[1.0], d=[1.0], n3=1.0)

    X, a = ray_tracing(0.0, 0.0, cal, mm)

    assert X.shape == (3,)
    assert a.shape == (3,)


def test_ray_tracing_all_air_direction_is_unit():
    """In all-air, the output direction vector should be a unit vector."""
    cal = Calibration()
    cal.set_pos([0.0, 0.0, 100.0])
    cal.set_primary_point(np.array([0.0, 0.0, 10.0]))
    cal.glass_par = np.array([0.0, 0.0, 1.0])
    mm = MultimediaPar(n1=1.0, n2=[1.0], d=[1.0], n3=1.0)

    _, a = ray_tracing(5.0, 3.0, cal, mm)

    assert abs(np.linalg.norm(a) - 1.0) < EPS

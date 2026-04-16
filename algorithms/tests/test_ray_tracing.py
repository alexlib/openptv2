import numpy as np
import pytest

from algorithms.calibration import Calibration
from algorithms.parameters import MultimediaPar
from algorithms.ray_tracing import ray_tracing

EPS = 1e-5

def test_ray_tracing():
    # Input: image-plane position
    x = 100.0
    y = 100.0

    # Calibration matching C test
    cal = Calibration()
    cal.set_pos([0.0, 0.0, 100.0])
    # Set dm explicitly (same non-identity matrix as C test)
    cal.ext_par.dm = np.array([
        [1.0,  0.2, -0.3],
        [0.2,  1.0,  0.0],
        [-0.3, 0.0,  1.0]
    ])
    
    cal.set_primary_point(np.array([0.0, 0.0, 100.0]))  # xh, yh, cc
    cal.glass_par = np.array([0.0001, 0.00001, 1.0])

    # Multimedia: 3-media (air – glass – water), only first layer used
    mm = MultimediaPar(nlay=3, n1=1.0, n2=[1.49, 0.0, 0.0], d=[5.0, 0.0, 0.0], n3=1.33)

    X, a = ray_tracing(x, y, cal.ext_par.dm, cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.int_par.cc, cal.glass_par[0], cal.glass_par[1], cal.glass_par[2], mm.n1, mm.n2[0], mm.n3, mm.d[0])

    # Expected from check_ray_tracing.c
    np.testing.assert_allclose(
        X, [110.406944, 88.325788, 0.988076], atol=EPS,
        err_msg=f"Crossing point X wrong: got {X}"
    )
    np.testing.assert_allclose(
        a, [0.387960, 0.310405, -0.867834], atol=EPS,
        err_msg=f"Direction vector a wrong: got {a}"
    )

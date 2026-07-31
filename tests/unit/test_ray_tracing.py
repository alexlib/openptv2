import numpy as np

from openptv2.algorithms.calibration import (
    Exterior,
    Glass,
    Interior,
)
from openptv2.algorithms.ray_tracing import ray_tracing

EPS = 1e-5


def test_ray_tracing():
    x = 100.0
    y = 100.0

    ext = Exterior(
        x0=0.0, y0=0.0, z0=100.0,
        omega=0.0, phi=0.0, kappa=0.0,
        dm=np.array([
            [1.0,  0.2, -0.3],
            [0.2,  1.0,  0.0],
            [-0.3, 0.0,  1.0],
        ], dtype=np.float64),
    )
    int_par = Interior(xh=0.0, yh=0.0, cc=100.0)
    glass = Glass(vec_x=0.0001, vec_y=0.00001, vec_z=1.0)

    mm_n1 = 1.0
    mm_n2_0 = 1.49
    mm_n3 = 1.33
    mm_d0 = 5.0

    X, a = ray_tracing(
        x, y,
        ext.dm, ext.x0, ext.y0, ext.z0,
        int_par.cc,
        glass.vec_x, glass.vec_y, glass.vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    np.testing.assert_allclose(
        X, [110.406944, 88.325788, 0.988076], atol=EPS,
        err_msg=f"Crossing point X wrong: got {X}",
    )
    np.testing.assert_allclose(
        a, [0.387960, 0.310405, -0.867834], atol=EPS,
        err_msg=f"Direction vector a wrong: got {a}",
    )

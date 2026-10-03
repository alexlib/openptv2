import numpy as np
import pytest

from openptv2.algorithms.calibration import (
    Calibration,
    Exterior,
    Glass,
    Interior,
)
from openptv2.algorithms.parameters import MmNp
from openptv2.algorithms.ray_tracing import ray_tracing, ray_tracing_batch

EPS = 1e-5


def test_ray_tracing():
    x = 100.0
    y = 100.0

    ext = Exterior(
        x0=0.0,
        y0=0.0,
        z0=100.0,
        omega=0.0,
        phi=0.0,
        kappa=0.0,
        dm=np.array(
            [
                [1.0, 0.2, -0.3],
                [0.2, 1.0, 0.0],
                [-0.3, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    int_par = Interior(xh=0.0, yh=0.0, cc=100.0)
    glass = Glass(vec_x=0.0001, vec_y=0.00001, vec_z=1.0)

    mm_n1 = 1.0
    mm_n2_0 = 1.49
    mm_n3 = 1.33
    mm_d0 = 5.0

    X, a = ray_tracing(
        x,
        y,
        ext.dm,
        ext.x0,
        ext.y0,
        ext.z0,
        int_par.cc,
        glass.vec_x,
        glass.vec_y,
        glass.vec_z,
        mm_n1,
        mm_n2_0,
        mm_n3,
        mm_d0,
    )

    np.testing.assert_allclose(
        X,
        [110.406944, 88.325788, 0.988076],
        atol=EPS,
        err_msg=f"Crossing point X wrong: got {X}",
    )
    np.testing.assert_allclose(
        a,
        [0.387960, 0.310405, -0.867834],
        atol=EPS,
        err_msg=f"Direction vector a wrong: got {a}",
    )


def _batch_test_cal(x0, z0, look_plus_z, glass_vec):
    """Calibration mimicking the two Ilmenau rig sides.

    look_plus_z=True: camera at -Z looking toward +Z (rig 5-8 side,
    ray has positive component along a +Z glass normal, n_dot > 0 --
    the regime where ray_tracing_batch used to mirror the ray).
    """
    dm = np.diag([1.0, -1.0, -1.0]) if look_plus_z else np.eye(3)
    return Calibration(
        ext_par=Exterior(x0=x0, y0=0.0, z0=z0, dm=np.ascontiguousarray(dm)),
        int_par=Interior(xh=0.0, yh=0.0, cc=8.5858),
        glass_par=Glass(vec_x=glass_vec[0], vec_y=glass_vec[1],
                        vec_z=glass_vec[2]),
    )


@pytest.mark.parametrize("look_plus_z", [True, False])
@pytest.mark.parametrize("glass_vec", [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
@pytest.mark.parametrize(
    "mm",
    [
        MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),  # all-air (Ilmenau)
        MmNp(n1=1.0, n2=[1.49], d=[5.0], n3=1.33),  # glass/water tank
    ],
    ids=["air", "water"],
)
def test_ray_tracing_batch_matches_scalar(look_plus_z, glass_vec, mm):
    """ray_tracing_batch must equal scalar ray_tracing for every ray/glass
    sign combination.

    Regression test: the batch path forced a negative glass-normal
    component (the pre-fix scalar behaviour), mirroring outgoing rays
    whenever n_dot = glass . ray_dir > 0 -- i.e. cameras looking along
    +Z with a +Z glass normal (Ilmenau rig 5-8). All file-based fixtures
    (test_cavity: cams at -Z, glass (0,0,-125)) and the old unit tests
    only exercised n_dot < 0, where the forced sign happens to be
    correct, so the suite was blind to it. Production symptom: rig 5-8
    correspondences collapsed into a Z~=0 pancake.
    """
    z0 = -3000.0 if look_plus_z else 3000.0
    cal = _batch_test_cal(1400.0, z0, look_plus_z, glass_vec)
    xy = np.array(
        [[0.0, 0.0], [100.0, -50.0], [-200.0, 150.0]], dtype=np.float64
    )

    pos_b, dir_b = ray_tracing_batch(xy, cal, mm)

    for i in range(len(xy)):
        pos_s, dir_s = ray_tracing(
            xy[i, 0],
            xy[i, 1],
            cal.ext_par.dm,
            cal.ext_par.x0,
            cal.ext_par.y0,
            cal.ext_par.z0,
            cal.int_par.cc,
            cal.glass_par.vec_x,
            cal.glass_par.vec_y,
            cal.glass_par.vec_z,
            mm.n1,
            mm.n2[0],
            mm.n3,
            mm.d[0],
        )
        np.testing.assert_allclose(
            pos_b[i], np.asarray(pos_s), atol=1e-9,
            err_msg=f"pos mismatch (look_plus_z={look_plus_z}, "
                    f"glass={glass_vec}, mm={mm.n2[0]}/{mm.d[0]})",
        )
        np.testing.assert_allclose(
            dir_b[i], np.asarray(dir_s), atol=1e-9,
            err_msg=f"dir mismatch (look_plus_z={look_plus_z}, "
                    f"glass={glass_vec}, mm={mm.n2[0]}/{mm.d[0]})",
        )

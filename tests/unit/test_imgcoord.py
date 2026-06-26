import math
import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration, Exterior, Interior, Glass, AddedPar
from openptv2.algorithms.imgcoord import flat_image_coord, img_coord
from openptv2.algorithms.parameters import MmNp

EPS = 1e-5

def _air_mm():
    """Multimedia: all air (no refraction effect)."""
    return MmNp(nlay=1, n1=1.0, n2=[1.0, 0.0, 0.0], d=[1.0, 0.0, 0.0], n3=1.0)

def _centered_cal():
    """Camera at (0, 0, 40) looking straight down; cc=10; glass at z=20."""
    cal = Calibration()
    cal.ext_par = Exterior(
        x0=0.0, y0=0.0, z0=40.0,
        omega=0.0, phi=0.0, kappa=0.0,
        dm=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    cal.int_par = Interior(xh=0.0, yh=0.0, cc=10.0)
    cal.glass_par = Glass(vec_x=0.0, vec_y=0.0, vec_z=20.0)
    cal.added_par = AddedPar(k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, scx=1.0, she=0.0)
    return cal

def test_flat_image_coord_centered_cam():
    pos = np.array([10.0, 5.0, -20.0])
    cal = _centered_cal()
    mm = _air_mm()

    x, y = flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )

    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS
    assert abs(x - 2.0 * y) < EPS

def test_flat_image_coord_decentered_cam():
    angle = math.atan(0.5)
    pos = np.array([10.0, 0.0, -20.0])
    cal = Calibration()
    cal.ext_par = Exterior(
        x0=-20.0, y0=0.0, z0=40.0,
        omega=0.0, phi=-angle, kappa=0.0
    )
    cal.ext_par.compute_rotation_matrix()
    cal.int_par = Interior(xh=0.0, yh=0.0, cc=10.0)
    cal.glass_par = Glass(vec_x=0.0, vec_y=0.0, vec_z=20.0)
    cal.added_par = AddedPar()
    mm = _air_mm()

    x, y = flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )

    assert abs(x) < EPS
    assert abs(y) < EPS

def test_flat_image_coord_multilayer_on_axis():
    angle = math.atan(0.5)
    pos = np.array([10.0, 0.0, -20.0])
    cal = Calibration()
    cal.ext_par = Exterior(
        x0=-20.0, y0=0.0, z0=40.0,
        omega=0.0, phi=-angle, kappa=0.0
    )
    cal.ext_par.compute_rotation_matrix()
    cal.int_par = Interior(xh=0.0, yh=0.0, cc=10.0)
    cal.glass_par = Glass(vec_x=-20.0 * math.sin(angle), vec_y=0.0, vec_z=20.0 * math.cos(angle))
    cal.added_par = AddedPar()
    mm = MmNp(nlay=1, n1=1.0, n2=[1.5, 0.0, 0.0], d=[1.0, 0.0, 0.0], n3=1.0)

    x, y = flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )

    assert abs(x) < EPS
    assert abs(y) < EPS

def test_img_coord_shifted_sensor():
    pos = np.array([10.0, 5.0, -20.0])
    cal = _centered_cal()
    cal.int_par.xh = 0.1
    cal.int_par.yh = 0.1
    mm = _air_mm()

    x, y = img_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.ext_par.dm, cal.int_par.cc,
        cal.int_par.xh, cal.int_par.yh,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
        cal.added_par.p1, cal.added_par.p2, cal.added_par.scx, cal.added_par.she
    )

    assert abs(x - (10.0 / 6.0 + 0.1)) < EPS
    assert abs(x - 2.0 * (y - 0.1) - 0.1) < EPS

def test_img_coord_barrel_distortion():
    pos = np.array([10.0, 5.0, -20.0])
    cal = _centered_cal()
    cal.added_par.k1 = -0.01
    mm = _air_mm()

    x, y = img_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.ext_par.dm, cal.int_par.cc,
        cal.int_par.xh, cal.int_par.yh,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
        cal.added_par.p1, cal.added_par.p2, cal.added_par.scx, cal.added_par.she
    )

    r_sq = (10.0 / 6.0) ** 2 + (5.0 / 6.0) ** 2
    x_expected = (10.0 / 6.0) * (1.0 - 0.01 * r_sq)

    assert abs(x - x_expected) < EPS
    assert abs(x - 2.0 * y) < EPS

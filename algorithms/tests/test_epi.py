import numpy as np
import pytest

from algorithms.calibration import Calibration, Exterior, Interior, Glass, AddedPar
from algorithms.parameters import VolumePar, MmNp
from algorithms.epi import epi_mm_2d, epi_mm, find_candidate

EPS = 1e-5


def make_cal(x0, y0, z0, gx, gy, gz, cc=100.0):
    ext = Exterior(x0=x0, y0=y0, z0=z0)
    ext.dm = np.eye(3, dtype=np.float64)
    cal = Calibration(
        ext_par=ext,
        int_par=Interior(xh=0.0, yh=0.0, cc=cc),
        glass_par=Glass(vec_x=gx, vec_y=gy, vec_z=gz),
        added_par=AddedPar(),
    )
    return cal


def make_mm(n1, n2_0, n3, d0):
    return MmNp(nlay=1, n1=n1, n2=np.array([n2_0, 1.0, 1.0]), d=np.array([d0, 0.0, 0.0]), n3=n3)


def make_vpar(X_lay, Zmin_lay, Zmax_lay):
    return VolumePar(X_lay=X_lay, Zmin_lay=Zmin_lay, Zmax_lay=Zmax_lay)


def test_epi_mm_2d():
    cal = make_cal(0.0, 0.0, 100.0, 0.0, 0.0, 50.0)
    mmp = make_mm(1.0, 1.49, 1.33, 5.0)
    vpar = make_vpar((-250.0, 250.0), (-100.0, -100.0), (100.0, 100.0))

    out = epi_mm_2d(1.0, 10.0, cal, mmp, vpar)
    assert np.abs(out[0] - 0.85858163) < EPS
    assert np.abs(out[1] - 8.58581626) < EPS
    assert np.abs(out[2] - 0.0) < EPS

    out = epi_mm_2d(0.0, 0.0, cal, mmp, vpar)
    assert np.abs(out[0] - 0.0) < EPS
    assert np.abs(out[1] - 0.0) < EPS
    assert np.abs(out[2] - 0.0) < EPS


def test_epi_mm():
    cal1 = make_cal(10.0, 0.0, 100.0, 0.0, 0.0, 50.0)
    cal2 = make_cal(-10.0, 0.0, 100.0, 0.0, 0.0, 50.0)
    mmp = make_mm(1.0, 1.49, 1.33, 5.0)
    vpar = make_vpar((-250.0, 250.0), (-50.0, -50.0), (50.0, 50.0))

    xmin, ymin, xmax, ymax = epi_mm(10.0, 10.0, cal1, cal2, mmp, vpar)
    assert np.abs(xmin - 26.44927852) < EPS
    assert np.abs(xmax - 51.60078764) < EPS
    assert np.abs(ymin - 10.08218486) < EPS
    assert np.abs(ymax - 10.04378909) < EPS


def test_epi_mm_perpendicular():
    cal1 = make_cal(0.0, 0.0, 100.0, 0.0, 0.0, 50.0)
    cal2 = make_cal(100.0, 0.0, 0.0, 0.0, 0.0, 50.0)
    mmp = make_mm(1.0, 1.0, 1.0, 1.0)
    vpar = make_vpar((-100.0, 100.0), (-100.0, -100.0), (100.0, 100.0))

    xmin, ymin, xmax, ymax = epi_mm(0.0, 0.0, cal1, cal2, mmp, vpar)
    assert np.abs(xmin + 100.0) < EPS
    assert np.abs(xmax - 100.0) < EPS
    assert np.abs(ymin - 0.0) < EPS
    assert np.abs(ymax - 0.0) < EPS

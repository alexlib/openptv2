import numpy as np
import pytest

from algorithms.calibration import Calibration, Exterior, Interior, Glass, AddedPar
from algorithms.parameters import VolumePar, ControlPar, MmNp
from algorithms.epi import epi_mm_2d, epi_mm, epipolar_curve, find_candidate

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


def _set_angles(cal, omega, phi, kappa):
    """Set angles and recompute rotation matrix."""
    cal.ext_par.omega = omega
    cal.ext_par.phi = phi
    cal.ext_par.kappa = kappa
    cal.ext_par.dm = cal.ext_par.compute_rotation_matrix()


def test_epipolar_curve_central_point():
    """Central point translates to central point for opposing cameras."""
    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    orig_cal = Calibration.from_file(ori_tmpl.format(1), add_file)
    proj_cal = Calibration.from_file(ori_tmpl.format(3), add_file)

    _set_angles(orig_cal, 0.0, -np.pi / 4.0, 0.0)
    _set_angles(proj_cal, 0.0, 3 * np.pi / 4.0, 0.0)

    cpar = ControlPar.from_file("test_data/corresp/control.par")
    cpar.mm.n1 = 1.0
    cpar.mm.n2[:] = 1.0
    cpar.mm.n3 = 1.0
    cpar.mm.d[:] = 1.0

    vpar = VolumePar.from_file("test_data/corresp/criteria.par")
    vpar.Zmin_lay = np.array([-10.0, -10.0])
    vpar.Zmax_lay = np.array([10.0, 10.0])

    mid = np.array([cpar.imx / 2.0, cpar.imy / 2.0])
    line = epipolar_curve(mid, orig_cal, proj_cal, 5, cpar, vpar)

    assert line.shape == (5, 2)
    assert np.all(np.abs(line - mid) < 1e-6)


def test_epipolar_curve_equatorial():
    """Off-center point draws a monotonic epipolar curve."""
    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    orig_cal = Calibration.from_file(ori_tmpl.format(1), add_file)
    proj_cal = Calibration.from_file(ori_tmpl.format(3), add_file)

    _set_angles(orig_cal, 0.0, -np.pi / 4.0, 0.0)
    _set_angles(proj_cal, 0.0, 3 * np.pi / 4.0, 0.0)

    cpar = ControlPar.from_file("test_data/corresp/control.par")
    cpar.mm.n1 = 1.0
    cpar.mm.n2[:] = 1.0
    cpar.mm.n3 = 1.0
    cpar.mm.d[:] = 1.0

    vpar = VolumePar.from_file("test_data/corresp/criteria.par")
    vpar.Zmin_lay = np.array([-10.0, -10.0])
    vpar.Zmax_lay = np.array([10.0, 10.0])

    mid = np.array([cpar.imx / 2.0, cpar.imy / 2.0])
    line = epipolar_curve(mid - np.array([100.0, 0.0]), orig_cal, proj_cal,
                          5, cpar, vpar)

    # x-coords should be monotonically decreasing
    np.testing.assert_array_equal(np.argsort(line[:, 0]), np.arange(5)[::-1])
    # y-coords stay on the equator
    assert np.all(np.abs(line[:, 1] - mid[1]) < 1e-6)


def _has_optv():
    import os
    if os.environ.get("OPENPTV_ENGINE") == "python":
        return False
    try:
        from optv.epipolar import epipolar_curve  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_optv(), reason="optv not installed")
def test_epipolar_curve_parity():
    """Compare Python epipolar_curve against Cython binding."""
    from optv.calibration import Calibration as CCalibration
    from optv.parameters import ControlParams as CControlParams
    from optv.parameters import VolumeParams as CVolumeParams
    from optv.epipolar import epipolar_curve as c_epipolar_curve

    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    # Python
    py_orig = Calibration.from_file(ori_tmpl.format(1), add_file)
    py_proj = Calibration.from_file(ori_tmpl.format(3), add_file)
    _set_angles(py_orig, 0.0, -np.pi / 4.0, 0.0)
    _set_angles(py_proj, 0.0, 3 * np.pi / 4.0, 0.0)
    py_cpar = ControlPar.from_file("test_data/corresp/control.par")
    py_cpar.mm.n1 = 1.0
    py_cpar.mm.n2[:] = 1.0
    py_cpar.mm.n3 = 1.0
    py_cpar.mm.d[:] = 1.0
    py_vpar = VolumePar.from_file("test_data/corresp/criteria.par")
    py_vpar.Zmin_lay = np.array([-10.0, -10.0])
    py_vpar.Zmax_lay = np.array([10.0, 10.0])

    # Cython
    c_orig = CCalibration()
    c_orig.from_file(ori_tmpl.format(1).encode(), add_file.encode())
    c_proj = CCalibration()
    c_proj.from_file(ori_tmpl.format(3).encode(), add_file.encode())
    c_orig.set_angles(np.r_[0.0, -np.pi / 4.0, 0.0])
    c_proj.set_angles(np.r_[0.0, 3 * np.pi / 4.0, 0.0])
    c_cpar = CControlParams(4)
    c_cpar.read_control_par("test_data/corresp/control.par")
    mult = c_cpar.get_multimedia_params()
    mult.set_n1(1.0)
    mult.set_layers(np.array([1.0]), np.array([1.0]))
    mult.set_n3(1.0)
    c_vpar = CVolumeParams()
    c_vpar.read_volume_par("test_data/corresp/criteria.par")
    c_vpar.set_Zmin_lay([-10, -10])
    c_vpar.set_Zmax_lay([10, 10])

    mid = np.array([py_cpar.imx / 2.0, py_cpar.imy / 2.0])

    for offset in [np.array([0.0, 0.0]), np.array([-100.0, 0.0]),
                   np.array([50.0, -30.0])]:
        pt = mid + offset
        py_line = epipolar_curve(pt, py_orig, py_proj, 10, py_cpar, py_vpar)
        c_line = c_epipolar_curve(pt, c_orig, c_proj, 10, c_cpar, c_vpar)
        np.testing.assert_allclose(py_line, c_line, atol=1e-4,
                                   err_msg=f"Mismatch at offset {offset}")


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

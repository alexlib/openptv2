from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar
from openptv2.algorithms.trafo import (
    correct_brown_affin,
    dist_to_flat,
    distort_brown_affin,
    flat_to_dist,
    metric_to_pixel,
    old_metric_to_pixel,
    old_pixel_to_metric,
    pixel_to_metric,
)

EPS = 1e-6


def test_old_metric_to_pixel():
    xc = 0.0
    yc = 0.0
    imx = 1024
    imy = 1008
    pix_x = 0.010
    pix_y = 0.010
    field = 0

    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    assert abs(xp - 512.0) < EPS
    assert abs(yp - 504.0) < EPS

    xc = 1.0
    yc = 0.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    assert abs(xp - 612.0) < EPS
    assert abs(yp - 504.0) < EPS

    xc = 0.0
    yc = -1.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    assert abs(xp - 512.0) < EPS
    assert abs(yp - 604.0) < EPS


def test_metric_to_pixel():
    xc = 0.0
    yc = 0.0
    cpar = ControlPar(imx=1024, imy=1008, pix_x=0.01, pix_y=0.01, chfield=0)

    xp, yp = metric_to_pixel(
        xc, yc, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    assert abs(xp - 512.0) < EPS
    assert abs(yp - 504.0) < EPS

    xc = 1.0
    yc = 0.0
    xp, yp = metric_to_pixel(
        xc, yc, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    assert abs(xp - 612.0) < EPS
    assert abs(yp - 504.0) < EPS

    xc = 0.0
    yc = -1.0
    xp, yp = metric_to_pixel(
        xc, yc, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    assert abs(xp - 512.0) < EPS
    assert abs(yp - 604.0) < EPS


def test_old_pixel_to_metric():
    xc = 0.0
    yc = 0.0
    imx = 1024
    imy = 1008
    pix_x = 0.010
    pix_y = 0.010
    field = 0

    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    xc1, yc1 = old_pixel_to_metric(xp, yp, imx, imy, pix_x, pix_y, field)
    assert abs(xc1 - xc) < EPS
    assert abs(yc1 - yc) < EPS

    xc = 1.0
    yc = 0.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    xc1, yc1 = old_pixel_to_metric(xp, yp, imx, imy, pix_x, pix_y, field)
    assert abs(xc1 - xc) < EPS
    assert abs(yc1 - yc) < EPS

    xc = 0.0
    yc = -1.0
    xp, yp = old_metric_to_pixel(xc, yc, imx, imy, pix_x, pix_y, field)
    xc1, yc1 = old_pixel_to_metric(xp, yp, imx, imy, pix_x, pix_y, field)
    assert abs(xc1 - xc) < EPS
    assert abs(yc1 - yc) < EPS


def test_pixel_to_metric():
    xc = 0.0
    yc = 0.0
    cpar = ControlPar(imx=1024, imy=1008, pix_x=0.01, pix_y=0.01, chfield=0)

    xp, yp = metric_to_pixel(
        xc, yc, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    xc1, yc1 = pixel_to_metric(
        xp, yp, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    assert abs(xc1 - xc) < EPS
    assert abs(yc1 - yc) < EPS

    xc = 1.0
    yc = 0.0
    xp, yp = metric_to_pixel(
        xc, yc, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    xc1, yc1 = pixel_to_metric(
        xp, yp, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    assert abs(xc1 - xc) < EPS
    assert abs(yc1 - yc) < EPS

    xc = 0.0
    yc = -1.0
    xp, yp = metric_to_pixel(
        xc, yc, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    xc1, yc1 = pixel_to_metric(
        xp, yp, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )
    assert abs(xc1 - xc) < EPS
    assert abs(yc1 - yc) < EPS


def test_shear():
    x = 1.0
    y = 1.0
    xp, yp = distort_brown_affin(x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    assert abs(xp - 0.158529) < EPS
    assert abs(yp - 0.540302) < EPS


def test_shear_round_trip():
    x = -1.0
    y = 10.0

    k1, k2, k3, p1, p2, scx, she = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1
    xp, yp = distort_brown_affin(x, y, k1, k2, k3, p1, p2, scx, she)
    x1, y1 = correct_brown_affin(xp, yp, k1, k2, k3, p1, p2, scx, she)

    assert abs(x1 - x) < EPS
    assert abs(y1 - y) < EPS

    x = 0.5
    y = -5.0
    xp, yp = distort_brown_affin(x, y, k1, k2, k3, p1, p2, scx, she)
    x1, y1 = correct_brown_affin(xp, yp, k1, k2, k3, p1, p2, scx, she)

    assert abs(x1 - x) < EPS
    assert abs(y1 - y) < EPS


def test_dummy_distortion_round_trip():
    x = 1.0
    y = 1.0
    k1, k2, k3, p1, p2, scx, she = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0

    xres, yres = distort_brown_affin(x, y, k1, k2, k3, p1, p2, scx, she)
    xres, yres = correct_brown_affin(xres, yres, k1, k2, k3, p1, p2, scx, she)

    assert abs(xres - x) < EPS
    assert abs(yres - y) < EPS


def test_radial_distortion_round_trip():
    x = 1.0
    y = 1.0
    iter_eps = 0.05
    k1, k2, k3, p1, p2, scx, she = 0.05, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0

    xres, yres = distort_brown_affin(x, y, k1, k2, k3, p1, p2, scx, she)
    xres, yres = correct_brown_affin(xres, yres, k1, k2, k3, p1, p2, scx, she)

    assert abs(xres - x) < iter_eps
    assert abs(yres - y) < iter_eps


def test_dist_flat_round_trip():
    from openptv2.algorithms.calibration import AddedPar

    x = 10.0
    y = 10.0
    iter_eps = 1e-3

    cal = Calibration()
    cal.int_par.xh = 1.5
    cal.int_par.yh = 1.5
    cal.int_par.cc = 60.0
    cal.added_par = AddedPar(
        k1=0.0005, k2=0.0, k3=0.0, p1=0.0, p2=0.0, scx=1.0, she=0.0
    )

    ap = cal.added_par
    xres, yres = flat_to_dist(
        x,
        y,
        cal.int_par.xh,
        cal.int_par.yh,
        ap.k1,
        ap.k2,
        ap.k3,
        ap.p1,
        ap.p2,
        ap.scx,
        ap.she,
    )
    xres, yres = dist_to_flat(
        xres,
        yres,
        cal.int_par.xh,
        cal.int_par.yh,
        ap.k1,
        ap.k2,
        ap.k3,
        ap.p1,
        ap.p2,
        ap.scx,
        ap.she,
        iter_eps,
    )

    assert abs(xres - x) < iter_eps
    assert abs(yres - y) < iter_eps

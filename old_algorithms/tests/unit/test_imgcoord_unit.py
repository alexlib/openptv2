"""Unit tests for imgcoord module.

Each function is tested with explicit known inputs and analytically-derivable
expected outputs, following the pattern of lib/tests/check_imgcoord.c.
"""

import math

import numpy as np
import pytest

from algorithms.calibration import Calibration
from algorithms.imgcoord import flat_image_coord, img_coord
from algorithms.parameters import MultimediaPar

EPS = 1e-5


def _air_mm():
    """Multimedia: all air (no refraction effect)."""
    return MultimediaPar(nlay=1, n1=1.0, n2=[1.0], d=[1.0], n3=1.0)


def _centered_cal():
    """Camera at (0, 0, 40) looking straight down; cc=10; glass at z=20."""
    cal = Calibration()
    cal.set_pos([0.0, 0.0, 40.0])
    cal.set_angles([0.0, 0.0, 0.0])         # identity rotation matrix
    cal.set_primary_point(np.array([0.0, 0.0, 10.0]))  # xh=0, yh=0, cc=10
    cal.glass_par = np.array([0.0, 0.0, 20.0])
    return cal


def test_flat_image_coord_centered_cam():
    """Camera on z-axis, point off-axis: x = 10/6, y = 5/6.

    Geometry from check_imgcoord.c test_flat_centered_cam.
    Camera at (0,0,40), point at (10,5,-20), cc=10 → ratio = cc/(Δz) = 10/60.
    """
    pos = np.array([10.0, 5.0, -20.0])
    cal = _centered_cal()
    mm = _air_mm()

    x, y = flat_image_coord(pos, cal, mm)

    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS
    assert abs(x - 2.0 * y) < EPS   # x/y = 10/5 = 2


def test_flat_image_coord_decentered_cam():
    """Camera axis passes through the measured point → projected to (0, 0).

    Geometry from check_imgcoord.c test_flat_decentered_cam.
    Camera at (-20, 0, 40) tilted by phi=-atan(0.5) so it looks directly at
    the point (10, 0, -20).
    """
    angle = math.atan(0.5)
    pos = np.array([10.0, 0.0, -20.0])
    cal = Calibration()
    cal.set_pos([-20.0, 0.0, 40.0])
    cal.set_angles([0.0, -angle, 0.0])
    cal.set_primary_point(np.array([0.0, 0.0, 10.0]))
    cal.glass_par = np.array([0.0, 0.0, 20.0])
    mm = _air_mm()

    x, y = flat_image_coord(pos, cal, mm)

    assert abs(x) < EPS
    assert abs(y) < EPS


def test_flat_image_coord_multilayer_on_axis():
    """Glass layer does not shift a point on the optical axis.

    Geometry from check_imgcoord.c test_flat_multilayer.
    Even with n2=1.5, if the ray hits the glass plane at normal incidence
    (axis-aligned), refraction has no lateral effect.
    """
    angle = math.atan(0.5)
    pos = np.array([10.0, 0.0, -20.0])
    cal = Calibration()
    cal.set_pos([-20.0, 0.0, 40.0])
    cal.set_angles([0.0, -angle, 0.0])
    cal.set_primary_point(np.array([0.0, 0.0, 10.0]))
    # Glass normal parallel to the camera-to-point line
    cal.glass_par = np.array([-20.0 * math.sin(angle), 0.0, 20.0 * math.cos(angle)])
    mm = MultimediaPar(nlay=1, n1=1.0, n2=[1.5], d=[1.0], n3=1.0)

    x, y = flat_image_coord(pos, cal, mm)

    assert abs(x) < EPS
    assert abs(y) < EPS


def test_img_coord_shifted_sensor():
    """Internal shifts (xh, yh) are added to the projected coordinate.

    Geometry from check_imgcoord.c test_shifted_sensor.
    Flat result is (10/6, 5/6); adding (xh=0.1, yh=0.1) gives (10/6+0.1, 5/6+0.1).
    """
    pos = np.array([10.0, 5.0, -20.0])
    cal = Calibration()
    cal.set_pos([0.0, 0.0, 40.0])
    cal.set_angles([0.0, 0.0, 0.0])
    cal.set_primary_point(np.array([0.1, 0.1, 10.0]))  # xh=0.1, yh=0.1
    cal.glass_par = np.array([0.0, 0.0, 20.0])
    mm = _air_mm()

    x, y = img_coord(pos, cal, mm)

    assert abs(x - (10.0 / 6.0 + 0.1)) < EPS
    assert abs(y - (5.0 / 6.0 + 0.1)) < EPS


def test_img_coord_barrel_distortion():
    """Barrel distortion k1=-0.01 scales the projected point by (1 + k1*r^2).

    Geometry from check_imgcoord.c test_distorted_centered_cam.
    Flat result (10/6, 5/6); r^2 = (10/6)^2 + (5/6)^2 = 125/36.
    x_dist = (10/6)*(1 - 0.01*125/36), and x_dist = 2*y_dist.
    """
    pos = np.array([10.0, 5.0, -20.0])
    cal = _centered_cal()
    # k1=-0.01, scx=1, she=0, rest zero
    cal.added_par = np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    mm = _air_mm()

    x, y = img_coord(pos, cal, mm)

    r_sq = (10.0 / 6.0) ** 2 + (5.0 / 6.0) ** 2   # = 125/36
    x_expected = (10.0 / 6.0) * (1.0 - 0.01 * r_sq)

    assert abs(x - x_expected) < EPS
    assert abs(x - 2.0 * y) < EPS

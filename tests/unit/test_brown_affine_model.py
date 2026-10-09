"""The Brown + affine image model, checked against the Tcl/Tk 3DPTV original.

Reference: 3dptv_tcltk/src_c/trafo.c:distort_brown_affin (identical in liboptv
until 2016)::

    x' = scx * x_d - sin(she) * y_d
    y' = cos(she) * y_d

``scx`` is the x-only scale factor of the El-Hakim / Gruen-Beyer affine pair
(pixel-clock vs frame-grabber mismatch), so y' does not depend on it.  A 2025
liboptv rewrite applied ``scx`` to both axes, which makes it a copy of ``cc``
and left ``correct_brown_affine_exact`` inverting a different model; openptv2
inherited both.  The orientation Jacobian (``orient``: dy/dscx = 0) always
assumed the original.

One deliberate departure from the C: both decentering terms are evaluated at
the same undistorted point (Brown 1971).  The C updated x in place before the
y line; the difference is second order in the distortion.

Every implementation of the forward and inverse model is checked here, since
the formula is inlined in several kernels.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import (
    flat_image_coord_batch,
    img_coord,
    img_coord_batch,
)
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.trafo import (
    correct_brown_affin,
    correct_brown_affine_batch,
    dist_to_flat,
    distort_brown_affin,
    distort_brown_affine_batch,
    flat_to_dist,
    metric_to_pixel,
)

DIST = (1e-3, 1e-5, 0.0, 2e-4, -3e-4)  # k1, k2, k3, p1, p2
AFFINE = [(1.0, 0.0), (1.02, 0.0), (1.0, 0.01), (0.97, -0.02), (1.02, 0.01)]
POINTS = [(3.0, -2.0), (-4.5, 1.2), (0.7, 5.1), (-2.2, -3.3)]
CAL_DIR = Path(__file__).resolve().parents[2] / "test_data" / "track" / "cal"


def ref_distort(x, y, k1, k2, k3, p1, p2, scx, she):
    """3DPTV distort_brown_affin, decentering evaluated at one point."""
    r2 = x * x + y * y
    rad = k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x + x * rad + p1 * (r2 + 2 * x * x) + 2 * p2 * x * y
    yd = y + y * rad + p2 * (r2 + 2 * y * y) + 2 * p1 * x * y
    return scx * xd - math.sin(she) * yd, math.cos(she) * yd


@pytest.mark.parametrize("scx,she", AFFINE)
def test_forward_matches_tcltk_model(scx, she):
    batch = np.asarray(distort_brown_affine_batch(np.array(POINTS), *DIST, scx, she))
    for (x, y), b in zip(POINTS, batch):
        ref = ref_distort(x, y, *DIST, scx, she)
        assert np.allclose(
            distort_brown_affin(x, y, *DIST, scx, she), ref, atol=1e-12, rtol=0
        )
        assert np.allclose(b, ref, atol=1e-12, rtol=0)
        # flat_to_dist adds the principal point first
        assert np.allclose(
            flat_to_dist(x - 0.1, y + 0.05, 0.1, -0.05, *DIST, scx, she),
            ref,
            atol=1e-12,
            rtol=0,
        )


def test_scx_scales_x_only():
    x, y = 3.0, -2.0
    a = distort_brown_affin(x, y, *DIST, 1.0, 0.01)
    b = distort_brown_affin(x, y, *DIST, 1.05, 0.01)
    assert b[1] == pytest.approx(a[1], abs=1e-15)
    assert b[0] != pytest.approx(a[0], abs=1e-6)


@pytest.mark.parametrize("scx,she", AFFINE)
def test_inverses_round_trip(scx, she):
    from openptv2.algorithms.track_kernels_pixel import _dist_to_flat_out

    xh, yh = 0.1, -0.05
    out = np.zeros(2)
    dist = np.array([ref_distort(x, y, *DIST, scx, she) for x, y in POINTS])
    batch = np.asarray(correct_brown_affine_batch(dist, *DIST, scx, she))
    for (x, y), (xd, yd), b in zip(POINTS, dist, batch):
        # correct_brown_affin and its batch stop at a relative change of 1e-8
        assert np.allclose(
            correct_brown_affin(xd, yd, *DIST, scx, she), (x, y), atol=1e-7, rtol=0
        )
        assert np.allclose(b, (x, y), atol=1e-7, rtol=0)
        # dist_to_flat / the tracking kernel also remove the principal point
        flat = dist_to_flat(xd, yd, xh, yh, *DIST, scx, she, 1e-12)
        assert np.allclose(flat, (x - xh, y - yh), atol=1e-9, rtol=0)
        _dist_to_flat_out(xd, yd, xh, yh, *DIST, scx, she, 1e-12, out)
        assert np.allclose(out, (x - xh, y - yh), atol=1e-9, rtol=0)


@pytest.mark.parametrize("scx,she", [(1.0, 0.0), (1.02, 0.01), (0.97, -0.02)])
def test_projection_paths_agree(scx, she):
    """img_coord, img_coord_batch and the three tracking-kernel projections."""
    from openptv2.algorithms.track import _point_to_pixel_fast
    from openptv2.algorithms.track_kernels import pack_cal_array, pack_mmlut
    from openptv2.algorithms.track_kernels_geom import point_to_pixel_fast
    from openptv2.algorithms.track_kernels_pixel import _point_to_pixel_out

    cal = Calibration()
    cal.from_file(str(CAL_DIR / "cam1.tif.ori"), str(CAL_DIR / "cam1.tif.addpar"))
    ap = cal.added_par
    ap.k1, ap.k2, ap.k3, ap.p1, ap.p2 = DIST
    ap.scx, ap.she = scx, she
    mm = MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)
    cpar = ControlPar(
        num_cams=1,
        imx=1280,
        imy=1024,
        pix_x=0.012,
        pix_y=0.012,
        mm=mm,
        chfield=0,
        tiff_flag=1,
        hp_flag=1,
        allCam_flag=0,
        img_base_name=[""],
        cal_img_base_name=[""],
    )
    pos = np.array([[0.0, 0.0, 0.0], [20.0, -15.0, 10.0], [-30.0, 25.0, -5.0]])
    batch = np.asarray(img_coord_batch(pos, cal, mm))
    pc = pack_cal_array(cal, mm)
    lut = pack_mmlut(cal)
    out = np.zeros(2)
    half = (cpar.imx / 2, cpar.imy / 2, 1 / cpar.pix_x, 1 / cpar.pix_y)
    flat = np.asarray(flat_image_coord_batch(pos, cal, mm))
    for p, b, (fx, fy) in zip(pos, batch, flat):
        ref = ref_distort(fx + cal.int_par.xh, fy + cal.int_par.yh, *DIST, scx, she)
        assert np.allclose(img_coord(p, cal, mm), ref, atol=1e-9, rtol=0)
        assert np.allclose(b, ref, atol=1e-9, rtol=0)
        ref_px = metric_to_pixel(*ref, cpar)
        assert np.allclose(
            _point_to_pixel_fast(
                p, cal, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, 0, mm
            ),
            ref_px,
            atol=1e-6,
            rtol=0,
        )
        assert np.allclose(
            point_to_pixel_fast(p, pc, *lut, 1, *half, 0), ref_px, atol=1e-6, rtol=0
        )
        _point_to_pixel_out(p, pc, *lut, 1, *half, 0, out)
        assert np.allclose(out, ref_px, atol=1e-6, rtol=0)


@pytest.mark.parametrize("she", [0.0, 0.01, -0.02])
def test_matches_liboptv_where_models_agree(she):
    """With scx = 1 the 2025 liboptv model coincides with the original."""
    T = pytest.importorskip("optv.transforms")
    from optv.calibration import Calibration as OptvCalibration

    oc = OptvCalibration()
    oc.set_radial_distortion(np.array(DIST[:3]))
    oc.set_decentering(np.array(DIST[3:]))
    oc.set_affine_trans(np.array([1.0, she]))
    pts = np.array(POINTS)
    for p, rd, rf in zip(
        pts, T.distort_arr_brown_affine(pts, oc), T.correct_arr_brown_affine(pts, oc)
    ):
        assert np.allclose(
            distort_brown_affin(*p, *DIST, 1.0, she), rd, atol=1e-9, rtol=0
        )
        assert np.allclose(
            dist_to_flat(*p, 0.0, 0.0, *DIST, 1.0, she, 1e-12), rf, atol=1e-7, rtol=0
        )

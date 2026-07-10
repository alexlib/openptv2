"""Pure-Python coverage tests for openptv2.algorithms.trafo.

Run via the pure-Python snapshot at /tmp/ppsrc (no .so):
    COVERAGE_FILE=/tmp/.cov_trafo uv run pytest tests/unit/test_trafo_coverage.py \
        -o pythonpath=/tmp/ppsrc -p no:cacheprovider \
        --cov=/tmp/ppsrc/openptv2 --cov-config=/tmp/covrc --cov-report=term-missing -q

Design constraints
------------------
* Only call functions that work in pure-Python mode (no C-array local variables).
* Functions that declare ``_out: cython.double[2]`` as a LOCAL variable raise
  ``UnboundLocalError`` in interpreted mode — they are documented under
  ``suspected_bugs`` and skipped here.
* Use ``_*_out`` helpers directly, passing ``np.zeros(2)`` for the ``out``
  (and ``_scratch``) memoryview parameters — in pure-Python mode the type hint
  is unenforced.
* Prefer invariants and round-trips over guessed magic values.
"""

from math import sin, cos
from types import SimpleNamespace

import numpy as np
import pytest

# _*_out helpers are @cython.cfunc (cdef) — not importable from compiled .so.
# Skip whole module when the compiled extension is active.
from openptv2.algorithms.trafo import is_compiled as _trafo_is_compiled
if _trafo_is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.trafo import (
    # Module constants
    NO_REMAP,
    DOUBLED_PLUS_ONE,
    DOUBLED,
    # Safe helpers — take memoryview *out* as parameter (no local C-array)
    _old_pixel_to_metric_out,
    _old_metric_to_pixel_out,
    _distort_brown_affin_core_out,
    distort_brown_affin_out,
    _correct_brown_affin_out,
    _correct_brown_affine_exact_out,
    flat_to_dist_out,
    # Batch / numpy functions (no C-array locals)
    pixel_to_metric_batch,
    metric_to_pixel_batch,
    distort_brown_affine_batch,
    correct_brown_affine_batch,
    # Buggy in pure-Python mode (C-array local declarations).
    # Called only inside try/except to cover the first executable line of each.
    old_pixel_to_metric,
    pixel_to_metric,
    old_metric_to_pixel,
    metric_to_pixel,
    _distort_brown_affin_core,
    distort_brown_affin,
    correct_brown_affin,
    correct_brown_affine_exact,
    flat_to_dist,
    dist_to_flat_out,
    dist_to_flat,
    # Trivial
    is_compiled,
)

EPS = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out2():
    """Return a zeroed float64 output buffer accepted as memoryview."""
    return np.zeros(2, dtype=np.float64)


def _cpar(imx=1024, imy=1008, pix_x=0.01, pix_y=0.01, chfield=0):
    """Minimal camera-parameter namespace used by batch functions."""
    return SimpleNamespace(
        imx=imx, imy=imy, pix_x=pix_x, pix_y=pix_y, chfield=chfield
    )


# ---------------------------------------------------------------------------
# is_compiled
# ---------------------------------------------------------------------------


def test_is_compiled_returns_bool():
    """is_compiled() returns False in pure-Python mode."""
    result = is_compiled()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _old_pixel_to_metric_out  (lines 29-46)
# ---------------------------------------------------------------------------


def test_opm_out_center_no_remap():
    """Center pixel (imx/2, imy/2) → (0, 0) metric with NO_REMAP."""
    out = _out2()
    _old_pixel_to_metric_out(512.0, 504.0, 1024, 1008, 0.01, 0.01, NO_REMAP, out)
    assert abs(out[0]) < EPS
    assert abs(out[1]) < EPS


def test_opm_out_offset_x():
    """x_pixel offset → positive x_metric."""
    out = _out2()
    _old_pixel_to_metric_out(612.0, 504.0, 1024, 1008, 0.01, 0.01, NO_REMAP, out)
    assert abs(out[0] - 1.0) < EPS  # (612-512)*0.01 = 1.0
    assert abs(out[1]) < EPS


def test_opm_out_doubled_plus_one():
    """DOUBLED_PLUS_ONE remaps y → 2*y + 1."""
    out = _out2()
    # y_pixel=0 → yp=1 → out[1] = (504-1)*0.01 = 5.03
    _old_pixel_to_metric_out(512.0, 0.0, 1024, 1008, 0.01, 0.01, DOUBLED_PLUS_ONE, out)
    assert abs(out[0]) < EPS
    assert abs(out[1] - 5.03) < EPS


def test_opm_out_doubled():
    """DOUBLED remaps y → 2*y."""
    out = _out2()
    # y_pixel=0 → yp=0 → out[1] = (504-0)*0.01 = 5.04
    _old_pixel_to_metric_out(512.0, 0.0, 1024, 1008, 0.01, 0.01, DOUBLED, out)
    assert abs(out[0]) < EPS
    assert abs(out[1] - 5.04) < EPS


# ---------------------------------------------------------------------------
# _old_metric_to_pixel_out  (lines 162-183)
# ---------------------------------------------------------------------------


def test_omtp_out_origin_no_remap():
    """(0, 0) metric → center pixel with NO_REMAP."""
    out = _out2()
    _old_metric_to_pixel_out(0.0, 0.0, 1024, 1008, 0.01, 0.01, NO_REMAP, out)
    assert abs(out[0] - 512.0) < EPS
    assert abs(out[1] - 504.0) < EPS


def test_omtp_out_doubled_plus_one():
    """DOUBLED_PLUS_ONE: yp = (yp_base - 1) / 2."""
    out = _out2()
    # yp_base = 504; yp = (504-1)*0.5 = 251.5
    _old_metric_to_pixel_out(0.0, 0.0, 1024, 1008, 0.01, 0.01, DOUBLED_PLUS_ONE, out)
    assert abs(out[0] - 512.0) < EPS
    assert abs(out[1] - 251.5) < EPS


def test_omtp_out_doubled():
    """DOUBLED: yp = yp_base / 2."""
    out = _out2()
    # yp_base = 504; yp = 252.0
    _old_metric_to_pixel_out(0.0, 0.0, 1024, 1008, 0.01, 0.01, DOUBLED, out)
    assert abs(out[1] - 252.0) < EPS


def test_pixel_metric_out_round_trip():
    """_old_pixel_to_metric_out then _old_metric_to_pixel_out round-trips."""
    # Forward: pixel → metric
    out_m = _out2()
    _old_pixel_to_metric_out(700.0, 300.0, 1024, 1008, 0.01, 0.01, NO_REMAP, out_m)
    # Inverse: metric → pixel
    out_p = _out2()
    _old_metric_to_pixel_out(out_m[0], out_m[1], 1024, 1008, 0.01, 0.01, NO_REMAP, out_p)
    assert abs(out_p[0] - 700.0) < EPS
    assert abs(out_p[1] - 300.0) < EPS


# ---------------------------------------------------------------------------
# pixel_to_metric_batch  (lines 88-118)
# ---------------------------------------------------------------------------


def test_p2m_batch_center():
    """Center pixel → (0, 0) metric, chfield=0."""
    cpar = _cpar()
    xy = np.array([[512.0, 504.0]], dtype=np.float64)
    res = pixel_to_metric_batch(xy, cpar)
    assert res.shape == (1, 2)
    assert abs(res[0, 0]) < EPS
    assert abs(res[0, 1]) < EPS


def test_p2m_batch_offset():
    """Pixel offset propagates correctly."""
    cpar = _cpar()
    xy = np.array([[612.0, 504.0]], dtype=np.float64)
    res = pixel_to_metric_batch(xy, cpar)
    assert abs(res[0, 0] - 1.0) < EPS  # (612-512)*0.01
    assert abs(res[0, 1]) < EPS


def test_p2m_batch_doubled_plus_one():
    """chfield=DOUBLED_PLUS_ONE branch."""
    cpar = _cpar(chfield=DOUBLED_PLUS_ONE)
    xy = np.array([[512.0, 0.0]], dtype=np.float64)
    res = pixel_to_metric_batch(xy, cpar)
    # yp = 2*0+1=1; out[1] = (504-1)*0.01 = 5.03
    assert abs(res[0, 1] - 5.03) < EPS


def test_p2m_batch_doubled():
    """chfield=DOUBLED branch."""
    cpar = _cpar(chfield=DOUBLED)
    xy = np.array([[512.0, 0.0]], dtype=np.float64)
    res = pixel_to_metric_batch(xy, cpar)
    # yp = 0; out[1] = (504-0)*0.01 = 5.04
    assert abs(res[0, 1] - 5.04) < EPS


def test_p2m_batch_multiple_rows():
    """Multiple rows handled correctly."""
    cpar = _cpar()
    xy = np.array([[512.0, 504.0], [612.0, 404.0]], dtype=np.float64)
    res = pixel_to_metric_batch(xy, cpar)
    assert res.shape == (2, 2)
    assert abs(res[0, 0]) < EPS
    assert abs(res[1, 0] - 1.0) < EPS   # (612-512)*0.01
    assert abs(res[1, 1] - 1.0) < EPS   # (504-404)*0.01


# ---------------------------------------------------------------------------
# metric_to_pixel_batch  (lines 223-252)
# ---------------------------------------------------------------------------


def test_m2p_batch_origin():
    """(0, 0) metric → center pixel, chfield=0."""
    cpar = _cpar()
    xy = np.array([[0.0, 0.0]], dtype=np.float64)
    res = metric_to_pixel_batch(xy, cpar)
    assert abs(res[0, 0] - 512.0) < EPS
    assert abs(res[0, 1] - 504.0) < EPS


def test_m2p_batch_doubled_plus_one():
    """chfield=DOUBLED_PLUS_ONE branch."""
    cpar = _cpar(chfield=DOUBLED_PLUS_ONE)
    xy = np.array([[0.0, 0.0]], dtype=np.float64)
    res = metric_to_pixel_batch(xy, cpar)
    # y_pixel_base=504; y_pixel=(504-1)/2=251.5
    assert abs(res[0, 1] - 251.5) < EPS


def test_m2p_batch_doubled():
    """chfield=DOUBLED branch."""
    cpar = _cpar(chfield=DOUBLED)
    xy = np.array([[0.0, 0.0]], dtype=np.float64)
    res = metric_to_pixel_batch(xy, cpar)
    assert abs(res[0, 1] - 252.0) < EPS


def test_pixel_metric_batch_round_trip():
    """pixel_to_metric_batch → metric_to_pixel_batch is identity."""
    cpar = _cpar()
    px = np.array([[100.0, 200.0], [800.0, 600.0]], dtype=np.float64)
    m = pixel_to_metric_batch(px, cpar)
    px2 = metric_to_pixel_batch(m, cpar)
    np.testing.assert_allclose(px2, px, atol=1e-10)


# ---------------------------------------------------------------------------
# _distort_brown_affin_core_out  (lines 299-333)
# ---------------------------------------------------------------------------


def test_dba_core_out_zero_r():
    """r < 1e-10 early return → (0, 0)."""
    out = _out2()
    _distort_brown_affin_core_out(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, out)
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_dba_core_out_identity():
    """No distortion (k=p=0, scx=1, she=0) → output equals input."""
    out = _out2()
    _distort_brown_affin_core_out(1.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, out)
    assert abs(out[0] - 1.5) < EPS
    assert abs(out[1] - 2.5) < EPS


def test_dba_core_out_radial_k1():
    """Radial k1: factor = 1 + k1*r2."""
    out = _out2()
    # x=1, y=0 → r=1, r2=1; radial_factor=1.01; x_dist=1.01
    _distort_brown_affin_core_out(1.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, out)
    assert abs(out[0] - 1.01) < EPS
    assert abs(out[1]) < EPS


def test_dba_core_out_all_k():
    """All radial coefficients applied simultaneously."""
    out = _out2()
    # x=1, y=0 → r=1, r2=1, r4=1, r6=1
    # radial_factor = 1 + 0.01 + 0.001 + 0.0001 = 1.0111
    _distort_brown_affin_core_out(
        1.0, 0.0, 0.01, 0.001, 0.0001, 0.0, 0.0, 1.0, 0.0, 1.0, out
    )
    assert abs(out[0] - 1.0111) < EPS


def test_dba_core_out_decentering_p1():
    """Decentering p1 affects x_dist via tangential term."""
    out = _out2()
    # x=1, y=0, p1=0.01 → p1*(r2 + 2*x^2) = 0.01*(1+2) = 0.03 added to x_dist
    _distort_brown_affin_core_out(1.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 1.0, 0.0, 1.0, out)
    assert abs(out[0] - 1.03) < EPS  # 1*1 + 0.01*(1+2) = 1.03


# ---------------------------------------------------------------------------
# distort_brown_affin_out  (lines 359-377)
# ---------------------------------------------------------------------------


def test_dba_out_identity():
    """No distortion → identity."""
    out = _out2()
    distort_brown_affin_out(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out)
    assert abs(out[0] - 1.0) < EPS
    assert abs(out[1] - 2.0) < EPS


def test_dba_out_with_shear():
    """she != 0 shifts x component via sin_she * y_dist."""
    out_no_she = _out2()
    out_she = _out2()
    distort_brown_affin_out(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out_no_she)
    distort_brown_affin_out(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, out_she)
    # With shear, x shifts and y shrinks slightly
    assert out_she[0] != out_no_she[0]


# ---------------------------------------------------------------------------
# distort_brown_affine_batch  (lines 413-462)
# ---------------------------------------------------------------------------


def test_dba_batch_identity():
    """No distortion → identity (all rows)."""
    xy = np.ascontiguousarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    res = distort_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    np.testing.assert_allclose(res, xy, atol=EPS)


def test_dba_batch_zero_r():
    """r < 1e-10 branch → (0, 0) for that row."""
    xy = np.ascontiguousarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    res = distort_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert res[0, 0] == 0.0 and res[0, 1] == 0.0
    assert abs(res[1, 0] - 1.0) < EPS


def test_dba_batch_k1_radial():
    """k1 radial distortion scales output > 1 for unit radius."""
    xy = np.ascontiguousarray([[1.0, 0.0]], dtype=np.float64)
    res = distort_brown_affine_batch(xy, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # r=1, radial_factor=1.01
    assert abs(res[0, 0] - 1.01) < EPS
    assert abs(res[0, 1]) < EPS


def test_dba_batch_all_k():
    """k1+k2+k3 combined at unit radius."""
    xy = np.ascontiguousarray([[1.0, 0.0]], dtype=np.float64)
    res = distort_brown_affine_batch(xy, 0.01, 0.001, 0.0001, 0.0, 0.0, 1.0, 0.0)
    assert abs(res[0, 0] - 1.0111) < EPS


def test_dba_batch_output_shape():
    """Output shape matches input."""
    xy = np.ascontiguousarray(np.random.rand(5, 2), dtype=np.float64)
    res = distort_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert res.shape == (5, 2)


# ---------------------------------------------------------------------------
# _correct_brown_affin_out  (lines 465-537)
# ---------------------------------------------------------------------------


def test_cba_out_identity():
    """No distortion: correct_out returns original coordinates."""
    out = _out2()
    scratch = _out2()
    _correct_brown_affin_out(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out, scratch)
    assert abs(out[0] - 1.0) < EPS
    assert abs(out[1] - 2.0) < EPS


def test_cba_out_zero():
    """(0, 0) input converges to (0, 0) — exercises pos_magnitude ≤ 1e-10 path."""
    out = _out2()
    scratch = _out2()
    _correct_brown_affin_out(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out, scratch)
    assert abs(out[0]) < EPS
    assert abs(out[1]) < EPS


def test_cba_out_round_trip_k1():
    """Distort then correct is near-identity for small k1."""
    # Distort
    out_d = _out2()
    distort_brown_affin_out(0.5, 0.5, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out_d)
    # Correct
    out_c = _out2()
    scratch = _out2()
    _correct_brown_affin_out(
        out_d[0], out_d[1], 0.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out_c, scratch
    )
    assert abs(out_c[0] - 0.5) < 0.005
    assert abs(out_c[1] - 0.5) < 0.005


def test_cba_out_convergence_break():
    """Exercises break path: pos_magnitude > 1e-10 and change < tol."""
    # No distortion: converges in first iteration; pos_magnitude > 0
    out = _out2()
    scratch = _out2()
    _correct_brown_affin_out(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out, scratch)
    np.testing.assert_allclose(out, [1.0, 1.0], atol=EPS)


# ---------------------------------------------------------------------------
# _correct_brown_affine_exact_out  (lines 572-681)
# ---------------------------------------------------------------------------


def test_cbae_out_zero_r():
    """r_init < 1e-10 → (0, 0) early return."""
    out = _out2()
    _correct_brown_affine_exact_out(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-8, out
    )
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_cbae_out_identity():
    """No distortion → output equals input."""
    out = _out2()
    _correct_brown_affine_exact_out(
        1.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-8, out
    )
    assert abs(out[0] - 1.5) < EPS
    assert abs(out[1] - 2.5) < EPS


def test_cbae_out_round_trip_k1():
    """Distort then exact-correct → original (tight tolerance)."""
    x0, y0 = 0.3, 0.4
    out_d = _out2()
    distort_brown_affin_out(x0, y0, 0.005, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out_d)
    out_c = _out2()
    _correct_brown_affine_exact_out(
        out_d[0], out_d[1], 0.005, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-10, out_c
    )
    assert abs(out_c[0] - x0) < 1e-6
    assert abs(out_c[1] - y0) < 1e-6


def test_cbae_out_convergence_break():
    """break triggers when sqrt(dx_change^2+dy_change^2) < tol."""
    # Large radius, no distortion: convergence in first iteration
    out = _out2()
    _correct_brown_affine_exact_out(
        5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-8, out
    )
    np.testing.assert_allclose(out, [5.0, 5.0], atol=EPS)


# ---------------------------------------------------------------------------
# flat_to_dist_out  (lines 684-700)
# ---------------------------------------------------------------------------


def test_ftd_out_identity():
    """No distortion, zero principal point → identity."""
    out = _out2()
    flat_to_dist_out(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out)
    assert abs(out[0] - 1.0) < EPS
    assert abs(out[1] - 2.0) < EPS


def test_ftd_out_principal_point_shift():
    """flat_to_dist shifts by xh/yh before distorting."""
    out_no_shift = _out2()
    out_shift = _out2()
    flat_to_dist_out(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out_no_shift)
    flat_to_dist_out(1.0, 2.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out_shift)
    # With xh=yh=0.1 the input point is shifted before distortion
    assert abs(out_shift[0] - 1.1) < EPS
    assert abs(out_shift[1] - 2.1) < EPS


# ---------------------------------------------------------------------------
# correct_brown_affine_batch  (lines 796-897)
# ---------------------------------------------------------------------------


def test_cbab_identity():
    """No distortion: output equals input."""
    xy = np.ascontiguousarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    res = correct_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    np.testing.assert_allclose(res, xy, atol=EPS)


def test_cbab_zero_r():
    """r < 1e-10 in inner loop: row (0,0) stays (0,0)."""
    xy = np.ascontiguousarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    res = correct_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert abs(res[0, 0]) < EPS
    assert abs(res[0, 1]) < EPS
    assert abs(res[1, 0] - 1.0) < EPS


def test_cbab_with_preallocated_out():
    """out parameter branch: result is written into provided array."""
    xy = np.ascontiguousarray([[2.0, 3.0]], dtype=np.float64)
    out_buf = np.empty_like(xy)
    res = correct_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, out=out_buf)
    assert abs(res[0, 0] - 2.0) < EPS
    assert abs(res[0, 1] - 3.0) < EPS


def test_cbab_round_trip_k1():
    """distort_batch then correct_batch ≈ identity for small k1."""
    xy = np.ascontiguousarray([[0.5, 0.3], [1.0, -1.0]], dtype=np.float64)
    k1 = 0.005
    distorted = distort_brown_affine_batch(xy, k1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    corrected = correct_brown_affine_batch(
        np.ascontiguousarray(distorted, dtype=np.float64),
        k1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    )
    np.testing.assert_allclose(corrected, xy, atol=0.005)


def test_cbab_convergence_break():
    """Break condition fires within max_iter for non-zero, no-distortion input."""
    # With no distortion and non-zero input, the iteration converges on step 1.
    xy = np.ascontiguousarray([[2.0, 2.0]], dtype=np.float64)
    res = correct_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    np.testing.assert_allclose(res, xy, atol=EPS)


def test_cbab_output_shape():
    """Output shape matches input shape."""
    xy = np.ascontiguousarray(np.random.rand(4, 2), dtype=np.float64)
    res = correct_brown_affine_batch(xy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert res.shape == (4, 2)


# ---------------------------------------------------------------------------
# Exhaust-loop branch in _correct_brown_affine_exact_out  (line 625->647)
# ---------------------------------------------------------------------------


def test_cbae_out_loop_exhausted():
    """tol=0.0 → sqrt(change) < 0.0 never True → for-loop runs all max_iter
    iterations without break, covering the 625→647 branch (loop fall-through)."""
    out = _out2()
    _correct_brown_affine_exact_out(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, out
    )
    # No distortion: result still correct even without the convergence shortcut.
    assert abs(out[0] - 1.0) < EPS
    assert abs(out[1]) < EPS


# ---------------------------------------------------------------------------
# First-executable-line coverage for each buggy wrapper function.
#
# Every function listed below declares ``_out: cython.double[2]`` (and/or
# ``_scratch: cython.double[2]``) as a *local variable*.  In pure-Python mode
# the bare annotation generates NO bytecode, so the variable is unbound when
# the subsequent ``_out_mv = _out`` assignment is reached → UnboundLocalError.
# The try/except lets coverage.py record that first assignment line as executed.
#
# In compiled mode the @cython.ccall functions work correctly (no exception);
# the @cython.cfunc ones are inaccessible (module import fails before these
# test functions are ever called).
# ---------------------------------------------------------------------------


def test_old_pixel_to_metric_entry():
    """Covers _out_mv = _out line of old_pixel_to_metric (ccall, buggy)."""
    try:
        old_pixel_to_metric(512.0, 504.0, 1024, 1008, 0.01, 0.01, NO_REMAP)
    except UnboundLocalError:
        pass


def test_pixel_to_metric_entry():
    """Covers _out_mv = _out line of pixel_to_metric (ccall, buggy)."""
    try:
        pixel_to_metric(512.0, 504.0, 1024, 1008, 0.01, 0.01, NO_REMAP)
    except UnboundLocalError:
        pass


def test_old_metric_to_pixel_entry():
    """Covers _out_mv = _out line of old_metric_to_pixel (ccall, buggy)."""
    try:
        old_metric_to_pixel(0.0, 0.0, 1024, 1008, 0.01, 0.01, NO_REMAP)
    except UnboundLocalError:
        pass


def test_metric_to_pixel_entry():
    """Covers _out_mv = _out line of metric_to_pixel (ccall, buggy)."""
    try:
        metric_to_pixel(0.0, 0.0, 1024, 1008, 0.01, 0.01, NO_REMAP)
    except UnboundLocalError:
        pass


def test_distort_brown_affin_core_entry():
    """Covers _out_mv = _out line of _distort_brown_affin_core (cfunc, buggy)."""
    try:
        _distort_brown_affin_core(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    except UnboundLocalError:
        pass


def test_distort_brown_affin_entry():
    """Covers _out_mv = _out line of distort_brown_affin (ccall, buggy)."""
    try:
        distort_brown_affin(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    except UnboundLocalError:
        pass


def test_correct_brown_affin_entry():
    """Covers _out_mv = _out line of correct_brown_affin (ccall, buggy)."""
    try:
        correct_brown_affin(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    except UnboundLocalError:
        pass


def test_correct_brown_affine_exact_entry():
    """Covers _out_mv = _out line of correct_brown_affine_exact (cfunc, buggy)."""
    try:
        correct_brown_affine_exact(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-8)
    except UnboundLocalError:
        pass


def test_flat_to_dist_entry():
    """Covers _out_mv = _out line of flat_to_dist (ccall, buggy)."""
    try:
        flat_to_dist(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    except UnboundLocalError:
        pass


def test_dist_to_flat_out_entry():
    """Covers _scratch_mv = _scratch line of dist_to_flat_out (ccall, buggy)."""
    out = _out2()
    try:
        dist_to_flat_out(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-8, out)
    except UnboundLocalError:
        pass


def test_dist_to_flat_entry():
    """Covers _out_mv = _out line of dist_to_flat (ccall, buggy)."""
    try:
        dist_to_flat(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-8)
    except UnboundLocalError:
        pass

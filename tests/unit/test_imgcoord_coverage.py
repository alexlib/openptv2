"""Pure-Python line-coverage tests for imgcoord.py.

Must run against the /tmp/ppsrc snapshot, not the compiled .so:
    COVERAGE_FILE=/tmp/.cov_imgcoord uv run pytest tests/unit/test_imgcoord_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing -q

Source bugs found (do NOT fix here — report only):
  NONE — imgcoord.py has no C-array static-allocation bugs.
  All typed locals use annotated assignments (bound) or are declared then
  immediately assigned inside the same scope.  The cython.double[:]
  memoryview annotations are no-ops in pure-Python mode.
"""

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Guard: skip the whole module when running against the compiled .so
# ---------------------------------------------------------------------------
from openptv2.algorithms.imgcoord import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

# ---------------------------------------------------------------------------
# Imports (after the skip guard so they resolve against ppsrc)
# ---------------------------------------------------------------------------
from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Exterior,
    Glass,
    Interior,
    MmLut,
)
from openptv2.algorithms.imgcoord import (
    _flat_image_coord_core,
    _flat_to_dist_core,
    _get_mmf_from_mmlut_core,
    _img_coord_params,
    flat_image_coord,
    flat_image_coord_batch,
    img_coord,
    img_coord_batch,
    img_coord_typed,
)
from openptv2.algorithms.parameters import MmNp

EPS = 1e-6

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _air_mm():
    """All-air multimedia params (no refraction)."""
    return MmNp(nlay=1, n1=1.0, n2=[1.0, 0.0, 0.0], d=[1.0, 0.0, 0.0], n3=1.0)


def _glass_mm():
    """Glass-water multimedia (n2=1.5) to trigger the nlay iteration."""
    return MmNp(nlay=1, n1=1.0, n2=[1.5, 0.0, 0.0], d=[2.0, 0.0, 0.0], n3=1.33)


def _identity_dm():
    return np.eye(3, dtype=np.float64)


def _basic_cal(glass_z=20.0, xh=0.0, yh=0.0):
    """Camera at (0,0,40) looking down (identity rotation); glass at z=glass_z."""
    cal = Calibration()
    cal.ext_par = Exterior(
        x0=0.0, y0=0.0, z0=40.0,
        omega=0.0, phi=0.0, kappa=0.0,
        dm=_identity_dm(),
    )
    cal.int_par = Interior(xh=xh, yh=yh, cc=10.0)
    cal.glass_par = Glass(vec_x=0.0, vec_y=0.0, vec_z=glass_z)
    cal.added_par = AddedPar(k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, scx=1.0, she=0.0)
    return cal


def _mmlut_with_factor(factor: float):
    """MmLut whose bilinear lookup at pos_t_0=0, pos_t_2=0 returns *factor*."""
    # nr=2, nz=3, rw=1.0 → max_v=6 → indices 0..6 all valid
    # LUT origin at (0,0,0); call with pos=(0,0,0) → ir=0, iz=0
    # bilinear result = data[0]*1*1 + data[1]*1*0 + data[3]*0*1 + data[4]*0*0 = data[0]
    data = np.ones(2 * 3, dtype=np.float64) * factor
    lut = MmLut()
    lut.origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    lut.nr = 2
    lut.nz = 3
    lut.rw = 1.0
    lut.data = data
    return lut


def _cal_with_mmlut(factor: float):
    cal = _basic_cal()
    cal.mmlut = _mmlut_with_factor(factor)
    return cal


# ---------------------------------------------------------------------------
# is_compiled
# ---------------------------------------------------------------------------

def test_is_compiled_false():
    assert _is_compiled() is False


# ---------------------------------------------------------------------------
# _flat_to_dist_core — internal cfunc, accessible in pure-Python
# ---------------------------------------------------------------------------

def test_flat_to_dist_core_near_zero():
    """r < 1e-10 → early return (0.0, 0.0)."""
    # flat_x + xh ≈ 0 and flat_y + yh ≈ 0
    x1, y1 = _flat_to_dist_core(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert x1 == 0.0
    assert y1 == 0.0


def test_flat_to_dist_core_no_distortion():
    """No distortion coeffs → output equals input (with scx=1, she=0)."""
    x1, y1 = _flat_to_dist_core(0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert abs(x1 - 0.5) < EPS
    assert abs(y1 - 0.3) < EPS


def test_flat_to_dist_core_radial():
    """k1 distortion shrinks the projected radius."""
    k1 = -0.01
    flat_x, flat_y = 1.0, 0.0
    x1, y1 = _flat_to_dist_core(flat_x, flat_y, 0.0, 0.0, k1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    r2 = flat_x**2 + flat_y**2
    expected = flat_x * (1.0 + k1 * r2)
    assert abs(x1 - expected) < EPS


def test_flat_to_dist_core_she_rotation():
    """Non-zero shear (she) rotates x/y."""
    she = 0.1
    x1, y1 = _flat_to_dist_core(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, she)
    # x1 = scx*(x_dist - sin(she)*y_dist), y1 = scx*cos(she)*y_dist
    # With flat_x=0, flat_y=1 → x=0, y=1, radial_factor≈1 (r2=1 but k1=0)
    # x_dist=0, y_dist=1 → x1=-sin(she), y1=cos(she)
    assert abs(x1 - (-math.sin(she))) < EPS
    assert abs(y1 - math.cos(she)) < EPS


def test_flat_to_dist_core_tangential():
    """p1 tangential distortion with x=y=0.5."""
    p1, p2 = 0.001, 0.0
    x1, y1 = _flat_to_dist_core(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, p1, p2, 1.0, 0.0)
    # r2 = 0.5, x_dist = 0.5 + p1*(0.5 + 2*0.25) = 0.5 + p1*1.0
    assert abs(x1 - (0.5 + p1 * 1.0)) < EPS


def test_flat_to_dist_core_scx():
    """scx != 1.0 scales the output."""
    x1, y1 = _flat_to_dist_core(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0)
    assert abs(x1 - 2.0) < EPS


# ---------------------------------------------------------------------------
# _get_mmf_from_mmlut_core — internal cfunc, accessible in pure-Python
# ---------------------------------------------------------------------------

def _make_lut_data(nr, nz, fill=1.0):
    return np.full(nr * nz, fill, dtype=np.float64)


def test_get_mmf_ir_out_of_range():
    """ir > mmlut_nr → return 0.0."""
    data = _make_lut_data(2, 3)
    # R / rw > nr: pos_x very large
    result = _get_mmf_from_mmlut_core(
        1000.0, 0.0, 0.0,   # pos: R >> nr*rw
        0.0, 0.0, 0.0,      # origin
        2, 3, 1.0,          # nr, nz, rw
        data,
    )
    assert result == 0.0


def test_get_mmf_iz_negative():
    """iz < 0 → return 0.0.  tz/rw < 0 gives negative iz."""
    data = _make_lut_data(2, 3)
    result = _get_mmf_from_mmlut_core(
        0.0, 0.0, -1.0,    # pos_z=-1 → tz=-1, iz=int(-1)=-1
        0.0, 0.0, 0.0,
        2, 3, 1.0,
        data,
    )
    assert result == 0.0


def test_get_mmf_iz_out_of_range_high():
    """iz > mmlut_nz → return 0.0."""
    data = _make_lut_data(2, 3)
    result = _get_mmf_from_mmlut_core(
        0.0, 0.0, 100.0,   # tz=100 → iz=100 >> nz=3
        0.0, 0.0, 0.0,
        2, 3, 1.0,
        data,
    )
    assert result == 0.0


def test_get_mmf_v4_2_out_of_range():
    """v4_2 > max_v → return 0.0.  Use ir == nr so (ir+1)*nz+iz > max_v."""
    nr, nz = 2, 3
    data = _make_lut_data(nr, nz)
    # ir = nr (check: ir > mmlut_nr is False since int(sr)==nr exactly equals mmlut_nr)
    # sr = nr exactly, iz=0: v4_2 = (nr+1)*nz=9 > max_v=6
    # But wait: ir > mmlut_nr check: `ir > mmlut_nr` i.e. nr > nr == False.
    # So it proceeds to v4 checks.
    result = _get_mmf_from_mmlut_core(
        float(nr), 0.0, 0.0,  # R = nr → sr = nr, ir = nr
        0.0, 0.0, 0.0,
        nr, nz, 1.0,
        data,
    )
    assert result == 0.0


def test_get_mmf_normal_bilinear():
    """Normal case: bilinear interpolation with ir=0, iz=0, sr=0, sz=0."""
    nr, nz = 2, 3
    # data[0]=0.5 → bilinear at (sr=0, sz=0) = data[0]*1*1 = 0.5
    data = np.zeros(nr * nz, dtype=np.float64)
    data[0] = 0.5
    result = _get_mmf_from_mmlut_core(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        nr, nz, 1.0,
        data,
    )
    assert abs(result - 0.5) < EPS


def test_get_mmf_bilinear_interpolated():
    """Bilinear interpolation at fractional sr, sz."""
    nr, nz = 3, 4
    data = np.zeros(nr * nz, dtype=np.float64)
    # ir=0, iz=0 → v4_0=0, v4_1=1, v4_2=4, v4_3=5
    data[0] = 1.0   # (1-sr)*(1-sz)
    data[1] = 2.0   # (1-sr)*sz
    data[nz] = 3.0  # sr*(1-sz)
    data[nz + 1] = 4.0  # sr*sz
    # sr = 0.5, sz = 0.5: result = 1*0.25 + 2*0.25 + 3*0.25 + 4*0.25 = 2.5
    result = _get_mmf_from_mmlut_core(
        0.5, 0.0, 0.5,  # R=0.5 → sr=0.5; tz=0.5 → sz=0.5
        0.0, 0.0, 0.0,
        nr, nz, 1.0,
        data,
    )
    assert abs(result - 2.5) < EPS


# ---------------------------------------------------------------------------
# _flat_image_coord_core — internal cfunc
# ---------------------------------------------------------------------------

def test_flat_image_coord_core_zero_glass_denom_zero():
    """Zero glass vector, denom==0 → (0.0, 0.0)."""
    dm = _identity_dm()
    # Camera at (0,0,40), point at same position → dx=dy=dz=0 → denom=0
    x, y = _flat_image_coord_core(
        0.0, 0.0, 40.0,    # pos == camera
        0.0, 0.0, 40.0,    # ext_x0/y0/z0
        dm, 10.0,
        0.0, 0.0, 0.0,     # glass=0 → zero glass path
        1.0, 1.0, 1.0, 1.0,
        False, 0.0, 0.0, 0.0, 0, 0, 0.0, np.zeros(1, dtype=np.float64),
    )
    assert x == 0.0 and y == 0.0


def test_flat_image_coord_core_zero_glass_normal():
    """Zero glass vector, normal projection."""
    dm = _identity_dm()
    # Camera at (0,0,40), point at (10,5,-20)
    # dx=10, dy=5, dz=-60; denom=dz=-60
    # x = -cc * dx / denom = -10*10/(-60) = 100/60
    x, y = _flat_image_coord_core(
        10.0, 5.0, -20.0,
        0.0, 0.0, 40.0,
        dm, 10.0,
        0.0, 0.0, 0.0,    # zero glass
        1.0, 1.0, 1.0, 1.0,
        False, 0.0, 0.0, 0.0, 0, 0, 0.0, np.zeros(1, dtype=np.float64),
    )
    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS


def test_flat_image_coord_core_with_glass_air():
    """Non-zero glass vector, all-air multimedia."""
    dm = _identity_dm()
    dummy_data = np.zeros(1, dtype=np.float64)
    x, y = _flat_image_coord_core(
        10.0, 5.0, -20.0,
        0.0, 0.0, 40.0,
        dm, 10.0,
        0.0, 0.0, 20.0,    # glass at z=20
        1.0, 1.0, 1.0, 1.0,
        False, 0.0, 0.0, 0.0, 0, 0, 0.0, dummy_data,
    )
    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS


def test_flat_image_coord_core_with_glass_mmlut():
    """Non-zero glass vector with mmlut enabled."""
    dm = _identity_dm()
    nr, nz = 2, 3
    lut_data = np.ones(nr * nz, dtype=np.float64) * 0.8
    x, y = _flat_image_coord_core(
        10.0, 5.0, -20.0,
        0.0, 0.0, 40.0,
        dm, 10.0,
        0.0, 0.0, 20.0,
        1.0, 1.0, 1.0, 1.0,
        True, 0.0, 0.0, 0.0, nr, nz, 1.0, lut_data,
    )
    # Result differs slightly from air case; just check it returns floats
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_flat_image_coord_core_mmlut_zero_mmf():
    """mmlut returns mmf <= 0 → falls back to mmf=1.0."""
    dm = _identity_dm()
    nr, nz = 2, 3
    lut_data = np.zeros(nr * nz, dtype=np.float64)  # all zeros → mmf=0 → fallback
    x, y = _flat_image_coord_core(
        10.0, 5.0, -20.0,
        0.0, 0.0, 40.0,
        dm, 10.0,
        0.0, 0.0, 20.0,
        1.0, 1.0, 1.0, 1.0,
        True, 0.0, 0.0, 0.0, nr, nz, 1.0, lut_data,
    )
    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS


def test_flat_image_coord_core_n_ve_zero():
    """n_ve (pos_t_0) == 0: skip the s_x adjustment branch."""
    dm = _identity_dm()
    dummy_data = np.zeros(1, dtype=np.float64)
    # On-axis point (0,0, below glass) → tmp vector is zero → pos_t_0 = 0
    x, y = _flat_image_coord_core(
        0.0, 0.0, -20.0,
        0.0, 0.0, 40.0,
        dm, 10.0,
        0.0, 0.0, 20.0,
        1.0, 1.0, 1.0, 1.0,
        False, 0.0, 0.0, 0.0, 0, 0, 0.0, dummy_data,
    )
    assert abs(x) < EPS
    assert abs(y) < EPS


# ---------------------------------------------------------------------------
# flat_image_coord — public ccall
# ---------------------------------------------------------------------------

def test_flat_image_coord_no_mmlut():
    pos = np.array([10.0, 5.0, -20.0])
    cal = _basic_cal()
    mm = _air_mm()
    x, y = flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
    )
    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS


def test_flat_image_coord_with_mmlut():
    """flat_image_coord with a populated mmlut (has_mmlut=True branch)."""
    pos = np.array([10.0, 5.0, -20.0])
    cal = _basic_cal()
    mm = _air_mm()
    lut = _mmlut_with_factor(0.9)
    x, y = flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        mmlut=lut,
    )
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_flat_image_coord_mmlut_none_data():
    """mmlut.data is None → has_mmlut stays False."""
    pos = np.array([10.0, 5.0, -20.0])
    cal = _basic_cal()
    mm = _air_mm()
    empty_lut = MmLut()  # data=None → is_initialized=False → not passed
    x, y = flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        mmlut=None,
    )
    assert abs(x - 10.0 / 6.0) < EPS


# ---------------------------------------------------------------------------
# img_coord — public function (two call paths)
# ---------------------------------------------------------------------------

def test_img_coord_explicit_params():
    """img_coord with all params explicit (ext_z0 is not None)."""
    pos = np.array([10.0, 5.0, -20.0])
    cal = _basic_cal()
    mm = _air_mm()
    x, y = img_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.int_par.xh, cal.int_par.yh,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
        cal.added_par.p1, cal.added_par.p2,
        cal.added_par.scx, cal.added_par.she,
    )
    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS


def test_img_coord_calibration_object():
    """img_coord with a Calibration object (ext_z0 is None path)."""
    pos = np.array([10.0, 5.0, -20.0])
    cal = _basic_cal()
    mm = _air_mm()
    x, y = img_coord(pos, cal, mm)
    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS


def test_img_coord_calibration_with_mmlut():
    """img_coord via Calibration path when mmlut is initialized."""
    pos = np.array([10.0, 5.0, -20.0])
    cal = _cal_with_mmlut(0.9)
    mm = _air_mm()
    x, y = img_coord(pos, cal, mm)
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_img_coord_list_input():
    """img_coord accepts Python list — np.ascontiguousarray conversion."""
    cal = _basic_cal()
    mm = _air_mm()
    x, y = img_coord(
        [10.0, 5.0, -20.0],
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.int_par.xh, cal.int_par.yh,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    )
    assert abs(x - 10.0 / 6.0) < EPS


def test_img_coord_with_distortion():
    """img_coord: k1 barrel distortion is applied."""
    pos = np.array([10.0, 5.0, -20.0])
    cal = _basic_cal()
    cal.added_par.k1 = -0.01
    mm = _air_mm()
    x_dist, y_dist = img_coord(pos, cal, mm)
    x_flat, _ = img_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        0.0, 0.0,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    )
    # With k1<0, distorted x < flat x (barrel)
    assert x_dist < x_flat


# ---------------------------------------------------------------------------
# _img_coord_params (tested indirectly but worth a direct call too)
# ---------------------------------------------------------------------------

def test_img_coord_params_direct():
    pos = np.ascontiguousarray([10.0, 5.0, -20.0], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    x, y = _img_coord_params(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.int_par.xh, cal.int_par.yh,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    )
    assert abs(x - 10.0 / 6.0) < EPS


def test_img_coord_params_with_mmlut():
    pos = np.ascontiguousarray([10.0, 5.0, -20.0], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    lut = _mmlut_with_factor(0.8)
    x, y = _img_coord_params(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.int_par.xh, cal.int_par.yh,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        mmlut=lut,
    )
    assert isinstance(x, float)


# ---------------------------------------------------------------------------
# img_coord_typed
# ---------------------------------------------------------------------------

def test_img_coord_typed_basic():
    pos = np.ascontiguousarray([10.0, 5.0, -20.0], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    x, y = img_coord_typed(pos, cal, mm)
    assert abs(x - 10.0 / 6.0) < EPS
    assert abs(y - 5.0 / 6.0) < EPS


def test_img_coord_typed_with_mmlut():
    pos = np.ascontiguousarray([10.0, 5.0, -20.0], dtype=np.float64)
    cal = _cal_with_mmlut(0.7)
    mm = _air_mm()
    x, y = img_coord_typed(pos, cal, mm)
    assert isinstance(x, float)


# ---------------------------------------------------------------------------
# img_coord_batch — covers _img_coord_batch_impl branches
# ---------------------------------------------------------------------------

def test_img_coord_batch_single_point_air():
    """Batch with N=1, all-air: hits all-air fast path (radial_shift=1.0)."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    result = img_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)
    assert abs(result[0, 0] - 10.0 / 6.0) < EPS
    assert abs(result[0, 1] - 5.0 / 6.0) < EPS


def test_img_coord_batch_multiple_points():
    """N=3 points, all-air."""
    positions = np.array([
        [10.0, 5.0, -20.0],
        [-5.0, 3.0, -20.0],
        [0.0, 0.0, -20.0],
    ], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    result = img_coord_batch(positions, cal, mm)
    assert result.shape == (3, 2)
    # Third point is on-axis → flat coords 0,0 → r_pt < 1e-10 branch for flat-to-dist
    assert abs(result[2, 0]) < EPS
    assert abs(result[2, 1]) < EPS


def test_img_coord_batch_glass_mm_nlay_iteration():
    """n1 != n2 triggers the nlay iteration loop (convergence path)."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    cal = _basic_cal()
    mm = _glass_mm()
    result = img_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)
    # Result should be close to the single-point function
    x_single, y_single = img_coord_typed(
        np.ascontiguousarray([10.0, 5.0, -20.0], dtype=np.float64), cal, mm
    )
    assert abs(result[0, 0] - x_single) < 1e-4
    assert abs(result[0, 1] - y_single) < 1e-4


def test_img_coord_batch_with_mmlut_nonzero():
    """Batch with mmlut returning non-1.0 value → mmf branch."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    cal = _cal_with_mmlut(0.8)
    mm = _air_mm()
    result = img_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)
    assert isinstance(result[0, 0], float)


def test_img_coord_batch_with_mmlut_zero_fallback():
    """Batch with mmlut returning 0 → mmf=1.0 fallback, then nlay all-air."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    # mmlut with all-zero data → mmf=0 → fallback to 1.0
    cal = _cal_with_mmlut(0.0)
    mm = _air_mm()
    result = img_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)


def test_img_coord_batch_on_axis_n_ve_zero():
    """On-axis point: pos_t_0=0 → n_ve=0 → skip s_x adjustment branch."""
    positions = np.array([[0.0, 0.0, -20.0]], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    result = img_coord_batch(positions, cal, mm)
    assert abs(result[0, 0]) < EPS
    assert abs(result[0, 1]) < EPS


def test_img_coord_batch_list_input():
    """img_coord_batch converts list to array (not-compiled branch)."""
    positions = [[10.0, 5.0, -20.0], [0.0, 0.0, -20.0]]
    cal = _basic_cal()
    mm = _air_mm()
    result = img_coord_batch(positions, cal, mm)
    assert result.shape == (2, 2)


def test_img_coord_batch_mmlut_with_glass():
    """Batch: mmlut + glass refraction (n!=1) → mmf branch then maybe iteration."""
    positions = np.array([[5.0, 3.0, -10.0], [1.0, 1.0, -5.0]], dtype=np.float64)
    cal = _cal_with_mmlut(1.2)  # mmf=1.2 > 0 and != 1.0 → radial_shift=mmf path
    mm = _glass_mm()
    result = img_coord_batch(positions, cal, mm)
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# flat_image_coord_batch — covers _flat_image_coord_batch_impl branches
# ---------------------------------------------------------------------------

def test_flat_image_coord_batch_single_air():
    """Batch flat: N=1, all-air."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)
    assert abs(result[0, 0] - 10.0 / 6.0) < EPS
    assert abs(result[0, 1] - 5.0 / 6.0) < EPS


def test_flat_image_coord_batch_multiple_points():
    """Batch flat: N=3."""
    positions = np.array([
        [10.0, 5.0, -20.0],
        [-5.0, 3.0, -20.0],
        [0.0, 0.0, -20.0],
    ], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert result.shape == (3, 2)
    assert abs(result[2, 0]) < EPS
    assert abs(result[2, 1]) < EPS


def test_flat_image_coord_batch_glass_nlay():
    """Batch flat: glass multimedia triggers nlay iteration."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    cal = _basic_cal()
    mm = _glass_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)


def test_flat_image_coord_batch_with_mmlut():
    """Batch flat with mmlut (non-1.0 factor → radial_shift=mmf)."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    cal = _cal_with_mmlut(1.1)
    mm = _air_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)


def test_flat_image_coord_batch_mmlut_zero_fallback():
    """Batch flat: mmlut returns 0 → fallback to mmf=1.0 → nlay path."""
    positions = np.array([[10.0, 5.0, -20.0]], dtype=np.float64)
    cal = _cal_with_mmlut(0.0)
    mm = _air_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)


def test_flat_image_coord_batch_on_axis_n_ve_zero():
    """Batch flat: on-axis point → pos_t_0=0 → skip s_x branch."""
    positions = np.array([[0.0, 0.0, -20.0]], dtype=np.float64)
    cal = _basic_cal()
    mm = _air_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert abs(result[0, 0]) < EPS
    assert abs(result[0, 1]) < EPS


def test_flat_image_coord_batch_list_input():
    """flat_image_coord_batch converts list to array (not-compiled branch)."""
    positions = [[10.0, 5.0, -20.0]]
    cal = _basic_cal()
    mm = _air_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)


def test_flat_image_coord_batch_n_ve_zero_glass():
    """Batch flat: on-axis with glass nlay (r_val=0 → radial_shift=1 path)."""
    positions = np.array([[0.0, 0.0, -20.0]], dtype=np.float64)
    cal = _basic_cal()
    mm = _glass_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert abs(result[0, 0]) < EPS
    assert abs(result[0, 1]) < EPS


def test_flat_image_coord_batch_mmlut_plus_glass():
    """Batch flat: mmlut with non-1 factor + glass mm → mmf branch."""
    positions = np.array([[5.0, 3.0, -10.0]], dtype=np.float64)
    cal = _cal_with_mmlut(0.9)
    mm = _glass_mm()
    result = flat_image_coord_batch(positions, cal, mm)
    assert result.shape == (1, 2)


# ---------------------------------------------------------------------------
# Consistency cross-checks
# ---------------------------------------------------------------------------

def test_batch_matches_single_point():
    """img_coord_batch result matches img_coord for each point."""
    points = [
        [10.0, 5.0, -20.0],
        [-3.0, 7.0, -15.0],
        [0.0, 4.0, -25.0],
    ]
    cal = _basic_cal()
    mm = _air_mm()
    positions = np.array(points, dtype=np.float64)
    batch_result = img_coord_batch(positions, cal, mm)
    for i, pt in enumerate(points):
        x, y = img_coord_typed(np.array(pt, dtype=np.float64), cal, mm)
        assert abs(batch_result[i, 0] - x) < 1e-10
        assert abs(batch_result[i, 1] - y) < 1e-10


def test_flat_batch_matches_single_point():
    """flat_image_coord_batch matches flat_image_coord for each point."""
    points = [
        [10.0, 5.0, -20.0],
        [-3.0, 7.0, -15.0],
    ]
    cal = _basic_cal()
    mm = _air_mm()
    positions = np.array(points, dtype=np.float64)
    batch_result = flat_image_coord_batch(positions, cal, mm)
    for i, pt in enumerate(points):
        x, y = flat_image_coord(
            np.array(pt, dtype=np.float64),
            cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
            cal.ext_par.dm, cal.int_par.cc,
            cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
            mm.n1, mm.n2[0], mm.n3, mm.d[0],
        )
        assert abs(batch_result[i, 0] - x) < 1e-10
        assert abs(batch_result[i, 1] - y) < 1e-10


def test_img_coord_vs_flat_with_no_distortion():
    """img_coord == flat_image_coord when all distortion coefficients are zero."""
    pos = np.array([10.0, 5.0, -20.0], dtype=np.float64)
    cal = _basic_cal()  # xh=yh=0, k1=k2=k3=p1=p2=0, scx=1, she=0
    mm = _air_mm()
    x_img, y_img = img_coord(pos, cal, mm)
    x_flat, y_flat = flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
    )
    assert abs(x_img - x_flat) < EPS
    assert abs(y_img - y_flat) < EPS

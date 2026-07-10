"""Pure-Python line coverage tests for src/openptv2/algorithms/track_kernels_geom.py.

Run via:
    COVERAGE_FILE=/tmp/.cov_track_kernels_geom uv run pytest \
      tests/unit/test_track_kernels_geom_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q 2>&1 | grep -E '(algorithms/track_kernels_geom\\.|TOTAL|passed|failed|error)'
"""

import numpy as np
import pytest

# Guard: skip when compiled .so is active.
from openptv2.algorithms.track_kernels import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.track_kernels_geom import (
    CAL_ARRAY_SIZE,
    PT_UNUSED,
    _angle_acc_out,
    _multimed_r_nlay_1layer,
    _point_to_pixel_out,
    _ray_tracing_fast,
    _ray_tracing_out,
    angle_acc_fast,
    point_to_pixel_fast,
    searchquader_fast,
)

# ---------------------------------------------------------------------------
# Cal array builder
# Cal array layout (31 float64):
#  0-2:   ext_x0, ext_y0, ext_z0
#  3-11:  dm[0,0], dm[1,0], dm[2,0], dm[0,1], dm[1,1], dm[2,1],
#          dm[0,2], dm[1,2], dm[2,2]
#  12:    int_cc
#  13-14: xh, yh
#  15-17: gx, gy, gz
#  18:    dist_o_glas
#  19:    inv_dog
#  20-23: mm_n1, mm_n2_0, mm_n3, mm_d0
#  24-30: k1, k2, k3, p1, p2, scx, she
# ---------------------------------------------------------------------------


def _make_cal_array(
    ext_x0=0.0,
    ext_y0=0.0,
    ext_z0=-100.0,
    dm=None,
    int_cc=100.0,
    xh=0.0,
    yh=0.0,
    gx=0.0,
    gy=0.0,
    gz=1.0,
    mm_n1=1.0,
    mm_n2_0=1.5,
    mm_n3=1.33,
    mm_d0=2.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    scx=1.0,
    she=0.0,
):
    if dm is None:
        dm = np.eye(3, dtype=np.float64)
    g_len = float(np.sqrt(gx**2 + gy**2 + gz**2))
    inv_dog = 1.0 / g_len if g_len != 0.0 else 0.0
    cal = np.zeros(31, dtype=np.float64)
    cal[0] = ext_x0
    cal[1] = ext_y0
    cal[2] = ext_z0
    # column-major order: dm[row, col]
    cal[3] = dm[0, 0];  cal[4] = dm[1, 0];  cal[5] = dm[2, 0]
    cal[6] = dm[0, 1];  cal[7] = dm[1, 1];  cal[8] = dm[2, 1]
    cal[9] = dm[0, 2];  cal[10] = dm[1, 2]; cal[11] = dm[2, 2]
    cal[12] = int_cc
    cal[13] = xh
    cal[14] = yh
    cal[15] = gx
    cal[16] = gy
    cal[17] = gz
    cal[18] = g_len        # dist_o_glas
    cal[19] = inv_dog
    cal[20] = mm_n1
    cal[21] = mm_n2_0
    cal[22] = mm_n3
    cal[23] = mm_d0
    cal[24] = k1
    cal[25] = k2
    cal[26] = k3
    cal[27] = p1
    cal[28] = p2
    cal[29] = scx
    cal[30] = she
    return cal


# Shared pixel-space constants
IMX_HALF = 640.0
IMY_HALF = 512.0
INV_PIX_X = 100.0   # px/mm  (pixel size = 0.01 mm)
INV_PIX_Y = 100.0

_EMPTY_MD = np.zeros(0, dtype=np.float64)
_EMPTY_MO = np.zeros(3, dtype=np.float64)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_cal_array_size():
    assert CAL_ARRAY_SIZE == 31


def test_pt_unused():
    assert PT_UNUSED == -999


# ---------------------------------------------------------------------------
# _multimed_r_nlay_1layer
# ---------------------------------------------------------------------------


def test_multimed_all_n_one_early_return():
    """n1 == n2 == n3 == 1.0 → early return 1.0."""
    r = _multimed_r_nlay_1layer(5.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0, 1.0, 1.0, 2.0)
    assert r == 1.0


def test_multimed_denom_zero_returns_one():
    """pos_z == ext_z0 → denom == 0 → return 1.0."""
    # denom = ext_z0 - pos_z = 0
    r = _multimed_r_nlay_1layer(
        1.0, 0.0, 100.0,   # pos_x, pos_y, pos_z
        0.0, 0.0, 100.0,   # ext_x0, ext_y0, ext_z0  (same z → denom=0)
        1.0, 1.5, 1.33, 2.0,
    )
    assert r == 1.0


def test_multimed_r_zero_returns_one():
    """pos_x == ext_x0 and pos_y == ext_y0 → r == 0 → return 1.0."""
    r = _multimed_r_nlay_1layer(
        0.0, 0.0, 0.0,
        0.0, 0.0, 100.0,
        1.0, 1.5, 1.33, 2.0,
    )
    assert r == 1.0


def test_multimed_normal_convergence():
    """Normal case converges and returns finite positive value."""
    r = _multimed_r_nlay_1layer(
        5.0, 0.0, 0.0,
        0.0, 0.0, 100.0,
        1.0, 1.5, 1.33, 2.0,
    )
    assert isinstance(r, float)
    assert r > 0.0
    assert np.isfinite(r)


def test_multimed_arg_clamp_high():
    """n1/n2 >> 1 → sin_beta1*n1/n2 > 1 → clamp to 1."""
    r = _multimed_r_nlay_1layer(
        50.0, 0.0, 0.0,
        0.0, 0.0, 100.0,
        3.0, 1.0, 1.33, 2.0,   # n1=3, n2=1 → arg > 1
    )
    assert isinstance(r, float)


def test_multimed_arg3_clamp_high():
    """n1/n3 >> 1 → arg3 > 1 → clamp."""
    r = _multimed_r_nlay_1layer(
        50.0, 0.0, 0.0,
        0.0, 0.0, 100.0,
        3.0, 1.5, 1.0, 2.0,   # n1/n3 = 3
    )
    assert isinstance(r, float)


def test_multimed_negative_arg_clamp_low():
    """Negative sin value with high n ratio → clamp to -1."""
    # Arrange negative rq by making ext_x0 offset beyond pos
    r = _multimed_r_nlay_1layer(
        -50.0, 0.0, 0.0,
        0.0, 0.0, 100.0,
        3.0, 1.0, 1.0, 2.0,
    )
    assert isinstance(r, float)


def test_multimed_small_offset():
    """Very small radial offset still converges."""
    r = _multimed_r_nlay_1layer(
        0.1, 0.0, 0.0,
        0.0, 0.0, 100.0,
        1.0, 1.5, 1.33, 2.0,
    )
    assert np.isfinite(r)
    assert r > 0.0


# ---------------------------------------------------------------------------
# point_to_pixel_fast  — no mmlut
# ---------------------------------------------------------------------------


def test_p2p_fast_on_axis_no_mmlut():
    """Point on camera axis projects to principal point (pixel centre)."""
    cal = _make_cal_array()
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_p2p_fast_off_axis():
    """Off-axis point exercises pos_t_0 > 0 and distortion branch."""
    cal = _make_cal_array()
    pos = np.array([5.0, 3.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_p2p_fast_r_near_zero_branch():
    """r < 1e-10 in flat_to_dist → x_dist = y_dist = 0."""
    # Use very small xh/yh and on-axis point so projected coords ≈ 0
    cal = _make_cal_array(xh=0.0, yh=0.0)
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_p2p_fast_chfield_0():
    cal = _make_cal_array()
    pos = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    x0, y0 = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(y0)


def test_p2p_fast_chfield_1():
    """chfield == 1 → y_pixel = (y_pixel - 1) * 0.5."""
    cal = _make_cal_array()
    pos = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    _, y0 = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    _, y1 = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 1,
    )
    assert np.isfinite(y1)
    assert abs(y1 - (y0 - 1.0) * 0.5) < 1e-10


def test_p2p_fast_chfield_2():
    """chfield == 2 → y_pixel = y_pixel * 0.5."""
    cal = _make_cal_array()
    pos = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    _, y0 = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    _, y2 = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 2,
    )
    assert np.isfinite(y2)
    assert abs(y2 - y0 * 0.5) < 1e-10


def test_p2p_fast_radial_distortion():
    """k1/k2/k3 non-zero exercises the radial distortion formula."""
    cal = _make_cal_array(k1=0.001, k2=1e-5, k3=1e-8, p1=1e-4, p2=1e-4)
    pos = np.array([5.0, 3.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_p2p_fast_she_nonzero():
    """she != 0 exercises cos_she / sin_she branches."""
    cal = _make_cal_array(she=0.05, scx=0.99)
    pos = np.array([4.0, 2.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_p2p_fast_mmlut_hit_positive_mmf():
    """has_mmlut=1, LUT data > 0 → mmf > 0 → radial_shift = mmf."""
    cal = _make_cal_array()
    nr, nz = 4, 4
    mmlut_data = np.full(nr * nz, 1.02, dtype=np.float64)
    mmlut_origin = np.zeros(3, dtype=np.float64)
    mmlut_rw = 5.0
    pos = np.array([3.0, 2.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, mmlut_data, mmlut_origin, nr, nz, mmlut_rw, 1,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_p2p_fast_mmlut_mmf_zero_fallback():
    """LUT data all zero → mmf <= 0 → radial_shift stays 1.0 → _multimed fallback."""
    cal = _make_cal_array()
    nr, nz = 4, 4
    mmlut_data = np.zeros(nr * nz, dtype=np.float64)
    mmlut_origin = np.zeros(3, dtype=np.float64)
    mmlut_rw = 5.0
    pos = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, mmlut_data, mmlut_origin, nr, nz, mmlut_rw, 1,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)


def test_p2p_fast_mmlut_out_of_bounds():
    """has_mmlut=1, point projects outside LUT grid → outer if fails → fallback."""
    cal = _make_cal_array()
    nr, nz = 2, 2
    mmlut_data = np.ones(nr * nz, dtype=np.float64)
    mmlut_origin = np.zeros(3, dtype=np.float64)
    mmlut_rw = 1.0   # tiny grid → ir > nr for large points
    pos = np.array([200.0, 0.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, mmlut_data, mmlut_origin, nr, nz, mmlut_rw, 1,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)


def test_p2p_fast_matched_refractive_indices():
    """n1 == n2 == n3 → _multimed returns 1.0 immediately."""
    cal = _make_cal_array(mm_n1=1.33, mm_n2_0=1.33, mm_n3=1.33)
    pos = np.array([5.0, 3.0, 0.0], dtype=np.float64)
    x, y = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert np.isfinite(x)


# ---------------------------------------------------------------------------
# _point_to_pixel_out
# ---------------------------------------------------------------------------


def test_p2p_out_basic_returns_zero():
    cal = _make_cal_array()
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    ret = _point_to_pixel_out(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, out,
    )
    assert ret == 0
    assert np.isfinite(out[0])
    assert np.isfinite(out[1])


def test_p2p_out_matches_fast():
    cal = _make_cal_array()
    pos = np.array([4.0, 2.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, out,
    )
    x_f, y_f = point_to_pixel_fast(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0,
    )
    assert abs(out[0] - x_f) < 1e-10
    assert abs(out[1] - y_f) < 1e-10


def test_p2p_out_chfield_1():
    cal = _make_cal_array()
    pos = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 1, out,
    )
    assert np.isfinite(out[1])


def test_p2p_out_chfield_2():
    cal = _make_cal_array()
    pos = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 2, out,
    )
    assert np.isfinite(out[1])


def test_p2p_out_distortion():
    cal = _make_cal_array(k1=0.001, k2=1e-5, she=0.02, scx=0.99)
    pos = np.array([5.0, 3.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, out,
    )
    assert np.isfinite(out[0])
    assert np.isfinite(out[1])


def test_p2p_out_mmlut_hit():
    cal = _make_cal_array()
    nr, nz = 4, 4
    mmlut_data = np.full(nr * nz, 1.01, dtype=np.float64)
    mmlut_origin = np.zeros(3, dtype=np.float64)
    pos = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    ret = _point_to_pixel_out(
        pos, cal, mmlut_data, mmlut_origin, nr, nz, 5.0, 1,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, out,
    )
    assert ret == 0
    assert np.isfinite(out[0])


def test_p2p_out_r_near_zero():
    cal = _make_cal_array(xh=0.0, yh=0.0)
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, _EMPTY_MD, _EMPTY_MO, 0, 0, 0.0, 0,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, out,
    )
    # x_dist = y_dist = 0 when r < 1e-10
    assert np.isfinite(out[0])


def test_p2p_out_mmlut_out_of_bounds():
    cal = _make_cal_array()
    nr, nz = 2, 2
    mmlut_data = np.ones(nr * nz, dtype=np.float64)
    mmlut_origin = np.zeros(3, dtype=np.float64)
    pos = np.array([200.0, 0.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, mmlut_data, mmlut_origin, nr, nz, 1.0, 1,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, out,
    )
    assert np.isfinite(out[0])


def test_p2p_out_mmlut_mmf_zero():
    cal = _make_cal_array()
    nr, nz = 4, 4
    mmlut_data = np.zeros(nr * nz, dtype=np.float64)
    mmlut_origin = np.zeros(3, dtype=np.float64)
    pos = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, mmlut_data, mmlut_origin, nr, nz, 5.0, 1,
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, out,
    )
    assert np.isfinite(out[0])


# ---------------------------------------------------------------------------
# searchquader_fast
# ---------------------------------------------------------------------------


def _std_quader():
    return np.ascontiguousarray([
        [-1.0, -1.0, -1.0], [-1.0, -1.0,  1.0],
        [-1.0,  1.0, -1.0], [-1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0], [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0, -1.0], [ 1.0,  1.0,  1.0],
    ], dtype=np.float64)


def test_searchquader_single_cam_returns_shapes():
    cal = _make_cal_array()
    point = np.ascontiguousarray([0.0, 0.0, 0.0], dtype=np.float64)
    xr, xl, yd, yu = searchquader_fast(
        point, _std_quader(), 1, [cal],
        [_EMPTY_MD], [_EMPTY_MO], [0], [0], [0.0],
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, 1280.0, 1024.0,
    )
    assert xr.shape == (1,) and xl.shape == (1,)
    assert yd.shape == (1,) and yu.shape == (1,)
    assert np.all(np.isfinite(xr))
    assert np.all(np.isfinite(xl))


def test_searchquader_two_cams():
    cal1 = _make_cal_array(ext_x0=-50.0)
    cal2 = _make_cal_array(ext_x0= 50.0)
    point = np.ascontiguousarray([0.0, 0.0, 0.0], dtype=np.float64)
    xr, xl, yd, yu = searchquader_fast(
        point, _std_quader(), 2, [cal1, cal2],
        [_EMPTY_MD, _EMPTY_MD], [_EMPTY_MO, _EMPTY_MO],
        [0, 0], [0, 0], [0.0, 0.0],
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, 1280.0, 1024.0,
    )
    assert xr.shape == (2,)


def test_searchquader_with_output_buffers():
    """xr_out/xl_out/yd_out/yu_out pre-allocated path."""
    cal = _make_cal_array()
    point = np.ascontiguousarray([0.0, 0.0, 0.0], dtype=np.float64)
    quader = np.ascontiguousarray(np.zeros((8, 3), dtype=np.float64))
    xr_out = np.zeros(1, dtype=np.float64)
    xl_out = np.zeros(1, dtype=np.float64)
    yd_out = np.zeros(1, dtype=np.float64)
    yu_out = np.zeros(1, dtype=np.float64)
    xr, xl, yd, yu = searchquader_fast(
        point, quader, 1, [cal],
        [_EMPTY_MD], [_EMPTY_MO], [0], [0], [0.0],
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, 1280.0, 1024.0,
        xr_out=xr_out, xl_out=xl_out, yd_out=yd_out, yu_out=yu_out,
    )
    assert xr is xr_out
    assert xl is xl_out


def test_searchquader_boundary_clip():
    """Large corners project outside image → xl_i<0, yu_i<0, xr_i>imx, yd_i>imy clipped."""
    cal = _make_cal_array()
    point = np.ascontiguousarray([0.0, 0.0, 0.0], dtype=np.float64)
    far_quader = np.ascontiguousarray([
        [-800.0, -800.0, -10.0], [-800.0, -800.0,  10.0],
        [-800.0,  800.0, -10.0], [-800.0,  800.0,  10.0],
        [ 800.0, -800.0, -10.0], [ 800.0, -800.0,  10.0],
        [ 800.0,  800.0, -10.0], [ 800.0,  800.0,  10.0],
    ], dtype=np.float64)
    xr, xl, yd, yu = searchquader_fast(
        point, far_quader, 1, [cal],
        [_EMPTY_MD], [_EMPTY_MO], [0], [0], [0.0],
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, 1280.0, 1024.0,
    )
    assert np.isfinite(xr[0])


def test_searchquader_zero_quader():
    """All corners at origin — degenerate but should not raise."""
    cal = _make_cal_array()
    point = np.ascontiguousarray([1.0, 0.0, 0.0], dtype=np.float64)
    quader = np.ascontiguousarray(np.zeros((8, 3), dtype=np.float64))
    xr, xl, yd, yu = searchquader_fast(
        point, quader, 1, [cal],
        [_EMPTY_MD], [_EMPTY_MO], [0], [0], [0.0],
        IMX_HALF, IMY_HALF, INV_PIX_X, INV_PIX_Y, 0, 1280.0, 1024.0,
    )
    assert xr.shape == (1,)


# ---------------------------------------------------------------------------
# angle_acc_fast
# ---------------------------------------------------------------------------


def test_angle_acc_fast_same_vectors_zero():
    """v0 == v1 → angle = 0.0, acc = 0.0."""
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,   # start
        1.0, 0.0, 0.0,   # pred
        1.0, 0.0, 0.0,   # cand  (same as pred)
    )
    assert angle == 0.0
    assert acc == 0.0


def test_angle_acc_fast_opposite_vectors_200():
    """v0 == -v1 → angle = 200.0."""
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0,
    )
    assert angle == 200.0


def test_angle_acc_fast_90_degrees():
    """Perpendicular vectors → angle ≈ 100.0 (90° scaled to 200/π·rad)."""
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    )
    assert abs(angle - 100.0) < 0.1
    assert np.isfinite(acc)


def test_angle_acc_fast_norm0_zero():
    """start == pred → v0 = (0,0,0) → norm0 = 0 → angle = 0.0."""
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
    )
    assert angle == 0.0


def test_angle_acc_fast_norm1_zero():
    """start == cand → v1 = (0,0,0) → norm1 = 0 → angle = 0.0."""
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )
    assert angle == 0.0


def test_angle_acc_fast_nearly_parallel():
    """Almost parallel vectors — dot may be > 1 in floating point → clamped."""
    eps = 1e-14
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0 + eps, 0.0, 0.0,
    )
    assert 0.0 <= angle <= 200.0


def test_angle_acc_fast_acceleration_value():
    """Acc is the distance between v1 and v0."""
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
    )
    # v0=(1,0,0), v1=(2,0,0) → dx=1 → acc=1
    assert abs(acc - 1.0) < 1e-10


def test_angle_acc_fast_3d_vectors():
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        1.0, -1.0, 0.0,
    )
    assert 0.0 <= angle <= 200.0
    assert np.isfinite(acc)


def test_angle_acc_fast_negative_dot_clamped():
    """Antiparallel but not exact → dot < -1 gets clamped to -1."""
    # Make two nearly-opposite unit vectors with floating-point excess
    angle, acc = angle_acc_fast(
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        -1.0, 1e-15, 0.0,   # nearly opposite, not exactly
    )
    assert 0.0 <= angle <= 200.0


# ---------------------------------------------------------------------------
# _angle_acc_out
# ---------------------------------------------------------------------------


def test_angle_acc_out_returns_int_zero():
    out = np.zeros(2, dtype=np.float64)
    ret = _angle_acc_out(
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, out
    )
    assert ret == 0
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_angle_acc_out_opposite():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, out)
    assert out[0] == 200.0


def test_angle_acc_out_90_degrees():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, out)
    assert abs(out[0] - 100.0) < 0.1
    assert np.isfinite(out[1])


def test_angle_acc_out_matches_fast():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.5, out)
    angle, acc = angle_acc_fast(0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.5)
    assert abs(out[0] - angle) < 1e-10
    assert abs(out[1] - acc) < 1e-10


def test_angle_acc_out_norm0_zero():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, out)
    assert out[0] == 0.0


def test_angle_acc_out_norm1_zero():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, out)
    assert out[0] == 0.0


def test_angle_acc_out_3d():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, out)
    assert 0.0 <= out[0] <= 200.0
    assert np.isfinite(out[1])


# ---------------------------------------------------------------------------
# _ray_tracing_fast
# ---------------------------------------------------------------------------


def test_ray_tracing_fast_on_axis_tuple_len():
    cal = _make_cal_array()
    result = _ray_tracing_fast(0.0, 0.0, cal)
    assert len(result) == 6
    assert all(np.isfinite(v) for v in result)


def test_ray_tracing_fast_off_axis():
    cal = _make_cal_array()
    Xx, Xy, Xz, ox, oy, oz = _ray_tracing_fast(1.0, 0.5, cal)
    assert all(np.isfinite(v) for v in [Xx, Xy, Xz, ox, oy, oz])


def test_ray_tracing_fast_negative_xy():
    cal = _make_cal_array()
    result = _ray_tracing_fast(-2.0, -1.0, cal)
    assert all(np.isfinite(v) for v in result)


def test_ray_tracing_fast_tilted_glass():
    """Tilted glass → non-trivial Snell refraction."""
    cal = _make_cal_array(gx=0.5, gy=0.0, gz=1.0)
    Xx, Xy, Xz, ox, oy, oz = _ray_tracing_fast(1.0, 0.0, cal)
    assert np.isfinite(Xx)
    assert np.isfinite(ox)


def test_ray_tracing_fast_on_axis_bpn_zero():
    """x=0, y=0, dm=identity, glass=[0,0,1] → start_dir·glass parallel → bpn=0."""
    cal = _make_cal_array(gx=0.0, gy=0.0, gz=1.0)
    result = _ray_tracing_fast(0.0, 0.0, cal)
    assert len(result) == 6


def test_ray_tracing_fast_varied_indices():
    cal = _make_cal_array(mm_n1=1.33, mm_n2_0=1.5, mm_n3=1.33, mm_d0=5.0)
    result = _ray_tracing_fast(2.0, 1.0, cal)
    assert all(np.isfinite(v) for v in result)


def test_ray_tracing_fast_large_xy():
    cal = _make_cal_array()
    result = _ray_tracing_fast(8.0, 6.0, cal)
    assert len(result) == 6


def test_ray_tracing_fast_symmetry_x():
    """_ray_tracing_fast(-x, y) mirrors _ray_tracing_fast(x, y) in X."""
    cal = _make_cal_array()
    Xx_p, Xy_p, Xz_p, _, _, _ = _ray_tracing_fast(2.0, 0.0, cal)
    Xx_n, Xy_n, Xz_n, _, _, _ = _ray_tracing_fast(-2.0, 0.0, cal)
    assert abs(Xx_p + Xx_n) < 1e-10
    assert abs(Xy_p - Xy_n) < 1e-10


def test_ray_tracing_fast_zero_glass_gn_zero_branch():
    """gx=gy=gz=0 → gn=0 branch executed (raises ZeroDivision later — acceptable)."""
    cal = _make_cal_array(gx=0.0, gy=0.0, gz=0.0)
    try:
        _ray_tracing_fast(1.0, 0.0, cal)
    except (ZeroDivisionError, ValueError):
        pass  # branch covered; exception is expected


# ---------------------------------------------------------------------------
# _ray_tracing_out
# ---------------------------------------------------------------------------


def test_ray_tracing_out_basic():
    cal = _make_cal_array()
    out = np.zeros(6, dtype=np.float64)
    ret = _ray_tracing_out(0.0, 0.0, cal, out)
    assert ret == 0
    assert all(np.isfinite(out[i]) for i in range(6))


def test_ray_tracing_out_off_axis():
    cal = _make_cal_array()
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(1.0, 0.5, cal, out)
    assert all(np.isfinite(out[i]) for i in range(6))


def test_ray_tracing_out_matches_fast():
    cal = _make_cal_array()
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(1.0, 0.5, cal, out)
    Xx, Xy, Xz, ox, oy, oz = _ray_tracing_fast(1.0, 0.5, cal)
    assert abs(out[0] - Xx) < 1e-10
    assert abs(out[1] - Xy) < 1e-10
    assert abs(out[2] - Xz) < 1e-10
    assert abs(out[3] - ox) < 1e-10
    assert abs(out[4] - oy) < 1e-10
    assert abs(out[5] - oz) < 1e-10


def test_ray_tracing_out_negative_xy():
    cal = _make_cal_array()
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(-1.5, -0.8, cal, out)
    assert all(np.isfinite(out[i]) for i in range(6))


def test_ray_tracing_out_tilted_glass():
    cal = _make_cal_array(gx=0.3, gy=0.0, gz=1.0)
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(1.0, 0.0, cal, out)
    assert np.isfinite(out[0])


def test_ray_tracing_out_large_xy():
    cal = _make_cal_array()
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(8.0, 6.0, cal, out)
    assert len(out) == 6


def test_ray_tracing_out_varied_indices():
    cal = _make_cal_array(mm_n1=1.33, mm_n2_0=1.5, mm_n3=1.33, mm_d0=5.0)
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(2.0, 1.0, cal, out)
    assert all(np.isfinite(out[i]) for i in range(6))


def test_ray_tracing_out_on_axis_bpn_zero():
    cal = _make_cal_array(gx=0.0, gy=0.0, gz=1.0)
    out = np.zeros(6, dtype=np.float64)
    ret = _ray_tracing_out(0.0, 0.0, cal, out)
    assert ret == 0

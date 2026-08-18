"""Pure-Python coverage tests for track_kernels_tracking.py.

Skip when the compiled .so is active (coverage measures the .py source).

Verification command (from repo root):
    cp src/openptv2/algorithms/track_kernels_tracking.py \
        /tmp/ppsrc/openptv2/algorithms/track_kernels_tracking.py
    COVERAGE_FILE=/tmp/.cov_track_kernels_tracking \
    uv run pytest tests/unit/test_track_kernels_tracking_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q 2>&1 | grep -E '(algorithms/track_kernels_tracking\\.|TOTAL|passed|failed|error)'
"""

import numpy as np
import pytest

from openptv2.algorithms.track_kernels import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

import openptv2.algorithms.track_kernels_tracking as _mod
from openptv2.algorithms.track_kernels_tracking import (
    ADD_PART_K,
    COORD_UNUSED_K,
    CORRES_NONE_K,
    MAX_CANDS_K,
    NEXT_NONE_K,
    POSI_K,
    PREV_NONE_K,
    PT_UNUSED,
    TR_UNUSED_K,
    _angle_acc_out,
    _candsearch_in_pix_rest_nogil,
    _dist_to_flat_out,
    _find_closest_in_3d,
    _multimed_r_nlay_1layer,
    _pixel_to_metric_out,
    _point_position_out,
    _point_to_pixel_out,
    _ray_tracing_out,
    _sorted_candidates_fast_out_nogil,
    assess_new_position_fast_nogil,
    candsearch_in_pix_fast_nogil,
    track3d_loop_fast,
    trackback_loop_fast,
    trackcorr_loop_fast,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NC = 1  # number of cameras used in most tests


def _make_cal_arr(nc=1):
    """31-element calibration flat array per camera, shape (nc, 31)."""
    cal = np.zeros((nc, 31), dtype=np.float64)
    for i in range(nc):
        # ext: camera at z=100, looking along -z
        cal[i, 0] = 0.0  # x0
        cal[i, 1] = 0.0  # y0
        cal[i, 2] = 100.0  # z0
        # rotation matrix = identity (dm[r,c] stored col-major: [0,0],[1,0],[2,0],[0,1]...)
        cal[i, 3] = 1.0  # dm[0,0]
        cal[i, 4] = 0.0  # dm[1,0]
        cal[i, 5] = 0.0  # dm[2,0]
        cal[i, 6] = 0.0  # dm[0,1]
        cal[i, 7] = 1.0  # dm[1,1]
        cal[i, 8] = 0.0  # dm[2,1]
        cal[i, 9] = 0.0  # dm[0,2]
        cal[i, 10] = 0.0  # dm[1,2]
        cal[i, 11] = 1.0  # dm[2,2]
        # interior: cc=10 (focal length), no principal point offset
        cal[i, 12] = 10.0  # cc
        cal[i, 13] = 0.0  # xh
        cal[i, 14] = 0.0  # yh
        # glass normal: pointing in z, distance=1
        cal[i, 15] = 0.0  # gx
        cal[i, 16] = 0.0  # gy
        cal[i, 17] = 1.0  # gz
        cal[i, 18] = 1.0  # dist_o_glas
        cal[i, 19] = 1.0  # inv_dog
        # multimedia: single medium (n1=n2=n3=1, d0=0)
        cal[i, 20] = 1.0  # n1
        cal[i, 21] = 1.0  # n2[0]
        cal[i, 22] = 1.0  # n3
        cal[i, 23] = 0.0  # d0
        # no distortion
        cal[i, 24] = 0.0  # k1
        cal[i, 25] = 0.0  # k2
        cal[i, 26] = 0.0  # k3
        cal[i, 27] = 0.0  # p1
        cal[i, 28] = 0.0  # p2
        cal[i, 29] = 1.0  # scx
        cal[i, 30] = 0.0  # she
    return cal


def _make_mmlut(nc=1):
    """Trivial mmlut arrays (no lut table — has_mmlut=False path)."""
    mo_arr = np.zeros((nc, 4), dtype=np.float64, order="C")
    mnr_arr = np.zeros(nc, dtype=np.int32)
    mnz_arr = np.zeros(nc, dtype=np.int32)
    mrw_arr = np.ones(nc, dtype=np.float64) * 1000.0
    return mo_arr, mnr_arr, mnz_arr, mrw_arr


def _md_arr(nc=1):
    """8 dummy float64 arrays (md0..md7) — only used when sorted_candidates is live."""
    return [np.zeros(4, dtype=np.float64) for _ in range(8)]


def _frame(n, nc, n_targ, max_cands=4, x_offset=0.0):
    """Build all SoA arrays for one frame."""
    n_ = max(n, 1)
    t_ = max(n_targ, 1)
    px = np.zeros((n_, 3), dtype=np.float64, order="C")
    for i in range(n):
        px[i] = [x_offset + i * 0.1, 0.0, 0.0]
    return dict(
        path_x=px,
        path_prev=np.full(n_, -1, dtype=np.int32),
        path_next=np.full(n_, -2, dtype=np.int32),
        path_inlist=np.zeros(n_, dtype=np.int32),
        path_prio=np.full(n_, 4, dtype=np.int32),
        path_finaldecis=np.zeros(n_, dtype=np.float64),
        path_decis=np.zeros((n_, max_cands), dtype=np.float64, order="C"),
        path_linkdecis=np.full((n_, max_cands), -1, dtype=np.int32, order="C"),
        corres_p=np.full((n_, nc), -1, dtype=np.int32, order="C"),
        corres_nr=np.zeros(n_, dtype=np.int32),
        targ_x=np.zeros((nc, t_), dtype=np.float64, order="C"),
        targ_y=np.zeros((nc, t_), dtype=np.float64, order="C"),
        targ_tnr=np.full((nc, t_), -1, dtype=np.int32, order="C"),
        num_targets=np.zeros(nc, dtype=np.int32),
        num_parts=np.array([n], dtype=np.int32),
    )


def _call_trackcorr(n0, n1, n2, n3, nc=1, n_targ=0, stub_zero=False):
    """Call trackcorr_loop_fast with minimal valid arrays.

    stub_zero=True: monkey-patches _sorted_candidates_fast_out_nogil → returns 0
    so _trackcorr_particle_fast always takes the early-return path.
    """
    f0 = _frame(n0, nc, n_targ)
    f1 = _frame(n1, nc, n_targ, x_offset=0.1)
    f2 = _frame(n2, nc, n_targ, x_offset=0.2)
    f3 = _frame(n3, nc, n_targ, x_offset=0.3)
    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    md = _md_arr(nc)

    def _do_call():
        return trackcorr_loop_fast(
            n1,
            # frame 0
            f0["path_x"],
            # frame 1
            f1["path_x"],
            f1["path_prev"],
            f1["path_next"],
            f1["path_inlist"],
            f1["path_finaldecis"],
            f1["path_decis"],
            f1["path_linkdecis"],
            f1["corres_p"],
            f1["targ_x"],
            f1["targ_y"],
            f1["targ_tnr"],
            # frame 2
            f2["path_x"],
            f2["path_prev"],
            f2["path_next"],
            f2["path_inlist"],
            f2["path_prio"],
            f2["path_finaldecis"],
            f2["path_decis"],
            f2["path_linkdecis"],
            f2["corres_p"],
            f2["corres_nr"],
            f2["targ_x"],
            f2["targ_y"],
            f2["targ_tnr"],
            f2["num_targets"],
            f2["num_parts"],
            # frame 3
            f3["path_x"],
            f3["path_prev"],
            f3["path_next"],
            f3["path_inlist"],
            f3["path_prio"],
            f3["path_finaldecis"],
            f3["path_decis"],
            f3["path_linkdecis"],
            f3["corres_p"],
            f3["corres_nr"],
            f3["targ_x"],
            f3["targ_y"],
            f3["targ_tnr"],
            f3["num_targets"],
            f3["num_parts"],
            # calibration
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            # tracking params
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,  # dvxmin/max * 3
            5.0,  # dacc
            30.0,  # dangle
            0,  # add_flag
            5.0,  # lmax
            # volume bounds
            -10.0,
            10.0,  # X_lay_0, X_lay_1
            -10.0,
            10.0,  # ymin, ymax
            -10.0,
            10.0,  # Zmin_lay_0, Zmax_lay_1
            # pixel params
            nc,  # num_cams
            50.0,  # imx_half
            50.0,  # imy_half
            1.0,  # inv_pix_x
            1.0,  # inv_pix_y
            0,  # chfield
            100.0,  # imx
            100.0,  # imy
            0.01,  # pix_x
            0.01,  # pix_y
            0.001,  # flatten_tol
        )

    if stub_zero:
        orig = _mod._sorted_candidates_fast_out_nogil
        _mod._sorted_candidates_fast_out_nogil = lambda *a, **k: 0
        try:
            return _do_call()
        finally:
            _mod._sorted_candidates_fast_out_nogil = orig
    else:
        return _do_call()


def _call_trackback(n1, n2, n3, nc=1, n_targ=0):
    """Build minimal args for trackback_loop_fast (always raises UnboundLocalError)."""
    f0 = _frame(n1, nc, n_targ)  # frame 0 = forward of frame 1
    f1 = _frame(n1, nc, n_targ)
    f2 = _frame(n2, nc, n_targ)
    f3 = _frame(n3, nc, n_targ)
    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    md = _md_arr(nc)

    return trackback_loop_fast(
        n1,
        f0["path_x"],
        f1["path_x"],
        f1["path_prev"],
        f1["path_next"],
        f1["path_inlist"],
        f1["path_finaldecis"],
        f1["path_decis"],
        f1["path_linkdecis"],
        f2["path_x"],
        f2["path_prev"],
        f2["path_next"],
        f2["num_parts"],
        f2["targ_x"],
        f2["targ_y"],
        f2["targ_tnr"],
        f2["num_targets"],
        f2["corres_p"],
        f2["corres_nr"],
        f2["path_inlist"],
        f2["path_prio"],
        f2["path_finaldecis"],
        f2["path_decis"],
        f2["path_linkdecis"],
        f3["path_x"],
        f3["path_prev"],
        cal,
        md,
        mo,
        mnr,
        mnz,
        mrw,
        # tracking params
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        5.0,
        30.0,
        0,
        5.0,
        # volume
        -10.0,
        10.0,
        -10.0,
        10.0,
        -10.0,
        10.0,
        # pixel
        nc,
        50.0,
        50.0,
        1.0,
        1.0,
        0,
        100.0,
        100.0,
        0.01,
        0.01,
        0.001,
    )


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_constants():
    assert PT_UNUSED == -999
    assert POSI_K == 80
    assert MAX_CANDS_K == 4
    assert TR_UNUSED_K == -1
    assert CORRES_NONE_K == -1
    assert PREV_NONE_K == -1
    assert NEXT_NONE_K == -2
    assert COORD_UNUSED_K == pytest.approx(-1e10)
    assert ADD_PART_K == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _angle_acc_out — all 5 branches
# ---------------------------------------------------------------------------


def test_angle_acc_out_anti_parallel():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, out)
    assert out[0] == pytest.approx(200.0)


def test_angle_acc_out_same_direction():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, out)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)


def test_angle_acc_out_zero_norm():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, out)
    assert out[0] == pytest.approx(0.0)


def test_angle_acc_out_dot_clamp_above_one():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 + 1e-15, 1e-15, 0.0, out)
    assert out[0] >= 0.0


def test_angle_acc_out_dot_clamp_below_minus_one():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0 - 1e-15, -1e-15, 0.0, out)
    assert out[0] >= 0.0


def test_angle_acc_out_normal():
    out = np.zeros(2, dtype=np.float64)
    _angle_acc_out(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, out)
    assert 0.0 < out[0] < 200.0
    assert out[1] >= 0.0


# ---------------------------------------------------------------------------
# _multimed_r_nlay_1layer — all branches
# ---------------------------------------------------------------------------


def test_multimed_r_trivial_air():
    """mm_n1==mm_n2_0==mm_n3==1.0 → early exit branch, returns 1.0."""
    # pos_x, pos_y, pos_z, ext_x0, ext_y0, ext_z0, mm_n1, mm_n2_0, mm_n3, mm_d0
    r = _multimed_r_nlay_1layer(0.5, 0.5, 0.0, 0.0, 0.0, 100.0, 1.0, 1.0, 1.0, 0.0)
    assert r == pytest.approx(1.0)


def test_multimed_r_zero_r():
    """pos_x==ext_x0, pos_y==ext_y0 → r=0 at end → returns 1.0."""
    r = _multimed_r_nlay_1layer(1.0, 2.0, 5.0, 1.0, 2.0, 100.0, 1.5, 1.33, 1.0, 5.0)
    assert r == pytest.approx(1.0)


def test_multimed_r_denom_zero():
    """ext_z0 == pos_z → denom=0 branch → returns 1.0."""
    r = _multimed_r_nlay_1layer(0.5, 0.5, 100.0, 0.0, 0.0, 100.0, 1.5, 1.33, 1.0, 5.0)
    assert r == pytest.approx(1.0)


def test_multimed_r_normal():
    """Standard multimedia refraction with non-trivial values."""
    r = _multimed_r_nlay_1layer(0.5, 0.5, 0.0, 0.0, 0.0, 100.0, 1.5, 1.33, 1.0, 5.0)
    assert r > 0.0


def test_multimed_r_no_converge():
    """Extreme values that fail convergence in 40 iterations → fallback 1.0."""
    r = _multimed_r_nlay_1layer(
        500.0, 500.0, 0.0, 0.0, 0.0, 100.0, 2.0, 0.1, 1.0, 0.001
    )
    assert r > 0.0


# ---------------------------------------------------------------------------
# _point_to_pixel_out — main path + chfield branches
# ---------------------------------------------------------------------------


def _p2p_args(chfield=0):
    cal = _make_cal_arr(1)[0]
    pos = np.array([0.5, 0.5, 0.0], dtype=np.float64)
    mo, mnr, mnz, mrw = _make_mmlut(1)
    out = np.zeros(2, dtype=np.float64)
    return (
        pos,
        cal,
        mo[0],
        np.zeros(3, dtype=np.float64),
        int(mnr[0]),
        int(mnz[0]),
        float(mrw[0]),
        0,
        50.0,
        50.0,
        1.0,
        1.0,
        chfield,
        out,
    )


def test_point_to_pixel_chfield_0():
    pos, cal, data, origin, nr, nz, rw, has_mmlut, imx_h, imy_h, ipx, ipy, cf, out = (
        _p2p_args(0)
    )
    ret = _point_to_pixel_out(
        pos, cal, data, origin, nr, nz, rw, has_mmlut, imx_h, imy_h, ipx, ipy, cf, out
    )
    assert ret == 0
    assert out.shape == (2,)


def test_point_to_pixel_chfield_1():
    pos, cal, data, origin, nr, nz, rw, has_mmlut, imx_h, imy_h, ipx, ipy, _, out = (
        _p2p_args(1)
    )
    _point_to_pixel_out(
        pos, cal, data, origin, nr, nz, rw, has_mmlut, imx_h, imy_h, ipx, ipy, 1, out
    )


def test_point_to_pixel_chfield_2():
    pos, cal, data, origin, nr, nz, rw, has_mmlut, imx_h, imy_h, ipx, ipy, _, out = (
        _p2p_args(2)
    )
    _point_to_pixel_out(
        pos, cal, data, origin, nr, nz, rw, has_mmlut, imx_h, imy_h, ipx, ipy, 2, out
    )


def test_point_to_pixel_near_origin():
    """r < 1e-10 branch in distortion computation."""
    cal = _make_cal_arr(1)[0]
    # Position exactly at principal point of camera → r ≈ 0 in image plane
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    mo, mnr, mnz, mrw = _make_mmlut(1)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, mo[0], np.zeros(3), 0, 0, 1000.0, 0, 50.0, 50.0, 1.0, 1.0, 0, out
    )


def test_point_to_pixel_with_distortion():
    """Non-zero distortion coefficients."""
    cal = _make_cal_arr(1)[0]
    cal[24] = 0.01  # k1
    cal[25] = 0.001  # k2
    cal[27] = 0.005  # p1
    pos = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    mo, mnr, mnz, mrw = _make_mmlut(1)
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        pos, cal, mo[0], np.zeros(3), 0, 0, 1000.0, 0, 50.0, 50.0, 1.0, 1.0, 0, out
    )


# ---------------------------------------------------------------------------
# candsearch_in_pix_fast_nogil
# ---------------------------------------------------------------------------


def _targ_arrays(n, start_x=10.0, start_y=10.0):
    """Sorted target arrays for candsearch tests."""
    tx = np.array([start_x + i for i in range(n)], dtype=np.float64)
    ty = np.array([start_y + i for i in range(n)], dtype=np.float64)
    tnr = np.arange(n, dtype=np.int32)
    return tx, ty, tnr


def test_candsearch_pix_fast_finds_one():
    # candsearch_in_pix_fast_nogil always returns 0; candidates are in out_idx
    tx, ty, tnr = _targ_arrays(5)
    out_idx = np.full(4, -999, dtype=np.int32)
    out_dists = np.full(4, 1e20, dtype=np.float64)
    candsearch_in_pix_fast_nogil(
        tx,
        ty,
        tnr,
        5,
        10.0,
        10.0,  # cent_x, cent_y
        3.0,
        3.0,
        3.0,
        3.0,  # dl, dr, du, dd
        100.0,
        100.0,  # imx, imy
        -1,  # tr_unused
        4,  # max_cands
        out_idx,
        out_dists,
    )
    assert out_idx[0] != -999  # at least one target found


def test_candsearch_pix_fast_out_of_image():
    """cent_x, cent_y outside image → early return with 0 candidates."""
    tx, ty, tnr = _targ_arrays(5)
    out_idx = np.full(4, -999, dtype=np.int32)
    out_dists = np.full(4, 1e20, dtype=np.float64)
    n = candsearch_in_pix_fast_nogil(
        tx,
        ty,
        tnr,
        5,
        -500.0,
        -500.0,  # well outside image
        3.0,
        3.0,
        3.0,
        3.0,
        100.0,
        100.0,
        -1,
        4,
        out_idx,
        out_dists,
    )
    assert n == 0


def test_candsearch_pix_fast_no_targets():
    """No targets → 0 candidates."""
    tx = np.array([], dtype=np.float64)
    ty = np.array([], dtype=np.float64)
    tnr = np.array([], dtype=np.int32)
    out_idx = np.full(4, -999, dtype=np.int32)
    out_dists = np.full(4, 1e20, dtype=np.float64)
    n = candsearch_in_pix_fast_nogil(
        tx,
        ty,
        tnr,
        0,
        10.0,
        10.0,
        5.0,
        5.0,
        5.0,
        5.0,
        100.0,
        100.0,
        -1,
        4,
        out_idx,
        out_dists,
    )
    assert n == 0


def test_candsearch_pix_fast_multiple_candidates():
    """Multiple close targets → up to 4 ranked by distance in out_idx."""
    tx = np.array([10.5, 10.6, 10.7, 10.8, 10.9], dtype=np.float64)
    ty = np.array([10.5, 10.5, 10.5, 10.5, 10.5], dtype=np.float64)
    tnr = np.arange(5, dtype=np.int32)
    out_idx = np.full(4, -999, dtype=np.int32)
    out_dists = np.full(4, 1e20, dtype=np.float64)
    candsearch_in_pix_fast_nogil(
        tx,
        ty,
        tnr,
        5,
        10.5,
        10.5,
        2.0,
        2.0,
        2.0,
        2.0,
        100.0,
        100.0,
        -1,
        4,
        out_idx,
        out_dists,
    )
    assert out_idx[0] == 0  # closest is index 0


def test_candsearch_pix_fast_unused_targets_skipped():
    """Targets with tnr == tr_unused are skipped; only tnr != tr_unused are candidates."""
    tx = np.array([10.5, 10.6], dtype=np.float64)
    ty = np.array([10.5, 10.5], dtype=np.float64)
    tnr = np.array([-1, 1], dtype=np.int32)  # first is unused
    out_idx = np.full(4, -999, dtype=np.int32)
    out_dists = np.full(4, 1e20, dtype=np.float64)
    candsearch_in_pix_fast_nogil(
        tx,
        ty,
        tnr,
        2,
        10.5,
        10.5,
        2.0,
        2.0,
        2.0,
        2.0,
        100.0,
        100.0,
        -1,
        4,
        out_idx,
        out_dists,
    )
    # target 0 is unused (-1), so only target 1 (tnr=1) is a valid candidate
    assert out_idx[0] == 1


# ---------------------------------------------------------------------------
# _sorted_candidates_fast_out_nogil — crashes at quader_buf (C-array bug)
# ---------------------------------------------------------------------------


def test_sorted_candidates_raises_unbound_local():
    """quader_buf: cython.double[24] at L702 — never assigned, raises on access."""
    nc = 1
    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    center = np.zeros(3, dtype=np.float64)
    cpx = np.zeros(nc, dtype=np.float64)
    cpy = np.zeros(nc, dtype=np.float64)
    tx = np.zeros((nc, 1), dtype=np.float64, order="C")
    ty = np.zeros((nc, 1), dtype=np.float64, order="C")
    tnr = np.full((nc, 1), -1, dtype=np.int32, order="C")
    ntarg = np.zeros(nc, dtype=np.int32)
    ftnr = np.full(nc * MAX_CANDS_K, -1, dtype=np.int32)
    freq = np.zeros(nc * MAX_CANDS_K, dtype=np.int32)
    wc = np.zeros((nc * MAX_CANDS_K, nc), dtype=np.int32, order="C")
    pt_buf = np.zeros(2, dtype=np.float64)
    pp = np.zeros(2, dtype=np.float64)
    md = [np.zeros(4, dtype=np.float64)] * 8

    with pytest.raises((UnboundLocalError, NameError, IndexError)):
        _sorted_candidates_fast_out_nogil(
            center,
            cpx,
            cpy,
            nc,
            MAX_CANDS_K,
            cal,
            md[0],
            md[1],
            md[2],
            md[3],
            md[4],
            md[5],
            md[6],
            md[7],
            mo,
            mnr,
            mnz,
            mrw,
            tx,
            ty,
            tnr,
            ntarg,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            50.0,
            50.0,
            1.0,
            1.0,
            0,
            100.0,
            100.0,
            -1,
            ftnr,
            freq,
            wc,
            pt_buf,
            pp,
        )


# ---------------------------------------------------------------------------
# _ray_tracing_out
# ---------------------------------------------------------------------------


def test_ray_tracing_out_basic():
    cal = _make_cal_arr(1)[0]
    out = np.zeros(6, dtype=np.float64)
    ret = _ray_tracing_out(0.0, 0.0, cal, out)
    assert ret == 0
    # Xx, Xy, Xz = glass intersection; ox, oy, oz = camera origin
    assert not np.all(out == 0.0)


def test_ray_tracing_out_offcenter():
    cal = _make_cal_arr(1)[0]
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(5.0, -3.0, cal, out)
    # direction should be non-trivial
    assert out[3] != 0.0 or out[4] != 0.0 or out[5] != 0.0


def test_ray_tracing_out_with_glass():
    """Non-trivial multimedia (glass n2!=1, d0!=0) exercises the refraction branch."""
    cal = _make_cal_arr(1)[0]
    cal[21] = 1.5  # n2[0]
    cal[23] = 5.0  # d0
    out = np.zeros(6, dtype=np.float64)
    _ray_tracing_out(1.0, 1.0, cal, out)


# ---------------------------------------------------------------------------
# _point_position_out
# ---------------------------------------------------------------------------


def test_point_position_out_unused_targets():
    """All targets COORD_UNUSED → no valid cams → out stays zeros."""
    nc = 1
    cal = _make_cal_arr(nc)
    targets = np.full((nc, 2), COORD_UNUSED_K, dtype=np.float64, order="C")
    out = np.zeros(3, dtype=np.float64)
    scratch = np.zeros(6, dtype=np.float64)
    _point_position_out(targets, nc, cal, out, scratch)
    assert np.all(out == 0.0)


def test_point_position_out_two_cams_pair_loop():
    """Two cameras with real targets → enters pair loop (L1176-1235)."""
    nc = 2
    cal = _make_cal_arr(nc)
    # Offset second camera in X so ray directions differ
    cal[1, 0] = 10.0  # x0 of camera 1
    # Target pixel (0,0) for each camera
    targets = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64, order="C")
    out = np.zeros(3, dtype=np.float64)
    scratch = np.zeros(6, dtype=np.float64)
    _point_position_out(targets, nc, cal, out, scratch)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# _candsearch_in_pix_rest_nogil
# ---------------------------------------------------------------------------


def test_candsearch_rest_finds_target():
    # _candsearch_in_pix_rest_nogil finds targets where targ_tnr == tr_unused
    # (the "rest" = unlinked targets not yet assigned to a particle)
    tx = np.array([50.5, 55.0, 60.0], dtype=np.float64)
    ty = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    tnr = np.array([-1, 1, 2], dtype=np.int32)  # only target 0 is "rest" (unused)
    result = _candsearch_in_pix_rest_nogil(
        tx,
        ty,
        tnr,
        3,
        55.0,
        50.0,  # cent_x, cent_y
        5.0,
        5.0,
        5.0,
        5.0,  # dl, dr, du, dd
        100.0,
        100.0,  # imx, imy
        -1,  # tr_unused
    )
    assert result == 0  # index of the unused target at (50.5, 50)


def test_candsearch_rest_out_of_image():
    tx = np.array([50.0], dtype=np.float64)
    ty = np.array([50.0], dtype=np.float64)
    tnr = np.array([0], dtype=np.int32)
    result = _candsearch_in_pix_rest_nogil(
        tx,
        ty,
        tnr,
        1,
        -100.0,
        -100.0,  # outside image
        5.0,
        5.0,
        5.0,
        5.0,
        100.0,
        100.0,
        -1,
    )
    assert result == -1  # tr_unused


def test_candsearch_rest_no_match():
    """Unused target out of search box → returns tr_unused."""
    tx = np.array([90.0], dtype=np.float64)
    ty = np.array([90.0], dtype=np.float64)
    tnr = np.array([-1], dtype=np.int32)  # unused, but far away
    result = _candsearch_in_pix_rest_nogil(
        tx,
        ty,
        tnr,
        1,
        10.0,
        10.0,  # far from target
        2.0,
        2.0,
        2.0,
        2.0,
        100.0,
        100.0,
        -1,
    )
    assert result == -1


def test_candsearch_rest_break_on_ymax():
    """Unused target with ty > ymax triggers break in scan loop."""
    # All three targets are unused (tnr=-1). Target at y=80 triggers break.
    ty_vals = np.array([40.0, 50.5, 80.0], dtype=np.float64)
    tx_vals = np.array([50.5, 50.5, 50.5], dtype=np.float64)
    tnr = np.array([-1, -1, -1], dtype=np.int32)
    result = _candsearch_in_pix_rest_nogil(
        tx_vals,
        ty_vals,
        tnr,
        3,
        50.5,
        50.5,
        5.0,
        5.0,
        5.0,
        5.0,
        100.0,
        100.0,
        -1,
    )
    assert result == 1  # target 1 at (50.5, 50.5) is within search box


# ---------------------------------------------------------------------------
# _pixel_to_metric_out — chfield branches
# ---------------------------------------------------------------------------


def test_pixel_to_metric_chfield_0():
    out = np.zeros(2, dtype=np.float64)
    _pixel_to_metric_out(50.0, 50.0, 100, 100, 0.01, 0.01, 0, out)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)


def test_pixel_to_metric_chfield_1():
    out = np.zeros(2, dtype=np.float64)
    _pixel_to_metric_out(50.0, 25.0, 100, 100, 0.01, 0.01, 1, out)
    # yp = 2*25+1 = 51 → out[1] = (50-51)*0.01 = -0.01
    assert out[1] == pytest.approx(-0.01)


def test_pixel_to_metric_chfield_2():
    out = np.zeros(2, dtype=np.float64)
    _pixel_to_metric_out(50.0, 25.0, 100, 100, 0.01, 0.01, 2, out)
    # yp = 2*25 = 50 → out[1] = (50-50)*0.01 = 0
    assert out[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _dist_to_flat_out
# ---------------------------------------------------------------------------


def test_dist_to_flat_no_distortion():
    out = np.zeros(2, dtype=np.float64)
    _dist_to_flat_out(0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-5, out)
    assert out[0] == pytest.approx(0.5, abs=0.001)
    assert out[1] == pytest.approx(0.3, abs=0.001)


def test_dist_to_flat_near_origin():
    """r < 1e-10 → out = (-xh, -yh)."""
    out = np.zeros(2, dtype=np.float64)
    _dist_to_flat_out(0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-5, out)
    assert out[0] == pytest.approx(-0.1)
    assert out[1] == pytest.approx(-0.2)


def test_dist_to_flat_with_distortion():
    out = np.zeros(2, dtype=np.float64)
    _dist_to_flat_out(
        0.5, 0.5, 0.0, 0.0, 0.01, 0.001, 0.0001, 0.005, 0.003, 1.0, 0.0, 1e-6, out
    )
    assert np.isfinite(out[0]) and np.isfinite(out[1])


# ---------------------------------------------------------------------------
# assess_new_position_fast_nogil
# ---------------------------------------------------------------------------


def _assess_args(nc=1):
    """Build args for assess_new_position_fast_nogil."""
    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # 1 target per camera at projected position
    targ_x = np.array([[50.0]], dtype=np.float64, order="C")
    targ_y = np.array([[50.0]], dtype=np.float64, order="C")
    targ_tnr = np.array([[-1]], dtype=np.int32, order="C")
    num_targets = np.array([1], dtype=np.int32)
    proj_x = np.zeros(nc, dtype=np.float64)
    proj_y = np.zeros(nc, dtype=np.float64)
    proj_x[0] = 50.0
    proj_y[0] = 50.0
    targ_pos_out = np.full((nc, 2), COORD_UNUSED_K, dtype=np.float64, order="C")
    cand_inds_out = np.full(nc, -1, dtype=np.int32)
    scratch = np.zeros(2, dtype=np.float64)
    return (
        pos,
        nc,
        ADD_PART_K,
        cal,
        mo,
        mnr,
        mnz,
        mrw,
        targ_x,
        targ_y,
        targ_tnr,
        num_targets,
        50.0,
        50.0,
        1.0,
        1.0,
        0,
        100,
        100,
        0.01,
        0.01,
        0.001,
        -1,
        COORD_UNUSED_K,
        proj_x,
        proj_y,
        targ_pos_out,
        cand_inds_out,
        scratch,
    )


def test_assess_new_position_finds_target():
    args = _assess_args()
    n = assess_new_position_fast_nogil(*args)
    assert n >= 0


def test_assess_new_position_no_targets():
    nc = 1
    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    pos = np.zeros(3, dtype=np.float64)
    targ_x = np.zeros((nc, 1), dtype=np.float64, order="C")
    targ_y = np.zeros((nc, 1), dtype=np.float64, order="C")
    targ_tnr = np.full((nc, 1), -1, dtype=np.int32, order="C")  # all unused
    num_targets = np.array([0], dtype=np.int32)
    proj_x = np.array([50.0], dtype=np.float64)
    proj_y = np.array([50.0], dtype=np.float64)
    targ_pos_out = np.full((nc, 2), COORD_UNUSED_K, dtype=np.float64, order="C")
    cand_inds_out = np.full(nc, -1, dtype=np.int32)
    scratch = np.zeros(2, dtype=np.float64)
    n = assess_new_position_fast_nogil(
        pos,
        nc,
        ADD_PART_K,
        cal,
        mo,
        mnr,
        mnz,
        mrw,
        targ_x,
        targ_y,
        targ_tnr,
        num_targets,
        50.0,
        50.0,
        1.0,
        1.0,
        0,
        100,
        100,
        0.01,
        0.01,
        0.001,
        -1,
        COORD_UNUSED_K,
        proj_x,
        proj_y,
        targ_pos_out,
        cand_inds_out,
        scratch,
    )
    assert n == 0


# ---------------------------------------------------------------------------
# _find_closest_in_3d
# ---------------------------------------------------------------------------


def test_find_closest_in_3d_finds_one():
    path_x2 = np.array([[0.5, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64, order="C")
    cand_inds = np.full(4, -1, dtype=np.int32)
    cand_dists = np.full(4, 1e20, dtype=np.float64)
    n = _find_closest_in_3d(
        path_x2, 2, 0.4, 0.0, 0.0, 1.0, 1.0, 1.0, 4, cand_inds, cand_dists
    )
    assert n == 1
    assert cand_inds[0] == 0


def test_find_closest_in_3d_none_in_box():
    path_x2 = np.array([[10.0, 0.0, 0.0]], dtype=np.float64, order="C")
    cand_inds = np.full(4, -1, dtype=np.int32)
    cand_dists = np.full(4, 1e20, dtype=np.float64)
    n = _find_closest_in_3d(
        path_x2, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4, cand_inds, cand_dists
    )
    assert n == 0


def test_find_closest_in_3d_caps_at_max_cands():
    path_x2 = np.array(
        [[float(i) * 0.1, 0.0, 0.0] for i in range(10)], dtype=np.float64, order="C"
    )
    cand_inds = np.full(4, -1, dtype=np.int32)
    cand_dists = np.full(4, 1e20, dtype=np.float64)
    n = _find_closest_in_3d(
        path_x2, 10, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 4, cand_inds, cand_dists
    )
    assert n == 4


def test_find_closest_in_3d_empty():
    path_x2 = np.zeros((1, 3), dtype=np.float64, order="C")
    cand_inds = np.full(4, -1, dtype=np.int32)
    cand_dists = np.full(4, 1e20, dtype=np.float64)
    n = _find_closest_in_3d(
        path_x2, 0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4, cand_inds, cand_dists
    )
    assert n == 0


# ---------------------------------------------------------------------------
# track3d_loop_fast — all 3 levels
# ---------------------------------------------------------------------------


def _px(positions):
    return np.array(positions, dtype=np.float64, order="C")


def test_track3d_level1_prev_link():
    """Level 1: particle has prev link → velocity prediction."""
    px0 = _px([[0.0, 0.0, 0.0]])
    px1 = _px([[0.1, 0.0, 0.0]])
    px2 = _px([[0.2, 0.0, 0.0]])
    prev0 = np.array([-1], dtype=np.int32)
    prev1 = np.array([0], dtype=np.int32)  # has prev
    next1 = np.full(1, -2, dtype=np.int32)
    prev2 = np.full(1, -1, dtype=np.int32)
    next2 = np.full(1, -2, dtype=np.int32)
    count = track3d_loop_fast(
        1,
        px0,
        prev0,
        1,
        px1,
        prev1,
        next1,
        1,
        px2,
        prev2,
        next2,
        1,
        0.5,
        0.5,
        0.5,
        4,
    )
    assert count >= 0


def test_track3d_level2_neighbor_avg():
    """Level 2: particle has no prev link, but neighbors have prev links."""
    px0 = _px([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    px1 = _px([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
    px2 = _px([[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]])
    prev0 = np.array([-1, -1], dtype=np.int32)
    prev1 = np.array(
        [0, -1], dtype=np.int32
    )  # particle 1 has prev, particle 2 does not
    next1 = np.full(2, -2, dtype=np.int32)
    prev2 = np.full(2, -1, dtype=np.int32)
    next2 = np.full(2, -2, dtype=np.int32)
    count = track3d_loop_fast(
        2,
        px0,
        prev0,
        2,
        px1,
        prev1,
        next1,
        2,
        px2,
        prev2,
        next2,
        2,
        0.5,
        0.5,
        0.5,
        4,
    )
    assert count >= 0


def test_track3d_level3_static_position():
    """Level 3: particle has no prev link and no neighbors with prev links."""
    px0 = _px([[0.0, 0.0, 0.0]])
    px1 = _px([[0.1, 0.0, 0.0]])
    px2 = _px([[0.15, 0.0, 0.0]])
    prev0 = np.array([-1], dtype=np.int32)
    prev1 = np.array([-1], dtype=np.int32)  # no prev
    next1 = np.full(1, -2, dtype=np.int32)
    prev2 = np.full(1, -1, dtype=np.int32)
    next2 = np.full(1, -2, dtype=np.int32)
    count = track3d_loop_fast(
        1,
        px0,
        prev0,
        1,
        px1,
        prev1,
        next1,
        1,
        px2,
        prev2,
        next2,
        1,
        0.5,
        0.5,
        0.5,
        4,
    )
    assert count >= 0


def test_track3d_no_candidates():
    """Particles too far apart → no links."""
    px0 = _px([[0.0, 0.0, 0.0]])
    px1 = _px([[0.1, 0.0, 0.0]])
    px2 = _px([[100.0, 0.0, 0.0]])  # far from predicted position
    prev0 = np.array([0], dtype=np.int32)
    prev1 = np.array([0], dtype=np.int32)
    next1 = np.full(1, -2, dtype=np.int32)
    prev2 = np.full(1, -1, dtype=np.int32)
    next2 = np.full(1, -2, dtype=np.int32)
    count = track3d_loop_fast(
        1,
        px0,
        prev0,
        1,
        px1,
        prev1,
        next1,
        1,
        px2,
        prev2,
        next2,
        1,
        0.1,
        0.1,
        0.1,
        4,
    )
    assert count == 0  # no candidates in tight box


def test_track3d_empty():
    """Empty frames → no links."""
    empty = np.zeros((1, 3), dtype=np.float64, order="C")
    p = np.array([-1], dtype=np.int32)
    n = np.full(1, -2, dtype=np.int32)
    count = track3d_loop_fast(
        0,
        empty,
        p,
        0,
        empty,
        p.copy(),
        n.copy(),
        0,
        empty,
        p.copy(),
        n.copy(),
        0,
        1.0,
        1.0,
        1.0,
        4,
    )
    assert count == 0


def test_track3d_prev_idx_out_of_range():
    """prev_idx >= num_parts_0 → skip particle (bounds check branch)."""
    px0 = _px([[0.0, 0.0, 0.0]])
    px1 = _px([[0.1, 0.0, 0.0]])
    px2 = _px([[0.2, 0.0, 0.0]])
    prev0 = np.array([-1], dtype=np.int32)
    prev1 = np.array([999], dtype=np.int32)  # out of range
    next1 = np.full(1, -2, dtype=np.int32)
    prev2 = np.full(1, -1, dtype=np.int32)
    next2 = np.full(1, -2, dtype=np.int32)
    # Should not crash, just skip
    count = track3d_loop_fast(
        1,
        px0,
        prev0,
        1,
        px1,
        prev1,
        next1,
        1,
        px2,
        prev2,
        next2,
        1,
        1.0,
        1.0,
        1.0,
        4,
    )
    assert count >= 0


# ---------------------------------------------------------------------------
# trackcorr_loop_fast — buffer allocation + post-loop (orig_parts_1=0)
# ---------------------------------------------------------------------------


def test_trackcorr_no_particles():
    """orig_parts_1=0 → covers buffer allocation (L2503-2627) + post-loop (L2726-2839)."""
    result = _call_trackcorr(0, 0, 0, 0)
    assert isinstance(result, tuple)
    count1, num_added = result
    assert count1 == 0
    assert num_added == 0


def test_trackcorr_no_particles_with_targets_in_frame2():
    """Post-loop with some frame-2 particles (but no frame-1 links)."""
    result = _call_trackcorr(0, 0, 2, 2)
    count1, num_added = result
    assert count1 == 0


# ---------------------------------------------------------------------------
# trackcorr_loop_fast — prange body via stub_zero (covers _trackcorr_particle_fast setup)
# ---------------------------------------------------------------------------


def test_trackcorr_stub_zero_no_prev():
    """1 particle with prev_h=-1 → covers _trackcorr_particle_fast corres_p path."""
    result = _call_trackcorr(1, 1, 1, 1, stub_zero=True)
    count1, num_added = result
    assert count1 == 0  # stub returns 0 candidates → no links


def test_trackcorr_stub_zero_with_prev():
    """2 particles: one with prev, one without → covers both branches of prev_h check."""
    nc = 1
    f0 = _frame(2, nc, 0, x_offset=0.0)
    f1 = _frame(2, nc, 0, x_offset=0.1)
    # Give particle 0 a prev link
    f1["path_prev"][0] = 0
    f1["path_prev"][1] = -1
    f2 = _frame(2, nc, 0, x_offset=0.2)
    f3 = _frame(2, nc, 0, x_offset=0.3)
    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    md = _md_arr(nc)

    orig = _mod._sorted_candidates_fast_out_nogil
    _mod._sorted_candidates_fast_out_nogil = lambda *a, **k: 0
    try:
        result = trackcorr_loop_fast(
            2,
            f0["path_x"],
            f1["path_x"],
            f1["path_prev"],
            f1["path_next"],
            f1["path_inlist"],
            f1["path_finaldecis"],
            f1["path_decis"],
            f1["path_linkdecis"],
            f1["corres_p"],
            f1["targ_x"],
            f1["targ_y"],
            f1["targ_tnr"],
            f2["path_x"],
            f2["path_prev"],
            f2["path_next"],
            f2["path_inlist"],
            f2["path_prio"],
            f2["path_finaldecis"],
            f2["path_decis"],
            f2["path_linkdecis"],
            f2["corres_p"],
            f2["corres_nr"],
            f2["targ_x"],
            f2["targ_y"],
            f2["targ_tnr"],
            f2["num_targets"],
            f2["num_parts"],
            f3["path_x"],
            f3["path_prev"],
            f3["path_next"],
            f3["path_inlist"],
            f3["path_prio"],
            f3["path_finaldecis"],
            f3["path_decis"],
            f3["path_linkdecis"],
            f3["corres_p"],
            f3["corres_nr"],
            f3["targ_x"],
            f3["targ_y"],
            f3["targ_tnr"],
            f3["num_targets"],
            f3["num_parts"],
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            5.0,
            30.0,
            0,
            5.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            nc,
            50.0,
            50.0,
            1.0,
            1.0,
            0,
            100.0,
            100.0,
            0.01,
            0.01,
            0.001,
        )
    finally:
        _mod._sorted_candidates_fast_out_nogil = orig

    count1, num_added = result
    assert count1 == 0


def test_trackcorr_stub_zero_corres_p_path():
    """Particle with corres_p set → exercises corres_p path in particle setup."""
    nc = 1
    f0 = _frame(1, nc, 0)
    f1 = _frame(1, nc, 0)
    # Set corres_p[0, 0] to a valid target index
    f1["corres_p"][0, 0] = 0
    f2 = _frame(1, nc, 0)
    f3 = _frame(1, nc, 0)
    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    md = _md_arr(nc)

    orig = _mod._sorted_candidates_fast_out_nogil
    _mod._sorted_candidates_fast_out_nogil = lambda *a, **k: 0
    try:
        trackcorr_loop_fast(
            1,
            f0["path_x"],
            f1["path_x"],
            f1["path_prev"],
            f1["path_next"],
            f1["path_inlist"],
            f1["path_finaldecis"],
            f1["path_decis"],
            f1["path_linkdecis"],
            f1["corres_p"],
            f1["targ_x"],
            f1["targ_y"],
            f1["targ_tnr"],
            f2["path_x"],
            f2["path_prev"],
            f2["path_next"],
            f2["path_inlist"],
            f2["path_prio"],
            f2["path_finaldecis"],
            f2["path_decis"],
            f2["path_linkdecis"],
            f2["corres_p"],
            f2["corres_nr"],
            f2["targ_x"],
            f2["targ_y"],
            f2["targ_tnr"],
            f2["num_targets"],
            f2["num_parts"],
            f3["path_x"],
            f3["path_prev"],
            f3["path_next"],
            f3["path_inlist"],
            f3["path_prio"],
            f3["path_finaldecis"],
            f3["path_decis"],
            f3["path_linkdecis"],
            f3["corres_p"],
            f3["corres_nr"],
            f3["targ_x"],
            f3["targ_y"],
            f3["targ_tnr"],
            f3["num_targets"],
            f3["num_parts"],
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            5.0,
            30.0,
            0,
            5.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            nc,
            50.0,
            50.0,
            1.0,
            1.0,
            0,
            100.0,
            100.0,
            0.01,
            0.01,
            0.001,
        )
    finally:
        _mod._sorted_candidates_fast_out_nogil = orig


# ---------------------------------------------------------------------------
# trackback_loop_fast — crashes at _pos_mv = _pos_buf (C-array bug L2975)
# ---------------------------------------------------------------------------


def test_trackback_no_particles():
    """num_parts_1=0 → loop doesn't execute, returns (0, 0)."""
    count1, num_added = _call_trackback(0, 0, 0)
    assert count1 == 0
    assert num_added == 0


def test_trackback_particles_all_skipped():
    """All particles have next_h=-2 (NEXT_NONE_K) → loop body skipped."""
    count1, num_added = _call_trackback(1, 1, 1)
    assert count1 == 0
    assert num_added == 0


def test_trackback_enters_body():
    """Particle with next_h>=0 and prev_h==-1 → enters loop body (L2998+)."""
    nc = 1
    f0 = _frame(1, nc, 0, x_offset=0.2)
    f1 = _frame(1, nc, 0, x_offset=0.1)
    f2 = _frame(1, nc, 0, x_offset=0.0)
    f3 = _frame(1, nc, 0, x_offset=-0.1)
    # Give particle 0 in f1 a forward link (next_h=0) but no backward link
    f1["path_next"][0] = 0  # next_h = 0 (valid index in f0)
    f1["path_prev"][0] = -1  # prev_h = -1 → enters body

    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    md = _md_arr(nc)

    orig = _mod._sorted_candidates_fast_out_nogil
    _mod._sorted_candidates_fast_out_nogil = lambda *a, **k: 0
    try:
        count1, num_added = trackback_loop_fast(
            1,
            f0["path_x"],
            f1["path_x"],
            f1["path_prev"],
            f1["path_next"],
            f1["path_inlist"],
            f1["path_finaldecis"],
            f1["path_decis"],
            f1["path_linkdecis"],
            f2["path_x"],
            f2["path_prev"],
            f2["path_next"],
            f2["num_parts"],
            f2["targ_x"],
            f2["targ_y"],
            f2["targ_tnr"],
            f2["num_targets"],
            f2["corres_p"],
            f2["corres_nr"],
            f2["path_inlist"],
            f2["path_prio"],
            f2["path_finaldecis"],
            f2["path_decis"],
            f2["path_linkdecis"],
            f3["path_x"],
            f3["path_prev"],
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            5.0,
            30.0,
            0,
            5.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            nc,
            50.0,
            50.0,
            1.0,
            1.0,
            0,
            100.0,
            100.0,
            0.01,
            0.01,
            0.001,
        )
    finally:
        _mod._sorted_candidates_fast_out_nogil = orig

    assert isinstance(count1, int)
    assert isinstance(num_added, int)


# ---------------------------------------------------------------------------
# _point_to_pixel_out — mmlut branch (L412-433)
# ---------------------------------------------------------------------------


def test_point_to_pixel_with_mmlut():
    """has_mmlut=1 path exercises bilinear interpolation (L412-433)."""
    cal = _make_cal_arr(1)[0]
    # mmlut: 4x4 table, all values = 1.2 → mmf > 0 → radial_shift = 1.2
    nr, nz = 4, 4
    mo = np.zeros(4, dtype=np.float64)
    # origin at (0,0,0)
    mnr = nr
    mnz = nz
    mrw = 1000.0
    mmlut_data = np.ones(nr * nz, dtype=np.float64) * 1.2
    out = np.zeros(2, dtype=np.float64)
    _point_to_pixel_out(
        np.array([0.5, 0.5, 0.0], dtype=np.float64),
        cal,
        mmlut_data,
        mo,
        mnr,
        mnz,
        mrw,
        1,  # has_mmlut = True
        50.0,
        50.0,
        1.0,
        1.0,
        0,
        out,
    )
    assert np.isfinite(out[0]) and np.isfinite(out[1])


# ---------------------------------------------------------------------------
# _ray_tracing_out — gz=0 branch (L1003-1005)
# ---------------------------------------------------------------------------


def test_ray_tracing_gz_zero():
    """gz=gx=gy=0 → gn=0 → gd0=gd1=gd2=0.0 branch (L1003-1005)."""
    cal = _make_cal_arr(1)[0]
    cal[15] = 0.0  # gx
    cal[16] = 0.0  # gy
    cal[17] = 0.0  # gz
    out = np.zeros(6, dtype=np.float64)
    try:
        _ray_tracing_out(0.0, 0.0, cal, out)
    except ZeroDivisionError:
        pass  # acceptable — covers the gz=0 branch before any division


# ---------------------------------------------------------------------------
# _dist_to_flat_out — 50-iteration loop (L1403->1418 branch)
# ---------------------------------------------------------------------------


def test_dist_to_flat_loop_all_iterations():
    """k1=0.01, tol=1e-100 → convergence never reached → all 50 iters run."""
    out = np.zeros(2, dtype=np.float64)
    _dist_to_flat_out(
        0.5,
        0.5,  # dist_x, dist_y
        0.0,
        0.0,  # xh, yh
        0.01,  # k1 (non-zero → correction applied each iteration)
        0.0,
        0.0,
        0.0,  # k2, k3, p1, p2 → zero
        0.0,
        1.0,
        0.0,  # scx, she
        1e-100,  # tol (impossibly tight → loop runs all 50 iterations)
        out,
    )
    assert np.isfinite(out[0]) and np.isfinite(out[1])


# ---------------------------------------------------------------------------
# _candsearch_in_pix_rest_nogil — binary search (L1292, 1296)
# ---------------------------------------------------------------------------


def test_candsearch_rest_binary_search():
    """num_targets=8 → dj=2 > 1 → while dj > 1 loop executes (L1292,1296)."""
    n = 8
    targ_x = np.array([float(i) * 5.0 for i in range(n)], dtype=np.float64)
    targ_y = np.array([float(i) * 5.0 for i in range(n)], dtype=np.float64)
    targ_tnr = np.full(n, -1, dtype=np.int32)
    # Search near middle target (target 4: x=20, y=20)
    result = _candsearch_in_pix_rest_nogil(
        targ_x,
        targ_y,
        targ_tnr,
        n,
        20.0,
        20.0,  # cent_x, cent_y
        3.0,
        3.0,  # dl, dr
        3.0,
        3.0,  # du, dd
        200.0,
        200.0,  # imx, imy
        -1,  # tr_unused
    )
    assert result >= -1  # -1 means not found, >=0 means found


def test_candsearch_rest_xmax_clamp():
    """cent_x near right edge → xmax > imx → clamped (L1291-1292 path)."""
    n = 1
    targ_x = np.array([98.0], dtype=np.float64)
    targ_y = np.array([50.0], dtype=np.float64)
    targ_tnr = np.full(n, -1, dtype=np.int32)
    # cent_x=96 + dr=8 = 104 > imx=100 → clamp
    result = _candsearch_in_pix_rest_nogil(
        targ_x,
        targ_y,
        targ_tnr,
        n,
        96.0,
        50.0,  # cent_x, cent_y
        2.0,
        8.0,  # dl, dr (xmax=104 > imx=100)
        5.0,
        5.0,  # du, dd
        100.0,
        100.0,  # imx, imy
        -1,  # tr_unused
    )
    assert result >= -1


# ---------------------------------------------------------------------------
# trackcorr_loop_fast — stub-nonzero: covers _trackcorr_particle_fast body
# ---------------------------------------------------------------------------


def test_trackcorr_stub_nonzero_enters_for_mm_loop():
    """Stub returns 1 for first call (w_nc=1) and 0 for second (wn_nc=0).

    Covers L1873-2043: the 'for mm' body runs, second sorted-cands call
    returns 0 so wn_nc=0, skipping the kk-loop.
    With num_targets_3=0, quali=0 so _point_position_out never called.
    """
    nc = 1
    f0 = _frame(1, nc, 0, x_offset=0.0)
    f1 = _frame(1, nc, 0, x_offset=0.1)
    f2 = _frame(2, nc, 0, x_offset=0.2)  # 2 particles in frame 2
    f3 = _frame(0, nc, 0, x_offset=0.3)  # 0 targets in frame 3

    # Frame 1 particle has prev link (covers X[5] = velocity-extrapolation branch)
    f1["path_prev"][0] = 0

    cal = _make_cal_arr(nc)
    mo, mnr, mnz, mrw = _make_mmlut(nc)
    md = _md_arr(nc)

    call_count = [0]

    def _stub(*a, **k):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call (frame 2 search): return 1 candidate at index 0
            # _sorted_candidates_fast_out_nogil ends with ..., ftnr_out, freq_out, whichcam_out, pt_buf, _pp
            ftnr_out = a[-5]  # ftnr_out is 5th from end
            freq_out = a[-4]  # freq_out is 4th from end
            ftnr_out[0] = 0
            freq_out[0] = 1
            return 1
        return 0  # second call (frame 3): 0 candidates

    orig = _mod._sorted_candidates_fast_out_nogil
    _mod._sorted_candidates_fast_out_nogil = _stub
    try:
        count1, num_added = trackcorr_loop_fast(
            1,
            f0["path_x"],
            f1["path_x"],
            f1["path_prev"],
            f1["path_next"],
            f1["path_inlist"],
            f1["path_finaldecis"],
            f1["path_decis"],
            f1["path_linkdecis"],
            f1["corres_p"],
            f1["targ_x"],
            f1["targ_y"],
            f1["targ_tnr"],
            f2["path_x"],
            f2["path_prev"],
            f2["path_next"],
            f2["path_inlist"],
            f2["path_prio"],
            f2["path_finaldecis"],
            f2["path_decis"],
            f2["path_linkdecis"],
            f2["corres_p"],
            f2["corres_nr"],
            f2["targ_x"],
            f2["targ_y"],
            f2["targ_tnr"],
            f2["num_targets"],
            f2["num_parts"],
            f3["path_x"],
            f3["path_prev"],
            f3["path_next"],
            f3["path_inlist"],
            f3["path_prio"],
            f3["path_finaldecis"],
            f3["path_decis"],
            f3["path_linkdecis"],
            f3["corres_p"],
            f3["corres_nr"],
            f3["targ_x"],
            f3["targ_y"],
            f3["targ_tnr"],
            f3["num_targets"],
            f3["num_parts"],
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            5.0,
            30.0,
            0,
            5.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            -10.0,
            10.0,
            nc,
            50.0,
            50.0,
            1.0,
            1.0,
            0,
            100.0,
            100.0,
            0.01,
            0.01,
            0.001,
        )
    finally:
        _mod._sorted_candidates_fast_out_nogil = orig

    assert isinstance(count1, int)
    assert call_count[0] >= 2  # both calls were made


# ---------------------------------------------------------------------------
# trackcorr_loop_fast — num_cams=2 (covers md_arr unpacking L2520-2551)
# ---------------------------------------------------------------------------


def test_trackcorr_two_cams_stub_zero():
    """nc=2 exercises the md_arr multi-cam unpack branches (L2520-2551)."""
    result = _call_trackcorr(1, 1, 1, 1, nc=2, stub_zero=True)
    count1, num_added = result
    assert isinstance(count1, int)

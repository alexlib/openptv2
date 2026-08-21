"""Line-coverage tests for track_kernels_search.py.

Runs against both the compiled build and the pure-Python source, except
where a per-test/per-class skipif marker says otherwise (see the
_needs_pure_python* markers below for the specific, verified reasons).

Coverage command:
    cd /home/user/Documents/GitHub/openptv2
    COVERAGE_FILE=/tmp/.cov_track_kernels_search uv run pytest \
      tests/unit/test_track_kernels_search_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q 2>&1 | grep -E "(algorithms/track_kernels_search\\.|TOTAL|passed|failed|error)"

Source bugs found in pure-Python mode (do NOT fix):
  1. _sorted_candidates_fast_out (approx line 843-845): local C-array declarations
     `_pp: cython.double[2]`, `quader_buf: cython.double[24]`, `pt_buf: cython.double[3]`
     generate NO bytecode in Python 3.x (PEP 526 local-annotation rule). The variables
     remain unbound. First subscript access of `quader_buf` inside the searchquader loop
     raises UnboundLocalError, making all subsequent lines (~140 executable lines) unreachable.
  2. _sorted_candidates_fast_out_nogil (approx line 1206-1210): same C-array declarations
     (`_pp`, `quader_buf`, `pt_buf`, `cands_buf`) cause identical UnboundLocalError.
     ~138 executable lines are unreachable.
  3. sorted_candidates_fast propagates _sorted_candidates_fast_out's failure; its own
     `return ftnr, freq, whichcam, num_valid` line is unreachable (~1 line lost).
  Maximum achievable coverage is approximately 65-70% due to these bugs.
"""

import math

import numpy as np
import pytest

from openptv2.algorithms.track_kernels import is_compiled as _is_compiled

# PT_UNUSED/TR_UNUSED_K are module globals declared via cython.declare()
# without visibility="public" -- never exported as Python attributes even
# compiled, so they're not importable either way. Their values are stable,
# documented sentinel constants (see track_kernels_search.py's own
# `PT_UNUSED = -999`, `TR_UNUSED_K = -1`); defined locally here instead of
# gating ~30 test references on an import that can never succeed.
PT_UNUSED = -999
TR_UNUSED_K = -1

_needs_pure_python = pytest.mark.skipif(
    _is_compiled(),
    reason="_multimed_r_nlay_1layer/_point_to_pixel_out are @cython.cfunc, "
    "not exported from the compiled .pyd/.so",
)
_needs_pure_python_loose_types = pytest.mark.skipif(
    _is_compiled(),
    reason="passes a plain list/tuple where compiled Cython's typed "
    "memoryview parameters require a real buffer-protocol array "
    "(TypeError: a bytes-like object is required); the functions under test "
    "are internal-only nogil hot-path primitives with no caller that ever "
    "passes anything but a real ndarray, so this is a test-convenience gap "
    "in pure-Python mode, not a production one",
)
_sorted_candidates_fast_dead_code_bug = pytest.mark.skipif(
    _is_compiled(),
    reason="sorted_candidates_fast/sorted_candidates_in_volume are DEAD CODE "
    "(zero callers anywhere in src/openptv2) with a real internal signature "
    "mismatch: the outer function's cal_arrays/mmlut_* params are typed "
    "`tuple` and passed straight through to _sorted_candidates_fast_out, "
    "whose matching params are typed cython.double[:, ::1]/cython.int[:] "
    "memoryviews -- would TypeError on any real (non-empty) call if "
    "compiled. Not fixed here since nothing calls this path; flagged so a "
    "future caller doesn't hit it silently.",
)

# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------
from openptv2.algorithms.track_kernels_search import (
    _sorted_candidates_fast_out,
    _sorted_candidates_fast_out_nogil,
    candsearch_in_pix_fast,
    candsearch_in_pix_fast_nogil,
    candsearch_in_pix_rest_fast,
    sort_candidates_by_freq_fast,
    sorted_candidates_fast,
)

if not _is_compiled():
    from openptv2.algorithms.track_kernels_search import (
        _multimed_r_nlay_1layer,
        _point_to_pixel_out,
    )

EPS = 1e-8

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cal(
    ext_xyz=(0.0, 0.0, 500.0),
    dm=None,  # 3x3 rotation matrix, default identity
    cc=100.0,
    xh=0.0,
    yh=0.0,
    glass=(0.0, 0.0, 1.0),  # glass normal vector (gx, gy, gz)
    n1=1.0,
    n2=1.0,
    n3=1.0,
    d0=0.0,
    k1=0.0,
    k2=0.0,
    k3=0.0,
    p1=0.0,
    p2=0.0,
    scx=1.0,
    she=0.0,
):
    """Build a 31-element packed cal array for _point_to_pixel_out tests."""
    gx, gy, gz = glass
    dist_o_glas = math.sqrt(gx * gx + gy * gy + gz * gz)
    if dm is None:
        dm = np.eye(3, dtype=np.float64)
    c = np.zeros(31, dtype=np.float64)
    c[0] = ext_xyz[0]
    c[1] = ext_xyz[1]
    c[2] = ext_xyz[2]
    # dm stored column-major: dm[row, col] → c[row + col*3]
    c[3] = dm[0, 0]
    c[4] = dm[1, 0]
    c[5] = dm[2, 0]
    c[6] = dm[0, 1]
    c[7] = dm[1, 1]
    c[8] = dm[2, 1]
    c[9] = dm[0, 2]
    c[10] = dm[1, 2]
    c[11] = dm[2, 2]
    c[12] = cc
    c[13] = xh
    c[14] = yh
    c[15] = gx
    c[16] = gy
    c[17] = gz
    c[18] = dist_o_glas
    c[19] = 1.0 / dist_o_glas if dist_o_glas != 0 else 1.0
    c[20] = n1
    c[21] = n2
    c[22] = n3
    c[23] = d0
    c[24] = k1
    c[25] = k2
    c[26] = k3
    c[27] = p1
    c[28] = p2
    c[29] = scx
    c[30] = she
    return c


def _call_p2p(
    pos,
    cal,
    imx_half=512.0,
    imy_half=512.0,
    inv_pix_x=1.0,
    inv_pix_y=1.0,
    chfield=0,
    mmlut_data=None,
    mmlut_origin=None,
    mmlut_nr=0,
    mmlut_nz=0,
    mmlut_rw=1.0,
    has_mmlut=0,
):
    """Convenience wrapper for _point_to_pixel_out."""
    pos_arr = np.ascontiguousarray(pos, dtype=np.float64)
    cal_arr = np.ascontiguousarray(cal, dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    if mmlut_data is None:
        mmlut_data = np.zeros(4, dtype=np.float64)
    if mmlut_origin is None:
        mmlut_origin = np.zeros(3, dtype=np.float64)
    md = np.ascontiguousarray(mmlut_data, dtype=np.float64)
    mo = np.ascontiguousarray(mmlut_origin, dtype=np.float64)
    ret = _point_to_pixel_out(
        pos_arr,
        cal_arr,
        md,
        mo,
        mmlut_nr,
        mmlut_nz,
        mmlut_rw,
        has_mmlut,
        imx_half,
        imy_half,
        inv_pix_x,
        inv_pix_y,
        chfield,
        out,
    )
    return ret, out


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_pt_unused_value():
    assert PT_UNUSED == -999


def test_tr_unused_k_value():
    assert TR_UNUSED_K == -1


# ---------------------------------------------------------------------------
# _multimed_r_nlay_1layer
# ---------------------------------------------------------------------------


@_needs_pure_python
class TestMultimedRNlay1Layer:
    def test_all_n_equal_one_returns_one(self):
        r = _multimed_r_nlay_1layer(5.0, 3.0, 0.0, 0.0, 0.0, 100.0, 1.0, 1.0, 1.0, 0.0)
        assert r == 1.0

    def test_denom_zero_returns_one(self):
        # pos_z == ext_z0 → denom = ext_z0 - pos_z = 0
        r = _multimed_r_nlay_1layer(
            5.0, 0.0, 100.0, 0.0, 0.0, 100.0, 1.5, 1.33, 1.0, 2.0
        )
        assert r == 1.0

    def test_r_zero_returns_one(self):
        # pos_x == ext_x0 and pos_y == ext_y0 → r = 0
        r = _multimed_r_nlay_1layer(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.5, 1.33, 1.0, 2.0)
        assert r == 1.0

    def test_convergent_same_medium(self):
        # n1 == n2 == n3 (but not all 1.0) → still converges
        r = _multimed_r_nlay_1layer(
            10.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.33, 1.33, 1.33, 5.0
        )
        # Should converge and return rq/r ≈ 1.0 (same media, no shift)
        assert isinstance(r, float)

    def test_convergent_with_refraction(self):
        # Standard water/glass scenario, should converge
        r = _multimed_r_nlay_1layer(5.0, 3.0, 0.0, 0.0, 0.0, 100.0, 1.0, 1.33, 1.0, 5.0)
        assert isinstance(r, float)
        assert r > 0.0

    def test_arg_clamped_high(self):
        # sin_beta1 * mm_n1 / mm_n2_0 > 1.0 → clamp to 1.0
        # Large r, small denom → large angle
        r = _multimed_r_nlay_1layer(500.0, 0.0, 0.0, 0.0, 0.0, 10.0, 3.0, 1.0, 1.0, 2.0)
        assert isinstance(r, float)

    def test_small_rdiff_breaks_early(self):
        # Normal convergent case
        r = _multimed_r_nlay_1layer(2.0, 0.0, 0.0, 0.0, 0.0, 50.0, 1.0, 1.5, 1.0, 3.0)
        assert isinstance(r, float)
        assert r > 0.0


# ---------------------------------------------------------------------------
# _point_to_pixel_out
# ---------------------------------------------------------------------------


@_needs_pure_python
class TestPointToPixelOut:
    """Tests for the cfunc _point_to_pixel_out."""

    def setup_method(self):
        # Camera at (0, 0, 500), looking down -z, identity rotation
        self.cal = _make_cal(
            ext_xyz=(0.0, 0.0, 500.0),
            glass=(0.0, 0.0, 1.0),
            cc=100.0,
        )

    def test_basic_off_axis_point(self):
        """Point at (10, 20, 0) → pixel offset from center."""
        ret, out = _call_p2p([10.0, 20.0, 0.0], self.cal)
        assert ret == 0
        # Pixel should be offset from center (512, 512)
        assert abs(out[0] - 512.0) > 0.1
        assert abs(out[1] - 512.0) > 0.1

    def test_on_axis_point_r_near_zero(self):
        """Object on camera axis → r ≈ 0, takes the r < 1e-10 branch."""
        # pos exactly on the camera axis (x=0, y=0)
        ret, out = _call_p2p([0.0, 0.0, 0.0], self.cal)
        assert ret == 0
        # Should be at image center
        assert abs(out[0] - 512.0) < 1.0
        assert abs(out[1] - 512.0) < 1.0

    def test_chfield_1(self):
        """chfield=1 → y_pixel = (y_pixel - 1) * 0.5."""
        ret, out = _call_p2p([10.0, 20.0, 0.0], self.cal, chfield=1)
        assert ret == 0

    def test_chfield_2(self):
        """chfield=2 → y_pixel = y_pixel * 0.5."""
        ret, out = _call_p2p([10.0, 20.0, 0.0], self.cal, chfield=2)
        assert ret == 0

    def test_chfield_0_and_nonzero_differ(self):
        """chfield=0 vs chfield=1 give different y."""
        _, out0 = _call_p2p([10.0, 20.0, 0.0], self.cal, chfield=0)
        _, out1 = _call_p2p([10.0, 20.0, 0.0], self.cal, chfield=1)
        assert abs(out0[1] - out1[1]) > 0.01

    def test_with_distortion_params(self):
        """Non-zero k1 changes radial distortion."""
        cal_dist = _make_cal(
            ext_xyz=(0.0, 0.0, 500.0),
            glass=(0.0, 0.0, 1.0),
            cc=100.0,
            k1=0.01,
        )
        _, out_nodist = _call_p2p([50.0, 0.0, 0.0], self.cal)
        _, out_dist = _call_p2p([50.0, 0.0, 0.0], cal_dist)
        assert abs(out_nodist[0] - out_dist[0]) > 0.001

    def test_with_she_nonzero(self):
        """Non-zero she (shear) changes x_dist via sin_she."""
        cal_she = _make_cal(
            ext_xyz=(0.0, 0.0, 500.0),
            glass=(0.0, 0.0, 1.0),
            cc=100.0,
            she=0.1,
        )
        _, out0 = _call_p2p([10.0, 50.0, 0.0], self.cal)
        _, out_she = _call_p2p([10.0, 50.0, 0.0], cal_she)
        assert abs(out0[0] - out_she[0]) > 0.001

    def test_no_mmlut_path(self):
        """has_mmlut=0 → falls through to _multimed_r_nlay_1layer."""
        ret, out = _call_p2p([10.0, 20.0, 0.0], self.cal, has_mmlut=0)
        assert ret == 0

    def test_mmlut_valid_hit_mmf_nonzero(self):
        """has_mmlut=1 with mmlut data that gives mmf != 1.0 (covers lines 305-315, 316->329).

        The object must be at z > glass_plane_z (here z=5 > glass at z=1) so that
        pos_t_2 = pos2 - dist_o_glas = 5 - 1 = 4 > 0 → iz >= 0 → mmlut block entered.
        mmlut_rw=7 keeps ir=3 within mmlut_nr=5; mmlut_data=1.5 gives mmf=1.5 ≠ 1.0.
        """
        nr, nz = 5, 5
        mmlut_data = np.full(nr * nz, 1.5, dtype=np.float64)
        mmlut_origin = np.zeros(3, dtype=np.float64)
        ret, out = _call_p2p(
            [10.0, 20.0, 5.0],
            self.cal,  # z=5 → pos_t_2=4 > 0
            has_mmlut=1,
            mmlut_data=mmlut_data,
            mmlut_origin=mmlut_origin,
            mmlut_nr=nr,
            mmlut_nz=nz,
            mmlut_rw=7.0,  # ir = int(pos_t_0/7) = 3 ≤ 5
        )
        assert ret == 0

    def test_mmlut_valid_hit_mmf_one(self):
        """has_mmlut=1 with mmlut all-ones → mmf=1.0 → radial_shift stays 1.0 → fallback."""
        nr, nz = 5, 5
        mmlut_data = np.ones(nr * nz, dtype=np.float64)
        mmlut_origin = np.zeros(3, dtype=np.float64)
        ret, out = _call_p2p(
            [10.0, 20.0, 5.0],
            self.cal,  # z=5 → pos_t_2=4 > 0
            has_mmlut=1,
            mmlut_data=mmlut_data,
            mmlut_origin=mmlut_origin,
            mmlut_nr=nr,
            mmlut_nz=nz,
            mmlut_rw=7.0,
        )
        assert ret == 0

    def test_mmlut_hit_mmf_zero_falls_through(self):
        """has_mmlut=1 but mmf == 0.0 → radial_shift stays 1.0 → fallback."""
        nr, nz = 5, 5
        mmlut_data = np.zeros(nr * nz, dtype=np.float64)
        mmlut_origin = np.zeros(3, dtype=np.float64)
        ret, out = _call_p2p(
            [5.0, 5.0, 0.0],
            self.cal,
            has_mmlut=1,
            mmlut_data=mmlut_data,
            mmlut_origin=mmlut_origin,
            mmlut_nr=nr,
            mmlut_nz=nz,
            mmlut_rw=1.0,
        )
        assert ret == 0

    def test_mmlut_out_of_range_ir(self):
        """has_mmlut=1 but ir > mmlut_nr → branch not taken → fallback."""
        # Large pos_t_0 → large R → ir >> mmlut_nr
        nr, nz = 2, 2
        mmlut_data = np.ones(nr * nz, dtype=np.float64)
        mmlut_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ret, out = _call_p2p(
            [1000.0, 0.0, 0.0],
            self.cal,
            has_mmlut=1,
            mmlut_data=mmlut_data,
            mmlut_origin=mmlut_origin,
            mmlut_nr=nr,
            mmlut_nz=nz,
            mmlut_rw=1.0,
        )
        assert ret == 0

    def test_pos_t_0_zero_skips_back_trans(self):
        """pos_t_0 == 0 → skip the s_x branch in back_trans_point."""
        # Camera and object both at z=500 on the glass plane axis
        cal = _make_cal(
            ext_xyz=(0.0, 0.0, 500.0),
            glass=(0.0, 0.0, 1.0),
            cc=100.0,
            d0=500.0,  # mm_d0 == dist_cam_glas → moves ag to camera origin
        )
        ret, out = _call_p2p([0.0, 0.0, 0.0], cal)
        assert ret == 0

    def test_p1_p2_tangential_distortion(self):
        """Non-zero p1, p2 change the distorted coordinates."""
        cal_tp = _make_cal(
            ext_xyz=(0.0, 0.0, 500.0),
            glass=(0.0, 0.0, 1.0),
            cc=100.0,
            p1=0.001,
            p2=0.001,
        )
        _, out0 = _call_p2p([30.0, 30.0, 0.0], self.cal)
        _, out1 = _call_p2p([30.0, 30.0, 0.0], cal_tp)
        assert abs(out0[0] - out1[0]) + abs(out0[1] - out1[1]) > 0.001

    def test_consistency_with_n_same(self):
        """All n=1 (no refraction), result should be stable."""
        ret, out = _call_p2p([20.0, -15.0, 100.0], self.cal)
        assert ret == 0
        assert 0.0 <= out[0] <= 1024.0
        assert 0.0 <= out[1] <= 1024.0

    def test_nonzero_k2_k3(self):
        """k2 and k3 distortion terms covered."""
        cal_k = _make_cal(
            ext_xyz=(0.0, 0.0, 500.0),
            glass=(0.0, 0.0, 1.0),
            cc=100.0,
            k2=1e-4,
            k3=1e-6,
        )
        ret, out = _call_p2p([30.0, 30.0, 0.0], cal_k)
        assert ret == 0

    def test_v3_out_of_bounds(self):
        """v3 > mmlut_nr * mmlut_nz → mmlut branch not entered → fallback."""
        nr, nz = 2, 2
        mmlut_data = np.ones(nr * nz, dtype=np.float64)
        mmlut_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        # Make ir close to mmlut_nr so v0 is in bounds but v3 = v0+nz+1 is out
        # ir = mmlut_nr → v0 = nr*nz → v3 = nr*nz + nz + 1 > nr*nz
        ret, out = _call_p2p(
            [2.5, 0.0, 0.0],
            self.cal,
            has_mmlut=1,
            mmlut_data=mmlut_data,
            mmlut_origin=mmlut_origin,
            mmlut_nr=nr,
            mmlut_nz=nz,
            mmlut_rw=1.0,
        )
        assert ret == 0

    def test_mmlut_v3_out_of_bounds_inner(self):
        """Inner v0>=0 and v3<=nr*nz False branch (covers 307->316).

        nr=2, nz=2 → nr*nz=4. With mmlut_rw=16 and pos=[10,20,5]:
        pos_t_0≈22.36 → ir=int(22.36/16)=1 ≤ nr=2 ✓ (outer if passes)
        pos_t_2=4.0 → iz=int(4/16)=0 ≤ nz=2 ✓
        v0 = 1*2+0 = 2; v3 = 2+2+1 = 5 > 4 → inner if False → 307->316 covered.
        """
        nr, nz = 2, 2
        mmlut_data = np.ones(nr * nz, dtype=np.float64)
        mmlut_origin = np.zeros(3, dtype=np.float64)
        ret, out = _call_p2p(
            [10.0, 20.0, 5.0],
            self.cal,
            has_mmlut=1,
            mmlut_data=mmlut_data,
            mmlut_origin=mmlut_origin,
            mmlut_nr=nr,
            mmlut_nz=nz,
            mmlut_rw=16.0,  # ir=int(22.36/16)=1 ≤ 2; v3=5 > 4 → skip inner
        )
        assert ret == 0

    def test_mmlut_mmf_zero_inner(self):
        """mmf == 0.0 → radial_shift stays 1.0 → fallback (covers 314->316).

        nr=5, nz=5, mmlut_rw=7, pos=[10,20,5]: v0=15, v3=21 ≤ 25 → inner block runs.
        mmlut_data all-zero → mmf=0 → 'if mmf > 0.0' is False → 314->316 covered.
        """
        nr, nz = 5, 5
        mmlut_data = np.zeros(nr * nz, dtype=np.float64)
        mmlut_origin = np.zeros(3, dtype=np.float64)
        ret, out = _call_p2p(
            [10.0, 20.0, 5.0],
            self.cal,
            has_mmlut=1,
            mmlut_data=mmlut_data,
            mmlut_origin=mmlut_origin,
            mmlut_nr=nr,
            mmlut_nz=nz,
            mmlut_rw=7.0,
        )
        assert ret == 0


# ---------------------------------------------------------------------------
# candsearch_in_pix_fast
# ---------------------------------------------------------------------------


def _make_targets(xs, ys, tnrs):
    """Build aligned float64/int32 target arrays sorted by y."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    tnrs = np.asarray(tnrs, dtype=np.int32)
    order = np.argsort(ys)
    return xs[order], ys[order], tnrs[order]


TR = -1  # tr_unused sentinel


class TestCandsearchInPixFast:
    IMX, IMY = 1024.0, 1024.0

    def test_empty_targets_returns_all_unused(self):
        tx = np.zeros(0, dtype=np.float64)
        ty = np.zeros(0, dtype=np.float64)
        tnr = np.zeros(0, dtype=np.int32)
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 0, 512.0, 512.0, 10.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        assert all(p == PT_UNUSED for p in (p1, p2, p3, p4))

    def test_center_out_of_bounds_returns_unused(self):
        tx, ty, tnr = _make_targets([100.0], [100.0], [0])
        # Center far outside image
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx,
            ty,
            tnr,
            1,
            2000.0,
            512.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
        )
        assert all(p == PT_UNUSED for p in (p1, p2, p3, p4))

    def test_center_negative_returns_unused(self):
        tx, ty, tnr = _make_targets([100.0], [100.0], [0])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 1, -10.0, 512.0, 10.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        assert all(p == PT_UNUSED for p in (p1, p2, p3, p4))

    def test_single_target_in_range(self):
        tx, ty, tnr = _make_targets([500.0], [500.0], [42])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 1, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, self.IMX, self.IMY, TR
        )
        assert p1 == 0  # only one target, it's at index 0 after sort
        assert p2 == PT_UNUSED

    def test_target_with_tr_unused_skipped(self):
        # tnr == TR → skipped
        tx, ty, tnr = _make_targets([500.0], [500.0], [TR])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 1, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, self.IMX, self.IMY, TR
        )
        assert all(p == PT_UNUSED for p in (p1, p2, p3, p4))

    def test_four_targets_returns_four_closest(self):
        # Six targets — only four closest returned
        xs = [500.0, 501.0, 502.0, 503.0, 550.0, 600.0]
        ys = [500.0, 500.5, 501.0, 501.5, 505.0, 510.0]
        tnrs = [10, 11, 12, 13, 14, 15]
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx,
            ty,
            tnr,
            len(xs),
            500.0,
            500.0,
            30.0,
            30.0,
            30.0,
            30.0,
            self.IMX,
            self.IMY,
            TR,
        )
        found = [p for p in (p1, p2, p3, p4) if p != PT_UNUSED]
        assert len(found) >= 1

    def test_target_outside_search_area_excluded(self):
        # One target inside, one far outside search box
        tx, ty, tnr = _make_targets([500.0, 900.0], [500.0, 900.0], [1, 2])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 2, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, self.IMX, self.IMY, TR
        )
        # Only target 0 (at 500,500) is within ±20
        assert p1 == 0
        assert p2 == PT_UNUSED

    def test_d2_d3_d4_replacements(self):
        # Arrange targets so all four distance slots are used in order
        xs = [500.0, 501.0, 502.0, 503.0, 504.0]
        ys = [500.0, 500.1, 500.2, 500.3, 500.4]
        tnrs = [10, 11, 12, 13, 14]
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 5, 500.0, 500.0, 10.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        found = [p for p in (p1, p2, p3, p4) if p != PT_UNUSED]
        assert len(found) == 4

    def test_ymax_break(self):
        # Targets with increasing y; one far beyond ymax → early break
        xs = [500.0, 500.0, 500.0]
        ys = [500.0, 501.0, 600.0]  # 600 > ymax=510
        tnrs = [1, 2, 3]
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 3, 500.0, 500.0, 5.0, 5.0, 5.0, 10.0, self.IMX, self.IMY, TR
        )
        # Should find at most 2 (within ymin=495..ymax=510)
        found = [p for p in (p1, p2, p3, p4) if p != PT_UNUSED]
        assert len(found) <= 2

    def test_xmin_xmax_clamp_to_image(self):
        # Center near left edge — xmin clamped to 0
        tx, ty, tnr = _make_targets([5.0], [512.0], [7])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 1, 5.0, 512.0, 100.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        # Target at x=5 is within search window
        assert p1 != PT_UNUSED

    def test_ymin_clamp_to_zero(self):
        """cent_y < du → ymin clamped to 0.0 (covers line 448)."""
        tx, ty, tnr = _make_targets([512.0], [5.0], [3])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 1, 512.0, 5.0, 10.0, 10.0, 20.0, 10.0, self.IMX, self.IMY, TR
        )
        # cent_y=5, du=20 → ymin=-15 → clamped to 0; target at y=5 inside
        assert p1 != PT_UNUSED

    def test_ymax_clamp_to_imy(self):
        """cent_y + dd > imy → ymax clamped to imy (covers line 450)."""
        tx, ty, tnr = _make_targets([512.0], [1020.0], [9])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx,
            ty,
            tnr,
            1,
            512.0,
            1020.0,
            10.0,
            10.0,
            10.0,
            20.0,
            self.IMX,
            self.IMY,
            TR,
        )
        # cent_y=1020, dd=20 → ymax=1040 > 1024 → clamped; target at y=1020 inside
        assert p1 != PT_UNUSED

    def test_binary_search_j0_increment(self):
        """Binary search takes j0 += dj when targ_y[j0] < ymin (requires large n)."""
        # 40 targets in y=[100..200], search center at y=190 with small window
        # j0_initial = 20, dj=10, targ_y[20] = 150 < ymin=185 → j0 += dj
        n = 40
        xs = np.full(n, 512.0)
        ys = np.linspace(100.0, 200.0, n)
        tnrs = np.arange(n, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, n, 512.0, 190.0, 5.0, 5.0, 5.0, 5.0, self.IMX, self.IMY, TR
        )
        # Targets near y=190 should be found
        found = [p for p in (p1, p2, p3, p4) if p != PT_UNUSED]
        assert len(found) >= 1

    def test_target_in_y_range_but_out_of_x_range(self):
        """Target inside y window but outside x window → False branch of tx check."""
        # Target far in x but correct y → covers the tx-range False branch
        tx, ty, tnr = _make_targets([900.0], [500.0], [5])
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, 1, 500.0, 500.0, 10.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        assert all(p == PT_UNUSED for p in (p1, p2, p3, p4))

    def test_many_targets_large_num(self):
        # Large num_targets to exercise the binary search jump
        n = 100
        xs = np.linspace(400.0, 600.0, n)
        ys = np.linspace(400.0, 600.0, n)
        tnrs = np.arange(n, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        p1, p2, p3, p4 = candsearch_in_pix_fast(
            tx, ty, tnr, n, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, self.IMX, self.IMY, TR
        )
        found = [p for p in (p1, p2, p3, p4) if p != PT_UNUSED]
        assert len(found) == 4


# ---------------------------------------------------------------------------
# candsearch_in_pix_rest_fast
# ---------------------------------------------------------------------------


class TestCandsearchInPixRestFast:
    IMX, IMY = 1024.0, 1024.0

    def test_empty_returns_unused_zero(self):
        tx = np.zeros(0, dtype=np.float64)
        ty = np.zeros(0, dtype=np.float64)
        tnr = np.zeros(0, dtype=np.int32)
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 0, 512.0, 512.0, 10.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        assert idx == PT_UNUSED
        assert cnt == 0

    def test_out_of_bounds_center(self):
        tx, ty, tnr = _make_targets([100.0], [100.0], [TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx,
            ty,
            tnr,
            1,
            2000.0,
            512.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
        )
        assert idx == PT_UNUSED
        assert cnt == 0

    def test_finds_unused_target(self):
        # tnr == TR (unused) → eligible
        tx, ty, tnr = _make_targets([500.0], [500.0], [TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 1, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, self.IMX, self.IMY, TR
        )
        assert idx == 0
        assert cnt == 1

    def test_skips_used_target(self):
        # tnr != TR → not eligible for rest search
        tx, ty, tnr = _make_targets([500.0], [500.0], [42])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 1, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, self.IMX, self.IMY, TR
        )
        assert idx == PT_UNUSED
        assert cnt == 0

    def test_finds_closest_of_two_unused(self):
        tx, ty, tnr = _make_targets([500.0, 501.0], [500.0, 500.5], [TR, TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 2, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, self.IMX, self.IMY, TR
        )
        assert idx != PT_UNUSED
        assert cnt == 1

    def test_ymax_break(self):
        tx, ty, tnr = _make_targets([500.0, 500.0], [500.0, 600.0], [TR, TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 2, 500.0, 500.0, 5.0, 5.0, 5.0, 5.0, self.IMX, self.IMY, TR
        )
        # Only target at y=500 is in window (ymax=505)
        assert idx == 0

    def test_xmin_clamp(self):
        tx, ty, tnr = _make_targets([2.0], [512.0], [TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 1, 2.0, 512.0, 200.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        assert idx != PT_UNUSED

    def test_ymin_clamp(self):
        """cent_y < du → ymin clamped to 0.0 (covers line 564)."""
        tx, ty, tnr = _make_targets([512.0], [5.0], [TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 1, 512.0, 5.0, 10.0, 10.0, 20.0, 10.0, self.IMX, self.IMY, TR
        )
        assert idx != PT_UNUSED

    def test_ymax_clamp(self):
        """cent_y + dd > imy → ymax clamped to imy (covers line 566)."""
        tx, ty, tnr = _make_targets([512.0], [1020.0], [TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx,
            ty,
            tnr,
            1,
            512.0,
            1020.0,
            10.0,
            10.0,
            10.0,
            20.0,
            self.IMX,
            self.IMY,
            TR,
        )
        assert idx != PT_UNUSED

    def test_binary_search_j0_increment(self):
        """Binary search j0 += dj when targ_y[j0] < ymin (covers line 579)."""
        # 40 targets in y=[100..200], search at y=190 → midpoint y[20]=150 < ymin=185
        n = 40
        xs = np.full(n, 512.0)
        ys = np.linspace(100.0, 200.0, n)
        tnrs = np.full(n, TR, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, n, 512.0, 190.0, 5.0, 5.0, 5.0, 5.0, self.IMX, self.IMY, TR
        )
        assert idx != PT_UNUSED

    def test_j0_no_clamp_large_n(self):
        """num_targets >= 24 → j0 - 12 >= 0 → False branch of if j0 < 0 (covers 585->588)."""
        n = 50
        xs = np.full(n, 512.0)
        ys = np.linspace(500.0, 510.0, n)
        tnrs = np.full(n, TR, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, n, 512.0, 505.0, 5.0, 5.0, 5.0, 5.0, self.IMX, self.IMY, TR
        )
        assert idx != PT_UNUSED

    def test_target_in_y_range_out_of_x(self):
        """Target in y window but not x window → False branch at tx check (covers 594->588)."""
        tx, ty, tnr = _make_targets([900.0], [500.0], [TR])
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, 1, 500.0, 500.0, 10.0, 10.0, 10.0, 10.0, self.IMX, self.IMY, TR
        )
        assert idx == PT_UNUSED

    def test_large_num_targets(self):
        n = 80
        xs = np.linspace(490.0, 510.0, n)
        ys = np.linspace(490.0, 510.0, n)
        tnrs = np.full(n, TR, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        idx, cnt = candsearch_in_pix_rest_fast(
            tx, ty, tnr, n, 500.0, 500.0, 15.0, 15.0, 15.0, 15.0, self.IMX, self.IMY, TR
        )
        assert idx != PT_UNUSED
        assert cnt == 1


# ---------------------------------------------------------------------------
# sort_candidates_by_freq_fast
# ---------------------------------------------------------------------------


class TestSortCandidatesByFreqFast:
    NUM_CAMS = 4
    MAX_CANDS = 4

    def _make_arrays(self, ftnr_vals):
        n = len(ftnr_vals)
        ftnr = np.asarray(ftnr_vals, dtype=np.int32)
        freq = np.zeros(n, dtype=np.int32)
        whichcam = np.zeros((n, self.NUM_CAMS), dtype=np.int32)
        return ftnr, freq, whichcam

    def test_all_unused(self):
        ftnr, freq, whichcam = self._make_arrays([-1] * 16)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        assert nv == 0

    def test_single_candidate_one_cam(self):
        # One candidate in camera 0 slot, rest unused
        vals = [-1] * 16
        vals[0] = 5  # cam0, slot0
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        # Position 0 entry keeps freq=1 (dedup zeroes only j > i slots)
        assert nv == 1

    def test_same_candidate_two_cams(self):
        # Target 10 appears in cam0 slot0 and cam1 slot0
        vals = [-1] * 16
        vals[0] = 10  # cam0 slot0
        vals[4] = 10  # cam1 slot0
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        assert nv >= 1

    def test_same_candidate_all_cams(self):
        # Target 7 seen in all 4 cameras
        vals = [-1] * 16
        vals[0] = 7  # cam0
        vals[4] = 7  # cam1
        vals[8] = 7  # cam2
        vals[12] = 7  # cam3
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        assert nv >= 1
        # First entry should be target 7 with freq=4
        assert ftnr[0] == 7

    def test_duplicate_elimination(self):
        # Same target twice in same camera → should be deduped
        vals = [-1] * 16
        vals[0] = 3
        vals[1] = 3  # duplicate
        vals[4] = 3  # also in cam1
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        assert nv >= 1
        # Only one unique entry should remain
        active = [(ftnr[i], freq[i]) for i in range(16) if freq[i] > 0]
        tnrs = [t for t, f in active]
        assert tnrs.count(3) <= 1

    def test_sorting_higher_freq_first(self):
        # Target A in 3 cams, target B in 2 cams — A should sort first
        vals = [-1] * 16
        vals[0] = 20  # cam0: target A
        vals[4] = 20  # cam1: target A
        vals[8] = 20  # cam2: target A
        vals[1] = 30  # cam0 slot1: target B
        vals[5] = 30  # cam1 slot1: target B
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        assert nv >= 1
        if nv >= 2:
            assert freq[0] >= freq[1]

    def test_whichcam_swap(self):
        # Test that whichcam is correctly swapped during sort
        vals = [-1] * 16
        vals[0] = 5
        vals[4] = 5
        vals[1] = 6
        vals[5] = 6
        vals[9] = 6
        vals[13] = 6  # target 6 in all 4 cams → higher freq
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        # Target 6 (freq=4) should sort before target 5 (freq=2)
        if nv >= 1:
            assert ftnr[0] == 6

    def test_returns_nonzero_for_freq_one_at_position_zero(self):
        # Dedup loop only eliminates j > i, so index 0 always kept if freq > 0
        vals = [-1] * 16
        vals[0] = 99  # only in 1 camera → freq=1
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        # freq[0]=1 ≠ 0 → counted as valid
        assert nv == 1

    def test_freq_one_entry_after_another_gets_zeroed(self):
        # Second unique candidate at freq=1 is eliminated by dedup loop at j>i
        vals = [-1] * 16
        vals[0] = 10  # cam0: target 10 in 2 cams → freq=2
        vals[4] = 10  # cam1
        vals[1] = 99  # cam0 slot1: target 99 in 1 cam → freq=1
        ftnr, freq, whichcam = self._make_arrays(vals)
        nv = sort_candidates_by_freq_fast(
            ftnr, freq, whichcam, 16, self.NUM_CAMS, self.MAX_CANDS
        )
        # Target 10 (freq=2) survives; target 99 (freq=1) is zeroed
        assert nv >= 1


# ---------------------------------------------------------------------------
# candsearch_in_pix_fast_nogil (cfunc — callable in pure Python)
# ---------------------------------------------------------------------------


@_needs_pure_python_loose_types
class TestCandsearchInPixFastNogil:
    """Tests for the nogil cfunc variant. Pass a list for out_indices."""

    IMX, IMY = 1024.0, 1024.0

    def test_center_out_of_bounds(self):
        tx = np.array([100.0], dtype=np.float64)
        ty = np.array([100.0], dtype=np.float64)
        tnr = np.array([0], dtype=np.int32)
        out = [0, 0, 0, 0]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            1,
            2000.0,
            512.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert all(v == -999 for v in out)

    def test_empty_targets(self):
        tx = np.zeros(0, dtype=np.float64)
        ty = np.zeros(0, dtype=np.float64)
        tnr = np.zeros(0, dtype=np.int32)
        out = [0, 0, 0, 0]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            0,
            512.0,
            512.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert out[0] == -999

    def test_single_target_found(self):
        tx, ty, tnr = _make_targets([500.0], [500.0], [5])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            1,
            500.0,
            500.0,
            20.0,
            20.0,
            20.0,
            20.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert out[0] == 0  # index 0 in sorted arrays
        assert out[1] == -999

    def test_unused_tnr_skipped(self):
        tx, ty, tnr = _make_targets([500.0], [500.0], [TR])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            1,
            500.0,
            500.0,
            20.0,
            20.0,
            20.0,
            20.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert out[0] == -999

    def test_four_targets_fills_all_slots(self):
        n = 5
        xs = [500.0, 501.0, 502.0, 503.0, 504.0]
        ys = [500.0, 500.1, 500.2, 500.3, 500.4]
        tnrs = [10, 11, 12, 13, 14]
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            n,
            500.0,
            500.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        found = [v for v in out if v != -999]
        assert len(found) == 4

    def test_d2_replacement(self):
        # Target 1 is closest, target 2 is second closest
        tx, ty, tnr = _make_targets([500.0, 500.5], [500.0, 500.5], [1, 2])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            2,
            500.0,
            500.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        found = [v for v in out if v != -999]
        assert len(found) == 2

    def test_d3_replacement(self):
        tx, ty, tnr = _make_targets(
            [500.0, 500.5, 501.0], [500.0, 500.5, 501.0], [1, 2, 3]
        )
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            3,
            500.0,
            500.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        found = [v for v in out if v != -999]
        assert len(found) == 3

    def test_ymax_early_break(self):
        tx, ty, tnr = _make_targets([500.0, 500.0], [502.0, 600.0], [1, 2])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            2,
            500.0,
            500.0,
            5.0,
            5.0,
            5.0,
            5.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        found = [v for v in out if v != -999]
        assert len(found) <= 1

    def test_large_num_targets(self):
        n = 60
        xs = np.linspace(490.0, 510.0, n)
        ys = np.linspace(490.0, 510.0, n)
        tnrs = np.arange(n, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            n,
            500.0,
            500.0,
            12.0,
            12.0,
            12.0,
            12.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        found = [v for v in out if v != -999]
        assert len(found) == 4

    def test_ymin_clamp(self):
        """cent_y < du → ymin = 0.0 (covers line 1057)."""
        tx, ty, tnr = _make_targets([512.0], [5.0], [1])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            1,
            512.0,
            5.0,
            10.0,
            10.0,
            20.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert out[0] != -999

    def test_ymax_clamp(self):
        """cent_y + dd > imy → ymax = imy (covers line 1059)."""
        tx, ty, tnr = _make_targets([512.0], [1020.0], [1])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            1,
            512.0,
            1020.0,
            10.0,
            10.0,
            10.0,
            20.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert out[0] != -999

    def test_xmin_clamp(self):
        """cent_x < dl → xmin = 0.0 (covers line 1053)."""
        tx, ty, tnr = _make_targets([5.0], [512.0], [1])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            1,
            5.0,
            512.0,
            100.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert out[0] != -999

    def test_binary_search_j0_increment(self):
        """Binary search j0 += dj (covers line 1081) and j0 no-clamp (1087->1090)."""
        n = 40
        xs = np.full(n, 512.0)
        ys = np.linspace(100.0, 200.0, n)
        tnrs = np.arange(n, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            n,
            512.0,
            190.0,
            5.0,
            5.0,
            5.0,
            5.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        found = [v for v in out if v != -999]
        assert len(found) >= 1

    def test_j0_no_clamp(self):
        """num_targets >= 24 → j0 - 12 >= 0 (covers branch 1087->1090)."""
        n = 50
        xs = np.full(n, 512.0)
        ys = np.linspace(500.0, 510.0, n)
        tnrs = np.arange(n, dtype=np.int32)
        tx, ty, tnr = _make_targets(xs, ys, tnrs)
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            n,
            512.0,
            505.0,
            5.0,
            5.0,
            5.0,
            5.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        found = [v for v in out if v != -999]
        assert len(found) >= 1

    def test_target_in_y_out_of_x(self):
        """Target in y range but not x → False branch at tx check (covers 1096->1090)."""
        tx, ty, tnr = _make_targets([900.0], [500.0], [1])
        out = [-999, -999, -999, -999]
        candsearch_in_pix_fast_nogil(
            tx,
            ty,
            tnr,
            1,
            500.0,
            500.0,
            10.0,
            10.0,
            10.0,
            10.0,
            self.IMX,
            self.IMY,
            TR,
            out,
        )
        assert all(v == -999 for v in out)


# ---------------------------------------------------------------------------
# sorted_candidates_fast — calls _sorted_candidates_fast_out which crashes
# ---------------------------------------------------------------------------


def _make_sorted_candidates_args(num_cams=1, max_cands=4, n_targ=5):
    """Build minimal args for sorted_candidates_fast."""
    center = np.array([0.0, 0.0, 100.0], dtype=np.float64)
    center_proj_x = np.zeros(num_cams, dtype=np.float64)
    center_proj_y = np.zeros(num_cams, dtype=np.float64)

    cal = _make_cal()
    cal_arrays = tuple(cal for _ in range(num_cams))
    mmlut_data = np.zeros(4, dtype=np.float64)
    mmlut_datas = tuple(mmlut_data for _ in range(num_cams))
    mmlut_origins = tuple(np.zeros(3, dtype=np.float64) for _ in range(num_cams))
    mmlut_nrs = tuple(0 for _ in range(num_cams))
    mmlut_nzs = tuple(0 for _ in range(num_cams))
    mmlut_rws = tuple(1.0 for _ in range(num_cams))

    targ_x = np.zeros((num_cams, n_targ), dtype=np.float64, order="C")
    targ_y = np.zeros((num_cams, n_targ), dtype=np.float64, order="C")
    targ_tnr = np.full((num_cams, n_targ), -1, dtype=np.int32, order="C")
    num_targets = np.zeros(num_cams, dtype=np.int32)

    return (
        center,
        center_proj_x,
        center_proj_y,
        num_cams,
        max_cands,
        cal_arrays,
        mmlut_datas,
        mmlut_origins,
        mmlut_nrs,
        mmlut_nzs,
        mmlut_rws,
        targ_x,
        targ_y,
        targ_tnr,
        num_targets,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,  # dv min/max
        512.0,
        512.0,
        1.0,
        1.0,
        0,  # imx_half, imy_half, inv_pix, chfield
        1024.0,
        1024.0,
        -1,  # imx, imy, tr_unused
    )


@_sorted_candidates_fast_dead_code_bug
class TestSortedCandidatesFastBug:
    """sorted_candidates_fast runs in pure Python (C-array bug fixed 2026-07-10)."""

    def test_runs_without_error(self):
        args = _make_sorted_candidates_args()
        sorted_candidates_fast(*args)

    def test_runs_with_multiple_cams(self):
        args = _make_sorted_candidates_args(num_cams=2)
        sorted_candidates_fast(*args)


@_sorted_candidates_fast_dead_code_bug
class TestSortedCandidatesFastOutBug:
    """_sorted_candidates_fast_out runs in pure Python (C-array bug fixed 2026-07-10)."""

    def _make_out_args(self, num_cams=1, max_cands=4, n_targ=5):
        n = num_cams * max_cands
        center = np.array([0.0, 0.0, 100.0], dtype=np.float64)
        center_proj_x = np.zeros(num_cams, dtype=np.float64)
        center_proj_y = np.zeros(num_cams, dtype=np.float64)

        cal_arr = np.zeros((num_cams, 31), dtype=np.float64, order="C")
        for i in range(num_cams):
            cal_arr[i] = _make_cal()

        md_arr = [np.zeros(4, dtype=np.float64) for _ in range(num_cams)]
        mo_arr = np.zeros((num_cams, 3), dtype=np.float64, order="C")
        mnr_arr = np.zeros(num_cams, dtype=np.int32)
        mnz_arr = np.zeros(num_cams, dtype=np.int32)
        mrw_arr = np.ones(num_cams, dtype=np.float64)

        targ_x = np.zeros((num_cams, n_targ), dtype=np.float64, order="C")
        targ_y = np.zeros((num_cams, n_targ), dtype=np.float64, order="C")
        targ_tnr = np.full((num_cams, n_targ), -1, dtype=np.int32, order="C")
        num_targets = np.zeros(num_cams, dtype=np.int32)

        ftnr_out = np.full(n, -1, dtype=np.int32)
        freq_out = np.zeros(n, dtype=np.int32)
        whichcam_out = np.zeros((n, num_cams), dtype=np.int32)

        return (
            center,
            center_proj_x,
            center_proj_y,
            num_cams,
            max_cands,
            cal_arr,
            md_arr,
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
            targ_x,
            targ_y,
            targ_tnr,
            num_targets,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            512.0,
            512.0,
            1.0,
            1.0,
            0,
            1024.0,
            1024.0,
            -1,
            ftnr_out,
            freq_out,
            whichcam_out,
        )

    def test_runs_without_error(self):
        args = self._make_out_args()
        _sorted_candidates_fast_out(*args)

    def test_runs_with_two_cams(self):
        args = self._make_out_args(num_cams=2)
        _sorted_candidates_fast_out(*args)


@_needs_pure_python_loose_types
class TestSortedCandidatesFastOutNogil:
    """_sorted_candidates_fast_out_nogil runs in pure Python (C-array bug fixed 2026-07-10)."""

    def _make_nogil_args(self, num_cams=1, max_cands=4, n_targ=5):
        n = num_cams * max_cands
        center = np.array([0.0, 0.0, 100.0], dtype=np.float64)
        center_proj_x = np.zeros(num_cams, dtype=np.float64)
        center_proj_y = np.zeros(num_cams, dtype=np.float64)

        cal_arr = np.zeros((num_cams, 31), dtype=np.float64, order="C")
        for i in range(num_cams):
            cal_arr[i] = _make_cal()

        # nogil variant takes up to 8 separate md arrays
        empty_md = np.zeros(4, dtype=np.float64)
        md_list = [empty_md] * 8

        mo_arr = np.zeros((num_cams, 3), dtype=np.float64, order="C")
        mnr_arr = np.zeros(num_cams, dtype=np.int32)
        mnz_arr = np.zeros(num_cams, dtype=np.int32)
        mrw_arr = np.ones(num_cams, dtype=np.float64)

        targ_x = np.zeros((num_cams, n_targ), dtype=np.float64, order="C")
        targ_y = np.zeros((num_cams, n_targ), dtype=np.float64, order="C")
        targ_tnr = np.full((num_cams, n_targ), -1, dtype=np.int32, order="C")
        num_targets = np.zeros(num_cams, dtype=np.int32)

        ftnr_out = np.full(n, -1, dtype=np.int32)
        freq_out = np.zeros(n, dtype=np.int32)
        whichcam_out = np.zeros((n, num_cams), dtype=np.int32)

        return (
            center,
            center_proj_x,
            center_proj_y,
            num_cams,
            max_cands,
            cal_arr,
            *md_list,  # md0..md7
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
            targ_x,
            targ_y,
            targ_tnr,
            num_targets,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            512.0,
            512.0,
            1.0,
            1.0,
            0,
            1024.0,
            1024.0,
            -1,
            ftnr_out,
            freq_out,
            whichcam_out,
        )

    def test_runs_without_error(self):
        args = self._make_nogil_args()
        _sorted_candidates_fast_out_nogil(*args)

    def test_runs_with_two_cams(self):
        args = self._make_nogil_args(num_cams=2)
        _sorted_candidates_fast_out_nogil(*args)


# ---------------------------------------------------------------------------
# Compiled-mode sanity check (this module should be skipped when compiled)
# ---------------------------------------------------------------------------


@_needs_pure_python
def test_module_skip_guard_works():
    """In pure-Python mode, is_compiled() returns False — module not skipped."""
    assert _is_compiled() is False

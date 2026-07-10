"""Pure-Python line-coverage tests for src/openptv2/algorithms/track_kernels.py.

Skip automatically when the compiled .so shadows the .py (coverage would be 0%).

Verification command (run from repo root):
    cp src/openptv2/algorithms/track_kernels.py /tmp/ppsrc/openptv2/algorithms/track_kernels.py
    COVERAGE_FILE=/tmp/.cov_track_kernels uv run pytest tests/unit/test_track_kernels_coverage.py \\
      -o pythonpath=/tmp/ppsrc \\
      -p no:cacheprovider \\
      --cov=/tmp/ppsrc/openptv2 \\
      --cov-config=/tmp/covrc \\
      --cov-report=term-missing \\
      -q 2>&1 | grep -E "(algorithms/track_kernels\\.|TOTAL|passed|failed|error)"
"""

import types

import numpy as np
import pytest

# Guard: skip entire module when compiled Cython .so is active.
from openptv2.algorithms.track_kernels import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.track_kernels import (
    is_compiled,
    pack_cal_array,
    pack_mmlut,
    # re-exports from sub-modules
    point_to_pixel_fast,
    searchquader_fast,
    candsearch_in_pix_fast,
    candsearch_in_pix_rest_fast,
    sort_candidates_by_freq_fast,
    sorted_candidates_fast,
    point_position_fast,
    trackcorr_loop_fast,
    trackback_loop_fast,
    track3d_loop_fast,
    targ_rec_fast,
    init_mmlut_data_fast,
)


# ---------------------------------------------------------------------------
# Helpers: minimal stub objects
# ---------------------------------------------------------------------------

def _make_cal():
    """Build a minimal stub Calibration with all fields pack_cal_array reads."""
    ext = types.SimpleNamespace(
        x0=1.0,
        y0=2.0,
        z0=3.0,
        dm=np.eye(3, dtype=np.float64),
    )
    int_ = types.SimpleNamespace(cc=100.0, xh=0.1, yh=0.2)
    glass = types.SimpleNamespace(vec_x=0.0, vec_y=0.0, vec_z=1.0)
    added = types.SimpleNamespace(
        k1=0.01, k2=0.02, k3=0.03,
        p1=0.04, p2=0.05,
        scx=1.0, she=0.0,
    )
    mmlut = types.SimpleNamespace(
        data=None,
        origin=np.zeros(3, dtype=np.float64),
        nr=0,
        nz=0,
        rw=2,
    )
    cal = types.SimpleNamespace(
        ext_par=ext,
        int_par=int_,
        glass_par=glass,
        added_par=added,
        mmlut=mmlut,
    )
    return cal


def _make_mm():
    """Stub multimedia parameters as pack_cal_array expects."""
    return types.SimpleNamespace(
        n1=1.0,
        n2=np.array([1.5, 1.5, 1.5], dtype=np.float64),
        n3=1.0,
        d=np.array([5.0, 0.0, 0.0], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# is_compiled
# ---------------------------------------------------------------------------

def test_is_compiled_returns_false_in_pure_python():
    result = is_compiled()
    assert result is False


# ---------------------------------------------------------------------------
# pack_cal_array
# ---------------------------------------------------------------------------

def test_pack_cal_array_returns_float64_array_of_31():
    cal = _make_cal()
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    assert isinstance(c, np.ndarray)
    assert c.dtype == np.float64
    assert c.shape == (31,)


def test_pack_cal_array_position_values():
    cal = _make_cal()
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    # ext_par x0, y0, z0
    assert c[0] == 1.0
    assert c[1] == 2.0
    assert c[2] == 3.0
    # identity dm
    assert c[3] == 1.0   # dm[0,0]
    assert c[4] == 0.0   # dm[1,0]
    assert c[5] == 0.0   # dm[2,0]
    assert c[6] == 0.0   # dm[0,1]
    assert c[7] == 1.0   # dm[1,1]
    assert c[8] == 0.0   # dm[2,1]
    assert c[9] == 0.0   # dm[0,2]
    assert c[10] == 0.0  # dm[1,2]
    assert c[11] == 1.0  # dm[2,2]


def test_pack_cal_array_interior_params():
    cal = _make_cal()
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    assert c[12] == 100.0  # cc
    assert c[13] == 0.1    # xh
    assert c[14] == 0.2    # yh


def test_pack_cal_array_glass_params():
    cal = _make_cal()
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    # glass vec = (0, 0, 1) → dist_o_glas = 1.0
    assert c[15] == 0.0    # gx
    assert c[16] == 0.0    # gy
    assert c[17] == 1.0    # gz
    assert c[18] == pytest.approx(1.0)   # dist_o_glas
    assert c[19] == pytest.approx(1.0)   # 1/dist_o_glas


def test_pack_cal_array_multimedia_params():
    cal = _make_cal()
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    assert c[20] == mm.n1       # n1
    assert c[21] == mm.n2[0]    # n2[0]
    assert c[22] == mm.n3       # n3
    assert c[23] == mm.d[0]     # d[0]


def test_pack_cal_array_added_params():
    cal = _make_cal()
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    assert c[24] == pytest.approx(0.01)   # k1
    assert c[25] == pytest.approx(0.02)   # k2
    assert c[26] == pytest.approx(0.03)   # k3
    assert c[27] == pytest.approx(0.04)   # p1
    assert c[28] == pytest.approx(0.05)   # p2
    assert c[29] == pytest.approx(1.0)    # scx
    assert c[30] == pytest.approx(0.0)    # she


def test_pack_cal_array_nonunit_glass_vector():
    """dist_o_glas and its reciprocal computed correctly for non-unit vector."""
    cal = _make_cal()
    cal.glass_par.vec_x = 3.0
    cal.glass_par.vec_y = 4.0
    cal.glass_par.vec_z = 0.0
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    import math
    expected = math.sqrt(9.0 + 16.0)  # 5.0
    assert c[18] == pytest.approx(expected)
    assert c[19] == pytest.approx(1.0 / expected)


def test_pack_cal_array_nontrivial_dm():
    """Rotation matrix entries are written in column-major order as expected."""
    cal = _make_cal()
    dm = np.arange(9, dtype=np.float64).reshape(3, 3)
    cal.ext_par.dm = dm
    mm = _make_mm()
    c = pack_cal_array(cal, mm)
    # c[3..11] = dm[0,0] dm[1,0] dm[2,0] dm[0,1] dm[1,1] dm[2,1] dm[0,2] dm[1,2] dm[2,2]
    assert c[3] == dm[0, 0]
    assert c[4] == dm[1, 0]
    assert c[5] == dm[2, 0]
    assert c[6] == dm[0, 1]
    assert c[7] == dm[1, 1]
    assert c[8] == dm[2, 1]
    assert c[9] == dm[0, 2]
    assert c[10] == dm[1, 2]
    assert c[11] == dm[2, 2]


# ---------------------------------------------------------------------------
# pack_mmlut — path 1: no real data (None)
# ---------------------------------------------------------------------------

def test_pack_mmlut_no_data_returns_synthetic():
    cal = _make_cal()
    # mmlut.data is None → synthetic table
    data, origin, nr, nz, rw = pack_mmlut(cal)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float64
    assert len(data) == 4
    assert np.all(data == 1.0)
    assert isinstance(origin, np.ndarray)
    assert len(origin) == 3
    assert np.all(origin == 0.0)
    assert nr == 2
    assert nz == 2
    assert rw == 1000.0


def test_pack_mmlut_empty_data_returns_synthetic():
    cal = _make_cal()
    # mmlut.data is empty array → still synthetic (len == 0)
    cal.mmlut.data = np.array([], dtype=np.float64)
    data, origin, nr, nz, rw = pack_mmlut(cal)
    assert len(data) == 4
    assert np.all(data == 1.0)


# ---------------------------------------------------------------------------
# pack_mmlut — path 2: real data present
# ---------------------------------------------------------------------------

def test_pack_mmlut_with_real_data():
    cal = _make_cal()
    real_data = np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32)
    real_origin = np.array([0.5, 0.0, -1.0], dtype=np.float32)
    cal.mmlut.data = real_data
    cal.mmlut.origin = real_origin
    cal.mmlut.nr = 2
    cal.mmlut.nz = 2
    cal.mmlut.rw = 10

    data_out, origin_out, nr, nz, rw = pack_mmlut(cal)
    # Should be float64 and match values
    assert data_out.dtype == np.float64
    assert origin_out.dtype == np.float64
    np.testing.assert_allclose(data_out, real_data.astype(np.float64))
    np.testing.assert_allclose(origin_out, real_origin.astype(np.float64))
    assert nr == 2
    assert nz == 2
    assert rw == pytest.approx(10.0)


def test_pack_mmlut_real_data_already_float64_no_copy():
    """When data is already float64, astype(copy=False) should return same buffer."""
    cal = _make_cal()
    real_data = np.ones(6, dtype=np.float64)
    cal.mmlut.data = real_data
    cal.mmlut.origin = np.zeros(3, dtype=np.float64)
    cal.mmlut.nr = 3
    cal.mmlut.nz = 2
    cal.mmlut.rw = 5

    data_out, origin_out, nr, nz, rw = pack_mmlut(cal)
    assert data_out is real_data  # no copy when already float64
    assert nr == 3
    assert nz == 2


# ---------------------------------------------------------------------------
# Re-exported symbols — existence checks (covers the import lines)
# ---------------------------------------------------------------------------

def test_reexported_callables_exist():
    """All re-exported names must be callable."""
    names = [
        point_to_pixel_fast,
        searchquader_fast,
        candsearch_in_pix_fast,
        candsearch_in_pix_rest_fast,
        sort_candidates_by_freq_fast,
        sorted_candidates_fast,
        point_position_fast,
        trackcorr_loop_fast,
        trackback_loop_fast,
        track3d_loop_fast,
        targ_rec_fast,
        init_mmlut_data_fast,
    ]
    for fn in names:
        assert callable(fn), f"{fn!r} is not callable"

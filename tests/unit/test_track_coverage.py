"""
Pure-Python line-coverage tests for src/openptv2/algorithms/track.py.

Skip automatically when the compiled .so shadows the .py (coverage would be 0%).

Verification command (run from repo root):
    cp src/openptv2/algorithms/track.py /tmp/ppsrc/openptv2/algorithms/track.py
    COVERAGE_FILE=/tmp/.cov_track uv run pytest tests/unit/test_track_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q 2>&1 | grep -E '(algorithms/track\\.|TOTAL|passed|failed|error)'
"""

import os
import types

import numpy as np
import pytest

from openptv2.algorithms.track import is_compiled as _is_compiled

_needs_pure_python = pytest.mark.skipif(
    _is_compiled(), reason="asserts is_compiled() is False by design"
)

import openptv2.algorithms.track as _track_mod
from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.constants import (
    COORD_UNUSED,
    PT_UNUSED,
    TR_UNUSED,
)
from openptv2.algorithms.parameters import (
    ControlPar,
    TrackParTuple,
    VolumePar,
)
from openptv2.algorithms.track import (
    MAX_CANDS,
    _make_foundpix,
    _make_foundpix_array,
    _pack_cal,
    _pack_cams_fast,
    _pack_cams_fast_tuples,
    _point_to_pixel_fast,
    _point_to_pixel_packed,
    _ptp_fast,
    _sync_soa_to_aos,
    _vec3_dist,
    add_particle,
    angle_acc,
    assess_new_position,
    candsearch_in_pix,
    candsearch_in_pix_rest,
    copy_foundpix_array,
    is_compiled,
    point_to_pixel,
    pos3d_in_bounds,
    predict,
    register_closest_neighbs,
    reset_foundpix_array,
    search_volume_center_moving,
    searchquader,
    sort,
    sort_candidates_by_freq,
    sorted_candidates_in_volume,
    track_forward_start,
    trackback_c,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.tracking_frame_buf import Frame, Target

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(_HERE, "../../test_data/track"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_cal(cam: int = 1) -> Calibration:
    return Calibration.from_file(
        os.path.join(DATA_DIR, f"cal/cam{cam}.tif.ori"),
        os.path.join(DATA_DIR, f"cal/cam{cam}.tif.addpar"),
    )


def _read_cpar(num_cams: int | None = None) -> ControlPar:
    return ControlPar.from_yaml(
        os.path.join(DATA_DIR, "parameters.yaml"),
        num_cams=num_cams,
    )


def _read_vpar() -> VolumePar:
    return VolumePar.from_yaml(os.path.join(DATA_DIR, "parameters.yaml"))


def _make_tpar(**kw) -> TrackParTuple:
    defaults = dict(
        dvxmin=-5.0,
        dvxmax=5.0,
        dvymin=-5.0,
        dvymax=5.0,
        dvzmin=-5.0,
        dvzmax=5.0,
        dangle=2.0,
        dacc=0.4,
        add=0,
        dsumg=0.0,
        dn=0.0,
        dnx=0.0,
        dny=0.0,
    )
    defaults.update(kw)
    return TrackParTuple(**defaults)


def _targs(*xyt_triples) -> list:
    """Build Target list from (x, y, tnr) triples."""
    return [
        Target(pnr=i, x=x, y=y, n=1, nx=1, ny=1, sumg=1, tnr=tnr)
        for i, (x, y, tnr) in enumerate(xyt_triples)
    ]


def _frame(num_cams: int = 2, max_targets: int = 20) -> Frame:
    return Frame(num_cams=num_cams, max_targets=max_targets)


def _fast_tuples_for(cpar: ControlPar, cals: list) -> tuple:
    fc, fm = _pack_cams_fast(cals, cpar.mm)
    return _pack_cams_fast_tuples(fc, fm)


def _pix_info(cpar: ControlPar) -> tuple:
    return (
        cpar.imx,
        cpar.imy,
        cpar.imx * 0.5,
        cpar.imy * 0.5,
        1.0 / cpar.pix_x,
        1.0 / cpar.pix_y,
        cpar.chfield,
    )


# ---------------------------------------------------------------------------
# Mock FrameBuffer (no file I/O)
# ---------------------------------------------------------------------------


class _MockFB:
    def __init__(self, num_cams: int = 1, max_targets: int = 20):
        self.num_cams = num_cams
        self.buf_len = 4
        self.buf = [_frame(num_cams, max_targets) for _ in range(4)]

    def fb_next(self):
        self.buf.append(self.buf.pop(0))

    def fb_prev(self):
        self.buf.insert(0, self.buf.pop())

    def read_frame_at_end(self, frame_num, read_links=True):
        pass

    def write_frame_from_start(self, frame_num):
        pass


def _make_run(num_cams: int = 1) -> types.SimpleNamespace:
    cpar = _read_cpar(num_cams=num_cams)
    cals = [_read_cal(cam=c + 1) for c in range(num_cams)]
    return types.SimpleNamespace(
        fb=_MockFB(num_cams=num_cams),
        cal=cals,
        tpar=_make_tpar(dangle=120.0, dacc=5.0),
        vpar=_read_vpar(),
        cpar=cpar,
        seq_par=types.SimpleNamespace(first=1, last=4),
        npart=0.0,
        nlinks=0.0,
        lmax=0.0,
        ymin=0.0,
        ymax=0.0,
        flatten_tol=0.0001,
    )


# ===========================================================================
# Tier 1 — pure stateless helpers
# ===========================================================================


@_needs_pure_python
def test_is_compiled_returns_false():
    assert is_compiled() is False


def test_vec3_dist_zero():
    a = [1.0, 2.0, 3.0]
    assert _vec3_dist(a, a) == pytest.approx(0.0)


def test_vec3_dist_345():
    assert _vec3_dist([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)


def test_predict_extrapolates():
    prev = np.array([0.0, 0.0, 0.0])
    curr = np.array([1.0, 1.0, 1.0])
    out = np.zeros(3)
    predict(prev, curr, out)
    assert np.allclose(out, [2.0, 2.0, 2.0])


def test_search_volume_center_moving():
    c = search_volume_center_moving(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0])
    )
    assert np.allclose(c, [2.0, 4.0, 6.0])


def test_pos3d_in_bounds_inside():
    assert pos3d_in_bounds(np.array([0.0, 0.0, 0.0]), _make_tpar()) is True


def test_pos3d_in_bounds_outside_x():
    assert pos3d_in_bounds(np.array([10.0, 0.0, 0.0]), _make_tpar()) is False


def test_pos3d_in_bounds_outside_z():
    assert pos3d_in_bounds(np.array([0.0, 0.0, 10.0]), _make_tpar()) is False


def test_angle_acc_identical_vecs():
    s = np.array([0.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    angle, acc = angle_acc(s, v, v)
    assert angle == pytest.approx(0.0)
    assert acc == pytest.approx(0.0)


def test_angle_acc_opposite_vecs():
    s = np.array([0.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    w = np.array([-1.0, 0.0, 0.0])
    angle, acc = angle_acc(s, v, w)
    assert angle == pytest.approx(200.0)


def test_angle_acc_general():
    s = np.array([0.0, 0.0, 0.0])
    v = np.array([1.0, 1.0, 1.0])
    w = np.array([1.1, 1.0, 1.0])
    angle, acc = angle_acc(s, v, w)
    assert 0.0 < angle < 200.0
    assert acc > 0.0


def test_angle_acc_zero_norm_cand_eq_start():
    """cand == start → v1 = zero vector → zero-norm branch."""
    s = np.array([1.0, 2.0, 3.0])
    v = np.array([2.0, 3.0, 4.0])
    angle, acc = angle_acc(s, v, s.copy())
    assert angle == pytest.approx(0.0)


def test_make_foundpix_structure():
    fp = _make_foundpix(3)
    assert fp[0] == TR_UNUSED
    assert fp[1] == 0
    assert len(fp[2]) == 3


def test_make_foundpix_array_length():
    arr = _make_foundpix_array(5, 2)
    assert len(arr) == 5
    for fp in arr:
        assert fp[0] == TR_UNUSED
        assert fp[1] == 0


def test_reset_foundpix_array():
    arr = _make_foundpix_array(3, 2)
    arr[0][0] = 99
    arr[0][1] = 7
    arr[0][2][0] = 1
    reset_foundpix_array(arr, 3, 2)
    assert arr[0][0] == TR_UNUSED
    assert arr[0][1] == 0
    assert arr[0][2][0] == 0


def test_copy_foundpix_array():
    src = _make_foundpix_array(2, 2)
    src[0][0] = 7
    src[0][1] = 3
    src[0][2][1] = 1
    dest = _make_foundpix_array(2, 2)
    copy_foundpix_array(dest, src, 2, 2)
    assert dest[0][0] == 7
    assert dest[0][2][1] == 1


def test_sort_basic():
    a = [3.0, 1.0, 2.0]
    b = [30, 10, 20]
    sort(3, a, b)
    assert a[0] == pytest.approx(1.0)
    assert b[0] == 10


def test_sort_single_element():
    a = [9.0]
    b = [90]
    sort(1, a, b)
    assert a[0] == pytest.approx(9.0)


def test_sort_two_elements_already_sorted():
    a = [1.0, 2.0]
    b = [10, 20]
    sort(2, a, b)
    assert a[0] == pytest.approx(1.0)


def test_sort_candidates_by_freq_all_unused():
    n = 2 * MAX_CANDS
    items = _make_foundpix_array(n, 2)
    diff = sort_candidates_by_freq(items, 2)
    assert diff == 0


def test_sort_candidates_by_freq_one_seen_twice():
    """Particle 5 seen in both cams → freq=2, rises to top."""
    n = 2 * MAX_CANDS
    items = _make_foundpix_array(n, 2)
    items[0][0] = 5  # cam-0 slot 0
    items[MAX_CANDS][0] = 5  # cam-1 slot 0
    diff = sort_candidates_by_freq(items, 2)
    assert diff >= 1
    assert items[0][0] == 5
    assert items[0][1] == 2


def test_sort_candidates_by_freq_unique_particles():
    """All different particles → no merging, diff=0 or small."""
    n = 2 * MAX_CANDS
    items = _make_foundpix_array(n, 2)
    items[0][0] = 1
    items[1][0] = 2
    items[MAX_CANDS][0] = 3
    items[MAX_CANDS + 1][0] = 4
    sort_candidates_by_freq(items, 2)  # must not crash


# ---------------------------------------------------------------------------
# candsearch helpers
# ---------------------------------------------------------------------------


def test_candsearch_in_pix_finds_near_target():
    cpar = _read_cpar()
    targets = _targs((640.0, 512.0, PT_UNUSED), (700.0, 600.0, PT_UNUSED))
    p = candsearch_in_pix(targets, 2, 640.5, 512.5, 2.0, 2.0, 2.0, 2.0, cpar)
    assert 0 in p  # target 0 is within 2px


def test_candsearch_in_pix_none_in_range():
    cpar = _read_cpar()
    targets = _targs((100.0, 100.0, PT_UNUSED))
    p = candsearch_in_pix(targets, 1, 640.0, 512.0, 1.0, 1.0, 1.0, 1.0, cpar)
    assert all(x == PT_UNUSED for x in p)


def test_candsearch_in_pix_center_outside_image():
    cpar = _read_cpar()
    targets = _targs((640.0, 512.0, PT_UNUSED))
    p = candsearch_in_pix(targets, 1, -100.0, 512.0, 2.0, 2.0, 2.0, 2.0, cpar)
    assert all(x == PT_UNUSED for x in p)


def test_candsearch_in_pix_zero_targets():
    cpar = _read_cpar()
    p = candsearch_in_pix([], 0, 640.0, 512.0, 10.0, 10.0, 10.0, 10.0, cpar)
    assert all(x == PT_UNUSED for x in p)


def test_candsearch_in_pix_rest_finds_unused_target():
    cpar = _read_cpar()
    targets = _targs(
        (640.0, 512.0, TR_UNUSED),
        (641.0, 513.0, 5),  # already used → skipped
    )
    p = [PT_UNUSED] * 4
    count = candsearch_in_pix_rest(
        targets, 2, 640.0, 512.0, 2.0, 2.0, 2.0, 2.0, p, cpar
    )
    assert count == 1
    assert p[0] == 0  # index of the unused target


def test_candsearch_in_pix_rest_outside_image():
    cpar = _read_cpar()
    targets = _targs((640.0, 512.0, TR_UNUSED))
    p = [PT_UNUSED] * 4
    count = candsearch_in_pix_rest(
        targets, 1, -200.0, 512.0, 2.0, 2.0, 2.0, 2.0, p, cpar
    )
    assert count == 0


def test_candsearch_in_pix_rest_no_targets():
    cpar = _read_cpar()
    p = [PT_UNUSED] * 4
    count = candsearch_in_pix_rest([], 0, 640.0, 512.0, 10.0, 10.0, 10.0, 10.0, p, cpar)
    assert count == 0


# ===========================================================================
# Tier 2 — calibration-based projection
# ===========================================================================


def test_pack_cal_returns_tuple():
    cal = _read_cal()
    cpar = _read_cpar()
    pc = _pack_cal(cal, cpar.mm)
    assert isinstance(pc, tuple)
    assert len(pc) >= 35  # 35 or 36 depending on version


def test_pack_cal_no_mmlut_data():
    cal = _read_cal()
    cal.mmlut.data = None
    cpar = _read_cpar()
    pc = _pack_cal(cal, cpar.mm)
    # mmlut_data stored somewhere in the tuple; value should reflect None
    assert any(v is None for v in pc)


def test_pack_cams_fast_and_tuples():
    cpar = _read_cpar(num_cams=1)
    cal = _read_cal(1)
    fc, fm = _pack_cams_fast([cal], cpar.mm)
    result = _pack_cams_fast_tuples(fc, fm)
    assert len(result) == 6
    # cal_t has one element per cam
    assert len(result[0]) == 1


def test_point_to_pixel_packed_chfield0():
    cal = _read_cal()
    cpar = _read_cpar()
    pc = _pack_cal(cal, cpar.mm)
    pos = np.array([185.5, 3.2, 203.9])
    x, y = _point_to_pixel_packed(
        pos,
        pc,
        cpar.imx * 0.5,
        cpar.imy * 0.5,
        1.0 / cpar.pix_x,
        1.0 / cpar.pix_y,
        0,
    )
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_point_to_pixel_packed_chfield1():
    cal = _read_cal()
    cpar = _read_cpar()
    pc = _pack_cal(cal, cpar.mm)
    pos = np.array([185.5, 3.2, 203.9])
    x, y = _point_to_pixel_packed(
        pos,
        pc,
        cpar.imx * 0.5,
        cpar.imy * 0.5,
        1.0 / cpar.pix_x,
        1.0 / cpar.pix_y,
        1,
    )
    assert isinstance(x, float)


def test_point_to_pixel_packed_chfield2():
    cal = _read_cal()
    cpar = _read_cpar()
    pc = _pack_cal(cal, cpar.mm)
    pos = np.array([185.5, 3.2, 203.9])
    x, y = _point_to_pixel_packed(
        pos,
        pc,
        cpar.imx * 0.5,
        cpar.imy * 0.5,
        1.0 / cpar.pix_x,
        1.0 / cpar.pix_y,
        2,
    )
    assert isinstance(x, float)


def test_point_to_pixel_packed_with_mmlut_data():
    """Set up a synthetic mmlut so the lookup branch executes."""
    cal = _read_cal()
    cpar = _read_cpar()
    nr, nz = 100, 100
    cal.mmlut.data = np.ones(nr * nz, dtype=np.float64) * 1.001
    cal.mmlut.origin = np.zeros(3, dtype=np.float64)
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = 1000
    pc = _pack_cal(cal, cpar.mm)
    pos = np.array([185.5, 3.2, 203.9])
    x, y = _point_to_pixel_packed(
        pos,
        pc,
        cpar.imx * 0.5,
        cpar.imy * 0.5,
        1.0 / cpar.pix_x,
        1.0 / cpar.pix_y,
        0,
    )
    assert isinstance(x, float)


def test_point_to_pixel_packed_camera_center():
    """pos near camera center → may trigger r~0 branch (no crash required)."""
    cal = _read_cal()
    cpar = _read_cpar()
    pc = _pack_cal(cal, cpar.mm)
    pos = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0])
    try:
        _point_to_pixel_packed(
            pos,
            pc,
            cpar.imx * 0.5,
            cpar.imy * 0.5,
            1.0 / cpar.pix_x,
            1.0 / cpar.pix_y,
            0,
        )
    except (ZeroDivisionError, FloatingPointError):
        pass  # degenerate geometry; branch was still reached


def test_ptp_fast_returns_floats():
    cpar = _read_cpar(num_cams=1)
    cal = _read_cal(1)
    fc, fm = _pack_cams_fast([cal], cpar.mm)
    pos = np.array([185.5, 3.2, 203.9])
    x, y = _ptp_fast(
        pos,
        fc[0],
        fm[0],
        cpar.imx * 0.5,
        cpar.imy * 0.5,
        1.0 / cpar.pix_x,
        1.0 / cpar.pix_y,
        0,
    )
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_point_to_pixel_fast():
    cal = _read_cal()
    cpar = _read_cpar()
    pos = np.array([185.5, 3.2, 203.9])
    x, y = _point_to_pixel_fast(
        pos,
        cal,
        cpar.imx,
        cpar.imy,
        cpar.pix_x,
        cpar.pix_y,
        0,
        cpar.mm,
    )
    assert isinstance(x, float)


def test_point_to_pixel():
    cal = _read_cal()
    cpar = _read_cpar()
    x, y = point_to_pixel(np.array([185.5, 3.2, 203.9]), cal, cpar)
    assert isinstance(x, float)


def test_searchquader_returns_four_arrays():
    cpar = _read_cpar()
    calib = [_read_cal(c + 1) for c in range(cpar.num_cams)]
    tpar = _make_tpar(
        dvxmin=-0.2,
        dvxmax=0.2,
        dvymin=-0.2,
        dvymax=0.2,
        dvzmin=-0.1,
        dvzmax=0.1,
        dangle=120.0,
        dacc=0.4,
        add=1,
    )
    point = np.array([185.5, 3.2, 203.9])
    xr, xl, yd, yu = searchquader(point, tpar, cpar, calib)
    assert len(xr) == cpar.num_cams
    assert len(xl) == cpar.num_cams


def test_searchquader_zero_velocity():
    cpar = _read_cpar(num_cams=1)
    calib = [_read_cal(1)]
    tpar = _make_tpar(
        dvxmin=0.0, dvxmax=0.0, dvymin=0.0, dvymax=0.0, dvzmin=0.0, dvzmax=0.0
    )
    point = np.array([185.5, 3.2, 203.9])
    xr, xl, yd, yu = searchquader(point, tpar, cpar, calib)
    assert abs(xr[0] - xl[0]) < 1e-6


# ===========================================================================
# Tier 3 — Frame-based helpers
# ===========================================================================


def test_sync_soa_to_aos_no_parts():
    frm = _frame()
    frm.num_parts = 0
    _sync_soa_to_aos(frm)  # must not raise


def test_sync_soa_to_aos_one_particle():
    frm = _frame(num_cams=2, max_targets=20)
    frm.num_parts = 1
    frm.path_x[0] = [1.0, 2.0, 3.0]
    frm.path_prev[0] = 0
    frm.path_next[0] = 1
    frm.path_prio[0] = 2
    frm.corres_nr[0] = 5
    frm.corres_p[0, :] = [0, 1, -1, -1]
    frm.num_targets[0] = 1
    frm.targ_tnr[0, 0] = 3
    _sync_soa_to_aos(frm)
    assert np.allclose(frm.path_info[0].x, [1.0, 2.0, 3.0])
    assert frm.path_info[0].prev == 0
    assert frm.targets[0][0].tnr == 3


def test_add_particle_no_cand():
    frm = _frame(num_cams=2, max_targets=20)
    frm.num_parts = 0
    pos = np.array([1.0, 2.0, 3.0])
    cand_inds = [[PT_UNUSED] for _ in range(4)]
    add_particle(frm, pos, cand_inds)
    assert frm.num_parts == 1
    assert np.allclose(frm.path_info[0].x, [1.0, 2.0, 3.0])


def test_add_particle_with_cand():
    frm = _frame(num_cams=2, max_targets=20)
    frm.num_parts = 0
    frm.targets[0][0] = Target(
        pnr=0, x=5.0, y=5.0, n=1, nx=1, ny=1, sumg=1, tnr=PT_UNUSED
    )
    frm.targ_tnr[0, 0] = PT_UNUSED

    pos = np.array([0.0, 0.0, 0.0])
    cand_inds = [[0], [PT_UNUSED], [PT_UNUSED], [PT_UNUSED]]
    add_particle(frm, pos, cand_inds)

    assert frm.num_parts == 1
    assert frm.targets[0][0].tnr == 0
    assert frm.targ_tnr[0, 0] == 0


def test_register_closest_neighbs_nothing_found(monkeypatch):
    monkeypatch.setattr(
        _track_mod,
        "_candsearch_in_pix_fast",
        lambda *a, **kw: (PT_UNUSED, PT_UNUSED, PT_UNUSED, PT_UNUSED),
    )
    cpar = _read_cpar(num_cams=2)
    frm = _frame(num_cams=2)
    reg = _make_foundpix_array(MAX_CANDS, 2)
    targ_x = np.full(20, COORD_UNUSED)
    targ_y = np.full(20, COORD_UNUSED)
    targ_tnr = np.full(20, PT_UNUSED, dtype=np.int32)

    register_closest_neighbs(
        frm.targets[0],
        0,
        0,
        640.0,
        512.0,
        10.0,
        10.0,
        10.0,
        10.0,
        reg,
        cpar,
        _targ_x=targ_x,
        _targ_y=targ_y,
        _targ_tnr=targ_tnr,
    )
    assert all(reg[i][0] == TR_UNUSED for i in range(MAX_CANDS))


def test_register_closest_neighbs_one_found(monkeypatch):
    targ_tnr = np.full(20, PT_UNUSED, dtype=np.int32)
    targ_tnr[2] = 7  # target index 2 has tnr=7

    monkeypatch.setattr(
        _track_mod,
        "_candsearch_in_pix_fast",
        lambda *a, **kw: (2, PT_UNUSED, PT_UNUSED, PT_UNUSED),
    )
    cpar = _read_cpar(num_cams=2)
    frm = _frame(num_cams=2)
    reg = _make_foundpix_array(MAX_CANDS, 2)
    targ_x = np.full(20, COORD_UNUSED)
    targ_y = np.full(20, COORD_UNUSED)

    register_closest_neighbs(
        frm.targets[0],
        20,
        0,
        640.0,
        512.0,
        10.0,
        10.0,
        10.0,
        10.0,
        reg,
        cpar,
        _targ_x=targ_x,
        _targ_y=targ_y,
        _targ_tnr=targ_tnr,
    )
    assert reg[0][0] == 7  # tnr of target at index 2


def test_sorted_candidates_in_volume_returns_none(monkeypatch):
    monkeypatch.setattr(
        _track_mod,
        "_sorted_candidates_fast",
        lambda *a, **kw: (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 1), dtype=np.int32),
            0,
        ),
    )
    cpar = _read_cpar(num_cams=1)
    frm = _frame(num_cams=1)
    run = types.SimpleNamespace(cpar=cpar, tpar=_make_tpar())
    ft = _fast_tuples_for(cpar, [_read_cal(1)])
    pi = _pix_info(cpar)

    result = sorted_candidates_in_volume(
        center=np.zeros(3),
        center_proj=[[0.0, 0.0]],
        frm=frm,
        run=run,
        _pix_info=pi,
        _fast_tuples=ft,
    )
    assert result is None


def test_sorted_candidates_in_volume_returns_list(monkeypatch):
    monkeypatch.setattr(
        _track_mod,
        "_sorted_candidates_fast",
        lambda *a, **kw: (
            np.array([5], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([[1]], dtype=np.int32),
            1,
        ),
    )
    cpar = _read_cpar(num_cams=1)
    frm = _frame(num_cams=1)
    run = types.SimpleNamespace(cpar=cpar, tpar=_make_tpar())
    ft = _fast_tuples_for(cpar, [_read_cal(1)])
    pi = _pix_info(cpar)

    result = sorted_candidates_in_volume(
        center=np.zeros(3),
        center_proj=[[0.0, 0.0]],
        frm=frm,
        run=run,
        _pix_info=pi,
        _fast_tuples=ft,
    )
    assert result is not None
    assert len(result) == 2  # 1 real candidate + sentinel
    assert result[0]["ftnr"] == 5
    assert result[0]["freq"] == 2
    assert result[-1]["ftnr"] == TR_UNUSED


def test_assess_new_position_no_cands(monkeypatch):
    monkeypatch.setattr(
        _track_mod,
        "_candsearch_in_pix_rest_fast",
        lambda *a, **kw: (PT_UNUSED, 0),
    )
    cpar = _read_cpar(num_cams=1)
    cal = _read_cal(1)
    frm = _frame(num_cams=1)
    run = types.SimpleNamespace(cpar=cpar, cal=[cal], flatten_tol=0.0001)
    fc, fm = _pack_cams_fast([cal], cpar.mm)
    pi = _pix_info(cpar)

    targ_pos = [[COORD_UNUSED, COORD_UNUSED] for _ in range(4)]
    cand_inds = [[PT_UNUSED] for _ in range(4)]

    valid = assess_new_position(
        pos=np.array([0.0, 0.0, 0.0]),
        targ_pos=targ_pos,
        cand_inds=cand_inds,
        frm=frm,
        run=run,
        _fast_cals=fc,
        _fast_mmluts=fm,
        _pix_info=pi,
    )
    assert valid == 0


def test_assess_new_position_no_pix_info(monkeypatch):
    """Exercise the else-branch where _pix_info is None."""
    monkeypatch.setattr(
        _track_mod,
        "_candsearch_in_pix_rest_fast",
        lambda *a, **kw: (PT_UNUSED, 0),
    )
    cpar = _read_cpar(num_cams=1)
    cal = _read_cal(1)
    frm = _frame(num_cams=1)
    run = types.SimpleNamespace(cpar=cpar, cal=[cal], flatten_tol=0.0001)
    fc, fm = _pack_cams_fast([cal], cpar.mm)

    targ_pos = [[COORD_UNUSED, COORD_UNUSED] for _ in range(4)]
    cand_inds = [[PT_UNUSED] for _ in range(4)]

    valid = assess_new_position(
        pos=np.array([0.0, 0.0, 0.0]),
        targ_pos=targ_pos,
        cand_inds=cand_inds,
        frm=frm,
        run=run,
        _fast_cals=fc,
        _fast_mmluts=fm,
        _pix_info=None,  # forces the else branch
    )
    assert valid == 0


def test_assess_new_position_candidate_found(monkeypatch):
    """When the rest-search finds index 0, valid_cams increments."""
    monkeypatch.setattr(
        _track_mod,
        "_candsearch_in_pix_rest_fast",
        lambda *a, **kw: (0, 1),
    )
    cpar = _read_cpar(num_cams=1)
    cal = _read_cal(1)
    frm = _frame(num_cams=1, max_targets=20)
    frm.targ_x[0, 0] = 640.0
    frm.targ_y[0, 0] = 512.0
    frm.targ_tnr[0, 0] = TR_UNUSED
    frm.num_targets[0] = 1

    run = types.SimpleNamespace(cpar=cpar, cal=[cal], flatten_tol=0.0001)
    fc, fm = _pack_cams_fast([cal], cpar.mm)
    pi = _pix_info(cpar)

    targ_pos = [[COORD_UNUSED, COORD_UNUSED] for _ in range(4)]
    cand_inds = [[PT_UNUSED] for _ in range(4)]

    valid = assess_new_position(
        pos=np.array([185.5, 3.2, 203.9]),
        targ_pos=targ_pos,
        cand_inds=cand_inds,
        frm=frm,
        run=run,
        _fast_cals=fc,
        _fast_mmluts=fm,
        _pix_info=pi,
    )
    assert valid >= 1  # candidate found in 1 camera


# ===========================================================================
# Tier 4 — tracking-loop functions (monkeypatch fast kernels)
# ===========================================================================


def test_track_forward_start():
    run = _make_run(num_cams=1)
    track_forward_start(run)  # reads 4 frames (all no-ops) and rotates buffer


def test_trackcorr_c_loop_early_step(monkeypatch):
    """step < last-2 → read_frame_at_end branch."""
    monkeypatch.setattr(
        _track_mod,
        "_trackcorr_loop_fast",
        lambda *a, **kw: (0, 0),
    )
    run = _make_run(num_cams=1)
    trackcorr_c_loop(run, 1)
    assert run.npart >= 0
    assert run.nlinks >= 0


def test_trackcorr_c_loop_late_step(monkeypatch):
    """step >= last-2 → sets buf[-1].num_parts = 0."""
    monkeypatch.setattr(
        _track_mod,
        "_trackcorr_loop_fast",
        lambda *a, **kw: (0, 0),
    )
    run = _make_run(num_cams=1)
    trackcorr_c_loop(run, run.seq_par.last - 1)


def test_trackcorr_c_finish():
    run = _make_run(num_cams=1)
    run.npart = 12.0
    run.nlinks = 9.0
    trackcorr_c_finish(run, run.seq_par.last)
    # averages printed; fb_next + write_frame called (no-ops)


def test_trackback_c_mocked(monkeypatch):
    """Cover trackback_c including both loop branches."""
    monkeypatch.setattr(
        _track_mod,
        "_trackback_loop_fast",
        lambda *a, **kw: (0, 0),
    )
    run = _make_run(num_cams=1)
    run.seq_par = types.SimpleNamespace(first=1, last=4)
    nlinks = trackback_c(run)
    assert nlinks >= 0.0


def test_trackback_c_read_old_frames(monkeypatch):
    """seq_par spans 5 steps → _bk_step > first+2 triggers extra read."""
    monkeypatch.setattr(
        _track_mod,
        "_trackback_loop_fast",
        lambda *a, **kw: (0, 0),
    )
    run = _make_run(num_cams=1)
    # first=1, last=6 → inner loop range(5,1,-1)=[5,4,3,2]
    # step 5 > first+2=3 → read_frame_at_end branch hit
    run.seq_par = types.SimpleNamespace(first=1, last=6)
    nlinks = trackback_c(run)
    assert nlinks >= 0.0

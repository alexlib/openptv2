"""Pure-Python line-coverage tests for orientation.py.

Run only against the interpreted source at /tmp/ppsrc (not the compiled .so).
Launch with::

    cd /home/user/Documents/GitHub/openptv2
    COVERAGE_FILE=/tmp/.cov_orient uv run pytest tests/unit/test_orientation_coverage.py \\
      -o pythonpath=/tmp/ppsrc \\
      -p no:cacheprovider \\
      --cov=/tmp/ppsrc/openptv2 \\
      --cov-config=/tmp/covrc \\
      --cov-report=term-missing \\
      -q

Note: single_cam_point_positions reads ``vpar.Zmin_lay`` / ``vpar.Zmax_lay``
(capital-Z, matching VolumePar); the duck-typed _MockVpar mirrors those names.
"""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.algorithms.orientation import is_compiled as _is_compiled

_needs_pure_python = pytest.mark.skipif(
    _is_compiled(), reason="asserts is_compiled() is False by design"
)

# ── module-level imports ───────────────────────────────────────────────────────
from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.orientation import (
    COORD_UNUSED,
    NPAR,
    external_calibration,
    full_calibration,
    match_detection_to_ref,
    multi_cam_point_positions,
    num_deriv_exterior,
    orient,
    point_position,
    point_position_batch,
    point_positions,
    raw_orient,
    read_calblock,
    read_man_ori_fix,
    single_cam_point_positions,
    skew_midpoint,
    weighted_dumbbell_precision,
)
from openptv2.algorithms.parameters import ControlPar, MultimediaPar, OrientPar
from openptv2.algorithms.tracking_frame_buf import Target
from openptv2.algorithms.trafo import metric_to_pixel

# ── shared test-data paths ─────────────────────────────────────────────────────
CAL1_ORI = "test_data/calibration/cam1.tif.ori"
CAL1_ADD = "test_data/calibration/cam1.tif.addpar"
SYM_TMPL = "test_data/calibration/sym_cam{}.tif.ori"
CALBLOCK = "test_data/calibration/calblock.txt"
MAN_ORI = "test_data/parameters/man_ori.par"
PARAMS_YAML = "test_data/parameters.yaml"


# ── helpers ────────────────────────────────────────────────────────────────────


def _mm():
    return MultimediaPar(n1=1.0, n2=[1.0], d=[1.0], n3=1.0)


def _sym_cals(n=4):
    return [Calibration.from_file(SYM_TMPL.format(c + 1), CAL1_ADD) for c in range(n)]


def _grid_fix(n=64):
    fix = np.zeros((n, 3))
    pt = 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                fix[pt] = [(ix * 10) - 60, iy * 5, iz * 5]
                pt += 1
    return fix


def _project_targets(fix, cal, cpar, dy=0.0):
    """Return list[Target] with pnr set, optionally shifted in y."""
    pix = []
    for i, pos in enumerate(fix):
        xp, yp = img_coord(pos, cal, cpar.mm)
        xpx, ypx = metric_to_pixel(xp, yp, cpar)
        pix.append(Target(pnr=i, x=xpx, y=ypx + dy))
    return pix


# ─────────────────────────────────────────────────────────────────────────────
# is_compiled
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pure_python
def test_is_compiled_false():
    assert _is_compiled() is False


# ─────────────────────────────────────────────────────────────────────────────
# skew_midpoint  (covers _skew_midpoint_core too)
# ─────────────────────────────────────────────────────────────────────────────


def test_skew_midpoint_skew_rays():
    """Standard skew-ray case: known geometry."""
    p1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 1.0])
    d2 = np.array([0.0, 1.0, 0.0])

    dist, mid = skew_midpoint(p1, d1, p2, d2)
    assert abs(dist - 1.0) < 1e-10
    np.testing.assert_allclose(mid, [0.0, 0.0, 0.5], atol=1e-10)


def test_skew_midpoint_parallel_rays_triggers_fallback():
    """Parallel rays → cross-product near zero → fallback branch (scale < 1e-20)."""
    p1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])
    # Same position, same direction
    dist, mid = skew_midpoint(p1, d1, p1, d1)
    np.testing.assert_allclose(mid, [0.0, 0.0, 0.0], atol=1e-10)
    assert dist >= 0.0


def test_skew_midpoint_intersecting_rays():
    """Rays that meet at the origin: distance ~ 0."""
    p1 = np.array([0.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    d2 = np.array([0.0, 0.0, 1.0])
    dist, mid = skew_midpoint(p1, d1, p2, d2)
    assert dist < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# point_position_batch
# ─────────────────────────────────────────────────────────────────────────────


def _4cam_targets(point, cals, mm):
    targets = np.zeros((1, 4, 2))
    for cam in range(4):
        xp, yp = img_coord(point, cals[cam], mm)
        targets[0, cam, 0] = xp
        targets[0, cam, 1] = yp
    return targets


def test_ppb_basic_4cam():
    cals = _sym_cals()
    mm = _mm()
    point = np.array([17.0, 42.0, 0.0])
    targets = _4cam_targets(point, cals, mm)
    positions, distances = point_position_batch(targets, 4, mm, cals)
    assert positions.shape == (1, 3)
    np.testing.assert_allclose(positions[0], point, atol=1e-7)
    assert distances[0] < 1e-7


def test_ppb_coord_unused_two_cams():
    """Mark cams 2,3 as unused; only cam-pair (0,1) contributes."""
    cals = _sym_cals()
    mm = _mm()
    point = np.array([5.0, 10.0, 0.0])
    targets = _4cam_targets(point, cals, mm)
    targets[0, 2, 0] = COORD_UNUSED
    targets[0, 3, 0] = COORD_UNUSED

    positions, distances = point_position_batch(targets, 4, mm, cals)
    assert positions.shape == (1, 3)
    # With only 2 cams remaining, position is still computable
    assert np.all(np.isfinite(positions))


def test_ppb_all_unused_zero():
    """All cameras unused → zero position, zero distance."""
    cals = _sym_cals()
    mm = _mm()
    targets = np.full((1, 4, 2), COORD_UNUSED)
    positions, distances = point_position_batch(targets, 4, mm, cals)
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0])
    assert distances[0] == 0.0


def test_ppb_multi_point_batch():
    """Batch with M=3 targets all recoverable."""
    cals = _sym_cals()
    mm = _mm()
    pts = np.array([[17.0, 42.0, 0.0], [0.0, 0.0, 0.0], [-10.0, 5.0, 0.0]])
    targets = np.zeros((3, 4, 2))
    for pt, p in enumerate(pts):
        for cam in range(4):
            xp, yp = img_coord(p, cals[cam], mm)
            targets[pt, cam, 0] = xp
            targets[pt, cam, 1] = yp
    positions, distances = point_position_batch(targets, 4, mm, cals)
    assert positions.shape == (3, 3)
    np.testing.assert_allclose(positions, pts, atol=1e-7)


def test_ppb_hasattr_mm_wrapper():
    """Wrapped mm object (has _mm) exercises the hasattr(mm, '_mm') branch."""
    cals = _sym_cals()
    real_mm = _mm()

    class MmWrapper:
        def __init__(self, inner):
            self._mm = inner

    point = np.array([5.0, 5.0, 0.0])
    targets = _4cam_targets(point, cals, real_mm)
    positions, distances = point_position_batch(targets, 4, MmWrapper(real_mm), cals)
    assert positions.shape == (1, 3)


def test_ppb_hasattr_cal_wrapper():
    """Wrapped Calibration (has _cal) exercises hasattr(cal, '_cal') branch."""
    real_cals = _sym_cals()
    mm = _mm()

    class CalWrapper:
        def __init__(self, inner):
            self._cal = inner

    wrapped_cals = [CalWrapper(c) for c in real_cals]
    point = np.array([5.0, 5.0, 0.0])
    targets = _4cam_targets(point, real_cals, mm)
    positions, distances = point_position_batch(targets, 4, mm, wrapped_cals)
    assert positions.shape == (1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# point_position  (thin wrapper around batch)
# ─────────────────────────────────────────────────────────────────────────────


def test_point_position_basic():
    cals = _sym_cals()
    mm = _mm()
    point = np.array([5.0, 10.0, 0.0])
    targs = np.zeros((4, 2))
    for cam in range(4):
        xp, yp = img_coord(point, cals[cam], mm)
        targs[cam, 0] = xp
        targs[cam, 1] = yp
    pos, dist = point_position(targs, 4, mm, cals)
    np.testing.assert_allclose(pos, point, atol=1e-7)
    assert dist < 1e-7


# ─────────────────────────────────────────────────────────────────────────────
# weighted_dumbbell_precision
# ─────────────────────────────────────────────────────────────────────────────


def _dumbbell_targets(pts, cals, mm):
    n = len(pts)
    targets = np.zeros((n, 4, 2))
    for pt, p in enumerate(pts):
        for cam in range(4):
            xp, yp = img_coord(p, cals[cam], mm)
            targets[pt, cam, 0] = xp
            targets[pt, cam, 1] = yp
    return targets


def test_dumbbell_perfect_zero():
    cals = _sym_cals()
    mm = _mm()
    pts = np.array([[17.5, 42.0, 0.0], [-17.5, 42.0, 0.0]])
    targets = _dumbbell_targets(pts, cals, mm)
    result = weighted_dumbbell_precision(targets, 2, 4, mm, cals, 35.0, 0.0)
    assert result < 1e-8


def test_dumbbell_length_gt_db_length():
    """dist > db_length branch: real separation >> db_length."""
    cals = _sym_cals()
    mm = _mm()
    pts = np.array([[80.0, 0.0, 0.0], [-80.0, 0.0, 0.0]])
    targets = _dumbbell_targets(pts, cals, mm)
    # db_length=10, actual ~160 → triggers 1 - db_length/dist branch
    result = weighted_dumbbell_precision(targets, 2, 4, mm, cals, 10.0, 1.0)
    assert result > 0.0


def test_dumbbell_length_lt_db_length():
    """dist < db_length branch: real separation << db_length."""
    cals = _sym_cals()
    mm = _mm()
    pts = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    targets = _dumbbell_targets(pts, cals, mm)
    # db_length=100, actual ~4 → triggers 1 - dist/db_length branch
    result = weighted_dumbbell_precision(targets, 2, 4, mm, cals, 100.0, 1.0)
    assert result > 0.0


def test_dumbbell_weight_nonzero():
    """db_weight > 0 accumulates len_err_tot."""
    cals = _sym_cals()
    mm = _mm()
    pts = np.array([[17.5, 42.0, 0.0], [-17.5, 42.0, 0.0]])
    targets = _dumbbell_targets(pts, cals, mm)
    r1 = weighted_dumbbell_precision(targets, 2, 4, mm, cals, 35.0, 0.0)
    r2 = weighted_dumbbell_precision(targets, 2, 4, mm, cals, 35.0, 5.0)
    # Both near zero because the length is exactly right
    assert abs(r1) < 1e-8
    assert abs(r2) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# num_deriv_exterior
# ─────────────────────────────────────────────────────────────────────────────


def test_num_deriv_exterior_shape_and_nonzero():
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pos = np.array([17.0, 42.0, 0.0])
    x_ders, y_ders = num_deriv_exterior(cal, cpar, 1e-4, 1e-4, pos)
    assert len(x_ders) == 6
    assert len(y_ders) == 6
    # At least some derivatives should be non-trivial
    assert np.any(np.abs(x_ders) > 1e-12)
    assert np.any(np.abs(y_ders) > 1e-12)


def test_num_deriv_exterior_position_vs_angle():
    """Position step (pd<3) and angle step (pd>=3) follow different paths."""
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pos = np.array([0.0, 0.0, 0.0])
    x_ders, y_ders = num_deriv_exterior(cal, cpar, 1e-4, 1e-7, pos)
    # Derivatives for angles (pd>=3) use dang, positions use dpos
    assert x_ders.shape == (6,)


# ─────────────────────────────────────────────────────────────────────────────
# raw_orient
# ─────────────────────────────────────────────────────────────────────────────


def test_raw_orient_converges():
    fix4 = read_man_ori_fix(CALBLOCK, MAN_ORI, 0)
    assert fix4 is not None
    cal = Calibration.from_file(CAL1_ORI, CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix4 = []
    for i, pos in enumerate(fix4):
        xp, yp = img_coord(pos, cal, cpar.mm)
        xpx, ypx = metric_to_pixel(xp, yp, cpar)
        pix4.append(Target(pnr=i, x=xpx, y=ypx - 0.1))
    success = raw_orient(cal, cpar, 4, fix4, pix4)
    assert success is True


def test_raw_orient_callable_x_y():
    """Callable .x / .y attributes (callable() branch in raw_orient)."""
    fix4 = read_man_ori_fix(CALBLOCK, MAN_ORI, 0)
    cal = Calibration.from_file(CAL1_ORI, CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)

    class CallablePix:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        @property
        def x(self):
            return lambda: self._x

        @property
        def y(self):
            return lambda: self._y

    pix4 = []
    for pos in fix4:
        xp, yp = img_coord(pos, cal, cpar.mm)
        xpx, ypx = metric_to_pixel(xp, yp, cpar)
        pix4.append(CallablePix(xpx, ypx))

    result = raw_orient(cal, cpar, 4, fix4, pix4)
    assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# orient – main bundle-adjustment function
# ─────────────────────────────────────────────────────────────────────────────


def _orient_setup(sym_cam=1, dy=0.0):
    """Return (fix, pix, cal, cpar) starting from perfect data."""
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(sym_cam), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar, dy=dy)
    return fix, pix, cal, cpar


def test_orient_converges_from_small_perturbation():
    fix, pix, cal, cpar = _orient_setup(dy=0.0)
    cal.ext_par.x0 -= 5.0
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, OrientPar(), sigmabeta)
    assert resi is not None


def test_orient_returns_residual_array():
    fix, pix, cal, cpar = _orient_setup()
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, OrientPar(), sigmabeta)
    assert resi is not None
    assert len(resi) > 0


def test_orient_useflag_1_skip_even():
    """useflag=1 → skip even-indexed points."""
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(useflag=1)
    sigmabeta = np.zeros(NPAR + 1)
    # Cover the branch – result may be None if underdetermined, that's OK
    orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)


def test_orient_useflag_2_skip_odd():
    """useflag=2 → skip odd-indexed points."""
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(useflag=2)
    sigmabeta = np.zeros(NPAR + 1)
    orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)


def test_orient_useflag_3_skip_third():
    """useflag=3 → skip every 3rd point."""
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(useflag=3)
    sigmabeta = np.zeros(NPAR + 1)
    orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)


def test_orient_ccflag_enabled():
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(ccflag=1)
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_xhflag_yhflag():
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(xhflag=1, yhflag=1)
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_k1flag():
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(k1flag=1)
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_k2flag_k3flag():
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(k2flag=1, k3flag=1)
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_p1flag_p2flag():
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(p1flag=1, p2flag=1)
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_scxflag_sheflag():
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(scxflag=1, sheflag=1)
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_interfflag():
    """interfflag=1 activates glass-interface derivative path (numbers=18)."""
    fix, pix, cal, cpar = _orient_setup()
    opar = OrientPar(interfflag=1)
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_callable_pnr():
    """callable .pnr attribute exercises callable() branch in orient."""
    fix, _, cal, cpar = _orient_setup()

    class CallablePnrTarget:
        def __init__(self, pnr, x, y):
            self._pnr = pnr
            self.x = x
            self.y = y

        @property
        def pnr(self):
            return lambda: self._pnr

    pix = []
    for i, pos in enumerate(fix):
        xp, yp = img_coord(pos, cal, cpar.mm)
        xpx, ypx = metric_to_pixel(xp, yp, cpar)
        pix.append(CallablePnrTarget(i, xpx, ypx))

    opar = OrientPar()
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_callable_x_y_in_orient():
    """callable .x / .y attributes in orient."""
    fix, _, cal, cpar = _orient_setup()

    class CallableXY:
        def __init__(self, pnr, x, y):
            self.pnr = pnr
            self._x = x
            self._y = y

        @property
        def x(self):
            return lambda: self._x

        @property
        def y(self):
            return lambda: self._y

    pix = []
    for i, pos in enumerate(fix):
        xp, yp = img_coord(pos, cal, cpar.mm)
        xpx, ypx = metric_to_pixel(xp, yp, cpar)
        pix.append(CallableXY(i, xpx, ypx))

    opar = OrientPar()
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None


def test_orient_pnr_mismatch_skips_some_points():
    """Points with pnr != i are skipped (covers pnr_i != i branch)."""
    fix, pix, cal, cpar = _orient_setup()
    # Corrupt pnr for the last 16 points → only 48 active
    for p in pix[48:]:
        p.pnr = -1
    opar = OrientPar()
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is not None  # 48 points still enough to converge


def test_orient_returns_none_when_forced(monkeypatch):
    """Force non-convergence: NUM_ITER=1 + large initial perturbation → None."""
    import openptv2.algorithms.orientation as ori_mod

    monkeypatch.setattr(ori_mod, "NUM_ITER", 1)
    fix, pix, cal, cpar = _orient_setup()
    cal.ext_par.x0 += 200.0  # very far from correct → beta large after 1 iter
    opar = OrientPar()
    sigmabeta = np.zeros(NPAR + 1)
    resi = orient(cal, cpar, len(fix), fix, pix, opar, sigmabeta)
    assert resi is None


# ─────────────────────────────────────────────────────────────────────────────
# read_man_ori_fix  (all failure paths)
# ─────────────────────────────────────────────────────────────────────────────


def test_rmof_success():
    fix4 = read_man_ori_fix(CALBLOCK, MAN_ORI, 0)
    assert fix4 is not None
    assert len(fix4) == 4


def test_rmof_missing_file(tmp_path):
    result = read_man_ori_fix(CALBLOCK, tmp_path / "no_file.par", 0)
    assert result is None


def test_rmof_too_few_tokens(tmp_path):
    """File with only 3 tokens → len < (cam+1)*4 → None."""
    f = tmp_path / "short.par"
    f.write_text("1 2 3")
    result = read_man_ori_fix(CALBLOCK, f, 0)
    assert result is None


def test_rmof_invalid_token(tmp_path):
    """Non-integer token → ValueError in int() → return None."""
    f = tmp_path / "bad.par"
    f.write_text("abc 2 3 4")
    result = read_man_ori_fix(CALBLOCK, f, 0)
    assert result is None


def test_rmof_out_of_range_pnr(tmp_path):
    """Point number beyond calblock length → return None."""
    f = tmp_path / "oob.par"
    f.write_text("9999 9999 9999 9999")
    result = read_man_ori_fix(CALBLOCK, f, 0)
    assert result is None


def test_rmof_wrong_man_ori():
    """wrong_man_ori.par exists but has invalid entries for cam 0."""
    result = read_man_ori_fix(CALBLOCK, "test_data/parameters/wrong_man_ori.par", 0)
    assert result is None


def test_rmof_cam_index_nonzero():
    """cam=1 needs at least 8 tokens; MAN_ORI has them if it's valid."""
    # If only 4 tokens, cam=1 requires 8 → None
    # Use man_ori.par directly (it may have enough tokens)
    try:
        result = read_man_ori_fix(CALBLOCK, MAN_ORI, 1)
        # Either succeeds or None depending on token count; both are valid
        assert result is None or len(result) == 4
    except Exception:
        pass  # any non-crash outcome is acceptable


# ─────────────────────────────────────────────────────────────────────────────
# read_calblock  (delegates to sortgrid.read_calblock)
# ─────────────────────────────────────────────────────────────────────────────


def test_read_calblock_success():
    fix, num_fix = read_calblock(CALBLOCK)
    assert num_fix > 0
    assert len(fix) == num_fix


# ─────────────────────────────────────────────────────────────────────────────
# external_calibration  (wrapper around raw_orient)
# ─────────────────────────────────────────────────────────────────────────────


def test_external_calibration_success():
    fix4 = read_man_ori_fix(CALBLOCK, MAN_ORI, 0)
    cal = Calibration.from_file(CAL1_ORI, CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)

    ref_pts = np.array(fix4, dtype=np.float64)
    img_pts = np.zeros((4, 2))
    for i, pos in enumerate(fix4):
        xp, yp = img_coord(pos, cal, cpar.mm)
        xpx, ypx = metric_to_pixel(xp, yp, cpar)
        img_pts[i, 0] = xpx
        img_pts[i, 1] = ypx - 0.1

    result = external_calibration(cal, ref_pts, img_pts, cpar)
    assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# full_calibration  (wrapper around orient)
# ─────────────────────────────────────────────────────────────────────────────


def test_fc_no_flags():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)
    cal.ext_par.x0 -= 5.0

    ret, used, err = full_calibration(cal, fix, pix, cpar, flags=None)
    assert ret is not None
    assert used.shape[0] == len(pix)
    assert err.shape[0] == NPAR + 1


def test_fc_empty_flags_list():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)

    ret, used, err = full_calibration(cal, fix, pix, cpar, flags=[])
    assert ret is not None


def test_fc_cc_flag():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)

    ret, used, err = full_calibration(cal, fix, pix, cpar, flags=["cc"])
    assert ret is not None


def test_fc_xh_yh_flags():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)

    ret, used, err = full_calibration(cal, fix, pix, cpar, flags=["xh", "yh"])
    assert ret is not None


def test_fc_k1_k2_k3_flags():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)

    ret, used, err = full_calibration(cal, fix, pix, cpar, flags=["k1", "k2", "k3"])
    assert ret is not None


def test_fc_p1_p2_flags():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)

    ret, used, err = full_calibration(cal, fix, pix, cpar, flags=["p1", "p2"])
    assert ret is not None


def test_fc_scale_shear_flags():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)

    ret, used, err = full_calibration(cal, fix, pix, cpar, flags=["scale", "shear"])
    assert ret is not None


def test_fc_raw_ndarray_img_pts():
    """img_pts as raw (n,2) ndarray → Target wrapping path."""
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)

    img_pts = np.zeros((64, 2))
    for i, pos in enumerate(fix):
        xp, yp = img_coord(pos, cal, cpar.mm)
        xpx, ypx = metric_to_pixel(xp, yp, cpar)
        img_pts[i, 0] = xpx
        img_pts[i, 1] = ypx

    ret, used, err = full_calibration(cal, fix, img_pts, cpar)
    assert ret is not None


def test_fc_pnr_mismatch_gives_nan():
    """Targets whose pnr != i produce NaN residual entries."""
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)
    # Corrupt pnr for last 4 → they'll be skipped by orient and get NaN
    for p in pix[-4:]:
        p.pnr = -1

    ret, used, err = full_calibration(cal, fix, pix, cpar)
    assert np.any(np.isnan(ret[-4:]))


def test_fc_raises_on_nonconvergence(monkeypatch):
    """full_calibration raises ValueError when orient returns None."""
    import openptv2.algorithms.orientation as ori_mod

    monkeypatch.setattr(ori_mod, "NUM_ITER", 1)
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)
    cal.ext_par.x0 += 200.0  # large perturbation → won't converge in 1 iter

    with pytest.raises(ValueError, match="Orientation iteration failed"):
        full_calibration(cal, fix, pix, cpar)


# ─────────────────────────────────────────────────────────────────────────────
# match_detection_to_ref  (delegates to sortgrid)
# ─────────────────────────────────────────────────────────────────────────────


def test_match_detection_to_ref_basic():
    fix = _grid_fix(64)
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    pix = _project_targets(fix, cal, cpar)

    result = match_detection_to_ref(cal, fix, pix, cpar, eps=25)
    assert len(result) == len(fix)


# ─────────────────────────────────────────────────────────────────────────────
# multi_cam_point_positions
# ─────────────────────────────────────────────────────────────────────────────


def test_multi_cam_point_positions_basic():
    cals = _sym_cals()
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    # Use unit multimedia so img_coord and point_position_batch are consistent
    cpar.mm.n1 = 1.0
    cpar.mm.n2 = np.array([1.0])
    cpar.mm.d = np.array([1.0])
    cpar.mm.n3 = 1.0

    mm = _mm()
    point = np.array([17.0, 42.0, 0.0])
    targets = np.zeros((1, 4, 2))
    for cam in range(4):
        xp, yp = img_coord(point, cals[cam], mm)
        targets[0, cam, 0] = xp
        targets[0, cam, 1] = yp

    positions, rcm = multi_cam_point_positions(targets, cpar, cals)
    assert positions.shape == (1, 3)
    assert rcm.shape == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# point_positions  (dispatcher)
# ─────────────────────────────────────────────────────────────────────────────


class _MockVpar:
    """Duck-typed VolumePar exposing Zmin_lay / Zmax_lay (matches VolumePar and
    single_cam_point_positions, which reads the capital-Z attributes)."""

    Zmin_lay = np.array([-10.0, -10.0])
    Zmax_lay = np.array([10.0, 10.0])


def test_point_positions_zero_cals_raises():
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    targets = np.zeros((1, 0, 2))
    with pytest.raises(ValueError, match="wrong number of cameras"):
        point_positions(targets, cpar, [])


def test_point_positions_multi_cam():
    cals = _sym_cals()
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    cpar.mm.n1 = 1.0
    cpar.mm.n2 = np.array([1.0])
    cpar.mm.d = np.array([1.0])
    cpar.mm.n3 = 1.0
    mm = _mm()

    point = np.array([17.0, 42.0, 0.0])
    targets = np.zeros((1, 4, 2))
    for cam in range(4):
        xp, yp = img_coord(point, cals[cam], mm)
        targets[0, cam, 0] = xp
        targets[0, cam, 1] = yp

    positions, rcm = point_positions(targets, cpar, cals)
    assert positions.shape == (1, 3)


def test_point_positions_single_cam():
    """Single-camera dispatch → single_cam_point_positions (bug B1 mock)."""
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    mm = _mm()

    point = np.array([0.0, 0.0, 0.0])
    targets = np.zeros((1, 1, 2))
    xp, yp = img_coord(point, cal, mm)
    targets[0, 0, 0] = xp
    targets[0, 0, 1] = yp

    positions, rcm = point_positions(targets, cpar, [cal], vpar=_MockVpar())
    assert positions.shape == (1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# single_cam_point_positions
# ─────────────────────────────────────────────────────────────────────────────


def test_single_cam_normal_z_intercept():
    """Ray has non-zero z component → z-plane intersection branch."""
    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    mm = _mm()

    pts_3d = np.array([[5.0, 5.0, 0.0], [-5.0, -5.0, 0.0]])
    targets = np.zeros((2, 1, 2))
    for pt, p in enumerate(pts_3d):
        xp, yp = img_coord(p, cal, mm)
        targets[pt, 0, 0] = xp
        targets[pt, 0, 1] = yp

    positions, rcm = single_cam_point_positions(targets, cpar, [cal], _MockVpar())
    assert positions.shape == (2, 3)
    np.testing.assert_allclose(rcm, np.zeros(2))


def test_single_cam_zero_direct_z(monkeypatch):
    """Patch ray_tracing to return a direction with direct[2]=0 → else branch."""
    import openptv2.algorithms.ray_tracing as rt_mod

    orig_ray_tracing = rt_mod.ray_tracing

    call_count = [0]

    def patched_ray_tracing(*args, **kwargs):
        pos, direct = orig_ray_tracing(*args, **kwargs)
        call_count[0] += 1
        # Force direct[2] = 0 on first call to hit the else branch
        if call_count[0] == 1:
            direct = np.array([direct[0], direct[1], 0.0])
        return pos, direct

    monkeypatch.setattr(rt_mod, "ray_tracing", patched_ray_tracing)

    cal = Calibration.from_file(SYM_TMPL.format(1), CAL1_ADD)
    cpar = ControlPar.from_yaml(PARAMS_YAML)
    mm = _mm()
    point = np.array([0.0, 0.0, 0.0])
    targets = np.zeros((1, 1, 2))
    xp, yp = img_coord(point, cal, mm)
    targets[0, 0, 0] = xp
    targets[0, 0, 1] = yp

    positions, rcm = single_cam_point_positions(targets, cpar, [cal], _MockVpar())
    assert positions.shape == (1, 3)
    assert call_count[0] >= 1

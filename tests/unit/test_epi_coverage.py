"""Coverage-focused unit tests for openptv2.algorithms.epi.

Small, fast tests that exercise find_candidate, _quality_ratio, is_compiled,
epi_mm and epi_mm_2d via geometric invariants (no magic expected values).

Note: epipolar_curve and find_candidate both depend on trafo functions that
use the `_out: cython.double[2]` C-array pattern, which raises UnboundLocalError
when the module is interpreted as pure Python (they only work compiled).
find_candidate calls trafo.correct_brown_affin, which for zero-distortion
parameters (all k/p = 0, scx = 1, she = 0) is mathematically the identity, so
we monkeypatch it with an exact identity to exercise find_candidate's own
logic in pure-Python mode without asserting any invented values.
"""

import numpy as np
import pytest

import openptv2.algorithms.trafo as trafo
from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Exterior,
    Glass,
    Interior,
)
from openptv2.algorithms.epi import (
    MAXCAND,
    Candidate,
    Coord2d,
    _quality_ratio,
    epi_mm_2d,
    epipolar_curve,
    find_candidate,
    is_compiled,
)
from openptv2.algorithms.parameters import ControlPar, MmNp, VolumePar
from openptv2.algorithms.tracking_frame_buf import Target

EPS = 1e-5


# --------------------------------------------------------------------------
# Fixtures / builders
# --------------------------------------------------------------------------
def make_cal(x0, y0, z0, gx, gy, gz, cc=100.0):
    ext = Exterior(x0=x0, y0=y0, z0=z0)
    ext.dm = np.eye(3, dtype=np.float64)
    return Calibration(
        ext_par=ext,
        int_par=Interior(xh=0.0, yh=0.0, cc=cc),
        glass_par=Glass(vec_x=gx, vec_y=gy, vec_z=gz),
        added_par=AddedPar(),
    )


def make_mm(n1, n2_0, n3, d0):
    return MmNp(
        nlay=1, n1=n1, n2=np.array([n2_0, 1.0, 1.0]), d=np.array([d0, 0.0, 0.0]), n3=n3
    )


def make_vpar(X_lay, Zmin_lay, Zmax_lay, **kw):
    return VolumePar(X_lay=X_lay, Zmin_lay=Zmin_lay, Zmax_lay=Zmax_lay, **kw)


@pytest.fixture
def identity_brown(monkeypatch):
    """Force trafo.correct_brown_affin to an exact identity.

    Correct for zero-distortion calibration; lets find_candidate run in
    pure-Python mode (and is a no-op behaviourally when compiled).
    """

    def _identity(x, y, *args, **kwargs):
        return (x, y)

    monkeypatch.setattr(trafo, "correct_brown_affin", _identity)
    return _identity


def _fc_cpar():
    # sensor half-size = pix_x * imx / 2 = 0.1 * 1000 / 2 = 50 mm, xh = yh = 0
    return ControlPar(imx=1000, imy=1000, pix_x=0.1, pix_y=0.1)


def _fc_vpar():
    return make_vpar(
        (-50.0, 50.0),
        (-50.0, -50.0),
        (50.0, 50.0),
        cn=0.5,
        cnx=0.5,
        cny=0.5,
        csumg=0.5,
        eps0=1.0,
    )


def _fc_cal():
    return make_cal(0.0, 0.0, 100.0, 0.0, 0.0, 50.0)


def _cand_arrays():
    return (
        np.zeros(MAXCAND + 1, dtype=np.int32),
        np.zeros(MAXCAND + 1, dtype=np.float64),
        np.zeros(MAXCAND + 1, dtype=np.float64),
    )


# --------------------------------------------------------------------------
# Trivial helpers / dataclasses
# --------------------------------------------------------------------------
def test_is_compiled_returns_bool():
    assert isinstance(is_compiled(), bool)


def test_dataclasses_construct():
    c = Candidate(pnr=3, tol=0.5, corr=1.5)
    assert (c.pnr, c.tol, c.corr) == (3, 0.5, 1.5)
    p = Coord2d(pnr=2, x=1.0, y=-1.0)
    assert (p.pnr, p.x, p.y) == (2, 1.0, -1.0)


def test_quality_ratio_both_branches():
    # a < b branch and a >= b branch both yield min/max
    assert _quality_ratio(2.0, 4.0) == pytest.approx(0.5)
    assert _quality_ratio(4.0, 2.0) == pytest.approx(0.5)
    assert _quality_ratio(3.0, 3.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# epi_mm_2d / epi_mm — reference values reused from tests/unit/test_epi.py
# --------------------------------------------------------------------------
def test_epi_mm_2d_reference():
    cal = make_cal(0.0, 0.0, 100.0, 0.0, 0.0, 50.0)
    mmp = make_mm(1.0, 1.49, 1.33, 5.0)
    vpar = make_vpar((-250.0, 250.0), (-100.0, -100.0), (100.0, 100.0))

    out = epi_mm_2d(1.0, 10.0, cal, mmp, vpar)
    assert np.abs(out[0] - 0.85858163) < EPS
    assert np.abs(out[1] - 8.58581626) < EPS
    assert np.abs(out[2] - 0.0) < EPS

    out = epi_mm_2d(0.0, 0.0, cal, mmp, vpar)
    assert np.allclose(out, [0.0, 0.0, 0.0], atol=EPS)


# epi_mm is covered by tests/unit/test_epi.py::test_epi_mm (identical case);
# not duplicated here.


# --------------------------------------------------------------------------
# find_candidate
# --------------------------------------------------------------------------
def _line_setup(n_pts=12):
    """Points on the x-axis inside the sensor, x-sorted, plus one far right."""
    xs = [-22.0, -18.0, -14.0, -10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0, 100.0]
    xs = xs[:n_pts]
    crd = [Coord2d(pnr=i, x=x, y=0.0) for i, x in enumerate(xs)]
    pix = [Target(pnr=i, x=x, y=0.0, n=10, nx=5, ny=5, sumg=100) for i, x in enumerate(xs)]
    return crd, pix, len(xs)


def test_find_candidate_finds_points_on_line(identity_brown):
    crd, pix, num = _line_setup()
    cp, ct, cc = _cand_arrays()
    # horizontal epipolar line y = 0 across the sensor
    count = find_candidate(
        crd, pix, num, -25.0, 0.0, 25.0, 0.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    # 11 points lie exactly on the line and within bounds; the 12th (x=100)
    # is beyond xb + tol and terminates the scan.
    assert count == 11
    # exact-on-line points have zero perpendicular distance
    assert np.allclose(ct[:count], 0.0, atol=1e-12)
    # candidate indices are the first 11 crd indices, in order
    assert list(cp[:count]) == list(range(11))
    # identical target quality -> positive, equal correlation scores
    assert np.all(cc[:count] > 0.0)
    assert len(set(np.round(cc[:count], 9))) == 1


def test_find_candidate_order_independent(identity_brown):
    crd, pix, num = _line_setup()
    args = (crd, pix, num)
    kw = dict()  # noqa: C408
    fwd_arrays = _cand_arrays()
    rev_arrays = _cand_arrays()
    fwd = find_candidate(
        *args, -25.0, 0.0, 25.0, 0.0, 10, 5, 5, 100, *fwd_arrays,
        _fc_vpar(), _fc_cpar(), _fc_cal(), **kw,
    )
    # reversed x endpoints -> triggers the xa > xb swap
    rev = find_candidate(
        *args, 25.0, 0.0, -25.0, 0.0, 10, 5, 5, 100, *rev_arrays,
        _fc_vpar(), _fc_cpar(), _fc_cal(), **kw,
    )
    assert fwd == rev


def test_find_candidate_ya_gt_yb_swap(identity_brown):
    crd, pix, num = _line_setup()
    cp, ct, cc = _cand_arrays()
    # ya > yb triggers the y-swap branch; tilted line still valid geometry
    count = find_candidate(
        crd, pix, num, -25.0, 5.0, 25.0, -5.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    assert 0 <= count <= num


def test_find_candidate_vertical_line(identity_brown):
    crd, pix, num = _line_setup()
    cp, ct, cc = _cand_arrays()
    # xa == xb hits the degenerate-slope guard (xb += 1e-10)
    count = find_candidate(
        crd, pix, num, 0.0, -25.0, 0.0, 25.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    assert 0 <= count <= num


def test_find_candidate_line_outside_sensor(identity_brown):
    crd, pix, num = _line_setup()
    cp, ct, cc = _cand_arrays()
    # entire line left of the sensor -> -1
    count = find_candidate(
        crd, pix, num, -100.0, 0.0, -60.0, 0.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    assert count == -1


def test_find_candidate_pnr_out_of_range(identity_brown):
    # single on-line point whose pnr >= num -> early -1
    crd = [Coord2d(pnr=1, x=0.0, y=0.0)]
    pix = [Target(pnr=0, x=0.0, y=0.0, n=10, nx=5, ny=5, sumg=100)]
    cp, ct, cc = _cand_arrays()
    count = find_candidate(
        crd, pix, 1, -5.0, 0.0, 5.0, 0.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    assert count == -1


def test_find_candidate_quality_rejected(identity_brown):
    # on-line point but quality ratio far below threshold -> skipped
    crd = [Coord2d(pnr=0, x=0.0, y=0.0)]
    pix = [Target(pnr=0, x=0.0, y=0.0, n=10000, nx=5, ny=5, sumg=100)]
    cp, ct, cc = _cand_arrays()
    count = find_candidate(
        crd, pix, 1, -5.0, 0.0, 5.0, 0.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    assert count == 0


def test_find_candidate_off_band_skipped(identity_brown):
    # point far off the tolerance band in y -> continue, no candidates
    crd = [Coord2d(pnr=0, x=0.0, y=40.0)]
    pix = [Target(pnr=0, x=0.0, y=40.0, n=10, nx=5, ny=5, sumg=100)]
    cp, ct, cc = _cand_arrays()
    count = find_candidate(
        crd, pix, 1, -5.0, 0.0, 5.0, 0.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    assert count == 0


def test_find_candidate_binary_search_forward_branch(identity_brown):
    # Line shifted right so crd[j0].x < xa - tol on the first probe,
    # exercising the j0 += dj branch of the binary search.
    crd, pix, num = _line_setup()
    cp, ct, cc = _cand_arrays()
    count = find_candidate(
        crd, pix, num, 15.0, 0.0, 45.0, 0.0, 10, 5, 5, 100, cp, ct, cc,
        _fc_vpar(), _fc_cpar(), _fc_cal(),
    )
    assert 0 <= count <= num


# --------------------------------------------------------------------------
# epipolar_curve
#
# pixel_to_metric / metric_to_pixel / dist_to_flat use the trafo
# `_out: cython.double[2]` pattern that only works when compiled, so we patch
# them with pass-throughs to let epipolar_curve execute in pure-Python mode.
# We assert only the output SHAPE, which is independent of the patched values.
# --------------------------------------------------------------------------
@pytest.fixture
def passthrough_trafo(monkeypatch):
    monkeypatch.setattr(trafo, "pixel_to_metric", lambda x, y, cpar: (x, y))
    monkeypatch.setattr(trafo, "metric_to_pixel", lambda x, y, cpar: (x, y))
    monkeypatch.setattr(
        trafo, "dist_to_flat", lambda xp, yp, *a, **k: (xp, yp)
    )


def test_epipolar_curve_shape(passthrough_trafo):
    orig_cal = make_cal(10.0, 0.0, 100.0, 0.0, 0.0, 50.0)
    proj_cal = make_cal(-10.0, 0.0, 100.0, 0.0, 0.0, 50.0)
    cpar = ControlPar(imx=1280, imy=1024, pix_x=0.017, pix_y=0.017)
    cpar.mm = make_mm(1.0, 1.0, 1.0, 1.0)
    vpar = make_vpar((-250.0, 250.0), (-10.0, -10.0), (10.0, 10.0))

    num = 7
    line = epipolar_curve(
        np.array([cpar.imx / 2.0, cpar.imy / 2.0]),
        orig_cal, proj_cal, num, cpar, vpar,
    )
    assert line.shape == (num, 2)

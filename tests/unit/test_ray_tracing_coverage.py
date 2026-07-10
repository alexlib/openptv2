"""Pure-Python line coverage tests for src/openptv2/algorithms/ray_tracing.py.

Run via:
    COVERAGE_FILE=/tmp/.cov_ray uv run pytest tests/unit/test_ray_tracing_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q
"""

import math

import numpy as np
import pytest

from openptv2.algorithms.ray_tracing import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.ray_tracing import (
    _ray_tracing_core,
    ray_tracing,
    ray_tracing_batch,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_cal(x0=0.0, y0=0.0, z0=-100.0, glass_vec=(0.0, 0.0, 1.0)):
    """Return a minimal stub calibration object."""

    class _Ext:
        dm = np.eye(3, dtype=np.float64)

    class _Int:
        cc = 10.0

    class _Glass:
        pass

    class _Cal:
        ext_par = _Ext()
        int_par = _Int()
        glass_par = _Glass()

    cal = _Cal()
    cal.ext_par.x0 = x0
    cal.ext_par.y0 = y0
    cal.ext_par.z0 = z0
    cal.glass_par.vec_x = glass_vec[0]
    cal.glass_par.vec_y = glass_vec[1]
    cal.glass_par.vec_z = glass_vec[2]
    return cal


def _make_mm(n1=1.0, n2=None, n3=1.33, d=None):
    """Return a minimal multimedia parameters stub."""
    if n2 is None:
        n2 = [1.5]
    if d is None:
        d = [2.0]

    class _Mm:
        pass

    mm = _Mm()
    mm.n1 = n1
    mm.n2 = np.array(n2, dtype=np.float64)
    mm.n3 = n3
    mm.d = np.array(d, dtype=np.float64)
    return mm


GLASS_VEC = (0.0, 0.0, 1.0)  # standard vertical glass
CC = 10.0  # focal length
MM_N1 = 1.0
MM_N2 = 1.5
MM_N3 = 1.33
MM_D = 2.0
EXT_X0, EXT_Y0, EXT_Z0 = 0.0, 0.0, -100.0


def _standard_kwargs():
    """Return scalar kwargs for ray_tracing."""
    dm = np.eye(3, dtype=np.float64)
    return dict(
        ext_dm=dm,
        ext_x0=EXT_X0,
        ext_y0=EXT_Y0,
        ext_z0=EXT_Z0,
        int_cc=CC,
        glass_vec_x=GLASS_VEC[0],
        glass_vec_y=GLASS_VEC[1],
        glass_vec_z=GLASS_VEC[2],
        mm_n1=MM_N1,
        mm_n2_0=MM_N2,
        mm_n3=MM_N3,
        mm_d0=MM_D,
    )


# ---------------------------------------------------------------------------
# is_compiled
# ---------------------------------------------------------------------------


def test_is_compiled_false():
    assert _is_compiled() is False


# ---------------------------------------------------------------------------
# ray_tracing — off-axis (norm_bp > 0 branch exercised)
# ---------------------------------------------------------------------------


def test_ray_tracing_off_axis_returns_tuple():
    pos, direction = ray_tracing(x=1.0, y=0.0, **_standard_kwargs())
    assert isinstance(pos, np.ndarray)
    assert isinstance(direction, np.ndarray)
    assert pos.shape == (3,)
    assert direction.shape == (3,)


def test_ray_tracing_off_axis_finite():
    pos, direction = ray_tracing(x=1.0, y=0.5, **_standard_kwargs())
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(direction))


def test_ray_tracing_off_axis_direction_unit_length():
    pos, direction = ray_tracing(x=1.0, y=0.5, **_standard_kwargs())
    length = np.linalg.norm(direction)
    assert abs(length - 1.0) < 1e-10


def test_ray_tracing_negative_xy():
    pos, direction = ray_tracing(x=-2.0, y=-1.0, **_standard_kwargs())
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(direction))


def test_ray_tracing_large_xy():
    pos, direction = ray_tracing(x=5.0, y=5.0, **_standard_kwargs())
    assert np.all(np.isfinite(pos))


# ---------------------------------------------------------------------------
# ray_tracing — on-axis (norm_bp == 0 branch; ray along glass normal)
# ---------------------------------------------------------------------------

def test_ray_tracing_on_axis_norm_bp_zero_branch():
    """x=0, y=0 with identity dm and glass_dir=[0,0,1] → norm_bp=0 both times."""
    pos, direction = ray_tracing(x=0.0, y=0.0, **_standard_kwargs())
    assert pos.shape == (3,)
    assert direction.shape == (3,)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(direction))


def test_ray_tracing_on_axis_direction_unit():
    pos, direction = ray_tracing(x=0.0, y=0.0, **_standard_kwargs())
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# _ray_tracing_core — direct call coverage
# ---------------------------------------------------------------------------


def test_ray_tracing_core_direct_off_axis():
    dm = np.eye(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    _ray_tracing_core(
        1.0, 0.5,
        dm, EXT_X0, EXT_Y0, EXT_Z0, CC,
        0.0, 0.0, 1.0,
        MM_N1, MM_N2, MM_N3, MM_D,
        pos, out,
    )
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(out))


def test_ray_tracing_core_direct_on_axis():
    dm = np.eye(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    _ray_tracing_core(
        0.0, 0.0,
        dm, EXT_X0, EXT_Y0, EXT_Z0, CC,
        0.0, 0.0, 1.0,
        MM_N1, MM_N2, MM_N3, MM_D,
        pos, out,
    )
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(out))


def test_ray_tracing_core_modifies_pos_and_out():
    dm = np.eye(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    _ray_tracing_core(
        1.0, 0.0,
        dm, 0.0, 0.0, -50.0, 8.0,
        0.0, 0.0, 1.0,
        1.0, 1.5, 1.33, 3.0,
        pos, out,
    )
    # pos should not still be zeros
    assert not np.allclose(pos, 0.0)


# ---------------------------------------------------------------------------
# ray_tracing symmetry checks
# ---------------------------------------------------------------------------


def test_ray_tracing_x_symmetry():
    """ray_tracing(x, y) and ray_tracing(-x, y) should be mirror images in X."""
    kw = _standard_kwargs()
    pos_p, dir_p = ray_tracing(x=2.0, y=0.0, **kw)
    pos_n, dir_n = ray_tracing(x=-2.0, y=0.0, **kw)
    assert abs(pos_p[0] + pos_n[0]) < 1e-10  # symmetric in X
    assert abs(pos_p[1] - pos_n[1]) < 1e-10  # same Y
    assert abs(pos_p[2] - pos_n[2]) < 1e-10  # same Z


def test_ray_tracing_y_symmetry():
    kw = _standard_kwargs()
    pos_p, dir_p = ray_tracing(x=0.0, y=1.5, **kw)
    pos_n, dir_n = ray_tracing(x=0.0, y=-1.5, **kw)
    assert abs(pos_p[1] + pos_n[1]) < 1e-10
    assert abs(pos_p[0] - pos_n[0]) < 1e-10


# ---------------------------------------------------------------------------
# ray_tracing — non-trivial glass normals
# ---------------------------------------------------------------------------


def test_ray_tracing_tilted_glass():
    """Glass tilted at 45° in X-Z plane."""
    kw = _standard_kwargs()
    kw["glass_vec_x"] = 1.0
    kw["glass_vec_y"] = 0.0
    kw["glass_vec_z"] = 1.0
    pos, direction = ray_tracing(x=1.0, y=0.0, **kw)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(direction))


def test_ray_tracing_tilted_glass_off_center():
    kw = _standard_kwargs()
    kw["glass_vec_x"] = 0.5
    kw["glass_vec_y"] = 0.5
    kw["glass_vec_z"] = 1.0
    pos, direction = ray_tracing(x=0.5, y=0.5, **kw)
    assert np.all(np.isfinite(pos))


# ---------------------------------------------------------------------------
# ray_tracing — varied refractive indices / thickness
# ---------------------------------------------------------------------------


def test_ray_tracing_thin_glass():
    kw = _standard_kwargs()
    kw["mm_d0"] = 0.1
    pos, direction = ray_tracing(x=1.0, y=0.0, **kw)
    assert np.all(np.isfinite(pos))


def test_ray_tracing_thick_glass():
    kw = _standard_kwargs()
    kw["mm_d0"] = 20.0
    pos, direction = ray_tracing(x=1.0, y=0.0, **kw)
    assert np.all(np.isfinite(pos))


def test_ray_tracing_matched_indices():
    """n1 == n2 == n3 → no refraction, straight ray."""
    kw = _standard_kwargs()
    kw["mm_n1"] = kw["mm_n2_0"] = kw["mm_n3"] = 1.33
    pos, direction = ray_tracing(x=1.0, y=0.0, **kw)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(direction))


# ---------------------------------------------------------------------------
# ray_tracing_batch — basic functionality
# ---------------------------------------------------------------------------


def test_ray_tracing_batch_single_ray_matches_ray_tracing():
    cal = _make_cal()
    mm = _make_mm()
    xy = np.array([[1.0, 0.5]], dtype=np.float64)
    positions, directions = ray_tracing_batch(xy, cal, mm)

    kw = _standard_kwargs()
    pos_s, dir_s = ray_tracing(x=1.0, y=0.5, **kw)

    np.testing.assert_allclose(positions[0], pos_s, atol=1e-12)
    np.testing.assert_allclose(directions[0], dir_s, atol=1e-12)


def test_ray_tracing_batch_returns_correct_shapes():
    cal = _make_cal()
    mm = _make_mm()
    N = 5
    xy = np.random.rand(N, 2).astype(np.float64) * 3.0 - 1.5
    positions, directions = ray_tracing_batch(xy, cal, mm)
    assert positions.shape == (N, 3)
    assert directions.shape == (N, 3)


def test_ray_tracing_batch_all_finite():
    cal = _make_cal()
    mm = _make_mm()
    xy = np.array([[1.0, 0.0], [0.5, 0.5], [-1.0, 0.3], [0.0, -0.8]], dtype=np.float64)
    positions, directions = ray_tracing_batch(xy, cal, mm)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(directions))


def test_ray_tracing_batch_multiple_directions_unit():
    cal = _make_cal()
    mm = _make_mm()
    xy = np.array([[1.0, 0.5], [2.0, -1.0], [-0.5, 0.3]], dtype=np.float64)
    _, directions = ray_tracing_batch(xy, cal, mm)
    norms = np.linalg.norm(directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_ray_tracing_batch_on_axis_norm_bp_zero():
    """On-axis ray (0,0) exercises norm_bp==0 in batch loop."""
    cal = _make_cal()
    mm = _make_mm()
    xy = np.array([[0.0, 0.0]], dtype=np.float64)
    positions, directions = ray_tracing_batch(xy, cal, mm)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(directions))
    assert abs(np.linalg.norm(directions[0]) - 1.0) < 1e-10


def test_ray_tracing_batch_mixed_on_off_axis():
    """Mix of on-axis and off-axis rays in one batch."""
    cal = _make_cal()
    mm = _make_mm()
    xy = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]], dtype=np.float64)
    positions, directions = ray_tracing_batch(xy, cal, mm)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(directions))


def test_ray_tracing_batch_results_dtype_float64():
    cal = _make_cal()
    mm = _make_mm()
    xy = np.array([[1.0, 0.0]], dtype=np.float64)
    positions, directions = ray_tracing_batch(xy, cal, mm)
    assert positions.dtype == np.float64
    assert directions.dtype == np.float64


def test_ray_tracing_batch_large_batch():
    cal = _make_cal()
    mm = _make_mm()
    rng = np.random.default_rng(42)
    xy = rng.uniform(-3.0, 3.0, (50, 2)).astype(np.float64)
    positions, directions = ray_tracing_batch(xy, cal, mm)
    assert positions.shape == (50, 3)
    norms = np.linalg.norm(directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# ray_tracing_batch — consistency with individual ray_tracing calls
# ---------------------------------------------------------------------------


def test_ray_tracing_batch_matches_individual_calls():
    cal = _make_cal()
    mm = _make_mm()
    xys = [(1.0, 0.5), (-0.5, 1.2), (2.0, -0.3)]
    xy_arr = np.array(xys, dtype=np.float64)

    positions, directions = ray_tracing_batch(xy_arr, cal, mm)

    dm = np.eye(3, dtype=np.float64)
    for i, (x, y) in enumerate(xys):
        pos_s, dir_s = ray_tracing(
            x=x, y=y,
            ext_dm=dm, ext_x0=cal.ext_par.x0, ext_y0=cal.ext_par.y0,
            ext_z0=cal.ext_par.z0, int_cc=cal.int_par.cc,
            glass_vec_x=cal.glass_par.vec_x, glass_vec_y=cal.glass_par.vec_y,
            glass_vec_z=cal.glass_par.vec_z,
            mm_n1=mm.n1, mm_n2_0=mm.n2[0], mm_n3=mm.n3, mm_d0=mm.d[0],
        )
        np.testing.assert_allclose(positions[i], pos_s, atol=1e-12)
        np.testing.assert_allclose(directions[i], dir_s, atol=1e-12)


# ---------------------------------------------------------------------------
# ray_tracing_batch — tilted glass normal
# ---------------------------------------------------------------------------


def test_ray_tracing_batch_tilted_glass():
    cal = _make_cal(glass_vec=(1.0, 0.0, 1.0))
    mm = _make_mm()
    xy = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    positions, directions = ray_tracing_batch(xy, cal, mm)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(directions))


# ---------------------------------------------------------------------------
# ray_tracing — rotated camera (non-identity dm)
# ---------------------------------------------------------------------------


def test_ray_tracing_rotated_camera():
    """Use a small rotation to exercise non-trivial dm path."""
    angle = math.radians(5.0)
    dm = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle),  math.cos(angle), 0.0],
        [0.0,              0.0,             1.0],
    ], dtype=np.float64)
    kw = _standard_kwargs()
    kw["ext_dm"] = dm
    pos, direction = ray_tracing(x=1.0, y=0.0, **kw)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(direction))

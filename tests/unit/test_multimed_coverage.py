"""Coverage tests for openptv2.algorithms.multimed.

Target: >= 90% pure-Python line coverage of multimed.py.
All tests use analytic/invariant assertions — no guessed expected values.
"""
from unittest.mock import patch

import numpy as np
import pytest

from openptv2.algorithms.multimed import (
    _multimed_nlay_core,
    _multimed_r_1lay_iterative,
    back_trans_point,
    get_mmf_from_mmlut,
    init_mmlut,
    is_compiled,
    move_along_ray,
    multimed_nlay,
    multimed_r_nlay_iterative,
    trans_cam_point,
    volumedimension,
)

# ── trafo monkeypatch helpers ─────────────────────────────────────────────────
# pixel_to_metric and correct_brown_affin have _out: cython.double[2] bug in
# pure-Python mode; patch them wherever multimed functions do a local import.

def _mock_p2m(x, y, *args, **kwargs):
    return float(x) * 0.01 - 6.4, 5.12 - float(y) * 0.01


def _mock_cba(x, y, *args, **kwargs):
    return x, y


# ── _multimed_r_1lay_iterative ────────────────────────────────────────────────

def test_1lay_trivial_all_n_one():
    """Early return when n1 == n2 == n3 == 1.0."""
    result = _multimed_r_1lay_iterative(5.0, 0.0, 100.0, 1.0, 1.0, 1.0, 5.0)
    assert result == 1.0


def test_1lay_pos_x_zero_returns_one():
    """r == 0: final else returns 1.0 (no division by zero)."""
    result = _multimed_r_1lay_iterative(0.0, 0.0, 100.0, 1.0, 1.49, 1.33, 5.0)
    assert result == 1.0


def test_1lay_normal_convergence():
    """Standard water/glass geometry converges to a shift near 1."""
    result = _multimed_r_1lay_iterative(10.0, 0.0, 100.0, 1.0, 1.49, 1.33, 5.0)
    assert isinstance(result, float)
    assert 0.8 < result < 1.2


def test_1lay_arg_clamp_and_nonconvergence():
    """Extreme geometry: arg > 1 clamped (it. 1), arg < -1 clamped (it. 2+),
    loop never converges → for-else fires, returns 1.0 fallback."""
    # Large pos_x, small ext_z0-pos_z, mm_n1 >> mm_n2 → tan(pi/2) → diverge
    result = _multimed_r_1lay_iterative(100.0, 0.0, 1.0, 3.0, 1.0, 1.0, 0.1)
    # Either converged or hit the fallback; must be a float
    assert isinstance(result, float)


# ── _multimed_nlay_core ───────────────────────────────────────────────────────

def test_nlay_core_preset_mmf():
    """mmf > 0 and != 1.0 → radial_shift = mmf, Xq = pos_x * mmf."""
    mmf = 1.5
    Xq, Yq = _multimed_nlay_core(10.0, 0.0, 100.0, 1.0, 1.49, 1.33, 5.0, mmf)
    assert abs(Xq - 10.0 * mmf) < 1e-12
    assert Yq == 0.0


def test_nlay_core_mmf_one_calls_iterative():
    """mmf == 1.0 → calls iterative; trivial n=1 gives shift=1, Xq=pos_x."""
    Xq, Yq = _multimed_nlay_core(10.0, 0.0, 100.0, 1.0, 1.0, 1.0, 5.0, 1.0)
    assert abs(Xq - 10.0) < 1e-12
    assert Yq == 0.0


def test_nlay_core_mmf_zero_calls_iterative():
    """mmf == 0 → mmf > 0 is False → iterative path."""
    Xq, Yq = _multimed_nlay_core(8.0, 0.0, 100.0, 1.0, 1.0, 1.0, 5.0, 0.0)
    assert abs(Xq - 8.0) < 1e-12


# ── multimed_nlay ─────────────────────────────────────────────────────────────

def test_multimed_nlay_with_mmf():
    """mmf != 1.0 → Xq = ext_x0 + (pos_x - ext_x0) * mmf."""
    Xq, Yq = multimed_nlay(
        5.0, 3.0, 0.0,
        0.0, 0.0, 100.0,
        1.0, 1.49, 1.33, 5.0,
        mmf=2.0,
    )
    # ext = (0,0): Xq = 0 + (5-0)*2 = 10; Yq = 0 + (3-0)*2 = 6
    assert abs(Xq - 10.0) < 1e-12
    assert abs(Yq - 6.0) < 1e-12


def test_multimed_nlay_iterative_trivial():
    """mmf == 1.0, all n=1 → shift=1, Xq=pos_x, Yq=pos_y."""
    Xq, Yq = multimed_nlay(
        5.0, 3.0, 0.0,
        0.0, 0.0, 100.0,
        1.0, 1.0, 1.0, 5.0,
        mmf=1.0,
    )
    assert abs(Xq - 5.0) < 1e-12
    assert abs(Yq - 3.0) < 1e-12


def test_multimed_nlay_mmf_zero_iterative():
    """mmf == 0 triggers the iterative path."""
    Xq, Yq = multimed_nlay(
        4.0, 2.0, 0.0,
        0.0, 0.0, 100.0,
        1.0, 1.0, 1.0, 5.0,
        mmf=0.0,
    )
    assert abs(Xq - 4.0) < 1e-12
    assert abs(Yq - 2.0) < 1e-12


# ── multimed_r_nlay_iterative ─────────────────────────────────────────────────

def test_r_nlay_trivial_all_n_one():
    """All n=1, single layer → immediate return 1.0."""
    result = multimed_r_nlay_iterative(
        10.0, 5.0, 0.0, 0.0, 0.0, 100.0, 1.0, 1.0, 1.0, 5.0
    )
    assert result == 1.0


def test_r_nlay_r_zero_returns_one():
    """Camera at same XY as particle → r=0 → returns 1.0 (no div by zero)."""
    result = multimed_r_nlay_iterative(
        0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0, 1.49, 1.33, 5.0
    )
    assert result == 1.0


def test_r_nlay_default_lists_converges():
    """mm_n2=None and mm_d=None → both default from mm_n2_0/mm_d0; converges."""
    result = multimed_r_nlay_iterative(
        10.0, 5.0, 0.0, 0.0, 0.0, 100.0, 1.0, 1.49, 1.33, 5.0,
        mm_n2=None, mm_d=None,
    )
    assert isinstance(result, float)
    assert 0.8 < result < 1.2


def test_r_nlay_two_layers_zout_loop():
    """mm_nlay=2 → zout accumulation loop runs; returns valid shift."""
    result = multimed_r_nlay_iterative(
        10.0, 5.0, 0.0, 0.0, 0.0, 100.0,
        1.0, 1.49, 1.33, 5.0,
        mm_nlay=2,
        mm_n2=[1.49, 1.37],
        mm_d=[5.0, 2.0],
    )
    assert isinstance(result, float)
    assert result > 0.0


def test_r_nlay_arg_clamp_diverges():
    """mm_n1 >> mm_n3 → arg > 1 clamped (it.1); oscillation → arg < -1 clamped
    (it.2+); 40 iterations exhaust → for-else or fallback fires."""
    result = multimed_r_nlay_iterative(
        50.0, 0.0, 0.0, 0.0, 0.0, 100.0,
        3.0, 3.0, 1.0, 5.0,
    )
    assert isinstance(result, float)


# ── trans_cam_point ───────────────────────────────────────────────────────────

def test_trans_cam_point_shapes():
    """Output arrays have correct shapes."""
    pos = np.array([10.0, 20.0, 0.0])
    pos_t, cross_p, cross_c, ext_t_z0 = trans_cam_point(
        pos, 0.0, 0.0, 100.0, 0.0, 0.0, 50.0, 1.0, 1.49, 1.33, 5.0
    )
    assert pos_t.shape == (3,)
    assert cross_p.shape == (3,)
    assert cross_c.shape == (3,)
    assert isinstance(ext_t_z0, float)


def test_trans_cam_point_z_glass_cross_p_xy():
    """Glass vector along Z: cross_p XY == pos XY; pos_t[1] == 0."""
    pos = np.array([10.0, 20.0, 0.0])
    pos_t, cross_p, cross_c, _ = trans_cam_point(
        pos, 0.0, 0.0, 100.0, 0.0, 0.0, 50.0, 1.0, 1.49, 1.33, 5.0
    )
    assert abs(cross_p[0] - pos[0]) < 1e-10
    assert abs(cross_p[1] - pos[1]) < 1e-10
    # Y component is always 0 in local frame
    assert pos_t[1] == 0.0


# ── back_trans_point ──────────────────────────────────────────────────────────

def test_back_trans_point_roundtrip():
    """trans_cam_point → back_trans_point recovers original position."""
    pos = np.array([10.0, 20.0, 0.0])
    gv = [0.0001, 0.00001, 1.0]
    pos_t, cross_p, cross_c, _ = trans_cam_point(
        pos, 0.0, 0.0, 100.0, gv[0], gv[1], gv[2], 1.0, 1.49, 1.33, 5.0
    )
    pos_back = back_trans_point(
        pos_t, cross_p, cross_c, gv[0], gv[1], gv[2], 1.0, 1.49, 1.33, 5.0
    )
    assert abs(pos_back[0] - pos[0]) < 1e-6
    assert abs(pos_back[1] - pos[1]) < 1e-6
    assert abs(pos_back[2] - pos[2]) < 1e-6


def test_back_trans_point_nve_zero():
    """n_ve == 0 branch: cross_p == 0 → tmp = 0 → skip s_x correction."""
    # glass_vec = [0,0,1], cross_c = [0,0,mm_d0] → ag = [0,0,0]
    # cross_p = [0,0,0] → tmp = dot(cross_p - ag, ngl) = 0 → n_ve = 0
    pos_t = np.array([0.0, 0.0, -50.0])
    cross_p = np.array([0.0, 0.0, 0.0])
    cross_c = np.array([0.0, 0.0, 5.0])
    result = back_trans_point(
        pos_t, cross_p, cross_c, 0.0, 0.0, 1.0, 1.0, 1.49, 1.33, 5.0
    )
    assert result.shape == (3,)
    # s_z = -pos_t[2]/1 = 50; px=0-0*50=0, pz=0-1*50=-50
    assert abs(result[0]) < 1e-12
    assert abs(result[1]) < 1e-12
    assert abs(result[2] - (-50.0)) < 1e-12


def test_back_trans_point_nve_nonzero():
    """n_ve > 0 branch: pos_t[0] > 0 → s_x correction applied."""
    pos_t = np.array([5.0, 0.0, -10.0])
    cross_p = np.array([3.0, 4.0, 0.0])
    cross_c = np.array([0.0, 0.0, 5.0])
    result = back_trans_point(
        pos_t, cross_p, cross_c, 0.0, 0.0, 1.0, 1.0, 1.49, 1.33, 5.0
    )
    assert result.shape == (3,)


# ── move_along_ray ────────────────────────────────────────────────────────────

def test_move_along_ray_45deg():
    """45-degree ray from origin: result[0] = glob_Z, result[2] = glob_Z."""
    vertex = np.array([0.0, 0.0, 0.0])
    direct = np.array([1.0, 0.0, 1.0])
    result = move_along_ray(10.0, vertex, direct)
    assert abs(result[0] - 10.0) < 1e-12
    assert abs(result[1]) < 1e-12
    assert abs(result[2] - 10.0) < 1e-12


def test_move_along_ray_offset_vertex():
    """Non-zero vertex; analytic check."""
    vertex = np.array([1.0, 2.0, 5.0])
    direct = np.array([0.0, 1.0, 2.0])
    result = move_along_ray(9.0, vertex, direct)
    # dZ = 9-5=4; x = 1+4*0/2=1; y = 2+4*1/2=4; z = 9
    assert abs(result[0] - 1.0) < 1e-12
    assert abs(result[1] - 4.0) < 1e-12
    assert abs(result[2] - 9.0) < 1e-12


# ── get_mmf_from_mmlut ────────────────────────────────────────────────────────

def _flat_lut(nr=5, nz=10, rw=2.0, val=1.0, origin=(0.0, 0.0, -20.0)):
    """Uniform LUT — all data cells equal val."""
    org = np.array(origin, dtype=np.float64)
    data = np.full(nr * nz, val, dtype=np.float64)
    return org, nr, nz, rw, data


def test_get_mmf_inside_lut_uniform():
    """Point at LUT origin → bilinear corner → mmf == val."""
    org, nr, nz, rw, data = _flat_lut(val=0.5)
    pos = np.array([0.0, 0.0, -20.0], dtype=np.float64)
    mmf = get_mmf_from_mmlut(pos, org, nr, nz, rw, data)
    assert abs(mmf - 0.5) < 1e-12


def test_get_mmf_uniform_mid_point():
    """Mid-point in uniform LUT → bilinear interpolation == val."""
    org, nr, nz, rw, data = _flat_lut(nr=8, nz=8, rw=2.0, val=0.75)
    pos = np.array([1.5, 0.5, -17.0], dtype=np.float64)
    mmf = get_mmf_from_mmlut(pos, org, nr, nz, rw, data)
    assert abs(mmf - 0.75) < 1e-12


def test_get_mmf_ir_exceeds_nr_returns_zero():
    """R >> LUT radius → ir > nr → returns 0.0."""
    org, nr, nz, rw, data = _flat_lut(nr=2, nz=5)
    pos = np.array([200.0, 0.0, -20.0], dtype=np.float64)
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


def test_get_mmf_iz_negative_returns_zero():
    """Z below origin → tz < 0 → iz < 0 → returns 0.0."""
    org, nr, nz, rw, data = _flat_lut()
    pos = np.array([0.0, 0.0, -30.0], dtype=np.float64)  # tz = -10
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


def test_get_mmf_iz_exceeds_nz_returns_zero():
    """Z above LUT top → iz >= nz → returns 0.0."""
    org, nr, nz, rw, data = _flat_lut(nr=5, nz=4, rw=2.0)
    # LUT top: origin_z + nz*rw = -20+8 = -12; pos_z=0 → tz=20 > 8
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


def test_get_mmf_v4_0_exceeds_max():
    """ir == nr, iz > 0 → v4_0 index out of range → returns 0.0."""
    nr, nz, rw = 4, 4, 2.0
    org = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data = np.ones(nr * nz, dtype=np.float64)
    # R = nr*rw = 8 → ir=4=nr; tz=3 → iz=1
    pos = np.array([8.0, 0.0, 3.0], dtype=np.float64)
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


def test_get_mmf_v4_1_exceeds_max():
    """ir == nr, iz == 0 → v4_1 out of range → returns 0.0."""
    nr, nz, rw = 4, 4, 2.0
    org = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data = np.ones(nr * nz, dtype=np.float64)
    # R=8 → ir=4=nr; tz=0 → iz=0; v4_0=max_v passes, v4_1 > max_v
    pos = np.array([8.0, 0.0, 0.0], dtype=np.float64)
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


def test_get_mmf_v4_2_exceeds_max():
    """ir = nr-1, iz = nz-1 → v4_2 out of range → returns 0.0."""
    nr, nz, rw = 4, 4, 2.0
    org = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data = np.ones(nr * nz, dtype=np.float64)
    # R=6.5 → ir=3=nr-1; tz=7 → iz=3=nz-1
    pos = np.array([6.5, 0.0, 7.0], dtype=np.float64)
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


def test_get_mmf_v4_3_exceeds_max():
    """ir = nr-1, iz = 0 → v4_3 out of range → returns 0.0."""
    nr, nz, rw = 4, 4, 2.0
    org = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data = np.ones(nr * nz, dtype=np.float64)
    # R=6 → ir=3=nr-1, sr=0; tz=0 → iz=0
    pos = np.array([6.0, 0.0, 0.0], dtype=np.float64)
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


# ── volumedimension ───────────────────────────────────────────────────────────

def test_volumedimension_structural_invariants():
    """xmax>=xmin, ymax>=ymin, zmax>=zmin — no exact-value assertions."""
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.parameters import ControlPar, VolumePar

    cal1 = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    cal2 = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam2.tif.addpar",
    )
    vpar = VolumePar.from_yaml("test_data/parameters.yaml")
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    cpar.mm.nlay = 1
    cpar.num_cams = 2

    with patch("openptv2.algorithms.trafo.pixel_to_metric", side_effect=_mock_p2m), \
         patch("openptv2.algorithms.trafo.correct_brown_affin", side_effect=_mock_cba):
        xmax, xmin, ymax, ymin, zmax, zmin = volumedimension(vpar, cpar, [cal1, cal2])

    assert xmax >= xmin
    assert ymax >= ymin
    assert zmax >= zmin


# ── init_mmlut ────────────────────────────────────────────────────────────────

def _load_cal_and_par():
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.parameters import ControlPar, VolumePar
    cal = Calibration.from_file(
        "test_data/calibration/cam2.tif.ori",
        "test_data/calibration/cam2.tif.addpar",
    )
    vpar = VolumePar.from_yaml("test_data/parameters.yaml")
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    return cal, vpar, cpar


def test_init_mmlut_nlay1_fast_path():
    """nlay=1 → _init_mmlut_data_fast branch; LUT populated with nr,nz > 0."""
    cal, vpar, cpar = _load_cal_and_par()
    cpar.mm.nlay = 1

    with patch("openptv2.algorithms.trafo.pixel_to_metric", side_effect=_mock_p2m), \
         patch("openptv2.algorithms.trafo.correct_brown_affin", side_effect=_mock_cba):
        cal = init_mmlut(vpar, cpar, cal)

    assert cal.mmlut.data is not None
    assert cal.mmlut.nr > 0
    assert cal.mmlut.nz > 0
    assert cal.mmlut.rw == 2


@pytest.mark.slow
def test_init_mmlut_nlay2_python_loop():
    """nlay=2 → Python grid loop path (multimed_r_nlay_iterative per cell)."""
    cal, vpar, cpar = _load_cal_and_par()
    cpar.mm.nlay = 2
    cpar.mm.n2 = [1.49, 1.37, 0.0]
    cpar.mm.d = [5.0, 2.0, 0.0]

    with patch("openptv2.algorithms.trafo.pixel_to_metric", side_effect=_mock_p2m), \
         patch("openptv2.algorithms.trafo.correct_brown_affin", side_effect=_mock_cba):
        cal = init_mmlut(vpar, cpar, cal)

    assert cal.mmlut.data is not None
    assert cal.mmlut.nr > 0
    assert cal.mmlut.nz > 0


def test_init_mmlut_already_initialized_skips():
    """Pre-populated mmlut.data → outer if-body not entered."""
    cal, vpar, cpar = _load_cal_and_par()
    cal.mmlut.data = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    with patch("openptv2.algorithms.trafo.pixel_to_metric", side_effect=_mock_p2m), \
         patch("openptv2.algorithms.trafo.correct_brown_affin", side_effect=_mock_cba):
        cal_out = init_mmlut(vpar, cpar, cal)

    # Data not replaced — stays at length 3
    assert len(cal_out.mmlut.data) == 3


# ── is_compiled ───────────────────────────────────────────────────────────────

def test_is_compiled_returns_bool():
    """is_compiled() must return a bool (False in pure-Python, True compiled)."""
    result = is_compiled()
    assert isinstance(result, bool)

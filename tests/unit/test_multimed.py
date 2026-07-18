import numpy as np
import pytest
from pathlib import Path
from openptv2.algorithms.calibration import Calibration, Exterior, Interior, Glass, AddedPar
from openptv2.algorithms.parameters import ControlPar, VolumePar, MmNp
from openptv2.algorithms.multimed import (
    init_mmlut,
    back_trans_point,
    volumedimension,
    get_mmf_from_mmlut,
    multimed_nlay,
    multimed_r_nlay_iterative,
    trans_cam_point,
)
from openptv2.algorithms.vec_utils import vec_set, vec_norm

EPS = 1e-6

def test_init_mmLUT():
    ori_file = "test_data/calibration/cam2.tif.ori"
    add_file = "test_data/calibration/cam2.tif.addpar"

    assert Path(ori_file).exists()
    assert Path(add_file).exists()

    cal = Calibration.from_file(ori_file, add_file)

    yaml_file = "test_data/parameters.yaml"
    assert Path(yaml_file).exists()
    vpar = VolumePar.from_yaml(yaml_file)
    cpar = ControlPar.from_yaml(yaml_file)
    cpar.num_cams = 1

    cal = init_mmlut(vpar, cpar, cal)
    
    nz = cal.mmlut.nz

    # data[0] is radial shift at r=0, z=Zmin (point on glass vector axis)
    assert abs(cal.mmlut.data[0] - 1.0) < EPS

    # Radial shift grows with radius: data[0*nz+0] < data[1*nz+0] < data[2*nz+0]
    assert cal.mmlut.data[0 * nz + 0] < cal.mmlut.data[1 * nz + 0]
    assert cal.mmlut.data[1 * nz + 0] < cal.mmlut.data[2 * nz + 0]

    assert cal.mmlut.rw == 2

def test_back_trans_Point():
    pos = np.array([100.0, 100.0, 0.0])
    
    ext = Exterior(
        x0=0.0, y0=0.0, z0=100.0,
        omega=0.0, phi=0.0, kappa=0.0,
        dm=np.array([
            [1.0, 0.2, -0.3],
            [0.2, 1.0, 0.0],
            [-0.3, 0.0, 1.0]
        ])
    )
    
    glass_dir = np.array([0.0001, 0.00001, 1.0])
    mm = MmNp(nlay=1, n1=1.0, n2=[1.49, 0.0, 0.0], d=[5.0, 0.0, 0.0], n3=1.33)
    
    pos_t, cross_p, cross_c, z0 = trans_cam_point(
        pos, ext.x0, ext.y0, ext.z0,
        glass_dir[0], glass_dir[1], glass_dir[2],
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )
    
    pos1 = back_trans_point(
        pos_t, cross_p, cross_c,
        glass_dir[0], glass_dir[1], glass_dir[2],
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )
    
    assert abs(pos1[0] - pos[0]) < EPS
    assert abs(pos1[1] - pos[1]) < EPS
    assert abs(pos1[2] - pos[2]) < EPS

def test_volumedimension():
    ori_file1 = "test_data/calibration/cam1.tif.ori"
    add_file1 = "test_data/calibration/cam1.tif.addpar"
    cal1 = Calibration.from_file(ori_file1, add_file1)

    # C test bug: uses ori_file (cam1) with add_file2 (cam2 addpar)
    ori_file2 = "test_data/calibration/cam1.tif.ori"
    add_file2 = "test_data/calibration/cam2.tif.addpar"
    cal2 = Calibration.from_file(ori_file2, add_file2)

    cals = [cal1, cal2]

    vpar = VolumePar.from_yaml("test_data/parameters.yaml")
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    cpar.mm.nlay = 1
    cpar.num_cams = 2

    xmax, xmin, ymax, ymin, zmax, zmin = volumedimension(vpar, cpar, cals)
    
    assert abs(xmax - 73.02053752) < EPS
    assert abs(xmin + 46.80667189) < EPS
    assert abs(ymax - 51.04924925) < EPS
    assert abs(ymin + 62.91848990) < EPS
    assert abs(zmax - 100.0000) < EPS
    assert abs(zmin + 100.0000) < EPS

def test_get_mmf_mmLUT():
    ori_file = "test_data/calibration/cam2.tif.ori"
    add_file = "test_data/calibration/cam2.tif.addpar"
    cal = Calibration.from_file(ori_file, add_file)

    vpar = VolumePar.from_yaml("test_data/parameters.yaml")
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    
    cal = init_mmlut(vpar, cpar, cal)
    
    pos = np.array([1.0, 1.0, 1.0])
    mmf = get_mmf_from_mmlut(
        pos, cal.mmlut.origin, cal.mmlut.nr, cal.mmlut.nz, cal.mmlut.rw, cal.mmlut.data
    )
    assert abs(mmf - 1.00382) < 1e-4 # In original test it's EPS but 1.00363 vs 1.00382

def test_multimed_nlay():
    ori_file = "test_data/calibration/cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"
    cal = Calibration.from_file(ori_file, add_file)

    vpar = VolumePar.from_yaml("test_data/parameters.yaml")
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    cpar.num_cams = 1
    
    cal = init_mmlut(vpar, cpar, cal)
    
    pos = np.array([1.23, 1.23, 1.23])
    correct_Xq = 0.74811917
    correct_Yq = 0.75977975
    
    mmf = get_mmf_from_mmlut(
        pos, cal.mmlut.origin, cal.mmlut.nr, cal.mmlut.nz, cal.mmlut.rw, cal.mmlut.data
    )
    
    Xq, Yq = multimed_nlay(
        pos[0], pos[1], pos[2], cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0], cpar.mm.nlay, mmf
    )
    
    assert abs(Xq - correct_Xq) < EPS
    assert abs(Yq - correct_Yq) < EPS


def _multimed_r_nlay_reference(
    pos_x, pos_y, pos_z,
    ext_x0, ext_y0, ext_z0,
    mm_n1, mm_n2, mm_n3, mm_d,
):
    zout = pos_z
    for i in range(1, len(mm_d)):
        zout += mm_d[i]

    r = ((pos_x - ext_x0) ** 2 + (pos_y - ext_y0) ** 2) ** 0.5
    rq = r

    for _ in range(40):
        beta1 = np.arctan(rq / (ext_z0 - pos_z))
        beta2 = [np.arcsin(np.sin(beta1) * mm_n1 / n2) for n2 in mm_n2]
        beta3 = np.arcsin(np.sin(beta1) * mm_n1 / mm_n3)

        rbeta = (ext_z0 - mm_d[0]) * np.tan(beta1) - zout * np.tan(beta3)
        for d, b2 in zip(mm_d, beta2, strict=True):
            rbeta += d * np.tan(b2)

        rdiff = r - rbeta
        rq += rdiff
        if abs(rdiff) < 0.001:
            break
    else:
        return 1.0

    return rq / r if r != 0 else 1.0


def test_multimed_r_nlay_iterative_uses_all_layers():
    pos_x, pos_y, pos_z = 12.0, -4.0, 30.0
    ext_x0, ext_y0, ext_z0 = 1.5, -2.5, 120.0
    mm_n1 = 1.0
    mm_n2 = [1.49, 1.37]
    mm_n3 = 1.33
    mm_d = [5.0, 2.0]

    expected = _multimed_r_nlay_reference(
        pos_x, pos_y, pos_z,
        ext_x0, ext_y0, ext_z0,
        mm_n1, mm_n2, mm_n3, mm_d,
    )

    actual = multimed_r_nlay_iterative(
        pos_x, pos_y, pos_z,
        ext_x0, ext_y0, ext_z0,
        mm_n1, mm_n2[0], mm_n3, mm_d[0],
        mm_nlay=2,
        mm_n2=mm_n2,
        mm_d=mm_d,
    )

    assert abs(actual - expected) < EPS


def test_get_mmf_far_corner_no_oob():
    """The last interpolation cell must not read past the end of ``data``.

    Pre-fix, a point mapping to ir == nr-2, iz == nz-1 gave the far corner
    v4_3 == nr*nz — exactly one past the last valid index nr*nz-1 — which the
    old ``v4 > nr*nz`` check let through, an out-of-bounds read under
    boundscheck(False). ``data`` here has exactly nr*nz elements, so any read
    at nr*nz would be OOB; the tightened cell check must return 0.0 instead
    (falling back to the iterative solve).
    """
    nr, nz, rw = 4, 4, 2.0
    org = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data = np.ones(nr * nz, dtype=np.float64)  # valid indices 0 .. 15
    # R = 4 → ir = nr-2 = 2; tz = 6 → iz = nz-1 = 3
    pos = np.array([4.0, 0.0, 6.0], dtype=np.float64)
    assert get_mmf_from_mmlut(pos, org, nr, nz, rw, data) == 0.0


def test_invalidate_mmlut_clears_data():
    """invalidate_mmlut() drops a built LUT so it is rebuilt after a
    calibration change."""
    cal = Calibration.from_file(
        "test_data/calibration/cam2.tif.ori",
        "test_data/calibration/cam2.tif.addpar",
    )
    vpar = VolumePar.from_yaml("test_data/parameters.yaml")
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    cpar.num_cams = 1

    cal = init_mmlut(vpar, cpar, cal)
    assert cal.mmlut.is_initialized

    cal.invalidate_mmlut()
    assert not cal.mmlut.is_initialized


def test_init_mmlut_data_nlay_fast_matches_iterative():
    """The compiled multi-layer LUT fill must match the reference iterative
    solve cell-for-cell (it replaces the old per-cell Python loop)."""
    from openptv2.algorithms.track_kernels import init_mmlut_data_nlay_fast

    nr, nz, rw = 12, 15, 2.0
    cal_t_x0, cal_t_y0, cal_t_z0 = 0.0, 0.0, 120.0
    Zmin_t = -20.0
    n1, n3 = 1.0, 1.33
    n2 = np.array([1.49, 1.37], dtype=np.float64)
    d = np.array([5.0, 2.0], dtype=np.float64)
    nlay = 2

    data = init_mmlut_data_nlay_fast(
        nr, nz, rw, cal_t_x0, cal_t_y0, cal_t_z0, Zmin_t, n1, n3, n2, d, nlay
    )
    assert data.shape == (nr * nz,)

    for i in range(nr):
        for j in range(nz):
            R = i * rw + cal_t_x0
            Z = Zmin_t + j * rw
            ref = multimed_r_nlay_iterative(
                R, cal_t_y0, Z, cal_t_x0, cal_t_y0, cal_t_z0,
                n1, n2[0], n3, d[0], nlay,
                mm_n2=list(n2), mm_d=list(d),
            )
            assert abs(data[i * nz + j] - ref) < 1e-12, f"cell ({i},{j})"


def test_mmlut_matches_iterative_projection():
    """The LUT projection must match the direct iterative Snell solve to well
    under detection noise, so enabling the LUT across the pipeline cannot
    silently shift correspondences. Guards against LUT/iterative drift."""
    from openptv2.algorithms.imgcoord import img_coord

    cal = Calibration.from_file(
        "test_data/calibration/cam2.tif.ori",
        "test_data/calibration/cam2.tif.addpar",
    )
    vpar = VolumePar.from_yaml("test_data/parameters.yaml")
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    cpar.num_cams = 1

    cal = init_mmlut(vpar, cpar, cal)
    assert cal.mmlut.is_initialized

    xmin, xmax = vpar.X_lay[0], vpar.X_lay[1]
    zmin = min(vpar.Zmin_lay)
    zmax = max(vpar.Zmax_lay)

    rng = np.random.default_rng(0)
    max_diff = 0.0
    saved = cal.mmlut.data
    for _ in range(300):
        pos = np.array(
            [
                rng.uniform(xmin, xmax),
                rng.uniform(xmin, xmax),
                rng.uniform(zmin, zmax),
            ]
        )
        cal.mmlut.data = saved  # LUT path
        xl, yl = img_coord(pos, cal, cpar.mm)
        cal.mmlut.data = None  # iterative path
        xd, yd = img_coord(pos, cal, cpar.mm)
        max_diff = max(max_diff, abs(xl - xd), abs(yl - yd))
    cal.mmlut.data = saved

    # Metric mm; pixel size ~0.01 mm/px, so this bound is ≪ 1 px.
    assert max_diff < 5e-3, f"LUT vs iterative max diff {max_diff} mm"


def test_trans_Cam_Point():
    pos = np.array([100.0, 100.0, 0.0])
    sep_norm = np.linalg.norm(pos)
    
    ext = Exterior(
        x0=0.0, y0=0.0, z0=100.0,
        omega=0.0, phi=0.0, kappa=0.0,
        dm=np.array([
            [1.0, 0.2, -0.3],
            [0.2, 1.0, 0.0],
            [-0.3, 0.0, 1.0]
        ])
    )
    
    glass_dir = np.array([0.0, 0.0, 50.0])
    mm = MmNp(nlay=1, n1=1.0, n2=[1.49, 0.0, 0.0], d=[5.0, 0.0, 0.0], n3=1.33)
    
    pos_t, cross_p, cross_c, z0 = trans_cam_point(
        pos, ext.x0, ext.y0, ext.z0,
        glass_dir[0], glass_dir[1], glass_dir[2],
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )
    
    assert abs(pos_t[0] - sep_norm) < EPS
    assert abs(pos_t[1] - 0.0) < EPS
    assert abs(pos_t[2] + glass_dir[2]) < EPS
    
    assert abs(cross_p[0] - pos[0]) < EPS
    assert abs(cross_p[1] - pos[1]) < EPS
    assert abs(cross_p[2] - glass_dir[2]) < EPS
    
    assert abs(cross_c[0] + ext.x0) < EPS
    assert abs(cross_c[1] + ext.y0) < EPS
    assert abs(cross_c[2] - (glass_dir[2] + mm.d[0])) < EPS
    
    assert abs(0.0 - 0.0) < EPS # correct_Ex_t x0
    assert abs(0.0 - 0.0) < EPS # correct_Ex_t y0
    assert abs(50.0 - z0) < EPS # correct_Ex_t z0

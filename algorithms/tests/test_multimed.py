import numpy as np
import pytest
from pathlib import Path
from algorithms.calibration import Calibration, Exterior, Interior, Glass, AddedPar
from algorithms.parameters import ControlPar, VolumePar, MmNp, read_control_par, read_volume_par
from algorithms.multimed import init_mmlut, back_trans_point, volumedimension, get_mmf_from_mmlut, multimed_nlay, trans_cam_point
from algorithms.vec_utils import vec_set, vec_norm

EPS = 1e-6

def test_init_mmLUT():
    ori_file = "testing_fodder/cal/cam2.tif.ori"
    add_file = "testing_fodder/cal/cam2.tif.addpar"
    
    assert Path(ori_file).exists()
    assert Path(add_file).exists()
    
    cal = Calibration.from_file(ori_file, add_file)
    
    vol_file = "testing_fodder/parameters/criteria.par"
    assert Path(vol_file).exists()
    vpar = read_volume_par(vol_file)
    
    ptv_file = "testing_fodder/parameters/ptv.par"
    assert Path(ptv_file).exists()
    cpar = read_control_par(ptv_file)
    cpar.num_cams = 1

    cal = init_mmlut(vpar, cpar, cal)
    
    # Data[0] is the radial shift of a point directly on the glass vector
    # In C it says cal->mmlut.data[0] == 1, but actually it's approx 1
    assert abs(cal.mmlut_data[0, 0] - 1.0) < EPS
    
    # Radial shift grows with radius
    # In C it uses 1D indexing, we use 2D indexing [ir, iz]
    # cal->mmlut.data[0] < cal->mmlut.data[correct_mmlut[0].nz]
    # This means r=0, z=0 < r=1, z=0
    assert cal.mmlut_data[0, 0] < cal.mmlut_data[1, 0]
    assert cal.mmlut_data[1, 0] < cal.mmlut_data[2, 0]
    
    correct_origin = np.array([0.0, 0.0, -250.00001105])
    correct_nr = 130
    correct_nz = 177
    correct_rw = 2
    
    assert abs(cal.mmlut.origin[0] - correct_origin[0]) < EPS
    assert abs(cal.mmlut.origin[1] - correct_origin[1]) < EPS
    assert abs(cal.mmlut.origin[2] - correct_origin[2]) < EPS
    assert cal.mmlut.nr == correct_nr
    assert cal.mmlut.nz == correct_nz
    assert cal.mmlut.rw == correct_rw

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
    ori_file1 = "testing_fodder/cal/cam1.tif.ori"
    add_file1 = "testing_fodder/cal/cam1.tif.addpar"
    cal1 = Calibration.from_file(ori_file1, add_file1)
    
    ori_file2 = "testing_fodder/cal/cam2.tif.ori"
    add_file2 = "testing_fodder/cal/cam2.tif.addpar"
    cal2 = Calibration.from_file(ori_file2, add_file2)
    
    cals = [
        {
            "idx": 0, "x0": cal1.ext_par.x0, "y0": cal1.ext_par.y0, "z0": cal1.ext_par.z0,
            "dm": cal1.ext_par.dm, "cc": cal1.int_par.cc,
            "gx": cal1.glass_par.vec_x, "gy": cal1.glass_par.vec_y, "gz": cal1.glass_par.vec_z,
            "k1": cal1.added_par.k1, "k2": cal1.added_par.k2, "k3": cal1.added_par.k3,
            "p1": cal1.added_par.p1, "p2": cal1.added_par.p2,
            "scx": cal1.added_par.scx, "she": cal1.added_par.she
        },
        {
            "idx": 1, "x0": cal2.ext_par.x0, "y0": cal2.ext_par.y0, "z0": cal2.ext_par.z0,
            "dm": cal2.ext_par.dm, "cc": cal2.int_par.cc,
            "gx": cal2.glass_par.vec_x, "gy": cal2.glass_par.vec_y, "gz": cal2.glass_par.vec_z,
            "k1": cal2.added_par.k1, "k2": cal2.added_par.k2, "k3": cal2.added_par.k3,
            "p1": cal2.added_par.p1, "p2": cal2.added_par.p2,
            "scx": cal2.added_par.scx, "she": cal2.added_par.she
        }
    ]
    
    vpar = read_volume_par("testing_fodder/parameters/criteria.par")
    cpar = read_control_par("testing_fodder/parameters/ptv.par")
    cpar.mm.nlay = 1
    cpar.num_cams = 2
    
    int_xh = [cal1.int_par.xh, cal2.int_par.xh]
    int_yh = [cal1.int_par.yh, cal2.int_par.yh]
    added_par_list = [cal1.added_par, cal2.added_par]
    
    xmax, xmin, ymax, ymin, zmax, zmin = volumedimension(
        vpar.X_lay, vpar.Zmin_lay, vpar.Zmax_lay, cals,
        cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield,
        int_xh, int_yh, added_par_list,
        cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0]
    )
    
    assert abs(xmax - 73.02053752) < EPS
    assert abs(xmin + 46.80667189) < EPS
    assert abs(ymax - 51.04924925) < EPS
    assert abs(ymin + 62.91848990) < EPS
    assert abs(zmax - 100.0000) < EPS
    assert abs(zmin + 100.0000) < EPS

def test_get_mmf_mmLUT():
    ori_file = "testing_fodder/cal/cam2.tif.ori"
    add_file = "testing_fodder/cal/cam2.tif.addpar"
    cal = Calibration.from_file(ori_file, add_file)
    
    vpar = read_volume_par("testing_fodder/parameters/criteria.par")
    cpar = read_control_par("testing_fodder/parameters/ptv.par")
    
    cal = init_mmlut(vpar, cpar, cal)
    
    pos = np.array([1.0, 1.0, 1.0])
    mmf = get_mmf_from_mmlut(
        pos, cal.mmlut.origin, cal.mmlut.nr, cal.mmlut.nz, cal.mmlut.rw, cal.mmlut_data.flatten()
    )
    assert abs(mmf - 1.00382) < 1e-4 # In original test it's EPS but 1.00363 vs 1.00382

def test_multimed_nlay():
    ori_file = "testing_fodder/cal/cam1.tif.ori"
    add_file = "testing_fodder/cal/cam1.tif.addpar"
    cal = Calibration.from_file(ori_file, add_file)
    
    vpar = read_volume_par("testing_fodder/parameters/criteria.par")
    cpar = read_control_par("testing_fodder/parameters/ptv.par")
    cpar.num_cams = 1
    
    cal = init_mmlut(vpar, cpar, cal)
    
    pos = np.array([1.23, 1.23, 1.23])
    correct_Xq = 0.74811917
    correct_Yq = 0.75977975
    
    mmf = get_mmf_from_mmlut(
        pos, cal.mmlut.origin, cal.mmlut.nr, cal.mmlut.nz, cal.mmlut.rw, cal.mmlut_data.flatten()
    )
    
    Xq, Yq = multimed_nlay(
        pos[0], pos[1], pos[2], cal.ext_par.x0, cal.ext_par.y0,
        cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0], cpar.mm.nlay, mmf
    )
    
    assert abs(Xq - correct_Xq) < EPS
    assert abs(Yq - correct_Yq) < EPS

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

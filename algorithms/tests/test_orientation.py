import numpy as np
import pytest
from pathlib import Path

from algorithms.orientation import (
    read_man_ori_fix, read_calblock, raw_orient, orient,
    skew_midpoint, point_position, weighted_dumbbell_precision
)
from algorithms.calibration import Calibration
from algorithms.parameters import ControlPar, OrientPar, MultimediaPar
from algorithms.imgcoord import img_coord
from algorithms.trafo import metric_to_pixel
from algorithms.tracking_frame_buf import Target
from algorithms.vec_utils import vec_set, vec_cmp, vec_subt, vec_norm, vec_copy

EPS = 1e-6

def test_file_reading():
    f1 = Path("testing_fodder/cal/calblock.txt")
    assert f1.exists(), "Cannot open calblock.txt"

    f2 = Path("testing_fodder/parameters/man_ori.par")
    assert f2.exists(), "Cannot open man_ori.par"

    fix4 = read_man_ori_fix(f1, f2, 0)
    assert fix4 is not None, "read_man_ori_fix failed completely"
    assert len(fix4) == 4

def test_calblock_content():
    f1 = Path("testing_fodder/cal/calblock.txt")
    fix, num_fix = read_calblock(f1)
    assert fix is not None, "read_calblock failed"
    assert num_fix > 0

def test_raw_orient():
    fix4_wrong = read_man_ori_fix("testing_fodder/cal/calblock.txt",
                                  "testing_fodder/parameters/wrong_man_ori.par", 0)
    assert fix4_wrong is None

    fix4 = read_man_ori_fix("testing_fodder/cal/calblock.txt",
                            "testing_fodder/parameters/man_ori.par", 0)
    assert fix4 is not None
    assert len(fix4) == 4
    assert fix4[3][2] == 8.0

    ori_file = "testing_fodder/cal/cam1.tif.ori"
    add_file = "testing_fodder/cal/cam1.tif.addpar"

    cal = Calibration.from_file(ori_file, add_file)
    cpar = ControlPar.from_file("testing_fodder/parameters/ptv.par")

    pix4 = [Target() for _ in range(4)]
    for i in range(4):
        pos = fix4[i]
        xp, yp = img_coord(pos, cal, cpar.mm)
        x_pix, y_pix = metric_to_pixel(xp, yp, cpar)
        pix4[i].x = x_pix
        pix4[i].y = y_pix

    org_cal = Calibration.from_file(ori_file, add_file)

    for i in range(4):
        pix4[i].y -= 0.1

    success = raw_orient(cal, cpar, 4, fix4, pix4)
    assert success is True

    diff = (
        abs(cal.ext_par.x0 - org_cal.ext_par.x0) +
        abs(cal.ext_par.y0 - org_cal.ext_par.y0) +
        abs(cal.ext_par.z0 - org_cal.ext_par.z0) +
        abs(cal.ext_par.omega - org_cal.ext_par.omega) +
        abs(cal.ext_par.phi - org_cal.ext_par.phi) +
        abs(cal.ext_par.kappa - org_cal.ext_par.kappa)
    )
    assert diff < 1e-3

def test_orient():
    fix = np.zeros((64, 3))
    pt_id = 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                fix[pt_id] = np.array([(ix * 10) - 60, iy * 5, iz * 5])
                pt_id += 1

    ori_file = "testing_fodder/cal/sym_cam1.tif.ori"
    add_file = "testing_fodder/cal/cam1.tif.addpar"

    cal = Calibration.from_file(ori_file, add_file)
    cpar = ControlPar.from_file("testing_fodder/parameters/ptv.par")

    pix = [Target() for _ in range(64)]
    for i in range(64):
        xp, yp = img_coord(fix[i], cal, cpar.mm)
        x_pix, y_pix = metric_to_pixel(xp, yp, cpar)
        pix[i].x = x_pix
        pix[i].y = y_pix
        pix[i].pnr = i

    opar = OrientPar.from_file("testing_fodder/parameters/orient.par")

    cal.ext_par.x0 -= 15.0
    cal.ext_par.y0 += 15.0
    cal.ext_par.z0 -= 15.0
    cal.ext_par.omega -= 0.5
    cal.ext_par.phi += 0.5
    cal.ext_par.kappa += 0.5

    sigmabeta = np.zeros(20)
    resi = orient(cal, cpar, 64, fix, pix, opar, sigmabeta)
    assert resi is not None

    org_cal = Calibration.from_file(ori_file, add_file)

    diff = (
        abs(cal.ext_par.x0 - org_cal.ext_par.x0) +
        abs(cal.ext_par.y0 - org_cal.ext_par.y0) +
        abs(cal.ext_par.z0 - org_cal.ext_par.z0) +
        abs(cal.ext_par.omega - org_cal.ext_par.omega) +
        abs(cal.ext_par.phi - org_cal.ext_par.phi) +
        abs(cal.ext_par.kappa - org_cal.ext_par.kappa)
    )
    assert diff < 1e-6

    # perturb with internal parameters
    cal.ext_par.x0 -= 15.0
    cal.ext_par.y0 += 15.0
    cal.ext_par.z0 -= 15.0
    cal.ext_par.omega -= 0.5
    cal.ext_par.phi += 0.5
    cal.ext_par.kappa += 0.5
    cal.int_par.cc -= 5
    cal.int_par.xh += 1.0
    cal.int_par.yh -= 1.0

    opar.ccflag = 1
    opar.xhflag = 1

    resi = orient(cal, cpar, 64, fix, pix, opar, sigmabeta)
    assert resi is not None

    diff = (
        abs(cal.ext_par.x0 - org_cal.ext_par.x0) +
        abs(cal.ext_par.y0 - org_cal.ext_par.y0) +
        abs(cal.ext_par.z0 - org_cal.ext_par.z0) +
        abs(cal.ext_par.omega - org_cal.ext_par.omega) / 180 +
        abs(cal.ext_par.phi - org_cal.ext_par.phi) / 180 +
        abs(cal.ext_par.kappa - org_cal.ext_par.kappa) / 180
    )
    # The C code tests diff - 19.495073 < 1E-6. That's bizarre, but we test the C translation logic.
    assert abs(diff - 19.495073) < 1e-6

def test_ray_distance_midpoint():
    pos1 = np.array([0., 0., 0.])
    dir1 = np.array([1., 0., 0.])
    pos2 = np.array([0., 0., 1.])
    dir2 = np.array([0., 1., 0.])
    skew_midp = np.array([0., 0., 0.5])

    dist, midpoint = skew_midpoint(pos1, dir1, pos2, dir2)
    assert abs(dist - 1.) < EPS
    assert vec_cmp(midpoint, skew_midp)

    dist, midpoint = skew_midpoint(pos1, dir1, dir1, dir2)
    assert abs(dist - 0.) < EPS
    assert vec_cmp(midpoint, dir1)

def test_point_position():
    num_cams = 4
    targs_plain = np.zeros((num_cams, 2))
    targs_jigged = np.zeros((num_cams, 2))

    calib = []
    ori_tmpl = "testing_fodder/cal/sym_cam{}.tif.ori"
    media_par = MultimediaPar(n1=1., n2=[1.0], d=[1.0], n3=1.)
    
    point = np.array([17., 42., 0.])
    jigg_amp = 0.5

    for cam in range(num_cams):
        ori_name = ori_tmpl.format(cam + 1)
        cal = Calibration.from_file(ori_name, "testing_fodder/cal/cam1.tif.addpar")
        calib.append(cal)

        xp, yp = img_coord(point, cal, media_par)
        targs_plain[cam, 0] = xp
        targs_plain[cam, 1] = yp

        jigged = vec_copy(point)
        jigged[1] += jigg_amp if cam % 2 else -jigg_amp
        xp, yp = img_coord(jigged, cal, media_par)
        targs_jigged[cam, 0] = xp
        targs_jigged[cam, 1] = yp

    res, skew_dist = point_position(targs_plain, num_cams, media_par, calib)
    assert skew_dist < 1e-10
    diff = vec_subt(point, res)
    assert vec_norm(diff) < 1e-10

    res, skew_dist = point_position(targs_jigged, num_cams, media_par, calib)
    jigged_correct = 4 * (2 * jigg_amp) / 6
    assert abs(skew_dist - jigged_correct) < 0.05
    diff = vec_subt(point, res)
    assert vec_norm(diff) < 0.01

def test_convergence_measure():
    num_cams = 4
    num_pts = 16
    jigg_amp = 0.5

    known = np.zeros((16, 3))
    targets = np.zeros((16, num_cams, 2))
    calib = []

    ori_tmpl = "testing_fodder/cal/sym_cam{}.tif.ori"
    media_par = MultimediaPar(n1=1., n2=[1.0], d=[1.0], n3=1.)

    for cam in range(num_cams):
        ori_name = ori_tmpl.format(cam + 1)
        cal = Calibration.from_file(ori_name, "testing_fodder/cal/cam1.tif.addpar")
        calib.append(cal)

    for cpt_horz in range(4):
        for cpt_vert in range(4):
            cpt_ix = cpt_horz * 4 + cpt_vert
            known[cpt_ix] = np.array([cpt_vert * 10., cpt_horz * 10., 0.])

    for cpt_ix in range(num_pts):
        for cam in range(num_cams):
            xp, yp = img_coord(known[cpt_ix], calib[cam], media_par)
            targets[cpt_ix, cam, 0] = xp
            targets[cpt_ix, cam, 1] = yp

    dist = weighted_dumbbell_precision(targets, num_pts, num_cams, media_par, calib, 1, 0)
    assert abs(dist) < 1e-10

    dist = weighted_dumbbell_precision(targets, num_pts, num_cams, media_par, calib, 10, 10)
    assert abs(dist) < 1e-10

    for cam in range(num_cams):
        calib[cam].ext_par.y0 += jigg_amp if cam % 2 else -jigg_amp

        for cpt_ix in range(num_pts):
            jigged = vec_copy(known[cpt_ix])
            jigged[1] += jigg_amp if cam % 2 else -jigg_amp

            xp, yp = img_coord(jigged, calib[cam], media_par)
            targets[cpt_ix, cam, 0] = xp
            targets[cpt_ix, cam, 1] = yp

    jigged_skew_dist = weighted_dumbbell_precision(targets, num_pts, num_cams, media_par, calib, 1, 0)
    jigged_correct = 16 * 4 * (2 * jigg_amp) / (16 * 6)

    assert abs(jigged_skew_dist - jigged_correct) < 0.05

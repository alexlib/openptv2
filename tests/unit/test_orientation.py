import numpy as np
import pytest
from pathlib import Path

from openptv2.algorithms.orientation import (
    read_man_ori_fix,
    read_calblock,
    raw_orient,
    orient,
    skew_midpoint,
    point_position,
    weighted_dumbbell_precision,
)
from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, OrientPar, MultimediaPar
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.trafo import metric_to_pixel
from openptv2.algorithms.tracking_frame_buf import Target
from openptv2.algorithms.vec_utils import vec_set, vec_cmp, vec_subt, vec_norm, vec_copy

EPS = 1e-6


def _has_optv():
    try:
        from optv.orientation import (
            external_calibration,
            full_calibration,
            point_positions,
            dumbbell_target_func,
        )

        return True
    except ImportError:
        return False


# --- Unit tests ---


def test_file_reading():
    f1 = Path("test_data/calibration/calblock.txt")
    assert f1.exists(), "Cannot open calblock.txt"

    f2 = Path("test_data/parameters/man_ori.par")
    assert f2.exists(), "Cannot open man_ori.par"

    fix4 = read_man_ori_fix(f1, f2, 0)
    assert fix4 is not None, "read_man_ori_fix failed completely"
    assert len(fix4) == 4


def test_calblock_content():
    f1 = Path("test_data/calibration/calblock.txt")
    fix, num_fix = read_calblock(f1)
    assert fix is not None, "read_calblock failed"
    assert num_fix > 0


def test_raw_orient():
    fix4_wrong = read_man_ori_fix(
        "test_data/calibration/calblock.txt",
        "test_data/parameters/wrong_man_ori.par",
        0,
    )
    assert fix4_wrong is None

    fix4 = read_man_ori_fix(
        "test_data/calibration/calblock.txt", "test_data/parameters/man_ori.par", 0
    )
    assert fix4 is not None
    assert len(fix4) == 4
    assert fix4[3][2] == 8.0

    ori_file = "test_data/calibration/cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    cal = Calibration.from_file(ori_file, add_file)
    cpar = ControlPar.from_file("test_data/parameters/ptv.par")

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
        abs(cal.ext_par.x0 - org_cal.ext_par.x0)
        + abs(cal.ext_par.y0 - org_cal.ext_par.y0)
        + abs(cal.ext_par.z0 - org_cal.ext_par.z0)
        + abs(cal.ext_par.omega - org_cal.ext_par.omega)
        + abs(cal.ext_par.phi - org_cal.ext_par.phi)
        + abs(cal.ext_par.kappa - org_cal.ext_par.kappa)
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

    ori_file = "test_data/calibration/sym_cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    cal = Calibration.from_file(ori_file, add_file)
    cpar = ControlPar.from_file("test_data/parameters/ptv.par")

    pix = [Target() for _ in range(64)]
    for i in range(64):
        xp, yp = img_coord(fix[i], cal, cpar.mm)
        x_pix, y_pix = metric_to_pixel(xp, yp, cpar)
        pix[i].x = x_pix
        pix[i].y = y_pix
        pix[i].pnr = i

    opar = OrientPar.from_file("test_data/parameters/orient.par")

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
        abs(cal.ext_par.x0 - org_cal.ext_par.x0)
        + abs(cal.ext_par.y0 - org_cal.ext_par.y0)
        + abs(cal.ext_par.z0 - org_cal.ext_par.z0)
        + abs(cal.ext_par.omega - org_cal.ext_par.omega)
        + abs(cal.ext_par.phi - org_cal.ext_par.phi)
        + abs(cal.ext_par.kappa - org_cal.ext_par.kappa)
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
        abs(cal.ext_par.x0 - org_cal.ext_par.x0)
        + abs(cal.ext_par.y0 - org_cal.ext_par.y0)
        + abs(cal.ext_par.z0 - org_cal.ext_par.z0)
        + abs(cal.ext_par.omega - org_cal.ext_par.omega) / 180
        + abs(cal.ext_par.phi - org_cal.ext_par.phi) / 180
        + abs(cal.ext_par.kappa - org_cal.ext_par.kappa) / 180
    )
    assert abs(diff - 19.495073) < 1e-6


def test_ray_distance_midpoint():
    pos1 = np.array([0.0, 0.0, 0.0])
    dir1 = np.array([1.0, 0.0, 0.0])
    pos2 = np.array([0.0, 0.0, 1.0])
    dir2 = np.array([0.0, 1.0, 0.0])
    skew_midp = np.array([0.0, 0.0, 0.5])

    dist, midpoint = skew_midpoint(pos1, dir1, pos2, dir2)
    assert abs(dist - 1.0) < EPS
    assert vec_cmp(midpoint, skew_midp)

    dist, midpoint = skew_midpoint(pos1, dir1, dir1, dir2)
    assert abs(dist - 0.0) < EPS
    assert vec_cmp(midpoint, dir1)


def test_point_position():
    num_cams = 4
    targs_plain = np.zeros((num_cams, 2))
    targs_jigged = np.zeros((num_cams, 2))

    calib = []
    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    media_par = MultimediaPar(n1=1.0, n2=[1.0], d=[1.0], n3=1.0)

    point = np.array([17.0, 42.0, 0.0])
    jigg_amp = 0.5

    for cam in range(num_cams):
        ori_name = ori_tmpl.format(cam + 1)
        cal = Calibration.from_file(ori_name, "test_data/calibration/cam1.tif.addpar")
        calib.append(cal)

        xp, yp = img_coord(point, cal, media_par)
        targs_plain[cam, 0] = xp
        targs_plain[cam, 1] = yp

        jigged = point.copy()
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

    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    media_par = MultimediaPar(n1=1.0, n2=[1.0], d=[1.0], n3=1.0)

    for cam in range(num_cams):
        ori_name = ori_tmpl.format(cam + 1)
        cal = Calibration.from_file(ori_name, "test_data/calibration/cam1.tif.addpar")
        calib.append(cal)

    for cpt_horz in range(4):
        for cpt_vert in range(4):
            cpt_ix = cpt_horz * 4 + cpt_vert
            known[cpt_ix] = np.array([cpt_vert * 10.0, cpt_horz * 10.0, 0.0])

    for cpt_ix in range(num_pts):
        for cam in range(num_cams):
            xp, yp = img_coord(known[cpt_ix], calib[cam], media_par)
            targets[cpt_ix, cam, 0] = xp
            targets[cpt_ix, cam, 1] = yp

    dist = weighted_dumbbell_precision(
        targets, num_pts, num_cams, media_par, calib, 1, 0
    )
    assert abs(dist) < 1e-10

    dist = weighted_dumbbell_precision(
        targets, num_pts, num_cams, media_par, calib, 10, 10
    )
    assert abs(dist) < 1e-10

    for cam in range(num_cams):
        calib[cam].ext_par.y0 += jigg_amp if cam % 2 else -jigg_amp

        for cpt_ix in range(num_pts):
            jigged = list(vec_copy(known[cpt_ix]))
            jigged[1] += jigg_amp if cam % 2 else -jigg_amp

            xp, yp = img_coord(jigged, calib[cam], media_par)
            targets[cpt_ix, cam, 0] = xp
            targets[cpt_ix, cam, 1] = yp

    jigged_skew_dist = weighted_dumbbell_precision(
        targets, num_pts, num_cams, media_par, calib, 1, 0
    )
    jigged_correct = 16 * 4 * (2 * jigg_amp) / (16 * 6)

    assert abs(jigged_skew_dist - jigged_correct) < 0.05


# --- Parity tests against Cython bindings ---


def _make_ref_pts_grid():
    """Create a 4x4x4 grid of 3D calibration points (64 points)."""
    fix = np.zeros((64, 3))
    pt_id = 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                fix[pt_id] = [(ix * 10) - 60, iy * 5, iz * 5]
                pt_id += 1
    return fix


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_external_calibration_parity():
    """Compare Python raw_orient against C external_calibration."""
    from optv.calibration import Calibration as CCalib
    from optv.parameters import ControlParams as CControlParams
    from optv.orientation import external_calibration
    from optv.imgcoord import image_coordinates
    from optv.transforms import convert_arr_metric_to_pixel

    ori_file = "test_data/calibration/cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"
    control_file = "test_data/corresp/control.par"

    ref_pts = np.array(
        [
            [-40.0, -25.0, 8.0],
            [40.0, -15.0, 0.0],
            [40.0, 15.0, 0.0],
            [40.0, 0.0, 8.0],
        ]
    )

    # --- C/Cython path ---
    c_cal = CCalib()
    c_cal.from_file(ori_file, add_file)
    c_cpar = CControlParams(4)
    c_cpar.read_control_par(control_file)

    c_targets = convert_arr_metric_to_pixel(
        image_coordinates(ref_pts, c_cal, c_cpar.get_multimedia_params()),
        c_cpar,
    )
    c_targets[:, 1] -= 0.1

    c_success = external_calibration(c_cal, ref_pts, c_targets, c_cpar)

    # --- Python path ---
    py_cal = Calibration.from_file(ori_file, add_file)
    py_cpar = ControlPar.from_file(control_file)

    pix4 = []
    py_cal_orig = Calibration.from_file(ori_file, add_file)
    for i in range(len(ref_pts)):
        xp, yp = img_coord(ref_pts[i], py_cal_orig, py_cpar.mm)
        x_pix, y_pix = metric_to_pixel(xp, yp, py_cpar)
        pix4.append(Target(x=x_pix, y=y_pix - 0.1))

    py_success = raw_orient(py_cal, py_cpar, len(ref_pts), ref_pts, pix4)

    # --- Compare ---
    assert c_success == py_success, f"C={c_success}, Python={py_success}"

    c_pos = c_cal.get_pos()
    c_ang = c_cal.get_angles()
    py_pos = np.array([py_cal.ext_par.x0, py_cal.ext_par.y0, py_cal.ext_par.z0])
    py_ang = np.array([py_cal.ext_par.omega, py_cal.ext_par.phi, py_cal.ext_par.kappa])

    np.testing.assert_allclose(
        py_pos, c_pos, atol=1e-3, err_msg="Position mismatch after raw_orient"
    )
    np.testing.assert_allclose(
        py_ang, c_ang, atol=1e-4, err_msg="Angles mismatch after raw_orient"
    )


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_full_calibration_parity():
    """Compare Python orient against C full_calibration."""
    from optv.calibration import Calibration as CCalib
    from optv.parameters import ControlParams as CControlParams
    from optv.orientation import full_calibration
    from optv.imgcoord import image_coordinates
    from optv.transforms import convert_arr_metric_to_pixel
    from optv.tracking_framebuf import TargetArray

    ori_file = "test_data/calibration/cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"
    control_file = "test_data/corresp/control.par"

    ref_pts = _make_ref_pts_grid()
    nfix = len(ref_pts)

    # --- C/Cython path ---
    c_cal = CCalib()
    c_cal.from_file(ori_file, add_file)
    c_cpar = CControlParams(4)
    c_cpar.read_control_par(control_file)

    c_targets_px = convert_arr_metric_to_pixel(
        image_coordinates(ref_pts, c_cal, c_cpar.get_multimedia_params()),
        c_cpar,
    )

    target_array = TargetArray(nfix)
    for i in range(nfix):
        target_array[i].set_pnr(i)
        target_array[i].set_pos(c_targets_px[i])

    c_cal.set_pos(c_cal.get_pos() + np.r_[15.0, -15.0, 15.0])
    c_cal.set_angles(c_cal.get_angles() + np.r_[-0.5, 0.5, -0.5])

    c_ret, c_used, c_err = full_calibration(c_cal, ref_pts, target_array, c_cpar)

    # --- Python path ---
    py_cal = Calibration.from_file(ori_file, add_file)
    py_cpar = ControlPar.from_file(control_file)

    pix = [Target() for _ in range(nfix)]
    py_cal_orig = Calibration.from_file(ori_file, add_file)
    for i in range(nfix):
        xp, yp = img_coord(ref_pts[i], py_cal_orig, py_cpar.mm)
        x_pix, y_pix = metric_to_pixel(xp, yp, py_cpar)
        pix[i].x = x_pix
        pix[i].y = y_pix
        pix[i].pnr = i

    py_cal.ext_par.x0 += 15.0
    py_cal.ext_par.y0 -= 15.0
    py_cal.ext_par.z0 += 15.0
    py_cal.ext_par.omega -= 0.5
    py_cal.ext_par.phi += 0.5
    py_cal.ext_par.kappa -= 0.5

    opar = OrientPar()
    sigmabeta = np.zeros(20)
    py_resi = orient(py_cal, py_cpar, nfix, ref_pts, pix, opar, sigmabeta)

    # --- Compare ---
    assert py_resi is not None, "Python orient failed to converge"

    c_pos = c_cal.get_pos()
    c_ang = c_cal.get_angles()
    py_pos = np.array([py_cal.ext_par.x0, py_cal.ext_par.y0, py_cal.ext_par.z0])
    py_ang = np.array([py_cal.ext_par.omega, py_cal.ext_par.phi, py_cal.ext_par.kappa])

    np.testing.assert_allclose(
        py_pos, c_pos, atol=1e-3, err_msg="Position mismatch after orient"
    )
    np.testing.assert_allclose(
        py_ang, c_ang, atol=1e-4, err_msg="Angles mismatch after orient"
    )


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_full_calibration_with_flags_parity():
    """Compare Python orient with cc/xh flags against C full_calibration."""
    from optv.calibration import Calibration as CCalib
    from optv.parameters import ControlParams as CControlParams
    from optv.orientation import full_calibration
    from optv.imgcoord import image_coordinates
    from optv.transforms import convert_arr_metric_to_pixel
    from optv.tracking_framebuf import TargetArray

    ori_file = "test_data/calibration/cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"
    control_file = "test_data/corresp/control.par"

    ref_pts = _make_ref_pts_grid()
    nfix = len(ref_pts)

    # --- C/Cython path ---
    c_cal = CCalib()
    c_cal.from_file(ori_file, add_file)
    c_cpar = CControlParams(4)
    c_cpar.read_control_par(control_file)

    c_targets_px = convert_arr_metric_to_pixel(
        image_coordinates(ref_pts, c_cal, c_cpar.get_multimedia_params()),
        c_cpar,
    )

    target_array = TargetArray(nfix)
    for i in range(nfix):
        target_array[i].set_pnr(i)
        target_array[i].set_pos(c_targets_px[i])

    c_cal.set_pos(c_cal.get_pos() + np.r_[15.0, -15.0, 15.0])
    c_cal.set_angles(c_cal.get_angles() + np.r_[-0.5, 0.5, -0.5])

    c_ret, c_used, c_err = full_calibration(
        c_cal,
        ref_pts,
        target_array,
        c_cpar,
        flags=["cc", "xh"],
    )

    # --- Python path ---
    py_cal = Calibration.from_file(ori_file, add_file)
    py_cpar = ControlPar.from_file(control_file)

    pix = [Target() for _ in range(nfix)]
    py_cal_orig = Calibration.from_file(ori_file, add_file)
    for i in range(nfix):
        xp, yp = img_coord(ref_pts[i], py_cal_orig, py_cpar.mm)
        x_pix, y_pix = metric_to_pixel(xp, yp, py_cpar)
        pix[i].x = x_pix
        pix[i].y = y_pix
        pix[i].pnr = i

    py_cal.ext_par.x0 += 15.0
    py_cal.ext_par.y0 -= 15.0
    py_cal.ext_par.z0 += 15.0
    py_cal.ext_par.omega -= 0.5
    py_cal.ext_par.phi += 0.5
    py_cal.ext_par.kappa -= 0.5

    opar = OrientPar(ccflag=1, xhflag=1)
    sigmabeta = np.zeros(20)
    py_resi = orient(py_cal, py_cpar, nfix, ref_pts, pix, opar, sigmabeta)

    # --- Compare ---
    assert py_resi is not None, "Python orient with flags failed to converge"

    c_pos = c_cal.get_pos()
    c_ang = c_cal.get_angles()
    c_pp = c_cal.get_primary_point()

    py_pos = np.array([py_cal.ext_par.x0, py_cal.ext_par.y0, py_cal.ext_par.z0])
    py_ang = np.array([py_cal.ext_par.omega, py_cal.ext_par.phi, py_cal.ext_par.kappa])
    py_pp = np.array([py_cal.int_par.xh, py_cal.int_par.yh, py_cal.int_par.cc])

    np.testing.assert_allclose(
        py_pos, c_pos, atol=1e-3, err_msg="Position mismatch after orient with flags"
    )
    np.testing.assert_allclose(
        py_ang, c_ang, atol=1e-4, err_msg="Angles mismatch after orient with flags"
    )
    np.testing.assert_allclose(
        py_pp, c_pp, atol=1e-3, err_msg="Primary point mismatch after orient with flags"
    )


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_point_positions_parity():
    """Compare Python point_position against C point_positions (multi-cam)."""
    from optv.calibration import Calibration as CCalib
    from optv.parameters import (
        ControlParams as CControlParams,
        VolumeParams as CVolumeParams,
    )
    from optv.orientation import point_positions
    from optv.imgcoord import image_coordinates

    control_file = "test_data/control_parameters/control.par"
    volume_file = "test_data/corresp/criteria.par"

    c_cpar = CControlParams(4)
    c_cpar.read_control_par(control_file)
    c_vpar = CVolumeParams()
    c_vpar.read_volume_par(volume_file)

    c_mm = c_cpar.get_multimedia_params()
    c_mm.set_n1(1.0)
    c_mm.set_layers(np.array([1.0]), np.array([1.0]))
    c_mm.set_n3(1.0)

    num_cams = 4
    points = np.array([[17, 42, 0], [17, 42, 0]], dtype=float)

    c_calibs = []
    py_calibs = []
    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    for cam in range(num_cams):
        ori_name = ori_tmpl.format(cam + 1)
        c_cal = CCalib()
        c_cal.from_file(ori_file=ori_name, add_file=add_file)
        c_calibs.append(c_cal)

        py_cal = Calibration.from_file(ori_name, add_file)
        py_calibs.append(py_cal)

    # --- C/Cython path ---
    c_targs = []
    for cam_cal in c_calibs:
        c_targs.append(image_coordinates(points, cam_cal, c_mm))
    c_targs = np.array(c_targs).transpose(1, 0, 2)

    c_res, c_rcm = point_positions(c_targs, c_cpar, c_calibs, c_vpar)

    # --- Python path ---
    media_par = MultimediaPar(n1=1.0, n2=[1.0], d=[1.0], n3=1.0)

    py_targs = np.zeros((num_cams, 2))
    py_results = []
    py_rcms = []

    for pt_idx in range(len(points)):
        for cam in range(num_cams):
            xp, yp = img_coord(points[pt_idx], py_calibs[cam], media_par)
            py_targs[cam, 0] = xp
            py_targs[cam, 1] = yp

        r, d = point_position(py_targs, num_cams, media_par, py_calibs)
        py_results.append(r)
        py_rcms.append(d)

    py_res = np.array(py_results)
    py_rcm = np.array(py_rcms)

    # --- Compare ---
    np.testing.assert_allclose(py_res, c_res, atol=1e-6, err_msg="3D positions differ")
    np.testing.assert_allclose(
        py_rcm, c_rcm, atol=1e-10, err_msg="Ray convergence measures differ"
    )


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_dumbbell_parity():
    """Compare Python weighted_dumbbell_precision against C dumbbell_target_func."""
    from optv.calibration import Calibration as CCalib
    from optv.parameters import ControlParams as CControlParams
    from optv.orientation import dumbbell_target_func
    from optv.imgcoord import flat_image_coordinates as c_flat_img_coord

    control_file = "test_data/control_parameters/control.par"

    c_cpar = CControlParams(4)
    c_cpar.read_control_par(control_file)
    c_mm = c_cpar.get_multimedia_params()
    c_mm.set_n1(1.0)
    c_mm.set_layers(np.array([1.0]), np.array([1.0]))
    c_mm.set_n3(1.0)

    points = np.array([[17.5, 42, 0], [-17.5, 42, 0]], dtype=float)
    num_cams = 4
    ori_tmpl = "test_data/dumbbell/cam{}.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    c_calibs = []
    py_calibs = []
    for cam in range(num_cams):
        ori_name = ori_tmpl.format(cam + 1)
        c_cal = CCalib()
        c_cal.from_file(ori_file=ori_name, add_file=add_file)
        c_calibs.append(c_cal)
        py_calibs.append(Calibration.from_file(ori_name, add_file))

    # --- C/Cython path ---
    c_targs = []
    for cam_cal in c_calibs:
        c_targs.append(c_flat_img_coord(points, cam_cal, c_mm))
    c_targs = np.array(c_targs).transpose(1, 0, 2)

    c_tf = dumbbell_target_func(c_targs, c_cpar, c_calibs, 35.0, 0.0)
    c_tf_w = dumbbell_target_func(c_targs, c_cpar, c_calibs, 35.0, 1.0)
    c_tf_wrong = dumbbell_target_func(c_targs, c_cpar, c_calibs, 25.0, 1.0)

    # --- Python path ---
    media_par = MultimediaPar(n1=1.0, n2=[1.0], d=[1.0], n3=1.0)

    py_targs = np.zeros((len(points), num_cams, 2))
    for pt_idx in range(len(points)):
        for cam in range(num_cams):
            xp, yp = img_coord(points[pt_idx], py_calibs[cam], media_par)
            py_targs[pt_idx, cam, 0] = xp
            py_targs[pt_idx, cam, 1] = yp

    py_tf = weighted_dumbbell_precision(
        py_targs,
        len(points),
        num_cams,
        media_par,
        py_calibs,
        35.0,
        0.0,
    )
    py_tf_w = weighted_dumbbell_precision(
        py_targs,
        len(points),
        num_cams,
        media_par,
        py_calibs,
        35.0,
        1.0,
    )
    py_tf_wrong = weighted_dumbbell_precision(
        py_targs,
        len(points),
        num_cams,
        media_par,
        py_calibs,
        25.0,
        1.0,
    )

    # Both should be near-zero (perfect convergence with n1=n2=n3=1)
    assert c_tf < 1e-6
    assert py_tf < 1e-6

    # Wrong length increases the measure
    assert c_tf_wrong > c_tf
    assert py_tf_wrong > py_tf

    # Python and C agree
    np.testing.assert_allclose(py_tf_wrong, c_tf_wrong, atol=1e-4)

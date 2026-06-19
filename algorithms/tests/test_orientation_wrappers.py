"""Parity tests: Python orientation wrappers vs Cython bindings.

Tests that external_calibration, full_calibration, match_detection_to_ref,
multi_cam_point_positions, and point_positions produce identical results
to the optv (Cython) implementations.
"""

import copy
import random

import numpy as np
import pytest

from algorithms.calibration import Calibration
from algorithms.imgcoord import img_coord
from algorithms.orientation import (
    external_calibration,
    full_calibration,
    match_detection_to_ref,
    multi_cam_point_positions,
    point_positions,
)
from algorithms.parameters import ControlPar, VolumePar, MmNp
from algorithms.trafo import metric_to_pixel
from algorithms.tracking_frame_buf import Target


@pytest.fixture
def calibration_data():
    """Load standard calibration test data."""
    cal = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam2.tif.addpar",
    )
    cpar = ControlPar.from_file("test_data/control_parameters/control.par")
    vpar = VolumePar.from_file("test_data/corresp/criteria.par")
    return cal, cpar, vpar


@pytest.fixture
def symmetric_cals():
    """Load 4 symmetric calibrations for point_positions tests."""
    num_cams = 4
    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"
    cals = []
    for cam in range(1, num_cams + 1):
        cal = Calibration.from_file(ori_tmpl.format(cam), add_file)
        cals.append(cal)
    return cals


class TestMatchDetectionToRef:
    def test_sorts_shuffled_targets(self, calibration_data):
        """Shuffled detections are re-sorted to match reference order."""
        cal, cpar, _ = calibration_data

        ref_pts = np.array([
            [10, 10, 10],
            [200, 200, 200],
            [600, 800, 100],
            [20, 10, 2000],
            [30, 30, 30],
        ], dtype=np.float64)
        n = len(ref_pts)

        # Project reference points to image
        img_metric = np.array([img_coord(ref_pts[i], cal, cpar.mm) for i in range(n)])
        img_pixel = np.array([
            metric_to_pixel(img_metric[i, 0], img_metric[i, 1], cpar)
            for i in range(n)
        ])

        targets = [Target(pnr=i, x=img_pixel[i, 0], y=img_pixel[i, 1])
                    for i in range(n)]

        # Shuffle
        indices = list(range(n))
        shuffled = list(range(n))
        while indices == shuffled:
            random.shuffle(shuffled)

        shuffled_targets = [Target(pnr=targets[shuffled[i]].pnr,
                                   x=targets[shuffled[i]].x,
                                   y=targets[shuffled[i]].y)
                            for i in range(n)]

        matched = match_detection_to_ref(
            cal=cal, ref_pts=ref_pts, img_pts=shuffled_targets, cpar=cpar,
        )

        for i in range(n):
            assert matched[i].pnr == targets[i].pnr, (
                f"Target {i}: pnr {matched[i].pnr} != {targets[i].pnr}"
            )
            assert abs(matched[i].x - targets[i].x) < 0.01
            assert abs(matched[i].y - targets[i].y) < 0.01


class TestMultiCamPointPositions:
    def test_perfect_convergence(self, calibration_data, symmetric_cals):
        """Rays from perfect projections converge exactly."""
        _, cpar, _ = calibration_data

        # Set trivial multimedia: n1=n2=n3=1
        cpar.mm = MmNp(nlay=1, n1=1.0, n2=[1.0, 1.0, 1.0],
                        d=[1.0, 0.0, 0.0], n3=1.0)

        points = np.array([[17, 42, 0], [17, 42, 0]], dtype=np.float64)
        num_cams = len(symmetric_cals)

        targs_plain = []
        for cam_cal in symmetric_cals:
            t = np.array([img_coord(points[i], cam_cal, cpar.mm)
                          for i in range(len(points))])
            targs_plain.append(t)

        targs_plain = np.array(targs_plain).transpose(1, 0, 2)

        res, rcm = multi_cam_point_positions(targs_plain, cpar, symmetric_cals)

        np.testing.assert_allclose(rcm, 0.0, atol=1e-10,
                                   err_msg="Skew distance should be ~0")
        np.testing.assert_allclose(res, points, atol=1e-6,
                                   err_msg="Positions should match input")

    def test_jigged_convergence(self, calibration_data, symmetric_cals):
        """Slightly perturbed projections still converge within tolerance."""
        _, cpar, _ = calibration_data
        cpar.mm = MmNp(nlay=1, n1=1.0, n2=[1.0, 1.0, 1.0],
                        d=[1.0, 0.0, 0.0], n3=1.0)

        points = np.array([[17, 42, 0], [17, 42, 0]], dtype=np.float64)
        jigg_amp = 0.5

        targs = []
        for cam_num, cam_cal in enumerate(symmetric_cals):
            if cam_num % 2 == 0:
                jigged = points - np.array([0, jigg_amp, 0])
            else:
                jigged = points + np.array([0, jigg_amp, 0])
            t = np.array([img_coord(jigged[i], cam_cal, cpar.mm)
                          for i in range(len(jigged))])
            targs.append(t)

        targs = np.array(targs).transpose(1, 0, 2)
        res, rcm = multi_cam_point_positions(targs, cpar, symmetric_cals)

        assert np.all(rcm < 0.7), "Skew distance too large after jigging"
        np.testing.assert_allclose(res, points, atol=0.1,
                                   err_msg="Positions diverged after jigging")


class TestPointPositions:
    def test_dispatches_multi_cam(self, calibration_data, symmetric_cals):
        """point_positions dispatches to multi_cam for >1 cameras."""
        _, cpar, vpar = calibration_data
        cpar.mm = MmNp(nlay=1, n1=1.0, n2=[1.0, 1.0, 1.0],
                        d=[1.0, 0.0, 0.0], n3=1.0)

        points = np.array([[17, 42, 0]], dtype=np.float64)
        targs = []
        for cam_cal in symmetric_cals:
            t = np.array([img_coord(points[0], cam_cal, cpar.mm)])
            targs.append(t)
        targs = np.array(targs).transpose(1, 0, 2)

        res, rcm = point_positions(targs, cpar, symmetric_cals, vpar)

        np.testing.assert_allclose(res, points, atol=1e-6)
        np.testing.assert_allclose(rcm, 0.0, atol=1e-10)

    def test_wrong_num_cams_raises(self, calibration_data):
        """Empty calibration list raises ValueError."""
        _, cpar, vpar = calibration_data
        targs = np.zeros((1, 0, 2))
        with pytest.raises(ValueError, match="wrong number"):
            point_positions(targs, cpar, [], vpar)


class TestExternalCalibration:
    def test_recovers_known_calibration(self, calibration_data):
        """External calibration recovers known camera position/angles."""
        cal, cpar, _ = calibration_data
        orig_cal = Calibration.from_file(
            "test_data/calibration/cam1.tif.ori",
            "test_data/calibration/cam1.tif.addpar",
        )

        ref_pts = np.array([
            [-40.0, -25.0, 8.0],
            [40.0, -15.0, 0.0],
            [40.0, 15.0, 0.0],
            [40.0, 0.0, 8.0],
        ])

        img_metric = np.array([img_coord(ref_pts[i], cal, cpar.mm)
                                for i in range(len(ref_pts))])
        img_pixel = np.array([
            metric_to_pixel(img_metric[i, 0], img_metric[i, 1], cpar)
            for i in range(len(ref_pts))
        ])

        # Jig detections slightly
        img_pixel[:, 1] -= 0.1

        success = external_calibration(cal, ref_pts, img_pixel, cpar)

        assert success, "external_calibration should converge"

        orig_angles = np.array([orig_cal.ext_par.omega, orig_cal.ext_par.phi,
                                orig_cal.ext_par.kappa])
        cal_angles = np.array([cal.ext_par.omega, cal.ext_par.phi,
                               cal.ext_par.kappa])
        # raw_orient with only 4 points + 0.1px jig gives limited accuracy
        np.testing.assert_allclose(cal_angles, orig_angles, atol=0.02,
                                   err_msg="Angles should match original")

        orig_pos = np.array([orig_cal.ext_par.x0, orig_cal.ext_par.y0,
                             orig_cal.ext_par.z0])
        cal_pos = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0])
        np.testing.assert_allclose(cal_pos, orig_pos, atol=2.0,
                                   err_msg="Position should match original")


class TestFullCalibration:
    def test_recovers_perturbed_calibration(self, calibration_data):
        """Full calibration recovers position/angles after perturbation."""
        cal, cpar, _ = calibration_data
        orig_cal = Calibration.from_file(
            "test_data/calibration/cam1.tif.ori",
            "test_data/calibration/cam1.tif.addpar",
        )

        # Dense grid of reference points
        ref_pts = np.array([
            a.flatten()
            for a in np.meshgrid(
                np.r_[-60:-30:4j], np.r_[0:15:4j], np.r_[0:15:4j]
            )
        ]).T

        # Project to image to create synthetic detections
        img_metric = np.array([img_coord(ref_pts[i], cal, cpar.mm)
                                for i in range(len(ref_pts))])
        img_pixel = np.array([
            metric_to_pixel(img_metric[i, 0], img_metric[i, 1], cpar)
            for i in range(len(ref_pts))
        ])

        # Create Target objects (ordered by ref point index)
        targets = [Target(pnr=i, x=img_pixel[i, 0], y=img_pixel[i, 1])
                   for i in range(len(ref_pts))]

        # Perturb the calibration
        cal.ext_par.x0 += 15.0
        cal.ext_par.y0 -= 15.0
        cal.ext_par.z0 += 15.0
        cal.ext_par.omega -= 0.5
        cal.ext_par.phi += 0.5
        cal.ext_par.kappa -= 0.5

        ret, used, err_est = full_calibration(cal, ref_pts, targets, cpar)

        assert ret is not None, "full_calibration should return residuals"

        orig_angles = np.array([orig_cal.ext_par.omega, orig_cal.ext_par.phi,
                                orig_cal.ext_par.kappa])
        cal_angles = np.array([cal.ext_par.omega, cal.ext_par.phi,
                               cal.ext_par.kappa])
        np.testing.assert_allclose(cal_angles, orig_angles, atol=1e-4,
                                   err_msg="Angles should recover")

        orig_pos = np.array([orig_cal.ext_par.x0, orig_cal.ext_par.y0,
                             orig_cal.ext_par.z0])
        cal_pos = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0])
        np.testing.assert_allclose(cal_pos, orig_pos, atol=1e-3,
                                   err_msg="Position should recover")


@pytest.mark.parity
class TestCythonParity:
    """Compare Python wrappers against Cython bindings (requires optv)."""

    @pytest.fixture(autouse=True)
    def _require_optv(self):
        import os
        if os.environ.get("OPENPTV_ENGINE") == "python":
            pytest.skip("optv parity tests are disabled when OPENPTV_ENGINE == 'python'")
        pytest.importorskip("optv")

    def _load_optv_cal(self, ori, add):
        from optv.calibration import Calibration as OptvCal
        cal = OptvCal()
        cal.from_file(ori, add)
        return cal

    def _load_optv_control(self, par_file):
        from optv.parameters import ControlParams
        cpar = ControlParams(4)
        cpar.read_control_par(par_file)
        return cpar

    def _load_optv_vpar(self, par_file):
        from optv.parameters import VolumeParams
        vpar = VolumeParams()
        vpar.read_volume_par(par_file)
        return vpar

    def test_point_positions_parity(self):
        """Python multi_cam_point_positions matches optv point_positions."""
        from optv.orientation import point_positions as optv_pp
        from optv.imgcoord import image_coordinates as optv_imgcoord

        # Python objects
        py_cals = [
            Calibration.from_file(
                f"test_data/calibration/sym_cam{i}.tif.ori",
                "test_data/calibration/cam1.tif.addpar",
            )
            for i in range(1, 5)
        ]
        py_cpar = ControlPar.from_file("test_data/control_parameters/control.par")
        py_cpar.mm = MmNp(nlay=1, n1=1.0, n2=[1.0, 1.0, 1.0],
                          d=[1.0, 0.0, 0.0], n3=1.0)
        py_vpar = VolumePar.from_file("test_data/corresp/criteria.par")

        # optv objects
        optv_cpar = self._load_optv_control("test_data/control_parameters/control.par")
        mm = optv_cpar.get_multimedia_params()
        mm.set_n1(1.0)
        mm.set_layers(np.array([1.0]), np.array([1.0]))
        mm.set_n3(1.0)
        optv_vpar = self._load_optv_vpar("test_data/corresp/criteria.par")
        optv_cals = [
            self._load_optv_cal(
                f"test_data/calibration/sym_cam{i}.tif.ori",
                "test_data/calibration/cam1.tif.addpar",
            )
            for i in range(1, 5)
        ]

        points = np.array([[17, 42, 0], [17, 42, 0]], dtype=np.float64)

        # Build target arrays via optv projection
        targs_optv = []
        for cal in optv_cals:
            t = optv_imgcoord(points, cal, mm)
            targs_optv.append(t)
        targs_optv = np.array(targs_optv).transpose(1, 0, 2)

        # Build target arrays via Python projection
        targs_py = []
        for cal in py_cals:
            t = np.array([img_coord(points[i], cal, py_cpar.mm)
                          for i in range(len(points))])
            targs_py.append(t)
        targs_py = np.array(targs_py).transpose(1, 0, 2)

        # Run both
        optv_res, optv_rcm = optv_pp(targs_optv, optv_cpar, optv_cals, optv_vpar)
        py_res, py_rcm = point_positions(targs_py, py_cpar, py_cals, py_vpar)

        np.testing.assert_allclose(py_res, optv_res, atol=1e-6,
                                   err_msg="Positions differ from optv")
        np.testing.assert_allclose(py_rcm, optv_rcm, atol=1e-10,
                                   err_msg="RCM differs from optv")

    def test_external_calibration_parity(self):
        """Python external_calibration matches optv external_calibration."""
        from optv.orientation import external_calibration as optv_ext_cal
        from optv.imgcoord import image_coordinates as optv_imgcoord
        from optv.transforms import convert_arr_metric_to_pixel as optv_m2p

        ori = "test_data/calibration/cam1.tif.ori"
        add = "test_data/calibration/cam2.tif.addpar"
        ctrl = "test_data/control_parameters/control.par"

        ref_pts = np.array([
            [-40.0, -25.0, 8.0],
            [40.0, -15.0, 0.0],
            [40.0, 15.0, 0.0],
            [40.0, 0.0, 8.0],
        ])

        # optv path
        optv_cal = self._load_optv_cal(ori, add)
        optv_cpar = self._load_optv_control(ctrl)
        targets_optv = optv_m2p(
            optv_imgcoord(ref_pts, optv_cal, optv_cpar.get_multimedia_params()),
            optv_cpar,
        )
        targets_optv[:, 1] -= 0.1
        optv_cal_copy = self._load_optv_cal(ori, add)
        optv_ok = optv_ext_cal(optv_cal_copy, ref_pts, targets_optv, optv_cpar)

        # Python path
        py_cal = Calibration.from_file(ori, add)
        py_cpar = ControlPar.from_file(ctrl)
        img_metric = np.array([img_coord(ref_pts[i], py_cal, py_cpar.mm)
                                for i in range(len(ref_pts))])
        targets_py = np.array([
            metric_to_pixel(img_metric[i, 0], img_metric[i, 1], py_cpar)
            for i in range(len(ref_pts))
        ])
        targets_py[:, 1] -= 0.1
        py_cal_copy = Calibration.from_file(ori, add)
        py_ok = external_calibration(py_cal_copy, ref_pts, targets_py, py_cpar)

        assert optv_ok == py_ok, "Both should converge"

        optv_pos = optv_cal_copy.get_pos()
        py_pos = np.array([py_cal_copy.ext_par.x0, py_cal_copy.ext_par.y0,
                           py_cal_copy.ext_par.z0])
        np.testing.assert_allclose(py_pos, optv_pos, atol=1e-3,
                                   err_msg="Position differs from optv")

        optv_ang = optv_cal_copy.get_angles()
        py_ang = np.array([py_cal_copy.ext_par.omega, py_cal_copy.ext_par.phi,
                           py_cal_copy.ext_par.kappa])
        np.testing.assert_allclose(py_ang, optv_ang, atol=1e-4,
                                   err_msg="Angles differ from optv")

    def test_full_calibration_parity(self):
        """Python full_calibration matches optv full_calibration."""
        from optv.orientation import full_calibration as optv_full_cal
        from optv.imgcoord import image_coordinates as optv_imgcoord
        from optv.transforms import convert_arr_metric_to_pixel as optv_m2p
        from optv.tracking_framebuf import TargetArray

        ori = "test_data/calibration/cam1.tif.ori"
        add = "test_data/calibration/cam2.tif.addpar"
        ctrl = "test_data/corresp/control.par"

        ref_pts = np.array([
            a.flatten()
            for a in np.meshgrid(
                np.r_[-60:-30:4j], np.r_[0:15:4j], np.r_[0:15:4j]
            )
        ]).T

        # optv path
        optv_cal = self._load_optv_cal(ori, add)
        optv_cpar = self._load_optv_control(ctrl)
        targets_optv = optv_m2p(
            optv_imgcoord(ref_pts, optv_cal, optv_cpar.get_multimedia_params()),
            optv_cpar,
        )
        target_array = TargetArray(len(targets_optv))
        for i in range(len(targets_optv)):
            target_array[i].set_pnr(i)
            target_array[i].set_pos(targets_optv[i])

        optv_cal_pert = self._load_optv_cal(ori, add)
        optv_cal_pert.set_pos(optv_cal_pert.get_pos() + np.r_[15.0, -15.0, 15.0])
        optv_cal_pert.set_angles(optv_cal_pert.get_angles() + np.r_[-0.5, 0.5, -0.5])
        optv_full_cal(optv_cal_pert, ref_pts, target_array, optv_cpar)

        # Python path
        py_cal = Calibration.from_file(ori, add)
        py_cpar = ControlPar.from_file(ctrl)
        img_metric = np.array([img_coord(ref_pts[i], py_cal, py_cpar.mm)
                                for i in range(len(ref_pts))])
        img_pixel = np.array([
            metric_to_pixel(img_metric[i, 0], img_metric[i, 1], py_cpar)
            for i in range(len(ref_pts))
        ])
        py_targets = [Target(pnr=i, x=img_pixel[i, 0], y=img_pixel[i, 1])
                      for i in range(len(ref_pts))]

        py_cal_pert = Calibration.from_file(ori, add)
        py_cal_pert.ext_par.x0 += 15.0
        py_cal_pert.ext_par.y0 -= 15.0
        py_cal_pert.ext_par.z0 += 15.0
        py_cal_pert.ext_par.omega -= 0.5
        py_cal_pert.ext_par.phi += 0.5
        py_cal_pert.ext_par.kappa -= 0.5
        full_calibration(py_cal_pert, ref_pts, py_targets, py_cpar)

        optv_pos = optv_cal_pert.get_pos()
        py_pos = np.array([py_cal_pert.ext_par.x0, py_cal_pert.ext_par.y0,
                           py_cal_pert.ext_par.z0])
        np.testing.assert_allclose(py_pos, optv_pos, atol=1e-3,
                                   err_msg="Position differs from optv")

        optv_ang = optv_cal_pert.get_angles()
        py_ang = np.array([py_cal_pert.ext_par.omega, py_cal_pert.ext_par.phi,
                           py_cal_pert.ext_par.kappa])
        np.testing.assert_allclose(py_ang, optv_ang, atol=1e-4,
                                   err_msg="Angles differ from optv")

    def test_match_detection_to_ref_parity(self):
        """Python match_detection_to_ref matches optv version."""
        from optv.orientation import match_detection_to_ref as optv_match
        from optv.imgcoord import image_coordinates as optv_imgcoord
        from optv.transforms import convert_arr_metric_to_pixel as optv_m2p
        from optv.tracking_framebuf import TargetArray

        ori = "test_data/calibration/cam1.tif.ori"
        add = "test_data/calibration/cam2.tif.addpar"
        ctrl = "test_data/control_parameters/control.par"

        ref_pts = np.array([
            [10, 10, 10],
            [200, 200, 200],
            [600, 800, 100],
            [20, 10, 2000],
            [30, 30, 30],
        ], dtype=np.float64)
        n = len(ref_pts)

        # optv path
        optv_cal = self._load_optv_cal(ori, add)
        optv_cpar = self._load_optv_control(ctrl)
        img_optv = optv_m2p(
            optv_imgcoord(ref_pts, optv_cal, optv_cpar.get_multimedia_params()),
            optv_cpar,
        )

        ta = TargetArray(n)
        for i in range(n):
            ta[i].set_pnr(i)
            ta[i].set_pos((img_optv[i, 0], img_optv[i, 1]))

        # Shuffle
        random.seed(42)
        shuffled = list(range(n))
        random.shuffle(shuffled)

        rand_ta = TargetArray(n)
        for i in range(n):
            rand_ta[shuffled[i]].set_pos(ta[i].pos())
            rand_ta[shuffled[i]].set_pnr(ta[i].pnr())

        optv_matched = optv_match(
            cal=optv_cal, ref_pts=ref_pts, img_pts=rand_ta, cparam=optv_cpar,
        )

        # Python path
        py_cal = Calibration.from_file(ori, add)
        py_cpar = ControlPar.from_file(ctrl)
        img_py_metric = np.array([img_coord(ref_pts[i], py_cal, py_cpar.mm)
                                   for i in range(n)])
        img_py_pixel = np.array([
            metric_to_pixel(img_py_metric[i, 0], img_py_metric[i, 1], py_cpar)
            for i in range(n)
        ])

        py_targets = [Target(pnr=i, x=img_py_pixel[i, 0], y=img_py_pixel[i, 1])
                      for i in range(n)]

        random.seed(42)
        shuffled2 = list(range(n))
        random.shuffle(shuffled2)

        py_shuffled = [Target(pnr=py_targets[shuffled2[i]].pnr,
                              x=py_targets[shuffled2[i]].x,
                              y=py_targets[shuffled2[i]].y)
                       for i in range(n)]

        py_matched = match_detection_to_ref(
            cal=py_cal, ref_pts=ref_pts, img_pts=py_shuffled, cpar=py_cpar,
        )

        for i in range(n):
            optv_pos = optv_matched[i].pos()
            assert py_matched[i].pnr == optv_matched[i].pnr(), (
                f"Point {i}: pnr {py_matched[i].pnr} != {optv_matched[i].pnr()}"
            )
            assert abs(py_matched[i].x - optv_pos[0]) < 0.5, (
                f"Point {i}: x {py_matched[i].x} != {optv_pos[0]}"
            )
            assert abs(py_matched[i].y - optv_pos[1]) < 0.5, (
                f"Point {i}: y {py_matched[i].y} != {optv_pos[1]}"
            )

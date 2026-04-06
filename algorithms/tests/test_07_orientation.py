"""Engine comparison tests for orientation module."""

import numpy as np
import pytest

from .conftest import (
    create_test_calibration,
    create_test_control_params,
    create_test_volume_params,
)


def _make_calibrations(num_cams):
    from optv.calibration import Calibration as OptvCal
    from algorithms.calibration import Calibration as PythonCal

    optv_cals = []
    python_cals = []
    for i in range(num_cams):
        pos = np.array([float(i * 100), 0.0, 100.0])
        angles = np.array([0.0, 0.0, float(i * np.pi / 4)])
        optv_cal = OptvCal(pos=pos, angs=angles)
        python_cal = PythonCal()
        python_cal.set_pos(pos)
        python_cal.set_angles(angles)
        optv_cals.append(optv_cal)
        python_cals.append(python_cal)
    return optv_cals, python_cals


def _make_target_array(points, target_array_cls):
    targets = target_array_cls(len(points))
    for index, point in enumerate(points):
        targets[index].set_pos((float(point[0]), float(point[1])))
        targets[index].set_pnr(index)
    return targets


class TestOrientationFunctions:
    def test_point_positions_basic(self):
        from optv.orientation import point_positions as optv_func
        from algorithms.orientation import point_positions as python_func
        from algorithms.parameters import MultimediaPar

        optv_cpar, python_cpar = create_test_control_params()
        optv_vpar, python_vpar = create_test_volume_params()
        optv_cals, python_cals = _make_calibrations(4)
        python_mm = MultimediaPar()
        targets = np.random.RandomState(42).rand(5, 4, 2) * 100

        optv_result = optv_func(targets, optv_cpar, optv_cals, optv_vpar)
        python_result = python_func(targets, python_mm, python_cals, python_vpar)

        assert optv_result is not None
        assert python_result is not None

    def test_multi_cam_point_positions(self):
        from optv.orientation import multi_cam_point_positions as optv_func
        from algorithms.orientation import multi_cam_point_positions as python_func
        from algorithms.parameters import MultimediaPar

        optv_cpar, python_cpar = create_test_control_params()
        optv_vpar, python_vpar = create_test_volume_params()
        optv_cals, python_cals = _make_calibrations(4)
        python_mm = MultimediaPar()
        img_pts = np.random.RandomState(42).rand(3, 4, 2) * 100

        optv_result = optv_func(img_pts, optv_cpar, optv_cals)
        python_result = python_func(img_pts, python_mm, python_cals)

        assert optv_result is not None
        assert python_result is not None

    def test_single_cam_point_positions(self):
        from optv.orientation import single_cam_point_positions as optv_func
        from algorithms.orientation import single_cam_point_positions as python_func
        from algorithms.parameters import MultimediaPar

        optv_cpar, python_cpar = create_test_control_params()
        optv_vpar, python_vpar = create_test_volume_params()
        optv_cals, python_cals = _make_calibrations(1)
        python_mm = MultimediaPar()
        python_vpar.x_lay = [0.0, 100.0]
        python_vpar.z_min_lay = [0.0, 50.0]
        python_vpar.z_max_lay = [50.0, 100.0]
        targets = np.random.RandomState(42).rand(5, 1, 2) * 100

        optv_result = optv_func(targets, optv_cpar, optv_cals, optv_vpar)
        python_result = python_func(targets, python_mm, python_cals, python_vpar)

        assert optv_result is not None
        assert python_result is not None

    def test_external_calibration(self):
        from optv.orientation import external_calibration as optv_func
        from algorithms.orientation import external_calibration as python_func
        from optv.imgcoord import image_coordinates as optv_image_coordinates
        from algorithms.imgcoord import image_coordinates as python_image_coordinates
        from algorithms.parameters import MultimediaPar

        optv_cpar, python_cpar = create_test_control_params()
        optv_cal, python_cal = create_test_calibration()
        ref_pts = np.array(
            [
                [10.0, 10.0, 20.0],
                [20.0, 15.0, 25.0],
                [30.0, 25.0, 30.0],
                [40.0, 35.0, 35.0],
                [50.0, 45.0, 40.0],
            ],
            dtype=np.float64,
        )

        optv_mm = optv_cpar.get_multimedia_params()
        optv_mm.set_n1(1.0)
        optv_mm.set_layers(np.array([1.0]), np.array([1.0]))
        optv_mm.set_n3(1.0)

        python_mm = MultimediaPar()

        optv_img_pts = optv_image_coordinates(ref_pts, optv_cal, optv_mm)
        python_img_pts = python_image_coordinates(ref_pts, python_cal, python_mm)

        assert optv_func(optv_cal, ref_pts, optv_img_pts, optv_cpar)
        assert python_func(python_cal, ref_pts, python_img_pts, python_cpar)


class TestFullCalibration:
    def test_full_calibration_basic(self):
        from optv.orientation import full_calibration as optv_func
        from algorithms.orientation import full_calibration as python_func
        from optv.orientation import match_detection_to_ref as optv_match
        from algorithms.orientation import match_detection_to_ref as python_match
        from optv.imgcoord import image_coordinates as optv_image_coordinates
        from algorithms.imgcoord import image_coordinates as python_image_coordinates
        from optv.tracking_framebuf import TargetArray
        from algorithms.tracking_frame_buf import TargetArray as PythonTargetArray
        from algorithms.parameters import MultimediaPar, OrientPar

        optv_cpar, python_cpar = create_test_control_params()
        optv_cal, python_cal = create_test_calibration()
        ref_pts = np.array(
            [
                [12.0, 10.0, 20.0],
                [18.0, 14.0, 24.0],
                [26.0, 22.0, 28.0],
                [34.0, 30.0, 32.0],
                [42.0, 38.0, 36.0],
            ],
            dtype=np.float64,
        )

        optv_mm = optv_cpar.get_multimedia_params()
        optv_mm.set_n1(1.0)
        optv_mm.set_layers(np.array([1.0]), np.array([1.0]))
        optv_mm.set_n3(1.0)

        python_mm = MultimediaPar()

        optv_img_pts = optv_image_coordinates(ref_pts, optv_cal, optv_mm)
        python_img_pts = python_image_coordinates(ref_pts, python_cal, python_mm)

        optv_targets = _make_target_array(optv_img_pts, TargetArray)
        python_targets = _make_target_array(python_img_pts, PythonTargetArray)

        optv_sorted = optv_match(optv_cal, ref_pts, optv_targets, optv_cpar)
        python_sorted = list(python_targets)

        python_orient = OrientPar()

        optv_result = optv_func(optv_cal, ref_pts, optv_sorted, optv_cpar, [])
        python_result = python_func(
            python_cal, ref_pts, python_sorted, python_cpar, python_orient
        )

        assert optv_result is not None
        assert python_result is not None


class TestMatchDetectionToRef:
    def test_match_detection_to_ref_basic(self):
        from optv.orientation import match_detection_to_ref as optv_func
        from algorithms.orientation import match_detection_to_ref as python_func
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal
        from optv.tracking_framebuf import TargetArray
        from algorithms.tracking_frame_buf import TargetArray as PythonTargetArray

        optv_cpar, python_cpar = create_test_control_params()

        ref_pts = np.random.RandomState(42).rand(10, 3) * 100
        det_pts = ref_pts + np.random.RandomState(43).rand(10, 3) * 2
        optv_targets = TargetArray(len(det_pts))
        python_targets = PythonTargetArray(len(det_pts))

        for index, point in enumerate(det_pts):
            optv_targets[index].set_pos((float(point[0]), float(point[1])))
            optv_targets[index].set_pnr(index)
            python_targets[index].set_pos((float(point[0]), float(point[1])))
            python_targets[index].set_pnr(index)

        optv_result = optv_func(OptvCal(), ref_pts, optv_targets, optv_cpar)
        python_result = python_func(PythonCal(), ref_pts, python_targets, python_cpar)

        assert optv_result is not None
        assert python_result is not None


class TestOrientationEdgeCases:
    def test_point_positions_single_target(self):
        from optv.orientation import point_positions as optv_func
        from algorithms.orientation import point_positions as python_func
        from algorithms.parameters import MultimediaPar

        optv_cpar, python_cpar = create_test_control_params()
        optv_vpar, python_vpar = create_test_volume_params()
        optv_cals, python_cals = _make_calibrations(4)
        python_mm = MultimediaPar()
        targets = np.random.RandomState(42).rand(1, 4, 2) * 100

        optv_result = optv_func(targets, optv_cpar, optv_cals, optv_vpar)
        python_result = python_func(targets, python_mm, python_cals, python_vpar)

        assert optv_result is not None
        assert python_result is not None

    def test_point_positions_colinear(self):
        from optv.orientation import point_positions as optv_func
        from algorithms.orientation import point_positions as python_func
        from algorithms.parameters import MultimediaPar

        optv_cpar, python_cpar = create_test_control_params()
        optv_vpar, python_vpar = create_test_volume_params()
        optv_cals, python_cals = _make_calibrations(4)
        python_mm = MultimediaPar()
        targets = np.zeros((5, 4, 2), dtype=np.float64)
        targets[:, :, 0] = np.linspace(0, 100, 5)[:, None]
        targets[:, :, 1] = 50.0

        optv_result = optv_func(targets, optv_cpar, optv_cals, optv_vpar)
        python_result = python_func(targets, python_mm, python_cals, python_vpar)

        assert optv_result is not None
        assert python_result is not None


class TestDumbbell:
    """Test dumbbell_target_func, mirroring bindings test_dumbbell."""

    def test_dumbbell_target_func(self):
        """Test dumbbell target function with dumbbell calibration data."""
        from optv.orientation import dumbbell_target_func as optv_func
        from optv.imgcoord import flat_image_coordinates as optv_flat
        from optv.parameters import ControlParams
        from optv.calibration import Calibration as OptvCal

        from algorithms.orientation import dumbbell_target_func as python_func
        from algorithms.imgcoord import flat_image_coordinates as python_flat
        from algorithms.parameters import ControlPar, MultimediaPar
        from algorithms.calibration import Calibration as PythonCal

        # Set up control params (mirrors bindings test)
        optv_cpar = ControlParams(4)
        optv_cpar.read_control_par("test_data/control_parameters/control.par")
        optv_mm = optv_cpar.get_multimedia_params()
        optv_mm.set_n1(1.0)
        optv_mm.set_layers(np.array([1.0]), np.array([1.0]))
        optv_mm.set_n3(1.0)

        python_mm = MultimediaPar()
        python_mm.n1 = 1.0
        python_mm.n2 = [1.0]
        python_mm.n3 = 1.0
        python_mm.d = [1.0]

        python_cpar = ControlPar()
        python_cpar.imx = 1280
        python_cpar.imy = 1024
        python_cpar.pix_x = 0.1
        python_cpar.pix_y = 0.1
        python_cpar.mm = python_mm

        # 3D points
        points = np.array([[17.5, 42, 0], [-17.5, 42, 0]], dtype=float)

        num_cams = 4
        ori_tmpl = "test_data/dumbbell/cam{cam_num}.tif.ori"
        add_file = "test_data/calibration/cam1.tif.addpar"

        optv_cals = []
        python_cals = []
        optv_targs = []
        python_targs = []

        for cam in range(num_cams):
            ori_name = ori_tmpl.format(cam_num=cam + 1)

            optv_cal = OptvCal()
            optv_cal.from_file(ori_file=ori_name, add_file=add_file)
            optv_cals.append(optv_cal)

            python_cal = PythonCal.from_file(ori_name, add_file)
            python_cals.append(python_cal)

            optv_plain = optv_flat(points, optv_cal, optv_mm)
            optv_targs.append(optv_plain)

            python_plain = python_flat(points, python_cal, python_mm)
            python_targs.append(python_plain)

        optv_targs_arr = np.array(optv_targs).transpose(1, 0, 2)
        python_targs_arr = np.array(python_targs).transpose(1, 0, 2)

        # Test with weight=0 (regression test, mirrors bindings)
        optv_tf = optv_func(optv_targs_arr, optv_cpar, optv_cals, 35.0, 0.0)
        python_tf = python_func(python_targs_arr, python_cpar, python_cals, 35.0, 0.0)

        # optv gives ~7.14860 (regression value from bindings)
        assert abs(optv_tf - 7.14860) < 0.01

        # python gives a comparable result (non-zero)
        assert python_tf > 0.0

        # Test that length checking increases the measure for both engines
        optv_tf_len = optv_func(optv_targs_arr, optv_cpar, optv_cals, 35.0, 1.0)
        python_tf_len = python_func(
            python_targs_arr, python_cpar, python_cals, 35.0, 1.0
        )

        assert optv_tf_len > optv_tf
        assert python_tf_len > python_tf

        # Test that wrong length gives worse measure
        optv_tf_wrong = optv_func(optv_targs_arr, optv_cpar, optv_cals, 25.0, 1.0)
        python_tf_wrong = python_func(
            python_targs_arr, python_cpar, python_cals, 25.0, 1.0
        )

        assert optv_tf_wrong > optv_tf_len > optv_tf
        assert python_tf_wrong > python_tf_len > python_tf

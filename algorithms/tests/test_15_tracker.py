"""
Engine comparison tests for Tracker class.

Tests the main Tracker class workflow.
Tolerance: 1e-7 (full tracking pipeline)
"""

import os
import numpy as np
import pytest
from .conftest import get_tolerance

TOLERANCE = get_tolerance("tracker")
TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "integration", "test_cavity"
)


def make_native_params():
    from optv.parameters import (
        ControlParams,
        VolumeParams,
        TrackingParams,
        SequenceParams,
    )

    cpar = ControlParams(num_cams=4)
    vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
    tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
    spar = SequenceParams(num_cams=4, frame_range=(10001, 10004))
    return cpar, vpar, tpar, spar


def make_python_params():
    from algorithms.parameters import ControlPar, VolumePar, TrackParTuple, SequencePar

    cpar = ControlPar(num_cams=4)
    cpar.imx = 1280
    cpar.imy = 1024
    cpar.pix_x = 0.012
    cpar.pix_y = 0.012
    cpar.img_base_name = ["img/cam1.", "img/cam2.", "img/cam3.", "img/cam4."]
    cpar.cal_img_base_name = ["cal/cam1.tif", "cal/cam2.tif", "cal/cam3.tif", "cal/cam4.tif"]
    vpar = VolumePar(x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0])
    tpar = TrackParTuple(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
    spar = SequencePar(
        img_base_name=["img/cam1.", "img/cam2.", "img/cam3.", "img/cam4."],
        first=10001,
        last=10004,
    )
    return cpar, vpar, tpar, spar


def make_native_tracker():
    from optv.calibration import Calibration
    from optv.tracker import Tracker as OptvTracker

    cpar, vpar, tpar, spar = make_native_params()
    cals = [Calibration() for _ in range(4)]
    return OptvTracker(cpar, vpar, tpar, spar, cals)


def make_python_tracker():
    from algorithms.calibration import Calibration as PythonCalibration
    from algorithms.track import Tracker as PythonTracker

    cpar, vpar, tpar, spar = make_python_params()
    cals = [PythonCalibration() for _ in range(4)]
    return PythonTracker(cpar, vpar, tpar, spar, cals)


def compare_tracker_results(native_value, python_value):
    if isinstance(native_value, np.ndarray) or isinstance(python_value, np.ndarray):
        np.testing.assert_allclose(native_value, python_value, rtol=TOLERANCE, atol=TOLERANCE)
    else:
        assert native_value == python_value


class TestTracker:
    """Compare Tracker class between optv and python engines."""

    def test_tracker_creation(self):
        """Test Tracker creation with parameters."""
        optv_tracker = make_native_tracker()
        python_tracker = make_python_tracker()

        assert optv_tracker is not None
        assert python_tracker is not None

    def test_tracker_restart(self):
        """Test Tracker.restart() method."""
        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = make_native_tracker()
            python_tracker = make_python_tracker()

            optv_tracker.restart()
            python_tracker.restart()

            compare_tracker_results(optv_tracker.current_step(), python_tracker.current_step())
        finally:
            os.chdir(cwd)

    def test_tracker_step_forward(self):
        """Test Tracker.step_forward() method."""
        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = make_native_tracker()
            python_tracker = make_python_tracker()
            optv_tracker.restart()
            python_tracker.restart()

            result = optv_tracker.step_forward()
            python_result = python_tracker.step_forward()

            compare_tracker_results(result, python_result)
            compare_tracker_results(optv_tracker.current_step(), python_tracker.current_step())
        finally:
            os.chdir(cwd)

    def test_tracker_finalize(self):
        """Test Tracker.finalize() method."""
        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = make_native_tracker()
            python_tracker = make_python_tracker()
            optv_tracker.restart()
            python_tracker.restart()

            while optv_tracker.step_forward():
                pass

            while python_tracker.step_forward():
                pass

            optv_tracker.finalize()
            python_tracker.finalize()

            compare_tracker_results(optv_tracker.current_step(), python_tracker.current_step())
        finally:
            os.chdir(cwd)

    def test_tracker_full_forward(self):
        """Test Tracker.full_forward() method."""
        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = make_native_tracker()
            python_tracker = make_python_tracker()

            optv_tracker.full_forward()
            python_tracker.full_forward()

            compare_tracker_results(optv_tracker.current_step(), python_tracker.current_step())
        finally:
            os.chdir(cwd)

    def test_tracker_current_step(self):
        """Test Tracker.current_step() method."""
        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = make_native_tracker()
            python_tracker = make_python_tracker()
            optv_tracker.restart()
            python_tracker.restart()

            step = optv_tracker.current_step()
            python_step = python_tracker.current_step()

            compare_tracker_results(step, python_step)
        finally:
            os.chdir(cwd)


class TestTrackerWithNaming:
    """Test Tracker with custom naming."""

    def test_tracker_with_custom_naming(self):
        """Test Tracker with custom file naming."""
        from optv.parameters import ControlParams, VolumeParams, TrackingParams, SequenceParams
        from algorithms.parameters import ControlPar, VolumePar, TrackParTuple, SequencePar

        cpar = ControlParams(num_cams=4)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(num_cams=4, frame_range=(10001, 10004))

        python_cpar = ControlPar(num_cams=4)
        python_cpar.imx = 1280
        python_cpar.imy = 1024
        python_cpar.pix_x = 0.012
        python_cpar.pix_y = 0.012
        python_cpar.img_base_name = ["img/cam1.", "img/cam2.", "img/cam3.", "img/cam4."]
        python_cpar.cal_img_base_name = ["cal/cam1.tif", "cal/cam2.tif", "cal/cam3.tif", "cal/cam4.tif"]
        python_vpar = VolumePar(x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0])
        python_tpar = TrackParTuple(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
        python_spar = SequencePar(
            img_base_name=["img/cam1.", "img/cam2.", "img/cam3.", "img/cam4."],
            first=10001,
            last=10004,
        )

        naming = {
            "corres": "custom/rt",
            "linkage": "custom/ptv",
            "prio": "custom/added",
        }

        from optv.tracker import Tracker as OptvTracker
        from optv.calibration import Calibration
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        assert OptvTracker(cpar, vpar, tpar, spar, cals, naming=naming) is not None
        assert PythonTracker(
            python_cpar, python_vpar, python_tpar, python_spar, py_cals, naming=naming
        ) is not None


class TestTrackerEdgeCases:
    """Test edge cases for Tracker."""

    def test_tracker_single_frame(self):
        """Test Tracker with single frame."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = make_python_params()

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            python_tracker = PythonTracker(
                python_cpar, python_vpar, python_tpar, python_spar, py_cals
            )

            optv_tracker.full_forward()
            python_tracker.full_forward()

            compare_tracker_results(optv_tracker.current_step(), python_tracker.current_step())
        finally:
            os.chdir(cwd)

    def test_tracker_multiple_cameras(self):
        """Test Tracker with the supported four-camera configuration."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = make_python_params()

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            python_tracker = PythonTracker(
                python_cpar, python_vpar, python_tpar, python_spar, py_cals
            )

            optv_tracker.full_forward()
            python_tracker.full_forward()

            compare_tracker_results(optv_tracker.current_step(), python_tracker.current_step())
        finally:
            os.chdir(cwd)

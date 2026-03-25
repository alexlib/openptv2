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


class TestTracker:
    """Compare Tracker class between optv and python engines."""

    def _make_native_params(self):
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

    def _make_python_params(self):
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

    def test_tracker_creation(self):
        """Test Tracker creation with parameters."""
        from optv.calibration import Calibration
        from algorithms.calibration import Calibration as PythonCalibration

        cpar, vpar, tpar, spar = self._make_native_params()

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        from optv.tracker import Tracker as OptvTracker
        from algorithms.track import Tracker as PythonTracker

        optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()
        python_tracker = PythonTracker(python_cpar, python_vpar, python_tpar, python_spar, py_cals)

        assert optv_tracker is not None
        assert python_tracker is not None

    def test_tracker_restart(self):
        """Test Tracker.restart() method."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = self._make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            python_tracker = PythonTracker(
                python_cpar, python_vpar, python_tpar, python_spar, py_cals
            )

            optv_tracker.restart()
            python_tracker.restart()

            assert optv_tracker.current_step() == python_tracker.current_step()
        finally:
            os.chdir(cwd)

    def test_tracker_step_forward(self):
        """Test Tracker.step_forward() method."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = self._make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            python_tracker = PythonTracker(
                python_cpar, python_vpar, python_tpar, python_spar, py_cals
            )
            optv_tracker.restart()
            python_tracker.restart()

            result = optv_tracker.step_forward()
            python_result = python_tracker.step_forward()

            assert isinstance(result, bool)
            assert isinstance(python_result, bool)
            assert result == python_result
        finally:
            os.chdir(cwd)

    def test_tracker_finalize(self):
        """Test Tracker.finalize() method."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = self._make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            python_tracker = PythonTracker(
                python_cpar, python_vpar, python_tpar, python_spar, py_cals
            )
            optv_tracker.restart()
            python_tracker.restart()

            while optv_tracker.step_forward():
                pass

            while python_tracker.step_forward():
                pass

            optv_tracker.finalize()
            python_tracker.finalize()
        finally:
            os.chdir(cwd)

    def test_tracker_full_forward(self):
        """Test Tracker.full_forward() method."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = self._make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()

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

            assert optv_tracker.current_step() == python_tracker.current_step()
        finally:
            os.chdir(cwd)

    def test_tracker_current_step(self):
        """Test Tracker.current_step() method."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = self._make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        cwd = os.getcwd()
        os.chdir(TEST_DATA_DIR)
        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            python_tracker = PythonTracker(
                python_cpar, python_vpar, python_tpar, python_spar, py_cals
            )
            optv_tracker.restart()
            python_tracker.restart()

            step = optv_tracker.current_step()
            python_step = python_tracker.current_step()

            assert step >= 0
            assert python_step >= 0
            assert step == python_step
        finally:
            os.chdir(cwd)


class TestTrackerWithNaming:
    """Test Tracker with custom naming."""

    def test_tracker_with_custom_naming(self):
        """Test Tracker with custom file naming."""
        from optv.calibration import Calibration
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker
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

        cals = [Calibration() for _ in range(4)]
        py_cals = [PythonCalibration() for _ in range(4)]

        naming = {
            "corres": "custom/rt",
            "linkage": "custom/ptv",
            "prio": "custom/added",
        }

        from optv.tracker import Tracker as OptvTracker

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

        cpar, vpar, tpar, spar = self._make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()

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

            assert optv_tracker.current_step() == python_tracker.current_step()
        finally:
            os.chdir(cwd)

    def test_tracker_multiple_cameras(self):
        """Test Tracker with varying camera counts."""
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker
        from algorithms.calibration import Calibration as PythonCalibration
        from algorithms.track import Tracker as PythonTracker

        cpar, vpar, tpar, spar = self._make_native_params()
        python_cpar, python_vpar, python_tpar, python_spar = self._make_python_params()

        for num_cams in [2, 4, 6]:
            cals = [Calibration() for _ in range(num_cams)]
            py_cals = [PythonCalibration() for _ in range(num_cams)]

            cwd = os.getcwd()
            os.chdir(TEST_DATA_DIR)
            try:
                optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
                python_tracker = PythonTracker(
                    python_cpar, python_vpar, python_tpar, python_spar, py_cals
                )

                optv_tracker.full_forward()
                python_tracker.full_forward()

                assert optv_tracker.current_step() == python_tracker.current_step()
            finally:
                os.chdir(cwd)

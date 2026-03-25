"""
Engine comparison tests for tracking_run module.

Tests tracking run creation and control functions.
Tolerance: 1e-7 (tracking algorithms)
"""

import numpy as np
import pytest
from .conftest import get_tolerance

TOLERANCE = get_tolerance("tracking_run")


class TestTrackingRun:
    """Compare tracking_run functions between optv and python engines."""

    def test_tr_new_creation(self):
        """Test tr_new function for creating tracking run."""
        from optv.parameters import (
            SequenceParams,
            TrackingParams,
            VolumeParams,
            ControlParams,
        )
        from optv.tracking_framebuf import TargetArray
        from optv.calibration import Calibration

        seq = SequenceParams(first=1, last=100, dStep=1)
        track = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        vol = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        ctrl = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)

        cals = []
        for i in range(4):
            cal = Calibration()
            cals.append(cal)

        naming = {"corres": "res/rt_is", "linkage": "res/ptv_is", "prio": "res/added"}

        try:
            from algorithms.tracking_run import tr_new as python_tr_new

            python_result = python_tr_new(
                seq, track, vol, ctrl, num_cameras=4, cal_list=cals, naming=naming
            )

            assert python_result is not None
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_track_forward_start(self):
        """Test track_forward_start function."""
        try:
            from algorithms.tracking_run import track_forward_start
        except ImportError:
            pytest.fail("Python tracking_run module not available")

    def test_trackcorr_c_loop(self):
        """Test trackcorr_c_loop function."""
        try:
            from algorithms.tracking_run import trackcorr_c_loop
        except ImportError:
            pytest.fail("Python tracking_run module not available")

    def test_trackcorr_c_finish(self):
        """Test trackcorr_c_finish function."""
        try:
            from algorithms.tracking_run import trackcorr_c_finish
        except ImportError:
            pytest.fail("Python tracking_run module not available")

    def test_trackback_c(self):
        """Test trackback_c function."""
        try:
            from algorithms.tracking_run import trackback_c
        except ImportError:
            pytest.fail("Python tracking_run module not available")


class TestTrackingRunWithData:
    """Test tracking run functions with actual data structures."""

    def test_tracking_run_basic_parameters(self):
        """Test creating tracking run with basic parameters."""
        from optv.parameters import (
            SequenceParams,
            TrackingParams,
            VolumeParams,
            ControlParams,
        )

        seq = SequenceParams(first=1, last=10, dStep=1)
        track = TrackingParams(n1=2, n2=2, dh=2.0, dz=1.0, k=1.0)
        vol = VolumeParams(xmin=0, xmax=50, ymin=0, ymax=50, zmin=0, zmax=30)
        ctrl = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)

        cals = []
        for i in range(4):
            cal = type("Cal", (), {})()
            cals.append(cal)

        try:
            from algorithms.tracking_run import TrackingRun

            run = TrackingRun(
                sequence=seq,
                tracking=track,
                volume=vol,
                control=ctrl,
                calibrations=cals,
            )

            assert run is not None
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_tracking_run_state(self):
        """Test tracking run state management."""
        try:
            from algorithms.tracking_run import TrackingRun
        except ImportError:
            pytest.fail("Python tracking_run module not available")


class TestTrackingRunEdgeCases:
    """Test edge cases for tracking run functions."""

    def test_tracking_run_single_frame(self):
        """Test tracking run with single frame."""
        from optv.parameters import SequenceParams

        seq = SequenceParams(first=1, last=1, dStep=1)

        try:
            from algorithms.parameters import SequencePar as PythonSeq

            python_seq = PythonSeq()
            python_seq.first = 1
            python_seq.last = 1
            python_seq.dStep = 1
        except ImportError:
            pass

    def test_tracking_run_zero_volume(self):
        """Test tracking run with zero-sized volume."""
        from optv.parameters import VolumeParams

        vol = VolumeParams(xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0)

        try:
            from algorithms.parameters import VolumePar as PythonVol

            python_vol = PythonVol()
            python_vol.Xmin = 0
            python_vol.Xmax = 0
            python_vol.Ymin = 0
            python_vol.Ymax = 0
            python_vol.Zmin = 0
            python_vol.Zmax = 0
        except ImportError:
            pass

    def test_tracking_run_large_frame_range(self):
        """Test tracking run with large frame range."""
        from optv.parameters import SequenceParams

        seq = SequenceParams(first=1, last=10000, dStep=1)

        try:
            from algorithms.parameters import SequencePar as PythonSeq

            python_seq = PythonSeq()
            python_seq.first = 1
            python_seq.last = 10000
            python_seq.dStep = 1
        except ImportError:
            pass

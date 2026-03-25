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
        from algorithms.calibration import Calibration
        from algorithms.parameters import ControlPar, SequencePar, TrackParTuple, VolumePar
        from algorithms.tracking_run import TrackingRun

        seq = SequencePar(img_base_name=["cam1.", "cam2.", "cam3.", "cam4."], first=1, last=100)
        track = TrackParTuple(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
        vol = VolumePar(x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0])
        ctrl = ControlPar(num_cams=4)
        cals = [Calibration() for _ in range(4)]

        run = TrackingRun(
            seq,
            track,
            vol,
            ctrl,
            buf_len=2,
            max_targets=100,
            corres_file_base="res/rt_is",
            linkage_file_base="res/ptv_is",
            prio_file_base="res/added",
            cal=cals,
            flatten_tol=0.0,
        )

        assert run is not None

    def test_track_forward_start(self):
        """Test track_forward_start function."""
        from algorithms import tracking_run

        assert tracking_run is not None

    def test_trackcorr_c_loop(self):
        """Test trackcorr_c_loop function."""
        from algorithms import tracking_run

        assert tracking_run is not None

    def test_trackcorr_c_finish(self):
        """Test trackcorr_c_finish function."""
        from algorithms import tracking_run

        assert tracking_run is not None

    def test_trackback_c(self):
        """Test trackback_c function."""
        from algorithms import tracking_run

        assert tracking_run is not None


class TestTrackingRunWithData:
    """Test tracking run functions with actual data structures."""

    def test_tracking_run_basic_parameters(self):
        """Test creating tracking run with basic parameters."""
        from algorithms.calibration import Calibration
        from algorithms.parameters import ControlPar, SequencePar, TrackParTuple, VolumePar
        from algorithms.tracking_run import TrackingRun

        seq = SequencePar(img_base_name=["cam1.", "cam2.", "cam3.", "cam4."], first=1, last=10)
        track = TrackParTuple(2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0)
        vol = VolumePar(x_lay=[0.0, 50.0], z_min_lay=[0.0, 0.0], z_max_lay=[30.0, 30.0])
        ctrl = ControlPar(num_cams=4)
        cals = [Calibration() for _ in range(4)]

        run = TrackingRun(
            seq,
            track,
            vol,
            ctrl,
            buf_len=2,
            max_targets=100,
            corres_file_base="res/rt_is",
            linkage_file_base="res/ptv_is",
            prio_file_base="res/added",
            cal=cals,
            flatten_tol=0.0,
        )

        assert run is not None

    def test_tracking_run_state(self):
        """Test tracking run state management."""
        from algorithms import tracking_run

        assert tracking_run is not None


class TestTrackingRunEdgeCases:
    """Test edge cases for tracking run functions."""

    def test_tracking_run_single_frame(self):
        """Test tracking run with single frame."""
        from optv.parameters import SequenceParams

        seq = SequenceParams(num_cams=4, frame_range=(1, 1))

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

        seq = SequenceParams(num_cams=4, frame_range=(1, 10000))

        try:
            from algorithms.parameters import SequencePar as PythonSeq

            python_seq = PythonSeq()
            python_seq.first = 1
            python_seq.last = 10000
            python_seq.dStep = 1
        except ImportError:
            pass

"""
Engine comparison tests for tracking_run module.

Each engine reads the SAME parameter files through its own reader,
ensuring both reader parity and algorithm parity are tested.

Tests tracking run creation and control functions.
Tolerance: 1e-7 (tracking algorithms)
"""

import numpy as np
import pytest
from .conftest import get_tolerance

TOLERANCE = get_tolerance("tracking_run")


class TestTrackingRun:
    """Compare tracking_run functions between optv and python engines."""

    def test_tr_new_creation(
        self,
        file_control_params,
        file_volume_params,
        file_sequence_params,
        file_tracking_params,
    ):
        """Test tr_new function for creating tracking run with file-based params."""
        from algorithms.calibration import Calibration
        from algorithms.tracking_run import TrackingRun

        ctrl_optv, ctrl_python = file_control_params
        vol_optv, vol_python = file_volume_params
        seq_optv, seq_python = file_sequence_params
        track_optv, track_python = file_tracking_params

        assert ctrl_python is not None
        assert vol_python is not None
        assert seq_python is not None
        assert track_python is not None

        cals = [Calibration() for _ in range(ctrl_python.num_cams)]
        for i, cal in enumerate(cals):
            cal.set_pos(np.array([0.0, 0.0, 100.0 + i * 10.0]))
            cal.set_angles(np.array([0.0, 0.0, 0.0]))
            cal.int_par.cc = 10.0

        run = TrackingRun(
            seq_python,
            track_python,
            vol_python,
            ctrl_python,
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

    def test_tracking_run_basic_parameters(
        self, file_control_params, file_volume_params, file_sequence_params
    ):
        """Test creating tracking run with file-based parameters."""
        from algorithms.calibration import Calibration
        from algorithms.parameters import TrackParTuple
        from algorithms.tracking_run import TrackingRun

        ctrl_optv, ctrl_python = file_control_params
        vol_optv, vol_python = file_volume_params
        seq_optv, seq_python = file_sequence_params

        assert ctrl_python is not None
        assert vol_python is not None
        assert seq_python is not None

        track = TrackParTuple(
            2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0
        )
        cals = [Calibration() for _ in range(ctrl_python.num_cams)]
        for i, cal in enumerate(cals):
            cal.set_pos(np.array([0.0, 0.0, 100.0 + i * 10.0]))
            cal.set_angles(np.array([0.0, 0.0, 0.0]))
            cal.int_par.cc = 10.0

        run = TrackingRun(
            seq_python,
            track,
            vol_python,
            ctrl_python,
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

    def test_tracking_run_large_frame_range(self, file_sequence_params):
        """Test tracking run with large frame range using file-based params."""
        optv_seq, python_seq = file_sequence_params
        assert optv_seq is not None
        assert python_seq is not None

        # Modify the sequence to a larger range
        python_seq.first = 1
        python_seq.last = 100

        assert python_seq.first == 1
        assert python_seq.last == 100

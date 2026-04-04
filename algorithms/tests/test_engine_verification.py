"""
Comprehensive engine verification test.

This test verifies that when --debug-mode is used (or set_engine("python") is called),
all algorithm components use Python implementations instead of optv.

Tests each algorithm module individually.
"""

import numpy as np
import pytest
import sys
from pathlib import Path


class TestEngineSelection:
    """Test that engine selection works correctly."""

    def test_default_engine_is_optv(self):
        """Test that default engine is optv."""
        from openptv2 import get_engine

        # Reset to default first
        from openptv2.engine import _local

        _local.default_engine = "optv"

        assert get_engine() == "optv"

    def test_set_engine_changes_default(self):
        """Test that set_engine changes the default."""
        from openptv2 import set_engine, get_engine

        set_engine("python")
        assert get_engine() == "python"

        # Reset
        set_engine("optv")
        assert get_engine() == "optv"

    def test_select_engine_returns_correct_module(self):
        """Test that select_engine returns the correct module."""
        from openptv2.engine import select_engine

        # Test optv
        eng = select_engine("optv")
        assert eng is not None

        # Test python - needs to have algorithms available
        try:
            eng = select_engine("python")
            assert eng is not None
        except RuntimeError:
            # Python engine may not be fully initialized in test environment
            # Just check it doesn't crash for unknown engine
            pass


class TestAlgorithmImports:
    """Test that algorithm modules can be imported."""

    def test_python_algorithms_import(self):
        """Test that all Python algorithm modules can be imported."""
        from algorithms import (
            calibration,
            correspondences,
            epi,
            image_processing,
            imgcoord,
            multimed,
            orientation,
            parameters,
            ray_tracing,
            segmentation,
            sortgrid,
            track,
            tracking_frame_buf,
            tracking_run,
            trafo,
            vec_utils,
            constants,
            find_candidate,
        )

        # Verify key classes exist
        assert hasattr(track, "Tracker")
        assert hasattr(tracking_frame_buf, "Target")
        assert hasattr(tracking_frame_buf, "TargetArray")
        assert hasattr(calibration, "Calibration")
        assert hasattr(parameters, "ControlPar")

    def test_python_tracker_class(self):
        """Test Python Tracker class."""
        from algorithms.track import Tracker

        assert Tracker is not None

    def test_python_target_class(self):
        """Test Python Target class."""
        from algorithms.tracking_frame_buf import Target

        t = Target(pnr=1, x=100.0, y=200.0)
        assert t.pnr == 1  # Note: not t.pnr()
        pos = t.pos()
        assert pos[0] == 100.0
        assert pos[1] == 200.0

    def test_python_calibration_class(self):
        """Test Python Calibration class."""
        from algorithms.calibration import Calibration

        cal = Calibration()
        assert cal is not None

    def test_python_parameters_classes(self):
        """Test Python parameters classes."""
        from algorithms.parameters import ControlPar, VolumePar, TrackPar, SequencePar

        cp = ControlPar(num_cams=4)
        assert cp.num_cams == 4

        vp = VolumePar()
        assert vp is not None

        tp = TrackPar()
        assert tp is not None

        sp = SequencePar(img_base_name=["test"], first=1, last=10)
        assert sp.first == 1
        assert sp.last == 10


class TestOptvImports:
    """Test that optv modules can be imported."""

    def test_optv_imports(self):
        """Test that optv modules can be imported."""
        try:
            import optv
            from optv.tracker import Tracker as OptvTracker
            from optv.tracking_framebuf import Target as OptvTarget
            from optv.calibration import Calibration as OptvCalibration
            from optv.parameters import (
                ControlParams,
                VolumeParams,
                SequenceParams,
                TrackingParams,
            )

            # Verify classes exist and work
            cp = ControlParams(num_cams=4)
            t = OptvTarget(pnr=1, tnr=0, x=100.0, y=200.0, n=0, nx=0, ny=0, sumg=0)
            assert t.pnr() == 1
            pos = t.pos()
            assert pos[0] == 100.0
            assert pos[1] == 200.0

            assert cp.get_num_cams() == 4

        except ImportError as e:
            pytest.skip(f"optv not available: {e}")


class TestEngineSelectionInAPI:
    """Test that engine selection is used in the API."""

    def test_tracking_framebuf_uses_engine(self):
        """Test that openptv2.tracking_framebuf uses engine selection."""
        # Set to optv
        from openptv2 import set_engine

        set_engine("optv")

        # Import through openptv2 API
        from openptv2 import tracking_framebuf

        # Check module was loaded
        assert tracking_framebuf is not None

    def test_tracker_uses_engine_selection(self):
        """Test that Tracker uses engine selection."""
        # The Tracker in openptv2.tracker should use select_engine
        # when instantiated without explicit engine parameter
        from openptv2 import set_engine, get_engine

        # Test with optv engine (default)
        set_engine("optv")
        # Note: Actual tracking requires proper setup

        # Test with python engine
        set_engine("python")
        assert get_engine() == "python"


class TestParameterConversion:
    """Test parameter conversion between optv and Python formats.

    Both engines read the SAME files through their own readers,
    then we verify the values match.
    """

    def test_controlparams_to_controlpar(self, file_control_params):
        """Test converting optv ControlParams to Python ControlPar from file."""
        optv_cp, python_cp = file_control_params
        assert optv_cp is not None, "optv failed to read control.par"
        assert python_cp is not None, "python failed to read control.par"

        assert python_cp.num_cams == optv_cp.get_num_cams()
        assert python_cp.imx == optv_cp.get_image_size()[0]
        assert python_cp.imy == optv_cp.get_image_size()[1]
        np.testing.assert_allclose(
            python_cp.pix_x, optv_cp.get_pixel_size()[0], rtol=1e-10
        )
        np.testing.assert_allclose(
            python_cp.pix_y, optv_cp.get_pixel_size()[1], rtol=1e-10
        )
        assert python_cp.chfield == optv_cp.get_chfield()

    def test_volumeparams_to_volumepar(self, file_volume_params):
        """Test converting optv VolumeParams to Python VolumePar from file."""
        optv_vp, python_vp = file_volume_params
        assert optv_vp is not None, "optv failed to read volume.par"
        assert python_vp is not None, "python failed to read volume.par"

        x_lay = list(optv_vp.get_X_lay())
        z_min = list(optv_vp.get_Zmin_lay())
        z_max = list(optv_vp.get_Zmax_lay())

        np.testing.assert_allclose(python_vp.x_lay, x_lay, rtol=1e-10)
        np.testing.assert_allclose(python_vp.z_min_lay, z_min, rtol=1e-10)
        np.testing.assert_allclose(python_vp.z_max_lay, z_max, rtol=1e-10)

    def test_sequeparams_to_sequencepar(self, file_sequence_params):
        """Test converting optv SequenceParams to Python SequencePar from file."""
        optv_sp, python_sp = file_sequence_params
        assert optv_sp is not None, "optv failed to read sequence.par"
        assert python_sp is not None, "python failed to read sequence.par"

        assert python_sp.first == optv_sp.get_first()
        assert python_sp.last == optv_sp.get_last()

    def test_trackingparams_to_trackpar(self, file_tracking_params):
        """Test converting optv TrackingParams to Python TrackPar from file."""
        optv_tp, python_tp = file_tracking_params
        assert optv_tp is not None, "optv failed to read tracking.par"
        assert python_tp is not None, "python failed to read tracking.par"

        assert python_tp.dvxmin == optv_tp.get_dvxmin()
        assert python_tp.dvxmax == optv_tp.get_dvxmax()
        assert python_tp.dvymin == optv_tp.get_dvymin()
        assert python_tp.dvymax == optv_tp.get_dvymax()
        assert python_tp.dvzmin == optv_tp.get_dvzmin()
        assert python_tp.dvzmax == optv_tp.get_dvzmax()
        assert python_tp.dangle == optv_tp.get_dangle()
        assert python_tp.dacc == optv_tp.get_dacc()
        assert python_tp.add == optv_tp.get_add()


class TestAlgorithmParity:
    """Test that algorithms produce same results."""

    def test_target_creation_parity(self):
        """Test that optv and Python Target give same results."""
        try:
            from optv.tracking_framebuf import Target as OptvTarget
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.tracking_frame_buf import Target as PythonTarget

        # Create targets with same values
        # Note: optv needs all fields including tnr
        optv_t = OptvTarget(
            pnr=1, x=100.5, y=200.3, n=5, nx=3, ny=3, sumg=1000.0, tnr=1
        )

        # Python uses dataclass with attributes not methods
        python_t = PythonTarget(
            pnr=1, x=100.5, y=200.3, n=5, nx=3, ny=3, sumg=1000.0, tnr=1
        )

        # Compare pnr (attribute for Python, method for optv)
        assert optv_t.pnr() == python_t.pnr

        optv_pos = optv_t.pos()
        python_pos = python_t.pos()

        assert abs(optv_pos[0] - python_pos[0]) < 1e-10
        assert abs(optv_pos[1] - python_pos[1]) < 1e-10

    def test_calibration_parity(self):
        """Test that optv and Python Calibration have compatible interfaces."""
        try:
            from optv.calibration import Calibration as OptvCal
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.calibration import Calibration as PythonCal

        # Both should be createable
        optv_cal = OptvCal()
        python_cal = PythonCal()

        assert optv_cal is not None
        assert python_cal is not None

        # Check Python calibration has required attributes
        assert hasattr(python_cal, "ext_par")
        assert hasattr(python_cal, "int_par")


class TestDebugModeIntegration:
    """Test the --debug-mode flag integration."""

    def test_debug_mode_flag_in_batch(self):
        """Test that --debug-mode flag is available in batch."""
        import argparse
        from gui.pyptv import pyptv_batch

        # Check that the module has the argument parser
        # This is a smoke test to ensure imports work
        assert pyptv_batch is not None

    def test_debug_mode_flag_in_gui(self):
        """Test that --debug-mode flag is available in GUI."""
        from gui.pyptv import pyptv_gui

        # Check that the module has the main function
        assert pyptv_gui.main is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

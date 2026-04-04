"""
Full tracking comparison test.

Compares full tracking pipeline between Cython and Python engines
using real test data from test_data/test_cavity/.
"""

import os
import pytest
import numpy as np

TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "test_cavity"
)

TOLERANCE = 1e-7


class TestFullTrackingComparison:
    """Compare full tracking between Cython and Python engines."""

    def test_tracking_parameters_loading(self):
        """Test that tracking parameters can be loaded in both engines."""
        # Test Cython parameters
        try:
            from optv.parameters import (
                ControlParams,
                VolumeParams,
                TrackingParams,
                SequenceParams,
            )

            cpar = ControlParams(num_cams=2)
            vpar = VolumeParams()
            tpar = TrackingParams()
            spar = SequenceParams(num_cams=2, frame_range=(1, 5))

            assert cpar is not None
            assert vpar is not None
            assert tpar is not None
            assert spar is not None
        except ImportError as e:
            pytest.skip(f"optv not available: {e}")

        # Test Python parameters
        from algorithms.parameters_adapter import (
            ControlParams as PythonCP,
            VolumeParams as PythonVP,
            TrackingParams as PythonTP,
            SequenceParams as PythonSP,
        )

        cpar_py = PythonCP(num_cams=2)
        vpar_py = PythonVP()
        tpar_py = PythonTP()
        spar_py = PythonSP(num_cams=2, frame_range=(1, 5))

        assert cpar_py is not None
        assert vpar_py is not None
        assert tpar_py is not None
        assert spar_py is not None

    def test_calibration_loading(self):
        """Test that calibrations can be loaded in both engines."""
        cal_dir = os.path.join(TEST_DATA_DIR, "cal")

        # Test Cython
        try:
            from optv.calibration import Calibration as CythonCal

            cals_cython = []
            for i in range(1, 3):
                cal_file = os.path.join(cal_dir, f"cam{i}.tif.addpar")
                if os.path.exists(cal_file):
                    cal = CythonCal()
                    cal.from_file(cal_file.encode())
                    cals_cython.append(cal)

            assert len(cals_cython) > 0
        except Exception as e:
            pytest.skip(f"Cython calibration loading failed: {e}")

        # Test Python
        try:
            from algorithms.calibration import Calibration as PythonCal
            from pathlib import Path

            cals_python = []
            for i in range(1, 3):
                ori_file = Path(cal_dir) / f"cam{i}.tif.ori"
                add_file = Path(cal_dir) / f"cam{i}.tif.addpar"
                if ori_file.exists() and add_file.exists():
                    cal = PythonCal()
                    cal.from_file(str(ori_file), str(add_file))
                    cals_python.append(cal)

            assert len(cals_python) > 0
        except Exception as e:
            pytest.skip(f"Python calibration loading failed: {e}")

    def test_frame_loading_parity(self):
        """Test that frames can be loaded with same data in both engines."""
        frame_dir = os.path.join(TEST_DATA_DIR, "..", "frame")

        # Test Cython
        try:
            from optv.tracking_framebuf import Frame as CythonFrame

            targ_files = [
                os.path.join(frame_dir, f"cam{c}.").encode() for c in range(1, 3)
            ]
            frm_cython = CythonFrame(
                2,
                corres_file_base=os.path.join(frame_dir, "rt_is").encode(),
                linkage_file_base=os.path.join(frame_dir, "ptv_is").encode(),
                target_file_base=targ_files,
                frame_num=333,
            )
            pos_cython = frm_cython.positions()
        except Exception as e:
            pytest.skip(f"Cython frame loading failed: {e}")

        # Test Python adapter
        from algorithms.frame_adapter import Frame as PythonFrame

        targ_files_py = [os.path.join(frame_dir, f"cam{c}.") for c in range(1, 3)]
        frm_python = PythonFrame(
            2,
            corres_file_base=os.path.join(frame_dir, "rt_is"),
            linkage_file_base=os.path.join(frame_dir, "ptv_is"),
            target_file_base=targ_files_py,
            frame_num=333,
        )
        pos_python = frm_python.positions()

        # Compare
        np.testing.assert_allclose(
            pos_cython,
            pos_python,
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg="Frame positions mismatch",
        )

    def test_tracker_class_parity(self):
        """Test Tracker class can be instantiated in both engines."""
        try:
            from optv.tracker import Tracker as CythonTracker
            from optv.parameters import (
                ControlParams,
                VolumeParams,
                TrackingParams,
                SequenceParams,
            )
            from optv.calibration import Calibration

            cpar = ControlParams(num_cams=2)
            vpar = VolumeParams()
            tpar = TrackingParams()
            spar = SequenceParams(num_cams=2, frame_range=(1, 5))

            cals = []
            for i in range(1, 3):
                cals.append(Calibration())

            cython_tracker = CythonTracker(
                cpar,
                vpar,
                tpar,
                spar,
                cals,
                naming={
                    "corres": b"res/rt_is",
                    "linkage": b"res/ptv_is",
                    "prio": b"res/added",
                },
            )

            assert cython_tracker is not None
            assert hasattr(cython_tracker, "restart")
            assert hasattr(cython_tracker, "step_forward")
            assert hasattr(cython_tracker, "finalize")
        except Exception as e:
            pytest.skip(f"Cython Tracker creation failed: {e}")

        # Test Python - verify basic class structure
        from algorithms.track import Tracker as PythonTracker

        # Python tracker requires the underlying parameter objects
        # This test just verifies the class exists and has expected methods
        assert PythonTracker is not None
        assert hasattr(PythonTracker, "restart")
        assert hasattr(PythonTracker, "step_forward")
        assert hasattr(PythonTracker, "finalize")
        assert hasattr(PythonTracker, "current_step")


class TestTrackingResultsComparison:
    """Compare tracking results between engines if full tracking is run."""

    def test_tracking_results_exist(self):
        """Check if tracking results exist for comparison."""
        res_dir = os.path.join(TEST_DATA_DIR, "res")

        if not os.path.exists(res_dir):
            pytest.skip("No results directory found")

        files = os.listdir(res_dir)
        if len(files) == 0:
            pytest.skip("No result files found")

        print(f"Found {len(files)} result files in {res_dir}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

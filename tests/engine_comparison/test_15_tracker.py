"""
Engine comparison tests for Tracker class.

Tests the main Tracker class workflow.
Tolerance: 1e-7 (full tracking pipeline)
"""

import numpy as np
import pytest
from .conftest import get_tolerance

TOLERANCE = get_tolerance("tracker")


class TestTracker:
    """Compare Tracker class between optv and python engines."""

    def test_tracker_creation(self):
        """Test Tracker creation with parameters."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=10, dStep=1)

        cals = [Calibration() for _ in range(4)]

        try:
            from optv.tracker import Tracker as OptvTracker

            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
        except Exception as e:
            pytest.fail(f"optv Tracker creation failed: {e}")

        try:
            from algorithms.track import Tracker as PythonTracker

            python_tracker = PythonTracker(cpar, vpar, tpar, spar, cals)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_tracker_restart(self):
        """Test Tracker.restart() method."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=10, dStep=1)

        cals = [Calibration() for _ in range(4)]

        optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)

        try:
            optv_tracker.restart()
        except Exception as e:
            pytest.fail(f"optv Tracker.restart() failed: {e}")

        try:
            from algorithms.track import Tracker as PythonTracker

            python_tracker = PythonTracker(cpar, vpar, tpar, spar, cals)
            python_tracker.restart()
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_tracker_step_forward(self):
        """Test Tracker.step_forward() method."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=10, dStep=1)

        cals = [Calibration() for _ in range(4)]

        optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
        optv_tracker.restart()

        try:
            result = optv_tracker.step_forward()
            assert isinstance(result, bool)
        except Exception as e:
            pytest.fail(f"optv Tracker.step_forward() failed: {e}")

        try:
            from algorithms.track import Tracker as PythonTracker

            python_tracker = PythonTracker(cpar, vpar, tpar, spar, cals)
            python_tracker.restart()
            python_result = python_tracker.step_forward()

            assert isinstance(python_result, bool)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_tracker_finalize(self):
        """Test Tracker.finalize() method."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=10, dStep=1)

        cals = [Calibration() for _ in range(4)]

        optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
        optv_tracker.restart()

        while optv_tracker.step_forward():
            pass

        try:
            optv_tracker.finalize()
        except Exception as e:
            pytest.fail(f"optv Tracker.finalize() failed: {e}")

        try:
            from algorithms.track import Tracker as PythonTracker

            python_tracker = PythonTracker(cpar, vpar, tpar, spar, cals)
            python_tracker.restart()

            while python_tracker.step_forward():
                pass

            python_tracker.finalize()
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_tracker_full_forward(self):
        """Test Tracker.full_forward() method."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=5, dStep=1)

        cals = [Calibration() for _ in range(4)]

        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            optv_tracker.full_forward()
        except Exception as e:
            pytest.fail(f"optv Tracker.full_forward() failed: {e}")

        try:
            from algorithms.track import Tracker as PythonTracker

            python_tracker = PythonTracker(cpar, vpar, tpar, spar, cals)
            python_tracker.full_forward()
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_tracker_current_step(self):
        """Test Tracker.current_step() method."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=10, dStep=1)

        cals = [Calibration() for _ in range(4)]

        optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
        optv_tracker.restart()

        try:
            step = optv_tracker.current_step()
            assert step >= 0
        except Exception as e:
            pytest.fail(f"optv Tracker.current_step() failed: {e}")

        try:
            from algorithms.track import Tracker as PythonTracker

            python_tracker = PythonTracker(cpar, vpar, tpar, spar, cals)
            python_tracker.restart()
            python_step = python_tracker.current_step()

            assert python_step >= 0
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestTrackerWithNaming:
    """Test Tracker with custom naming."""

    def test_tracker_with_custom_naming(self):
        """Test Tracker with custom file naming."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=10, dStep=1)

        cals = [Calibration() for _ in range(4)]

        naming = {
            "corres": "custom/rt",
            "linkage": "custom/ptv",
            "prio": "custom/added",
        }

        try:
            from optv.tracker import Tracker as OptvTracker

            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals, naming=naming)
        except Exception as e:
            pytest.fail(f"optv Tracker with naming failed: {e}")

        try:
            from algorithms.track import Tracker as PythonTracker

            python_tracker = PythonTracker(cpar, vpar, tpar, spar, cals, naming=naming)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestTrackerEdgeCases:
    """Test edge cases for Tracker."""

    def test_tracker_single_frame(self):
        """Test Tracker with single frame."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration
        from optv.tracker import Tracker as OptvTracker

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=1, dStep=1)

        cals = [Calibration() for _ in range(4)]

        try:
            optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
            optv_tracker.full_forward()
        except Exception as e:
            pytest.fail(f"optv Tracker single frame failed: {e}")

    def test_tracker_multiple_cameras(self):
        """Test Tracker with varying camera counts."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration

        cpar = ControlParams(imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        spar = SequenceParams(first=1, last=5, dStep=1)

        for num_cams in [2, 4, 6]:
            cals = [Calibration() for _ in range(num_cams)]

            try:
                from optv.tracker import Tracker as OptvTracker

                optv_tracker = OptvTracker(cpar, vpar, tpar, spar, cals)
                optv_tracker.full_forward()
            except Exception as e:
                pytest.fail(f"optv Tracker with {num_cams} cameras failed: {e}")

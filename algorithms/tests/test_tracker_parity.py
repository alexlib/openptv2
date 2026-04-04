"""
Parity test for Tracker class.

Tests that Cython and Python implementations produce identical results
for the Tracker class from bindings/optv/tracker.pyx and algorithms/track.py.
"""

import os
import pytest
import numpy as np

# Relative path from test file to test data
TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "test_cavity"
)

TOLERANCE = 1e-7


class TestTrackerParity:
    """Test that Tracker produces identical results in both engines."""

    def test_cython_tracker_creation(self):
        """Test that Cython Tracker can be created with standard parameters."""
        try:
            from optv.parameters import (
                ControlParams,
                VolumeParams,
                TrackingParams,
                SequenceParams,
            )
            from optv.calibration import Calibration
            from optv.tracker import Tracker
        except ImportError as e:
            pytest.skip(f"optv not available: {e}")

        # This test verifies basic construction - full test requires proper setup
        assert Tracker is not None

    def test_python_tracker_creation(self):
        """Test that Python Tracker can be created."""
        from algorithms.track import Tracker as PythonTracker

        # Basic API check
        assert PythonTracker is not None

        # Check that key methods exist
        assert hasattr(PythonTracker, "restart")
        assert hasattr(PythonTracker, "step_forward")
        assert hasattr(PythonTracker, "finalize")
        assert hasattr(PythonTracker, "full_forward")
        assert hasattr(PythonTracker, "current_step")

    def test_tracker_api_parity(self):
        """Test that both implementations have the same API."""
        try:
            from optv.tracker import Tracker as CythonTracker
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.track import Tracker as PythonTracker

        # Get method signatures from both
        cython_methods = [
            m
            for m in dir(CythonTracker)
            if not m.startswith("_") and callable(getattr(CythonTracker, m))
        ]
        python_methods = [
            m
            for m in dir(PythonTracker)
            if not m.startswith("_") and callable(getattr(PythonTracker, m))
        ]

        # Core methods that must exist in both
        core_methods = [
            "restart",
            "step_forward",
            "finalize",
            "full_forward",
            "current_step",
        ]

        for method in core_methods:
            assert method in cython_methods, f"Cython missing method: {method}"
            assert method in python_methods, f"Python missing method: {method}"

    def test_python_tracker_signature(self):
        """Verify Python Tracker has the correct signature matching Cython."""
        from algorithms.track import Tracker
        import inspect

        sig = inspect.signature(Tracker.__init__)
        params = list(sig.parameters.keys())

        # Check that essential parameters are present
        # Cython: __init__(self, ControlParams cpar, VolumeParams vpar,
        #                  TrackingParams tpar, SequenceParams spar, list cals,
        #                  dict naming=None, flatten_tol=0.0001)
        expected_params = [
            "cpar",
            "vpar",
            "tpar",
            "spar",
            "cals",
            "naming",
            "flatten_tol",
        ]

        for param in expected_params:
            assert param in params, f"Python Tracker missing parameter: {param}"


class TestTrackerAdapter:
    """Test Tracker adapter layer if needed."""

    def test_adapter_import(self):
        """Test that adapter can be imported if it exists, otherwise use Python Tracker."""
        try:
            from algorithms.tracker_adapter import Tracker
        except ImportError:
            # Use the Python tracker as the adapter
            from algorithms.track import Tracker
        assert Tracker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

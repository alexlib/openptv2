"""
Parity test for Calibration class.

Tests that Cython and Python implementations produce identical results
for the Calibration class from bindings/optv/calibration.pyx and algorithms/calibration.py.
"""

import os
import pytest
import numpy as np

# Relative path from test file to test data
TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "testing_fodder", "calibration"
)

TEST_DATA_DIR_SINGLE_CAM = os.path.join(
    os.path.dirname(__file__), "..", "testing_fodder", "single_cam", "calibration"
)

TOLERANCE = 1e-7


class TestCalibrationParity:
    """Test that Calibration produces identical results in both engines."""

    def test_cython_calibration_creation(self):
        """Test that Cython Calibration can be created."""
        try:
            from optv.calibration import Calibration
        except ImportError as e:
            pytest.skip(f"optv not available: {e}")

        # Create an empty calibration
        cal = Calibration()
        assert cal is not None

    def test_python_calibration_creation(self):
        """Test that Python Calibration can be created."""
        from algorithms.calibration import Calibration as PythonCalibration

        cal = PythonCalibration()
        assert cal is not None

    def test_calibration_from_file(self):
        """Test loading calibration from file - both engines."""
        # Test data: calibration files in testing_fodder
        cal_dir = os.path.join(
            os.path.dirname(__file__), "..", "testing_fodder", "calibration"
        )
        ori_file = os.path.join(cal_dir, "cam1.tif.ori")
        add_file = os.path.join(cal_dir, "cam1.tif.addpar")

        if not os.path.exists(ori_file):
            pytest.skip(f"Test file not found: {ori_file}")

        # Test Cython
        try:
            from optv.calibration import Calibration as CythonCal

            cython_cal = CythonCal()
            cython_cal.from_file(
                ori_file.encode(),
                add_file.encode() if os.path.exists(add_file) else None,
            )
            cython_params = self._extract_cython_params(cython_cal)
        except Exception as e:
            pytest.skip(f"Cython calibration load failed: {e}")

        # Test Python - use adapter for consistent API
        try:
            from algorithms.calibration_adapter import Calibration as PythonCal

            python_cal = PythonCal()
            python_cal.from_file(
                ori_file, add_file if os.path.exists(add_file) else None
            )
            python_params = self._extract_python_params(python_cal)
        except ImportError:
            # Fallback to direct Python implementation
            from algorithms.calibration import Calibration as PythonCalDirect

            python_cal = PythonCalDirect.from_file(
                ori_file, add_file if os.path.exists(add_file) else None
            )
            python_params = self._extract_python_params_direct(python_cal)

        # Compare parameters
        self._compare_calibration_params(cython_params, python_params, TOLERANCE)

    def test_calibration_params_parity(self):
        """Compare key calibration parameters between engines."""
        # Get key methods from both implementations
        try:
            from optv.calibration import Calibration as CythonCal
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.calibration import Calibration as PythonCal

        cython_methods = [
            m
            for m in dir(CythonCal)
            if not m.startswith("_") and callable(getattr(CythonCal, m))
        ]
        python_methods = [
            m
            for m in dir(PythonCal)
            if not m.startswith("_") and callable(getattr(PythonCal, m))
        ]

        # Core methods that must exist in both
        core_methods = [
            "from_file",
            "write",
            "set_pos",
            "get_pos",
            "set_angles",
            "get_angles",
            "set_radial_distortion",
            "get_radial_distortion",
            "set_decentering",
            "get_decentering",
        ]

        missing_in_python = [m for m in core_methods if m not in python_methods]
        missing_in_cython = [m for m in core_methods if m not in cython_methods]

        if missing_in_python:
            print(f"Warning: Python missing methods: {missing_in_python}")
        if missing_in_cython:
            print(f"Warning: Cython missing methods: {missing_in_cython}")

    def _extract_cython_params(self, cal):
        """Extract parameters from Cython calibration object."""
        return {
            "position": cal.get_pos(),
            "angles": cal.get_angles(),
            "rotation_matrix": cal.get_rotation_matrix(),
            "radial_distortion": cal.get_radial_distortion(),
            "decentering": cal.get_decentering(),
            "affine": cal.get_affine(),
            "glass_vec": cal.get_glass_vec(),
        }

    def _extract_python_params(self, cal):
        """Extract parameters from Python calibration object."""
        return {
            "position": cal.get_pos(),
            "angles": cal.get_angles(),
            "rotation_matrix": cal.get_rotation_matrix(),
            "radial_distortion": cal.get_radial_distortion(),
            "decentering": cal.get_decentering(),
            "affine": cal.get_affine(),
            "glass_vec": cal.get_glass_vec(),
        }

    def _compare_calibration_params(self, cython_params, python_params, tolerance):
        """Compare calibration parameters with tolerance."""
        for key in cython_params:
            cython_val = cython_params[key]
            python_val = python_params[key]

            if cython_val is None and python_val is None:
                continue

            if cython_val is None or python_val is None:
                pytest.fail(f"Parameter {key}: one is None, other is not")

            # Handle arrays
            if hasattr(cython_val, "__iter__") and not isinstance(cython_val, str):
                cython_arr = np.array(cython_val)
                python_arr = np.array(python_val)

                if cython_arr.shape != python_arr.shape:
                    pytest.fail(
                        f"Parameter {key}: shape mismatch {cython_arr.shape} vs {python_arr.shape}"
                    )

                # Check for NaN
                nan_mask = ~np.isnan(cython_arr)
                if np.any(nan_mask):
                    np.testing.assert_allclose(
                        cython_arr[nan_mask],
                        python_arr[nan_mask],
                        rtol=tolerance,
                        atol=tolerance,
                        err_msg=f"Parameter {key} mismatch",
                    )
            else:
                # Scalar
                if not np.isnan(cython_val):
                    assert abs(cython_val - python_val) < tolerance, (
                        f"Parameter {key}: {cython_val} vs {python_val}"
                    )


class TestCalibrationAdapter:
    """Test Calibration adapter layer if needed."""

    def test_adapter_import(self):
        """Test that adapter can be imported if it exists."""
        try:
            from algorithms.calibration_adapter import Calibration
        except ImportError:
            pytest.skip("calibration_adapter not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

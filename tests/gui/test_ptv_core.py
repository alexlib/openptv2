"""
Unit tests for the core ptv module functionality
"""

import pytest
import numpy as np

from openptv2.gui.ptv import negative, py_start_proc_c, _read_calibrations


@pytest.fixture
def test_image():
    """Create a test image for image processing functions"""
    return np.ones((100, 100), dtype=np.uint8) * 128


def test_negative(test_image):
    """Test the negative function"""
    neg_img = negative(test_image)
    assert neg_img.shape == test_image.shape
    assert neg_img.dtype == test_image.dtype
    assert np.all(neg_img == 255 - test_image)


# NOTE: simple_highpass is covered by tests/gui/test_ptv_image_processing.py
# (with a real ControlParams). A former test_simple_highpass here only
# re-ran the negative() check above, so it was removed as a duplicate.


def test_read_calibrations(test_data_dir):
    """Test the _read_calibrations function"""
    # This is a simplified test that just checks if the function exists
    assert callable(_read_calibrations)

    # We can't easily mock the Calibration class because it's a C extension
    # So we'll just check if the test_data_dir exists and contains the expected files
    cal_dir = test_data_dir / "cal"
    assert cal_dir.exists(), f"Cal directory {cal_dir} not found"

    # Check if at least one calibration file exists
    cal_files = list(cal_dir.glob("*.ori"))
    assert len(cal_files) > 0, f"No calibration files found in {cal_dir}"


def test_py_start_proc_c():
    """Test the py_start_proc_c function"""
    # Just test that the function exists and is callable
    assert callable(py_start_proc_c)

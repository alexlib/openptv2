#!/usr/bin/env python
"""
Test script to verify pyptv installation
"""

import os
import sys
import pytest
from openptv2.calibration import Calibration


def test_installation(test_data_dir):
    """Test if pyptv and optv are installed correctly"""
    try:
        import openptv2.gui as pyptv

        print(f"PyPTV version: {pyptv.__version__}")
    except ImportError as e:
        pytest.fail(f"Error: PyPTV is not installed correctly: {e}")

    try:
        import optv

        print(f"OpenPTV version: {optv.__version__}")
    except ImportError as e:
        pytest.skip(f"Legacy optv package not installed: {e}")

    # Test path to test_cavity
    test_cavity_path = test_data_dir
    print(f"Test cavity path: {test_cavity_path}")

    # Test if we can load calibration
    try:
        cal = Calibration()
        cal_file = os.path.join(test_cavity_path, "cal", "cam1.tif.ori")
        addpar_file = os.path.join(test_cavity_path, "cal", "cam1.tif.addpar")

        if os.path.exists(cal_file) and os.path.exists(addpar_file):
            cal.from_file(cal_file.encode(), addpar_file.encode())
            print("Successfully loaded calibration")
            print(f"Calibration parameters: {cal.get_pos()}")
        else:
            pytest.fail(f"Calibration files not found: {cal_file} or {addpar_file}")
    except Exception as e:
        pytest.fail(f"Error loading calibration: {str(e)}")

    print("Installation test completed successfully!")


if __name__ == "__main__":
    try:
        # If run as standalone, use a default test_data path if sys.argv or environment is not set,
        # or require pytest to run it with the fixture
        test_dir = sys.argv[1] if len(sys.argv) > 1 else "test_data/test_cavity"
        test_installation(test_dir)
        print("Success!")
        sys.exit(0)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)

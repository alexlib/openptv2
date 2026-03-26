#!/usr/bin/env python
"""
Test script to verify pyptv installation
"""

import os
import sys
from optv.calibration import Calibration


def test_installation(test_data_dir):
    """Test if pyptv and optv are installed correctly"""
    try:
        import gui.pyptv as pyptv

        print(f"PyPTV version: {pyptv.__version__}")
    except ImportError:
        raise AssertionError("Error: PyPTV is not installed correctly")

    try:
        import optv

        print(f"OpenPTV version: {optv.__version__}")
    except ImportError:
        raise AssertionError("Error: OpenPTV is not installed correctly")

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
            print("Calibration files not found")
            raise AssertionError("Calibration files not found")
    except Exception as e:
        raise AssertionError(f"Error loading calibration: {str(e)}")

    print("Installation test completed successfully!")


if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)

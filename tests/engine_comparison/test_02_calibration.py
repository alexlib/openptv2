"""
Engine comparison tests for Calibration class.

Tests camera calibration get/set methods and file I/O.
Tolerance: 1e-9 (parameter structures)
"""

import numpy as np
import pytest
from pathlib import Path
from .conftest import get_tolerance, compare_arrays, compare_values

TOLERANCE = get_tolerance("calibration")


class TestCalibration:
    """Compare Calibration results between optv and python engines."""

    def test_calibration_creation_default(self):
        """Test calibration creation with default parameters."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        optv_pos = optv_cal.get_pos()
        python_pos = python_cal.get_pos()

        np.testing.assert_allclose(optv_pos, python_pos, rtol=TOLERANCE, atol=TOLERANCE)

    def test_calibration_creation_with_params(self):
        """Test calibration creation with provided parameters."""
        pos = np.array([10.0, 20.0, 100.0])
        angles = np.array([0.1, 0.2, 0.3])

        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal(pos=pos, angs=angles)
        python_cal = PythonCal()
        python_cal.set_pos(pos)
        python_cal.set_angles(angles)

        optv_pos = optv_cal.get_pos()
        python_pos = python_cal.get_pos()
        np.testing.assert_allclose(optv_pos, python_pos, rtol=TOLERANCE, atol=TOLERANCE)

        optv_angles = optv_cal.get_angles()
        python_angles = python_cal.get_angles()
        np.testing.assert_allclose(
            optv_angles, python_angles, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_pos_set_pos(self):
        """Test get_pos and set_pos methods."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        test_pos = np.array([50.0, 75.0, 200.0])

        optv_cal.set_pos(test_pos)
        python_cal.set_pos(test_pos)

        optv_result = optv_cal.get_pos()
        python_result = python_cal.get_pos()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_angles_set_angles(self):
        """Test get_angles and set_angles methods."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        test_angles = np.array([np.pi / 4, np.pi / 6, np.pi / 3])

        optv_cal.set_angles(test_angles)
        python_cal.set_angles(test_angles)

        optv_result = optv_cal.get_angles()
        python_result = python_cal.get_angles()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_rotation_matrix(self):
        """Test get_rotation_matrix method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        angles = np.array([0.1, 0.2, 0.3])

        optv_cal = OptvCal(angs=angles)
        python_cal = PythonCal()
        python_cal.set_angles(angles)

        optv_result = optv_cal.get_rotation_matrix()
        python_result = python_cal.get_rotation_matrix()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_primary_point(self):
        """Test get_primary_point method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        prim_point = np.array([512.0, 384.0, 0.0])

        optv_cal = OptvCal(prim_point=prim_point)
        python_cal = PythonCal()
        python_cal.set_primary_point(prim_point)

        optv_result = optv_cal.get_primary_point()
        python_result = python_cal.get_primary_point()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_set_primary_point(self):
        """Test set_primary_point method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        prim_point = np.array([500.0, 400.0, 0.0])

        optv_cal.set_primary_point(prim_point)
        python_cal.set_primary_point(prim_point)

        optv_result = optv_cal.get_primary_point()
        python_result = python_cal.get_primary_point()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_radial_distortion(self):
        """Test get_radial_distortion method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        rad_dist = np.array([0.001, 0.002, 0.0005])

        optv_cal = OptvCal(rad_dist=rad_dist)
        python_cal = PythonCal()
        python_cal.set_radial_distortion(rad_dist)

        optv_result = optv_cal.get_radial_distortion()
        python_result = python_cal.get_radial_distortion()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_set_radial_distortion(self):
        """Test set_radial_distortion method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        rad_dist = np.array([0.01, 0.02, 0.005])

        optv_cal.set_radial_distortion(rad_dist)
        python_cal.set_radial_distortion(rad_dist)

        optv_result = optv_cal.get_radial_distortion()
        python_result = python_cal.get_radial_distortion()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_decentering(self):
        """Test get_decentering method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        decent = np.array([0.0001, 0.0002])

        optv_cal = OptvCal(decent=decent)
        python_cal = PythonCal()
        python_cal.set_decentering(decent)

        optv_result = optv_cal.get_decentering()
        python_result = python_cal.get_decentering()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_set_decentering(self):
        """Test set_decentering method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        decent = np.array([0.001, 0.002])

        optv_cal.set_decentering(decent)
        python_cal.set_decentering(decent)

        optv_result = optv_cal.get_decentering()
        python_result = python_cal.get_decentering()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_affine_trans(self):
        """Test get_affine_trans method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        affine = np.array([1.0, 0.01])

        optv_cal = OptvCal(affine=affine)
        python_cal = PythonCal()
        python_cal.set_affine_trans(affine)

        optv_result = optv_cal.get_affine()
        python_result = python_cal.get_affine_trans()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_set_affine_trans(self):
        """Test set_affine_trans method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        affine = np.array([1.0, 0.02])

        optv_cal.set_affine_trans(affine)
        python_cal.set_affine_trans(affine)

        optv_result = optv_cal.get_affine()
        python_result = python_cal.get_affine_trans()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_get_glass_vec(self):
        """Test get_glass_vec method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        glass = np.array([0.0, 0.0, 10.0])

        optv_cal = OptvCal(glass=glass)
        python_cal = PythonCal()
        python_cal.set_glass_vec(glass)

        optv_result = optv_cal.get_glass_vec()
        python_result = python_cal.get_glass_vec()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_set_glass_vec(self):
        """Test set_glass_vec method."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        glass = np.array([0.0, 0.0, 20.0])

        optv_cal.set_glass_vec(glass)
        python_cal.set_glass_vec(glass)

        optv_result = optv_cal.get_glass_vec()
        python_result = python_cal.get_glass_vec()

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_calibration_from_file(self, calibration_files):
        """Test loading calibration from .ori and .addpar files."""
        import os
        import sys
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        cam1_ori, cam1_add = calibration_files["cam1"]

        print(f"\nCWD: {os.getcwd()}", file=sys.stderr)
        print(f"cam1_ori: {cam1_ori}, exists: {cam1_ori.exists()}", file=sys.stderr)

        optv_cal = OptvCal()
        optv_cal.from_file(str(cam1_ori), str(cam1_add) if cam1_add else None)
        print(f"optv pos: {optv_cal.get_pos()}", file=sys.stderr)

        python_cal = PythonCal()
        python_cal.from_file(str(cam1_ori), str(cam1_add) if cam1_add else None)
        print(f"python pos: {python_cal.get_pos()}", file=sys.stderr)

        np.testing.assert_allclose(
            optv_cal.get_pos(), python_cal.get_pos(), rtol=TOLERANCE, atol=TOLERANCE
        )
        np.testing.assert_allclose(
            optv_cal.get_angles(),
            python_cal.get_angles(),
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )

    def test_calibration_sym_cameras(self, calibration_files):
        """Test loading symmetric camera calibrations."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        sym_cams = ["sym_cam1", "sym_cam2", "sym_cam3", "sym_cam4"]

        for cam_name in sym_cams:
            ori_file, add_file = calibration_files[cam_name]

            optv_cal = OptvCal()
            optv_cal.from_file(str(ori_file), str(add_file) if add_file else None)

            python_cal = PythonCal()
            python_cal.from_file(ori_file, add_file if add_file else None)

            np.testing.assert_allclose(
                optv_cal.get_pos(),
                python_cal.get_pos(),
                rtol=TOLERANCE,
                atol=TOLERANCE,
                err_msg=f"Failed for {cam_name}",
            )
            np.testing.assert_allclose(
                optv_cal.get_angles(),
                python_cal.get_angles(),
                rtol=TOLERANCE,
                atol=TOLERANCE,
                err_msg=f"Failed for {cam_name}",
            )

    def test_rotation_matrix_identity(self):
        """Test rotation_matrix with identity angles."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal
        from algorithms.calibration import rotation_matrix

        angles = np.array([0.0, 0.0, 0.0])

        optv_cal = OptvCal(angs=angles)
        optv_matrix = optv_cal.get_rotation_matrix()

        python_cal = PythonCal()
        python_cal.set_angles(angles)
        python_matrix = python_cal.get_rotation_matrix()

        np.testing.assert_allclose(
            optv_matrix, python_matrix, rtol=TOLERANCE, atol=TOLERANCE
        )
        np.testing.assert_allclose(
            python_matrix, np.eye(3), rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_rotation_matrix_90_degrees(self):
        """Test rotation_matrix with 90 degree rotations."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        angles = np.array([np.pi / 2, 0.0, 0.0])

        optv_cal = OptvCal(angs=angles)
        optv_matrix = optv_cal.get_rotation_matrix()

        python_cal = PythonCal()
        python_cal.set_angles(angles)
        python_matrix = python_cal.get_rotation_matrix()

        np.testing.assert_allclose(
            optv_matrix, python_matrix, rtol=TOLERANCE, atol=TOLERANCE
        )

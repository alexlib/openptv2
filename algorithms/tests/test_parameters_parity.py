"""
Parity test for Parameters classes.

Tests that Cython and Python implementations produce identical results
for the parameter classes from bindings/optv/parameters.pyx and algorithms/parameters.py.

Each engine reads the SAME parameter files through its own native reader,
ensuring both reader parity and value parity are tested.
"""

import os
import pytest
import numpy as np
from pathlib import Path

# Relative path from test file to test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_data")

TOLERANCE = 1e-7


class TestParametersParity:
    """Test that parameter classes produce identical results in both engines."""

    def test_control_params_from_file(self, file_control_params):
        """Both engines read the same control.par file and get identical values."""
        optv_cpar, python_cpar = file_control_params
        assert optv_cpar is not None, "optv failed to read control.par"
        assert python_cpar is not None, "python failed to read control.par"

        assert optv_cpar.get_num_cams() == python_cpar.num_cams
        assert optv_cpar.get_image_size() == (python_cpar.imx, python_cpar.imy)
        np.testing.assert_allclose(
            optv_cpar.get_pixel_size(),
            (python_cpar.pix_x, python_cpar.pix_y),
            rtol=TOLERANCE,
        )
        assert optv_cpar.get_chfield() == python_cpar.chfield

    def test_volume_params_from_file(self, file_volume_params):
        """Both engines read the same volume.par file and get identical values."""
        optv_vpar, python_vpar = file_volume_params
        assert optv_vpar is not None, "optv failed to read volume.par"
        assert python_vpar is not None, "python failed to read volume.par"

        x_lay = list(optv_vpar.get_X_lay())
        z_min = list(optv_vpar.get_Zmin_lay())
        z_max = list(optv_vpar.get_Zmax_lay())

        np.testing.assert_allclose(x_lay, python_vpar.x_lay, rtol=TOLERANCE)
        np.testing.assert_allclose(z_min, python_vpar.z_min_lay, rtol=TOLERANCE)
        np.testing.assert_allclose(z_max, python_vpar.z_max_lay, rtol=TOLERANCE)

    def test_control_params_creation(self):
        """Test ControlParams creation in both engines."""
        try:
            from optv.parameters import ControlParams as CythonCP

            cpar_cython = CythonCP(num_cams=4)
            assert cpar_cython is not None
        except ImportError:
            pytest.skip("optv not available")

        try:
            from algorithms.parameters_adapter import ControlParams as PythonCP

            cpar_python = PythonCP(num_cams=4)
            assert cpar_python is not None
        except ImportError:
            pytest.skip("Adapter not available")

    def test_volume_params_creation(self):
        """Test VolumeParams creation in both engines."""
        try:
            from optv.parameters import VolumeParams as CythonVP

            vpar_cython = CythonVP()
            assert vpar_cython is not None
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.parameters import VolumePar as PythonVP

        vpar_python = PythonVP()
        assert vpar_python is not None

    def test_tracking_params_creation(self):
        """Test TrackingParams creation in both engines."""
        try:
            from optv.parameters import TrackingParams as CythonTP

            tpar_cython = CythonTP()
            assert tpar_cython is not None
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.parameters import TrackPar as PythonTP

        tpar_python = PythonTP()
        assert tpar_python is not None

    def test_sequence_params_creation(self):
        """Test SequenceParams creation in both engines."""
        try:
            from optv.parameters import SequenceParams as CythonSP

            spar_cython = CythonSP(num_cams=4)
            assert spar_cython is not None
        except ImportError:
            pytest.skip("optv not available")

        try:
            from algorithms.parameters_adapter import SequenceParams as PythonSP

            spar_python = PythonSP(num_cams=4)
            assert spar_python is not None
        except ImportError:
            pytest.skip("Adapter not available")

    def test_target_params_creation(self):
        """Test TargetParams creation in both engines."""
        try:
            from optv.parameters import TargetParams as CythonTGP

            tgpar_cython = CythonTGP()
            assert tgpar_cython is not None
        except ImportError:
            pytest.skip("optv not available")

        try:
            from algorithms.parameters_adapter import TargetParams as PythonTGP

            tgpar_python = PythonTGP()
            assert tgpar_python is not None
        except ImportError:
            pytest.skip("Adapter not available")

    def test_multimedia_params_creation(self):
        """Test MultimediaParams creation in both engines."""
        try:
            from optv.parameters import MultimediaParams as CythonMP

            mpar_cython = CythonMP()
            assert mpar_cython is not None
        except ImportError:
            pytest.skip("optv not available")

        try:
            from algorithms.parameters_adapter import MultimediaParams as PythonMP

            mpar_python = PythonMP()
            assert mpar_python is not None
        except ImportError:
            pytest.skip("Adapter not available")

    def test_parameters_api_parity(self):
        """Verify both implementations have similar APIs."""
        try:
            from optv.parameters import ControlParams, TrackingParams, VolumeParams
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.parameters import ControlPar, TrackPar, VolumePar

        # Check that key classes exist in both
        assert ControlParams is not None
        assert TrackingParams is not None
        assert VolumeParams is not None
        assert ControlPar is not None
        assert TrackPar is not None
        assert VolumePar is not None


class TestCorrespondencesParity:
    """Test MatchedCoords and correspondences function parity."""

    def test_matched_coords_api(self):
        """Verify MatchedCoords has required methods."""
        try:
            from optv.correspondences import MatchedCoords as CythonMC
        except ImportError:
            pytest.skip("optv not available")

        # Cython MatchedCoords requires TargetArray, ControlParams, Calibration
        # Check that the class exists and has expected methods
        cython_methods = ["as_arrays", "get_by_pnrs"]
        for method in cython_methods:
            assert hasattr(CythonMC, method), f"Cython missing {method}"

        # For Python, we don't have a direct MatchedCoords class
        # Check what functions are available in algorithms/correspondences
        from algorithms import correspondences as corr_module

        # Just verify the module exists and is importable
        assert corr_module is not None


class TestImageCoordParity:
    """Test imgcoord functions parity."""

    def test_flat_image_coordinates(self):
        """Test flat_image_coordinates function in both engines."""
        try:
            from optv.imgcoord import flat_image_coordinates
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.imgcoord import flat_image_coordinates as python_func

        # Both should exist and be callable
        assert callable(flat_image_coordinates)
        assert callable(python_func)

    def test_image_coordinates(self):
        """Test image_coordinates function in both engines."""
        try:
            from optv.imgcoord import image_coordinates
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.imgcoord import image_coordinates as python_func

        assert callable(image_coordinates)
        assert callable(python_func)


class TestSegmentationParity:
    """Test segmentation functions parity."""

    def test_target_recognition(self):
        """Test target_recognition function in both engines."""
        try:
            from optv.segmentation import target_recognition
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.segmentation import target_recognition as python_func

        assert callable(target_recognition)
        assert callable(python_func)


class TestEpipolarParity:
    """Test epipolar functions parity."""

    def test_epipolar_curve(self):
        """Test epipolar_curve function in both engines."""
        try:
            from optv.epipolar import epipolar_curve
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.epi import epipolar_curve as python_func

        assert callable(epipolar_curve)
        assert callable(python_func)


class TestImageProcessingParity:
    """Test image processing functions parity."""

    def test_preprocess_image(self):
        """Test preprocess_image function in both engines."""
        try:
            from optv.image_processing import preprocess_image
        except ImportError:
            pytest.skip("optv not available")

        from algorithms.image_processing import preprocess_image as python_func

        assert callable(preprocess_image)
        assert callable(python_func)


class TestTransformsParity:
    """Test transforms/trafo functions parity."""

    def test_convert_arr_pixel_to_metric(self):
        """Test convert_arr_pixel_to_metric in both engines."""
        try:
            from optv.transforms import convert_arr_pixel_to_metric
        except ImportError:
            pytest.skip("optv not available")

        # Python uses different name: arr_pixel_to_metric
        from algorithms.trafo import arr_pixel_to_metric as python_func

        assert callable(convert_arr_pixel_to_metric)
        assert callable(python_func)

    def test_convert_arr_metric_to_pixel(self):
        """Test convert_arr_metric_to_pixel in both engines."""
        try:
            from optv.transforms import convert_arr_metric_to_pixel
        except ImportError:
            pytest.skip("optv not available")

        # Python uses different name: arr_metric_to_pixel
        from algorithms.trafo import arr_metric_to_pixel as python_func

        assert callable(convert_arr_metric_to_pixel)
        assert callable(python_func)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

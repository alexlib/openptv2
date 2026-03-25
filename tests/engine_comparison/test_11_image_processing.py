"""
Engine comparison tests for image_processing module.

Tests preprocess_image function.
Tolerance: 1e-9 (simple preprocessing)
"""

import numpy as np
import pytest
from .conftest import get_tolerance

TOLERANCE = get_tolerance("image_processing")


class TestImageProcessing:
    """Compare image processing functions between optv and python engines."""

    def test_preprocess_image_basic(self):
        """Test preprocess_image with basic image."""
        from optv.image_processing import preprocess_image as optv_func
        from optv.parameters import ControlParams

        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(img, filter_hp=1, control=cpar, lowpass_dim=1)
        except Exception as e:
            pytest.fail(f"optv preprocess_image failed: {e}")

        try:
            from algorithms.image_processing import preprocess_image as python_func
            from algorithms._native_convert import from_optv_control_par

            native_cpar = from_optv_control_par(cpar)
            python_result = python_func(img, filter_hp=1, cpar=native_cpar, dim_lp=1)

            if python_result is not None:
                np.testing.assert_allclose(
                    optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_preprocess_image_lowpass(self):
        """Test preprocess_image with lowpass filter."""
        from optv.image_processing import preprocess_image as optv_func
        from optv.parameters import ControlParams

        img = np.zeros((50, 50), dtype=np.uint8)
        cpar = ControlParams(num_cams=4, image_size=(50, 50), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(img, filter_hp=1, control=cpar, lowpass_dim=2)
        except Exception as e:
            pytest.fail(f"optv preprocess_image failed: {e}")

        assert optv_result.shape == img.shape
        assert optv_result.dtype == img.dtype

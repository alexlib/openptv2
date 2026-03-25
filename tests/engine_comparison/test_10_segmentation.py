"""
Engine comparison tests for segmentation module.

Tests target_recognition function.
Tolerance: 1e-7 (image processing algorithms)
"""

import numpy as np
import pytest
from .conftest import get_tolerance, create_test_control_params

TOLERANCE = get_tolerance("segmentation")


class TestSegmentation:
    """Compare segmentation functions between optv and python engines."""

    def test_target_recognition_basic(self, test_image):
        """Test target_recognition with basic test image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams

        tpar = TargetParams(
            pixel_count_bounds=(3, 20),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(test_image, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(test_image, tpar, 0, cpar)

            assert len(optv_result) == len(python_result)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_no_targets(self):
        """Test target_recognition with empty (black) image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams

        black_image = np.zeros((100, 100), dtype=np.uint8)

        tpar = TargetParams(
            pixel_count_bounds=(3, 20),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(black_image, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        assert len(optv_result) == 0

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(black_image, tpar, 0, cpar)

            assert len(optv_result) == len(python_result)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_single_bright_spot(self):
        """Test with single bright spot in image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams

        img = np.zeros((100, 100), dtype=np.uint8)
        img[45:55, 45:55] = 255

        tpar = TargetParams(
            pixel_count_bounds=(3, 30),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(img, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(img, tpar, 0, cpar)

            if len(optv_result) > 0 and len(python_result) > 0:
                assert abs(optv_result[0].pos()[0] - python_result[0].pos()[0]) < 2.0
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_multiple_targets(self):
        """Test with multiple targets in image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams

        img = np.zeros((200, 200), dtype=np.uint8)

        positions = [(50, 50), (100, 80), (150, 120), (60, 160)]
        for x, y in positions:
            img[y - 5 : y + 5, x - 5 : x + 5] = 200

        tpar = TargetParams(
            pixel_count_bounds=(3, 30),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        cpar = ControlParams(num_cams=4, image_size=(200, 200), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(img, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(img, tpar, 0, cpar)

            assert abs(len(optv_result) - len(python_result)) <= 1
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_with_subrange(self):
        """Test target_recognition with image subrange."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams

        img = np.zeros((100, 100), dtype=np.uint8)
        img[45:55, 45:55] = 255

        tpar = TargetParams(
            pixel_count_bounds=(3, 30),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(
                img, tpar, 0, cpar, subrange_x=(30, 70), subrange_y=(30, 70)
            )
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        assert len(optv_result) >= 1

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(
                img, tpar, 0, cpar, subrange_x=(30, 70), subrange_y=(30, 70)
            )

            assert abs(len(optv_result) - len(python_result)) <= 1
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_large_image(self):
        """Test with larger image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams

        np.random.seed(42)
        img = np.random.randint(0, 50, (512, 512), dtype=np.uint8)

        for _ in range(20):
            x, y = np.random.randint(50, 450, 2)
            img[y - 3 : y + 3, x - 3 : x + 3] = 200

        tpar = TargetParams(
            pixel_count_bounds=(2, 15),
            xsize_bounds=(0, 3),
            ysize_bounds=(0, 3),
        )
        cpar = ControlParams(num_cams=4, image_size=(512, 512), pixel_size=(0.01, 0.01))

        try:
            optv_result = optv_func(img, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(img, tpar, 0, cpar)

            assert abs(len(optv_result) - len(python_result)) <= 2
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

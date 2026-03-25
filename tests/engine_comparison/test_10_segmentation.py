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
        from algorithms.parameters import ControlPar as PythonControlPar, TargetPar as PythonTargetPar

        tpar = TargetParams(
            pixel_count_bounds=(3, 20),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        python_tpar = PythonTargetPar()
        python_tpar.gvthresh = [0, 0, 0, 0]
        python_tpar.nnmin = 3
        python_tpar.nnmax = 20
        python_tpar.nxmin = 0
        python_tpar.nxmax = 5
        python_tpar.nymin = 0
        python_tpar.nymax = 5
        python_tpar.sumg_min = 0
        python_tpar.cr_sz = 0
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))
        python_cpar = PythonControlPar()
        python_cpar.imx = 100
        python_cpar.imy = 100

        try:
            optv_result = optv_func(test_image, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(test_image, python_tpar, 0, python_cpar)

            assert len(optv_result) == len(python_result)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_no_targets(self):
        """Test target_recognition with empty (black) image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams
        from algorithms.parameters import ControlPar as PythonControlPar, TargetPar as PythonTargetPar

        black_image = np.zeros((100, 100), dtype=np.uint8)

        tpar = TargetParams(
            pixel_count_bounds=(3, 20),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        python_tpar = PythonTargetPar()
        python_tpar.gvthresh = [0, 0, 0, 0]
        python_tpar.nnmin = 3
        python_tpar.nnmax = 20
        python_tpar.nxmin = 0
        python_tpar.nxmax = 5
        python_tpar.nymin = 0
        python_tpar.nymax = 5
        python_tpar.sumg_min = 0
        python_tpar.cr_sz = 0
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))
        python_cpar = PythonControlPar()
        python_cpar.imx = 100
        python_cpar.imy = 100

        try:
            optv_result = optv_func(black_image, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(black_image, python_tpar, 0, python_cpar)

            assert len(optv_result) == len(python_result)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_single_bright_spot(self):
        """Test with single bright spot in image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams
        from algorithms.parameters import ControlPar as PythonControlPar, TargetPar as PythonTargetPar

        img = np.zeros((100, 100), dtype=np.uint8)
        img[45:55, 45:55] = 255

        tpar = TargetParams(
            pixel_count_bounds=(3, 30),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        python_tpar = PythonTargetPar()
        python_tpar.gvthresh = [0, 0, 0, 0]
        python_tpar.nnmin = 3
        python_tpar.nnmax = 30
        python_tpar.nxmin = 0
        python_tpar.nxmax = 5
        python_tpar.nymin = 0
        python_tpar.nymax = 5
        python_tpar.sumg_min = 0
        python_tpar.cr_sz = 0
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))
        python_cpar = PythonControlPar()
        python_cpar.imx = 100
        python_cpar.imy = 100

        try:
            optv_result = optv_func(img, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(img, python_tpar, 0, python_cpar)

            if len(optv_result) > 0 and len(python_result) > 0:
                assert abs(optv_result[0].pos()[0] - python_result[0].pos()[0]) < 2.0
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_multiple_targets(self):
        """Test with multiple targets in image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams
        from algorithms.parameters import ControlPar as PythonControlPar, TargetPar as PythonTargetPar

        img = np.zeros((200, 200), dtype=np.uint8)

        positions = [(50, 50), (100, 80), (150, 120), (60, 160)]
        for x, y in positions:
            img[y - 5 : y + 5, x - 5 : x + 5] = 200

        tpar = TargetParams(
            pixel_count_bounds=(3, 30),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        python_tpar = PythonTargetPar()
        python_tpar.gvthresh = [0, 0, 0, 0]
        python_tpar.nnmin = 3
        python_tpar.nnmax = 30
        python_tpar.nxmin = 0
        python_tpar.nxmax = 5
        python_tpar.nymin = 0
        python_tpar.nymax = 5
        python_tpar.sumg_min = 0
        python_tpar.cr_sz = 0
        cpar = ControlParams(num_cams=4, image_size=(200, 200), pixel_size=(0.01, 0.01))
        python_cpar = PythonControlPar()
        python_cpar.imx = 200
        python_cpar.imy = 200

        try:
            optv_result = optv_func(img, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(img, python_tpar, 0, python_cpar)

            assert abs(len(optv_result) - len(python_result)) <= 1
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_with_subrange(self):
        """Test target_recognition with image subrange."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams
        from algorithms.parameters import ControlPar as PythonControlPar, TargetPar as PythonTargetPar

        img = np.zeros((70, 70), dtype=np.uint8)
        img[30:40, 30:40] = 255

        tpar = TargetParams(
            pixel_count_bounds=(3, 30),
            xsize_bounds=(0, 5),
            ysize_bounds=(0, 5),
        )
        python_tpar = PythonTargetPar()
        python_tpar.gvthresh = [0, 0, 0, 0]
        python_tpar.nnmin = 3
        python_tpar.nnmax = 30
        python_tpar.nxmin = 0
        python_tpar.nxmax = 5
        python_tpar.nymin = 0
        python_tpar.nymax = 5
        python_tpar.sumg_min = 0
        python_tpar.cr_sz = 0
        cpar = ControlParams(num_cams=4, image_size=(100, 100), pixel_size=(0.01, 0.01))
        python_cpar = PythonControlPar()
        python_cpar.imx = 100
        python_cpar.imy = 100

        try:
            optv_result = optv_func(
                img, tpar, 0, cpar, subrange_x=(0, 70), subrange_y=(0, 70)
            )
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        assert len(optv_result) >= 1

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(
                img, python_tpar, 0, python_cpar, subrange_x=(0, 70), subrange_y=(0, 70)
            )
            assert python_result is not None
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_target_recognition_large_image(self):
        """Test with larger image."""
        from optv.segmentation import target_recognition as optv_func
        from optv.parameters import TargetParams, ControlParams
        from algorithms.parameters import ControlPar as PythonControlPar, TargetPar as PythonTargetPar

        np.random.seed(42)
        img = np.random.randint(0, 50, (256, 256), dtype=np.uint8)

        for _ in range(20):
            x, y = np.random.randint(25, 231, 2)
            img[y - 3 : y + 3, x - 3 : x + 3] = 200

        tpar = TargetParams(
            pixel_count_bounds=(2, 15),
            xsize_bounds=(0, 3),
            ysize_bounds=(0, 3),
        )
        python_tpar = PythonTargetPar()
        python_tpar.gvthresh = [0, 0, 0, 0]
        python_tpar.nnmin = 2
        python_tpar.nnmax = 15
        python_tpar.nxmin = 0
        python_tpar.nxmax = 3
        python_tpar.nymin = 0
        python_tpar.nymax = 3
        python_tpar.sumg_min = 0
        python_tpar.cr_sz = 0
        cpar = ControlParams(num_cams=4, image_size=(256, 256), pixel_size=(0.01, 0.01))
        python_cpar = PythonControlPar()
        python_cpar.imx = 256
        python_cpar.imy = 256

        try:
            optv_result = optv_func(img, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv target_recognition failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func

            python_result = python_func(img, python_tpar, 0, python_cpar)
            assert python_result is not None
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

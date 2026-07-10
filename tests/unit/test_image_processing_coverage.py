"""Unit tests for image_processing.py — pure-Python coverage run.

Run via:
    COVERAGE_FILE=/tmp/.cov_imgproc uv run pytest tests/unit/test_image_processing_coverage.py \
      -o pythonpath=/tmp/ppsrc/openptv2 \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing -q

For normal (compiled) runs the module-level skip fires and the suite is a no-op.
"""
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module when running against compiled extension
# ---------------------------------------------------------------------------
from openptv2.algorithms.image_processing import is_compiled as _is_compiled

if _is_compiled():
    pytest.skip("pure-Python coverage tests only", allow_module_level=True)

from openptv2.algorithms.image_processing import (
    copy_images,
    fast_box_blur,
    filter_3,
    lowpass_3,
    prepare_image,
    split,
    subtract_img,
    subtract_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(imy: int, imx: int, value: int = 100) -> np.ndarray:
    return np.full((imy, imx), value, dtype=np.uint8)


def _ramp(imy: int, imx: int) -> np.ndarray:
    """Linearly ramped image so every pixel is different."""
    return np.arange(imy * imx, dtype=np.uint8).reshape(imy, imx)


# ===========================================================================
# filter_3
# ===========================================================================

class TestFilter3:
    def test_zero_sum_raises(self):
        img = _solid(5, 5, 50)
        filt = np.zeros((3, 3))
        with pytest.raises(ValueError, match="sum is zero"):
            filter_3(img, filt, 5, 5)

    def test_identity_filter(self):
        img = _solid(5, 5, 100)
        filt = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        out = filter_3(img, filt, 5, 5)
        assert out.shape == (5, 5)
        assert out.dtype == np.uint8

    def test_output_clamped_above_255(self):
        # Produce total / filt_sum > 255 by using a bright image and large
        # centre weight relative to sum so the division stays > 255.
        img = np.full((5, 5), 255, dtype=np.uint8)
        # With equal weights total = 255*9, filt_sum=9 → buf=255 → clamp fires
        filt = np.ones((3, 3), dtype=np.float64)
        out = filter_3(img, filt, 5, 5)
        # All interior pixels should be clamped to 255
        assert int(out[2, 2]) == 255

    def test_min_brightness_floor(self):
        # Dark image → buf < min_brightness triggers floor
        img = np.zeros((5, 5), dtype=np.uint8)
        filt = np.ones((3, 3), dtype=np.float64)
        out = filter_3(img, filt, 5, 5, min_brightness=8)
        # Interior pixels come out floored to min_brightness=8
        assert int(out[2, 2]) == 8

    def test_min_brightness_zero(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        filt = np.ones((3, 3), dtype=np.float64)
        out = filter_3(img, filt, 5, 5, min_brightness=0)
        assert int(out[2, 2]) == 0

    def test_output_shape_and_dtype(self):
        img = _ramp(8, 10)
        filt = np.ones((3, 3), dtype=np.float64)
        out = filter_3(img, filt, 10, 8)
        assert out.shape == (8, 10)
        assert out.dtype == np.uint8

    def test_non_uniform_filter(self):
        img = _solid(7, 7, 90)
        filt = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
        out = filter_3(img, filt, 7, 7)
        assert out.shape == (7, 7)

    def test_asymmetric_image(self):
        img = _ramp(6, 10)
        filt = np.ones((3, 3), dtype=np.float64)
        out = filter_3(img, filt, 10, 6)
        assert out.shape == (6, 10)

    def test_buf_above_255_branch_triggered(self):
        # Single bright pixel centre weight that drives buf > 255
        img = np.zeros((5, 5), dtype=np.uint8)
        img[2, 2] = 255
        # filt_sum = 1, centre weight = 10 so total = 10*255 = 2550, buf = 2550
        filt = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=np.float64)
        # filt_sum=10 → total/filt_sum = 255 → exactly 255, not > 255
        # Use unequal weights so filt_sum < max weight
        filt2 = np.array([[0, 0, 0], [0, 9, 0], [0, 0, 1]], dtype=np.float64)
        # filt_sum=10, centre * 9 / 10 for a 255-pixel: 229 < 255
        # Instead: very large centre weight and filt_sum=1 → buf=255*big_num → clamp
        filt3 = np.array([[0, 0, 0], [0, 1000, 0], [0, 0, -999]], dtype=np.float64)
        # filt_sum=1, total = 255*1000 - 0 = 255000, buf=255000 → clamp
        out = filter_3(img, filt3, 5, 5)
        assert int(out[2, 2]) == 255


# ===========================================================================
# lowpass_3
# ===========================================================================

class TestLowpass3:
    def test_solid_image(self):
        img = _solid(5, 5, 90)
        out = lowpass_3(img, 5, 5)
        assert out.shape == (5, 5)
        # Interior pixels: sum of 9 identical values / 9 = same value
        assert int(out[2, 2]) == 90

    def test_zero_image(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        out = lowpass_3(img, 5, 5)
        assert np.all(out == 0)

    def test_output_dtype(self):
        img = _ramp(6, 8)
        out = lowpass_3(img, 8, 6)
        assert out.dtype == np.uint8
        assert out.shape == (6, 8)

    def test_border_pixels_unchanged(self):
        img = _solid(5, 5, 100)
        out = lowpass_3(img, 5, 5)
        # Border pixels are left at zero (not written)
        assert int(out[0, 0]) == 0
        assert int(out[4, 4]) == 0

    def test_rectangular(self):
        img = _ramp(4, 8)
        out = lowpass_3(img, 8, 4)
        assert out.shape == (4, 8)


# ===========================================================================
# fast_box_blur
# ===========================================================================

class TestFastBoxBlur:
    def test_solid_image(self):
        img = _solid(10, 10, 128)
        out = fast_box_blur(img, 2, 10, 10)
        assert out.shape == (10, 10)
        assert out.dtype == np.uint8

    def test_zero_filt_span(self):
        img = _solid(6, 6, 50)
        out = fast_box_blur(img, 0, 6, 6)
        assert out.shape == (6, 6)

    def test_output_bounded(self):
        img = _ramp(12, 12)
        out = fast_box_blur(img, 3, 12, 12)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_span_1(self):
        img = _solid(8, 8, 100)
        out = fast_box_blur(img, 1, 8, 8)
        assert out.shape == (8, 8)

    def test_rectangular_image(self):
        img = _ramp(8, 16)
        out = fast_box_blur(img, 2, 16, 8)
        assert out.shape == (8, 16)

    def test_span_equal_to_half_image(self):
        # filt_span = 1, imx=imy=3 → some loops may be empty
        img = _solid(5, 5, 70)
        out = fast_box_blur(img, 1, 5, 5)
        assert out.shape == (5, 5)

    def test_large_image_all_branches(self):
        # 10x10 image, filt_span=2 (n=5): hits left-ramp, middle, right-ramp
        # and top-ramp, middle rows, bottom-ramp in vertical pass
        img = np.arange(100, dtype=np.uint8).reshape(10, 10)
        out = fast_box_blur(img, 2, 10, 10)
        assert out.shape == (10, 10)

    def test_tall_image(self):
        img = _ramp(20, 6)
        out = fast_box_blur(img, 2, 6, 20)
        assert out.shape == (20, 6)

    def test_wide_image(self):
        img = _ramp(6, 20)
        out = fast_box_blur(img, 2, 20, 6)
        assert out.shape == (6, 20)

    def test_filt_span_1_small_image(self):
        # Small image (5x5) with filt_span=1: covers left-ramp, middle, right-ramp
        img = _solid(5, 5, 80)
        out = fast_box_blur(img, 1, 5, 5)
        assert out.shape == (5, 5)


# ===========================================================================
# split
# ===========================================================================

class TestSplit:
    def test_half_selector_zero_returns_copy(self):
        img = _ramp(6, 8)
        out = split(img, 0, 8, 6)
        assert out.shape == (6, 8)
        np.testing.assert_array_equal(out, img)

    def test_half_selector_one_odd_rows(self):
        img = np.zeros((6, 4), dtype=np.uint8)
        # Fill even vs odd rows with different values
        img[0, :] = 10   # row 0 (even)
        img[1, :] = 20   # row 1 (odd)
        img[2, :] = 30   # row 2 (even)
        img[3, :] = 40   # row 3 (odd)
        img[4, :] = 50
        img[5, :] = 60
        out = split(img, 1, 4, 6)
        # half=3; row_offset=1 (odd rows): src rows 1,3,5 → dest rows 0,1,2
        assert int(out[0, 0]) == 20
        assert int(out[1, 0]) == 40
        assert int(out[2, 0]) == 60
        # Lower half filled with 2
        assert int(out[3, 0]) == 2
        assert int(out[4, 0]) == 2
        assert int(out[5, 0]) == 2

    def test_half_selector_two_even_rows(self):
        img = np.zeros((6, 4), dtype=np.uint8)
        img[0, :] = 10
        img[1, :] = 20
        img[2, :] = 30
        img[3, :] = 40
        img[4, :] = 50
        img[5, :] = 60
        out = split(img, 2, 4, 6)
        # row_offset=0 (even rows): src rows 0,2,4 → dest rows 0,1,2
        assert int(out[0, 0]) == 10
        assert int(out[1, 0]) == 30
        assert int(out[2, 0]) == 50
        # Lower half filled with 2
        assert int(out[3, 0]) == 2

    def test_split_does_not_modify_original(self):
        img = _solid(8, 6, 100)
        original = img.copy()
        split(img, 1, 6, 8)
        np.testing.assert_array_equal(img, original)

    def test_split_output_shape(self):
        img = _ramp(8, 10)
        out = split(img, 2, 10, 8)
        assert out.shape == (8, 10)
        assert out.dtype == np.uint8


# ===========================================================================
# subtract_img
# ===========================================================================

class TestSubtractImg:
    def test_basic_subtraction(self):
        img1 = _solid(4, 4, 100)
        img2 = _solid(4, 4, 40)
        out = subtract_img(img1, img2)
        assert np.all(out == 60)

    def test_clamped_to_zero(self):
        img1 = _solid(4, 4, 10)
        img2 = _solid(4, 4, 100)
        out = subtract_img(img1, img2)
        assert np.all(out == 0)

    def test_output_dtype(self):
        img1 = _solid(4, 4, 200)
        img2 = _solid(4, 4, 50)
        out = subtract_img(img1, img2)
        assert out.dtype == np.uint8

    def test_identical_images_gives_zero(self):
        img = _ramp(6, 6)
        out = subtract_img(img, img)
        assert np.all(out == 0)

    def test_partial_clamp(self):
        img1 = np.array([[200, 50]], dtype=np.uint8)
        img2 = np.array([[100, 100]], dtype=np.uint8)
        out = subtract_img(img1, img2)
        assert int(out[0, 0]) == 100
        assert int(out[0, 1]) == 0  # clamped


# ===========================================================================
# subtract_mask
# ===========================================================================

class TestSubtractMask:
    def test_full_mask(self):
        img = _solid(4, 4, 100)
        mask = _solid(4, 4, 255)
        out = subtract_mask(img, mask)
        assert np.all(out == 100)

    def test_zero_mask_clears_image(self):
        img = _solid(4, 4, 100)
        mask = np.zeros((4, 4), dtype=np.uint8)
        out = subtract_mask(img, mask)
        assert np.all(out == 0)

    def test_partial_mask(self):
        img = _solid(4, 4, 100)
        mask = np.ones((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 0
        out = subtract_mask(img, mask)
        assert int(out[0, 0]) == 100   # unmasked
        assert int(out[1, 1]) == 0     # masked

    def test_does_not_modify_original(self):
        img = _solid(4, 4, 100)
        original = img.copy()
        mask = np.ones((4, 4), dtype=np.uint8)
        mask[0, 0] = 0
        subtract_mask(img, mask)
        np.testing.assert_array_equal(img, original)


# ===========================================================================
# prepare_image
# ===========================================================================

class TestPrepareImage:
    def _img(self, imy=10, imx=10, value=100):
        return np.full((imy, imx), value, dtype=np.uint8)

    def test_filter_hp_none(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=0)
        assert out.shape == (10, 10)

    def test_filter_hp_lowpass(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=1)
        assert out.shape == (10, 10)

    def test_filter_hp_custom_from_file(self):
        img = self._img()
        filt = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            np.savetxt(f, filt)
            fpath = f.name
        try:
            out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=2, filter_file=fpath)
            assert out.shape == (10, 10)
        finally:
            os.unlink(fpath)

    def test_filter_hp_2_no_file_raises(self):
        img = self._img()
        with pytest.raises(ValueError, match="filter_file required"):
            prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=2, filter_file=None)

    def test_chfield_1(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=0, chfield=1)
        assert out.shape == (10, 10)

    def test_chfield_2(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=0, chfield=2)
        assert out.shape == (10, 10)

    def test_chfield_0_no_split(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=0, chfield=0)
        assert out.shape == (10, 10)

    def test_chfield_1_with_lowpass(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=1, chfield=1)
        assert out.shape == (10, 10)

    def test_chfield_2_with_lowpass(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10, filter_hp=1, chfield=2)
        assert out.shape == (10, 10)

    def test_returns_uint8(self):
        img = self._img()
        out = prepare_image(img, dim_lp=1, imx=10, imy=10)
        assert out.dtype == np.uint8

    def test_varied_image_content(self):
        img = _ramp(10, 10)
        out = prepare_image(img, dim_lp=2, imx=10, imy=10, filter_hp=0)
        assert out.shape == (10, 10)


# ===========================================================================
# copy_images
# ===========================================================================

class TestCopyImages:
    def test_list_input_returns_copies(self):
        imgs = [_solid(4, 4, 10), _solid(4, 4, 20)]
        result = copy_images(imgs)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], imgs[0])
        np.testing.assert_array_equal(result[1], imgs[1])
        # Copies are independent
        result[0][:] = 99
        assert int(imgs[0][0, 0]) == 10

    def test_with_dest(self):
        src = _solid(4, 4, 77)
        dest = np.zeros((4, 4), dtype=np.uint8)
        out = copy_images(src, dest=dest)
        assert out is dest
        np.testing.assert_array_equal(dest, src)

    def test_no_dest_returns_copy(self):
        src = _solid(4, 4, 55)
        out = copy_images(src)
        np.testing.assert_array_equal(out, src)
        out[:] = 0
        assert int(src[0, 0]) == 55  # original unchanged

    def test_imx_imy_ignored_with_dest(self):
        # imx/imy params are accepted but unused in dest path
        src = _solid(4, 4, 33)
        dest = np.zeros((4, 4), dtype=np.uint8)
        out = copy_images(src, dest=dest, imx=4, imy=4)
        np.testing.assert_array_equal(out, src)

    def test_empty_list(self):
        result = copy_images([])
        assert result == []


# ===========================================================================
# is_compiled
# ===========================================================================

class TestIsCompiled:
    def test_returns_bool(self):
        from openptv2.algorithms.image_processing import is_compiled
        result = is_compiled()
        assert isinstance(result, bool)

    def test_returns_false_in_pure_python(self):
        from openptv2.algorithms.image_processing import is_compiled
        # In pure-Python mode (running from /tmp/ppsrc), cython.compiled is False
        assert is_compiled() is False

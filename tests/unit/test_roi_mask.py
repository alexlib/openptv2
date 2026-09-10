import os
import time

import numpy as np
import pytest

from openptv2.roi_mask import apply_roi_mask, load_mask_polygon, rasterize_mask


def test_no_mask_file_means_full_frame(tmp_path):
    img = np.full((10, 12), 200, dtype=np.uint8)
    assert load_mask_polygon(0, tmp_path) is None
    out = apply_roi_mask(img, 0, tmp_path)
    assert np.array_equal(out, img)


def test_saved_polygon_masks_outside_points(tmp_path):
    (tmp_path / "mask_0.txt").write_text("0 0\n5 0\n5 9\n0 9\n")

    polygon_xy = load_mask_polygon(0, tmp_path)
    assert polygon_xy is not None
    assert polygon_xy.shape == (4, 2)

    img = np.full((10, 12), 200, dtype=np.uint8)
    out = apply_roi_mask(img, 0, tmp_path)
    assert out[5, 2] == 200  # inside the polygon
    assert out[5, 10] == 0  # outside the polygon


def test_fewer_than_three_points_is_no_mask(tmp_path):
    (tmp_path / "mask_1.txt").write_text("0 0\n5 0\n")
    assert load_mask_polygon(1, tmp_path) is None


def test_rasterize_mask_shape():
    poly = np.array([[0, 0], [4, 0], [4, 9], [0, 9]], dtype=float)
    mask = rasterize_mask(poly, imx=10, imy=10)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8


def test_edited_mask_invalidates_the_rasterized_cache(tmp_path):
    img = np.full((10, 12), 200, dtype=np.uint8)
    mask_path = tmp_path / "mask_0.txt"

    mask_path.write_text("0 0\n5 0\n5 9\n0 9\n")
    out = apply_roi_mask(img, 0, tmp_path)
    assert out[5, 10] == 0  # right half masked out

    # Redraw a wider polygon covering the whole frame; bump mtime so the
    # cache (keyed on it) actually sees a change on fast filesystems.
    os.utime(mask_path, (time.time() + 1, time.time() + 1))
    mask_path.write_text("0 0\n11 0\n11 9\n0 9\n")
    out2 = apply_roi_mask(img, 0, tmp_path)
    assert out2[5, 10] == 200  # now inside the redrawn polygon

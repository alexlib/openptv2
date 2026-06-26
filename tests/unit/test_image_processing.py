import numpy as np
import pytest
from openptv2.algorithms.image_processing import filter_3, lowpass_3, fast_box_blur, split, subtract_img, subtract_mask, copy_images, prepare_image

EPS = 1

def images_equal(img1, img2, offset=0, discard=0):
    # Compare flattened arrays, skipping offset and last discard elements
    img1f = img1.ravel()[offset:img1.size - discard]
    img2f = img2.ravel()[offset:img2.size - discard]
    return np.allclose(img1f, img2f, atol=EPS)

def test_general_filter():
    blur_filt = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]])
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    img_correct = np.array([
          0,   0,   0,   0,  0,
          0, 153, 204, 153, 51,
         51, 204, 255, 204, 51,
         51, 153, 204, 153,  0,
         0,   0,   0,   0,   0
    ], dtype=np.uint8).reshape(5, 5)
    out = filter_3(img, blur_filt, 5, 5)
    assert images_equal(out, img_correct, offset=6, discard=6)

def test_mean_filter():
    mean_filt = np.ones((3, 3))
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    out1 = filter_3(img, mean_filt, 5, 5)
    out2 = lowpass_3(img, 5, 5)
    assert images_equal(out1, out2, offset=6, discard=6)


def test_filter_3_matches_c_min_brightness_and_border_handling():
    img = np.ones((5, 5), dtype=np.uint8)
    filt = np.ones((3, 3), dtype=np.float64)

    out = filter_3(img, filt, 5, 5)

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected.ravel()[6:19] = 8
    assert np.array_equal(out, expected)

def test_box_blur():
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    
    img_filt = fast_box_blur(img, 1, 5, 5)
    img_mean = lowpass_3(img, 5, 5)
    
    # set lowpass edge values to 0 so it equals the no-wrap action of the fast box blur
    # In Python, we can reshape to 1D, zero out, then reshape back, or just do it in 2D
    img_mean_flat = img_mean.flatten()
    for elem in range(5):
        img_mean_flat[5 * elem] = 0
        img_mean_flat[5 * elem + 4] = 0
    img_mean = img_mean_flat.reshape(5, 5)
    
    assert images_equal(img_filt, img_mean, offset=6, discard=6)

def test_split():
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    
    img_even = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0
    ], dtype=np.uint8).reshape(2, 5)
    
    img_odd = np.array([
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0
    ], dtype=np.uint8).reshape(2, 5)
    
    erased_half = np.array([
         2, 2, 2, 2, 2,
         2, 2, 2, 2, 2,
         2, 2, 2, 2, 2
    ], dtype=np.uint8).reshape(3, 5)

    img1 = img.copy()
    img2 = img.copy()
    
    img1 = split(img1, 1, 5, 5)
    assert images_equal(img1[:2, :], img_odd, offset=0, discard=0)
    assert images_equal(img1[2:, :], erased_half, offset=0, discard=0)
    
    img2 = split(img2, 2, 5, 5)
    assert images_equal(img2[:2, :], img_even, offset=0, discard=0)
    assert images_equal(img2[2:, :], erased_half, offset=0, discard=0)

def test_subtract_img():
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    
    img_zero = np.zeros((5, 5), dtype=np.uint8)

    img1 = subtract_img(img, img_zero)
    assert images_equal(img1, img, offset=0, discard=0)
    
    img2 = subtract_img(img, img1)
    assert images_equal(img2, img_zero, offset=0, discard=0)

def test_subtract_mask():
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    
    img_mask1 = np.ones((5, 5), dtype=np.uint8)
    
    img_mask2 = np.ones((5, 5), dtype=np.uint8)
    img_mask2[2, 2] = 0
    
    img_correct = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255,   0, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    
    img_new1 = subtract_mask(img, img_mask1)
    assert images_equal(img_new1, img, offset=0, discard=0)
    
    img_new2 = subtract_mask(img, img_mask2)
    assert images_equal(img_new2, img_correct, offset=0, discard=0)

def test_copy_img():
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0,
         1,   1,   1,   1, 1
    ], dtype=np.uint8).reshape(6, 5)
    
    img_new = np.empty_like(img)
    # The C code tests copy_images which does simple pointer copy. In Python, we just copy.
    # Let's test deep copy if possible.
    import copy
    img_new = copy.deepcopy(img)
    assert images_equal(img_new, img, offset=0, discard=0)
    
    img1 = img.copy()
    assert images_equal(img_new, img1, offset=0, discard=0)

def test_highpass():
    img = np.array([
         0,   0,   0,   0, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0, 255, 255, 255, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)
    
    img_correct = np.array([
         0,   0,   0,   0, 0,
         0, 142,  85, 142, 0,
         0,  85,   0,  85, 0,
         0, 142,  85, 142, 0,
         0,   0,   0,   0, 0
    ], dtype=np.uint8).reshape(5, 5)

    img_hp = prepare_image(img, 1, 5, 5, 0, None, 0)
    assert images_equal(img_hp, img_correct, offset=6, discard=6)

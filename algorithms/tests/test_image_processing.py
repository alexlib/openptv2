import numpy as np
import pytest
from algorithms.image_processing import filter_3, lowpass_3

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

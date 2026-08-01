import numpy as np

from openptv2.algorithms.image_processing import (
    fast_box_blur,
    filter_3,
    lowpass_3,
    prepare_image,
)


def print_array(arr, name):
    print(
        f"{name} = np.array({arr.ravel().tolist()}, dtype=np.uint8).reshape({arr.shape})"
    )


# test_general_filter
blur_filt = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]])
img = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        255,
        255,
        255,
        0,
        0,
        255,
        255,
        255,
        0,
        0,
        255,
        255,
        255,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.uint8,
).reshape(5, 5)
print_array(filter_3(img, blur_filt, 5, 5), "test_general_filter_out")

# test_box_blur
print_array(fast_box_blur(img, 1, 5, 5), "test_box_blur_out")
print_array(lowpass_3(img, 5, 5), "test_box_blur_mean")

# test_highpass
print_array(prepare_image(img, 1, 5, 5, 0, None, 0), "test_highpass_out")

# test_filter_3_matches_c...
img_ones = np.ones((5, 5), dtype=np.uint8)
filt_ones = np.ones((3, 3), dtype=np.float64)
print_array(filter_3(img_ones, filt_ones, 5, 5), "test_filter_3_ones")

import numpy as np
from scipy.ndimage import convolve, uniform_filter


def filter_3_scipy(img, filt, imx, imy, min_brightness=8):
    filt_arr = np.asarray(filt, dtype=np.float64).reshape(3, 3)
    filt_sum = filt_arr.sum()
    if filt_sum == 0:
        raise ValueError("Filter kernel sum is zero")

    img_float = np.asarray(img, dtype=np.float64)
    res = convolve(img_float, filt_arr, mode='constant', cval=0.0)

    res = np.trunc(res / filt_sum)

    # In the original, boundaries were left as 0. Mode 'constant' leaves them as filtered.
    # We should leave a 1-pixel border as 0 if we want to match exactly, but the plan was to fix boundaries.

    res = np.clip(res, min_brightness, 255).astype(np.uint8)
    return res

def lowpass_3_scipy(img, imx, imy):
    img_float = np.asarray(img, dtype=np.float64)
    res = uniform_filter(img_float, size=3, mode='constant', cval=0.0)
    return res.astype(np.uint8)

def fast_box_blur_scipy(img, filt_span, imx, imy):
    size = 2 * filt_span + 1
    img_float = np.asarray(img, dtype=np.float64)
    # The original box blur is essentially a uniform filter.
    res = uniform_filter(img_float, size=size, mode='constant', cval=0.0)
    return res.astype(np.uint8)

if __name__ == "__main__":
    from openptv2.algorithms.image_processing import fast_box_blur, filter_3, lowpass_3

    img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    filt = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int32)

    res_orig = filter_3(img, filt, 10, 10)
    res_scipy = filter_3_scipy(img, filt, 10, 10)

    print("Orig filter_3:\n", res_orig)
    print("SciPy filter_3:\n", res_scipy)

    res_orig_lp = lowpass_3(img, 10, 10)
    res_scipy_lp = lowpass_3_scipy(img, 10, 10)

    print("Orig lowpass_3:\n", res_orig_lp)
    print("SciPy lowpass_3:\n", res_scipy_lp)

    res_orig_bb = fast_box_blur(img, 1, 10, 10)
    res_scipy_bb = fast_box_blur_scipy(img, 1, 10, 10)

    print("Orig fast_box_blur:\n", res_orig_bb)
    print("SciPy fast_box_blur:\n", res_scipy_bb)

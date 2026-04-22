"""Image processing operations using NumPy vectorized operations.

Translation of lib/src/image_processing.c and lib/include/image_processing.h.

Provides:
- 3x3 filtering (general and lowpass)
- Fast box blur
- Image arithmetic (subtract, copy)
- Interlaced image splitting
- Image preparation for particle detection
"""

import numpy as np
from pathlib import Path

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# --------------- Numba JIT kernels ---------------

@njit(cache=True, parallel=True)
def _filter_3_jit(src, filt_flat, filt_sum, imx, imy, min_brightness):
    image_size = imx * imy
    result = np.zeros(image_size, dtype=np.uint8)
    f00, f01, f02 = filt_flat[0], filt_flat[1], filt_flat[2]
    f10, f11, f12 = filt_flat[3], filt_flat[4], filt_flat[5]
    f20, f21, f22 = filt_flat[6], filt_flat[7], filt_flat[8]
    end = image_size - imx - 1
    for idx in prange(imx + 1, end):
        buf = (f00 * src[idx - imx - 1] + f01 * src[idx - imx] + f02 * src[idx - imx + 1]
             + f10 * src[idx - 1]       + f11 * src[idx]       + f12 * src[idx + 1]
             + f20 * src[idx + imx - 1] + f21 * src[idx + imx] + f22 * src[idx + imx + 1])
        buf = buf / filt_sum
        if buf < min_brightness:
            buf = min_brightness
        elif buf > 255.0:
            buf = 255.0
        result[idx] = np.uint8(buf)
    return result


@njit(cache=True, parallel=True)
def _lowpass_3_jit(src, imx, imy):
    image_size = imx * imy
    result = np.zeros(image_size, dtype=np.uint8)
    end = image_size - imx - 1
    for idx in prange(imx + 1, end):
        buf = (src[idx - imx - 1] + src[idx - imx] + src[idx - imx + 1]
             + src[idx - 1]       + src[idx]       + src[idx + 1]
             + src[idx + imx - 1] + src[idx + imx] + src[idx + imx + 1])
        result[idx] = buf // 9
    return result


@njit(cache=True, parallel=True)
def _box_blur_row_pass_jit(src_flat, row_accum, filt_span, imx, imy):
    """Row pass — parallelized across rows."""
    n = 2 * filt_span + 1
    for i in prange(imy):
        rs = i * imx
        accum = np.int32(src_flat[rs])
        row_accum[rs] = accum * n

        for j in range(1, filt_span + 1):
            if j >= imx:
                break
            li = 2 * j - 1
            ri = 2 * j
            if ri < imx:
                accum += src_flat[rs + li] + src_flat[rs + ri]
            elif li < imx:
                accum += src_flat[rs + li]
            m = 2 * j + 1
            row_accum[rs + j] = accum * n // m

        mid_end = imx - filt_span if imx > filt_span else imx
        for j in range(filt_span + 1, mid_end):
            accum += src_flat[rs + j + filt_span] - src_flat[rs + j - filt_span - 1]
            row_accum[rs + j] = accum

        end_start = imx - filt_span
        if end_start < filt_span + 1:
            end_start = filt_span + 1
        left_ptr = imx - n
        m = n - 2
        for j in range(end_start, imx):
            if left_ptr >= 0 and left_ptr + 1 < imx:
                accum -= src_flat[rs + left_ptr] + src_flat[rs + left_ptr + 1]
            if m > 0:
                row_accum[rs + j] = accum * n // m
            left_ptr += 2
            m -= 2


@njit(cache=True)
def _fast_box_blur_jit(src_flat, filt_span, imx, imy):
    n = 2 * filt_span + 1
    nq = n * n
    image_size = imx * imy
    row_accum = np.zeros(image_size, dtype=np.int32)
    dest = np.zeros(image_size, dtype=np.uint8)

    _box_blur_row_pass_jit(src_flat, row_accum, filt_span, imx, imy)

    # Column pass (sequential — vertical dependency)
    col_accum = np.zeros(imx, dtype=np.int32)
    for j in range(imx):
        col_accum[j] = row_accum[j]
        dest[j] = np.uint8(col_accum[j] // n)

    for i in range(1, filt_span + 1):
        if i >= imy:
            break
        r1 = 2 * i - 1
        r2 = 2 * i
        ds = i * imx
        for j in range(imx):
            if r2 < imy:
                col_accum[j] += row_accum[r1 * imx + j] + row_accum[r2 * imx + j]
            elif r1 < imy:
                col_accum[j] += row_accum[r1 * imx + j]
            dest[ds + j] = np.uint8(n * col_accum[j] // nq // (2 * i + 1))

    ptr1_row = 0
    ptr2_row = n
    mid_upper = imy - filt_span
    if mid_upper < filt_span + 1:
        mid_upper = filt_span + 1
    for i in range(filt_span + 1, mid_upper):
        ds = i * imx
        if ptr2_row < imy:
            for j in range(imx):
                col_accum[j] += row_accum[ptr2_row * imx + j] - row_accum[ptr1_row * imx + j]
                dest[ds + j] = np.uint8(col_accum[j] // nq)
        else:
            for j in range(imx):
                dest[ds + j] = np.uint8(col_accum[j] // nq)
        ptr1_row += 1
        ptr2_row += 1

    for i in range(filt_span, 0, -1):
        r1 = imy - 2 * i - 1
        r2 = r1 + 1
        ds = (imy - i) * imx
        if 0 <= r1 < imy and r2 < imy:
            for j in range(imx):
                col_accum[j] -= row_accum[r1 * imx + j] + row_accum[r2 * imx + j]
                dest[ds + j] = np.uint8(n * col_accum[j] // nq // (2 * i + 1))
        else:
            for j in range(imx):
                dest[ds + j] = np.uint8(n * col_accum[j] // nq // (2 * i + 1))

    return dest


def filter_3(
    img: np.ndarray,
    filt: np.ndarray,
    imx: int,
    imy: int,
    min_brightness: int = 8,
) -> np.ndarray:
    """Apply a 3x3 filter kernel to an image.

    The first and last lines are not processed. Edges use wrap-around.
    Minimal brightness in output is enforced.

    Args:
        img: input image as 2D uint8 array (imy, imx).
        filt: 3x3 filter kernel.
        imx, imy: image dimensions.
        min_brightness: minimum output pixel value.

    Returns:
        Filtered image as 2D uint8 array.

    Raises:
        ValueError: if filter kernel is all zeros.
    """
    filt = np.asarray(filt, dtype=np.float64)
    filt_sum = filt.sum()
    if filt_sum == 0:
        raise ValueError("Filter kernel sum is zero")

    if HAS_NUMBA:
        src = np.ascontiguousarray(img, dtype=np.float64).ravel()
        return _filter_3_jit(src, filt.ravel(), filt_sum, imx, imy,
                             float(min_brightness)).reshape(imy, imx)

    src = np.asarray(img, dtype=np.float64).ravel()
    image_size = imx * imy
    result = np.zeros(image_size, dtype=np.float64)

    start = imx + 1
    end = image_size - imx - 1
    idx = np.arange(start, end)

    buf = (filt[0, 0] * src[idx - imx - 1] + filt[0, 1] * src[idx - imx] + filt[0, 2] * src[idx - imx + 1]
         + filt[1, 0] * src[idx - 1]       + filt[1, 1] * src[idx]       + filt[1, 2] * src[idx + 1]
         + filt[2, 0] * src[idx + imx - 1] + filt[2, 1] * src[idx + imx] + filt[2, 2] * src[idx + imx + 1])
    buf /= filt_sum
    np.clip(buf, min_brightness, 255, out=buf)
    result[start:end] = buf

    return result.reshape(imy, imx).astype(np.uint8)


def lowpass_3(img: np.ndarray, imx: int, imy: int) -> np.ndarray:
    """Apply a 3x3 averaging (box) filter.

    Simplified version of filter_3 with uniform kernel.

    Args:
        img: input image as 2D uint8 array (imy, imx).
        imx, imy: image dimensions.

    Returns:
        Blurred image as 2D uint8 array.
    """
    if HAS_NUMBA:
        src = np.ascontiguousarray(img, dtype=np.int16).ravel()
        return _lowpass_3_jit(src, imx, imy).reshape(imy, imx)

    src = np.asarray(img, dtype=np.float64).ravel()
    image_size = imx * imy
    result = np.zeros(image_size, dtype=np.float64)

    start = imx + 1
    end = image_size - imx - 1
    idx = np.arange(start, end)

    buf = (src[idx - imx - 1] + src[idx - imx] + src[idx - imx + 1]
         + src[idx - 1]       + src[idx]       + src[idx + 1]
         + src[idx + imx - 1] + src[idx + imx] + src[idx + imx + 1])
    result[start:end] = (buf / 9.0).astype(np.int64)

    return result.reshape(imy, imx).astype(np.uint8)


def _box_blur_row_pass(src, filt_span, imy, imx):
    """Row pass of fast_box_blur — vectorized across all rows."""
    n = 2 * filt_span + 1
    row_accum = np.zeros((imy, imx), dtype=np.float64)

    # col 0: accum = src[:, 0], row_accum[:, 0] = accum * n
    accum = src[:, 0].copy()
    row_accum[:, 0] = accum * n

    # Growing filter: cols 1..filt_span
    for j in range(1, min(filt_span + 1, imx)):
        left_idx = 2 * j - 1
        right_idx = 2 * j
        if right_idx < imx:
            accum += src[:, left_idx] + src[:, right_idx]
        elif left_idx < imx:
            accum += src[:, left_idx]
        m = 2 * j + 1
        row_accum[:, j] = accum * n / m

    # Middle: constant-size sliding window
    mid_end = imx - filt_span if imx > filt_span else imx
    for j in range(filt_span + 1, mid_end):
        accum += src[:, j + filt_span] - src[:, j - filt_span - 1]
        row_accum[:, j] = accum

    # Shrinking filter: last filt_span columns
    end_start = max(imx - filt_span, filt_span + 1)
    left_ptr = imx - n
    m = n - 2
    for j in range(end_start, imx):
        if left_ptr >= 0 and left_ptr + 1 < imx:
            accum -= src[:, left_ptr] + src[:, left_ptr + 1]
        if m > 0:
            row_accum[:, j] = accum * n / m
        left_ptr += 2
        m -= 2

    return row_accum


def fast_box_blur(
    img: np.ndarray,
    filt_span: int,
    imx: int,
    imy: int,
) -> np.ndarray:
    """Perform box blur using linear-time algorithm.

    Direct translation of C fast_box_blur. Uses sliding-window accumulation
    in both row and column directions.
    """
    if HAS_NUMBA:
        src_flat = np.ascontiguousarray(img, dtype=np.int32).ravel()
        return _fast_box_blur_jit(src_flat, filt_span, imx, imy).reshape(imy, imx)

    src = np.asarray(img, dtype=np.float64).reshape(imy, imx)
    n = 2 * filt_span + 1
    nq = n * n

    row_accum = _box_blur_row_pass(src, filt_span, imy, imx)

    # Column pass (already vectorized across columns via row slicing)
    col_accum = row_accum[0, :].copy()
    dest = np.zeros((imy, imx), dtype=np.float64)
    dest[0, :] = col_accum / n

    for i in range(1, min(filt_span + 1, imy)):
        r1 = 2 * i - 1
        r2 = 2 * i
        if r2 < imy:
            col_accum += row_accum[r1, :] + row_accum[r2, :]
        elif r1 < imy:
            col_accum += row_accum[r1, :]
        dest[i, :] = n * col_accum / nq / (2 * i + 1)

    ptr1_row = 0
    ptr2_row = n
    for i in range(filt_span + 1, max(imy - filt_span, filt_span + 1)):
        if ptr2_row < imy:
            col_accum += row_accum[ptr2_row, :] - row_accum[ptr1_row, :]
        dest[i, :] = col_accum / nq
        ptr1_row += 1
        ptr2_row += 1

    for i in range(filt_span, 0, -1):
        r1 = imy - 2 * i - 1
        r2 = r1 + 1
        if 0 <= r1 < imy and r2 < imy:
            col_accum -= row_accum[r1, :] + row_accum[r2, :]
        dest[imy - i, :] = n * col_accum / nq / (2 * i + 1)

    return dest.astype(np.uint8)


def split(img: np.ndarray, half_selector: int, imx: int, imy: int) -> np.ndarray:
    """Extract even or odd lines into first half of image.

    Used with interlaced cameras.

    Args:
        img: input image as 2D uint8 array (imy, imx).
        half_selector: 0=unchanged, 1=odd rows, 2=even rows.
        imx, imy: image dimensions.

    Returns:
        Modified image with selected lines crammed into first half,
        lower half filled with 2.
    """
    img = np.asarray(img, dtype=np.uint8).copy()

    if half_selector == 0:
        return img

    cond_offs = imx if half_selector % 2 else 0

    for row in range(imy // 2):
        img[row, :] = img[2 * row + (1 if cond_offs else 0), :]

    # Erase lower half with magic value 2
    img[imy // 2 :, :] = 2

    return img


def subtract_img(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Subtract img2 from img1, clamping to zero.

    Args:
        img1: minuend image (2D uint8).
        img2: subtrahend image (2D uint8).

    Returns:
        img1 - img2 clamped to [0, 255].
    """
    result = np.asarray(img1, dtype=np.int16) - np.asarray(img2, dtype=np.int16)
    return np.clip(result, 0, 255).astype(np.uint8)


def subtract_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image: zero pixels where mask is zero.

    Args:
        img: input image (2D uint8).
        mask: mask image (2D uint8), zero means clear.

    Returns:
        Masked image.
    """
    result = np.asarray(img, dtype=np.uint8).copy()
    result[np.asarray(mask, dtype=np.uint8) == 0] = 0
    return result


def prepare_image(
    img: np.ndarray,
    dim_lp: int,
    imx: int,
    imy: int,
    filter_hp: int = 0,
    filter_file: str | Path | None = None,
    chfield: int = 0,
) -> np.ndarray:
    """Prepare image for particle detection: smoothing + optional filtering.

    Steps:
    1. Low-pass blur (box filter)
    2. Subtract from original (high-pass result)
    3. Optionally split for interlaced
    4. Optionally apply additional filter

    Args:
        img: input image (2D uint8, shape (imy, imx)).
        dim_lp: half-width of low-pass blur.
        imx, imy: image dimensions.
        filter_hp: 0=none, 1=lowpass, 2=custom 3x3 from file.
        filter_file: path to filter matrix file (if filter_hp=2).
        chfield: 0=whole, 1=upper half, 2=lower half.

    Returns:
        Prepared image.
    """
    img = np.asarray(img, dtype=np.uint8)

    # Step 1: Low-pass blur
    img_lp = fast_box_blur(img, dim_lp, imx, imy)

    # Step 2: Subtract (high-pass)
    img_hp = subtract_img(img, img_lp)

    # Step 3: Handle interlaced
    if chfield in (1, 2):
        img_hp = split(img_hp, chfield, imx, imy)

    # Step 4: Additional filtering
    if filter_hp == 1:
        img_hp = lowpass_3(img_hp, imx, imy)
    elif filter_hp == 2:
        if filter_file is None:
            raise ValueError("filter_file required when filter_hp=2")
        filt = np.loadtxt(filter_file).reshape(3, 3)
        img_hp = filter_3(img_hp, filt, imx, imy)

    return img_hp


def copy_images(imgs):
    """Stub for copy_images: returns a copy of the input list of images."""
    return [np.copy(img) for img in imgs]

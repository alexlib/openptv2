"""Image processing operations using NumPy vectorized operations.

Translation of lib/src/image_processing.c and lib/include/image_processing.h.

Provides:
- 3x3 filtering (general and lowpass)
- Fast box blur
- Image arithmetic (subtract, copy)
- Interlaced image splitting
- Image preparation for particle detection
"""
import cython


import numpy as np
from pathlib import Path



@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def filter_3(
    img: np.ndarray,
    filt: np.ndarray,
    imx: cython.int,
    imy: cython.int,
    min_brightness: cython.int = 8,
) -> np.ndarray:
    """Apply the C 3x3 filter kernel with matching integer semantics."""
    filt = np.asarray(filt, dtype=np.float64).reshape(3, 3)
    filt_sum = float(filt.sum())
    if filt_sum == 0:
        raise ValueError("Filter kernel sum is zero")

    src = np.asarray(img, dtype=np.uint8).reshape(imy, imx).ravel()
    image_size = imx * imy
    result = np.zeros(image_size, dtype=np.uint8)

    start = imx + 1
    end = image_size - imx - 1
    for idx in range(start, end):
        total = (
            filt[0, 0] * src[idx - imx - 1]
            + filt[0, 1] * src[idx - imx]
            + filt[0, 2] * src[idx - imx + 1]
            + filt[1, 0] * src[idx - 1]
            + filt[1, 1] * src[idx]
            + filt[1, 2] * src[idx + 1]
            + filt[2, 0] * src[idx + imx - 1]
            + filt[2, 1] * src[idx + imx]
            + filt[2, 2] * src[idx + imx + 1]
        )
        buf = int(total / filt_sum)
        if buf > 255:
            buf = 255
        if buf < min_brightness:
            buf = min_brightness
        result[idx] = buf

    return result.reshape(imy, imx)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def lowpass_3(img: np.ndarray, imx: cython.int, imy: cython.int) -> np.ndarray:
    """Apply the C 3x3 low-pass filter with integer division."""
    src = np.asarray(img, dtype=np.uint8).reshape(imy, imx).ravel()
    image_size = imx * imy
    result = np.zeros(image_size, dtype=np.uint8)

    start = imx + 1
    end = image_size - imx - 1
    for idx in range(start, end):
        total = (
            int(src[idx])
            + int(src[idx - imx - 1])
            + int(src[idx - imx])
            + int(src[idx - imx + 1])
            + int(src[idx - 1])
            + int(src[idx + 1])
            + int(src[idx + imx - 1])
            + int(src[idx + imx])
            + int(src[idx + imx + 1])
        )
        result[idx] = total // 9

    return result.reshape(imy, imx)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def fast_box_blur(
    img: np.ndarray,
    filt_span: cython.int,
    imx: cython.int,
    imy: cython.int,
) -> np.ndarray:
    """Perform the C box blur with matching integer rounding."""
    src = np.asarray(img, dtype=np.uint8).reshape(imy, imx).ravel()
    n = 2 * filt_span + 1
    nq = n * n
    image_size = imx * imy
    row_accum = np.zeros(image_size, dtype=np.int64)
    col_accum = np.zeros(imx, dtype=np.int64)
    dest = np.zeros(image_size, dtype=np.uint8)

    for row in range(imy):
        row_start = row * imx
        accum = int(src[row_start])
        row_accum[row_start] = accum * n

        for col in range(1, min(filt_span + 1, imx)):
            left_idx = row_start + 2 * col - 1
            right_idx = row_start + 2 * col
            accum += int(src[left_idx]) + int(src[right_idx])
            m = 2 * col + 1
            row_accum[row_start + col] = accum * n // m

        for col in range(filt_span + 1, imx - filt_span):
            accum += int(src[row_start + col + filt_span]) - int(src[row_start + col - filt_span - 1])
            row_accum[row_start + col] = accum

        left_ptr = row_start + imx - n
        m = n - 2
        for col in range(max(imx - filt_span, filt_span + 1), imx):
            accum -= int(src[left_ptr]) + int(src[left_ptr + 1])
            row_accum[row_start + col] = accum * n // m
            left_ptr += 2
            m -= 2

    for col in range(imx):
        col_accum[col] = row_accum[col]
        dest[col] = col_accum[col] // n

    for row in range(1, min(filt_span + 1, imy)):
        ptr1 = (2 * row - 1) * imx
        ptr2 = ptr1 + imx
        out = row * imx
        for col in range(imx):
            col_accum[col] += row_accum[ptr1 + col] + row_accum[ptr2 + col]
            dest[out + col] = (n * col_accum[col]) // nq // (2 * row + 1)

    ptr1 = 0
    ptr2 = imx * n
    for row in range(filt_span + 1, imy - filt_span):
        out = row * imx
        for col in range(imx):
            col_accum[col] += row_accum[ptr2 + col] - row_accum[ptr1 + col]
            dest[out + col] = col_accum[col] // nq
        ptr1 += imx
        ptr2 += imx

    for row in range(filt_span, 0, -1):
        ptr1 = (imy - 2 * row - 1) * imx
        ptr2 = ptr1 + imx
        out = (imy - row) * imx
        for col in range(imx):
            col_accum[col] -= row_accum[ptr1 + col] + row_accum[ptr2 + col]
            dest[out + col] = (n * col_accum[col]) // nq // (2 * row + 1)

    return dest.reshape(imy, imx)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def split(img: np.ndarray, half_selector: cython.int, imx: cython.int, imy: cython.int) -> np.ndarray:
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


@cython.ccall
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


@cython.ccall
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


@cython.ccall
def prepare_image(
    img: np.ndarray,
    dim_lp: cython.int,
    imx: cython.int,
    imy: cython.int,
    filter_hp: cython.int = 0,
    filter_file: str | Path | None = None,
    chfield: cython.int = 0,
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


@cython.ccall
def copy_images(src, dest=None, imx=None, imy=None):
    """Copy image data from src to dest, matching C's copy_images semantics.

    If dest is provided, copies src into dest in-place and returns dest.
    If only src is provided (a list of images), returns copies (legacy API).
    """
    if isinstance(src, list):
        return [np.copy(img) for img in src]
    if dest is not None:
        np.copyto(dest, src)
        return dest
    return src.copy()


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled

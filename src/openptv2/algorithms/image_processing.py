"""Image processing operations using Cython-optimized kernels.

Translation of lib/src/image_processing.c and lib/include/image_processing.h.

Provides Cython Pure Python implementations using typed memoryviews,
cdivision, and contiguous layout for maximum compiled speed while
remaining valid Python when the .so is absent.

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
@cython.cdivision(True)
@cython.locals(
    filt_sum=cython.double,
    image_size=cython.int,
    start=cython.int,
    end=cython.int,
    idx=cython.int,
    total=cython.double,
    buf=cython.int,
    f0=cython.double,
    f1=cython.double,
    f2=cython.double,
    f3=cython.double,
    f4=cython.double,
    f5=cython.double,
    f6=cython.double,
    f7=cython.double,
    f8=cython.double,
    src_mv=cython.uchar[:],
    result_mv=cython.uchar[:],
)
def filter_3(
    img: np.ndarray,
    filt: np.ndarray,
    imx: cython.int,
    imy: cython.int,
    min_brightness: cython.int = 8,
) -> object:
    """Apply the C 3x3 filter kernel with matching integer semantics."""
    filt_arr = np.asarray(filt, dtype=np.float64).ravel()
    f0 = filt_arr[0]
    f1 = filt_arr[1]
    f2 = filt_arr[2]
    f3 = filt_arr[3]
    f4 = filt_arr[4]
    f5 = filt_arr[5]
    f6 = filt_arr[6]
    f7 = filt_arr[7]
    f8 = filt_arr[8]
    filt_sum = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    if filt_sum == 0:
        raise ValueError("Filter kernel sum is zero")

    src_arr = np.asarray(img, dtype=np.uint8).reshape(imy, imx).ravel()
    src_mv = src_arr
    image_size = imx * imy
    result_arr = np.zeros(image_size, dtype=np.uint8)
    result_mv = result_arr

    start = imx + 1
    end = image_size - imx - 1
    for idx in range(start, end):
        total = (
            f0 * src_mv[idx - imx - 1]
            + f1 * src_mv[idx - imx]
            + f2 * src_mv[idx - imx + 1]
            + f3 * src_mv[idx - 1]
            + f4 * src_mv[idx]
            + f5 * src_mv[idx + 1]
            + f6 * src_mv[idx + imx - 1]
            + f7 * src_mv[idx + imx]
            + f8 * src_mv[idx + imx + 1]
        )
        buf = int(total / filt_sum)
        if buf > 255:
            buf = 255
        if buf < min_brightness:
            buf = min_brightness
        result_mv[idx] = buf

    return result_arr.reshape(imy, imx)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.locals(
    image_size=cython.int,
    start=cython.int,
    end=cython.int,
    idx=cython.int,
    total=cython.int,
    src_mv=cython.uchar[:],
    result_mv=cython.uchar[:],
)
def lowpass_3(img: np.ndarray, imx: cython.int, imy: cython.int) -> object:
    """Apply the C 3x3 low-pass filter with integer division."""
    src_arr = np.asarray(img, dtype=np.uint8).reshape(imy, imx).ravel()
    src_mv = src_arr
    image_size = imx * imy
    result_arr = np.zeros(image_size, dtype=np.uint8)
    result_mv = result_arr

    start = imx + 1
    end = image_size - imx - 1
    for idx in range(start, end):
        total = (
            int(src_mv[idx])
            + int(src_mv[idx - imx - 1])
            + int(src_mv[idx - imx])
            + int(src_mv[idx - imx + 1])
            + int(src_mv[idx - 1])
            + int(src_mv[idx + 1])
            + int(src_mv[idx + imx - 1])
            + int(src_mv[idx + imx])
            + int(src_mv[idx + imx + 1])
        )
        result_mv[idx] = total // 9

    return result_arr.reshape(imy, imx)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.locals(
    filt_span=cython.int,
    imx=cython.int,
    imy=cython.int,
    n=cython.int,
    nq=cython.int,
    image_size=cython.int,
    row=cython.int,
    col=cython.int,
    row_start=cython.int,
    accum=cython.longlong,
    left_idx=cython.int,
    right_idx=cython.int,
    m=cython.int,
    left_ptr=cython.int,
    ptr1=cython.int,
    ptr2=cython.int,
    out=cython.int,
    src_mv=cython.longlong[:],
    row_accum_mv=cython.longlong[:],
    col_accum_mv=cython.longlong[:],
    col_acc=cython.longlong,
    dest_mv=cython.uchar[:],
)
def fast_box_blur(
    img: np.ndarray,
    filt_span: cython.int,
    imx: cython.int,
    imy: cython.int,
) -> object:
    """Perform the C box blur with matching integer rounding."""
    src_arr = np.asarray(img, dtype=np.uint8).reshape(imy, imx).ravel()
    # Widened int64 view to prevent uint8 overflow during accumulation
    src_mv = src_arr.astype(np.int64)
    n = 2 * filt_span + 1
    nq = n * n
    image_size = imx * imy
    row_accum_arr = np.zeros(image_size, dtype=np.int64)
    row_accum_mv = row_accum_arr
    col_accum_arr = np.zeros(imx, dtype=np.int64)
    col_accum_mv = col_accum_arr
    dest_arr = np.zeros(image_size, dtype=np.uint8)
    dest_mv = dest_arr

    # Horizontal pass
    for row in range(imy):
        row_start = row * imx
        accum = src_mv[row_start]
        row_accum_mv[row_start] = accum * n

        for col in range(1, min(filt_span + 1, imx)):
            left_idx = row_start + 2 * col - 1
            right_idx = row_start + 2 * col
            accum += src_mv[left_idx] + src_mv[right_idx]
            m = 2 * col + 1
            row_accum_mv[row_start + col] = accum * n // m

        for col in range(filt_span + 1, imx - filt_span):
            accum += (
                src_mv[row_start + col + filt_span]
                - src_mv[row_start + col - filt_span - 1]
            )
            row_accum_mv[row_start + col] = accum

        left_ptr = row_start + imx - n
        m = n - 2
        for col in range(max(imx - filt_span, filt_span + 1), imx):
            accum -= src_mv[left_ptr] + src_mv[left_ptr + 1]
            row_accum_mv[row_start + col] = accum * n // m
            left_ptr += 2
            m -= 2

    # Vertical pass — first rows (top ramp)
    for col in range(imx):
        col_acc = row_accum_mv[col]
        col_accum_mv[col] = col_acc
        dest_mv[col] = col_acc // n

    for row in range(1, min(filt_span + 1, imy)):
        ptr1 = (2 * row - 1) * imx
        ptr2 = ptr1 + imx
        out = row * imx
        for col in range(imx):
            col_acc = (
                col_accum_mv[col] + row_accum_mv[ptr1 + col] + row_accum_mv[ptr2 + col]
            )
            col_accum_mv[col] = col_acc
            dest_mv[out + col] = (n * col_acc) // nq // (2 * row + 1)

    # Vertical pass — middle rows (full window)
    ptr1 = 0
    ptr2 = imx * n
    for row in range(filt_span + 1, imy - filt_span):
        out = row * imx
        for col in range(imx):
            col_acc = (
                col_accum_mv[col] + row_accum_mv[ptr2 + col] - row_accum_mv[ptr1 + col]
            )
            col_accum_mv[col] = col_acc
            dest_mv[out + col] = col_acc // nq
        ptr1 += imx
        ptr2 += imx

    # Vertical pass — last rows (bottom ramp)
    for row in range(filt_span, 0, -1):
        ptr1 = (imy - 2 * row - 1) * imx
        ptr2 = ptr1 + imx
        out = (imy - row) * imx
        for col in range(imx):
            col_acc = (
                col_accum_mv[col] - row_accum_mv[ptr1 + col] - row_accum_mv[ptr2 + col]
            )
            col_accum_mv[col] = col_acc
            dest_mv[out + col] = (n * col_acc) // nq // (2 * row + 1)

    return dest_arr.reshape(imy, imx)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(
    half=cython.int,
    row=cython.int,
    imx=cython.int,
    imy=cython.int,
    src_mv=cython.uchar[:, ::1],
    dst_mv=cython.uchar[:, ::1],
)
def split(
    img: np.ndarray, half_selector: cython.int, imx: cython.int, imy: cython.int
) -> object:
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
    if half_selector == 0:
        return np.asarray(img, dtype=np.uint8).copy()

    src_arr = np.asarray(img, dtype=np.uint8).reshape(imy, imx).copy()
    src_mv = src_arr
    half = imy // 2
    # Hoist conditional out of inner loop: half_selector 1→odd(offset=1), 2→even(offset=0)
    row_offset: cython.int = 1 if half_selector == 1 else 0

    for row in range(half):
        src_row: cython.int = 2 * row + row_offset
        for col in range(imx):
            src_mv[row, col] = src_mv[src_row, col]

    for row in range(half, imy):
        for col in range(imx):
            src_mv[row, col] = 2

    return src_arr


@cython.ccall
def subtract_img(img1: np.ndarray, img2: np.ndarray) -> object:
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
def subtract_mask(img: np.ndarray, mask: np.ndarray) -> object:
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
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.locals(
    dim_lp=cython.int,
    imx=cython.int,
    imy=cython.int,
    filter_hp=cython.int,
    chfield=cython.int,
)
def prepare_image(
    img: np.ndarray,
    dim_lp: cython.int,
    imx: cython.int,
    imy: cython.int,
    filter_hp: cython.int = 0,
    filter_file: str | Path | None = None,
    chfield: cython.int = 0,
) -> object:
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
        img_hp = split(img_hp, half_selector=chfield, imx=imx, imy=imy)

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

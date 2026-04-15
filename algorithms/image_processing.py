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
    img = np.asarray(img, dtype=np.float64)
    img = np.asarray(img, dtype=np.float64)
    filt = np.asarray(filt, dtype=np.float64)

    filt_sum = filt.sum()
    if filt_sum == 0:
        raise ValueError("Filter kernel sum is zero")

    result = np.zeros_like(img)

    # Process the full image, but for borders, use partial neighborhoods (as in C)
    for i in range(imy):
        for j in range(imx):
            # For border pixels, use the nearest valid 3x3 region (pad with edge values)
            i0 = max(i - 1, 0)
            i1 = min(i + 2, imy)
            j0 = max(j - 1, 0)
            j1 = min(j + 2, imx)
            region = img[i0:i1, j0:j1]
            # Pad region to 3x3 if at edge/corner
            padded = np.full((3, 3), 0.0)
            padded[
                (1 - (i - i0)) : (1 + (i1 - i)),
                (1 - (j - j0)) : (1 + (j1 - j)),
            ] = region
            val = (padded * filt).sum() / filt_sum
            val = np.clip(val, min_brightness, 255)
            result[i, j] = val
    return result.astype(np.uint8)


def lowpass_3(img: np.ndarray, imx: int, imy: int) -> np.ndarray:
    """Apply a 3x3 averaging (box) filter.

    Simplified version of filter_3 with uniform kernel.

    Args:
        img: input image as 2D uint8 array (imy, imx).
        imx, imy: image dimensions.

    Returns:
        Blurred image as 2D uint8 array.
    """
    img = np.asarray(img, dtype=np.float64)
    result = np.zeros_like(img)
    # Use same border logic as filter_3: for each pixel, use nearest valid 3x3 region, pad with zeros
    for i in range(imy):
        for j in range(imx):
            i0 = max(i - 1, 0)
            i1 = min(i + 2, imy)
            j0 = max(j - 1, 0)
            j1 = min(j + 2, imx)
            region = img[i0:i1, j0:j1]
            padded = np.full((3, 3), 0.0)
            padded[
                (1 - (i - i0)) : (1 + (i1 - i)),
                (1 - (j - j0)) : (1 + (j1 - j)),
            ] = region
            val = padded.sum() / 9.0
            val = np.clip(val, 0, 255)
            result[i, j] = val
    return result.astype(np.uint8)


def fast_box_blur(
    img: np.ndarray,
    filt_span: int,
    imx: int,
    imy: int,
) -> np.ndarray:
    """Perform box blur using linear-time algorithm.

    Equivalent to an all-ones kernel of size (2*filt_span+1)^2,
    but runs in O(filt_span * imx * imy) instead of O(filt_span^2 * imx * imy).

    Args:
        img: input image as 2D uint8 array (imy, imx).
        filt_span: half-width of blur kernel (total size = 2*filt_span+1).
        imx, imy: image dimensions.

    Returns:
        Blurred image as 2D uint8 array.
    """
    img = np.asarray(img, dtype=np.float64)
    n = 2 * filt_span + 1
    nq = n * n

    # Horizontal pass
    row_accum = np.zeros((imy, imx), dtype=np.float64)

    for i in range(imy):
        row = img[i, :]
        # Cumulative sum approach
        cumsum = np.cumsum(np.pad(row, filt_span, mode="edge"))
        row_accum[i, :] = (cumsum[n:] - cumsum[:-n]) / n

    # Vertical pass
    col_accum = np.zeros(imx, dtype=np.float64)
    result = np.zeros_like(img)

    # First line
    col_accum = row_accum[0, :].copy()
    result[0, :] = col_accum / n

    # Middle lines with sliding window
    for i in range(1, imy):
        if i <= filt_span:
            col_accum += row_accum[i, :]
            result[i, :] = n * col_accum / nq / (2 * i + 1)
        elif i < imy - filt_span:
            col_accum += row_accum[i, :] - row_accum[i - n, :]
            result[i, :] = col_accum / nq
        else:
            col_accum -= row_accum[i - n, :]
            remaining = imy - i
            result[i, :] = n * col_accum / nq / (2 * remaining + 1)

    return result.astype(np.uint8)


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

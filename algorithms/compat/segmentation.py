"""
Segmentation compatibility wrapper providing optv-like API.
"""

from algorithms.segmentation import targ_rec
from algorithms.compat.tracking_framebuf import TargetArray


def target_recognition(img, tpar, cam, cpar, subrange_x=None, subrange_y=None):
    """
    Recognize targets in image using segmentation.

    Args:
        img: Input image array
        tpar: TargetParams instance
        cam: Camera index (unused, for API compatibility)
        cpar: ControlParams instance
        subrange_x: Optional x range tuple (xmin, xmax)
        subrange_y: Optional y range tuple (ymin, ymax)

    Returns:
        TargetArray containing detected targets
    """
    # Get image size
    imx, imy = cpar.get_image_size()

    # Set default subranges
    if subrange_x is None:
        xmin, xmax = 1, imx - 1
    else:
        xmin, xmax = subrange_x

    if subrange_y is None:
        ymin, ymax = 1, imy - 1
    else:
        ymin, ymax = subrange_y

    # Get target parameters
    gvthres = tpar.get_grey_thresholds()[cam]  # Use camera-specific threshold
    discont = tpar.get_max_discontinuity()
    nnmin, nnmax = tpar.get_pixel_count_bounds()
    nxmin, nxmax = tpar.get_xsize_bounds()
    nymin, nymax = tpar.get_ysize_bounds()
    sumg_min = tpar.get_min_sum_grey()

    # Call algorithms targ_rec
    targets = targ_rec(
        img=img,
        gvthres=gvthres,
        discont=discont,
        nnmin=nnmin,
        nnmax=nnmax,
        nxmin=nxmin,
        nxmax=nxmax,
        nymin=nymin,
        nymax=nymax,
        sumg_min=sumg_min,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    return TargetArray(targets)

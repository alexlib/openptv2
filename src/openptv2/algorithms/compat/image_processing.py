"""
Image processing compatibility wrapper providing optv-like API.
"""

import numpy as np
from openptv2.algorithms.image_processing import prepare_image


def preprocess_image(img, filter_hp, cpar, lowpass_dim=1, filter_file=None, output=None):
    """
    Preprocess image with high-pass and low-pass filters.

    Args:
        img: Input image array
        filter_hp: High-pass filter flag (0 or 1)
        cpar: ControlParams instance
        lowpass_dim: Lowpass filter dimension (default 1)
        filter_file: Optional filter file path
        output: Optional output array

    Returns:
        Processed image array
    """
    imx, imy = cpar.get_image_size()
    chfield = cpar.get_chfield()

    result = prepare_image(
        img=img,
        dim_lp=lowpass_dim,
        imx=imx,
        imy=imy,
        filter_hp=filter_hp,
        filter_file=filter_file,
        chfield=chfield
    )

    if output is not None:
        output[:] = result
        return output
    return result

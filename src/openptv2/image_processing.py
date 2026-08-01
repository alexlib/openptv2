"""Streamlined image preprocessing API."""

from openptv2.algorithms.image_processing import prepare_image


def preprocess_image(
    img, filter_hp, cpar, lowpass_dim=1, filter_file=None, output=None
):
    """
    Preprocess image with high-pass and low-pass filters.

    Args:
        img: Input image array
        filter_hp: High-pass filter flag (0 or 1)
        cpar: ControlPar instance
        lowpass_dim: Lowpass filter dimension (default 1)
        filter_file: Optional filter file path
        output: Optional output array

    Returns:
        Processed image array
    """
    raw_cpar = cpar._cpar if hasattr(cpar, "_cpar") else cpar

    if hasattr(raw_cpar, "get_image_size"):
        imx, imy = raw_cpar.get_image_size()
    else:
        imx, imy = raw_cpar.imx, raw_cpar.imy

    if hasattr(raw_cpar, "get_chfield"):
        chfield = raw_cpar.get_chfield()
    else:
        chfield = raw_cpar.chfield

    result = prepare_image(
        img=img,
        dim_lp=lowpass_dim,
        imx=imx,
        imy=imy,
        filter_hp=filter_hp,
        filter_file=filter_file,
        chfield=chfield,
    )

    if output is not None:
        output[:] = result
        return output
    return result


__all__ = ["preprocess_image"]

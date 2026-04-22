"""Transforms module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.transforms import (
            convert_arr_pixel_to_metric,
            convert_arr_metric_to_pixel,
            correct_arr_brown_affine,
            distort_arr_brown_affine,
            distorted_to_flat,
        )
    except ImportError:
        from algorithms.compat.transforms import (
            convert_arr_pixel_to_metric,
            convert_arr_metric_to_pixel,
            correct_arr_brown_affine,
            distort_arr_brown_affine,
            distorted_to_flat,
        )
else:
    from algorithms.compat.transforms import (
        convert_arr_pixel_to_metric,
        convert_arr_metric_to_pixel,
        correct_arr_brown_affine,
        distort_arr_brown_affine,
        distorted_to_flat,
    )

__all__ = [
    'convert_arr_pixel_to_metric',
    'convert_arr_metric_to_pixel',
    'correct_arr_brown_affine',
    'distort_arr_brown_affine',
    'distorted_to_flat',
]

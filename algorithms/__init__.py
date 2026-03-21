"""
Python/Numba fallback engine for openptv2.

This module provides pure Python implementations of OpenPTV algorithms
using NumPy and Numba for JIT acceleration. It produces identical results
to the C/Cython engine (within floating-point tolerance) and is used for:

- Debugging and development
- Real-time visualization of algorithm steps
- Algorithm prototyping before C implementation
- Systems where C compilation is not available

This implementation is based on openptv-python:
https://github.com/openptv/openptv-python
"""

try:
    from .version import __version__
except ImportError:  # pragma: no cover
    __version__ = "999"

# Core algorithm modules
from . import (
    calibration,
    correspondences,
    epi,
    image_processing,
    imgcoord,
    multimed,
    orientation,
    parameters,
    ray_tracing,
    segmentation,
    sortgrid,
    track,
    tracking_frame_buf,
    tracking_run,
    trafo,
    vec_utils,
    constants,
    find_candidate,
)

# Native compatibility layer (for optional optv backend)
from . import _native_compat
from . import _native_convert

__all__ = [
    "__version__",
    "calibration",
    "correspondences",
    "epi",
    "image_processing",
    "imgcoord",
    "multimed",
    "orientation",
    "parameters",
    "ray_tracing",
    "segmentation",
    "sortgrid",
    "track",
    "tracking_frame_buf",
    "tracking_run",
    "trafo",
    "vec_utils",
    "constants",
    "find_candidate",
    "_native_compat",
    "_native_convert",
]

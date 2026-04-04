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

# Core algorithm modules — imported lazily to avoid hard dependency on numba.
# The Python/Numba engine is optional; the C/Cython engine works without it.
_NUMBA_AVAILABLE = False


def __getattr__(name):
    """Lazy-load algorithm modules to avoid requiring numba at import time."""
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is False:
        try:
            import numba  # noqa: F401

            _NUMBA_AVAILABLE = True
        except ImportError:
            _NUMBA_AVAILABLE = None  # Mark as checked but unavailable

    algorithm_modules = [
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

    if name in algorithm_modules:
        if _NUMBA_AVAILABLE is None:
            raise ImportError(
                f"algorithms.{name} requires numba, which is not installed. "
                "Install with: pip install openptv2[numba]"
            )
        import importlib

        return importlib.import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

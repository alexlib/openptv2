"""One place that decides how a >8-bit image becomes the uint8 the detectors want.

Every detection threshold in openptv2 -- ``targ_rec.gvthres``,
``targ_rec.sumg_min``, ``detect_plate.gvth_*``, ``detect_plate.sum_grey`` -- is
written on a 0-255 scale, while cameras routinely deliver 16-bit frames.  The
mapping between the two decides what those numbers mean, and it used to be made
in three places that disagreed:

* ``segmentation._load_image_array`` and the GUI used ``img_as_ubyte`` -- a
  FIXED full-range map, 65535 -> 255;
* ``detect_plate`` used a per-image percentile STRETCH;
* the pure-Python ``targ_rec`` fallback did an unsafe cast that wraps mod 256
  (2112 -> 64, 65520 -> 240), so a compiled and an interpreted build could
  disagree catastrophically on the same input.

The practical cost of that on the Illmenau rig: a ``gvthres`` of 3, sensible
under a stretch, sat far below the noise floor under the fixed map, which found
550-1044 blobs per camera instead of a few hundred particles, saturated the
epipolar candidate lists and collapsed correspondences to a handful of
quadruplets -- with a calibration that was provably good.

So the rule is now named, recorded in the parameter file, and applied here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Mode = Literal["fixed", "stretch", "range"]

MODES: tuple[str, ...] = ("fixed", "stretch", "range")
DEFAULT_PERCENTILES = (1.0, 99.5)


def describe(
    mode: str,
    lo: float | None = None,
    hi: float | None = None,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> str:
    """One line saying what a threshold on the result actually means."""
    if mode == "fixed":
        return (
            "fixed full-range map (dtype max -> 255): a grey value always "
            "maps to the same 8-bit value, so thresholds are comparable "
            "across frames and cameras"
        )
    if mode == "stretch":
        return (
            f"per-image percentile stretch ({percentiles[0]}-{percentiles[1]} "
            "-> 0-255): adapts to drifting illumination, so a threshold "
            "means something slightly different in every frame"
        )
    if mode == "range":
        return (
            f"explicit range [{lo}, {hi}] -> 0-255: absolute and tuned to "
            "the sensor's real range"
        )
    raise ValueError(f"unknown grey-scaling mode {mode!r}; expected one of {MODES}")


def to_uint8(
    img,
    mode: Mode = "fixed",
    *,
    lo: float | None = None,
    hi: float | None = None,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> np.ndarray:
    """Convert an image to C-contiguous 2D ``uint8`` by an explicit rule.

    Args:
        img: 2D or 3D array. A 3D array is averaged over its last axis.
        mode: ``"fixed"``, ``"stretch"`` or ``"range"`` -- see :func:`describe`.
        lo, hi: bounds in the SOURCE dtype's units, required for ``"range"``.
        percentiles: low/high percentiles for ``"stretch"``.

    Returns:
        C-contiguous ``(H, W)`` ``uint8``.

    Never casts unsafely: a value outside the mapped range clips to 0 or 255
    rather than wrapping.  That is the whole point of this function existing.
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    if mode == "fixed":
        # skimage's img_as_ubyte downcasts integers with a pure BIT SHIFT, not a
        # rescale: 65535 >> 8 == 255, but also 511 >> 8 == 1 where a rescale
        # would give 2.  The loader paths and the GUI have always used it, so
        # match it exactly rather than approximately -- a one-grey-level drift
        # would silently move every threshold.
        if np.issubdtype(arr.dtype, np.unsignedinteger):
            shift = arr.dtype.itemsize * 8 - 8
            if shift <= 0:
                return np.ascontiguousarray(arr.astype(np.uint8))
            return np.ascontiguousarray((arr >> shift).astype(np.uint8))
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            f_lo, f_hi = float(info.min), float(info.max)
        else:
            # float images follow the skimage convention, 0..1
            f_lo, f_hi = 0.0, 1.0
    elif mode == "stretch":
        f_lo = float(np.percentile(arr, percentiles[0]))
        f_hi = float(np.percentile(arr, percentiles[1]))
        if f_hi <= f_lo:
            f_hi = float(arr.max()) or 1.0
    elif mode == "range":
        if lo is None or hi is None:
            raise ValueError("mode='range' needs both lo and hi")
        f_lo, f_hi = float(lo), float(hi)
        if f_hi <= f_lo:
            raise ValueError(f"grey_range must have hi > lo, got [{lo}, {hi}]")
    else:
        raise ValueError(f"unknown grey-scaling mode {mode!r}; expected one of {MODES}")

    # 'stretch' ends with a truncating cast, reproducing detect_plate exactly.
    scaled = (arr.astype(np.float64) - f_lo) / (f_hi - f_lo) * 255.0
    return np.ascontiguousarray(np.clip(scaled, 0.0, 255.0).astype(np.uint8))


def from_parameters(par: dict | None) -> dict:
    """Read the scaling rule out of a parsed parameter file.

    ``ControlPar`` is a Cython cclass with declared fields, so the rule is NOT
    bolted onto it -- it is read here and passed explicitly to whatever converts
    an image.  That keeps the hot path untouched and makes the dependency
    visible at every call site.

    A parameter file that says nothing keeps the historical behaviour.
    """
    ptv = ((par or {}).get("ptv") or {}) if isinstance(par, dict) else {}
    mode = str(ptv.get("grey_scaling") or "fixed")
    if mode not in MODES:
        raise ValueError(
            f"ptv.grey_scaling is {mode!r}; expected one of {MODES}. "
            f"See docs/plans/2026-08-31-16bit-image-handling.md"
        )
    rng = ptv.get("grey_range")
    lo, hi = (float(rng[0]), float(rng[1])) if rng else (None, None)
    if mode == "range" and (lo is None or hi is None):
        raise ValueError("ptv.grey_scaling: range needs ptv.grey_range: [lo, hi]")
    return {"mode": mode, "lo": lo, "hi": hi}


def suggest_range(
    img, percentiles: tuple[float, float] = (0.5, 99.9)
) -> tuple[int, int]:
    """A sensible ``grey_range`` for a dataset, from one representative image.

    Use it to turn an adaptive stretch into a fixed, reproducible mapping once
    the sensor's real range is known.
    """
    arr = np.asarray(img)
    return (
        int(np.percentile(arr, percentiles[0])),
        int(np.percentile(arr, percentiles[1])),
    )

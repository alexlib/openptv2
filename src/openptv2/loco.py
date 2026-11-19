"""Leave-one-camera-out (LOCO) consistency: a validation score that needs no truth.

For every 3D point seen by at least four cameras, each camera in turn is held out: the point is
triangulated from the other cameras' pixel positions, projected into the held-out camera, and compared
with the pixel position observed there. The residual (px) measures how well the observations of one
particle agree across cameras; a merged blob (shifted centre) or a wrong match shows up as a large
residual. It works on real recordings, so a correction of merged blobs can be judged without truth:
a corrected centre should agree with the other cameras better than the raw blob centre does.

Pixel positions come in as an array ``(N, C, 2)`` with NaN where a camera is not used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openptv2.algorithms.constants import COORD_UNUSED


def project_pixels(pts: np.ndarray, cal: Any, cpar: Any) -> np.ndarray:
    """Pixel positions (N, 2) of 3D points ``pts`` (mm) in the camera ``cal``."""
    from openptv2.algorithms.imgcoord import img_coord_batch

    m = np.asarray(img_coord_batch(np.asarray(pts, float), cal, cpar.mm), float)
    return np.column_stack(
        [m[:, 0] / cpar.pix_x + cpar.imx / 2, cpar.imy / 2 - m[:, 1] / cpar.pix_y]
    )


def pixels_to_flat(pixel_xy: np.ndarray, cal: Any, cpar: Any) -> np.ndarray:
    """Pixel positions (N, 2) -> flat image coordinates (mm) used by the triangulation."""
    from openptv2.transforms import convert_arr_pixel_to_metric, distorted_to_flat

    met = convert_arr_pixel_to_metric(np.asarray(pixel_xy, float), cpar)
    return np.asarray(distorted_to_flat(met, cal), float)


def loco_residuals(
    pixel_xy: np.ndarray,
    cpar: Any,
    cals: list,
    vpar: Any = None,
    method: str = "lsq",
    min_other: int = 3,
) -> np.ndarray:
    """Residual (px) of every observation against the point triangulated without that camera.

    ``pixel_xy`` is ``(N, C, 2)`` with NaN for unused cameras. Returns ``(N, C)``; NaN where the camera is
    unused or fewer than ``min_other`` other cameras are available.
    """
    from openptv2.orientation import point_positions

    pixel_xy = np.asarray(pixel_xy, float)
    n, n_cams, _ = pixel_xy.shape
    valid = ~np.isnan(pixel_xy[:, :, 0])
    flat = np.full((n, n_cams, 2), COORD_UNUSED)
    for c in range(n_cams):
        if valid[:, c].any():
            flat[valid[:, c], c] = pixels_to_flat(pixel_xy[valid[:, c], c], cals[c], cpar)
    out = np.full((n, n_cams), np.nan)
    for c in range(n_cams):
        rows = np.flatnonzero(valid[:, c] & ((valid.sum(1) - 1) >= min_other))
        if len(rows) == 0:
            continue
        held = flat[rows].copy()
        held[:, c] = COORD_UNUSED
        pos, _ = point_positions(held, cpar, cals, vpar, method=method)
        proj = project_pixels(np.asarray(pos, float), cals[c], cpar)
        out[rows, c] = np.linalg.norm(proj - pixel_xy[rows, c], axis=1)
    return out


def summarize(res: np.ndarray) -> dict:
    """Robust summary of a LOCO residual array (px)."""
    r = res[np.isfinite(res)]
    if len(r) == 0:
        return {"n": 0}
    return {"n": int(len(r)), "median_px": float(np.median(r)), "p90_px": float(np.percentile(r, 90)),
            "p99_px": float(np.percentile(r, 99)), "rms_px": float(np.sqrt(np.mean(r**2)))}

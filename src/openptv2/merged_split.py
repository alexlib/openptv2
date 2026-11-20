"""Mixture-centroid split of merged 3D-PTV blobs (Idea 4 + 9).

A detected blob is not a particle: ~27 % of blob observations hold >= 2 particles.
The merged centre is an intensity-weighted average of the constituents' projections.
Given the constituents' predicted positions and their per-particle brightnesses, this
module recovers both positions by minimising the reprojection error of the merged
centres across cameras, while keeping each constituent close to its prediction.

The result is a set of split 3D positions plus a ``merged`` flag per point (Idea 9)
so downstream tracking can inflate the measurement noise and never update velocity
from a split observation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares


def _project_one(pt: np.ndarray, cal: Any, cpar: Any) -> np.ndarray:
    from openptv2.algorithms.imgcoord import img_coord_batch

    m = np.asarray(img_coord_batch(np.asarray(pt, dtype=float).reshape(1, 3), cal, cpar.mm))
    pix_x, pix_y = float(cpar.pix_x), float(cpar.pix_y)
    imx, imy = float(cpar.imx), float(cpar.imy)
    x = m[0, 0] / pix_x + imx / 2.0
    y = imy / 2.0 - m[0, 1] / pix_y
    return np.array([x, y], dtype=float)


def _project(pt: np.ndarray, cals: list, cpar: Any) -> np.ndarray:
    out = np.full((len(cals), 2), np.nan)
    for ci, cal in enumerate(cals):
        try:
            out[ci] = _project_one(pt, cal, cpar)
        except Exception:
            out[ci] = np.nan
    return out


def split_merged_blob(
    merged_px: np.ndarray,          # (n_cams, 2) merged blob pixel coords per cam
    seen: np.ndarray,              # (n_cams,) bool
    intensities: np.ndarray,       # (n_constituents,) per-particle brightness
    preds: np.ndarray,             # (n_constituents, 3) predicted 3D positions
    cals: list,
    cpar: Any,
    weight_pred: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one merged blob into ``n_constituents`` 3D positions.

    Returns (positions (K,3), merged_flags (K,) all True)."""
    K = len(preds)
    intensities = np.asarray(intensities, dtype=float)
    w = intensities / intensities.sum()
    x0 = preds.reshape(-1).copy()

    def residuals(x):
        pts = x.reshape(K, 3)
        # merged centre reconstruction per camera
        res = []
        n_cams = len(merged_px)
        for ci in range(n_cams):
            if not seen[ci]:
                continue
            # project each constituent
            _cal_list = [cals[ci]] if (cals is not None and ci < len(cals)) else []
            projs = np.array([_project(pts[k], _cal_list, cpar)[0] for k in range(K)])
            # intensity-weighted centroid
            cen = (w[:, None] * projs).sum(axis=0)
            res.append(cen - merged_px[ci])
        # prediction regularisation
        res.append(weight_pred * (pts - preds).ravel())
        return np.concatenate(res)

    # initial residual: do the predicted constituents explain the merged centre?
    init_res = residuals(x0)
    init_rms = float(np.sqrt(np.mean(init_res ** 2))) if init_res.size else 0.0
    sol = least_squares(residuals, x0, method="trf", max_nfev=200)
    pts = sol.x.reshape(K, 3)
    final_rms = float(np.sqrt(np.mean(sol.fun ** 2))) if sol.fun.size else 0.0
    return pts, np.ones(K, dtype=bool), init_rms, final_rms

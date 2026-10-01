"""Per-point quality mark from ray convergence (plan A10).

A 3D point built from camera rays that do not meet well (large ray miss
distance, rcm) is more likely a ghost. The miss distance grows with the distance
from the volume centre on real data, so it is divided by a smooth per-run curve
first. The relative value and the camera count give a ghost probability
(table measured on the synthetic real-jitter case, docs/plans/2026-09-30-tracker-plan.md).

Nothing here deletes a point; callers use the probability softly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# relative rcm (rcm / local typical rcm) -> ghost probability, by camera count
_REL_X = np.array([0.8, 1.05, 1.4, 1.9, 2.6, 3.5])
_P_GHOST = {
    3: np.array([0.09, 0.10, 0.20, 0.37, 0.62, 0.75]),
    4: np.array([0.01, 0.03, 0.04, 0.09, 0.28, 0.45]),
}


def frame_rcm(store: Any, frame: int, cals: list, cpar: Any) -> np.ndarray:
    """Ray miss distance (mm) of every stored 3D point of one frame."""
    from openptv2.algorithms.orientation import (
        COORD_UNUSED,
        multi_cam_point_positions,
    )
    from openptv2.algorithms.trafo import (
        correct_brown_affine_batch,
        pixel_to_metric_batch,
    )

    pos, cam_ids = store.read_correspondences(frame)
    n, n_cams = len(pos), len(cals)
    if n == 0:
        return np.zeros(0)
    targets = np.full((n, n_cams, 2), COORD_UNUSED, dtype=np.float64)
    for cam in range(n_cams):
        rows = np.asarray(store.root[f"targets/cam_{cam}/frame_{frame:06d}"])
        sel = np.flatnonzero(cam_ids[:, cam] >= 0)
        if not len(sel):
            continue
        cal = cals[cam]
        a = cal.added_par
        xy = np.ascontiguousarray(rows[cam_ids[sel, cam]][:, 1:3], dtype=np.float64)
        met = np.ascontiguousarray(pixel_to_metric_batch(xy, cpar), dtype=np.float64)
        flat = np.asarray(
            correct_brown_affine_batch(met, a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she)
        )
        targets[sel, cam, 0] = flat[:, 0] - cal.int_par.xh
        targets[sel, cam, 1] = flat[:, 1] - cal.int_par.yh
    _, rcm = multi_cam_point_positions(targets, cpar, cals)
    return np.asarray(rcm, dtype=np.float64)


def fit_scale(
    pos: np.ndarray, rcm: np.ndarray, n_cams_seen: np.ndarray, centre=None
) -> tuple[np.ndarray, float, float]:
    """Typical rcm as a straight line in the distance from the volume centre,
    fitted on the medians of 4-camera points (robust to ghosts).
    Returns (centre, a, b): scale(d) = a + b * d."""
    centre = np.median(pos, axis=0) if centre is None else np.asarray(centre)
    d = np.linalg.norm(pos - centre, axis=1)
    m = (n_cams_seen >= 4) & np.isfinite(rcm)
    if m.sum() < 200:
        return centre, float(np.nanmedian(rcm)), 0.0
    q = np.quantile(d[m], np.linspace(0, 1, 13))
    b_idx = np.clip(np.searchsorted(q, d[m], side="right") - 1, 0, 11)
    xs = np.array([np.median(d[m][b_idx == i]) for i in range(12)])
    ys = np.array([np.median(rcm[m][b_idx == i]) for i in range(12)])
    slope, icpt = np.polyfit(xs, ys, 1)
    return centre, float(icpt), float(slope)


def ghost_probability(
    pos: np.ndarray,
    rcm: np.ndarray,
    n_cams_seen: np.ndarray,
    scale: tuple[np.ndarray, float, float],
) -> np.ndarray:
    """Probability that each point is a ghost, from relative rcm and camera
    count. Points seen by 2 cameras use the 3-camera curve; NaN rcm -> 0.5."""
    centre, a, b = scale
    d = np.linalg.norm(pos - centre, axis=1)
    rel = rcm / np.maximum(a + b * d, 1e-6)
    p = np.where(
        n_cams_seen >= 4,
        np.interp(rel, _REL_X, _P_GHOST[4]),
        np.interp(rel, _REL_X, _P_GHOST[3]),
    )
    return np.where(np.isfinite(rel), p, 0.5)

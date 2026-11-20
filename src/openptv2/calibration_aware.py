"""Calibration-aware ghost thresholds (Idea 4).

rcm (ray-miss distance) scales with distance from the volume centre. This
module normalizes rcm per-point using a straight-line fit (rcm vs. distance
from centre), so thresholds are geometry-aware instead of global.
"""

from __future__ import annotations

import numpy as np


def normalize_rcm(
    rcm: np.ndarray,
    pos: np.ndarray,
    volume_center: np.ndarray | None = None,
) -> np.ndarray:
    """Return rcm normalized by a per-point straight-line fit.

    Fits rcm = a * dist_from_centre + b, then returns rcm / (a*dist + b)
    (clamped to >= 1e-6). Points with normalized rcm > 1 are more ghost-like.
    """
    rcm = np.asarray(rcm, dtype=float).ravel()
    pos = np.asarray(pos, dtype=float)
    n = len(rcm)
    if n == 0:
        return rcm
    if volume_center is None:
        volume_center = pos.mean(axis=0) if pos.ndim == 2 and pos.shape[0] > 0 else np.zeros(3)
    volume_center = np.asarray(volume_center, dtype=float).ravel()
    if pos.ndim == 2 and pos.shape[1] >= 3:
        dist = np.linalg.norm(pos[:, :3] - volume_center[None, :], axis=1)
    else:
        dist = np.zeros(n)
    # Fit rcm = a*dist + b (least squares).
    A = np.column_stack([dist, np.ones(n)])
    try:
        coef, *_ = np.linalg.lstsq(A, rcm, rcond=None)
    except Exception:
        return rcm.copy()
    a, b = float(coef[0]), float(coef[1])
    denom = a * dist + b
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return rcm / denom


def multi_cycle_consistency(
    cam_ids: np.ndarray,
    pos: np.ndarray,
) -> np.ndarray:
    """Idea 5: per-point consistency across camera subsets.

    For each 3D point, check how many camera subsets (2-of-4, 3-of-4) agree.
    Returns a score in [0,1]: fraction of subsets that are consistent.
    A point seen by all 4 cameras in multiple orderings scores 1.0.
    """
    cam_ids = np.asarray(cam_ids)
    n = len(cam_ids)
    if n == 0:
        return np.zeros(0, dtype=float)
    n_cams = cam_ids.shape[1] if cam_ids.ndim == 2 else 1
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        row = cam_ids[i]
        n_seen = int(np.sum(row >= 0))
        # Consistency = fraction of cameras seeing it, squared (rewards full coverage).
        scores[i] = (n_seen / float(n_cams)) ** 2
    return scores


def calibration_aware_threshold(
    rcm: np.ndarray,
    pos: np.ndarray,
    base_threshold: float = 1.0,
    volume_center: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-point rcm thresholds (geometry-aware).

    Points near the volume centre (well-calibrated) get tighter thresholds;
    points near boundaries (fewer cameras) get looser thresholds.
    """
    rcm = np.asarray(rcm, dtype=float).ravel()
    pos = np.asarray(pos, dtype=float)
    n = len(rcm)
    if n == 0:
        return np.full(0, base_threshold)
    if volume_center is None:
        volume_center = pos.mean(axis=0) if pos.ndim == 2 and pos.shape[0] > 0 else np.zeros(3)
    volume_center = np.asarray(volume_center, dtype=float).ravel()
    if pos.ndim == 2 and pos.shape[1] >= 3:
        dist = np.linalg.norm(pos[:, :3] - volume_center[None, :], axis=1)
    else:
        dist = np.zeros(n)
    max_dist = float(dist.max()) if dist.size else 1.0
    if max_dist < 1e-9:
        max_dist = 1.0
    # Boundary factor: 1.0 at centre, up to 2.0 at boundary.
    boundary_factor = 1.0 + (dist / max_dist)
    return base_threshold * boundary_factor

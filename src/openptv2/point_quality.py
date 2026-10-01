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


def read_frame_arrays(store: Any, frame: int, n_cams: int | None = None):
    """(pos, cam_ids, target_rows_per_camera) of one frame, each array read once.
    ``target_rows_per_camera[c]`` is None when the store has no targets for camera c."""
    pos, cam_ids = store.read_correspondences(frame)
    n_cams = cam_ids.shape[1] if n_cams is None else n_cams
    rows = []
    for cam in range(n_cams):
        key = f"targets/cam_{cam}/frame_{frame:06d}"
        rows.append(np.asarray(store.root[key]) if key in store.root else None)
    return pos, cam_ids, rows


def rcm_from_arrays(
    pos: np.ndarray, cam_ids: np.ndarray, rows: list, cals: list, cpar: Any
) -> np.ndarray:
    """Ray miss distance (mm) of every 3D point of one frame, from its arrays."""
    from openptv2.algorithms.orientation import (
        COORD_UNUSED,
        multi_cam_point_positions,
    )
    from openptv2.algorithms.trafo import (
        correct_brown_affine_batch,
        pixel_to_metric_batch,
    )

    n, n_cams = len(pos), len(cals)
    if n == 0:
        return np.zeros(0)
    targets = np.full((n, n_cams, 2), COORD_UNUSED, dtype=np.float64)
    for cam in range(n_cams):
        sel = np.flatnonzero(cam_ids[:, cam] >= 0)
        if not len(sel):
            continue
        cal = cals[cam]
        a = cal.added_par
        xy = np.ascontiguousarray(
            rows[cam][cam_ids[sel, cam]][:, 1:3], dtype=np.float64
        )
        met = np.ascontiguousarray(pixel_to_metric_batch(xy, cpar), dtype=np.float64)
        flat = np.asarray(
            correct_brown_affine_batch(met, a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she)
        )
        targets[sel, cam, 0] = flat[:, 0] - cal.int_par.xh
        targets[sel, cam, 1] = flat[:, 1] - cal.int_par.yh
    _, rcm = multi_cam_point_positions(targets, cpar, cals)
    return np.asarray(rcm, dtype=np.float64)


def frame_rcm(store: Any, frame: int, cals: list, cpar: Any) -> np.ndarray:
    """Ray miss distance (mm) of every stored 3D point of one frame."""
    pos, cam_ids, rows = read_frame_arrays(store, frame, len(cals))
    return rcm_from_arrays(pos, cam_ids, rows, cals, cpar)


def log_brightness_from_arrays(cam_ids: np.ndarray, rows: list) -> np.ndarray:
    """(N, C) log blob brightness (``sumg``) in each camera that sees the point;
    NaN where the camera does not see it."""
    n, n_cams = len(cam_ids), cam_ids.shape[1]
    logb = np.full((n, n_cams), np.nan)
    for cam in range(n_cams):
        sel = np.flatnonzero(cam_ids[:, cam] >= 0)
        if len(sel):
            logb[sel, cam] = np.log(np.maximum(rows[cam][cam_ids[sel, cam]][:, 6], 1.0))
    return logb


def frame_log_brightness(store: Any, frame: int) -> np.ndarray:
    """(N, C) log blob brightness of every stored 3D point of one frame."""
    pos, cam_ids, rows = read_frame_arrays(store, frame)
    return log_brightness_from_arrays(cam_ids, rows)


def brightness_spread(
    logb: np.ndarray, offsets: np.ndarray | None = None
) -> np.ndarray:
    """Std over the seeing cameras of the log brightness, after subtracting each
    camera's typical value (``offsets``, e.g. its run median): cameras differ in gain
    by 0.13-0.17 in log on real data, which must not look like inconsistency. A real
    particle shows a similar brightness in all cameras (spread about 0.15); a ghost
    combines blobs of different particles (about 0.4)."""
    x = logb if offsets is None else logb - np.asarray(offsets)[None, :]
    with np.errstate(all="ignore"):
        return np.nanstd(x, axis=1)


def frame_brightness_spread(store: Any, frame: int) -> np.ndarray:
    """``brightness_spread`` of one frame without gain normalisation."""
    return brightness_spread(frame_log_brightness(store, frame))


# Logistic model on [1, log rel_rcm, 3-camera flag, log(spread + 0.05),
# log(spread + 0.05) * 3-camera flag]; fitted on the synthetic real-jitter case with
# realistic per-camera brightness scatter (docs/plans/2026-09-30-tracker-plan.md, A10).
_BLOB_W = np.array([-1.71, 2.34, 2.5, 1.73, 0.57])


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
    spread: np.ndarray | None = None,
) -> np.ndarray:
    """Probability that each point is a ghost, from relative rcm and camera
    count (and, with ``spread`` = ``frame_brightness_spread``, the brightness
    agreement across cameras; model ``rcm_blob``). Points seen by 2 cameras use the 3-camera curve; NaN rcm -> 0.5."""
    centre, a, b = scale
    d = np.linalg.norm(pos - centre, axis=1)
    rel = rcm / np.maximum(a + b * d, 1e-6)
    if spread is not None:
        three = (np.asarray(n_cams_seen) < 4).astype(np.float64)
        ls = np.log(np.asarray(spread, dtype=np.float64) + 0.05)
        x = np.c_[
            np.ones(len(rel)), np.log(np.maximum(rel, 1e-3)), three, ls, ls * three
        ]
        p = 1.0 / (1.0 + np.exp(-(x @ _BLOB_W)))
        return np.where(np.isfinite(rel) & np.isfinite(ls), p, 0.5)
    p = np.where(
        n_cams_seen >= 4,
        np.interp(rel, _REL_X, _P_GHOST[4]),
        np.interp(rel, _REL_X, _P_GHOST[3]),
    )
    return np.where(np.isfinite(rel), p, 0.5)

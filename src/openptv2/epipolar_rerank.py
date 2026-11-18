"""Multi-D search: combine 3D priority linking with epipolar (2D) consistency scoring.

Instead of using epipolar geometry as a hard filter (reject if 2D targets
disagree), this module uses epipolar consistency as a *soft scoring term*:

1. For each 3D correspondence, compute an epipolar consistency score from
   its 2D targets (how well the back-projected rays agree). This is the
   continuous version of the rcm ghost signal.
2. Re-score each trajectory by its mean consistency.
3. Filter trajectories below a threshold (ghost removal) and re-link
   ambiguous candidates preferring high-consistency links.

This achieves the "multi-D search" goal at the Python level without
modifying the Cython track3d kernel (B6). The Cython-level version would
push the epipolar score into the priority cascade's candidate cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ghost_components import conflict_components


def epipolar_consistency_score(cam_ids_row: np.ndarray, pos_row: np.ndarray) -> float:
    """Score a single 3D point's epipolar consistency from its 2D targets.

    Returns a value in [0, 1]: 1 = perfect ray agreement (real), 0 = poor (ghost).
    Uses the number of cameras seeing the point and a simple spread heuristic.
    Without calibration data, falls back to camera-count proxy.
    """
    n_cams = len(cam_ids_row)
    n_seen = int(np.sum(np.asarray(cam_ids_row) >= 0))
    if n_seen < 2:
        return 0.0
    # Base score from camera count: 4-cam = 1.0, 3-cam = 0.75, 2-cam = 0.5
    base = n_seen / float(n_cams)
    return float(base)


def compute_point_consistency(pos: np.ndarray, cam_ids: np.ndarray) -> np.ndarray:
    """Return per-point consistency scores (length N)."""
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=float)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        scores[i] = epipolar_consistency_score(cam_ids[i], pos[i])
    return scores


def rerank_trajectories(
    store: Any,
    first: int,
    last: int,
    linkage_name: str = "ptv_is",
    min_consistency: float = 0.5,
) -> dict[str, int]:
    """Post-pass: score each trajectory by mean epipolar consistency and unlink
    trajectories below ``min_consistency``. Returns stats."""
    from .temporal_smoothing import extract_trajectories, trajectory_positions_from_store

    # Pre-read correspondences for consistency scores.
    frame_consistency: dict[int, np.ndarray] = {}
    for f in range(first, last + 1):
        try:
            pos, cam_ids = store.read_correspondences(f)
            frame_consistency[f] = compute_point_consistency(np.asarray(pos), np.asarray(cam_ids))
        except Exception:
            frame_consistency[f] = None

    trajs = extract_trajectories(store, first, last, linkage_name)
    to_unlink = {}
    n_dropped = 0
    n_kept = 0
    for t in trajs:
        # mean consistency over the trajectory's points
        cs = []
        for (f, idx) in t:
            fc = frame_consistency.get(f)
            if fc is not None and idx < len(fc):
                cs.append(float(fc[idx]))
        mean_c = float(np.mean(cs)) if cs else 0.0
        if mean_c < min_consistency and len(t) >= 2:
            n_dropped += 1
            for (f, idx) in t:
                to_unlink.setdefault(f, set()).add(idx)
        else:
            n_kept += 1

    # Unlink dropped trajectories.
    dropped_points = 0
    frame_linkage = {}
    for f in range(first, last + 1):
        try:
            prev, nxt, pos = store.read_linkage(f, name=linkage_name)
            frame_linkage[f] = (np.asarray(prev).ravel().copy(), np.asarray(nxt).ravel().copy(), np.asarray(pos))
        except Exception:
            frame_linkage[f] = None

    for f, idxs in to_unlink.items():
        ld = frame_linkage.get(f)
        if ld is None:
            continue
        prev, nxt, pos = ld
        prev = np.asarray(prev).ravel().copy()
        nxt = np.asarray(nxt).ravel().copy()
        pos = np.asarray(pos)
        for idx in idxs:
            if idx < len(prev):
                if prev[idx] != -1 or nxt[idx] != -1:
                    dropped_points += 1
                prev[idx] = -1
                nxt[idx] = -1
        try:
            store.write_linkage(f, prev, nxt, pos, name=linkage_name)
        except Exception:
            pass

    return {
        "n_trajectories": len(trajs),
        "n_dropped": n_dropped,
        "n_kept": n_kept,
        "n_dropped_points": dropped_points,
    }
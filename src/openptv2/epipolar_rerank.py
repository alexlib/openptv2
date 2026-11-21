"""Multi-D search: combine 3D priority linking with epipolar (2D) consistency scoring.

Instead of using epipolar geometry as a hard filter (reject if 2D targets
disagree), this module uses epipolar consistency as a *soft scoring term*.

At the Python level this is a post-pass: trajectories whose mean epipolar
consistency (ray-miss distance) is below threshold are unlinked. The
Cython-level version (B6) would push the epipolar score into the track3d
priority cascade's candidate cost for speed.
"""

from __future__ import annotations

import numpy as np

from .temporal_smoothing import extract_trajectories


def epipolar_consistency_score(cam_ids_row, pos_row, rcm_val=None):
    """Score a single 3D point's epipolar consistency from its 2D targets.

    Returns [0,1]: 1 = perfect ray agreement (real), 0 = poor (ghost).
    If rcm_val (ray-miss mm) given, use exponential decay. Else camera-count.
    """
    n_cams = len(cam_ids_row)
    n_seen = int(np.sum(np.asarray(cam_ids_row) >= 0))
    if n_seen < 2:
        return 0.0
    if rcm_val is not None and rcm_val >= 0:
        return float(np.exp(-rcm_val / 5.0))
    return float(n_seen / float(n_cams))


def compute_point_consistency(pos, cam_ids, rcm=None):
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=float)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        rv = float(rcm[i]) if (rcm is not None and i < len(rcm)) else None
        scores[i] = epipolar_consistency_score(cam_ids[i], pos[i], rcm_val=rv)
    return scores


def rerank_trajectories(store, first, last, linkage_name="ptv_is", min_consistency=0.5):
    """Post-pass: score each trajectory by mean epipolar consistency and unlink
    trajectories below min_consistency. Uses rcm if available."""
    frame_consistency = {}
    for f in range(first, last + 1):
        try:
            pos, cam_ids = store.read_correspondences(f)
            rcm = store.read_rcm(f) if hasattr(store, "read_rcm") else None
            if rcm is not None:
                from openptv2.calibration_aware import normalize_rcm
                rcm = normalize_rcm(rcm, np.asarray(pos))
            frame_consistency[f] = compute_point_consistency(np.asarray(pos), np.asarray(cam_ids), rcm=rcm)
        except Exception:
            frame_consistency[f] = None

    trajs = extract_trajectories(store, first, last, linkage_name)

    to_unlink = {}
    n_dropped = 0
    n_kept = 0
    for t in trajs:
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

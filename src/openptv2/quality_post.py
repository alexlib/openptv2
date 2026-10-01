"""Use the per-point ghost probability AFTER tracking (plan A10).

Arrays follow the pred.npz layout: ``trajid``, ``frame``, ``pos`` per point,
any order. All functions return masks or new arrays; nothing is mutated.

* ``trim_doubtful_ends``  - cut doubtful points off both ends of a trajectory.
* ``drop_spikes``         - drop interior points whose ghost probability jumps up
                            against their neighbours in the same trajectory.
* ``quality_weights``     - smoothing weights from the ghost probability.
* ``weighted_savgol``     - Savitzky-Golay-style local polynomial fit with weights.
"""

from __future__ import annotations

import numpy as np


def _sorted(trajid: np.ndarray, frame: np.ndarray):
    order = np.lexsort((frame, trajid))
    cuts = np.flatnonzero(np.diff(trajid[order])) + 1
    return order, np.split(np.arange(len(order)), cuts)


def trim_doubtful_ends(
    trajid: np.ndarray,
    frame: np.ndarray,
    ghost: np.ndarray,
    thr: float,
    max_trim: int = 3,
    max_frac: float = 0.25,
) -> np.ndarray:
    """Keep-mask: from each end, remove up to ``max_trim`` consecutive points
    whose ghost probability is above ``thr`` (stops at the first good point),
    but never more than ``max_frac`` of the trajectory from one end (a short
    track of 8 points loses at most 2 per end)."""
    keep = np.ones(len(trajid), bool)
    order, groups = _sorted(trajid, frame)
    for g in groups:
        idx = order[g]
        bad = ghost[idx] > thr
        # never cut more than max_frac of a trajectory from one end
        limit = min(max_trim, int(len(idx) * max_frac))
        for seq in (range(len(idx)), range(len(idx) - 1, -1, -1)):
            for n, k in enumerate(seq):
                if n >= limit or not bad[k]:
                    break
                keep[idx[k]] = False
    return keep


def drop_spikes(
    trajid: np.ndarray,
    frame: np.ndarray,
    ghost: np.ndarray,
    thr: float,
    jump: float,
) -> np.ndarray:
    """Keep-mask: drop an interior point if its ghost probability is above
    ``thr`` AND exceeds the mean of its two neighbours (previous and next point
    of the same trajectory) by more than ``jump``."""
    keep = np.ones(len(trajid), bool)
    order, groups = _sorted(trajid, frame)
    for g in groups:
        if len(g) < 3:
            continue
        idx = order[g]
        p = ghost[idx]
        spike = (p[1:-1] > thr) & (p[1:-1] - 0.5 * (p[:-2] + p[2:]) > jump)
        keep[idx[1:-1][spike]] = False
    return keep


def quality_weights(ghost: np.ndarray, power: float = 2.0, floor: float = 0.05):
    """Weight of a point in smoothing: ``(1 - ghost) ** power`` (floored)."""
    return np.maximum((1.0 - np.asarray(ghost, dtype=np.float64)) ** power, floor)


def weighted_savgol(
    trajid: np.ndarray,
    frame: np.ndarray,
    pos: np.ndarray,
    weights: np.ndarray | None,
    fps: float,
    window: int,
    order: int = 3,
    min_window: int = 5,
):
    """Weighted local polynomial smoothing, one fit per point.

    Window = ``window`` points of the same trajectory, centred on the point and
    shifted inward at the ends (as scipy's ``mode='interp'``); trajectories
    shorter than ``window`` use all their points (at least ``min_window``).
    With unit weights this is a Savitzky-Golay filter.

    Returns (trajid, frame, pos, vel [per s], acc [per s^2]) for the kept points.
    """
    w_all = np.ones(len(trajid)) if weights is None else np.asarray(weights, float)
    order_idx, groups = _sorted(trajid, frame)
    out_id, out_fr, out_pos, out_vel, out_acc = [], [], [], [], []
    for g in groups:
        idx = order_idx[g]
        n = len(idx)
        if n < min_window:
            continue
        w = min(window, n)
        if w % 2 == 0:
            w -= 1  # largest odd window the trajectory fills (as flowtracks min_window)
        deg = min(order, w - 1)
        h = max(w // 2, 1)
        i = np.arange(n)
        start = np.clip(i - w // 2, 0, n - w)
        win = start[:, None] + np.arange(w)[None, :]  # (n, w)
        t = frame[idx].astype(np.float64)
        u = (t[win] - t[:, None]) / h  # (n, w) in [-1, 1]-ish
        A = u[:, :, None] ** np.arange(deg + 1)[None, None, :]  # (n, w, deg+1)
        W = w_all[idx][win]  # (n, w)
        AtW = A.transpose(0, 2, 1) * W[:, None, :]
        M = AtW @ A + 1e-12 * np.eye(deg + 1)
        Y = pos[idx][win]  # (n, w, 3)
        C = np.linalg.solve(M, AtW @ Y)  # (n, deg+1, 3)
        out_id.append(trajid[idx])
        out_fr.append(frame[idx])
        out_pos.append(C[:, 0, :])
        out_vel.append(C[:, 1, :] / h * fps if deg >= 1 else np.zeros((n, 3)))
        out_acc.append(2.0 * C[:, 2, :] / h**2 * fps**2 if deg >= 2 else np.zeros((n, 3)))
    if not out_id:
        z = np.zeros((0, 3))
        return np.zeros(0, int), np.zeros(0, int), z, z, z
    return (
        np.concatenate(out_id),
        np.concatenate(out_fr),
        np.vstack(out_pos),
        np.vstack(out_vel),
        np.vstack(out_acc),
    )

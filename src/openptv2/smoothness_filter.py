"""Smoothness filter for fake points inside trajectories (plan A9).

Each point is compared with a polynomial fitted to its neighbours in the same trajectory
WITHOUT the point itself (leave-one-out). A real particle follows a smooth path; a point
that does not (a ghost, or a wrong link) sits far from the curve its neighbours define.
Arrays follow the pred.npz layout. The fit runs on the real frame times, so a gap (a
reconnected piece) is not mistaken for a kink: the neighbours on each side simply define
the curve across it.
"""

from __future__ import annotations

import numpy as np


def leave_one_out_residual(
    trajid: np.ndarray,
    frame: np.ndarray,
    pos: np.ndarray,
    window: int = 11,
    deg: int = 2,
    min_len: int = 6,
) -> np.ndarray:
    """Distance (same units as ``pos``) of every point to the polynomial fitted to up to
    ``window`` neighbours around it, excluding the point itself. NaN for trajectories
    shorter than ``min_len``."""
    res = np.full(len(trajid), np.nan)
    order = np.lexsort((frame, trajid))
    cuts = np.flatnonzero(np.diff(trajid[order])) + 1
    for g in np.split(np.arange(len(order)), cuts):
        n = len(g)
        if n < min_len:
            continue
        idx = order[g]
        w = min(window, n)
        d = min(deg, w - 2)
        i = np.arange(n)
        start = np.clip(i - w // 2, 0, n - w)
        win = start[:, None] + np.arange(w)[None, :]
        t = frame[idx].astype(float)
        h = max(w // 2, 1)
        u = (t[win] - t[:, None]) / h
        A = u[:, :, None] ** np.arange(d + 1)[None, None, :]
        W = (win != i[:, None]).astype(float)
        AtW = A.transpose(0, 2, 1) * W[:, None, :]
        M = AtW @ A + 1e-9 * np.eye(d + 1)
        C = np.linalg.solve(M, AtW @ pos[idx][win].astype(float))
        res[idx] = np.linalg.norm(pos[idx] - C[:, 0, :], axis=1)
    return res


def smoothness_keep(
    trajid: np.ndarray,
    frame: np.ndarray,
    pos: np.ndarray,
    k: float = 6.0,
    window: int = 11,
    deg: int = 2,
    min_len: int = 6,
    min_keep_len: int = 0,
) -> np.ndarray:
    """Keep-mask: drop points whose leave-one-out residual exceeds ``k`` times the median
    residual of the whole run (a robust noise scale). Trajectories with fewer than
    ``min_keep_len`` points are dropped as a whole (0 = keep short ones: a real particle
    that enters late starts short)."""
    res = leave_one_out_residual(trajid, frame, pos, window, deg, min_len)
    ok = np.isfinite(res)
    scale = float(np.median(res[ok])) if ok.any() else np.inf
    keep = ~(ok & (res > k * scale))
    if min_keep_len > 0:
        ids, counts = np.unique(trajid, return_counts=True)
        short = set(ids[counts < min_keep_len].tolist())
        keep &= ~np.isin(trajid, list(short))
    return keep

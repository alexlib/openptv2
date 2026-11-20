"""Blob content model: what a detected blob contains, and where each particle in it is.

A blob is the image of m >= 1 particles. Its centre is (about) the brightness-weighted mean of the
particles' spot centres and its grey sum the sum of their brightnesses. Given a prediction ``p_j`` of
every member's pixel position (from its track) and its brightness share ``w_j``, the blob centre ``c``
is one linear measurement of the members' positions,

    c = sum_j w_j s_j + noise,

and a Kalman update gives each member a corrected pixel position ``s_j`` and a variance. Without noise
and with equal prior variances this is the minimum-norm split ``s_j = p_j + w_j r / sum w_k^2`` with
``r = c - sum w_k p_k``. A blob with one member returns the blob centre itself.

All functions are pure numpy; geometry (projection, triangulation) is in :mod:`openptv2.loco` and
:func:`triangulate_weighted`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openptv2.loco import project_pixels


def brightness_shares(brightness: np.ndarray) -> np.ndarray:
    """Normalised brightness shares ``w_j`` (sum 1) of the particles in a blob; equal if none is positive."""
    a = np.asarray(brightness, float)
    a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
    if np.all(np.isnan(a)):
        return np.full(len(a), 1.0 / len(a))
    a = np.where(np.isnan(a), np.nanmedian(a), a)
    return a / a.sum()


def split_blob(
    centre: np.ndarray,
    pred: np.ndarray,
    brightness: np.ndarray,
    var_centre: float,
    var_pred: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrected pixel position and variance of every particle in one blob.

    ``centre`` (2,) the blob centre, ``pred`` (m, 2) predicted pixel positions of the members,
    ``brightness`` (m,) their brightness (signature), ``var_centre`` the variance (px^2) of the centre of a
    blob that holds one particle, ``var_pred`` the variance (px^2) of the predictions (scalar or (m,)).
    Returns ``(s (m, 2), var (m,))``.
    """
    centre = np.asarray(centre, float).reshape(2)
    pred = np.asarray(pred, float).reshape(-1, 2)
    m = len(pred)
    var_p = np.broadcast_to(np.asarray(var_pred, float), (m,)).copy()
    if m == 1:
        # one particle: the blob centre is its measurement, weighted against the prediction
        k = var_p[0] / (var_p[0] + var_centre)
        return pred + k * (centre - pred[0]), np.array([var_p[0] * var_centre / (var_p[0] + var_centre)])
    w = brightness_shares(brightness)
    resid = centre - w @ pred
    s_var = float(np.sum(w**2 * var_p) + var_centre)
    gain = var_p * w / s_var
    s = pred + gain[:, None] * resid[None, :]
    var = var_p * (1.0 - w**2 * var_p / s_var)
    return s, var


def clean_centre(centre: np.ndarray, var_centre: float) -> tuple[np.ndarray, float]:
    """An unshared blob: the centre itself, with the centre variance."""
    return np.asarray(centre, float).reshape(2), float(var_centre)


def signature_from_history(sumg: np.ndarray, alone: np.ndarray, quantile: float = 0.5) -> float:
    """Typical brightness of one particle in one camera: a quantile (default the median) of the grey sums over the
    frames where its blob held only this particle. A lower quantile resists frames where an unseen particle hid in
    the blob and inflated it. NaN if there is no such frame."""
    sumg = np.asarray(sumg, float)
    alone = np.asarray(alone, bool)
    v = sumg[alone & np.isfinite(sumg) & (sumg > 0)]
    return float(np.quantile(v, quantile)) if len(v) else float("nan")


def triangulate_weighted(
    pixel_xy: np.ndarray,
    var_px: np.ndarray,
    cpar: Any,
    cals: list,
    x0: np.ndarray,
    n_iter: int = 3,
    step_mm: float = 0.01,
    cam_index: list[int] | None = None,
    return_cov: bool = False,
):
    """Weighted least-squares position (mm) of points from pixel observations, by Gauss-Newton on the
    reprojection error (maximum likelihood for pixel noise with variance ``var_px``).

    ``pixel_xy`` (N, S, 2) with NaN for unused slots, ``var_px`` (N, S), ``x0`` (N, 3) a start (e.g. the
    ordinary triangulation). Slot ``s`` is an observation in camera ``cam_index[s]`` (default: slot = camera),
    so a camera can appear twice: e.g. a second block of slots holding the *projected path prediction* of the
    particle (a temporal prior) next to the measured blobs. Points with fewer than two valid slots keep ``x0``.
    With ``return_cov`` also returns ``(N, 3, 3)`` position covariances (mm^2, inverse of the weighted normal
    matrix at the solution; NaN where fewer than two slots), the uncertainty a smoother can weight by.
    """
    pixel_xy = np.asarray(pixel_xy, float)
    n, n_cams, _ = pixel_xy.shape
    cam_of = list(range(n_cams)) if cam_index is None else list(cam_index)
    if len(cam_of) != n_cams:
        raise ValueError("cam_index must have one entry per observation slot")
    x = np.asarray(x0, float).copy()
    valid = ~np.isnan(pixel_xy[:, :, 0]) & np.isfinite(var_px) & (var_px > 0)
    ok = valid.sum(1) >= 2
    cov = np.full((n, 3, 3), np.nan)
    if not ok.any():
        return (x, cov) if return_cov else x
    w_all = np.where(valid, 1.0 / np.where(valid, var_px, 1.0), 0.0)
    eye = np.eye(3)
    for _ in range(n_iter):
        a_mat = np.zeros((n, 3, 3))
        b_vec = np.zeros((n, 3))
        for c in range(n_cams):
            rows = np.flatnonzero(ok & valid[:, c])
            if len(rows) == 0:
                continue
            base = project_pixels(x[rows], cals[cam_of[c]], cpar)
            jac = np.zeros((len(rows), 2, 3))
            for ax in range(3):
                d = np.zeros(3)
                d[ax] = step_mm
                jac[:, :, ax] = (project_pixels(x[rows] + d, cals[cam_of[c]], cpar)
                                 - project_pixels(x[rows] - d, cals[cam_of[c]], cpar)) / (2 * step_mm)
            r = pixel_xy[rows, c] - base                         # (k, 2)
            w = w_all[rows, c][:, None, None]
            a_mat[rows] += np.einsum("kij,kil->kjl", jac * np.sqrt(w), jac * np.sqrt(w))
            b_vec[rows] += np.einsum("kij,ki->kj", jac, r * w[:, :, 0])
        idx = np.flatnonzero(ok)
        step = np.linalg.solve(a_mat[idx] + 1e-9 * eye, b_vec[idx][:, :, None])[:, :, 0]
        x[idx] += step
    if return_cov:
        idx = np.flatnonzero(ok)
        cov[idx] = np.linalg.inv(a_mat[idx] + 1e-9 * eye)
        return x, cov
    return x


def split_blob_effective(
    centre: np.ndarray,
    pred: np.ndarray,
    brightness: np.ndarray,
    var_centre: float,
    var_pred: float | np.ndarray,
    min_share: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-particle *effective observation* of one blob, free of the particle's own prediction.

    With ``c = sum_k w_k s_k + noise``, the information the blob gives about particle ``j`` is
    ``s_j = (c - sum_{k != j} w_k p_k) / w_j`` with variance ``(var_c + sum_{k != j} w_k^2 var_p_k) / w_j^2``:
    only the *other* members' predictions enter, so a particle's own track is not counted twice when it is
    fitted with this observation (Gauss-Seidel over the members). A one-particle blob returns the blob
    centre and ``var_centre``. A dim member (small share) gets a large variance, i.e. little weight.
    Returns ``(s (m, 2), var (m,))``.
    """
    centre = np.asarray(centre, float).reshape(2)
    pred = np.asarray(pred, float).reshape(-1, 2)
    m = len(pred)
    if m == 1:
        return centre[None, :].copy(), np.array([float(var_centre)])
    var_p = np.broadcast_to(np.asarray(var_pred, float), (m,))
    w = np.maximum(brightness_shares(brightness), min_share)
    w = w / w.sum()
    wp = w[:, None] * pred
    others = wp.sum(0)[None, :] - wp                              # sum over k != j of w_k p_k
    s = (centre[None, :] - others) / w[:, None]
    var_others = np.sum(w**2 * var_p) - w**2 * var_p
    var = (var_centre + var_others) / w**2
    return s, var


def local_poly_predict(
    trajid: np.ndarray,
    frame: np.ndarray,
    pos: np.ndarray,
    weights: np.ndarray | None = None,
    window: int = 9,
    order: int = 2,
    min_points: int = 5,
    return_support: bool = False,
):
    """Leave-one-out prediction of every point of a trajectory from its neighbours.

    A weighted local polynomial of degree ``order`` in time is fitted to the ``window`` nearest points of the
    same trajectory *excluding the point itself* and evaluated at the point's frame. A bad observation (e.g.
    a merged blob) therefore does not pull its own prediction. Returns ``(N, 3)``; NaN for trajectories
    with fewer than ``min_points`` points. Gaps in frame numbers are handled (times, not indices, are used).
    With ``return_support`` also returns ``(N,)``: the sum of the weights of the window points actually used
    (excluding the point itself), so a caller that down-weights doubtful points can tell how well supported a
    prediction is (support >= 3 means at least about three good neighbours).
    """
    trajid = np.asarray(trajid)
    frame = np.asarray(frame, float)
    pos = np.asarray(pos, float)
    w_all = np.ones(len(trajid)) if weights is None else np.asarray(weights, float)
    out = np.full((len(trajid), 3), np.nan)
    support = np.zeros(len(trajid))
    order_idx = np.lexsort((frame, trajid))
    ids_sorted = trajid[order_idx]
    cuts = np.flatnonzero(np.r_[True, ids_sorted[1:] != ids_sorted[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order_idx[a:b]
        n = len(idx)
        if n < min_points:
            continue
        w = min(window, n)
        deg = min(order, w - 2)
        h = max(w // 2, 1)
        i = np.arange(n)
        start = np.clip(i - w // 2, 0, n - w)
        win = start[:, None] + np.arange(w)[None, :]
        t = frame[idx]
        u = (t[win] - t[:, None]) / h
        a_mat = u[:, :, None] ** np.arange(deg + 1)[None, None, :]
        wt = w_all[idx][win].copy()
        wt[i, i - start] = 0.0                                     # leave the point itself out
        atw = a_mat.transpose(0, 2, 1) * wt[:, None, :]
        m_mat = atw @ a_mat + 1e-9 * np.eye(deg + 1)
        coef = np.linalg.solve(m_mat, atw @ pos[idx][win])
        out[idx] = coef[:, 0, :]
        support[idx] = wt.sum(1)
    return (out, support) if return_support else out

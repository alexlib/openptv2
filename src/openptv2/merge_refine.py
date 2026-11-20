"""Refine particle positions that use merged (shared) blobs. Truth-free; works on tracks.

Input, one row per particle and frame (a track point):
``frame``, ``pid`` (track id), ``blob`` (N, C) the target index of the blob the particle falls in per camera
(-1 = not detected; rows of the same frame and camera with the same blob index share that blob),
``centre`` (N, C, 2) the blob centres (px, NaN where not detected) and ``sumg`` (N, C) the blob grey sums.

Loop (the blob model of ``docs/plans/2026-11-20-holistic-ptv.md``, H1):
1. positions from the raw blob centres (triangulation, then Gauss-Newton on the reprojection error);
2. per particle a leave-one-out path prediction from frames where it has >= 2 clean cameras
   (:func:`openptv2.blob_model.local_poly_predict`); a prediction counts as *trusted* only if enough such
   frames support it;
3. every particle in a shared blob gets its *effective observation* (blob centre minus the other members'
   predictions, divided by its brightness share, :func:`openptv2.blob_model.split_blob_effective`) with its
   variance; members without a trusted prediction count as unknown (a blob radius of doubt); an effective
   observation inconsistent with the particle's own trusted prediction is dropped (gate);
4. positions again by weighted Gauss-Newton, with the trusted path prediction as pseudo-observations (prior).
Rounds repeat 2-3 times.

Hidden partners: a blob used by one point only, whose grey sum is at least ``hidden_mass`` times that point's
typical grey sum (signature), holds another particle that is not among the points (the matcher did not form it).
Its centre is shifted by that particle, so it gets a phantom member of unknown position and the remaining
brightness: the observation keeps its centre but its variance grows, i.e. the camera nearly drops out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openptv2.blob_model import (
    local_poly_predict,
    signature_from_history,
    split_blob_effective,
    triangulate_weighted,
)
from openptv2.loco import pixels_to_flat, project_pixels


@dataclass
class RefineResult:
    x: np.ndarray                    # (N, 3) positions (mm); NaN where fewer than 2 cameras
    obs_px: np.ndarray               # (N, C, 2) pixel position used per camera (NaN = not used)
    obs_var: np.ndarray              # (N, C) its variance (px^2)
    n_share: np.ndarray              # (N, C) 0 not detected, 1 alone, k >= 2 particles in the blob
    pred_px: np.ndarray              # (N, C, 2) trusted path prediction (NaN = none)
    pos_var: np.ndarray | None = None  # (N,) position variance (mm^2, trace / 3 of the covariance)
    x_rounds: list = field(default_factory=list)
    info: list = field(default_factory=list)


def shared_blobs(frame: np.ndarray, blob: np.ndarray) -> np.ndarray:
    """(N, C) number of rows (particles) in the same frame that use the same blob in that camera."""
    n, n_cams = blob.shape
    out = np.zeros((n, n_cams), int)
    for c in range(n_cams):
        det = np.flatnonzero(blob[:, c] >= 0)
        key = frame[det].astype(np.int64) * 10_000_000 + blob[det, c]
        _u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        out[det, c] = cnt[inv]
    return out


def _signatures(pid: np.ndarray, clean: np.ndarray, sumg: np.ndarray, quantile: float = 0.5) -> dict:
    """Typical grey sum per (particle, camera): median over the frames where its blob held only it."""
    sig = {}
    for c in range(clean.shape[1]):
        m = np.flatnonzero(clean[:, c])
        o = m[np.argsort(pid[m], kind="stable")]
        if len(o) == 0:
            continue
        cuts = np.flatnonzero(np.r_[True, pid[o][1:] != pid[o][:-1], True])
        for a, b in zip(cuts[:-1], cuts[1:]):
            sig[(int(pid[o[a]]), c)] = signature_from_history(sumg[o[a:b], c], np.ones(b - a, bool), quantile)
    return sig


def refine_merged(
    frame: np.ndarray,
    pid: np.ndarray,
    blob: np.ndarray,
    centre: np.ndarray,
    sumg: np.ndarray,
    cpar: Any,
    cals: list,
    vpar: Any,
    *,
    rounds: int = 3,
    window: int = 9,
    order: int = 2,
    var_c: float = 0.01,
    trust: bool = True,
    trust_support: float = 3.0,
    gate: float = 4.0,
    prior: bool = True,
    prior_scale: float = 1.0,
    var_unknown: float = 1.0,
    hidden_mass: float | None = 1.4,
) -> RefineResult:
    from openptv2.orientation import point_positions

    frame = np.asarray(frame)
    pid = np.asarray(pid)
    n, n_cams = blob.shape
    det = blob >= 0
    share = shared_blobs(frame, blob)
    clean = det & (share == 1)
    est = np.flatnonzero(det.sum(1) >= 2)          # rows that can be triangulated
    pidx = np.full(n, -1)
    pidx[est] = np.arange(len(est))
    n_est = len(est)
    fr_e, pid_e = frame[est], pid[est]
    frames = np.unique(frame)
    rows_of = {f: np.flatnonzero(frame == f) for f in frames}
    est_rows_of = {f: np.flatnonzero(fr_e == f) for f in frames}
    sig = _signatures(pid, clean, sumg)
    raw = centre[est]
    raw_var = np.where(np.isnan(raw[:, :, 0]), np.nan, var_c)

    pos_var = np.full(n_est, np.nan)

    def fit(obs_px, obs_var, x_start=None, prior_px=None, prior_var=None):
        x = np.full((n_est, 3), np.nan) if x_start is None else x_start.copy()
        cam_ix = list(range(n_cams))
        if prior_px is not None:
            obs_px = np.concatenate([obs_px, prior_px], axis=1)
            obs_var = np.concatenate([obs_var, np.broadcast_to(prior_var, prior_px.shape[:2])], axis=1)
            cam_ix = cam_ix + cam_ix
        for f, rr in est_rows_of.items():
            if len(rr) == 0:
                continue
            if x_start is None:
                flat = np.full((len(rr), n_cams, 2), -1e10)
                for c in range(n_cams):
                    v = ~np.isnan(obs_px[rr, c, 0])
                    if v.any():
                        flat[v, c] = pixels_to_flat(obs_px[rr[v], c], cals[c], cpar)
                x0 = np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)
            else:
                x0 = x_start[rr]
            x[rr], cv = triangulate_weighted(obs_px[rr], obs_var[rr], cpar, cals, x0, n_iter=3, cam_index=cam_ix,
                                             return_cov=True)
            pos_var[rr] = np.trace(cv, axis1=1, axis2=2) / 3.0
        return x

    x = fit(raw, raw_var)
    res = RefineResult(x=x, obs_px=raw.copy(), obs_var=raw_var.copy(), n_share=share.copy(),
                       pred_px=np.full((n_est, n_cams, 2), np.nan), x_rounds=[x.copy()])
    for rnd in range(1, rounds + 1):
        wpt = np.where(clean[est].sum(1) >= 2, 1.0, 0.02) if trust else None
        pred3, supp = local_poly_predict(pid_e, fr_e, x, wpt, window=window, order=order, return_support=True)
        good = np.isfinite(pred3[:, 0]) & ((supp >= trust_support) if trust else True)
        pred_px = np.full((n_est, n_cams, 2), np.nan)
        for c in range(n_cams):
            ok = good & det[est, c]
            if ok.any():
                pred_px[ok, c] = project_pixels(pred3[ok], cals[c], cpar)
        cl = clean[est] & np.isfinite(pred_px[:, :, 0])
        var_p = max(float(np.mean((raw - pred_px)[cl] ** 2)) - var_c, 0.01) if cl.any() else 0.25
        obs_px, obs_var = raw.copy(), raw_var.copy()
        n_gated = n_hidden = 0
        for f in frames:
            sel = rows_of[f]
            for c in range(n_cams):
                idx = sel[det[sel, c]]
                b = blob[idx, c]
                o = np.argsort(b, kind="stable")
                idx, b = idx[o], b[o]
                cuts = np.flatnonzero(np.r_[True, b[1:] != b[:-1], True])
                for a, e in zip(cuts[:-1], cuts[1:]):
                    if e - a < 2:
                        if hidden_mass is None:
                            continue
                        r0 = idx[a]
                        s_own = sig.get((int(pid[r0]), c), np.nan)
                        g = sumg[r0, c]
                        if not (np.isfinite(s_own) and s_own > 0 and np.isfinite(g) and g >= hidden_mass * s_own):
                            continue
                        if pidx[r0] < 0:
                            continue
                        ctr = centre[r0, c]
                        _s, v_h = split_blob_effective(ctr, np.vstack([ctr, ctr]), np.array([s_own, g - s_own]),
                                                       var_c, np.array([var_p, var_unknown]))
                        obs_var[pidx[r0], c] = v_h[0]
                        n_hidden += 1
                        continue
                    mem = idx[a:e]
                    ctr = centre[mem[0], c]
                    pm = pidx[mem]
                    pr = np.where((pm >= 0)[:, None], pred_px[np.maximum(pm, 0), c], np.nan)
                    unknown = ~np.isfinite(pr[:, 0])
                    pr[unknown] = ctr
                    vp = np.where(unknown, var_unknown, var_p)
                    bright = np.array([sig.get((int(pid[m]), c), np.nan) for m in mem])
                    s_eff, v_eff = split_blob_effective(ctr, pr, bright, var_c, vp)
                    for j in range(len(mem)):
                        if pm[j] < 0:
                            continue
                        if gate > 0 and not unknown[j]:
                            if np.sum((s_eff[j] - pr[j]) ** 2) / (v_eff[j] + vp[j]) > gate**2:
                                obs_px[pm[j], c] = np.nan
                                obs_var[pm[j], c] = np.nan
                                n_gated += 1
                                continue
                        obs_px[pm[j], c] = s_eff[j]
                        obs_var[pm[j], c] = v_eff[j]
        if prior:
            pri = np.where(good[:, None, None], pred_px, np.nan)
            x = fit(obs_px, obs_var, x_start=x, prior_px=pri, prior_var=prior_scale * var_p)
        else:
            x = fit(obs_px, obs_var, x_start=x)
        res.x_rounds.append(x.copy())
        res.obs_px, res.obs_var, res.pred_px = obs_px, obs_var, pred_px
        res.info.append({"round": rnd, "var_p_px2": var_p, "trusted_share": float(np.mean(good)),
                         "gated_obs": n_gated, "hidden_partner_obs": n_hidden})
    res.x = x
    # back to the caller's rows
    full_x = np.full((n, 3), np.nan)
    full_x[est] = x
    out = RefineResult(x=full_x, obs_px=_expand(res.obs_px, est, n), obs_var=_expand(res.obs_var, est, n),
                       n_share=share, pred_px=_expand(res.pred_px, est, n), pos_var=_expand(pos_var, est, n),
                       x_rounds=[_expand(r, est, n) for r in res.x_rounds], info=res.info)
    return out


def _expand(a: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    out = np.full((n,) + a.shape[1:], np.nan)
    out[rows] = a
    return out

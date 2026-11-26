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

``signatures`` (N, C), if given, replaces the simple per-track signature (see
:func:`openptv2.merge_post.estimate_signatures`). ``partner_min_len``: a member whose track has fewer rows does not
lend its prediction to its partners (it counts as unknown for them): young tracks, most of them new births and some
of them wrong, cannot corrupt the effective observations of established particles.

``mode="joint"``: instead of per-member effective observations, every cluster of points that share blobs in a frame
is solved jointly (:func:`openptv2.blob_model.solve_cluster`): clean blobs give each member's own equations, a shared
blob one equation (centre = brightness-weighted mean of the members' projections), trusted path predictions
pseudo-observations. No member relies on a partner's prediction.

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
    split_blobs_effective,
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
    """Typical grey sum per (particle, camera): the ``quantile`` (median) over the frames where its blob held only
    it, as :func:`signature_from_history` (positive finite values; linear interpolation like ``np.quantile``)."""
    sig = {}
    for c in range(clean.shape[1]):
        m = np.flatnonzero(clean[:, c] & np.isfinite(sumg[:, c]) & (sumg[:, c] > 0))
        if len(m) == 0:
            continue
        o = m[np.lexsort((sumg[m, c], pid[m]))]               # by particle, then by value
        p_o, v_o = pid[o], sumg[o, c]
        cuts = np.flatnonzero(np.r_[True, p_o[1:] != p_o[:-1], True])
        a, cnt = cuts[:-1], np.diff(cuts)
        pos = (cnt - 1) * quantile
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, cnt - 1)
        frac = pos - lo
        v_lo, v_hi = v_o[a + lo], v_o[a + hi]
        val = v_lo + (v_hi - v_lo) * frac
        for pp, vv in zip(p_o[a], val):
            sig[(int(pp), c)] = float(vv)
    return sig


def _poly3_basis(x_norm: np.ndarray) -> np.ndarray:
    """20-term degree-3 3D polynomial basis on normalized coordinates (N, 3)."""
    x, y, z = x_norm[:, 0], x_norm[:, 1], x_norm[:, 2]
    return np.column_stack([
        np.ones_like(x), x, y, z,
        x * x, x * y, x * z, y * y, y * z, z * z,
        x * x * x, x * x * y, x * x * z, x * y * y, x * y * z, x * z * z,
        y * y * y, y * y * z, y * z * z, z * z * z,
    ])


def fit_4cam_residual_field(
    pos4: np.ndarray,
    centre4: np.ndarray,
    cals: list,
    cpar: Any,
    huber_px: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Fit a smooth 3D polynomial field to all-camera reprojection residuals (zero gauge drift).

    By the normal equations of least-squares multi-camera triangulation, the reprojection
    residuals sum to zero in the 3D Jacobian space, so subtracting the fitted field from
    the 2D pixel observations leaves the 4-camera 3D gauge invariant while aligning every
    3-camera subset and down-weighted shared view onto the same 4-camera gauge.
    """
    n_cams = len(cals)
    med = np.median(pos4, axis=0)
    scale = np.maximum(np.std(pos4, axis=0), 1.0)
    b_mat = _poly3_basis((pos4 - med) / scale)
    reg = np.diag([1e-9] + [1e-3] * (b_mat.shape[1] - 1))
    coefs: list[tuple[np.ndarray, np.ndarray]] = []
    for c in range(n_cams):
        proj = project_pixels(pos4, cals[c], cpar)
        res = centre4[:, c, :] - proj
        w = np.ones(len(pos4))
        cx = np.zeros(b_mat.shape[1])
        cy = np.zeros(b_mat.shape[1])
        for _ in range(3):
            bw = b_mat * w[:, None]
            lhs = bw.T @ b_mat + reg
            cx = np.linalg.solve(lhs, bw.T @ res[:, 0])
            cy = np.linalg.solve(lhs, bw.T @ res[:, 1])
            err = np.hypot(res[:, 0] - b_mat @ cx, res[:, 1] - b_mat @ cy)
            w = np.where(err < huber_px, 1.0, huber_px / np.maximum(err, 1e-6))
        coefs.append((cx, cy))
    return med, scale, coefs


def eval_4cam_residual_field(
    pos: np.ndarray,
    field: tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]],
) -> np.ndarray:
    """Evaluate the fitted (N, C, 2) pixel residual field at 3D positions ``pos``."""
    med, scale, coefs = field
    b_mat = _poly3_basis((pos - med) / scale)
    out = np.zeros((len(pos), len(coefs), 2))
    for c, (cx, cy) in enumerate(coefs):
        out[:, c, 0] = b_mat @ cx
        out[:, c, 1] = b_mat @ cy
    return out


def align_store_correspondences_4cam(
    store: Any,
    frames: list[int],
    cals: list,
    cpar: Any,
    vpar: Any,
    *,
    min_align_points: int = 100,
    var_c: float = 0.01,
) -> int:
    """Align partial-camera (e.g. 3-camera) correspondences in ``store`` onto the 4-camera gauge.

    Fits the zero-gauge-drift 4-camera residual field on all clean 4-camera points across
    ``frames``, then re-triangulates only the partial-camera points (2 <= seen < n_cams)
    with aligned 2D observations. Clean 4-camera points are untouched (zero scale drift).
    Works for every tracker (``two_phase``, ``trackcorr``, ``fast_3d``, custom plugins).

    Returns the number of partial-camera points updated.
    """
    from openptv2.orientation import point_positions

    n_cams = len(cals)
    if n_cams < 4 or not frames:
        return 0

    frame_data = []
    pos4_list = []
    ctr4_list = []
    for f in frames:
        try:
            pos, cam_ids = store.read_correspondences(f)
        except Exception:
            continue
        pos = np.asarray(pos, dtype=float)
        cam_ids = np.asarray(cam_ids, dtype=np.int64)
        if len(pos) == 0 or cam_ids.shape[1] < n_cams:
            continue
        ids = cam_ids[:, :n_cams]
        targs = []
        missing_targets = False
        for c in range(n_cams):
            key = f"targets/cam_{c}/frame_{int(f):06d}"
            if key not in store.root:
                missing_targets = True
                break
            targs.append(np.asarray(store.root[key], dtype=float))
        if missing_targets:
            continue
        n = len(pos)
        centre = np.full((n, n_cams, 2), np.nan, dtype=float)
        shared = np.zeros((n, n_cams), dtype=bool)
        for c in range(n_cams):
            sel = np.flatnonzero((ids[:, c] >= 0) & (ids[:, c] < len(targs[c])))
            if len(sel):
                centre[sel, c] = targs[c][ids[sel, c], 1:3]
                _, inv, cnt = np.unique(ids[sel, c], return_inverse=True, return_counts=True)
                shared[sel, c] = cnt[inv] >= 2
        clean = (ids >= 0) & ~shared & np.isfinite(centre[:, :, 0])
        ok4 = (clean.sum(axis=1) == n_cams) & np.isfinite(pos[:, 0])
        if ok4.any():
            pos4_list.append(pos[ok4])
            ctr4_list.append(centre[ok4])
        frame_data.append((f, pos, cam_ids, ids, centre))

    if not pos4_list:
        return 0
    pos4 = np.vstack(pos4_list)
    ctr4 = np.vstack(ctr4_list)
    if len(pos4) < min_align_points:
        return 0

    x4_px = triangulate_weighted(
        ctr4, np.full((len(ctr4), n_cams), var_c), cpar, cals, pos4, n_iter=3, frozen_jacobian=True
    )
    ok_px = np.isfinite(x4_px[:, 0])
    if ok_px.sum() < min_align_points:
        return 0
    field = fit_4cam_residual_field(x4_px[ok_px], ctr4[ok_px], cals, cpar)

    n_updated = 0
    for f, pos, cam_ids, ids, centre in frame_data:
        n_seen = (ids >= 0).sum(axis=1)
        partial = (n_seen >= 2) & (n_seen < n_cams) & np.isfinite(pos[:, 0])
        if not partial.any():
            continue
        p_idx = np.flatnonzero(partial)
        delta = eval_4cam_residual_field(pos[p_idx], field)
        c_ali = centre[p_idx] - delta
        flat = np.full((len(p_idx), n_cams, 2), -1e10, dtype=float)
        for c in range(n_cams):
            v = ~np.isnan(c_ali[:, c, 0])
            if v.any():
                flat[v, c] = pixels_to_flat(c_ali[v, c], cals[c], cpar)
        x_new = np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], dtype=float)
        ok_new = np.isfinite(x_new[:, 0])
        if ok_new.any():
            pos_out = pos.copy()
            pos_out[p_idx[ok_new]] = x_new[ok_new]
            store.write_correspondences(f, pos_out, cam_ids)
            n_updated += int(ok_new.sum())
    return n_updated


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
    prior: bool = False,
    prior_scale: float = 1.0,
    var_unknown: float = 1.0,
    hidden_mass: float | None = 1.4,
    signatures: np.ndarray | None = None,
    partner_min_len: int = 0,
    mode: str = "effective",
    x_init: np.ndarray | None = None,
    shared_only: bool = False,
    pixel_fit: Any = None,
    align_4cam: bool = True,
    min_align_points: int = 100,
) -> RefineResult:
    from openptv2.orientation import point_positions

    frame = np.asarray(frame)
    pid = np.asarray(pid)
    n, n_cams = blob.shape
    det = blob >= 0
    share = shared_blobs(frame, blob)
    clean = det & (share == 1)

    if align_4cam and n_cams >= 4:
        clean4 = clean.sum(1) == n_cams
        if hidden_mass is not None:
            sig_full = np.asarray(signatures, float) if signatures is not None else None
            if sig_full is None:
                sig_dict = _signatures(pid, clean, sumg)
                up_a, inv_a = np.unique(pid, return_inverse=True)
                tab_a = np.full((len(up_a), n_cams), np.nan)
                for (p_, c_), v_ in sig_dict.items():
                    tab_a[np.searchsorted(up_a, p_), c_] = v_
                sig_full = tab_a[inv_a]
            with np.errstate(invalid="ignore"):
                is_hid = (clean & np.isfinite(sig_full) & (sig_full > 0)
                          & (sumg >= hidden_mass * sig_full)).any(1)
            clean4 = clean4 & ~is_hid
        if x_init is not None:
            x_align = np.asarray(x_init, float)
        else:
            flat_align = np.full((n, n_cams, 2), -1e10)
            for c in range(n_cams):
                v = ~np.isnan(centre[:, c, 0])
                if v.any():
                    flat_align[v, c] = pixels_to_flat(centre[v, c], cals[c], cpar)
            x_align = np.asarray(point_positions(flat_align, cpar, cals, vpar, method="lsq")[0], float)
        ok4 = clean4 & np.isfinite(x_align[:, 0])
        if ok4.sum() >= min_align_points:
            # Solve pixel-space LSQ on clean4 so the 4-camera normal equations sum(J_c^T r_c) = 0 hold exactly
            c4 = centre[ok4]
            x4_px = triangulate_weighted(
                c4, np.full((len(c4), n_cams), var_c), cpar, cals, x_align[ok4], n_iter=3, frozen_jacobian=True
            )
            ok4_px = np.isfinite(x4_px[:, 0])
            if ok4_px.sum() >= min_align_points:
                field = fit_4cam_residual_field(x4_px[ok4_px], c4[ok4_px], cals, cpar)
                ok_x = np.isfinite(x_align[:, 0])
                delta = np.zeros_like(centre)
                delta[ok_x] = eval_4cam_residual_field(x_align[ok_x], field)
                centre = centre - delta
                if x_init is not None and shared_only:
                    three_clean = clean.sum(1) == (n_cams - 1)
                    if three_clean.any():
                        flat_3 = np.full((int(three_clean.sum()), n_cams, 2), -1e10)
                        c_sub = centre[three_clean]
                        m_sub = clean[three_clean]
                        for c in range(n_cams):
                            v = m_sub[:, c] & ~np.isnan(c_sub[:, c, 0])
                            if v.any():
                                flat_3[v, c] = pixels_to_flat(c_sub[v, c], cals[c], cpar)
                        x3 = np.asarray(point_positions(flat_3, cpar, cals, vpar, method="lsq")[0], float)
                        x_init_new = np.asarray(x_init, float).copy()
                        ok3 = np.isfinite(x3[:, 0])
                        if ok3.any():
                            idx3 = np.flatnonzero(three_clean)[ok3]
                            x_init_new[idx3] = x3[ok3]
                        x_init = x_init_new

    est = np.flatnonzero(det.sum(1) >= 2)          # rows that can be triangulated
    pidx = np.full(n, -1)
    pidx[est] = np.arange(len(est))
    n_est = len(est)
    fr_e, pid_e = frame[est], pid[est]
    frames = np.unique(frame)
    rows_of = {f: np.flatnonzero(frame == f) for f in frames}
    sig = _signatures(pid, clean, sumg)
    track_len = np.bincount(np.unique(pid, return_inverse=True)[1])[np.unique(pid, return_inverse=True)[1]]

    if signatures is not None:
        sig_mat = np.asarray(signatures, float)
    else:
        up, inv = np.unique(pid, return_inverse=True)
        table = np.full((len(up), n_cams), np.nan)
        for (p_, c_), v_ in sig.items():
            table[np.searchsorted(up, p_), c_] = v_
        sig_mat = table[inv]

    def sig_of(rows, c):
        return sig_mat[rows, c]
    raw = centre[est]
    raw_var = np.where(np.isnan(raw[:, :, 0]), np.nan, var_c)

    pos_var = np.full(n_est, np.nan)

    # rows whose position can change: a view on a shared blob, or a blob too bright for its own particle
    # (a hidden partner); with ``shared_only`` only those are fitted, the others keep their position
    sig_est = sig_mat[est]
    hidden_row = np.zeros(n_est, bool)
    if hidden_mass is not None:
        with np.errstate(invalid="ignore"):
            hidden_row = (det[est] & (share[est] == 1) & np.isfinite(sig_est) & (sig_est > 0)
                          & (sumg[est] >= hidden_mass * sig_est)).any(1)
    affected = (det[est] & (share[est] >= 2)).any(1) | hidden_row
    fit_rows = np.flatnonzero(affected) if shared_only else np.arange(n_est)
    if shared_only and x_init is None:
        raise ValueError("shared_only needs x_init, the positions the other points keep")

    def fit(obs_px, obs_var, x_start=None, prior_px=None, prior_var=None):
        x = np.full((n_est, 3), np.nan) if x_start is None else x_start.copy()
        cam_ix = list(range(n_cams))
        idx = fit_rows                        # every point is independent of the others: one call for all frames
        if n_est == 0 or len(idx) == 0:
            return x
        obs_px, obs_var = obs_px[idx], obs_var[idx]
        if prior_px is not None:
            obs_px = np.concatenate([obs_px, prior_px[idx]], axis=1)
            obs_var = np.concatenate([obs_var, np.broadcast_to(prior_var, (len(idx), n_cams))], axis=1)
            cam_ix = cam_ix + cam_ix
        if x_start is None:
            flat = np.full((len(idx), n_cams, 2), -1e10)
            for c in range(n_cams):
                v = ~np.isnan(obs_px[:, c, 0])
                if v.any():
                    flat[v, c] = pixels_to_flat(obs_px[v, c], cals[c], cpar)
            x0 = np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)
        else:
            x0 = x_start[idx]
        x[idx], cv = triangulate_weighted(obs_px, obs_var, cpar, cals, x0, n_iter=3, cam_index=cam_ix,
                                          frozen_jacobian=True, return_cov=True)
        pos_var[idx] = np.trace(cv, axis1=1, axis2=2) / 3.0
        return x

    if x_init is not None:
        x = np.asarray(x_init, float)[est].copy()
        if not shared_only:
            x = fit(raw, raw_var, x_start=x)
    else:
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
        n_gated = n_hidden = n_pixel = 0
        for c in range(n_cams):
            idx = np.flatnonzero(det[:, c])
            key = frame[idx].astype(np.int64) * 10_000_000 + blob[idx, c]
            o = np.argsort(key, kind="stable")
            idx, key = idx[o], key[o]
            if pixel_fit is not None:
                ukey, first = np.unique(key, return_index=True)
                blob_ctr, blob_fr = centre[idx[first], c], ukey // 10_000_000
                fcut = np.flatnonzero(np.r_[True, blob_fr[1:] != blob_fr[:-1], True])
                ctrs_of = {int(blob_fr[a]): blob_ctr[a:b] for a, b in zip(fcut[:-1], fcut[1:])}
            cuts = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
            starts, sizes = cuts[:-1], np.diff(cuts)
            # blobs of one particle: only a hidden partner can change them
            if hidden_mass is not None:
                r0 = idx[starts[sizes == 1]]
                s_own, g = sig_mat[r0, c], sumg[r0, c]
                hid = np.isfinite(s_own) & (s_own > 0) & np.isfinite(g) & (g >= hidden_mass * np.where(np.isfinite(s_own), s_own, np.inf))
                hid &= pidx[r0] >= 0
                if hid.any():
                    r0, s_own, g = r0[hid], s_own[hid], g[hid]
                    ctr = centre[r0, c]
                    _s, v_h = split_blobs_effective(
                        ctr, np.stack([ctr, ctr], axis=1), np.stack([s_own, g - s_own], axis=1), var_c,
                        np.broadcast_to([var_p, var_unknown], (len(r0), 2)))
                    obs_var[pidx[r0], c] = v_h[:, 0]
                    n_hidden += len(r0)
            for m in np.unique(sizes[sizes >= 2]):
                st = starts[sizes == m]
                mem = idx[st[:, None] + np.arange(m)[None, :]]            # (G, m) rows of the blobs
                ctr = centre[mem[:, 0], c]
                pm = pidx[mem]
                pr = np.where((pm >= 0)[:, :, None], pred_px[np.maximum(pm, 0), c], np.nan)
                unknown = ~np.isfinite(pr[:, :, 0])
                pr = np.where(unknown[:, :, None], ctr[:, None, :], pr)
                vp = np.where(unknown, var_unknown, var_p)
                bright = sig_mat[mem, c]
                young = track_len[mem] < partner_min_len
                s_eff, v_eff = split_blobs_effective(ctr, pr, bright, var_c, vp)
                yg = np.flatnonzero(young.any(1))
                if len(yg):
                    # a young member's own observation is computed normally, but for its partners it counts as
                    # unknown: one split per member, with the others' predictions as trusted or not
                    for j in range(m):
                        mask = (np.arange(m) != j)[None, :] & young[yg]
                        pr_j = np.where(mask[:, :, None], ctr[yg][:, None, :], pr[yg])
                        vp_j = np.where(mask, var_unknown, vp[yg])
                        s_j, v_j = split_blobs_effective(ctr[yg], pr_j, bright[yg], var_c, vp_j)
                        s_eff[yg, j], v_eff[yg, j] = s_j[:, j], v_j[:, j]
                if pixel_fit is not None and m <= 4:
                    # measure the members of the shared blob from the pixels; the track prediction is the seed
                    prior_px = max(float(np.sqrt(var_p)), float(getattr(pixel_fit, "min_prior_px", 0.15)))
                    for gi in np.flatnonzero(~unknown.any(1)):
                        f_g = int(frame[mem[gi, 0]])
                        r = pixel_fit(f_g, c, ctr[gi], pr[gi], prior_px, ctrs_of.get(f_g))
                        if r is None:
                            continue
                        p_fit, v_fit = r
                        okm = np.isfinite(v_fit)
                        s_eff[gi, okm], v_eff[gi, okm] = p_fit[okm], v_fit[okm]
                        n_pixel += int(okm.sum())
                keep = pm >= 0
                if gate > 0:
                    bad = keep & ~unknown & (np.sum((s_eff - pr) ** 2, axis=2) / (v_eff + vp) > gate**2)
                else:
                    bad = np.zeros_like(keep)
                n_gated += int(bad.sum())
                ok = keep & ~bad
                obs_px[pm[bad], c] = np.nan
                obs_var[pm[bad], c] = np.nan
                obs_px[pm[ok], c] = s_eff[ok]
                obs_var[pm[ok], c] = v_eff[ok]
        if prior:
            pri = np.where(good[:, None, None], pred_px, np.nan)
            x = fit(obs_px, obs_var, x_start=x, prior_px=pri, prior_var=prior_scale * var_p)
        else:
            x = fit(obs_px, obs_var, x_start=x)
        if mode == "joint":
            x = _joint_clusters(x, frame, est, pidx, blob, det, share, centre, obs_var, pred_px, good, var_c,
                                prior_scale * var_p if prior else None, sig_of, cpar, cals, rows_of)
        res.x_rounds.append(x.copy())
        res.obs_px, res.obs_var, res.pred_px = obs_px, obs_var, pred_px
        res.info.append({"round": rnd, "var_p_px2": var_p, "trusted_share": float(np.mean(good)),
                         "gated_obs": n_gated, "hidden_partner_obs": n_hidden, "pixel_fit_obs": n_pixel})
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


def _joint_clusters(x, frame, est, pidx, blob, det, share, centre, obs_var, pred_px, good, var_c, prior_var, sig_of,
                    cpar, cals, rows_of, max_move_mm: float = 1.0):
    """Replace the positions of every cluster of estimable rows that share blobs by a joint solve."""
    from openptv2.blob_model import brightness_shares, solve_cluster

    x = x.copy()
    n_cams = blob.shape[1]
    for f, sel in rows_of.items():
        rows = sel[pidx[sel] >= 0]
        if len(rows) == 0:
            continue
        parent = {int(r): int(r) for r in rows}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        groups: dict = {}
        for c in range(n_cams):
            for r in rows:
                if det[r, c] and share[r, c] >= 2:
                    groups.setdefault((c, int(blob[r, c])), []).append(int(r))
        for mem in groups.values():
            for r in mem[1:]:
                ra, rb = find(mem[0]), find(r)
                if ra != rb:
                    parent[rb] = ra
        clusters: dict = {}
        for r in rows:
            clusters.setdefault(find(int(r)), []).append(int(r))
        for mem in clusters.values():
            if len(mem) < 2:
                continue
            loc = {r: k for k, r in enumerate(mem)}
            clean_obs, prior_obs, shared_obs = [], [], []
            for r in mem:
                k = loc[r]
                e = pidx[r]
                for c in range(n_cams):
                    if det[r, c] and share[r, c] == 1 and np.isfinite(obs_var[e, c]):
                        clean_obs.append((k, c, centre[r, c], float(obs_var[e, c])))
                    if prior_var is not None and good[e] and np.isfinite(pred_px[e, c, 0]):
                        prior_obs.append((k, c, pred_px[e, c], float(prior_var)))
            for (c, b), gm in groups.items():
                if gm[0] not in loc:
                    continue
                allm = [r for r in sel if det[r, c] and blob[r, c] == b]
                if any(r not in loc for r in allm):
                    continue                      # a member outside the cluster (no 3D unknown): skip this equation
                w = brightness_shares(sig_of(np.array(allm), c))
                shared_obs.append((tuple(loc[r] for r in allm), tuple(w), c, centre[allm[0], c], var_c))
            idx = np.array([pidx[r] for r in mem])
            if not np.all(np.isfinite(x[idx])):
                continue
            x_new, (c0, c1) = solve_cluster(x[idx], clean_obs, shared_obs, prior_obs, cpar, cals, return_info=True)
            # keep the per-member result unless the joint solve clearly fits better and stays near it
            if c1 < c0 and np.all(np.linalg.norm(x_new - x[idx], axis=1) < max_move_mm):
                x[idx] = x_new
    return x

"""Merge-aware refinement of a finished tracking run (H2 of docs/plans/2026-11-20-holistic-ptv.md).

A tracker (e.g. ``two_phase``) leaves a run store with 3D points, the blobs they use and their links. Here the
tracks are used to find which blobs are shared by several particles and to correct the positions of the
particles in them (:func:`openptv2.merge_refine.refine_merged`). No truth is used.

Membership inference (v1): a tracked point that has no blob in a camera is a *member of the blob* its projected
position falls on (within ``radius`` px) if there is one. If another point already uses that blob, the blob is
shared; if the blob is free, the point simply takes it (a missed detection). Points never share a blob unless a
point without a view in that camera falls on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from openptv2.loco import project_pixels


def chains_from_linkage(prev_by_frame: dict, next_by_frame: dict, max_step: int = 3) -> dict:
    """Track id of every row, from the linkage arrays ``{frame: prev}``, ``{frame: next}`` (row indices in the
    neighbouring frame; a link may bridge a gap, the frame step is found where the neighbour reciprocates).

    Returns ``{frame: tid array}`` with tid >= 0 for rows in a chain (including single rows), numbered
    consecutively; every row gets an id.
    """
    frames = sorted(prev_by_frame)
    tid = {f: np.full(len(prev_by_frame[f]), -1, dtype=np.int64) for f in frames}
    nxt_id = 0
    for f in frames:
        prev = prev_by_frame[f]
        for i in range(len(prev)):
            if tid[f][i] >= 0 or prev[i] >= 0:
                continue                                    # not a chain start
            cur_f, cur_i = f, i
            while True:
                tid[cur_f][cur_i] = nxt_id
                n = int(next_by_frame[cur_f][cur_i])
                if n < 0:
                    break
                step = 0
                for s in range(1, max_step + 1):
                    p = prev_by_frame.get(cur_f + s)
                    if p is not None and 0 <= n < len(p) and p[n] == cur_i:
                        step = s
                        break
                step = step or 1                            # a one-sided link is followed one frame
                if cur_f + step not in tid or n >= len(tid[cur_f + step]) or tid[cur_f + step][n] >= 0:
                    break
                cur_f, cur_i = cur_f + step, n
            nxt_id += 1
    for f in frames:                                        # rows only reached backwards (broken links)
        for i in np.flatnonzero(tid[f] < 0):
            tid[f][i] = nxt_id
            nxt_id += 1
    return tid


def extend_membership(
    frame: np.ndarray,
    pos: np.ndarray,
    blob: np.ndarray,
    centre: np.ndarray,
    sumg: np.ndarray,
    blobs: dict,
    cpar: Any,
    cals: list,
    radius: float = 2.0,
    min_cams: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Give points without a blob in a camera the blob their projection falls on.

    ``blobs[(frame, cam)] = (centres (B, 2), grey sums (B,))``. Only points with >= ``min_cams`` cameras
    (a trustworthy position) are extended. Returns new ``blob``, ``centre``, ``sumg`` and the number of
    attachments.
    """
    blob, centre, sumg = blob.copy(), centre.copy(), sumg.copy()
    n_cams = blob.shape[1]
    ncam = (blob >= 0).sum(1)
    n_att = 0
    for f in np.unique(frame):
        rows = np.flatnonzero(frame == f)
        for c in range(n_cams):
            miss = rows[(blob[rows, c] < 0) & (ncam[rows] >= min_cams)]
            if len(miss) == 0 or (f, c) not in blobs:
                continue
            cent, sg = blobs[(f, c)]
            if len(cent) == 0:
                continue
            px = project_pixels(pos[miss], cals[c], cpar)
            d, j = cKDTree(cent).query(px, distance_upper_bound=radius)
            hit = np.isfinite(d)
            if not hit.any():
                continue
            r = miss[hit]
            blob[r, c] = j[hit]
            centre[r, c] = cent[j[hit]]
            sumg[r, c] = sg[j[hit]]
            n_att += int(hit.sum())
    return blob, centre, sumg, n_att


def arrays_from_store(store: Any, linkage_name: str, first: int, last: int, n_cams: int) -> dict:
    """All correspondence rows of a run store as flat arrays plus the blob tables of every frame and camera."""
    from openptv2.tracking_postprocess import read_linkage

    prev_by, next_by, pos_l, ids_l, fr_l, row_l = {}, {}, [], [], [], []
    blobs = {}
    for f in range(first, last + 1):
        if not store.has_correspondences(f):
            continue
        pos, ids = store.read_correspondences(f)
        ids = np.asarray(ids, dtype=np.int64)[:, :n_cams]
        lk = read_linkage(linkage_name, f, store=store)
        if lk is None:
            prev = np.full(len(pos), -1)
            nxt = np.full(len(pos), -2)
        else:
            prev, nxt, _ = lk
        prev_by[f], next_by[f] = np.asarray(prev), np.asarray(nxt)
        pos_l.append(np.asarray(pos, float))
        ids_l.append(ids)
        fr_l.append(np.full(len(pos), f))
        row_l.append(np.arange(len(pos)))
        for c in range(n_cams):
            rows = np.asarray(store.root[f"targets/cam_{c}/frame_{f:06d}"])
            blobs[(f, c)] = (rows[:, 1:3].astype(float), rows[:, 6].astype(float))
    frame = np.concatenate(fr_l)
    row = np.concatenate(row_l)
    pos = np.vstack(pos_l)
    blob = np.vstack(ids_l)
    tid_by = chains_from_linkage(prev_by, next_by)
    tid = np.concatenate([tid_by[f] for f in sorted(tid_by)])
    centre = np.full(blob.shape + (2,), np.nan)
    sumg = np.full(blob.shape, np.nan)
    for c in range(n_cams):
        for f in np.unique(frame):
            r = np.flatnonzero((frame == f) & (blob[:, c] >= 0))
            if len(r):
                cent, sg = blobs[(int(f), c)]
                centre[r, c] = cent[blob[r, c]]
                sumg[r, c] = sg[blob[r, c]]
    return {"frame": frame, "row": row, "pos": pos, "blob": blob, "centre": centre, "sumg": sumg,
            "tid": tid, "blobs": blobs}


def _hermite(p0, p1, v0, v1, g, span):
    """Cubic Hermite position ``g`` frames after p0 on a gap of ``span`` frames (velocities in mm/frame)."""
    u = g / span
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * p0 + h10 * span * v0 + h01 * p1 + h11 * span * v1


def track_signatures(a: dict) -> dict:
    """Typical grey sum per (track id, camera) from the frames where the track's blob is used by one point."""
    from openptv2.merge_refine import _signatures, shared_blobs

    share = shared_blobs(a["frame"], a["blob"])
    return _signatures(a["tid"], (a["blob"] >= 0) & (share == 1), a["sumg"])


def fill_gaps(
    a: dict,
    pos: np.ndarray,
    cpar: Any,
    cals: list,
    radius: float = 1.5,
    max_gap: int = 3,
    min_cams: int = 3,
    min_rows: int = 4,
    interp: str = "linear",
    radius_growth: float = 0.0,
    radius_max: float = 4.0,
    mass_range: tuple[float, float] | None = None,
) -> dict:
    """Rows for the frames a track skipped: where its path says the particle is, blobs are claimed.

    For every track with at least ``min_rows`` rows, every missing frame between two of its rows (at most
    ``max_gap`` frames) gets a predicted position (``interp``: ``linear``, or ``hermite`` using the end
    velocities), projected into every camera; the nearest blob within ``radius`` px (growing by
    ``radius_growth`` px per frame of distance from the nearer known row, at most ``radius_max``) is taken in
    each camera (a blob another point already uses is shared), and the row is added if at least ``min_cams``
    cameras have one. With ``mass_range = (lo, hi)`` a blob is accepted only if its grey sum is within
    ``lo..hi`` times the sum of the typical grey sums (signatures) of the points already using it and of the
    claiming track, when all are known: the brightness-mass check.
    Returns a dict with the new rows (``frame``, ``tid``, ``pos``, ``blob``, ``centre``, ``sumg``) in the same
    layout as :func:`arrays_from_store`.
    """
    frame, tid = a["frame"], a["tid"]
    blobs = a["blobs"]
    n_cams = a["blob"].shape[1]
    order = np.lexsort((frame, tid))
    t_sorted = tid[order]
    cuts = np.flatnonzero(np.r_[True, t_sorted[1:] != t_sorted[:-1], True])
    todo = []                                                  # (frame, tid, predicted position, distance to a row)
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        if s1 - s0 < min_rows:
            continue
        idx = order[s0:s1]
        f = frame[idx]
        for i in np.flatnonzero(np.diff(f) > 1):
            span = int(f[i + 1] - f[i])
            gap = span - 1
            if gap > max_gap:
                continue
            p0, p1 = pos[idx[i]], pos[idx[i + 1]]
            if interp == "hermite":
                v0 = (p0 - pos[idx[i - 1]]) / (f[i] - f[i - 1]) if i > 0 else (p1 - p0) / span
                v1 = ((pos[idx[i + 2]] - p1) / (f[i + 2] - f[i + 1])) if i + 2 < len(idx) else (p1 - p0) / span
            for g in range(1, gap + 1):
                pg = _hermite(p0, p1, v0, v1, g, span) if interp == "hermite" else p0 + (p1 - p0) * g / span
                todo.append((int(f[i]) + g, int(tid[idx[i]]), pg, min(g, span - g)))
    if not todo:
        return {k: np.zeros((0,) + v.shape[1:], v.dtype) for k, v in
                (("frame", frame), ("tid", tid), ("pos", pos), ("blob", a["blob"]), ("centre", a["centre"]),
                 ("sumg", a["sumg"]))}
    tf = np.array([t[0] for t in todo])
    tt = np.array([t[1] for t in todo])
    tp = np.array([t[2] for t in todo])
    td = np.array([t[3] for t in todo], float)
    blob = np.full((len(todo), n_cams), -1, dtype=a["blob"].dtype)
    centre = np.full((len(todo), n_cams, 2), np.nan)
    sumg = np.full((len(todo), n_cams), np.nan)
    if mass_range is not None:
        sig = track_signatures(a)
        users: dict = {}
        for r in np.flatnonzero((a["blob"] >= 0).any(1)):
            for c in range(n_cams):
                if a["blob"][r, c] >= 0:
                    users.setdefault((int(frame[r]), c, int(a["blob"][r, c])), []).append(int(tid[r]))
    for f in np.unique(tf):
        rows = np.flatnonzero(tf == f)
        for c in range(n_cams):
            if (int(f), c) not in blobs:
                continue
            cent, sg = blobs[(int(f), c)]
            if len(cent) == 0:
                continue
            px = project_pixels(tp[rows], cals[c], cpar)
            tree = cKDTree(cent)
            for k, r in enumerate(rows):
                rad = min(radius + radius_growth * td[r], radius_max)
                d, j = tree.query(px[k], distance_upper_bound=rad)
                if not np.isfinite(d):
                    continue
                if mass_range is not None:
                    owners = users.get((int(f), c, int(j)), [])
                    sigs = [sig.get((o, c), np.nan) for o in owners] + [sig.get((int(tt[r]), c), np.nan)]
                    if not np.any(np.isnan(sigs)):
                        ratio = sg[j] / float(np.sum(sigs))
                        if not (mass_range[0] <= ratio <= mass_range[1]):
                            continue
                blob[r, c] = j
                centre[r, c] = cent[j]
                sumg[r, c] = sg[j]
    ok = (blob >= 0).sum(1) >= min_cams
    return {"frame": tf[ok], "tid": tt[ok], "pos": tp[ok], "blob": blob[ok], "centre": centre[ok], "sumg": sumg[ok]}


def row_signatures(tid: np.ndarray, frame: np.ndarray, blob: np.ndarray, sumg: np.ndarray,
                   quantile: float = 0.5) -> np.ndarray:
    """(N, C) typical grey sum of each row's track in each camera (NaN if unknown); see :func:`track_signatures`."""
    from openptv2.merge_refine import _signatures, shared_blobs

    share = shared_blobs(frame, blob)
    sig = _signatures(tid, (blob >= 0) & (share == 1), sumg, quantile)
    out = np.full(blob.shape, np.nan)
    for c in range(blob.shape[1]):
        out[:, c] = [sig.get((int(t), c), np.nan) for t in tid]
    return out


def birth_targets(
    frame: np.ndarray,
    blob: np.ndarray,
    pred_px: np.ndarray,
    sig: np.ndarray,
    blobs: dict,
    counts: dict,
    min_resid: float = 0.5,
    max_shift_px: float = 3.0,
    fallback_px: np.ndarray | None = None,
) -> dict:
    """Virtual targets for births, per (frame, camera): the free blobs and the *residual targets*.

    A blob used by tracked points whose typical grey sums (``sig``) and trusted predicted pixel positions
    (``pred_px``) are known, and whose grey sum ``G`` exceeds the members' sum ``sum S_m`` by at least ``min_resid``
    times the camera's median signature, holds an unseen particle of brightness ``R = G - sum S_m``. The blob centre
    ``c`` is the brightness-weighted mean, so that particle sits at ``s = (G c - sum S_m p_m) / R``.
    ``blobs[(f, c)] = (centres, grey sums)``, ``counts[(f, c)] = (n, nx, ny)`` per blob.
    ``fallback_px`` (N, C, 2), if given, replaces a missing trusted prediction (e.g. the projection of the
    member's refined position).
    Returns ``{(f, c): dict(xy (k, 2), sumg (k,), npx (k, 3), src (k,), kind (k,))}`` with ``kind`` 0 = free blob,
    1 = residual target; ``src`` is the blob index the target comes from.
    """
    if fallback_px is not None:
        pred_px = np.where(np.isfinite(pred_px), pred_px, fallback_px)
    out = {}
    n_cams = blob.shape[1]
    med_sig = [np.nanmedian(sig[:, c]) if np.isfinite(sig[:, c]).any() else np.nan for c in range(n_cams)]
    for f in np.unique(frame):
        rows = np.flatnonzero(frame == f)
        for c in range(n_cams):
            if (int(f), c) not in blobs:
                continue
            cent, sg = blobs[(int(f), c)]
            npx = counts[(int(f), c)]
            used = blob[rows, c]
            members: dict = {}
            for r, b in zip(rows, used):
                if b >= 0:
                    members.setdefault(int(b), []).append(r)
            free = np.setdiff1d(np.arange(len(cent)), np.array(sorted(members), dtype=int))
            xy, s_, n_, src, kind = [cent[free]], [sg[free]], [npx[free]], [free], [np.zeros(len(free), int)]
            for b, mem in members.items():
                s_m = sig[mem, c]
                p_m = pred_px[mem, c]
                if not (np.all(np.isfinite(s_m)) and np.all(np.isfinite(p_m))):
                    continue
                resid = sg[b] - s_m.sum()
                if not (np.isfinite(med_sig[c]) and resid >= min_resid * med_sig[c]):
                    continue
                s = (sg[b] * cent[b] - (s_m[:, None] * p_m).sum(0)) / resid
                if np.linalg.norm(s - cent[b]) > max_shift_px:
                    continue
                xy.append(s[None, :])
                s_.append(np.array([resid]))
                n_.append(npx[b][None, :])
                src.append(np.array([b]))
                kind.append(np.array([1]))
            out[(int(f), c)] = {"xy": np.vstack(xy), "sumg": np.concatenate(s_), "npx": np.vstack(n_),
                                "src": np.concatenate(src).astype(int), "kind": np.concatenate(kind)}
    return out


def birth_points(targets: dict, frames, cpar: Any, cals: list, vpar: Any, need_residual: bool = True,
                 min_cams: int = 3) -> dict:
    """Match the virtual targets of every frame with the standard matcher; keep points with >= ``min_cams``
    cameras (and at least one residual target if ``need_residual``). Returns rows (``frame``, ``pos``, ``blob``
    = source blob index per camera, -1 none, ``n_resid``)."""
    from openptv2.correspondences import match_frame_correspondences
    from openptv2.tracking_framebuf import TargetArray

    n_cams = len(cals)
    fr_l, pos_l, blob_l, nres_l = [], [], [], []
    for f in frames:
        dets, keys = [], []
        for c in range(n_cams):
            t = targets.get((int(f), c))
            if t is None or len(t["xy"]) == 0:
                dets.append(TargetArray(0))
                keys.append(None)
                continue
            o = np.argsort(t["xy"][:, 1], kind="stable")          # the matcher wants y-sorted targets
            ta = TargetArray(len(o))
            for i, k in enumerate(o):
                ta[i].set_pnr(i)
                ta[i].set_pos((float(t["xy"][k, 0]), float(t["xy"][k, 1])))
                ta[i].set_pixel_counts(int(t["npx"][k, 0]), int(t["npx"][k, 1]), int(t["npx"][k, 2]))
                ta[i].set_sum_grey_value(int(max(t["sumg"][k], 1)))
                ta[i].set_tnr(-1)
            dets.append(ta)
            keys.append(o)
        if sum(len(d) for d in dets) == 0:
            continue
        pos, ids = match_frame_correspondences(dets, cpar, cals, vpar)
        ids = np.asarray(ids)[:, :n_cams]
        if len(pos) == 0:
            continue
        src = np.full(ids.shape, -1, int)
        nres = np.zeros(len(ids), int)
        for c in range(n_cams):
            u = ids[:, c] >= 0
            if not u.any():
                continue
            t = targets[(int(f), c)]
            k = keys[c][ids[u, c]]
            src[u, c] = t["src"][k]
            nres[u] += t["kind"][k]
        keep = ((src >= 0).sum(1) >= min_cams) & ((nres >= 1) if need_residual else True)
        if keep.any():
            fr_l.append(np.full(int(keep.sum()), f))
            pos_l.append(np.asarray(pos, float)[keep])
            blob_l.append(src[keep])
            nres_l.append(nres[keep])
    if not fr_l:
        return {"frame": np.zeros(0, int), "pos": np.zeros((0, 3)), "blob": np.zeros((0, n_cams), int),
                "n_resid": np.zeros(0, int)}
    return {"frame": np.concatenate(fr_l), "pos": np.vstack(pos_l), "blob": np.vstack(blob_l),
            "n_resid": np.concatenate(nres_l)}


def paired_births(
    frame: np.ndarray,
    blob: np.ndarray,
    sig: np.ndarray,
    blobs: dict,
    counts: dict,
    frames,
    cpar: Any,
    cals: list,
    vpar: Any,
    radius: float = 2.0,
    mass_lo: float = 0.4,
    mass_hi: float = 2.5,
    min_cams: int = 3,
) -> dict:
    """Births of particles that have only two free views: a 2-camera match of free blobs, *verified by mass* in the
    other cameras.

    The free blobs (used by no point) of every frame are matched with the standard matcher in pair mode (also
    giving triplets and quads of free blobs). In each camera without a free blob in the candidate, its projection
    must fall on a blob (within ``radius`` px) that is brighter than the signatures of the points using it by
    ``mass_lo``..``mass_hi`` times the candidate's own brightness (the mean grey sum of its free blobs): that
    camera then counts as a hidden view and the blob is shared. A candidate is kept with at least ``min_cams``
    free + hidden views. Returns rows (``frame``, ``pos``, ``blob``, ``n_hidden``).
    """
    from openptv2.correspondences import match_frame_correspondences
    from openptv2.tracking_framebuf import TargetArray

    n_cams = len(cals)
    old_flag = getattr(cpar, "pair_flag", 0)
    cpar.pair_flag = 1
    fr_l, pos_l, blob_l, nh_l = [], [], [], []
    try:
        for f in frames:
            f = int(f)
            rows = np.flatnonzero(frame == f)
            owners: list[dict] = []
            dets, free_of = [], []
            for c in range(n_cams):
                cent, sg = blobs[(f, c)]
                own: dict = {}
                for r in rows:
                    b = blob[r, c]
                    if b >= 0:
                        own.setdefault(int(b), []).append(r)
                owners.append(own)
                free = np.setdiff1d(np.arange(len(cent)), np.array(sorted(own), dtype=int))
                free = free[np.argsort(cent[free, 1], kind="stable")]       # y-sorted for the matcher
                ta = TargetArray(len(free))
                npx = counts[(f, c)]
                for i, k in enumerate(free):
                    ta[i].set_pnr(i)
                    ta[i].set_pos((float(cent[k, 0]), float(cent[k, 1])))
                    ta[i].set_pixel_counts(int(npx[k, 0]), int(npx[k, 1]), int(npx[k, 2]))
                    ta[i].set_sum_grey_value(int(max(sg[k], 1)))
                    ta[i].set_tnr(-1)
                dets.append(ta)
                free_of.append(free)
            if sum(len(d) for d in dets) == 0:
                continue
            pos, ids = match_frame_correspondences(dets, cpar, cals, vpar)
            ids = np.asarray(ids)[:, :n_cams]
            for p, row in zip(np.asarray(pos, float), ids):
                src = np.full(n_cams, -1, int)
                bright = []
                for c in range(n_cams):
                    if row[c] >= 0:
                        src[c] = free_of[c][row[c]]
                        bright.append(blobs[(f, c)][1][src[c]])
                if len(bright) == 0:
                    continue
                b_mean = float(np.mean(bright))
                n_hidden = 0
                for c in range(n_cams):
                    if src[c] >= 0:
                        continue
                    cent, sg = blobs[(f, c)]
                    if len(cent) == 0:
                        continue
                    px = project_pixels(p[None], cals[c], cpar)[0]
                    d = np.linalg.norm(cent - px, axis=1)
                    j = int(np.argmin(d))
                    if d[j] > radius or j not in owners[c]:
                        continue
                    s_own = sig[owners[c][j], c]
                    if not np.all(np.isfinite(s_own)):
                        continue
                    resid = sg[j] - float(np.sum(s_own))
                    if mass_lo * b_mean <= resid <= mass_hi * b_mean:
                        src[c] = j
                        n_hidden += 1
                if (src >= 0).sum() >= min_cams and n_hidden >= 1:
                    fr_l.append(f)
                    pos_l.append(p)
                    blob_l.append(src)
                    nh_l.append(n_hidden)
    finally:
        cpar.pair_flag = old_flag
    if not fr_l:
        return {"frame": np.zeros(0, int), "pos": np.zeros((0, 3)), "blob": np.zeros((0, n_cams), int),
                "n_hidden": np.zeros(0, int)}
    return {"frame": np.array(fr_l), "pos": np.array(pos_l), "blob": np.array(blob_l), "n_hidden": np.array(nh_l)}


def stamp_tnr(store: Any, frames, n_cams: int) -> None:
    """Set every target's ``tnr`` to the first point (row) of its frame that uses it, -1 if none (as the image
    pipeline does after matching; trackers read it to tell used from free blobs)."""
    from openptv2.storage.run_store import _frame_key

    for f in frames:
        _, ids = store.read_correspondences(f)
        ids = np.asarray(ids)
        for c in range(n_cams):
            key = f"targets/cam_{c}/{_frame_key(f)}"
            t = np.asarray(store.root[key]).copy()
            t[:, 7] = -1
            col = ids[:, c] if ids.ndim == 2 and ids.shape[1] > c else np.zeros(0, int)
            used = np.flatnonzero(col >= 0)
            for r in used[::-1]:                         # reversed: the first user wins
                t[int(col[r]), 7] = r
            store.root[key] = t


def estimate_signatures(
    tid: np.ndarray,
    frame: np.ndarray,
    blob: np.ndarray,
    sumg: np.ndarray,
    n_iter: int = 3,
    start_quantile: float = 0.2,
    lo: float = 0.75,
    hi: float = 1.3,
    min_obs: int = 2,
    cross_camera: bool = True,
    cross_ratio: float = 1.35,
) -> np.ndarray:
    """Typical grey sum of every track in every camera, estimated jointly with how much each blob holds.

    A track's frames where no other *tracked* point uses its blob are not all clean: an untracked particle may hide
    in the blob and inflate it. Start from a low quantile of the track's grey sums; then repeat: an observation is
    *explained* when its blob's grey sum is ``lo``..``hi`` times the sum of the signatures of the tracked points in it;
    each member's share of an explained blob is ``G * S_m / sum S``; a track's signature is the median of its explained
    shares (kept if it has at least ``min_obs`` of them).
    With ``cross_camera``: a particle's brightness per camera is about a camera gain times its own brightness, so
    where a camera's signature is missing, or more than ``cross_ratio`` times what the particle's other cameras predict
    (a partner hid in that camera's blob in all its frames), the cross-camera value replaces it.
    Returns ``(N, C)`` per row, NaN if unknown.
    """
    from openptv2.merge_refine import _signatures, shared_blobs

    n, n_cams = blob.shape
    det = blob >= 0
    share = shared_blobs(frame, blob)
    sig_d = _signatures(tid, det & (share == 1), sumg, start_quantile)
    sig = np.full((n, n_cams), np.nan)
    for c in range(n_cams):
        sig[:, c] = [sig_d.get((int(t), c), np.nan) for t in tid]
    for _ in range(n_iter):
        new = sig.copy()
        for c in range(n_cams):
            rows = np.flatnonzero(det[:, c])
            key = frame[rows].astype(np.int64) * 10_000_000 + blob[rows, c]
            order = np.argsort(key, kind="stable")
            rows, key = rows[order], key[order]
            cuts = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
            ex_rows, ex_share = [], []
            for a, b in zip(cuts[:-1], cuts[1:]):
                mem = rows[a:b]
                s_m = sig[mem, c]
                if not np.all(np.isfinite(s_m)) or s_m.sum() <= 0:
                    continue
                g = sumg[mem[0], c]
                ratio = g / s_m.sum()
                if lo <= ratio <= hi:
                    ex_rows.append(mem)
                    ex_share.append(g * s_m / s_m.sum())
            if not ex_rows:
                continue
            er = np.concatenate(ex_rows)
            es = np.concatenate(ex_share)
            t_er = tid[er]
            o = np.argsort(t_er, kind="stable")
            t_s, v_s = t_er[o], es[o]
            tc = np.flatnonzero(np.r_[True, t_s[1:] != t_s[:-1], True])
            med = {}
            for a, b in zip(tc[:-1], tc[1:]):
                if b - a >= min_obs:
                    med[int(t_s[a])] = float(np.median(v_s[a:b]))
            for t, m in med.items():
                new[tid == t, c] = m
        sig = new
    if cross_camera and n_cams > 1:
        ut, first = np.unique(tid, return_index=True)
        s_t = sig[first]                                               # (T, C) per track
        full = np.all(np.isfinite(s_t) & (s_t > 0), axis=1)
        if full.sum() >= 10:
            logs = np.log(s_t[full])
            gain = np.exp(np.median(logs - logs.mean(1, keepdims=True), axis=0))   # relative camera gains
            rel = s_t / gain[None, :]
            for k in range(len(ut)):
                for c in range(n_cams):
                    others = [rel[k, o] for o in range(n_cams) if o != c and np.isfinite(rel[k, o]) and rel[k, o] > 0]
                    if not others:
                        continue
                    pred = gain[c] * float(np.median(others))
                    if not np.isfinite(s_t[k, c]) or s_t[k, c] > cross_ratio * pred:
                        s_t[k, c] = pred
            inv = np.searchsorted(ut, tid)
            sig = s_t[inv]
    return sig

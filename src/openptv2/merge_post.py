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

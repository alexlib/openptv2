"""Blob-ownership outer loop around two-phase tracking.

A detected 2D blob is not always a single particle: at realistic densities
15-30% of blob observations contain two or more overlapping particles.
One-to-one stereo correspondence forces each blob onto a single 3D point,
which (1) shifts the kept particle's centroid, (2) drops the overlapping
partner from the point cloud, and (3) leaves the kept particle treating the
merged blob as a clean observation because its partner is hidden.

This module wraps :class:`openptv2.plugins.two_phase_tracking.TwoPhaseTracker`
(unchanged as the linker) in an alternating loop that maintains many-to-one
blob membership across frames:

1. Link the current 3D particle cloud with ``two_phase`` (using de-blended /
   reprojected 2D leaves on shared blobs).
2. Estimate each trajectory's intrinsic per-camera brightness and build the
   blob membership table ``M(frame, cam, blob)``.
3. Bring hidden partners back via:
   (a) track-driven gap-filling and head/tail extension into both free and
       already-occupied blobs, and
   (b) residual-pool correspondence discovery combining free blobs with
       virtual de-blended targets from excess-brightness blobs.
4. Jointly refine 3D positions of particles sharing blobs via damped
   Gauss-Newton on the brightness-weighted centroid forward model, and
   downweight still-unexplained excess-brightness blobs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class BlobLoopConfig:
    """Configuration for the blob-ownership outer loop."""

    n_iters: int = 3
    min_track_len: int = 3
    attach_radius_px: float = 2.5
    recover_radius_px: float = 3.0
    merge_radius_px: float = 3.5
    max_gap_fill: int = 3
    excess_frac: float = 0.35
    min_residual_sumg: float = 120.0
    brightness_quantile: float = 0.40
    dedup_mm: float = 0.25
    membership_radius_mm: float = 0.6
    max_shift_mm: float = 0.8
    gn_iters: int = 3
    gn_damping: float = 1e-4
    tether_weight: float = 0.05
    excess_downweight: float = 0.15
    flag_merged: bool = True
    merged_noise_scale: float = 1.4
    recover_tracks: bool = True
    discover_residual: bool = True
    joint_deblend: bool = True
    downweight_excess: bool = True
    final_deblend: bool = True
    verbose: bool = True


# ---------------------------------------------------------------------------
# Camera projection helpers
# ---------------------------------------------------------------------------


def project_points_per_cam(
    pts: np.ndarray, cals: list[Any], cpar: Any
) -> np.ndarray:
    """Project (N, 3) 3D points into all C cameras -> (N, C, 2) pixel coords."""
    from openptv2.algorithms.imgcoord import img_coord_batch

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    n_cams = len(cals)
    out = np.full((n, n_cams, 2), np.nan, dtype=np.float64)
    if n == 0:
        return out
    mm = cpar.mm
    pix_x, pix_y = float(cpar.pix_x), float(cpar.pix_y)
    hx, hy = float(cpar.imx) / 2.0, float(cpar.imy) / 2.0
    for c, cal in enumerate(cals):
        m = np.asarray(img_coord_batch(pts, cal, mm), dtype=np.float64)
        out[:, c, 0] = m[:, 0] / pix_x + hx
        out[:, c, 1] = hy - m[:, 1] / pix_y
    return out


def project_and_jacobian_per_cam(
    pts: np.ndarray, cals: list[Any], cpar: Any, eps_mm: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Batched projections (N, C, 2) and finite-diff Jacobians (N, C, 2, 3)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    n_cams = len(cals)
    if n == 0:
        return (
            np.empty((0, n_cams, 2), dtype=np.float64),
            np.empty((0, n_cams, 2, 3), dtype=np.float64),
        )
    stacked = np.empty((4 * n, 3), dtype=np.float64)
    stacked[0:n] = pts
    for ax in range(3):
        p_eps = pts.copy()
        p_eps[:, ax] += eps_mm
        stacked[(ax + 1) * n : (ax + 2) * n] = p_eps
    proj = project_points_per_cam(stacked, cals, cpar)
    u0 = proj[0:n]
    jac = np.empty((n, n_cams, 2, 3), dtype=np.float64)
    inv_eps = 1.0 / eps_mm
    for ax in range(3):
        jac[:, :, :, ax] = (proj[(ax + 1) * n : (ax + 2) * n] - u0) * inv_eps
    return u0, jac


def triangulate_pixels(
    pixel_xy: np.ndarray, cals: list[Any], cpar: Any
) -> tuple[np.ndarray, float] | None:
    """Triangulate a single 3D point from (C, 2) pixel coords (NaN = unused)."""
    from openptv2.algorithms.constants import COORD_UNUSED
    from openptv2.algorithms.orientation import point_position
    from openptv2.transforms import convert_arr_pixel_to_metric, distorted_to_flat

    n_cams = len(cals)
    valid = np.all(np.isfinite(pixel_xy), axis=1)
    if int(valid.sum()) < 2:
        return None
    targets_metric = np.full((n_cams, 2), COORD_UNUSED, dtype=np.float64)
    for c in np.flatnonzero(valid):
        met = convert_arr_pixel_to_metric(pixel_xy[c : c + 1], cpar)
        targets_metric[c] = distorted_to_flat(met, cals[c])[0]
    pos, dist = point_position(targets_metric, n_cams, cpar.mm, cals, method="lsq")
    pos = np.asarray(pos, dtype=np.float64)
    if not np.all(np.isfinite(pos)):
        return None
    return pos, float(dist)


# ---------------------------------------------------------------------------
# Trajectory extraction & brightness model
# ---------------------------------------------------------------------------


def links_to_tracks(
    links: list[tuple[int, int, int, int]],
) -> list[list[tuple[int, int]]]:
    """Convert two_phase link tuples (t0, i0, t1, i1) into sorted track lists."""
    fwd: dict[tuple[int, int], tuple[int, int]] = {}
    bwd: dict[tuple[int, int], tuple[int, int]] = {}
    nodes: set[tuple[int, int]] = set()
    for t0, i0, t1, i1 in links:
        a, b = (int(t0), int(i0)), (int(t1), int(i1))
        fwd[a] = b
        bwd[b] = a
        nodes.add(a)
        nodes.add(b)
    tracks: list[list[tuple[int, int]]] = []
    visited: set[tuple[int, int]] = set()
    # Start from track heads (nodes with no predecessor)
    for node in sorted(nodes):
        if node in bwd or node in visited:
            continue
        tr: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = node
        while cur is not None and cur not in visited:
            visited.add(cur)
            tr.append(cur)
            cur = fwd.get(cur)
        tracks.append(tr)
    return tracks


def compute_cam_gain_offsets(
    frame_cam_ids: list[np.ndarray],
    raw_rows: list[list[np.ndarray | None]],
    n_cams: int,
) -> np.ndarray:
    """Per-camera median log(sumg) offset for cross-camera brightness fallback."""
    vals = [[] for _ in range(n_cams)]
    for cam_ids, rows in zip(frame_cam_ids, raw_rows):
        for c in range(n_cams):
            if rows[c] is None or len(cam_ids) == 0:
                continue
            v = cam_ids[:, c]
            sel = v[v >= 0]
            if len(sel):
                vals[c].append(np.log(np.maximum(rows[c][sel, 6], 1.0)))
    med = np.zeros(n_cams, dtype=np.float64)
    for c in range(n_cams):
        if vals[c]:
            med[c] = float(np.median(np.concatenate(vals[c])))
    return med


def build_blob_membership(
    frame_cam_ids: list[np.ndarray],
    active_masks: list[np.ndarray],
    n_cams: int,
) -> list[list[dict[int, list[int]]]]:
    """Map [t][c][blob_idx] -> list of active particle indices in frame t."""
    membership: list[list[dict[int, list[int]]]] = []
    for cam_ids, active in zip(frame_cam_ids, active_masks):
        per_cam: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(n_cams)]
        for i in np.flatnonzero(active):
            ii = int(i)
            for c in range(n_cams):
                b = int(cam_ids[ii, c])
                if b >= 0:
                    per_cam[c][b].append(ii)
        membership.append(per_cam)
    return membership


def attach_missing_blobs(
    frame_pos: list[np.ndarray],
    frame_cam_ids: list[np.ndarray],
    active_masks: list[np.ndarray],
    raw_rows: list[list[np.ndarray | None]],
    raw_trees: list[list[cKDTree | None]],
    cals: list[Any],
    cpar: Any,
    attach_radius_px: float,
) -> int:
    """Attach active particles with cam_ids[i, c] == -1 to any blob they project
    onto within ``attach_radius_px``. This immediately exposes merges where
    particle p2 was triangulated from its other 2-3 cameras while p1 took the
    merged blob in camera c."""
    n_cams = len(cals)
    attached = 0
    for t in range(len(frame_pos)):
        act = np.flatnonzero(active_masks[t])
        if len(act) == 0:
            continue
        cam_ids = frame_cam_ids[t]
        need = np.any(cam_ids[act] < 0, axis=1)
        idx = act[need]
        if len(idx) == 0:
            continue
        proj = project_points_per_cam(frame_pos[t][idx], cals, cpar)
        for c in range(n_cams):
            tree = raw_trees[t][c]
            if tree is None:
                continue
            miss = cam_ids[idx, c] < 0
            if not np.any(miss):
                continue
            sub_idx = idx[miss]
            uv = proj[miss, c]
            finite = np.all(np.isfinite(uv), axis=1)
            if not np.any(finite):
                continue
            sub_idx = sub_idx[finite]
            uv = uv[finite]
            dist, b_idx = tree.query(uv, distance_upper_bound=attach_radius_px)
            hit = np.isfinite(dist)
            if np.any(hit):
                cam_ids[sub_idx[hit], c] = b_idx[hit].astype(np.int32)
                attached += int(hit.sum())
    return attached


def estimate_particle_brightness(
    tracks: list[list[tuple[int, int]]],
    frame_cam_ids: list[np.ndarray],
    raw_rows: list[list[np.ndarray | None]],
    membership: list[list[dict[int, list[int]]]],
    cam_gain_offsets: np.ndarray,
    n_cams: int,
    quantile: float = 0.40,
    min_track_len: int = 3,
) -> list[np.ndarray]:
    """Estimate intrinsic per-camera brightness A_{p, c} for every particle.

    For tracked particles, takes the lower-median (``quantile``) of ``sumg``
    over frames where the track is the sole known occupant of its blob in
    camera c (resisting upward bias from unrecovered hidden partners). Falls
    back to cross-camera gain scaling when a track has no sole-occupant frames
    in camera c, and to the raw blob ``sumg`` for untracked particles.
    """
    n_frames = len(frame_cam_ids)
    # Default: raw blob sumg (or NaN if unseen)
    amp = []
    for t in range(n_frames):
        n = len(frame_cam_ids[t])
        a = np.full((n, n_cams), np.nan, dtype=np.float64)
        for c in range(n_cams):
            rows_c = raw_rows[t][c]
            if rows_c is None or n == 0:
                continue
            v = frame_cam_ids[t][:, c]
            ok = v >= 0
            a[ok, c] = np.maximum(rows_c[v[ok], 6], 1.0)
        amp.append(a)

    for tr in tracks:
        if len(tr) < min_track_len:
            continue
        clean_vals: list[list[float]] = [[] for _ in range(n_cams)]
        all_vals: list[list[float]] = [[] for _ in range(n_cams)]
        for t, i in tr:
            for c in range(n_cams):
                b = int(frame_cam_ids[t][i, c])
                if b < 0 or raw_rows[t][c] is None:
                    continue
                sg = float(max(raw_rows[t][c][b, 6], 1.0))
                all_vals[c].append(sg)
                if len(membership[t][c].get(b, ())) <= 1:
                    clean_vals[c].append(sg)

        est = np.full(n_cams, np.nan, dtype=np.float64)
        for c in range(n_cams):
            if clean_vals[c]:
                est[c] = float(np.quantile(clean_vals[c], quantile))
            elif all_vals[c]:
                est[c] = float(np.quantile(all_vals[c], 0.25))

        # Cross-camera gain fallback for any camera still missing
        have = np.isfinite(est)
        if np.any(have) and not np.all(have):
            ref_log = float(np.median(np.log(est[have]) - cam_gain_offsets[have]))
            for c in np.flatnonzero(~have):
                est[c] = float(np.exp(ref_log + cam_gain_offsets[c]))

        if np.all(np.isfinite(est)):
            for t, i in tr:
                amp[t][i] = est

    # Fill any remaining NaN with camera median
    cam_med = np.exp(cam_gain_offsets)
    for t in range(n_frames):
        for c in range(n_cams):
            bad = ~np.isfinite(amp[t][:, c])
            if np.any(bad):
                amp[t][bad, c] = cam_med[c]
    return amp


# ---------------------------------------------------------------------------
# Residual virtual targets (exposing hidden partners)
# ---------------------------------------------------------------------------


def compute_residual_blobs_frame(
    pos: np.ndarray,
    cam_ids: np.ndarray,
    amp: np.ndarray,
    membership_t: list[dict[int, list[int]]],
    rows_t: list[np.ndarray | None],
    cals: list[Any],
    cpar: Any,
    cfg: BlobLoopConfig,
) -> tuple[list[dict[int, tuple[np.ndarray, float]]], list[set[int]]]:
    """For one frame, identify occupied blobs with significant excess brightness
    and compute their virtual de-blended 2D target (u_res, S_res).

    Returns
    -------
    residuals_by_cam : list[dict[int, tuple[np.ndarray, float]]]
        ``residuals_by_cam[c][blob_idx] = (u_res_xy, S_res)``
    excess_blobs_by_cam : list[set[int]]
        Set of blob indices per camera flagged as having unexplained excess
        brightness.
    """
    n_cams = len(cals)
    proj = project_points_per_cam(pos, cals, cpar)
    residuals_by_cam: list[dict[int, tuple[np.ndarray, float]]] = [
        {} for _ in range(n_cams)
    ]
    excess_blobs: list[set[int]] = [set() for _ in range(n_cams)]

    for c in range(n_cams):
        rows_c = rows_t[c]
        if rows_c is None:
            continue
        for b, occ in membership_t[c].items():
            if not occ:
                continue
            s_obs = float(max(rows_c[b, 6], 1.0))
            u_obs = rows_c[b, 1:3].astype(np.float64)
            s_known = 0.0
            moment_known = np.zeros(2, dtype=np.float64)
            ok = True
            for i in occ:
                if not np.all(np.isfinite(proj[i, c])):
                    ok = False
                    break
                a_ic = float(amp[i, c])
                s_known += a_ic
                moment_known += a_ic * proj[i, c]
            if not ok or s_known <= 0.0:
                continue
            s_res = s_obs - s_known
            if s_res > max(cfg.min_residual_sumg, cfg.excess_frac * s_known):
                excess_blobs[c].add(b)
                u_res = (s_obs * u_obs - moment_known) / s_res
                delta = u_res - u_obs
                norm = float(np.linalg.norm(delta))
                if norm > cfg.merge_radius_px:
                    u_res = u_obs + delta * (cfg.merge_radius_px / norm)
                residuals_by_cam[c][b] = (u_res, s_res)
    return residuals_by_cam, excess_blobs


# ---------------------------------------------------------------------------
# Hidden-partner recovery (3a: track-driven into shared blobs)
# ---------------------------------------------------------------------------


def _claim_into_any_blob(
    pred_pos: np.ndarray,
    rows_t: list[np.ndarray | None],
    trees_t: list[cKDTree | None],
    membership_t: list[dict[int, list[int]]],
    residuals_t: list[dict[int, tuple[np.ndarray, float]]],
    cals: list[Any],
    cpar: Any,
    cfg: BlobLoopConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Match a predicted 3D position against free OR occupied blobs in all
    cameras and triangulate.

    For occupied blobs with a virtual residual target ``u_res``, uses ``u_res``
    for triangulation; for occupied blobs without ``u_res``, allows matching
    within ``merge_radius_px`` and prefers clean cameras if >= 2 are clean.
    """
    n_cams = len(cals)
    proj = project_points_per_cam(pred_pos[None, :], cals, cpar)[0]
    cam_ids = np.full(n_cams, -1, dtype=np.int32)
    clean_xy = np.full((n_cams, 2), np.nan, dtype=np.float64)
    all_xy = np.full((n_cams, 2), np.nan, dtype=np.float64)
    n_clean = 0
    n_res_occ = 0

    for c in range(n_cams):
        tree = trees_t[c]
        rows_c = rows_t[c]
        if tree is None or rows_c is None or not np.all(np.isfinite(proj[c])):
            continue
        k = min(4, len(rows_c))
        dists, idxs = tree.query(
            proj[c], k=k, distance_upper_bound=max(cfg.recover_radius_px, cfg.merge_radius_px)
        )
        best_b = -1
        best_xy = None
        best_score = np.inf
        best_is_clean = False
        best_is_res = False
        for d, j in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
            if not np.isfinite(d):
                continue
            b = int(j)
            is_occ = len(membership_t[c].get(b, ())) > 0
            if not is_occ:
                if d <= cfg.recover_radius_px and d < best_score:
                    best_b = b
                    best_xy = rows_c[b, 1:3].astype(np.float64)
                    best_score = float(d)
                    best_is_clean = True
                    best_is_res = False
            else:
                if b in residuals_t[c]:
                    u_res, _ = residuals_t[c][b]
                    d_res = float(np.linalg.norm(proj[c] - u_res))
                    if d_res <= cfg.recover_radius_px and (d_res + 0.25) < best_score:
                        best_b = b
                        best_xy = u_res
                        best_score = d_res + 0.25
                        best_is_clean = False
                        best_is_res = True
                elif d <= cfg.merge_radius_px and (d + 0.75) < best_score:
                    best_b = b
                    best_xy = rows_c[b, 1:3].astype(np.float64)
                    best_score = float(d) + 0.75
                    best_is_clean = False
                    best_is_res = False

        if best_b >= 0 and best_xy is not None:
            cam_ids[c] = best_b
            all_xy[c] = best_xy
            if best_is_clean:
                clean_xy[c] = best_xy
                n_clean += 1
            elif best_is_res:
                n_res_occ += 1

    n_hits = int((cam_ids >= 0).sum())
    if n_hits < 2:
        return None
    # Require at least one clean camera OR at least two residual-confirmed occupied blobs
    if n_clean == 0 and n_res_occ < 2:
        return None

    # Prefer clean-only triangulation when >= 2 clean cameras exist so an
    # occupied blob cannot pull the initial 3D estimate before joint de-blending
    if n_clean >= 2:
        res = triangulate_pixels(clean_xy, cals, cpar)
    else:
        res = triangulate_pixels(all_xy, cals, cpar)
    if res is None:
        return None
    pos, _ = res
    return pos, cam_ids


def recover_from_tracks(
    tracks: list[list[tuple[int, int]]],
    frame_pos: list[np.ndarray],
    frame_cam_ids: list[np.ndarray],
    frame_amp: list[np.ndarray],
    active_masks: list[np.ndarray],
    membership: list[list[dict[int, list[int]]]],
    residuals_all: list[list[dict[int, tuple[np.ndarray, float]]]],
    raw_rows: list[list[np.ndarray | None]],
    raw_trees: list[list[cKDTree | None]],
    cals: list[Any],
    cpar: Any,
    v_max: float,
    cfg: BlobLoopConfig,
) -> tuple[int, list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    """Recover missing detections along track gaps and at track heads/tails,
    allowing matches into already-occupied (merged) blobs.

    Returns
    -------
    n_added : int
    additions_by_frame : list of (pos, cam_ids, est_amp) tuples per frame
    """
    n_frames = len(frame_pos)
    n_cams = len(cals)
    additions: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
        [] for _ in range(n_frames)
    ]

    # Map every existing particle (t, i) -> owning track index (-1 if unlinked)
    owner_track = [np.full(len(frame_pos[t]), -1, dtype=np.int32) for t in range(n_frames)]
    track_frames: list[set[int]] = []
    for tr_idx, tr in enumerate(tracks):
        fset = set()
        for t, i in tr:
            owner_track[t][i] = tr_idx
            fset.add(t)
        track_frames.append(fset)

    # Build per-frame KD-trees of existing particles
    pos_trees = [
        cKDTree(frame_pos[t]) if len(frame_pos[t]) else None for t in range(n_frames)
    ]
    added_pts: list[list[np.ndarray]] = [[] for _ in range(n_frames)]

    def _near_added(t: int, p: np.ndarray) -> bool:
        for q in added_pts[t]:
            if float(np.linalg.norm(p - q)) < cfg.dedup_mm:
                return True
        return False

    gate_mm = min(max(0.5, 0.75 * v_max), 2.5)
    n_added = 0

    def _split_from_partner(
        t: int, pred: np.ndarray, partner_idx: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Idea 11 + 4: predicted track lands within membership_radius_mm of an
        existing 3D point owned by another track. Build constituent cam_ids
        (clean blobs where separated, shared blobs where merged) and initial
        3D position for the hidden partner."""
        proj = project_points_per_cam(pred[None, :], cals, cpar)[0]
        cids = np.full(n_cams, -1, dtype=np.int32)
        clean_xy = np.full((n_cams, 2), np.nan, dtype=np.float64)
        n_clean = 0
        for c in range(n_cams):
            tree = raw_trees[t][c]
            rows_c = raw_rows[t][c]
            if tree is None or rows_c is None or not np.all(np.isfinite(proj[c])):
                continue
            k = min(4, len(rows_c))
            dists, idxs = tree.query(
                proj[c], k=k, distance_upper_bound=max(cfg.recover_radius_px, cfg.merge_radius_px)
            )
            best_b = -1
            best_score = np.inf
            best_clean = False
            for d, j in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                if not np.isfinite(d):
                    continue
                b = int(j)
                is_occ = len(membership[t][c].get(b, ())) > 0
                if not is_occ and d <= cfg.recover_radius_px and d < best_score:
                    best_b = b
                    best_score = float(d)
                    best_clean = True
                elif is_occ and d <= cfg.merge_radius_px and (d + 0.5) < best_score:
                    best_b = b
                    best_score = float(d) + 0.5
                    best_clean = False
            if best_b >= 0:
                cids[c] = best_b
                if best_clean:
                    clean_xy[c] = rows_c[best_b, 1:3].astype(np.float64)
                    n_clean += 1
            else:
                # Fall back to sharing partner's blob if pred projects within merge_radius_px
                pb = int(frame_cam_ids[t][partner_idx, c])
                if pb >= 0:
                    d_pb = float(np.linalg.norm(proj[c] - rows_c[pb, 1:3]))
                    if d_pb <= cfg.merge_radius_px:
                        cids[c] = pb
        if int((cids >= 0).sum()) < 2:
            return None
        if n_clean >= 2:
            res = triangulate_pixels(clean_xy, cals, cpar)
            if res is not None and float(np.linalg.norm(res[0] - pred)) <= gate_mm:
                return res[0], cids
        return pred.copy(), cids

    def _try_add(
        t: int, pred: np.ndarray, track_amp: np.ndarray, tr_idx: int
    ) -> np.ndarray | None:
        nonlocal n_added
        if t < 0 or t >= n_frames or t in track_frames[tr_idx]:
            return None
        if _near_added(t, pred):
            return None

        # Idea 11: check if pred lands inside an existing particle's membership ball
        tree = pos_trees[t]
        if tree is not None:
            d, j = tree.query(pred, distance_upper_bound=cfg.membership_radius_mm)
            if np.isfinite(d):
                j_int = int(j)
                owner = int(owner_track[t][j_int])
                if owner < 0 and d < cfg.dedup_mm:
                    # Unlinked existing particle already sits here; let two_phase link it
                    return None
                if owner >= 0 and owner != tr_idx:
                    # Merged 3D point! Split out constituent for tr_idx (Idea 4)
                    got = _split_from_partner(t, pred, j_int)
                    if got is None:
                        return None
                    pos, cids = got
                    if _near_added(t, pos):
                        return None
                    additions[t].append((pos, cids, track_amp.copy()))
                    added_pts[t].append(pos)
                    track_frames[tr_idx].add(t)
                    n_added += 1
                    return pos

        got = _claim_into_any_blob(
            pred,
            raw_rows[t],
            raw_trees[t],
            membership[t],
            residuals_all[t],
            cals,
            cpar,
            cfg,
        )
        if got is None:
            return None
        pos, cids = got
        if float(np.linalg.norm(pos - pred)) > gate_mm:
            return None
        if _near_added(t, pos):
            return None
        if tree is not None:
            d2, j2 = tree.query(pos, distance_upper_bound=cfg.dedup_mm)
            if np.isfinite(d2) and int(owner_track[t][int(j2)]) < 0:
                return None
        additions[t].append((pos, cids, track_amp.copy()))
        added_pts[t].append(pos)
        track_frames[tr_idx].add(t)
        n_added += 1
        return pos

    for tr_idx, tr in enumerate(tracks):
        if len(tr) < cfg.min_track_len:
            continue
        t0, i0 = tr[0]
        track_amp = frame_amp[t0][i0].copy()

        # 1. Internal gaps (linear interpolation)
        for k in range(len(tr) - 1):
            ta, ia = tr[k]
            tb, ib = tr[k + 1]
            dt = tb - ta
            if 1 < dt <= cfg.max_gap_fill + 1:
                pa = frame_pos[ta][ia]
                pb = frame_pos[tb][ib]
                for step in range(1, dt):
                    frac = step / float(dt)
                    pred = (1.0 - frac) * pa + frac * pb
                    _try_add(ta + step, pred, track_amp, tr_idx)

        # 2. Backward extension from track head (up to 3 frames)
        p_curr = frame_pos[tr[0][0]][tr[0][1]]
        p_next = frame_pos[tr[1][0]][tr[1][1]]
        dt_h = max(1, tr[1][0] - tr[0][0])
        vel_h = (p_curr - p_next) / float(dt_h)
        t_cur = tr[0][0]
        for _ in range(3):
            t_new = t_cur - 1
            pred = p_curr + vel_h
            got_pos = _try_add(t_new, pred, track_amp, tr_idx)
            if got_pos is None:
                break
            vel_h = got_pos - p_curr
            p_curr = got_pos
            t_cur = t_new

        # 3. Forward extension from track tail (up to 3 frames)
        p_curr = frame_pos[tr[-1][0]][tr[-1][1]]
        p_prev = frame_pos[tr[-2][0]][tr[-2][1]]
        dt_t = max(1, tr[-1][0] - tr[-2][0])
        vel_t = (p_curr - p_prev) / float(dt_t)
        t_cur = tr[-1][0]
        for _ in range(3):
            t_new = t_cur + 1
            pred = p_curr + vel_t
            got_pos = _try_add(t_new, pred, track_amp, tr_idx)
            if got_pos is None:
                break
            vel_t = got_pos - p_curr
            p_curr = got_pos
            t_cur = t_new

    return n_added, additions


# ---------------------------------------------------------------------------
# Hidden-partner recovery (3b: residual-pool correspondence discovery)
# ---------------------------------------------------------------------------


def discover_residual_correspondences(
    frame_pos: list[np.ndarray],
    active_masks: list[np.ndarray],
    membership: list[list[dict[int, list[int]]]],
    residuals_all: list[list[dict[int, tuple[np.ndarray, float]]]],
    raw_rows: list[list[np.ndarray | None]],
    cals: list[Any],
    cpar: Any,
    vpar: Any,
    cfg: BlobLoopConfig,
) -> tuple[int, list[list[tuple[np.ndarray, np.ndarray]]]]:
    """Run stereo correspondence matching on the per-frame residual 2D pool:
    all free blobs (not used by any tracked particle) plus virtual de-blended
    targets (u_res, S_res) from excess-brightness occupied blobs.
    """
    from openptv2.correspondences import match_frame_correspondences
    from openptv2.segmentation import TargetArray

    n_frames = len(frame_pos)
    n_cams = len(cals)
    discovered: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(n_frames)
    ]
    n_found = 0

    for t in range(n_frames):
        # Build residual TargetArray per camera + map back to raw blob index
        detections = []
        orig_blob_maps: list[np.ndarray] = []
        is_virtual_maps: list[np.ndarray] = []
        any_virtual = False

        for c in range(n_cams):
            rows_c = raw_rows[t][c]
            if rows_c is None or len(rows_c) == 0:
                detections.append(TargetArray(0))
                orig_blob_maps.append(np.empty(0, dtype=np.int32))
                is_virtual_maps.append(np.empty(0, dtype=bool))
                continue

            occ_set = set(membership[t][c].keys())
            res_dict = residuals_all[t][c]
            cand_rows = []
            cand_orig = []
            cand_virt = []

            for b in range(len(rows_c)):
                if b not in occ_set:
                    cand_rows.append(rows_c[b].copy())
                    cand_orig.append(b)
                    cand_virt.append(False)
                elif b in res_dict:
                    u_res, s_res = res_dict[b]
                    r = rows_c[b].copy()
                    r[1], r[2] = float(u_res[0]), float(u_res[1])
                    r[6] = float(s_res)
                    r[7] = -1
                    cand_rows.append(r)
                    cand_orig.append(b)
                    cand_virt.append(True)
                    any_virtual = True

            if not cand_rows:
                detections.append(TargetArray(0))
                orig_blob_maps.append(np.empty(0, dtype=np.int32))
                is_virtual_maps.append(np.empty(0, dtype=bool))
                continue

            arr = np.asarray(cand_rows, dtype=np.float64)
            orig = np.asarray(cand_orig, dtype=np.int32)
            virt = np.asarray(cand_virt, dtype=bool)
            # Sort by y so match_frame_correspondences' sort_y() preserves order
            order = np.argsort(arr[:, 2], kind="stable")
            arr = arr[order]
            orig = orig[order]
            virt = virt[order]

            ta = TargetArray(len(arr))
            for idx_r, row in enumerate(arr):
                targ = ta[idx_r]
                targ.set_pnr(idx_r)
                targ.set_pos((float(row[1]), float(row[2])))
                targ.set_pixel_counts(
                    int(max(1, round(row[3]))),
                    int(max(1, round(row[4]))),
                    int(max(1, round(row[5]))),
                )
                targ.set_sum_grey_value(int(max(1, round(row[6]))))
                targ.set_tnr(-1)
            detections.append(ta)
            orig_blob_maps.append(orig)
            is_virtual_maps.append(virt)

        if not any_virtual:
            continue

        pos_new, ids_new = match_frame_correspondences(detections, cpar, cals, vpar)
        if len(pos_new) == 0:
            continue

        ids_new = np.asarray(ids_new[:, :n_cams], dtype=np.int32)
        pos_new = np.asarray(pos_new, dtype=np.float64)
        tree_existing = cKDTree(frame_pos[t]) if len(frame_pos[t]) else None

        for k in range(len(pos_new)):
            # Only keep candidates that actually use at least one virtual target
            # (free-only combinations were already seen by initial matching)
            uses_virt = False
            mapped_ids = np.full(n_cams, -1, dtype=np.int32)
            for c in range(n_cams):
                j = int(ids_new[k, c])
                if j >= 0 and j < len(orig_blob_maps[c]):
                    mapped_ids[c] = int(orig_blob_maps[c][j])
                    if is_virtual_maps[c][j]:
                        uses_virt = True
            if not uses_virt or int((mapped_ids >= 0).sum()) < 2:
                continue
            if tree_existing is not None:
                d, _ = tree_existing.query(pos_new[k], distance_upper_bound=cfg.dedup_mm)
                if np.isfinite(d):
                    continue
            discovered[t].append((pos_new[k], mapped_ids))
            n_found += 1

    return n_found, discovered


# ---------------------------------------------------------------------------
# Joint multi-particle 3D de-blending (Gauss-Newton per connected component)
# ---------------------------------------------------------------------------


def joint_deblend_frame(
    pos: np.ndarray,
    cam_ids: np.ndarray,
    amp: np.ndarray,
    active_mask: np.ndarray,
    rows_t: list[np.ndarray | None],
    excess_blobs_t: list[set[int]],
    cals: list[Any],
    cpar: Any,
    cfg: BlobLoopConfig,
) -> tuple[np.ndarray, int]:
    """Jointly refine 3D positions of particles in shared or excess-brightness
    blobs at one frame via damped Gauss-Newton.

    Returns (updated_pos, n_refined_particles).
    """
    n_pts = len(pos)
    n_cams = len(cals)
    if n_pts == 0:
        return pos, 0

    out = pos.copy()
    act = np.flatnonzero(active_mask)
    if len(act) == 0:
        return out, 0

    # Build blob -> active particle list
    blob_occ: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(n_cams)]
    for i in act:
        ii = int(i)
        for c in range(n_cams):
            b = int(cam_ids[ii, c])
            if b >= 0 and rows_t[c] is not None:
                blob_occ[c][b].append(ii)

    # Identify particles that participate in at least one shared blob OR
    # excess-brightness blob
    interesting: set[int] = set()
    edge_u: list[int] = []
    edge_v: list[int] = []
    for c in range(n_cams):
        for b, occ in blob_occ[c].items():
            if len(occ) >= 2:
                interesting.update(occ)
                p0 = occ[0]
                for p1 in occ[1:]:
                    edge_u.append(p0)
                    edge_v.append(p1)
            elif len(occ) == 1 and cfg.downweight_excess and b in excess_blobs_t[c]:
                interesting.add(occ[0])

    if not interesting:
        return out, 0

    # Connected components over interesting particles
    int_list = sorted(interesting)
    local_id = {p: k for k, p in enumerate(int_list)}
    m_int = len(int_list)
    if edge_u:
        lu = [local_id[u] for u in edge_u]
        lv = [local_id[v] for v in edge_v]
        adj = coo_matrix(
            (np.ones(2 * len(lu)), (lu + lv, lv + lu)), shape=(m_int, m_int)
        )
        n_comp, labels = connected_components(adj, directed=False)
    else:
        n_comp = m_int
        labels = np.arange(m_int)

    comps: list[list[int]] = [[] for _ in range(n_comp)]
    for k, lab in enumerate(labels):
        comps[int(lab)].append(int_list[k])

    # Run GN iterations; batch all projection+Jacobian evaluations per iteration
    x_init = out[int_list].copy()
    x_curr = x_init.copy()

    for _it in range(cfg.gn_iters):
        proj, jac = project_and_jacobian_per_cam(x_curr, cals, cpar)
        for comp in comps:
            m = len(comp)
            loc = [local_id[p] for p in comp]
            comp_set = set(comp)
            # Collect all unique (c, b) blobs touching this component
            blobs: list[tuple[int, int, list[int]]] = []
            seen_cb: set[tuple[int, int]] = set()
            for p in comp:
                for c in range(n_cams):
                    b = int(cam_ids[p, c])
                    if b >= 0 and rows_t[c] is not None and (c, b) not in seen_cb:
                        seen_cb.add((c, b))
                        occ = [q for q in blob_occ[c][b] if q in comp_set]
                        if occ:
                            blobs.append((c, b, occ))

            if not blobs:
                continue

            # Count clean cameras per particle in this component (for excess downweighting)
            clean_count = {p: 0 for p in comp}
            for c, b, occ in blobs:
                if len(occ) == 1 and b not in excess_blobs_t[c]:
                    clean_count[occ[0]] += 1

            # Assemble normal equations H dx = g
            dim = 3 * m
            H = np.zeros((dim, dim), dtype=np.float64)
            g = np.zeros(dim, dtype=np.float64)
            comp_idx = {p: idx_c for idx_c, p in enumerate(comp)}
            eff_w = np.zeros(m, dtype=np.float64)

            for c, b, occ in blobs:
                u_obs = rows_t[c][b, 1:3].astype(np.float64)
                # Weight for this blob
                if len(occ) == 1:
                    p = occ[0]
                    if cfg.downweight_excess and b in excess_blobs_t[c] and clean_count[p] >= 2:
                        s_obs = float(max(rows_t[c][b, 6], 1.0))
                        ratio = min(1.0, float(amp[p, c]) / s_obs)
                        w = cfg.excess_downweight * (ratio * ratio)
                    else:
                        w = 1.0
                else:
                    # Shared blob: full weight when joint_deblend is enabled
                    w = 1.0 if cfg.joint_deblend else 0.2

                weights_occ = np.array([float(amp[q, c]) for q in occ], dtype=np.float64)
                w_sum = float(weights_occ.sum())
                if w_sum <= 0.0:
                    continue
                alpha = weights_occ / w_sum

                u_pred = np.zeros(2, dtype=np.float64)
                valid_blob = True
                for q, a_q in zip(occ, alpha):
                    lq = local_id[q]
                    if not np.all(np.isfinite(proj[lq, c])):
                        valid_blob = False
                        break
                    u_pred += a_q * proj[lq, c]
                if not valid_blob:
                    continue

                res = u_obs - u_pred  # (2,)
                # Accumulate J^T W J and J^T W r
                for idx_a, (qa, aa) in enumerate(zip(occ, alpha)):
                    ca = comp_idx[qa]
                    lqa = local_id[qa]
                    Ja = aa * jac[lqa, c]  # (2, 3)
                    eff_w[ca] += w * (aa * aa)
                    g[3 * ca : 3 * ca + 3] += w * (Ja.T @ res)
                    for idx_b, (qb, ab) in enumerate(zip(occ, alpha)):
                        cb = comp_idx[qb]
                        lqb = local_id[qb]
                        Jb = ab * jac[lqb, c]  # (2, 3)
                        H[3 * ca : 3 * ca + 3, 3 * cb : 3 * cb + 3] += w * (Ja.T @ Jb)

            # Levenberg-Marquardt damping + under-constrained tether
            for ca, p in enumerate(comp):
                lp = local_id[p]
                sl = slice(3 * ca, 3 * ca + 3)
                damp = cfg.gn_damping * max(1.0, float(np.trace(H[sl, sl])) / 3.0)
                H[sl, sl] += damp * np.eye(3)
                if eff_w[ca] < 1.2:
                    # Gently tether under-constrained particle to its initial 3D estimate
                    tw = cfg.tether_weight
                    H[sl, sl] += tw * np.eye(3)
                    g[sl] += tw * (x_init[lp] - x_curr[lp])

            try:
                dx = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                continue

            for ca, p in enumerate(comp):
                lp = local_id[p]
                step = dx[3 * ca : 3 * ca + 3]
                if not np.all(np.isfinite(step)):
                    continue
                cand = x_curr[lp] + step
                total_shift = cand - x_init[lp]
                norm = float(np.linalg.norm(total_shift))
                if norm > cfg.max_shift_mm:
                    cand = x_init[lp] + total_shift * (cfg.max_shift_mm / norm)
                x_curr[lp] = cand

    out[int_list] = x_curr
    return out, len(int_list)


# ---------------------------------------------------------------------------
# Main outer loop
# ---------------------------------------------------------------------------


def _build_leaves_and_seen(
    frame_pos: list[np.ndarray],
    frame_cam_ids: list[np.ndarray],
    raw_rows: list[list[np.ndarray | None]],
    membership: list[list[dict[int, list[int]]]] | None,
    excess_all: list[list[set[int]]] | None,
    cals: list[Any],
    cpar: Any,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build (frame_leaves, frame_seen) for TwoPhaseTracker.

    On clean blobs uses the raw 2D blob centroid; on shared or excess-brightness
    blobs uses the particle's reprojected (de-blended) 2D coordinate so
    TwoPhaseTracker's 2D leaf cost is not pulled by the overlapping partner.
    """
    from openptv2.storage.run_store import seen_mask

    n_cams = len(cals)
    frame_leaves = []
    frame_seen = []
    for t, (pos, cam_ids, rows) in enumerate(zip(frame_pos, frame_cam_ids, raw_rows)):
        n = len(pos)
        xy = np.full((n, n_cams, 2), np.nan, dtype=np.float64)
        for c in range(n_cams):
            if rows[c] is not None and n > 0:
                valid = cam_ids[:, c] >= 0
                xy[valid, c] = rows[c][cam_ids[valid, c], 1:3]

        if membership is not None and n > 0:
            # Replace leaves on shared or excess blobs with reprojected positions
            need_proj = np.zeros((n, n_cams), dtype=bool)
            for c in range(n_cams):
                for i in range(n):
                    b = int(cam_ids[i, c])
                    if b < 0:
                        continue
                    if len(membership[t][c].get(b, ())) >= 2 or (
                        excess_all is not None and b in excess_all[t][c]
                    ):
                        need_proj[i, c] = True
            any_row = np.any(need_proj, axis=1)
            if np.any(any_row):
                idx = np.flatnonzero(any_row)
                proj = project_points_per_cam(pos[idx], cals, cpar)
                for k, i in enumerate(idx):
                    for c in range(n_cams):
                        if need_proj[i, c] and np.all(np.isfinite(proj[k, c])):
                            xy[i, c] = proj[k, c]

        frame_leaves.append(np.nan_to_num(xy.reshape(n, -1)))
        frame_seen.append(seen_mask(cam_ids))
    return frame_leaves, frame_seen


def run_blob_ownership_loop(
    frame_pos: list[np.ndarray],
    frame_cam_ids: list[np.ndarray],
    raw_rows: list[list[np.ndarray | None]],
    cals: list[Any],
    cpar: Any,
    vpar: Any,
    track_cfg: dict[str, Any],
    cfg: BlobLoopConfig | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int, int, int]], dict[str, Any]]:
    """Run the blob-ownership outer loop around ``TwoPhaseTracker``.

    Returns
    -------
    frame_pos : list[np.ndarray]
        Refined (and augmented) 3D positions per frame.
    frame_cam_ids : list[np.ndarray]
        Updated camera-to-blob IDs per frame (may share blob indices).
    links : list[tuple[int, int, int, int]]
        Final links from ``TwoPhaseTracker``.
    stats : dict
        Per-iteration diagnostics.
    """
    from openptv2.plugins.two_phase_tracking import (
        AUTO_MAX_NEIGHBOURS,
        AUTO_MIN_KINK_RATIO,
        DEFAULT_Q_SEED,
        DEFAULT_Q_YOUNG,
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
        build_project_fn,
        ghost_probabilities,
        q_model_of,
        resolve_confirm,
        resolve_confirm_auto,
    )
    from openptv2.point_quality import log_brightness_from_arrays, rcm_from_arrays

    cfg = cfg or BlobLoopConfig()
    n_frames = len(frame_pos)
    n_cams = len(cals)

    # Working copies
    pos_list = [np.asarray(p, dtype=np.float64).copy() for p in frame_pos]
    cids_list = [np.asarray(c[:, :n_cams], dtype=np.int32).copy() for c in frame_cam_ids]
    raw_trees = [
        [
            cKDTree(rows[c][:, 1:3]) if (rows[c] is not None and len(rows[c])) else None
            for c in range(n_cams)
        ]
        for rows in raw_rows
    ]

    # Build TwoPhaseTrackerConfig identically to two_phase_tracking.track_store
    leaf_weight = float(track_cfg.get("leaf_weight", 1.0))
    v_max = float(track_cfg.get("v_max", track_cfg.get("dvxmax", 15.5)))
    use_velocity = bool(track_cfg.get("use_velocity", True))
    cost_mode = str(track_cfg.get("cost_mode", "projected"))
    max_gap = int(track_cfg.get("max_gap", 2))
    allow_shared = bool(track_cfg.get("allow_shared", False))
    max_shared = int(track_cfg.get("max_shared", 2))
    share_tol_raw = track_cfg.get("share_tol", 1.0)
    share_tol = None if share_tol_raw is None else float(share_tol_raw)
    max_group_size = int(track_cfg.get("max_group_size", 128))
    confirm_tol, confirm_ends, _ = resolve_confirm(track_cfg, v_max)
    confirm_auto = resolve_confirm_auto(track_cfg)
    confirm_guard = {
        "confirm_auto_max_neighbours": float(
            track_cfg.get("confirm_auto_max_neighbours", AUTO_MAX_NEIGHBOURS)
        ),
        "confirm_auto_min_ratio": float(
            track_cfg.get("confirm_auto_min_ratio", AUTO_MIN_KINK_RATIO)
        ),
        "confirm_auto_radius": float(track_cfg.get("confirm_auto_radius", 5.0)),
    }
    bidirectional = bool(track_cfg.get("bidirectional", False))
    bg_raw = track_cfg.get("blob_gate", None)
    blob_gate = None if bg_raw is None else float(bg_raw)
    q_weight = float(track_cfg.get("q_weight", 0.0))
    q_seed_raw = track_cfg.get("q_seed", DEFAULT_Q_SEED)
    q_seed = None if q_seed_raw is None else float(q_seed_raw)
    q_young = int(track_cfg.get("q_young", DEFAULT_Q_YOUNG))
    q_model = q_model_of(track_cfg)
    bwd_v_max_raw = track_cfg.get("bwd_v_max", None)
    bwd_v_max = None if bwd_v_max_raw is None else float(bwd_v_max_raw)

    want_q = q_weight > 0 or q_seed is not None or q_young > 0
    need_logb = bool(blob_gate) or (want_q and q_model == "rcm_blob")
    project_fn = build_project_fn(cals, cpar)

    tp_cfg = TwoPhaseTrackerConfig(
        v_max=v_max,
        blob_gate=blob_gate,
        q_weight=q_weight,
        q_seed=q_seed,
        q_young=q_young,
        leaf_weight=leaf_weight,
        use_velocity=use_velocity,
        cost_mode=cost_mode,
        max_gap=max_gap,
        allow_shared=allow_shared,
        max_shared=max_shared,
        share_tol=share_tol,
        max_group_size=max_group_size,
        confirm_tol=confirm_tol,
        confirm_ends=confirm_ends,
        confirm_auto=confirm_auto,
        **confirm_guard,
        bidirectional=bidirectional,
        parallel_directions=bool(track_cfg.get("parallel_directions", False)),
        bwd_v_max=bwd_v_max,
    )

    cam_gain_offsets = compute_cam_gain_offsets(cids_list, raw_rows, n_cams)
    membership: list[list[dict[int, list[int]]]] | None = None
    excess_all: list[list[set[int]]] | None = None
    links: list[tuple[int, int, int, int]] = []
    iter_stats: list[dict[str, Any]] = []

    def _run_linker() -> list[tuple[int, int, int, int]]:
        frame_leaves, frame_seen = _build_leaves_and_seen(
            pos_list, cids_list, raw_rows, membership, excess_all, cals, cpar
        )
        frame_ghost = None
        frame_logb = None
        if want_q or need_logb:
            rcm_list = [
                rcm_from_arrays(p, c, r, cals, cpar)
                for p, c, r in zip(pos_list, cids_list, raw_rows)
            ]
            nseen_list = [s.sum(axis=1) for s in frame_seen]
            logb_list = (
                [log_brightness_from_arrays(c, r) for c, r in zip(cids_list, raw_rows)]
                if need_logb
                else None
            )
            if want_q:
                frame_ghost = ghost_probabilities(
                    pos_list, rcm_list, nseen_list, logb_list, q_model
                )
            if blob_gate:
                frame_logb = logb_list
        tracker = TwoPhaseTracker(tp_cfg)
        return tracker.track_frames(
            pos_list,
            frame_leaves,
            project_fn=project_fn,
            frame_seen=frame_seen,
            frame_ghost=frame_ghost,
            frame_logb=frame_logb,
        )

    for it in range(cfg.n_iters):
        links = _run_linker()
        tracks = links_to_tracks(links)

        # Mark particles belonging to tracks of length >= min_track_len as active
        active_masks = [np.zeros(len(p), dtype=bool) for p in pos_list]
        for tr in tracks:
            if len(tr) >= cfg.min_track_len:
                for t, i in tr:
                    active_masks[t][i] = True

        # Attach missing camera blobs for active particles
        n_attached = attach_missing_blobs(
            pos_list,
            cids_list,
            active_masks,
            raw_rows,
            raw_trees,
            cals,
            cpar,
            cfg.attach_radius_px,
        )

        # Build blob membership & estimate per-particle intrinsic brightness
        membership = build_blob_membership(cids_list, active_masks, n_cams)
        frame_amp = estimate_particle_brightness(
            tracks,
            cids_list,
            raw_rows,
            membership,
            cam_gain_offsets,
            n_cams,
            quantile=cfg.brightness_quantile,
            min_track_len=cfg.min_track_len,
        )

        # Compute residual virtual targets for blobs with unexplained excess brightness
        residuals_all = []
        excess_all = []
        n_excess = 0
        for t in range(n_frames):
            res_t, exc_t = compute_residual_blobs_frame(
                pos_list[t],
                cids_list[t],
                frame_amp[t],
                membership[t],
                raw_rows[t],
                cals,
                cpar,
                cfg,
            )
            residuals_all.append(res_t)
            excess_all.append(exc_t)
            n_excess += sum(len(s) for s in exc_t)

        # On the final iteration, optionally do one last joint de-blending pass
        # on the final tracks without adding new unlinked candidates
        if it == cfg.n_iters - 1:
            n_refined = 0
            if cfg.final_deblend and (cfg.joint_deblend or cfg.downweight_excess):
                for t in range(n_frames):
                    pos_list[t], nr = joint_deblend_frame(
                        pos_list[t],
                        cids_list[t],
                        frame_amp[t],
                        active_masks[t],
                        raw_rows[t],
                        excess_all[t],
                        cals,
                        cpar,
                        cfg,
                    )
                    n_refined += nr
            st = {
                "iter": it,
                "links": len(links),
                "tracks": sum(1 for tr in tracks if len(tr) >= cfg.min_track_len),
                "attached": n_attached,
                "excess_blobs": n_excess,
                "recovered_tracks": 0,
                "discovered_residual": 0,
                "refined": n_refined,
            }
            iter_stats.append(st)
            if cfg.verbose:
                print(f"BlobLoop iter {it}: {st}")
            break

        # 3a. Track-driven recovery into free or occupied blobs
        n_rec_tr = 0
        if cfg.recover_tracks:
            n_rec_tr, tr_adds = recover_from_tracks(
                tracks,
                pos_list,
                cids_list,
                frame_amp,
                active_masks,
                membership,
                residuals_all,
                raw_rows,
                raw_trees,
                cals,
                cpar,
                v_max,
                cfg,
            )
            for t in range(n_frames):
                if not tr_adds[t]:
                    continue
                add_p = np.vstack([a[0] for a in tr_adds[t]])
                add_c = np.vstack([a[1] for a in tr_adds[t]])
                add_a = np.vstack([a[2] for a in tr_adds[t]])
                n_old = len(pos_list[t])
                pos_list[t] = np.vstack([pos_list[t], add_p])
                cids_list[t] = np.vstack([cids_list[t], add_c])
                frame_amp[t] = np.vstack([frame_amp[t], add_a])
                active_masks[t] = np.concatenate(
                    [active_masks[t], np.ones(len(add_p), dtype=bool)]
                )
                for k_add in range(len(add_p)):
                    ii = n_old + k_add
                    for c in range(n_cams):
                        b = int(add_c[k_add, c])
                        if b >= 0:
                            membership[t][c][b].append(ii)

        # 3b. Residual-pool correspondence discovery for unseeded hidden partners
        n_disc = 0
        if cfg.discover_residual and vpar is not None:
            # Refresh residuals after track-driven additions
            if n_rec_tr > 0:
                residuals_all = []
                excess_all = []
                for t in range(n_frames):
                    res_t, exc_t = compute_residual_blobs_frame(
                        pos_list[t],
                        cids_list[t],
                        frame_amp[t],
                        membership[t],
                        raw_rows[t],
                        cals,
                        cpar,
                        cfg,
                    )
                    residuals_all.append(res_t)
                    excess_all.append(exc_t)
            n_disc, disc_adds = discover_residual_correspondences(
                pos_list,
                active_masks,
                membership,
                residuals_all,
                raw_rows,
                cals,
                cpar,
                vpar,
                cfg,
            )
            cam_med = np.exp(cam_gain_offsets)
            for t in range(n_frames):
                if not disc_adds[t]:
                    continue
                add_p = np.vstack([a[0] for a in disc_adds[t]])
                add_c = np.vstack([a[1] for a in disc_adds[t]])
                # Estimate newly discovered particle brightness from residual sumg
                add_a = np.tile(cam_med, (len(add_p), 1))
                for k_add in range(len(add_p)):
                    for c in range(n_cams):
                        b = int(add_c[k_add, c])
                        if b >= 0:
                            if b in residuals_all[t][c]:
                                add_a[k_add, c] = max(residuals_all[t][c][b][1], 1.0)
                            elif raw_rows[t][c] is not None:
                                add_a[k_add, c] = max(float(raw_rows[t][c][b, 6]), 1.0)
                pos_list[t] = np.vstack([pos_list[t], add_p])
                cids_list[t] = np.vstack([cids_list[t], add_c])
                frame_amp[t] = np.vstack([frame_amp[t], add_a])
                # Mark discovered candidates inactive for de-blending until two_phase
                # links them on the next iteration (so false discoveries cannot pull
                # confirmed tracks)
                active_masks[t] = np.concatenate(
                    [active_masks[t], np.zeros(len(add_p), dtype=bool)]
                )

        # 4. Joint multi-particle 3D de-blending on all active particles
        n_refined = 0
        if cfg.joint_deblend or cfg.downweight_excess:
            for t in range(n_frames):
                pos_list[t], nr = joint_deblend_frame(
                    pos_list[t],
                    cids_list[t],
                    frame_amp[t],
                    active_masks[t],
                    raw_rows[t],
                    excess_all[t],
                    cals,
                    cpar,
                    cfg,
                )
                n_refined += nr

        st = {
            "iter": it,
            "links": len(links),
            "tracks": sum(1 for tr in tracks if len(tr) >= cfg.min_track_len),
            "attached": n_attached,
            "excess_blobs": n_excess,
            "recovered_tracks": n_rec_tr,
            "discovered_residual": n_disc,
            "refined": n_refined,
        }
        iter_stats.append(st)
        if cfg.verbose:
            print(f"BlobLoop iter {it}: {st}")

    return pos_list, cids_list, links, {"iterations": iter_stats}

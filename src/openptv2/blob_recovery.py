"""Predictor-based hidden-partner recovery (best validated result).

Two-pass procedure (validated on L1_d1: vErr 19.65 -> 19.35):
  1. Run two_phase on the baseline (merged) 3D centres -> tracks.
  2. For each 3D point claimed by >= 2 tracks, treat it as a merged blob and
     split it into constituents using merged_split.split_merged_blob, with the
     tracks' positions as predictions.
  3. Re-run two_phase on the split points.

This is the recommended path (sparse-case gain, dense-case neutral-to-negative
until intensity-based membership is added).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def recover(frames, frame_particles, frame_leaves, project_fn, cals, cpar,
            frame_brightness=None, brightness_threshold=1.3,
            frame_seen=None):
    """Return (new_particles, new_leaves, frame_merged) with hidden partners recovered.

    frame_brightness: optional list of per-point brightness arrays. When given,
    a point is only treated as a merged blob if its brightness exceeds
    brightness_threshold * median brightness (intensity-based membership)."""
    from openptv2.plugins.two_phase_tracking import TwoPhaseTracker, TwoPhaseTrackerConfig

    cfg = TwoPhaseTrackerConfig(v_max=5.0, leaf_weight=1.0, use_velocity=True,
                                cost_mode="projected", max_gap=2)
    tracker = TwoPhaseTracker(cfg)
    links = tracker.track_frames(
        [p.copy() for p in frame_particles],
        [lf.copy() for lf in frame_leaves],
        project_fn=project_fn,
    )
    # chain links into tracks
    nxt = defaultdict(list)
    for (a, ra, b, rb) in links:
        nxt[(a, ra)].append((b, rb))
    allpts = set()
    for fi, P in enumerate(frame_particles):
        for r in range(len(P)):
            allpts.add((fi, r))
    preds = defaultdict(dict)
    tid = 0
    visited = set()
    for start in allpts:
        if start in visited or any((start[0], start[1]) == (b, rb) for (_, _, b, rb) in links):
            continue
        cur = start
        chain = []
        while cur in allpts and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            cands = nxt.get(cur, [])
            if not cands:
                break
            cur = cands[0]
            if cur in visited:
                break
        for (fi, r) in chain:
            preds[tid][fi] = frame_particles[fi][r].copy()
        tid += 1
    tracks_by_frame = defaultdict(list)
    for t, fmap in preds.items():
        for fi, pos in fmap.items():
            tracks_by_frame[fi].append((t, pos))

    new_particles = []
    new_leaves = []
    new_merged = []
    new_seen_all = []
    for fi, P in enumerate(frame_particles):
        pos = np.asarray(P, dtype=float)
        new_pts = []
        new_merged_frame = []
        new_seen_frame = []
        tlist = tracks_by_frame.get(fi, [])
        tpos = np.array([tp for (_, tp) in tlist]) if tlist else np.zeros((0, 3))
        tree = cKDTree(tpos) if len(tpos) else None
        tids_list = [t for (t, _) in tlist]
        for i, p in enumerate(pos):
            close = []
            if tree is not None:
                for j in tree.query_ball_point(p, 0.6):
                    close.append((tids_list[j], tpos[j]))
            # intensity-based membership gate (adaptive by local density)
            bri_ok = True
            if frame_brightness is not None:
                all_bri = np.concatenate([b for b in frame_brightness if len(b)])
                med_bri = np.median(all_bri[all_bri > 0]) if np.any(all_bri > 0) else 1.0
                # local density: neighbour count within 1mm
                if len(pos) > 1:
                    from scipy.spatial import cKDTree as _T
                    _t = _T(pos)
                    nbr = len(_t.query_ball_point(p, 1.0)) - 1
                    med_nbr = np.median([len(_t.query_ball_point(pp, 1.0)) - 1 for pp in pos])
                    adapt = 1.0 + 0.05 * max(0.0, nbr - med_nbr)
                else:
                    adapt = 1.0
                bri_ok = (frame_brightness[fi][i] > (brightness_threshold * adapt) * med_bri)
            if len(close) >= 2 and bri_ok:
                from openptv2 import merged_split
                preds_arr = np.array([tp for (_, tp) in close])
                intensities = np.ones(len(close))
                mp_all = project_fn(p.reshape(1, 3)).reshape(-1)
                n_cams = len(cals)
                merged_px = mp_all.reshape(n_cams, 2)
                seen = np.ones(n_cams, dtype=bool)
                try:
                    split_pos, _, _, _ = merged_split.split_merged_blob(
                        merged_px, seen, intensities, preds_arr, cals, cpar
                    )
                    for sp in split_pos:
                        new_pts.append(sp)
                        new_merged_frame.append(True)
                        new_seen_frame.append(seen.copy())
                except Exception:
                    new_pts.append(p)
                    new_merged_frame.append(False)
                    new_seen_frame.append(seen.copy())
            else:
                new_pts.append(p)
                new_merged_frame.append(False)
                _s = np.ones(len(cals), dtype=bool)
                if frame_seen is not None and fi < len(frame_seen) and i < len(frame_seen[fi]):
                    _s = np.asarray(frame_seen[fi][i], dtype=bool).copy()
                new_seen_frame.append(_s)
        new_seen_all.append(np.array(new_seen_frame, dtype=bool) if new_seen_frame else np.zeros((0, len(cals)), dtype=bool))
        new_particles.append(np.array(new_pts, dtype=float) if new_pts else np.zeros((0, 3)))
        new_merged.append(np.array(new_merged_frame, dtype=bool))
        if new_pts:
            xy = project_fn(np.array(new_pts, dtype=float))
            new_leaves.append(np.nan_to_num(np.asarray(xy).reshape(len(new_pts), -1)))
        else:
            new_leaves.append(np.zeros((0, 0)))
    return new_particles, new_leaves, new_merged, new_seen_all

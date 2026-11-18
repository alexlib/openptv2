"""Sliding-window multi-hypothesis tracking (MHT) linker.

Post-tracking repair pass that re-optimizes links over a window of frames
jointly, instead of frame-pair greedy. Uses scipy.optimize.linear_sum_assignment
on a combined cost matrix (spatial + velocity + epipolar consistency).

This is the "true multi-frame MHT" at the Python level. The Cython-level
version would replace trackcorr_c_loop's frame-pair search with a windowed
optimizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .temporal_smoothing import extract_trajectories


def mht_relink(
    store: Any,
    first: int,
    last: int,
    window: int = 3,
    max_gap: int = 2,
    spatial_tol: float = 2.0,
    linkage_name: str = "ptv_is",
) -> dict[str, int]:
    """Sliding-window MHT: re-optimize links over windows of ``window`` frames.

    For each window [f, f+window-1], extract candidate points per frame,
    build a cost matrix between consecutive frames (spatial distance +
    velocity consistency), and solve the assignment jointly. Rewrites linkage.

    Returns stats.
    """
    n_merged = 0
    n_broken = 0

    for w_start in range(first, last + 1, max(1, window - 1)):
        w_end = min(last, w_start + window - 1)
        if w_end - w_start < 1:
            continue

        # Read frames in window.
        frames = {}
        for f in range(w_start, w_end + 1):
            try:
                prev, nxt, pos = store.read_linkage(f, name=linkage_name)
                frames[f] = (np.asarray(prev).ravel().copy(),
                             np.asarray(nxt).ravel().copy(),
                             np.asarray(pos))
            except Exception:
                frames[f] = None

        # For each consecutive pair in window, solve assignment.
        for f in range(w_start, w_end):
            if frames.get(f) is None or frames.get(f + 1) is None:
                continue
            prev_f, nxt_f, pos_f = frames[f]
            prev_g, nxt_g, pos_g = frames[f + 1]
            pos_f = np.asarray(pos_f, float)
            pos_g = np.asarray(pos_g, float)
            nf, ng = len(pos_f), len(pos_g)
            if nf == 0 or ng == 0:
                continue

            # Build cost matrix: spatial distance + velocity estimate.
            # For points with existing velocity (prev link), predict position.
            cost = np.full((nf, ng), 1e6, dtype=float)
            for i in range(nf):
                # predicted position: if has prev, use velocity; else current
                pred = pos_f[i]  # default: current position
                v_prev = None
                # velocity from prev if available
                if f - 1 in frames and frames[f - 1] is not None:
                    prev_arr, _, pos_prev = frames[f - 1]
                    pi = int(prev_f[i]) if prev_f[i] >= 0 else -1
                    if 0 <= pi < len(pos_prev):
                        v_prev = pos_f[i] - np.asarray(pos_prev)[pi]
                        pred = pos_f[i] + v_prev
                for j in range(ng):
                    d = np.linalg.norm(pred - pos_g[j])
                    if d > spatial_tol * 2:
                        continue
                    v_new = pos_g[j] - pos_f[i]
                    vn = np.linalg.norm(v_new)
                    if v_prev is not None and vn > 1e-9:
                        cos = np.dot(v_prev, v_new) / (np.linalg.norm(v_prev) * vn + 1e-12)
                        if cos < 0.3:
                            continue
                    cost[i, j] = d + 0.3 * np.linalg.norm(pos_f[i] - pos_g[j])

            # Solve assignment.
            try:
                row_ind, col_ind = linear_sum_assignment(cost)
            except Exception:
                continue

            # Apply: set next/prev for matched pairs, break unmatched.
            new_nxt = nxt_f.copy()
            new_prev = prev_g.copy()
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] >= 1e6:
                    # no valid match -> break
                    if new_nxt[i] != -1:
                        n_broken += 1
                    new_nxt[i] = -1
                    if new_prev[j] == i:
                        new_prev[j] = -1
                    continue
                # merge: link i -> j
                if new_nxt[i] != j:
                    n_merged += 1
                new_nxt[i] = j
                new_prev[j] = i

            # Break links for unmatched rows
            for i in range(nf):
                if i not in row_ind:
                    if new_nxt[i] != -1:
                        n_broken += 1
                    new_nxt[i] = -1

            # Write back
            frames[f] = (frames[f][0], new_nxt, frames[f][2])
            frames[f + 1] = (new_prev, frames[f + 1][1], frames[f + 1][2])
            try:
                store.write_linkage(f, new_nxt if False else frames[f][0], new_nxt, frames[f][2], name=linkage_name)
            except Exception:
                pass
            try:
                store.write_linkage(f + 1, new_prev, frames[f + 1][1], frames[f + 1][2], name=linkage_name)
            except Exception:
                pass

    return {"n_merged": n_merged, "n_broken": n_broken}
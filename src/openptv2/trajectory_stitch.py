"""Multi-frame trajectory stitching: merge fragments and split kinks.

Post-tracking repair pass that operates on the store's linkage. It does NOT
re-run trackcorr; instead it repairs the linkage graph:

* Merge: two trajectory fragments whose endpoints are spatially close and
  velocity-consistent across a short gap are linked (bridging missed links).
* Split: a trajectory is broken at frames where the velocity reverses (kink),
  so a ghost-induced kink does not corrupt a real track.

This is the "merging/splitting" repair that fixes fragmentation without
rewriting the trackcorr backbone.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _read_frames(store, first, last, linkage_name):
    out = {}
    for f in range(first, last + 1):
        try:
            prev, nxt, pos = store.read_linkage(f, name=linkage_name)
            out[f] = (np.asarray(prev).ravel().copy(),
                      np.asarray(nxt).ravel().copy(),
                      np.asarray(pos))
        except Exception:
            out[f] = None
    return out


def _extract(frames, first, last):
    """Return list of trajectories as list of (frame, slot)."""
    visited = set()
    trajs = []
    for start_f in range(first, last + 1):
        fd = frames.get(start_f)
        if fd is None:
            continue
        prev_arr, _, _ = fd
        n = len(prev_arr)
        for i in range(n):
            if int(prev_arr[i]) != -1:
                continue
            chain = []
            f, slot = start_f, i
            seen = set()
            while True:
                key = (f, slot)
                if key in seen or f not in frames or frames[f] is None:
                    break
                seen.add(key)
                chain.append(key)
                _, nxt_arr, _ = frames[f]
                if slot >= len(nxt_arr):
                    break
                nx = int(nxt_arr[slot])
                if nx < 0:
                    break
                # recover step via reciprocal prev of next frame
                step = 1
                nf = f + 1
                if nf <= last and frames.get(nf) is not None:
                    p_arr, _, _ = frames[nf]
                    if nx < len(p_arr) and int(p_arr[nx]) == slot:
                        step = 1
                    else:
                        # gap-bridged: find frame where prev==slot
                        for cand in range(f + 2, min(last, f + 4) + 1):
                            if frames.get(cand) is None:
                                continue
                            pa, _, _ = frames[cand]
                            if nx < len(pa) and int(pa[nx]) == slot:
                                step = cand - f
                                break
                nf = f + step
                if nf > last or frames.get(nf) is None:
                    break
                pa, _, _ = frames[nf]
                if nx >= len(pa):
                    break
                f, slot = nf, nx
            if chain:
                for k in chain:
                    visited.add(k)
                trajs.append(chain)
    # singletons for unvisited
    for f in range(first, last + 1):
        fd = frames.get(f)
        if fd is None:
            continue
        prev_arr, _, _ = fd
        for i in range(len(prev_arr)):
            if (f, i) not in visited:
                trajs.append([(f, i)])
                visited.add((f, i))
    return trajs


def _pos(frames, f, slot):
    fd = frames.get(f)
    if fd is None:
        return None
    _, _, pos = fd
    pos = np.asarray(pos)
    if slot >= len(pos):
        return None
    return np.asarray(pos[slot]).ravel()[:3].astype(float)


def _vel(frames, traj):
    if len(traj) < 2:
        return None
    p0 = _pos(frames, *traj[0])
    p1 = _pos(frames, *traj[-1])
    if p0 is None or p1 is None:
        return None
    dt = traj[-1][0] - traj[0][0]
    if dt <= 0:
        return None
    return (p1 - p0) / dt


def stitch_trajectories(
    store: Any,
    first: int,
    last: int,
    max_gap: int = 2,
    spatial_tol: float = 5.0,
    velocity_cos_tol: float = 0.8,
    kink_angle_deg: float = 120.0,
    linkage_name: str = "ptv_is",
) -> dict[str, int]:
    """Merge fragmented trajectories and split at kinks. Mutates store linkage.

    Returns stats dict.
    """
    frames = _read_frames(store, first, last, linkage_name)
    trajs = _extract(frames, first, last)

    # compute endpoints + velocities
    info = []
    for ti, t in enumerate(trajs):
        p_start = _pos(frames, *t[0])
        p_end = _pos(frames, *t[-1])
        v = _vel(frames, t)
        info.append({"idx": ti, "traj": t, "start": t[0], "end": t[-1],
                    "p_start": p_start, "p_end": p_end, "vel": v})

    # --- Merge candidates ---
    # A ends, B starts, gap in [1, max_gap], spatial close, velocity consistent.
    cands = []
    for a in range(len(info)):
        for b in range(len(info)):
            if a == b:
                continue
            ia, ib = info[a], info[b]
            if ia["end"][0] >= ib["start"][0]:
                continue
            gap = ib["start"][0] - ia["end"][0]
            if gap < 1 or gap > max_gap:
                continue
            if ia["p_end"] is None or ib["p_start"] is None:
                continue
            dist = float(np.linalg.norm(ia["p_end"] - ib["p_start"]))
            # expected displacement
            va = ia["vel"]
            vb = ib["vel"]
            vel_ok = True
            if va is not None and vb is not None:
                na = np.linalg.norm(va); nb = np.linalg.norm(vb)
                if na > 1e-9 and nb > 1e-9:
                    cos = float(np.dot(va, vb) / (na * nb))
                    vel_ok = cos >= velocity_cos_tol
            if not vel_ok:
                continue
            # spatial tolerance: allow velocity*gap + tol
            tol = spatial_tol + (np.linalg.norm(va) if va is not None else 1.0) * gap
            if dist > tol:
                continue
            # Smoothness check: merging must not create a kink. Compute the
            # combined trajectory residual; reject if too rough.
            combined = [p for p in ia["traj"]] + [p for p in ib["traj"]]
            cpos = []
            for (f, s) in combined:
                pp = _pos(frames, f, s)
                cpos.append(pp if pp is not None else [np.nan]*3)
            cpos = np.asarray(cpos, float)
            if np.isnan(cpos).any():
                continue
            cresid = float(np.sqrt(np.mean((cpos[2:] - 2*cpos[1:-1] + cpos[:-2])**2))) if len(cpos) >= 3 else 0.0
            if cresid > 2.0:
                continue
            score = dist + cresid
            cands.append((score, a, b, gap))

    cands.sort(key=lambda c: c[0])

    # greedy union
    parent = list(range(len(info)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    merged_pairs = []
    used_ends = set()
    used_starts = set()
    for score, a, b, gap in cands:
        if a in used_ends or b in used_starts:
            continue
        if find(a) == find(b):
            continue
        union(a, b)
        used_ends.add(a)
        used_starts.add(b)
        merged_pairs.append((a, b, gap))

    # --- Split at kinks ---
    # For each trajectory, find kink indices (velocity angle > threshold).
    split_points = {}  # traj index -> set of split-after positions
    cos_thr = np.cos(np.deg2rad(kink_angle_deg))
    for ti, inf in enumerate(info):
        t = inf["traj"]
        if len(t) < 3:
            continue
        splits = set()
        for i in range(1, len(t) - 1):
            p_prev = _pos(frames, *t[i-1])
            p_cur = _pos(frames, *t[i])
            p_next = _pos(frames, *t[i+1])
            if p_prev is None or p_cur is None or p_next is None:
                continue
            v1 = p_cur - p_prev
            v2 = p_next - p_cur
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos = float(np.dot(v1, v2) / (n1 * n2))
            if cos < cos_thr:
                splits.add(i)  # split after index i
        if splits:
            split_points[ti] = splits

    # --- Apply merges and splits to linkage ---
    # Build a map from (frame, slot) -> new prev/next. Start from current.
    # We'll modify frames' prev/next arrays in place, then write back.

    # Apply splits: break links at kink
    n_splits = 0
    for ti, splits in split_points.items():
        t = info[ti]["traj"]
        for i in splits:
            # break between t[i] and t[i+1]
            f1, s1 = t[i]
            f2, s2 = t[i+1]
            # set next of (f1,s1) = -1
            if frames[f1] is not None:
                prev1, nxt1, pos1 = frames[f1]
                if s1 < len(nxt1) and nxt1[s1] != -1:
                    nxt1[s1] = -1
                    n_splits += 1
            # set prev of (f2,s2) = -1
            if frames[f2] is not None:
                prev2, nxt2, pos2 = frames[f2]
                if s2 < len(prev2) and prev2[s2] != -1:
                    prev2[s2] = -1

    # Apply merges: link end of A to start of B
    n_merges = 0
    for a, b, gap in merged_pairs:
        ia, ib = info[a], info[b]
        fA, sA = ia["end"]
        fB, sB = ib["start"]
        if frames[fA] is None or frames[fB] is None:
            continue
        prevA, nxtA, posA = frames[fA]
        prevB, nxtB, posB = frames[fB]
        if sA >= len(nxtA) or sB >= len(prevB):
            continue
        # set next of A's end = sB (with gap encoded as step)
        # For gap=1, next stores slot in next frame. For gap>1, store slot
        # and rely on reciprocal prev; trackcorr's _link_step recovers gap.
        nxtA[sA] = sB
        prevB[sB] = sA
        n_merges += 1

    # Write back all frames
    for f in range(first, last + 1):
        fd = frames.get(f)
        if fd is None:
            continue
        prev, nxt, pos = fd
        try:
            store.write_linkage(f, np.asarray(prev), np.asarray(nxt), np.asarray(pos), name=linkage_name)
        except Exception:
            pass

    return {
        "n_trajectories_before": len(trajs),
        "n_merges": n_merges,
        "n_splits": n_splits,
        "n_trajectories_after_estimate": len(trajs) - n_merges + n_splits,
    }
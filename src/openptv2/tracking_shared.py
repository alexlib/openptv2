"""Shared-observation prototype for trackcorr, at linkage-file level.

trackcorr's kernel (``track_kernels_corr._trackcorr_particle_fast``) is
compiled per-particle greedy code -- the plan is to prototype the lesson in
pure Python on its linkage files first and move it into the kernel (behind a
flag) only if it wins:

1. :func:`mark_shared_observations` -- after the forward pass, find frames
   where one 3D point serves two tracks (detector undercount = occlusion)
   and mark it shared instead of letting one track die.
2. Downstream, shared points update position but never velocity (each
   track's speed comes only from its own points); gap bridging
   (``tracking_postprocess.relink_trajectory_gaps``) then stitches with
   honest speeds.

Linkage convention (mirrors ``tracking_postprocess``):
    frames[k] = (prev_k, next_k, xyz_k) with prev/next int arrays
    (PREV_NONE/NEXT_NONE = unlinked) and xyz_k (N,3) float positions.
Shared marks: dict mapping (frame, idx) -> list of track-end ids sharing it.
A "track end" here is identified by (frame, idx) of its last point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PREV_NONE = -1
NEXT_NONE = -2


def _velocity(frames, k, idx, shared, back_steps=1):
    """Velocity at (k, idx) from own points only: walk back over prev links,
    skipping shared marks. Returns None when fewer than two independent
    points exist."""
    pts = [(k, idx)]
    ck, ci = k, idx
    while True:
        prev, _, _ = frames[ck]
        pi = int(prev[ci])
        if pi < 0:
            break
        ck -= 1
        if ck not in frames:
            break
        ci = pi
        pts.append((ck, ci))
        if len(pts) >= back_steps + 2:
            break
    indep = [(fk, fi) for (fk, fi) in pts if (fk, fi) not in shared]
    if len(indep) < 2:
        return None
    (f1, i1), (f0, i0) = indep[-1], indep[-2]
    dt = max(f1 - f0, 1)
    _, _, xyz1 = frames[f1]
    _, _, xyz0 = frames[f0]
    return (xyz1[i1] - xyz0[i0]) / dt


def mark_shared_observations(frames, first, last, tol, max_share=2):
    """Find occlusions: a frame-(k+1) detection predicted well by >= 2
    distinct frame-k tracks that have a velocity (prev link), where at most
    one of them actually claimed it.

    Returns shared: dict (frame, idx) -> list of (frame_k, idx_k) sharers.
    Does not modify linkages; assembly/relink consume the marks.
    """
    shared: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for k in range(first, last):
        if k not in frames or (k + 1) not in frames:
            continue
        prev_k, next_k, xyz_k = frames[k]
        _, _, xyz_n = frames[k + 1]
        # predictors: frame-k particles with a prev link (have velocity)
        for j in range(len(xyz_n)):
            claimants = []
            for i in range(len(xyz_k)):
                pi = int(prev_k[i])
                if pi < 0 or k - 1 not in frames:
                    continue
                _, _, xyz_p = frames[k - 1]
                if pi >= len(xyz_p):
                    continue
                v = xyz_k[i] - xyz_p[pi]
                if np.linalg.norm(xyz_n[j] - (xyz_k[i] + v)) < tol:
                    claimants.append(
                        (i, float(np.linalg.norm(xyz_n[j] - (xyz_k[i] + v))))
                    )
            # undercount signature: >= 2 predictors, <= 1 actual claim.
            # actual claims: frame-k particles with next_k[i] == j
            actual = [i for i in range(len(xyz_k)) if int(next_k[i]) == j]
            if len(claimants) >= 2 and len(actual) <= 1:
                key = (k + 1, j)
                shared.setdefault(key, [])
                for i, _ in sorted(claimants, key=lambda t: t[1]):
                    if (k, i) not in shared[key]:
                        shared[key].append((k, i))
                shared[key] = shared[key][: max_share + 1]
    return shared


def assemble_with_shared(frames, first, last, shared):
    """Assemble tracks from linkage arrays, emitting a shared point into
    every sharing track (position shared, histories stay distinct because
    each track keeps its own prev chain).

    Returns list of dicts: {frames: [...], pos: (L,3)}.
    """
    # forward chains from every unclaimed start; shared points entered
    # once per sharing track.
    tracks = []
    visited: set[Any] = set()  # (frame, idx, owner-key) to allow shared re-entry
    # 1. ordinary chains from particles with no prev link
    for k in range(first, last + 1):
        if k not in frames:
            continue
        prev_k, next_k, xyz_k = frames[k]
        for i in range(len(xyz_k)):
            if int(prev_k[i]) >= 0:
                continue
            chain = []
            ck, ci = k, i
            while True:
                chain.append((ck, ci))
                if ck not in frames:
                    break
                _, nxt, _ = frames[ck]
                ni = int(nxt[ci]) if ci < len(nxt) else NEXT_NONE
                if ni < 0:
                    break
                ck += 1
                if ck not in frames:
                    break
                ci = ni
            tracks.append(chain)
            for node in chain:
                visited.add((node, "main"))
    # 2. sharing tracks: re-walk from each sharer end through the shared
    # point, then STOP. The shared point is observed jointly, but what
    # follows belongs to whoever links there next -- riding the winner's
    # tail would graft the wrong identity onto the sharer (a switch).
    # Continuation past separation is gap-relink's job, not assembly's.
    for (fk, fj), sharers in shared.items():
        for sk, si in sharers:
            # walk the sharer's own history up to (sk, si)
            hist = []
            ck, ci = sk, si
            while True:
                hist.append((ck, ci))
                if ck not in frames:
                    break
                pv, _, _ = frames[ck]
                pi = int(pv[ci]) if ci < len(pv) else PREV_NONE
                if pi < 0:
                    break
                ck -= 1
                if ck not in frames:
                    break
                ci = pi
            hist.reverse()
            # then the shared point itself -- and stop. Whatever continues
            # past separation belongs to whoever links there next; following
            # the winner's tail would graft the wrong identity (a switch).
            full = hist + [(fk, fj)] if (fk, fj) not in hist else list(hist)
            key = ("shared", sk, si, fk, fj)
            if key in visited:
                continue
            visited.add(key)
            # skip if an identical main chain already covers it
            if any(all(n in c for n in full) for c in tracks if len(c) >= len(full)):
                continue
            tracks.append(full)
    out = []
    for chain in tracks:
        fr, ps = [], []
        for ck, ci in chain:
            if ck not in frames:
                continue
            _, _, xyz = frames[ck]
            if ci >= len(xyz):
                continue
            fr.append(ck)
            ps.append(xyz[ci])
        if len(fr) >= 1:
            out.append({"frames": fr, "pos": np.array(ps)})
    return out

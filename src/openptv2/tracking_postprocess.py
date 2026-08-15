"""Disk-level tracking post-processes that improve trajectory quality without
touching the (validated) Cython tracking kernels.

All functions operate on the linkage files written by the tracker:
    ``{linkage_base}.{frame}``  lines: ``prev next x y z``   (0-based neighbour
    indices; ``prev == -1`` / ``next == -2`` mean "no link").

They are pure Python + NumPy, safe to run after ``full_forward`` /
``full_backward``, and each returns a small stats dict so the caller can measure
the effect.
"""

from __future__ import annotations

import os

import numpy as np

from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE

__all__ = [
    "read_linkage",
    "write_linkage",
    "count_links",
    "enforce_reciprocity",
    "seed_cold_start",
    "relink_trajectory_gaps",
]


from pathlib import Path


def _path(base: str, frame: int) -> str:
    return f"{base}.{frame}"


def read_linkage(linkage_base: str, frame: int, store=None):
    """Return (prev, next, xyz) arrays for a frame, or None if the file/store
    entry is absent or empty. ``prev``/``next`` are int32; ``xyz`` is (n, 3)
    float64.

    ``store``: an ``openptv2.storage.RunStore``, or None. When given, reads
    through it (its own frame-key convention) instead of guessing a
    ``run.zarr`` path with a hand-built, differently-padded key -- the two
    conventions collided silently before this (see
    docs/plans/2026-08-14-storage-formats-as-built.md).
    """
    base_path = Path(linkage_base)
    if store is not None:
        name = base_path.name
        if store.has_linkage(frame, name):
            prev, nxt, xyz = store.read_linkage(frame, name)
            return prev, nxt, xyz

    p = _path(linkage_base, frame)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        data = np.loadtxt(p, skiprows=1, ndmin=2)
    except Exception:
        return None
    if data.size == 0:
        return None
    prev = data[:, 0].astype(np.int32)
    nxt = data[:, 1].astype(np.int32)
    xyz = np.ascontiguousarray(data[:, 2:5], dtype=np.float64)
    return prev, nxt, xyz


def write_linkage(linkage_base: str, frame: int, prev, nxt, xyz, store=None) -> None:
    """Write a frame's linkage to the unified RunStore when ``store`` is
    given, otherwise to ASCII (store-backed runs no longer write ASCII --
    see docs/plans/2026-08-15-zarr-only-transition-plan.md and
    ``write_path_frame``'s identical store-vs-ASCII split). ``linkage_base``
    may be a store-only namespace (e.g. warmup's ``"warmup/cycle1"`` scratch
    group) with no real on-disk directory, so falling through to ASCII when
    a store is given would raise FileNotFoundError."""
    base_path = Path(linkage_base)
    if store is not None:
        store.write_linkage(frame, prev, nxt, xyz, name=base_path.name)
        return

    p = _path(linkage_base, frame)
    n = len(prev)
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(
                f"{int(prev[i]):4d} {int(nxt[i]):4d} "
                f"{xyz[i, 0]:10.3f} {xyz[i, 1]:10.3f} {xyz[i, 2]:10.3f}\n"
            )


def count_links(linkage_base: str, first: int, last: int, store=None) -> int:
    """Total forward links across the sequence (particles with next >= 0)."""
    total = 0
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k, store=store)
        if r is None:
            continue
        total += int((r[1] >= 0).sum())
    return total


def enforce_reciprocity(linkage_base: str, first: int, last: int, store=None):
    """Forward-backward consistency guard: keep only bidirectional links.

    A link between frame-k particle ``i`` and frame-(k+1) particle ``j`` is kept
    only if ``next_k[i] == j`` AND ``prev_{k+1}[j] == i``. One-sided links (which
    arise when the forward and backward passes disagree on dense/noisy data) are
    severed on both ends. On clean data where the tracker already produces
    reciprocal links this is a no-op; it is a precision guard for the hard cases.

    Returns a stats dict with the number of severed forward/backward links.
    """
    frames = {}
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k, store=store)
        if r is not None:
            frames[k] = r

    severed_next = 0
    severed_prev = 0
    dirty = set()

    for k in range(first, last):
        if k not in frames or (k + 1) not in frames:
            continue
        _p0, n0, _x0 = frames[k]
        p1, _n1, _x1 = frames[k + 1]
        for i in range(len(n0)):
            j = n0[i]
            if j < 0:
                continue
            if not (0 <= j < len(p1) and p1[j] == i):
                n0[i] = NEXT_NONE  # forward link not reciprocated
                severed_next += 1
                dirty.add(k)
        for j in range(len(p1)):
            i = p1[j]
            if i < 0:
                continue
            if not (0 <= i < len(n0) and n0[i] == j):
                p1[j] = PREV_NONE  # backward link not reciprocated
                severed_prev += 1
                dirty.add(k + 1)

    for k in dirty:
        prev, nxt, xyz = frames[k]
        write_linkage(linkage_base, k, prev, nxt, xyz, store=store)

    return {"severed_next": severed_next, "severed_prev": severed_prev}


def seed_cold_start(
    linkage_base: str,
    first: int,
    last: int,
    dv_max: float,
    accept_frac: float = 0.5,
    store=None,
):
    """Velocity-seeded recovery of the cold-start (first) transition.

    The first forward step has no velocity history, so it links far fewer
    particles than later steps (position-only search). Here we reuse the
    velocity that later frames *did* establish: for every frame-(first+1)
    particle ``j`` that is linked forward to frame first+2 (so its velocity is
    known) but is NOT yet linked back to frame first, we predict where it came
    from (``X_j - v_j``) and attach it to the nearest unlinked frame-first
    particle, provided the match is unambiguous and within tolerance.

    Only creates *bidirectional* links (sets both next and prev), so the result
    stays reciprocal. Returns a stats dict.
    """
    r0 = read_linkage(linkage_base, first, store=store)
    r1 = read_linkage(linkage_base, first + 1, store=store)
    r2 = read_linkage(linkage_base, first + 2, store=store)
    if r0 is None or r1 is None or r2 is None:
        return {"added": 0, "reason": "missing frames"}

    prev0, next0, xyz0 = r0
    prev1, next1, xyz1 = r1
    _prev2, _next2, xyz2 = r2

    # Tolerance: a fraction of the per-axis velocity gate. Tight by design so we
    # only accept confident recoveries.
    tol = accept_frac * float(dv_max)
    tol2 = tol * tol

    # Unlinked frame-first particles are the only valid targets.
    free0 = np.where(next0 < 0)[0]
    if len(free0) == 0:
        return {"added": 0, "reason": "no free source particles"}
    free_xyz = xyz0[free0]

    added = 0
    for j in range(len(prev1)):
        if prev1[j] >= 0:
            continue  # already linked back to frame first
        m = next1[j]
        if m < 0 or m >= len(xyz2):
            continue  # no forward velocity available
        v = xyz2[m] - xyz1[j]
        pred = xyz1[j] - v
        d2 = np.sum((free_xyz - pred) ** 2, axis=1)
        order = np.argsort(d2)
        best = order[0]
        if d2[best] > tol2:
            continue
        # Unambiguous: the runner-up must be clearly worse (2x distance).
        if len(order) > 1 and d2[order[1]] < 4.0 * d2[best]:
            continue
        i = int(free0[best])
        if next0[i] >= 0:
            continue  # taken by an earlier acceptance this pass
        next0[i] = j
        prev1[j] = i
        added += 1

    if added:
        write_linkage(linkage_base, first, prev0, next0, xyz0, store=store)
        write_linkage(linkage_base, first + 1, prev1, next1, xyz1, store=store)
    return {"added": added, "candidates": int((prev1 < 0).sum()), "tol": tol}


def relink_trajectory_gaps(
    linkage_base: str,
    first: int,
    last: int,
    max_gap: int = 2,
    max_velocity_err: float = 5.0,
    store=None,
) -> dict[str, int]:
    """
    Multi-pass post-processing gap relinking across linkage files.

    Identifies terminated tracks at frame k and unlinked track starts at frame k+gap+1,
    extrapolating particle positions using constant velocity to recover occluded
    particles over missing-frame gaps.

    A bridged gap is filled with one interpolated placeholder particle per
    skipped frame (constant-velocity interpolation between the two real
    endpoints), linked consecutively frame-to-frame -- NOT a single `next`
    pointer skipping straight from frame k to frame k+gap+1. Every other
    reader of prev/next (enforce_reciprocity, trajectory walkers) assumes
    next[k][i] always indexes into frame k+1; a cross-frame-skip pointer
    silently violates that and is actively incompatible with
    enforce_reciprocity specifically: it compares next_k against frame k+1
    (not k+gap+1), so it never recognizes the bridge as reciprocal and
    severs it right back out -- observed directly (286 bridged, 286 severed,
    net zero change) benchmarking Fast 3D with postprocess enabled.

    Returns:
        dict with count of recovered gaps (bridge chains, not individual
        hops): {'bridged_gaps': N}
    """
    bridged = 0
    frames = {}
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k, store=store)
        if r is not None:
            frames[k] = r

    dirty = set()

    def _frame(m):
        if m not in frames:
            frames[m] = (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.float64),
            )
        return frames[m]

    for gap in range(1, max_gap + 1):
        for k in range(first, last - gap):
            if k not in frames or (k + gap + 1) not in frames:
                continue

            prev_k, next_k, xyz_k = frames[k]
            prev_target, next_target, xyz_target = frames[k + gap + 1]

            # Unlinked ends at frame k that have a valid incoming link (known velocity)
            ends_k = np.where((next_k < 0) & (prev_k >= 0))[0]
            # Unlinked starts at frame k+gap+1 that have a valid outgoing link
            starts_target = np.where((prev_target < 0) & (next_target >= 0))[0]

            if len(ends_k) == 0 or len(starts_target) == 0:
                continue

            for end_idx in ends_k:
                p_prev_idx = prev_k[end_idx]
                if p_prev_idx < 0 or (k - 1) not in frames:
                    continue
                _, _, xyz_prev = frames[k - 1]
                if p_prev_idx >= len(xyz_prev):
                    continue

                v_est = xyz_k[end_idx] - xyz_prev[p_prev_idx]
                pred_pos = xyz_k[end_idx] + v_est * (gap + 1)

                cand_xyz = xyz_target[starts_target]
                dists = np.linalg.norm(cand_xyz - pred_pos, axis=1)
                best_cand = np.argmin(dists)

                if dists[best_cand] > max_velocity_err:
                    continue
                target_idx = starts_target[best_cand]

                # Chain: end_idx (frame k) -> placeholder(s) in k+1..k+gap -> target_idx (frame k+gap+1)
                start_xyz = xyz_k[end_idx]
                end_xyz = xyz_target[target_idx]
                prev_link_frame, prev_link_idx = k, int(end_idx)
                for step, m in enumerate(range(k + 1, k + gap + 1), start=1):
                    m_prev, m_next, m_xyz = _frame(m)
                    frac = step / (gap + 1)
                    placeholder_pos = start_xyz + (end_xyz - start_xyz) * frac
                    new_idx = len(m_next)
                    m_prev = np.append(m_prev, PREV_NONE).astype(np.int32)
                    m_next = np.append(m_next, NEXT_NONE).astype(np.int32)
                    m_xyz = np.vstack([m_xyz, placeholder_pos])
                    frames[m] = (m_prev, m_next, m_xyz)
                    dirty.add(m)

                    # Link the previous hop in the chain forward to this placeholder.
                    pf_prev, pf_next, pf_xyz = frames[prev_link_frame]
                    pf_next = pf_next.copy()
                    pf_next[prev_link_idx] = new_idx
                    frames[prev_link_frame] = (pf_prev, pf_next, pf_xyz)
                    dirty.add(prev_link_frame)

                    m_prev[new_idx] = prev_link_idx
                    prev_link_frame, prev_link_idx = m, new_idx

                # Final hop: last placeholder (or end_idx itself if gap's loop never ran,
                # i.e. unreachable here since gap >= 1) -> target_idx in frame k+gap+1.
                pf_prev, pf_next, pf_xyz = frames[prev_link_frame]
                pf_next = pf_next.copy()
                pf_next[prev_link_idx] = target_idx
                frames[prev_link_frame] = (pf_prev, pf_next, pf_xyz)
                dirty.add(prev_link_frame)

                prev_target = prev_target.copy()
                prev_target[target_idx] = prev_link_idx
                frames[k + gap + 1] = (prev_target, next_target, xyz_target)
                dirty.add(k + gap + 1)

                bridged += 1

    for m in dirty:
        prev_m, next_m, xyz_m = frames[m]
        write_linkage(linkage_base, m, prev_m, next_m, xyz_m, store=store)

    return {"bridged_gaps": bridged}

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
from pathlib import Path

import numpy as np

from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE

__all__ = [
    "read_linkage",
    "write_linkage",
    "count_links",
    "enforce_reciprocity",
    "seed_cold_start",
    "relink_trajectory_gaps",
    "MAX_LINK_STEP",
    "link_step",
    "back_link_step",
]

#: Largest frame step a forward/backward link may span. ``relink_trajectory_gaps``
#: bridges a gap by pointing ``next_k[i]`` straight into frame ``k + gap + 1``
#: (no fabricated measurement at the skipped frames -- those points would end up
#: in the Lagrangian velocity/acceleration statistics that are this project's
#: actual output). The linkage format cannot say *which* frame an index points
#: into, so consumers recover the step by looking for the reciprocal pointer;
#: this caps that search. Matches the default ``max_gap=2`` (=> steps 1..3).
MAX_LINK_STEP = 3


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


def link_step(prev_of, k: int, i: int, j: int, max_step: int = MAX_LINK_STEP) -> int:
    """Frame step spanned by the forward link ``next_k[i] == j``.

    Returns the smallest ``s`` in ``1..max_step`` with ``prev[k+s][j] == i``,
    or 0 when no frame reciprocates (a one-sided link). Smallest-first, so a
    genuine step-1 link always beats a coincidental match further out.

    ``prev_of(frame)`` returns that frame's ``prev`` array, or None if absent.
    """
    for s in range(1, max_step + 1):
        p = prev_of(k + s)
        if p is not None and 0 <= j < len(p) and p[j] == i:
            return s
    return 0


def back_link_step(
    next_of, k: int, j: int, i: int, max_step: int = MAX_LINK_STEP
) -> int:
    """Mirror of :func:`link_step` for the backward link ``prev_k[j] == i``:
    smallest ``s`` with ``next[k-s][i] == j``, else 0."""
    for s in range(1, max_step + 1):
        n = next_of(k - s)
        if n is not None and 0 <= i < len(n) and n[i] == j:
            return s
    return 0


def enforce_reciprocity(
    linkage_base: str,
    first: int,
    last: int,
    store=None,
    max_step: int = MAX_LINK_STEP,
):
    """Forward-backward consistency guard: keep only bidirectional links.

    A link between frame-k particle ``i`` and particle ``j`` some ``s`` frames
    later is kept only if ``next_k[i] == j`` AND ``prev_{k+s}[j] == i``.
    One-sided links (which arise when the forward and backward passes disagree
    on dense/noisy data) are severed on both ends. On clean data where the
    tracker already produces reciprocal links this is a no-op; it is a
    precision guard for the hard cases.

    ``s`` is searched over ``1..max_step`` rather than fixed at 1, so the
    cross-frame links ``relink_trajectory_gaps`` writes over a bridged gap are
    recognised as reciprocal instead of being severed straight back out.

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

    def prev_of(m):
        return frames[m][0] if m in frames else None

    def next_of(m):
        return frames[m][1] if m in frames else None

    for k in range(first, last + 1):
        if k not in frames:
            continue
        p0, n0, _x0 = frames[k]
        # Links at the sequence edges point outside it and are left alone
        # (as they always were: the old loop ran k in [first, last)).
        if k < last:
            for i in range(len(n0)):
                j = n0[i]
                if j < 0:
                    continue
                if not link_step(prev_of, k, i, j, max_step):
                    n0[i] = NEXT_NONE  # forward link not reciprocated
                    severed_next += 1
                    dirty.add(k)
        if k > first:
            for j in range(len(p0)):
                i = p0[j]
                if i < 0:
                    continue
                if not back_link_step(next_of, k, j, i, max_step):
                    p0[j] = PREV_NONE  # backward link not reciprocated
                    severed_prev += 1
                    dirty.add(k)

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
    max_accel_err: float = 5.0,
    store=None,
) -> dict[str, int]:
    """
    Multi-pass post-processing gap relinking across linkage files.

    Identifies terminated tracks at frame k and unlinked track starts at frame k+gap+1,
    extrapolating particle positions using constant velocity to recover occluded
    particles over missing-frame gaps.

    ``max_accel_err`` is an **acceleration** tolerance (mm/frame^2), not a
    velocity one: the candidate is compared against a position that has
    already been velocity-extrapolated, so what is left is the acceleration
    residual. The accepted distance is ``0.5 * max_accel_err * (gap+1)**2 / 2``
    -- it grows with the gap, because a longer extrapolation accumulates more
    of it. Pass ``dacc``.

    Passing ``dvxmax`` here (as every caller used to) is a live hazard: warmup
    suggests ``dvxmax ~ 52 mm`` on the JHU data against ~9 mm particle
    spacing, which would accept 52 mm bridges. It only looked sane on the
    synthetic set, where ``dvxmax == dacc == 6``.

    The 0.5 is measured, not assumed -- ``dacc`` gates the *maximum*
    acceleration, so the typical residual is well inside it. Bridges scored
    against ground-truth identity on ``test_data/synthetic_turbulent``
    (``dacc = 6``, so gap-1 tolerance = ``2 * max_accel_err``):

    ======  =============  =======  =========  ==================
    accel   tol at gap 1   bridges  % correct  true gaps recovered
    ======  =============  =======  =========  ==================
    1.5     3.0            137      92.0       26.5
    2.0     4.0            219      92.7       42.6
    3.0     6.0            309      92.6       60.1
    4.0     8.0            330      90.9       63.0
    6.0     12.0           347      85.6       62.4
    ======  =============  =======  =========  ==================

    Correctness holds until the gap-1 tolerance passes ~6 mm and recovery
    stops improving past it, which ``0.5 * dacc * 4 / 2 = dacc`` lands on.

    A bridged gap is a *single* cross-frame link: ``next_k[i]`` points straight
    into frame ``k+gap+1`` and ``prev_{k+gap+1}[j]`` points back, with nothing
    written at the skipped frames. No measurement is fabricated where the
    particle was never observed, and the representation matches ground truth
    (a gap is a link of step > 1). Consumers recover the step via
    :func:`link_step` / :func:`back_link_step`.

    Returns:
        dict with count of recovered links: {'bridged_gaps': N}
    """
    bridged = 0
    frames = {}
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k, store=store)
        if r is not None:
            frames[k] = r

    dirty = set()

    def next_of(m):
        return frames[m][1] if m in frames else None

    for gap in range(1, max_gap + 1):
        # Kinematic residual over dt = gap+1 frames is a*dt^2/2. ``dacc`` gates
        # the *maximum* acceleration, so the effective residual scale is about
        # half of it -- hence the 0.5 (measured, see the docstring).
        tol = 0.5 * max_accel_err * (gap + 1) ** 2 / 2.0
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
                if p_prev_idx < 0:
                    continue
                # The incoming link may itself be a bridge (step > 1) from an
                # earlier pass, so find the frame it really comes from and
                # divide the displacement by that step.
                back = back_link_step(next_of, k, int(end_idx), int(p_prev_idx))
                if back == 0:
                    back = 1  # non-reciprocal: take it at face value, as before
                if (k - back) not in frames:
                    continue
                _, _, xyz_prev = frames[k - back]
                if p_prev_idx >= len(xyz_prev):
                    continue

                v_est = (xyz_k[end_idx] - xyz_prev[p_prev_idx]) / back
                pred_pos = xyz_k[end_idx] + v_est * (gap + 1)

                cand_xyz = xyz_target[starts_target]
                dists = np.linalg.norm(cand_xyz - pred_pos, axis=1)
                best_cand = np.argmin(dists)

                if dists[best_cand] > tol:
                    continue
                target_idx = starts_target[best_cand]
                if prev_target[target_idx] >= 0:
                    continue  # claimed by an earlier end this pass

                next_k[end_idx] = target_idx
                prev_target[target_idx] = end_idx
                bridged += 1
                dirty.add(k)
                dirty.add(k + gap + 1)

    for m in dirty:
        prev_m, next_m, xyz_m = frames[m]
        write_linkage(linkage_base, m, prev_m, next_m, xyz_m, store=store)

    return {"bridged_gaps": bridged}

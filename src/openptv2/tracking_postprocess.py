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

from pathlib import Path
from typing import Any

import numpy as np

from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE

__all__ = [
    "read_linkage",
    "write_linkage",
    "count_links",
    "enforce_reciprocity",
    "confirm_links",
    "confirm_link_tuples",
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


def read_linkage(
    linkage_base: str, frame: int, store: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (prev, next, xyz) arrays for a frame from ``RunStore``, or None
    if ``store`` is None or the entry is absent/empty."""
    if store is None:
        return None
    name = Path(linkage_base).name
    if store.has_linkage(frame, name):
        prev, nxt, xyz = store.read_linkage(frame, name)
        return prev, nxt, xyz
    return None


def write_linkage(
    linkage_base: str,
    frame: int,
    prev: np.ndarray,
    nxt: np.ndarray,
    xyz: np.ndarray,
    store: Any = None,
) -> None:
    """Write a frame's linkage to the unified RunStore."""
    if store is None:
        raise ValueError("write_linkage requires a RunStore (ASCII linkage is removed)")
    base_path = Path(linkage_base)
    store.write_linkage(frame, prev, nxt, xyz, name=base_path.name)


def write_linkage_frames(
    linkage_base: str,
    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dirty: Any,
    store: Any = None,
) -> None:
    """Write the ``dirty`` frames of ``frames`` ({k: (prev, next, xyz)}). With a
    store this is ONE committed block write instead of a write per frame."""
    ks = sorted(dirty)
    if store is not None:
        if ks:
            store.write_linkage_many(
                ks,
                [(frames[k][0], frames[k][1]) for k in ks],
                [frames[k][2] for k in ks],
                name=Path(linkage_base).name,
            )
        return
    for k in ks:
        prev, nxt, xyz = frames[k]
        write_linkage(linkage_base, k, prev, nxt, xyz, store=None)


def count_links(linkage_base: str, first: int, last: int, store: Any = None) -> int:
    """Total forward links across the sequence (particles with next >= 0)."""
    total = 0
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k, store=store)
        if r is None:
            continue
        total += int((r[1] >= 0).sum())
    return total


def link_step(
    prev_of: Any, k: int, i: int, j: int, max_step: int = MAX_LINK_STEP
) -> int:
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
    next_of: Any, k: int, j: int, i: int, max_step: int = MAX_LINK_STEP
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
    store: Any = None,
    max_step: int = MAX_LINK_STEP,
) -> dict[str, int]:
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

    write_linkage_frames(linkage_base, frames, dirty, store=store)

    return {"severed_next": severed_next, "severed_prev": severed_prev}


def median_kink(
    links: list[tuple[int, int, int, int]], frame_particles: list[np.ndarray]
) -> float:
    """Median velocity kink |x2 - 2 x1 + x0| over all pairs of consecutive links.

    On tracks the kink is dominated by position noise (real accelerations are tiny at
    high frame rates), so it measures the noise level of the data itself; it scaled
    linearly with the position jitter on the benchmark (median 0.043 / 0.076 / 0.109 /
    0.140 mm for centroid noise 0.04 / 0.08 / 0.12 / 0.16 px) and was 0.06 on real
    CompleteTest data."""
    nxt: dict[tuple[int, int], tuple[int, int]] = {}
    for t0, r0, t1, r1 in links:
        if t1 - t0 == 1:
            nxt[(t0, r0)] = (t1, r1)
    P = [np.asarray(p, dtype=np.float64) for p in frame_particles]
    kinks = []
    for (t0, r0), (t1, r1) in nxt.items():
        n2 = nxt.get((t1, r1))
        if n2 is None or r0 >= len(P[t0]) or r1 >= len(P[t1]) or n2[1] >= len(P[n2[0]]):
            continue
        kinks.append(np.linalg.norm(P[n2[0]][n2[1]] - 2.0 * P[t1][r1] + P[t0][r0]))
    return float(np.median(kinks)) if kinks else float("nan")


def kink_statistics(
    links: list[tuple[int, int, int, int]],
    frame_particles: list[np.ndarray],
    radius: float = 5.0,
    n_frames: int = 5,
) -> dict[str, float]:
    """Data statistics that decide whether the noise-scaled confirmation tolerance is
    safe: ``median_kink``, ``median_step`` (both mm/frame), their ratio, and
    ``neighbours``: the median number of other 3D points within ``radius`` mm of a point
    (density), averaged over ``n_frames`` frames spread over the sequence."""
    from scipy.spatial import cKDTree

    mk = median_kink(links, frame_particles)
    nxt = {(t0, r0): (t1, r1) for t0, r0, t1, r1 in links if t1 - t0 == 1}
    P = [np.asarray(p, dtype=np.float64) for p in frame_particles]
    steps = [
        np.linalg.norm(P[t1][r1] - P[t0][r0])
        for (t0, r0), (t1, r1) in nxt.items()
        if r0 < len(P[t0]) and r1 < len(P[t1])
    ]
    ms = float(np.median(steps)) if steps else float("nan")
    idx = np.unique(np.linspace(0, len(P) - 1, min(n_frames, len(P))).astype(int))
    nb = []
    for i in idx:
        if len(P[i]) > 10:
            tree = cKDTree(P[i])
            counts = [len(x) - 1 for x in tree.query_ball_point(P[i][::6], radius)]
            nb.append(np.median(counts))
    return {
        "median_kink": mk,
        "median_step": ms,
        "ratio": mk / ms if ms and np.isfinite(ms) else float("nan"),
        "neighbours": float(np.mean(nb)) if nb else float("nan"),
    }


def confirm_link_tuples(
    links: list[tuple[int, int, int, int]],
    frame_particles: list[np.ndarray],
    tol: float | None,
    ends: bool,
    point_sigma: list[np.ndarray] | None = None,
) -> tuple[list[tuple[int, int, int, int]], set]:
    """Two-hop confirmation post-pass over consecutive links.

    A consecutive link (t0,r0)->(t1,r1) survives iff a consecutive onward
    link from (t1,r1) continues within velocity kink ``tol`` (mm/frame).
    Links into the last frame cannot be judged and are kept; gap links
    pass through untouched (gap logic already decided them). With ``ends``,
    dead-end consecutive links (no onward link, not last frame) are also
    severed -- a dying track grabbing a stranger. Severing fragments
    chains; rejoining them is gap-relink/repair's job, not assembly's.

    With ``point_sigma`` (per frame, the position uncertainty of each point in mm) the tolerance of a link widens
    with its points' noise: the kink ``|p0 - 2 p1 + p2|`` has variance ``s0^2 + 4 s1^2 + s2^2``, so the tolerance is
    ``tol * max(1, sqrt((s0^2 + 4 s1^2 + s2^2) / (6 s_ref^2)))`` with ``s_ref`` the median sigma. Never stricter
    than ``tol``: points known to be noisy (corrected merged blobs, births) are not cut for their own noise.

    Returns (kept_links, severed) with severed a set of node pairs.
    """
    last_t = len(frame_particles) - 1
    sig_ref = None
    if point_sigma is not None:
        allv = np.concatenate([np.asarray(v, float) for v in point_sigma if v is not None and len(v)])
        allv = allv[np.isfinite(allv) & (allv > 0)]
        sig_ref = float(np.median(allv)) if len(allv) else None

    def _sig(t, r):
        if sig_ref is None or point_sigma[t] is None or r >= len(point_sigma[t]):
            return sig_ref if sig_ref is not None else 0.0
        v = float(point_sigma[t][r])
        return v if np.isfinite(v) and v > 0 else sig_ref
    fwd: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for t0, r0, t1, r1 in links:
        if t1 - t0 == 1:
            fwd.setdefault((t0, r0), []).append((t1, r1))
    severed = set()
    if tol is not None:
        P = [np.asarray(p, dtype=np.float64) for p in frame_particles]
        for t0, r0, t1, r1 in links:
            if t1 - t0 != 1 or t1 >= last_t:
                continue
            if r0 >= len(P[t0]) or r1 >= len(P[t1]):
                continue
            onward = fwd.get((t1, r1), [])
            if not onward:
                if ends:
                    severed.add(((t0, r0), (t1, r1)))
                continue
            v = P[t1][r1] - P[t0][r0]
            keep = False
            for t2, r2 in onward:
                if t2 - t1 != 1 or r2 >= len(P[t2]):
                    continue
                kink = float(np.linalg.norm((P[t2][r2] - P[t1][r1]) - v))
                tol_eff = tol
                if sig_ref:
                    s2 = _sig(t0, r0) ** 2 + 4 * _sig(t1, r1) ** 2 + _sig(t2, r2) ** 2
                    tol_eff = tol * max(1.0, float(np.sqrt(s2 / (6 * sig_ref**2))))
                if kink <= tol_eff:
                    keep = True
                    break
            if not keep:
                severed.add(((t0, r0), (t1, r1)))
    kept = [
        L
        for L in links
        if not (L[2] - L[0] == 1 and ((L[0], L[1]), (L[2], L[3])) in severed)
    ]
    return kept, severed


def confirm_links(
    linkage_base: str,
    first: int,
    last: int,
    tol: float,
    ends: bool = False,
    store: Any = None,
) -> dict[str, int]:
    """Two-hop link confirmation over any tracker's stored linkage.

    Severs a consecutive link (frame k particle i -> frame k+1 particle j)
    unless the link that leaves j continues within a velocity kink of ``tol``
    (position units per frame, the scale of ``dacc``). A wrong link (to a
    stranger) almost never gets a second, kinematically consistent link
    after it. With ``ends`` also severs links into dead ends. This is the
    rule ``two_phase`` applies internally, which the benchmark identified as
    the reason it beats every other tracker (see ``scripts/synth_bench.py``).

    Run BEFORE ``relink_trajectory_gaps`` (which writes multi-frame links this
    pass would misread as consecutive) and rejoin the fragments afterwards.
    Returns ``{"links", "severed"}``.
    """
    frames = {}
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k, store=store)
        if r is not None:
            frames[k] = r
    ks = sorted(frames)
    if not ks:
        return {"links": 0, "severed": 0}
    t_of = {k: t for t, k in enumerate(ks)}
    particles = [frames[k][2] for k in ks]
    links = []
    for k in ks:
        nxt = frames[k][1]
        if k + 1 not in frames:
            continue
        n_next = len(frames[k + 1][0])
        for i in np.flatnonzero((nxt >= 0) & (nxt < n_next)):
            links.append((t_of[k], int(i), t_of[k + 1], int(nxt[i])))
    _, severed = confirm_link_tuples(links, particles, tol, ends)
    dirty = set()
    for (t0, r0), (t1, r1) in severed:
        frames[ks[t0]][1][r0] = NEXT_NONE
        frames[ks[t1]][0][r1] = PREV_NONE
        dirty.update((ks[t0], ks[t1]))
    write_linkage_frames(linkage_base, frames, dirty, store=store)
    return {"links": len(links), "severed": len(severed)}


def seed_cold_start(
    linkage_base: str,
    first: int,
    last: int,
    dv_max: float,
    accept_frac: float = 0.5,
    store: Any = None,
) -> dict[str, Any]:
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
    store: Any = None,
    *,
    require_bidir_vel: bool = False,
    uniqueness_ratio: float = 0.0,
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

    def prev_of(m):
        return frames[m][0] if m in frames else None

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
                if uniqueness_ratio > 1.0 and len(dists) >= 2:
                    order = np.argsort(dists)
                    best_cand = int(order[0])
                    if dists[best_cand] > tol:
                        continue
                    if dists[int(order[1])] < uniqueness_ratio * max(float(dists[best_cand]), 1e-9):
                        continue
                else:
                    best_cand = int(np.argmin(dists))
                    if dists[best_cand] > tol:
                        continue
                target_idx = starts_target[best_cand]
                if prev_target[target_idx] >= 0:
                    continue  # claimed by an earlier end this pass

                if require_bidir_vel and next_target[target_idx] >= 0:
                    t_next_idx = int(next_target[target_idx])
                    fwd = link_step(prev_of, k + gap + 1, int(target_idx), t_next_idx)
                    if fwd == 0:
                        fwd = 1
                    f_next = k + gap + 1 + fwd
                    if f_next in frames and t_next_idx < len(frames[f_next][2]):
                        v_out = (frames[f_next][2][t_next_idx] - xyz_target[target_idx]) / fwd
                        if float(np.linalg.norm(v_out - v_est)) > max_accel_err * (gap + 1):
                            continue

                next_k[end_idx] = target_idx
                prev_target[target_idx] = end_idx
                bridged += 1
                dirty.add(k)
                dirty.add(k + gap + 1)

    write_linkage_frames(linkage_base, frames, dirty, store=store)

    return {"bridged_gaps": bridged}


def prune_kinky_links(
    linkage_base: str,
    first: int,
    last: int,
    *,
    max_accel_err: float,
    max_angle_deg: float = 0.0,
    store: Any = None,
) -> dict[str, int]:
    """Break reciprocal links at kinematic velocity-jump kinks."""
    if max_accel_err <= 0.0:
        return {"broken_kinks": 0}

    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k, store=store)
        if r is not None:
            prv, nxt, xyz = r
            frames[k] = (prv.copy(), nxt.copy(), xyz)

    def next_of(m: int):
        return frames[m][1] if m in frames else None

    def prev_of(m: int):
        return frames[m][0] if m in frames else None

    cos_thr = float(np.cos(np.radians(max_angle_deg))) if max_angle_deg > 0.0 else 0.0
    dirty: set[int] = set()
    broken = 0

    for f1 in range(first + 1, last):
        if f1 not in frames:
            continue
        prv1, nxt1, xyz1 = frames[f1]
        for i1 in range(len(xyz1)):
            i0 = int(prv1[i1])
            i2 = int(nxt1[i1])
            if i0 < 0 or i2 < 0:
                continue
            dt1 = back_link_step(next_of, f1, i1, i0)
            dt2 = link_step(prev_of, f1, i1, i2)
            if dt1 == 0 or dt2 == 0:
                continue
            f0 = f1 - dt1
            f2 = f1 + dt2
            if f0 not in frames or f2 not in frames:
                continue
            if i0 >= len(frames[f0][2]) or i2 >= len(frames[f2][2]):
                continue

            v1 = (xyz1[i1] - frames[f0][2][i0]) / float(dt1)
            v2 = (frames[f2][2][i2] - xyz1[i1]) / float(dt2)
            acc = float(np.linalg.norm(v2 - v1))
            if acc <= max_accel_err:
                continue

            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            ang_bad = True
            if n1 > 1e-9 and n2 > 1e-9 and max_angle_deg > 0.0:
                cos_a = float(np.dot(v1, v2) / (n1 * n2))
                ang_bad = cos_a < cos_thr

            if not (ang_bad or acc > 1.5 * max_accel_err):
                continue

            # Break the trailing edge by default, or the leading edge if f0 is
            # a 1-step track head jumping onto an established continuation.
            if frames[f0][0][i0] < 0 and frames[f2][1][i2] >= 0 and n1 > 1.5 * max(n2, 1e-9):
                prv1[i1] = PREV_NONE
                frames[f0][1][i0] = NEXT_NONE
                dirty.update((f0, f1))
            else:
                nxt1[i1] = NEXT_NONE
                frames[f2][0][i2] = PREV_NONE
                dirty.update((f1, f2))
            broken += 1

    write_linkage_frames(linkage_base, frames, dirty, store=store)
    return {"broken_kinks": broken}

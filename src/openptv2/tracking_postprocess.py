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
]


def _path(base: str, frame: int) -> str:
    return f"{base}.{frame}"


def read_linkage(linkage_base: str, frame: int):
    """Return (prev, next, xyz) arrays for a frame, or None if the file is absent
    or empty. ``prev``/``next`` are int32; ``xyz`` is (n, 3) float64."""
    p = _path(linkage_base, frame)
    if not os.path.exists(p):
        return None
    data = np.loadtxt(p, skiprows=1, ndmin=2)
    if data.size == 0:
        return None
    prev = data[:, 0].astype(np.int32)
    nxt = data[:, 1].astype(np.int32)
    xyz = np.ascontiguousarray(data[:, 2:5], dtype=np.float64)
    return prev, nxt, xyz


def write_linkage(linkage_base: str, frame: int, prev, nxt, xyz) -> None:
    """Rewrite a linkage file, preserving the tracker's column format."""
    p = _path(linkage_base, frame)
    n = len(prev)
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(
                f"{int(prev[i]):4d} {int(nxt[i]):4d} "
                f"{xyz[i, 0]:10.3f} {xyz[i, 1]:10.3f} {xyz[i, 2]:10.3f}\n"
            )


def count_links(linkage_base: str, first: int, last: int) -> int:
    """Total forward links across the sequence (particles with next >= 0)."""
    total = 0
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k)
        if r is None:
            continue
        total += int((r[1] >= 0).sum())
    return total


def enforce_reciprocity(linkage_base: str, first: int, last: int):
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
        r = read_linkage(linkage_base, k)
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
        write_linkage(linkage_base, k, prev, nxt, xyz)

    return {"severed_next": severed_next, "severed_prev": severed_prev}


def seed_cold_start(
    linkage_base: str,
    first: int,
    last: int,
    dv_max: float,
    accept_frac: float = 0.5,
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
    r0 = read_linkage(linkage_base, first)
    r1 = read_linkage(linkage_base, first + 1)
    r2 = read_linkage(linkage_base, first + 2)
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
        write_linkage(linkage_base, first, prev0, next0, xyz0)
        write_linkage(linkage_base, first + 1, prev1, next1, xyz1)
    return {"added": added, "candidates": int((prev1 < 0).sum()), "tol": tol}

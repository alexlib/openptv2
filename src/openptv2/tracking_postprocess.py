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


def read_linkage(linkage_base: str, frame: int):
    """Return (prev, next, xyz) arrays for a frame, or None if the file/Zarr entry is absent
    or empty. ``prev``/``next`` are int32; ``xyz`` is (n, 3) float64."""
    base_path = Path(linkage_base)
    zarr_dir = base_path.parent / "run.zarr"
    if zarr_dir.exists():
        try:
            import zarr
            try:
                root = zarr.open_group(str(zarr_dir), mode="r")
            except Exception:
                root = zarr.open_group(str(zarr_dir), mode="a")
            key = f"linkage/{base_path.name}/frame_{frame:05d}"
            if key in root:
                fg = root[key]
                prev = np.asarray(fg["prev"], dtype=np.int32)
                nxt = np.asarray(fg["next"], dtype=np.int32)
                xyz = np.ascontiguousarray(fg["pos"], dtype=np.float64)
                return prev, nxt, xyz
        except Exception as e:
            print(f"[read_linkage] Zarr read warning frame {frame}: {e}")

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


def write_linkage(linkage_base: str, frame: int, prev, nxt, xyz) -> None:
    """Rewrite a linkage file or Zarr store entry, preserving the tracker's column format."""
    base_path = Path(linkage_base)
    zarr_dir = base_path.parent / "run.zarr"
    if zarr_dir.exists():
        try:
            import zarr
            root = zarr.open_group(str(zarr_dir), mode="r+")
            key = f"linkage/{base_path.name}/frame_{frame:05d}"
            if key in root:
                fg = root[key]
                fg.create_array("prev", data=np.asarray(prev, dtype=np.int32), overwrite=True)
                fg.create_array("next", data=np.asarray(nxt, dtype=np.int32), overwrite=True)
                fg.create_array("pos", data=np.asarray(xyz, dtype=np.float64), overwrite=True)
                return
        except Exception:
            pass

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


def relink_trajectory_gaps(
    linkage_base: str,
    first: int,
    last: int,
    max_gap: int = 2,
    max_velocity_err: float = 5.0,
) -> dict[str, int]:
    """
    Multi-pass post-processing gap relinking across linkage files.

    Identifies terminated tracks at frame k and unlinked track starts at frame k+gap+1,
    extrapolating particle positions using constant velocity to recover occluded
    particles over missing-frame gaps.

    Returns:
        dict with count of recovered links: {'bridged_gaps': N}
    """
    bridged = 0
    frames = {}
    for k in range(first, last + 1):
        r = read_linkage(linkage_base, k)
        if r is not None:
            frames[k] = r

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

                if dists[best_cand] <= max_velocity_err:
                    target_idx = starts_target[best_cand]
                    next_k[end_idx] = target_idx
                    prev_target[target_idx] = end_idx
                    bridged += 1

                    write_linkage(linkage_base, k, prev_k, next_k, xyz_k)
                    write_linkage(linkage_base, k + gap + 1, prev_target, next_target, xyz_target)

    return {"bridged_gaps": bridged}


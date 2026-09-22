"""Cascade tracker: trackcorr first, two-phase+vel only on leftovers.

Phase 1: full trackcorr forward (+optional backward) -- high precision base.
Phase 2: productized TwoPhaseTracker (+velocity, re-projected costs) runs
on the same positions; its links are merged ADDITIVELY: a two-phase link
(a -> b) is accepted iff trackcorr left BOTH ends free (next[a] == -1 and
prev[b] == -1). Existing links are never stolen or rewritten -- the merge
can only extend heads/tails and bridge gaps, exactly where maneuvers hide.

Outputs go to <res>_cascade/ (the trackcorr res/ is untouched).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def merge_links(corr_prev, corr_next, tp_links, nrows):
    """Additive merge of two-phase links into trackcorr linkage.

    corr_prev/next: lists per frame offset of prev/next arrays (row space).
    tp_links: iterable of (t0, r0, t1, r1) in the same frame-offset/row space.
    nrows: rows per frame offset.
    Returns (merged_prev, merged_next, n_added).
    """
    mprev = [np.array(p, dtype=np.int32) for p in corr_prev]
    mnxt = [np.array(p, dtype=np.int32) for p in corr_next]
    added = []
    for t0, r0, t1, r1 in tp_links:
        if not (0 <= t0 < len(mnxt) and 0 <= t1 < len(mprev)):
            continue
        if not (0 <= r0 < nrows[t0] and 0 <= r1 < nrows[t1]):
            continue
        # Free tail: trackcorr marks "no outgoing link" as -1 (never
        # claimed) or -2 (lost contest / dropped); both are linkable.
        # Free head: prev == -1. Anything else is taken -- never steal.
        if mnxt[t0][r0] < 0 and mprev[t1][r1] < 0:
            mnxt[t0][r0] = r1
            mprev[t1][r1] = r0
            added.append((t0, r0, t1, r1))
    return mprev, mnxt, len(added), added


def write_ptv_is(out: Path, frames, mprev, mnxt, positions):
    """Write merged linkage + positions in ptv_is format."""
    out.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(frames):
        n = len(positions[k])
        with open(out / f"ptv_is.{f}", "w") as fh:
            fh.write(f"{n}\n")
            for i in range(n):
                x, y, z = positions[k][i]
                fh.write(f"{int(mprev[k][i]):4d} {int(mnxt[k][i]):4d} "
                         f"{x:10.3f} {y:10.3f} {z:10.3f}\n")

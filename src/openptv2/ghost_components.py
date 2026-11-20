"""Connected-component decomposition of candidate 3D points by shared 2D targets.

Ghosts arise when several candidate 3D points compete for the same 2D image
targets. Decomposing the per-frame candidate set into connected components
(where an edge means "two points share at least one 2D target") bounds the
subsequent uniqueness solver (Hungarian / ILP) to small, independent sub-
problems instead of the whole cloud.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _union_find(n: int) -> tuple[list[int], callable]:
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return parent, union, find


def conflict_components(
    pos: np.ndarray,
    cam_ids: np.ndarray,
) -> list[list[int]]:
    """Return connected components of candidate 3D points sharing 2D targets.

    ``pos`` is (N, 3) (unused beyond length, kept for API symmetry).
    ``cam_ids`` is (N, n_cams) of 2D target ids (>=0) or -1.

    Two points are connected if they share at least one 2D target in any camera.
    Returns a list of components, each a list of row indices into ``pos``.
    """
    n = len(pos)
    if n == 0:
        return []
    if cam_ids.size == 0:
        return [[i] for i in range(n)]

    n_cams = cam_ids.shape[1]
    parent, union, find = _union_find(n)

    # Group point indices by (camera, target_id) to union sharers.
    for cam in range(n_cams):
        col = cam_ids[:, cam]
        valid = col >= 0
        if not np.any(valid):
            continue
        # map target_id -> list of point indices
        ids = col[valid]
        idxs = np.nonzero(valid)[0]
        order = np.argsort(ids, kind="stable")
        s_ids = ids[order]
        s_idx = idxs[order]
        # union consecutive equal ids
        for k in range(1, len(s_ids)):
            if s_ids[k] == s_ids[k - 1]:
                union(int(s_idx[k]), int(s_idx[k - 1]))

    # Collect components.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def component_sizes(components: Sequence[Sequence[int]]) -> list[int]:
    return [len(c) for c in components]


def classify_component(size: int) -> str:
    """Return 'trivial' (1), 'bipartite' (2), 'ilp' (3..max), 'large' (overflow)."""
    if size <= 1:
        return "trivial"
    if size == 2:
        return "bipartite"
    return "ilp"

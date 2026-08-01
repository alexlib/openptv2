"""Radius-limited min-cost assignment shared by the MyPTV tracking plugins.

Tracking links predictions to candidates by minimising total displacement,
subject to a per-prediction search radius. The direct formulation is a dense
Hungarian assignment over an (n_pred, n_cand) cost matrix where out-of-radius
pairs carry a big-M sentinel. That is exact but O(n^3), and in PTV the matrix
is overwhelmingly sentinel: a particle only ever competes with the handful of
candidates inside its search ball.

`match_within_radius` solves the same problem by building only the in-radius
edges (KD-tree) and running the Hungarian separately on each connected
component of that graph. Cross-component pairs are out of radius by
construction, so they can never appear in a valid link, and the optimum
decomposes exactly over components.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# Below this cost-matrix size the dense Hungarian beats building the KD-tree
# and decomposing the graph. Measured crossover is around 400x400; see
# docs/developer_guide/custom_tracking_plugins.md.
DENSE_CUTOFF = 150_000


def _match_dense(pred, cands, radius):
    """Big-M Hungarian over the full cost matrix. Exact, O(n^3)."""
    dists = cdist(pred, cands)
    in_radius = dists <= radius[:, None]
    if not in_radius.any():
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

    # One sentinel outweighs every real distance combined, so the assignment
    # maximises the number of links before it minimises their total length.
    sentinel = dists[in_radius].sum() + 1.0
    rows, cols = linear_sum_assignment(np.where(in_radius, dists, sentinel))
    keep = in_radius[rows, cols]
    return rows[keep], cols[keep]


def match_within_radius(pred, cands, radius):
    """Match predictions to candidates, minimising total distance.

    Parameters
    ----------
    pred : ndarray (n_pred, ndim)
        Predicted positions, one per active track.
    cands : ndarray (n_cand, ndim)
        Candidate positions in the new frame.
    radius : float or ndarray (n_pred,)
        Search radius, scalar or per-prediction.

    Returns
    -------
    rows, cols : ndarray of int
        Indices of matched (prediction, candidate) pairs. Every returned pair
        is within its radius; unmatched predictions and candidates are simply
        absent.

    Notes
    -----
    Equivalent to a dense big-M Hungarian over the full cost matrix, except
    that exact ties may be broken differently (the components are solved
    independently). Distances are floats, so ties need coincident points.
    """
    n_pred = len(pred)
    n_cand = len(cands)
    empty = (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
    if n_pred == 0 or n_cand == 0:
        return empty

    radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n_pred,))

    if n_pred * n_cand <= DENSE_CUTOFF:
        return _match_dense(pred, cands, radius)

    # In-radius edges only. query_ball_point takes a per-point radius array.
    neighbours = cKDTree(cands).query_ball_point(pred, r=radius)
    counts = np.fromiter((len(n) for n in neighbours), dtype=np.intp, count=n_pred)
    n_edges = int(counts.sum())
    if n_edges == 0:
        return empty

    rows = np.repeat(np.arange(n_pred, dtype=np.intp), counts)
    cols = np.fromiter((c for n in neighbours for c in n), dtype=np.intp, count=n_edges)
    dists = np.linalg.norm(pred[rows] - cands[cols], axis=1)

    # Connected components of the bipartite graph, predictions indexed
    # [0, n_pred) and candidates [n_pred, n_pred + n_cand).
    n_nodes = n_pred + n_cand
    graph = coo_matrix(
        (np.ones(n_edges, dtype=np.int8), (rows, cols + n_pred)),
        shape=(n_nodes, n_nodes),
    )
    n_comp, labels = connected_components(graph, directed=False)

    # A component holding exactly one edge is an unambiguous pair: the
    # Hungarian on a 1x1 matrix is the identity, so take those in bulk. This
    # is the overwhelming majority of components at PTV seeding densities.
    edge_comp = labels[rows]
    comp_edges = np.bincount(edge_comp, minlength=n_comp)
    trivial = comp_edges[edge_comp] == 1

    out_rows = [rows[trivial]]
    out_cols = [cols[trivial]]

    # Everything else: one small dense Hungarian per component. Group the
    # remaining edges by sorting on the component label once -- scanning the
    # full edge array per component would be quadratic in component count.
    rest = np.flatnonzero(~trivial)
    if len(rest):
        rest = rest[np.argsort(edge_comp[rest], kind="stable")]
        splits = np.flatnonzero(np.diff(edge_comp[rest])) + 1

        for group in np.split(rest, splits):
            c_rows = rows[group].tolist()
            c_cols = cols[group].tolist()
            c_dists = dists[group].tolist()

            # Components are small (a few particles), so plain dicts beat
            # np.unique, which pays a sort setup cost per call.
            uniq_r = sorted(set(c_rows))
            uniq_c = sorted(set(c_cols))
            r_local = {v: i for i, v in enumerate(uniq_r)}
            c_local = {v: i for i, v in enumerate(uniq_c)}

            # The dense formulation is lexicographic: maximise the number of
            # real links first, minimise total distance second. To reproduce
            # that, one sentinel must outweigh every real distance combined --
            # otherwise the Hungarian could drop a link to shorten the others.
            sentinel = sum(c_dists) + 1.0
            sub = np.full((len(uniq_r), len(uniq_c)), sentinel, dtype=np.float64)
            for rr, cc, dd in zip(c_rows, c_cols, c_dists):
                sub[r_local[rr], c_local[cc]] = dd

            r_ind, c_ind = linear_sum_assignment(sub)
            real = sub[r_ind, c_ind] < sentinel
            out_rows.append(np.asarray(uniq_r, dtype=np.intp)[r_ind[real]])
            out_cols.append(np.asarray(uniq_c, dtype=np.intp)[c_ind[real]])

    return np.concatenate(out_rows), np.concatenate(out_cols)

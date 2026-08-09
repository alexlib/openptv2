"""Stereo-3D tracking loop — position-space only, no camera projections."""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def _find_closest_in_3d(
    path_x_2: cython.double[:, ::1],
    np2: cython.int,
    pred_x: cython.double,
    pred_y: cython.double,
    pred_z: cython.double,
    dx: cython.double,
    dy: cython.double,
    dz: cython.double,
    max_cands: cython.int,
    cand_inds: cython.int[:],
    cand_dists: cython.double[:],
):
    """Find up to max_cands closest candidates by distance within a 3D box.

    Maintains a running top-N by distance, matching candsearch_in_pix logic.
    Writes into pre-allocated cand_inds/cand_dists arrays.
    Returns the number of candidates found.
    """
    s: cython.int
    k: cython.int
    slot: cython.int
    ddx: cython.double
    ddy: cython.double
    ddz: cython.double
    d: cython.double
    n_found = 0
    for s in range(max_cands):
        cand_inds[s] = -1
        cand_dists[s] = 1e20

    for k in range(np2):
        ddx = path_x_2[k, 0] - pred_x
        ddy = path_x_2[k, 1] - pred_y
        ddz = path_x_2[k, 2] - pred_z
        if abs(ddx) < dx and abs(ddy) < dy and abs(ddz) < dz:
            d = c_sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            for slot in range(max_cands):
                if d < cand_dists[slot]:
                    for s in range(max_cands - 1, slot, -1):
                        cand_inds[s] = cand_inds[s - 1]
                        cand_dists[s] = cand_dists[s - 1]
                    cand_inds[slot] = k
                    cand_dists[slot] = d
                    break

    for s in range(max_cands):
        if cand_inds[s] >= 0:
            n_found += 1
    return n_found


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def track3d_loop_fast(
    orig_parts: cython.int,
    # Frame 0 (prev) — read only
    path_x_0: cython.double[:, ::1],
    path_prev_0: cython.int[:],
    num_parts_0: cython.int,
    # Frame 1 (curr) — read/write
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    num_parts_1: cython.int,
    # Frame 2 (next) — read/write
    path_x_2: cython.double[:, ::1],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    num_parts_2: cython.int,
    # Tracking params
    dx: cython.double,
    dy: cython.double,
    dz: cython.double,
    max_cands: cython.int,
    dacc: cython.double = 0.0,
):
    """Full track3d loop (3 levels) — single compiled entry.

    Level 1: particles with previous links — predict from velocity (search box = dacc).
    Level 2: no prev link — average velocity from neighbors (search box = dacc).
    Level 3: no prev link, no neighbor info — use current position (search box = dx,dy,dz).

    Within each level, candidates are claimed in ascending cost order across
    ALL of that level's particles at once (one sort per level), not
    particle-by-particle in index order: otherwise particle 0 always wins a
    contested candidate over particle 500 regardless of which is the better
    match. The three levels still run as a strict cascade (level 2 only
    sees particles level 1 left unclaimed, etc.) — only the claim order
    *within* a level changed.

    Returns count1 (number of links established).
    """
    count1: cython.int
    np2: cython.int
    i: cython.int
    j: cython.int
    ci: cython.int
    prev_idx: cython.int
    pred_x: cython.double
    pred_y: cython.double
    pred_z: cython.double
    n_cands: cython.int
    k: cython.int
    d0: cython.double
    d1: cython.double
    d2: cython.double
    acc: cython.double
    vel_x: cython.double
    vel_y: cython.double
    vel_z: cython.double
    nvel: cython.int
    cx: cython.double
    cy: cython.double
    cz: cython.double
    pj: cython.int
    inv_nvel: cython.double
    ax: cython.double
    ay: cython.double
    az: cython.double
    n_edges: cython.int
    max_edges: cython.int
    oi: cython.int
    e: cython.int
    order: cython.int[:]

    count1 = 0
    np2 = num_parts_2
    ax = dacc if dacc > 0.0 else dx
    ay = dacc if dacc > 0.0 else dy
    az = dacc if dacc > 0.0 else dz

    _cand_inds = np.empty(max_cands, dtype=np.int32)
    _cand_dists = np.empty(max_cands, dtype=np.float64)
    cand_inds: cython.int[:] = _cand_inds
    cand_dists: cython.double[:] = _cand_dists

    # One (cost, particle, candidate) edge buffer, reused per level: upper
    # bound is orig_parts candidate-generation calls x max_cands each.
    max_edges = orig_parts * max_cands
    _edge_cost = np.empty(max_edges, dtype=np.float64)
    _edge_i = np.empty(max_edges, dtype=np.int32)
    _edge_k = np.empty(max_edges, dtype=np.int32)
    edge_cost: cython.double[:] = _edge_cost
    edge_i: cython.int[:] = _edge_i
    edge_k: cython.int[:] = _edge_k

    # ===== Level 1: Particles with previous links =====
    n_edges = 0
    for i in range(orig_parts):
        if path_prev_1[i] < 0:
            continue
        prev_idx = path_prev_1[i]
        if prev_idx < 0 or prev_idx >= num_parts_0:
            continue
        path_next_1[i] = -1  # default; a claim below may overwrite this

        pred_x = 2.0 * path_x_1[i, 0] - path_x_0[prev_idx, 0]
        pred_y = 2.0 * path_x_1[i, 1] - path_x_0[prev_idx, 1]
        pred_z = 2.0 * path_x_1[i, 2] - path_x_0[prev_idx, 2]

        n_cands = _find_closest_in_3d(
            path_x_2, np2, pred_x, pred_y, pred_z, ax, ay, az,
            max_cands, cand_inds, cand_dists,
        )
        for ci in range(n_cands):
            k = cand_inds[ci]
            # Acceleration residual X(t+1) - 2X(t) + X(t-1); equivalently the
            # distance from the candidate to the constant-velocity prediction.
            d0 = path_x_2[k, 0] - 2.0 * path_x_1[i, 0] + path_x_0[prev_idx, 0]
            d1 = path_x_2[k, 1] - 2.0 * path_x_1[i, 1] + path_x_0[prev_idx, 1]
            d2 = path_x_2[k, 2] - 2.0 * path_x_1[i, 2] + path_x_0[prev_idx, 2]
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            edge_cost[n_edges] = acc
            edge_i[n_edges] = i
            edge_k[n_edges] = k
            n_edges += 1

    if n_edges > 0:
        _order = np.argsort(_edge_cost[:n_edges]).astype(np.int32)
        order = _order
        for oi in range(n_edges):
            e = order[oi]
            i = edge_i[e]
            k = edge_k[e]
            if path_next_1[i] < 0 and path_prev_2[k] < 0:
                path_next_1[i] = k
                path_prev_2[k] = i
                count1 += 1

    # ===== Level 2: No previous link, neighbor velocity =====
    n_edges = 0
    for i in range(orig_parts):
        if path_prev_1[i] >= 0 or path_next_1[i] >= 0:
            continue

        vel_x = 0.0
        vel_y = 0.0
        vel_z = 0.0
        nvel = 0
        cx = path_x_1[i, 0]
        cy = path_x_1[i, 1]
        cz = path_x_1[i, 2]

        for j in range(orig_parts):
            if j == i:
                continue
            if (
                abs(path_x_1[j, 0] - cx) < dx
                and abs(path_x_1[j, 1] - cy) < dy
                and abs(path_x_1[j, 2] - cz) < dz
                and path_prev_1[j] >= 0
            ):
                pj = path_prev_1[j]
                vel_x += path_x_1[j, 0] - path_x_0[pj, 0]
                vel_y += path_x_1[j, 1] - path_x_0[pj, 1]
                vel_z += path_x_1[j, 2] - path_x_0[pj, 2]
                nvel += 1

        if nvel == 0:
            continue

        path_next_1[i] = -1  # default; a claim below may overwrite this
        inv_nvel = 1.0 / nvel
        pred_x = cx + vel_x * inv_nvel
        pred_y = cy + vel_y * inv_nvel
        pred_z = cz + vel_z * inv_nvel

        n_cands = _find_closest_in_3d(
            path_x_2, np2, pred_x, pred_y, pred_z, ax, ay, az,
            max_cands, cand_inds, cand_dists,
        )
        for ci in range(n_cands):
            k = cand_inds[ci]
            # pred already carries the neighbour-averaged velocity, so the
            # residual to it is the acceleration.
            d0 = path_x_2[k, 0] - pred_x
            d1 = path_x_2[k, 1] - pred_y
            d2 = path_x_2[k, 2] - pred_z
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            edge_cost[n_edges] = acc
            edge_i[n_edges] = i
            edge_k[n_edges] = k
            n_edges += 1

    if n_edges > 0:
        _order = np.argsort(_edge_cost[:n_edges]).astype(np.int32)
        order = _order
        for oi in range(n_edges):
            e = order[oi]
            i = edge_i[e]
            k = edge_k[e]
            if path_next_1[i] < 0 and path_prev_2[k] < 0:
                path_next_1[i] = k
                path_prev_2[k] = i
                count1 += 1

    # ===== Level 3: No previous link, no neighbors — static prediction =====
    n_edges = 0
    for i in range(orig_parts):
        if path_prev_1[i] >= 0 or path_next_1[i] >= 0:
            continue
        path_next_1[i] = -1  # default; a claim below may overwrite this

        pred_x = path_x_1[i, 0]
        pred_y = path_x_1[i, 1]
        pred_z = path_x_1[i, 2]

        n_cands = _find_closest_in_3d(
            path_x_2, np2, pred_x, pred_y, pred_z, dx, dy, dz,
            max_cands, cand_inds, cand_dists,
        )
        for ci in range(n_cands):
            k = cand_inds[ci]
            # No velocity estimate: pred == curr, so this is plain distance.
            d0 = path_x_2[k, 0] - pred_x
            d1 = path_x_2[k, 1] - pred_y
            d2 = path_x_2[k, 2] - pred_z
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            edge_cost[n_edges] = acc
            edge_i[n_edges] = i
            edge_k[n_edges] = k
            n_edges += 1

    if n_edges > 0:
        _order = np.argsort(_edge_cost[:n_edges]).astype(np.int32)
        order = _order
        for oi in range(n_edges):
            e = order[oi]
            i = edge_i[e]
            k = edge_k[e]
            if path_next_1[i] < 0 and path_prev_2[k] < 0:
                path_next_1[i] = k
                path_prev_2[k] = i
                count1 += 1

    return count1


# ============================================================
# Batch kernels for standalone API acceleration
# ============================================================

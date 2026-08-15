"""Stereo-3D tracking loop — position-space only, no camera projections."""

import cython
import numpy as np

from openptv2.algorithms.constants import NEXT_NONE

# Level 1 candidate cost = acceleration_residual + dist_weight *
# |candidate - current_position| (see track3d_loop_fast's dist_weight
# parameter). It exists purely to break near-ties in acceleration residual
# toward the physically smaller jump -- it must stay low enough that a
# genuine accelerating/curving continuation (large acceleration gap vs a
# near-but-wrong decoy) still wins on acceleration alone; see
# test_track3d_level1_ranks_by_forward_acceleration_not_decoy_behind, whose
# fixture requires this weight to stay under 3.0.
#
# LEVEL1_DIST_WEIGHT is the static fallback for callers that don't supply
# their own dist_weight. track3d_loop (the driver used by the default
# tracker) instead estimates a per-dataset value from the first two frames
# via track3d.estimate_level1_dist_weight: the right balance depends on how
# large true motion is relative to particle spacing, which is measurable
# before any acceleration assumption is made and varies a lot between
# datasets (a slow, densely-seeded flow wants a much higher weight than a
# fast, sparse one -- a single fixed constant can't serve both).
LEVEL1_DIST_WEIGHT = 1.0

# Cost offset applied to a 4BE candidate that has no real particle near its
# two-frames-ahead estimate (see track4be_loop_fast). It only has to exceed
# the largest possible supported cost, which is bounded by the search box
# diagonal, so any round number far above millimetre scale keeps the two
# tiers strictly ordered.
UNSUPPORTED_PENALTY = 1e6

if cython.compiled:
    from cython.cimports.libc.math import floor as c_floor, sqrt as c_sqrt
else:
    from math import floor as c_floor, sqrt as c_sqrt


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
def _find_closest_in_3d_grid(
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
    grid_head: cython.int[:],
    grid_next: cython.int[:],
    min_x: cython.double,
    min_y: cython.double,
    min_z: cython.double,
    cell_x: cython.double,
    cell_y: cython.double,
    cell_z: cython.double,
    nx: cython.int,
    ny: cython.int,
    nz: cython.int,
) -> cython.int:
    """Find up to max_cands closest candidates using 3D spatial grid cells.

    Declared @cython.cfunc: this is module-internal and sits on the hottest
    path in the file -- 4BE calls it once per particle for candidates plus
    once per candidate for n+2 support, so a Python calling convention here
    would dominate its runtime.
    """
    s: cython.int
    k: cython.int
    slot: cython.int
    ddx: cython.double
    ddy: cython.double
    ddz: cython.double
    d: cython.double
    n_found: cython.int = 0

    if np2 < 32:
        return _find_closest_in_3d(
            path_x_2, np2, pred_x, pred_y, pred_z, dx, dy, dz, max_cands, cand_inds, cand_dists
        )

    for s in range(max_cands):
        cand_inds[s] = -1
        cand_dists[s] = 1e20

    # Determine cell range covering [pred_x - dx, pred_x + dx], etc.
    c_x_min = int(c_floor((pred_x - dx - min_x) / cell_x))
    c_x_max = int(c_floor((pred_x + dx - min_x) / cell_x))
    c_y_min = int(c_floor((pred_y - dy - min_y) / cell_y))
    c_y_max = int(c_floor((pred_y + dy - min_y) / cell_y))
    c_z_min = int(c_floor((pred_z - dz - min_z) / cell_z))
    c_z_max = int(c_floor((pred_z + dz - min_z) / cell_z))

    if c_x_min < 0: c_x_min = 0
    if c_x_max >= nx: c_x_max = nx - 1
    if c_y_min < 0: c_y_min = 0
    if c_y_max >= ny: c_y_max = ny - 1
    if c_z_min < 0: c_z_min = 0
    if c_z_max >= nz: c_z_max = nz - 1

    cx: cython.int
    cy: cython.int
    cz: cython.int
    cell_idx: cython.int

    for cx in range(c_x_min, c_x_max + 1):
        for cy in range(c_y_min, c_y_max + 1):
            for cz in range(c_z_min, c_z_max + 1):
                cell_idx = (cx * ny + cy) * nz + cz
                k = grid_head[cell_idx]
                while k >= 0:
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
                    k = grid_next[k]

    for s in range(max_cands):
        if cand_inds[s] >= 0:
            n_found += 1
    return n_found


@cython.ccall
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
) -> cython.int:
    """Find up to max_cands closest candidates by distance within a 3D box.

    @cython.ccall rather than @cython.cfunc: track_kernels_tracking re-exports
    this one, so it has to stay importable from Python while still being
    C-callable from _find_closest_in_3d_grid's small-frame fallback.
    """
    s: cython.int
    k: cython.int
    slot: cython.int
    ddx: cython.double
    ddy: cython.double
    ddz: cython.double
    d: cython.double
    n_found: cython.int = 0
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
    dist_weight: cython.double = LEVEL1_DIST_WEIGHT,
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

    # ===== Construct 3D Spatial Grid for Frame 2 Candidates =====
    min_x: cython.double = 1e20
    min_y: cython.double = 1e20
    min_z: cython.double = 1e20
    max_x: cython.double = -1e20
    max_y: cython.double = -1e20
    max_z: cython.double = -1e20

    if np2 > 0:
        for k in range(np2):
            if path_x_2[k, 0] < min_x: min_x = path_x_2[k, 0]
            if path_x_2[k, 0] > max_x: max_x = path_x_2[k, 0]
            if path_x_2[k, 1] < min_y: min_y = path_x_2[k, 1]
            if path_x_2[k, 1] > max_y: max_y = path_x_2[k, 1]
            if path_x_2[k, 2] < min_z: min_z = path_x_2[k, 2]
            if path_x_2[k, 2] > max_z: max_z = path_x_2[k, 2]
    else:
        min_x = 0.0; max_x = 1.0
        min_y = 0.0; max_y = 1.0
        min_z = 0.0; max_z = 1.0

    cell_x: cython.double = ax if ax > 0.5 else dx
    cell_y: cython.double = ay if ay > 0.5 else dy
    cell_z: cython.double = az if az > 0.5 else dz
    if cell_x < 0.1: cell_x = 1.0
    if cell_y < 0.1: cell_y = 1.0
    if cell_z < 0.1: cell_z = 1.0

    nx: cython.int = int(c_floor((max_x - min_x) / cell_x)) + 1
    ny: cython.int = int(c_floor((max_y - min_y) / cell_y)) + 1
    nz: cython.int = int(c_floor((max_z - min_z) / cell_z)) + 1
    grid_size: cython.int = nx * ny * nz

    _grid_head = np.full(grid_size, -1, dtype=np.int32)
    _grid_next = np.full(np2 if np2 > 0 else 1, -1, dtype=np.int32)
    grid_head: cython.int[:] = _grid_head
    grid_next: cython.int[:] = _grid_next

    cx_k: cython.int
    cy_k: cython.int
    cz_k: cython.int
    c_idx: cython.int

    for k in range(np2):
        cx_k = int(c_floor((path_x_2[k, 0] - min_x) / cell_x))
        cy_k = int(c_floor((path_x_2[k, 1] - min_y) / cell_y))
        cz_k = int(c_floor((path_x_2[k, 2] - min_z) / cell_z))
        if cx_k < 0: cx_k = 0
        if cx_k >= nx: cx_k = nx - 1
        if cy_k < 0: cy_k = 0
        if cy_k >= ny: cy_k = ny - 1
        if cz_k < 0: cz_k = 0
        if cz_k >= nz: cz_k = nz - 1
        c_idx = (cx_k * ny + cy_k) * nz + cz_k
        grid_next[k] = grid_head[c_idx]
        grid_head[c_idx] = k

    # ===== Level 1: Particles with previous links =====
    n_edges = 0
    for i in range(orig_parts):
        if path_prev_1[i] < 0:
            continue
        prev_idx = path_prev_1[i]
        if prev_idx < 0 or prev_idx >= num_parts_0:
            continue
        path_next_1[i] = NEXT_NONE  # default; a claim below may overwrite this

        pred_x = 2.0 * path_x_1[i, 0] - path_x_0[prev_idx, 0]
        pred_y = 2.0 * path_x_1[i, 1] - path_x_0[prev_idx, 1]
        pred_z = 2.0 * path_x_1[i, 2] - path_x_0[prev_idx, 2]

        n_cands = _find_closest_in_3d_grid(
            path_x_2, np2, pred_x, pred_y, pred_z, ax, ay, az,
            max_cands, cand_inds, cand_dists,
            grid_head, grid_next, min_x, min_y, min_z, cell_x, cell_y, cell_z, nx, ny, nz
        )
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = path_x_2[k, 0] - 2.0 * path_x_1[i, 0] + path_x_0[prev_idx, 0]
            d1 = path_x_2[k, 1] - 2.0 * path_x_1[i, 1] + path_x_0[prev_idx, 1]
            d2 = path_x_2[k, 2] - 2.0 * path_x_1[i, 2] + path_x_0[prev_idx, 2]
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            # Acceleration residual alone ranks by fit to the (possibly
            # noisy) constant-velocity extrapolation, with no penalty for
            # how far the candidate actually sits from the particle's last
            # known position -- so once the gate is wider than the true
            # particle spacing, a distant candidate that happens to align
            # with a noisy prediction outranks a much closer, physically
            # plausible one (observed: a farther candidate beat a closer
            # available one in ~51% of links on a loosely-gated dataset).
            # Add a raw-displacement term, weighted below the acceleration
            # term (LEVEL1_DIST_WEIGHT < 1) so the existing decoy-vs-true-
            # continuation ordering (large acceleration gap) still wins --
            # this only breaks near-ties in acceleration toward the smaller
            # jump.
            dc0 = path_x_2[k, 0] - path_x_1[i, 0]
            dc1 = path_x_2[k, 1] - path_x_1[i, 1]
            dc2 = path_x_2[k, 2] - path_x_1[i, 2]
            dist_from_curr = c_sqrt(dc0 * dc0 + dc1 * dc1 + dc2 * dc2)
            edge_cost[n_edges] = acc + dist_weight * dist_from_curr
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

        path_next_1[i] = NEXT_NONE  # default; a claim below may overwrite this
        inv_nvel = 1.0 / nvel
        pred_x = cx + vel_x * inv_nvel
        pred_y = cy + vel_y * inv_nvel
        pred_z = cz + vel_z * inv_nvel

        n_cands = _find_closest_in_3d_grid(
            path_x_2, np2, pred_x, pred_y, pred_z, ax, ay, az,
            max_cands, cand_inds, cand_dists,
            grid_head, grid_next, min_x, min_y, min_z, cell_x, cell_y, cell_z, nx, ny, nz
        )
        for ci in range(n_cands):
            k = cand_inds[ci]
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
        path_next_1[i] = NEXT_NONE  # default; a claim below may overwrite this

        pred_x = path_x_1[i, 0]
        pred_y = path_x_1[i, 1]
        pred_z = path_x_1[i, 2]

        n_cands = _find_closest_in_3d_grid(
            path_x_2, np2, pred_x, pred_y, pred_z, dx, dy, dz,
            max_cands, cand_inds, cand_dists,
            grid_head, grid_next, min_x, min_y, min_z, cell_x, cell_y, cell_z, nx, ny, nz
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


# ============================================================
# 4BE - Four-Frame Best Estimate (Ouellette, Xu & Bodenschatz,
# Exp. Fluids 40:301-313, 2006, eqs. 10, 12, 14)
# ============================================================
#
# Where 3MA (track3d_loop_fast above) scores a candidate by the
# acceleration it implies, 4BE scores it by how well it PREDICTS a real
# particle one further frame ahead:
#
#   search centre in n+1   x^n + v*dt              = 2*q1 - x0     (eq. 10)
#   estimate for n+2       x^n + v*2dt + a*(2dt)^2 = 2*q - x1      (eq. 12)
#   cost                   || x^{n+2}_k - estimate ||              (eq. 14)
#
# Eq. 12 collapses to 2*q - x1 because the paper defines acceleration as
# (q - 2*x1 + x0)/(2*dt^2) (their eq. 11), so the x0 terms cancel exactly.
# That is precisely why 4BE "makes no attempt to estimate the third time
# derivative", and why it beats the change-in-acceleration heuristics on
# noisy data. For reference, Willneff's trackcorr predictor
# 0.5*(5q - 4*x1 + x0) is the mean of this constant-velocity estimate and
# the full constant-acceleration one (3q - 3*x1 + x0) -- a half-damped
# version of the same idea.
#
# Conflicts are resolved by giving up, not by global assignment: the paper
# measured Munkres/Hungarian conflict-breaking as DEGRADING every heuristic
# it tested except nearest-neighbour, and recommends stopping every track
# involved in a conflict.


@cython.cfunc
def _build_grid3d(px, np_pts: cython.int, cell_x: cython.double,
                  cell_y: cython.double, cell_z: cython.double):
    """Uniform-cell spatial hash over one frame's 3D positions.

    Returns (grid_head, grid_next, min_x, min_y, min_z, nx, ny, nz), the
    arguments _find_closest_in_3d_grid expects. Called twice per tracking
    step (frame n+1 for candidates, n+2 for their support), so the tuple
    return is not on any hot path.
    """
    k: cython.int
    cx: cython.int
    cy: cython.int
    cz: cython.int
    c_idx: cython.int
    min_x: cython.double = 1e20
    min_y: cython.double = 1e20
    min_z: cython.double = 1e20
    max_x: cython.double = -1e20
    max_y: cython.double = -1e20
    max_z: cython.double = -1e20

    if np_pts > 0:
        for k in range(np_pts):
            if px[k, 0] < min_x:
                min_x = px[k, 0]
            if px[k, 0] > max_x:
                max_x = px[k, 0]
            if px[k, 1] < min_y:
                min_y = px[k, 1]
            if px[k, 1] > max_y:
                max_y = px[k, 1]
            if px[k, 2] < min_z:
                min_z = px[k, 2]
            if px[k, 2] > max_z:
                max_z = px[k, 2]
    else:
        min_x = 0.0
        max_x = 1.0
        min_y = 0.0
        max_y = 1.0
        min_z = 0.0
        max_z = 1.0

    if cell_x < 0.1:
        cell_x = 1.0
    if cell_y < 0.1:
        cell_y = 1.0
    if cell_z < 0.1:
        cell_z = 1.0

    nx: cython.int = int(c_floor((max_x - min_x) / cell_x)) + 1
    ny: cython.int = int(c_floor((max_y - min_y) / cell_y)) + 1
    nz: cython.int = int(c_floor((max_z - min_z) / cell_z)) + 1

    _grid_head = np.full(nx * ny * nz, -1, dtype=np.int32)
    _grid_next = np.full(np_pts if np_pts > 0 else 1, -1, dtype=np.int32)
    grid_head: cython.int[:] = _grid_head
    grid_next: cython.int[:] = _grid_next

    for k in range(np_pts):
        cx = int(c_floor((px[k, 0] - min_x) / cell_x))
        cy = int(c_floor((px[k, 1] - min_y) / cell_y))
        cz = int(c_floor((px[k, 2] - min_z) / cell_z))
        if cx < 0:
            cx = 0
        if cx >= nx:
            cx = nx - 1
        if cy < 0:
            cy = 0
        if cy >= ny:
            cy = ny - 1
        if cz < 0:
            cz = 0
        if cz >= nz:
            cz = nz - 1
        c_idx = (cx * ny + cy) * nz + cz
        grid_next[k] = grid_head[c_idx]
        grid_head[c_idx] = k

    return _grid_head, _grid_next, min_x, min_y, min_z, nx, ny, nz


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def track4be_loop_fast(
    orig_parts: cython.int,
    path_x_0: cython.double[:, ::1],
    path_prev_0: cython.int[:],
    num_parts_0: cython.int,
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    num_parts_1: cython.int,
    path_x_2: cython.double[:, ::1],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    num_parts_2: cython.int,
    path_x_3: cython.double[:, ::1],
    num_parts_3: cython.int,
    dx: cython.double,
    dy: cython.double,
    dz: cython.double,
    max_cands: cython.int,
    strict_support: cython.int = 0,
):
    """Four-frame best-estimate linking of frame n to frame n+1.

    Pure 3D: consumes only stereo-matched particle positions, never 2D
    targets or camera models. Seeded particles (those with a previous link)
    are scored by 4BE; unseeded ones fall back to nearest neighbour, which
    is what the paper specifies for joining the first two points of a track.

    When frame n+2 is unavailable (the tail of a sequence), seeded scoring
    degrades to the 3MA acceleration residual so the last steps still link.

    ``strict_support``: 1 reproduces the paper literally -- a candidate with
    no real particle near its n+2 estimate is discarded. 0 (the default)
    keeps such candidates as a penalised 3MA fallback, which recovers the
    yield lost to genuine detection gaps without disturbing 4BE's ordering
    among supported candidates.

    Returns the number of links established.
    """
    i: cython.int
    k: cython.int
    ci: cython.int
    prev_idx: cython.int
    n_cands: cython.int
    n_sup: cython.int
    count1: cython.int = 0
    np2: cython.int = num_parts_2
    np3: cython.int = num_parts_3
    have_f3: cython.int = 1 if num_parts_3 > 0 else 0

    pred_x: cython.double
    pred_y: cython.double
    pred_z: cython.double
    est_x: cython.double
    est_y: cython.double
    est_z: cython.double
    cost: cython.double
    best_cost: cython.double
    d0: cython.double
    d1: cython.double
    d2: cython.double
    best: cython.int

    _cand_inds = np.empty(max_cands, dtype=np.int32)
    _cand_dists = np.empty(max_cands, dtype=np.float64)
    cand_inds: cython.int[:] = _cand_inds
    cand_dists: cython.double[:] = _cand_dists

    _sup_inds = np.empty(1, dtype=np.int32)
    _sup_dists = np.empty(1, dtype=np.float64)
    sup_inds: cython.int[:] = _sup_inds
    sup_dists: cython.double[:] = _sup_dists

    _best_k = np.full(orig_parts if orig_parts > 0 else 1, -1, dtype=np.int32)
    best_k: cython.int[:] = _best_k
    _claims = np.zeros(np2 if np2 > 0 else 1, dtype=np.int32)
    claims: cython.int[:] = _claims

    g2 = _build_grid3d(path_x_2, np2, dx, dy, dz)
    grid2_head: cython.int[:] = g2[0]
    grid2_next: cython.int[:] = g2[1]
    g2_min_x: cython.double = g2[2]
    g2_min_y: cython.double = g2[3]
    g2_min_z: cython.double = g2[4]
    g2_nx: cython.int = g2[5]
    g2_ny: cython.int = g2[6]
    g2_nz: cython.int = g2[7]

    g3 = _build_grid3d(path_x_3, np3, dx, dy, dz)
    grid3_head: cython.int[:] = g3[0]
    grid3_next: cython.int[:] = g3[1]
    g3_min_x: cython.double = g3[2]
    g3_min_y: cython.double = g3[3]
    g3_min_z: cython.double = g3[4]
    g3_nx: cython.int = g3[5]
    g3_ny: cython.int = g3[6]
    g3_nz: cython.int = g3[7]

    for i in range(orig_parts):
        path_next_1[i] = NEXT_NONE
        best = -1
        best_cost = 1e20
        prev_idx = path_prev_1[i]

        if prev_idx >= 0 and prev_idx < num_parts_0:
            # eq. 10 -- constant-velocity search centre in frame n+1
            pred_x = 2.0 * path_x_1[i, 0] - path_x_0[prev_idx, 0]
            pred_y = 2.0 * path_x_1[i, 1] - path_x_0[prev_idx, 1]
            pred_z = 2.0 * path_x_1[i, 2] - path_x_0[prev_idx, 2]

            n_cands = _find_closest_in_3d_grid(
                path_x_2, np2, pred_x, pred_y, pred_z, dx, dy, dz,
                max_cands, cand_inds, cand_dists,
                grid2_head, grid2_next, g2_min_x, g2_min_y, g2_min_z,
                dx, dy, dz, g2_nx, g2_ny, g2_nz
            )

            for ci in range(n_cands):
                k = cand_inds[ci]
                if have_f3 == 1:
                    # eq. 12 -- estimated position two frames ahead
                    est_x = 2.0 * path_x_2[k, 0] - path_x_1[i, 0]
                    est_y = 2.0 * path_x_2[k, 1] - path_x_1[i, 1]
                    est_z = 2.0 * path_x_2[k, 2] - path_x_1[i, 2]
                    n_sup = _find_closest_in_3d_grid(
                        path_x_3, np3, est_x, est_y, est_z, dx, dy, dz,
                        1, sup_inds, sup_dists,
                        grid3_head, grid3_next, g3_min_x, g3_min_y, g3_min_z,
                        dx, dy, dz, g3_nx, g3_ny, g3_nz
                    )
                    if n_sup > 0:
                        cost = sup_dists[0]  # eq. 14
                    elif strict_support == 1:
                        # Nothing real near the estimate: strict 4BE treats
                        # the candidate as unsupported and rejects it.
                        continue
                    else:
                        # Unsupported two frames out. Rejecting outright is
                        # expensive on real data, where the particle is
                        # genuinely missing from n+2 far more often than one
                        # would like (7.6% of ground-truth links on the
                        # synthetic set are gaps of 2+ frames). Fall back to
                        # the 3MA acceleration residual, offset so that ANY
                        # supported candidate still outranks EVERY
                        # unsupported one -- 4BE's ordering is preserved
                        # wherever the evidence for it exists, and only the
                        # otherwise-dead candidates compete on 3MA.
                        d0 = path_x_2[k, 0] - 2.0 * path_x_1[i, 0] + path_x_0[prev_idx, 0]
                        d1 = path_x_2[k, 1] - 2.0 * path_x_1[i, 1] + path_x_0[prev_idx, 1]
                        d2 = path_x_2[k, 2] - 2.0 * path_x_1[i, 2] + path_x_0[prev_idx, 2]
                        cost = UNSUPPORTED_PENALTY + c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                else:
                    d0 = path_x_2[k, 0] - 2.0 * path_x_1[i, 0] + path_x_0[prev_idx, 0]
                    d1 = path_x_2[k, 1] - 2.0 * path_x_1[i, 1] + path_x_0[prev_idx, 1]
                    d2 = path_x_2[k, 2] - 2.0 * path_x_1[i, 2] + path_x_0[prev_idx, 2]
                    cost = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)

                if cost < best_cost:
                    best_cost = cost
                    best = k
        else:
            # First two points of a track are joined by nearest neighbour.
            n_cands = _find_closest_in_3d_grid(
                path_x_2, np2,
                path_x_1[i, 0], path_x_1[i, 1], path_x_1[i, 2], dx, dy, dz,
                1, sup_inds, sup_dists,
                grid2_head, grid2_next, g2_min_x, g2_min_y, g2_min_z,
                dx, dy, dz, g2_nx, g2_ny, g2_nz
            )
            if n_cands > 0:
                best = sup_inds[0]
                best_cost = sup_dists[0]

        if best >= 0 and path_prev_2[best] < 0:
            best_k[i] = best
            claims[best] += 1

    # Conflict handling: a frame n+1 particle claimed by more than one
    # particle in frame n links to none of them -- every track involved
    # stops here and a new track begins at n+1.
    for i in range(orig_parts):
        k = best_k[i]
        if k >= 0 and claims[k] == 1:
            path_next_1[i] = k
            path_prev_2[k] = i
            count1 += 1

    return count1

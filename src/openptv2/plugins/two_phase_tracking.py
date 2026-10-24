"""Two-phase tracking: 3D search + 2D leaf ranking.

Phase 1: 3D KD-tree finds candidate matches within a search radius.
Phase 2: 2D pixel distances become the cost matrix; Hungarian assignment
         within connected components picks the best match globally.

This exploits the tree-forest architecture: 3D positions are the "trunk"
(structural search), per-camera 2D leaf positions are the "signature"
(disambiguation when 3D is noisy or ambiguous).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from openptv2.tracking_postprocess import confirm_link_tuples as _confirm_links
from openptv2.tracking_postprocess import kink_statistics as _kink_statistics

#: A10 defaults of the two_phase plugin (tuned over five synthetic cases, checked on
#: four real recordings; see docs/tracking_quality.md and the tracker plan).
DEFAULT_Q_SEED = 0.2
DEFAULT_Q_YOUNG = 3
#: ghost model: "rcm" (ray miss distance + camera count) or "rcm_blob" (adds the
#: brightness agreement of the blobs across cameras, gain-normalised per camera)
DEFAULT_Q_MODEL = "rcm_blob"


#: Two-hop link confirmation is ON by default in the plugin (the one measured big win:
#: lv_multi jump steps 8.4% -> 0.0%; every synthetic benchmark case uses it). The
#: tolerance is a velocity kink in mm/frame, tested for steps of 0.05-0.1 mm/frame, so
#: the default only applies when v_max is small (<= 3 mm/frame). Larger scales need an
#: explicit confirm_tol. An explicit ``confirm_tol: null`` switches it off.
DEFAULT_CONFIRM_TOL = 0.3
DEFAULT_CONFIRM_MAX_VMAX = 3.0
#: Data-driven tolerance: confirm_tol = DEFAULT_CONFIRM_AUTO * median kink of the
#: unconfirmed links (the kink measures the position noise, so the tolerance follows the
#: jitter). 8 beat the fixed 0.3 at every jitter level on the benchmark (real level:
#: -0.009 at a 21-frame window, 2x jitter: -0.054). The fixed value is only the fallback.
DEFAULT_CONFIRM_AUTO = 8.0
#: ... but ONLY in the regime where it was validated (sparse data with a noise-dominated
#: kink); everywhere else the fixed tolerance stays. On the benchmark the automatic value
#: helped at 6 neighbours within 5 mm and kink/step >= 0.68 and hurt badly at density x2/x4
#: (9-11 neighbours) and with frame skipping (kink/step 0.31-0.44).
AUTO_MAX_NEIGHBOURS = 7.0
AUTO_MIN_KINK_RATIO = 0.5


def resolve_confirm_auto(track_cfg: dict) -> float | None:
    """Multiplier of the median kink for the confirmation tolerance, or None.

    An explicit ``confirm_tol`` (number or null) switches the automatic tolerance off
    unless ``confirm_auto`` is given too; with neither, DEFAULT_CONFIRM_AUTO is used."""
    if "confirm_auto" in track_cfg:
        raw = track_cfg["confirm_auto"]
        return None if raw is None else float(raw)
    if "confirm_tol" in track_cfg:
        return None
    return DEFAULT_CONFIRM_AUTO


def resolve_confirm(track_cfg: dict, v_max: float) -> tuple[float | None, bool, str]:
    """(confirm_tol, confirm_ends, note) from the ``track`` section.

    Explicit ``confirm_tol`` (a number or null) always wins and keeps the old
    ``confirm_ends`` default (False). Absent: 0.3 mm/frame with ``confirm_ends``
    True when ``v_max`` is small, otherwise off with a note."""
    if "confirm_tol" in track_cfg:
        raw = track_cfg["confirm_tol"]
        tol = None if raw is None else float(raw)
        return tol, bool(track_cfg.get("confirm_ends", False)), ""
    ends = bool(track_cfg.get("confirm_ends", True))
    if v_max <= DEFAULT_CONFIRM_MAX_VMAX:
        return DEFAULT_CONFIRM_TOL, ends, ""
    return (
        None,
        ends,
        f"v_max={v_max:g} mm/frame is large; two-hop confirmation (confirm_tol) is "
        "not switched on by default. Set confirm_tol explicitly (about 3-4x your "
        "position noise scale).",
    )


@dataclass
class TwoPhaseTrackerConfig:
    """Configuration for the two-phase tracker.

    Attributes
    ----------
    v_max : float
        Maximum velocity in mm/frame. Search radius for frame-to-frame matching.
    max_gap : int
        Maximum number of frames a track can survive without a match.
    dt : float
        Time step between frames (for velocity computation).
    leaf_weight : float
        Weight for 2D leaf distances in the cost matrix. If 0, falls back
        to pure 3D matching.
    use_velocity : bool
        Predict each track forward with its constant-velocity estimate and
        match predictions (not current positions) against new detections.
        Required to cross steady trajectories correctly; without it every
        X-crossing resolves as a bounce. Default True.
    cost_mode : str
        "projected": leaf costs are evaluated at re-projected predicted
        positions (needs project_fn) -- the benchmarked fix for
        maneuver-at-crossing scenes. "3d": cost is the 3D distance between
        prediction and candidate. Falls back to "3d" when no project_fn
        is available.
    allow_shared : bool
        Prototype (shared-observation): in a contested component with more
        tracks than candidates (detector undercount = occlusion), let the
        losing tracks SHARE the winner's detection instead of dying.
        Shared links update position but never velocity (each track's speed
        comes only from its own points). Default False (legacy behaviour).
    max_shared : int
        Maximum consecutive shared frames per track before it must match
        alone again. Default 2.
    share_tol : float | None
        Maximum edge cost (same units as the cost matrix) for a shared
        claim. Sharing without it hijacks strangers: any unassigned track
        inside the gate would co-opt a foreign detection (seen live: a
        gap-stranded track shared two unrelated detections). None disables
        the gate. Default 1.0.
    max_group_size : int
        Groups bigger than this skip the cubic Hungarian and fall back to
        greedy claiming inside the group (sharing still applies). At
        production density the frame percolates into giant components;
        without the cap one frame stalls the run. Default 128.
    confirm_tol : float | None
        Two-hop confirmation (trackcorr's X4 lesson, ported): a consecutive
        link is kept only if its successor continues within this velocity
        kink (mm/frame, same scale as dacc) -- lies rarely confirm twice.
        None disables (legacy behaviour). Default None.
    confirm_ends : bool
        Also sever consecutive links into dead ends (no onward link, not
        the last frame): dying tracks grabbing strangers. Only meaningful
        with confirm_tol set. Default False.
    confirm_auto : float | None
        Set the confirmation tolerance from the data: ``confirm_tol = confirm_auto *
        median kink`` of the unconfirmed links (the kink is noise dominated, so this
        follows the position jitter). Overrides ``confirm_tol``. None = off.
    confirm_auto_max_neighbours, confirm_auto_min_ratio, confirm_auto_radius : float
        The guard of ``confirm_auto`` (all measured on the data, see
        ``docs/tracking_parameters_guide.md`` and ``scripts/tracking_advice.py``): the
        automatic tolerance is used only if the median number of other points within
        ``confirm_auto_radius`` mm of a point is at most ``confirm_auto_max_neighbours``
        (sparse data) and the median kink is at least ``confirm_auto_min_ratio`` times the
        median step (noise dominated). Otherwise the fixed ``confirm_tol`` is used.
    bidirectional : bool
        Run forward tracking, backward tracking on reversed frames, and
        merge the two sets (reciprocal-first core, non-conflicting links
        added greedily by 3D distance). Closes ~75% of the accuracy gap
        to 4-frame trackcorr on dense data in a fraction of the time.
        Default False (unidirectional forward).
    blob_gate : float | None
        Brightness continuity: a real particle keeps its blob brightness in each camera
        from frame to frame (mean |change of log brightness| median 0.06 on real data,
        0.33 for a random neighbour). A candidate whose mean absolute change over the
        cameras that see both points exceeds this is not a candidate. None = off
        (needs ``frame_logb`` in ``track_frames``). 0.5 is a good value for frame-skipped
        or low-frame-rate data (see docs/tracking_quality.md).
    q_weight : float
        A10: per-point ghost probability ``g`` (from ray convergence, see
        ``openptv2.point_quality``) raises the cost of linking onto that
        candidate: ``cost * (1 + q_weight * g)``. 0 = off.
    q_seed : float | None
        A detection with ghost probability above this does not start a new
        trajectory (it can still be picked up by an existing one). None = off.
    q_young : int
        A trajectory with fewer than this many points may not continue onto a
        point whose ghost probability is above ``q_seed``. 0 = off.
    bwd_v_max : float | None
        Optional search radius for the backward pass in bidirectional mode.
        None defaults to v_max. Setting a slightly wider bwd_v_max (e.g. 2.5
        when v_max=2.0) allows backward tracking to reach fast particles that
        forward missed, safely protected by the Forward-First lock.
    vfc_mode : str
        Spatial Neighborhood Consensus (VFC) mode: "off" (default), "pre"
        (Option A: pre-Hungarian gate and/or soft cost penalty), "post"
        (Option B: post-assignment Fast-VFC outlier rejection), or "both".
    vfc_radius : float
        Spatial neighborhood radius in mm for consensus evaluation.
        Default 10.0.
    vfc_min_neighbors : int
        Minimum neighbors required to establish a valid consensus. Default 3.
    vfc_gate : float | None
        Option A: maximum allowed velocity deviation (mm/frame) from local
        median neighbor consensus. Edges exceeding this are pruned before
        Hungarian assignment. None = disabled.
    vfc_weight : float
        Option A: soft cost penalty multiplier for deviation from consensus.
        0.0 = disabled.
    vfc_post_k : float
        Option B: multiplier over local median absolute deviation (MAD) for
        post-assignment outlier rejection. Default 3.0.
    vfc_post_min_tol : float
        Option B: absolute floor tolerance in mm/frame for post-assignment
        rejection. Default 0.5.
    vfc_post_reject_cold_start : bool
        Option B: whether detections rejected by post-VFC are prevented from
        spawning new cold-start tracks. Default True.
    vfc_bootstrap : bool
        Whether to bootstrap initial velocity for cold starts (including
        frame 0 -> 1) using local displacement mode extraction. Default False.
    vfc_bootstrap_grid_size : float
        Spatial binning cell size in mm for coarse velocity field bootstrapping.
        Default 20.0.
    vfc_bootstrap_passes : int
        How the initial velocity field is obtained.
        ``1``: one pass, the field is the displacement mode of the frame
        0 -> 1 candidate pairs (exact for a locally uniform flow, biased in a
        shear).
        ``2`` (default): two passes.  The first pass tracks the sequence with
        zero initial velocity; the field for the second pass is then built
        from the *established* tracks of that pass (local consensus, affine
        when ``vfc_affine`` is on), which is free of the candidate clutter
        and reproduces a shear exactly.  Twice the tracking time.
    vfc_bootstrap_shift_ratio : float
        The two-pass field is only trusted when the flow displacement per
        frame stays below this fraction of the local particle spacing
        (0.5 by default).  Above it the first link of a track is itself
        unreliable, the field of the first pass is scrambled, and the
        single-pass displacement mode is used instead.
    vfc_bootstrap_max_neighbors : int
        Number of samples of the first pass that are used for the field of
        one point (32 by default).  The whole sequence gives one sample per
        link, and the particles pass the same region again and again, so
        the number of samples inside the radius grows with the length of the
        recording.  The nearest ones are the relevant ones.
    vfc_affine : bool
        Whether to fit a local divergence-free affine velocity field
        v(x) = v0 + A(x - x0) (tr(A) = 0) to account for shear and rotation.
        Default False.
    vfc_affine_min_neighbors : int
        Minimum neighbors required to fit the 11-DOF divergence-free affine
        field (3 for v0 and 9 for the gradient, minus 1 for tr(A) = 0).
        Default 6.
    """

    v_max: float = 5.0
    blob_gate: float | None = None
    q_weight: float = 0.0
    q_seed: float | None = None
    q_young: int = 0
    max_gap: int = 2
    dt: float = 1.0
    leaf_weight: float = 1.0
    use_velocity: bool = True
    cost_mode: str = "projected"
    allow_shared: bool = False
    max_shared: int = 2
    share_tol: float | None = 1.0
    max_group_size: int = 128
    confirm_tol: float | None = None
    confirm_ends: bool = False
    confirm_auto: float | None = None
    confirm_auto_max_neighbours: float = 7.0
    confirm_auto_min_ratio: float = 0.5
    confirm_auto_radius: float = 5.0
    bidirectional: bool = False
    parallel_directions: bool = False
    bwd_v_max: float | None = None
    vfc_mode: str = "off"
    vfc_radius: float = 10.0
    vfc_min_neighbors: int = 3
    vfc_gate: float | None = None
    vfc_weight: float = 0.0
    vfc_post_k: float = 3.0
    vfc_post_min_tol: float = 0.5
    vfc_post_reject_cold_start: bool = True
    vfc_bootstrap: bool = False
    vfc_bootstrap_grid_size: float = 20.0
    vfc_bootstrap_passes: int = 2
    vfc_bootstrap_shift_ratio: float = 0.5
    vfc_bootstrap_max_neighbors: int = 32
    vfc_affine: bool = False
    vfc_affine_min_neighbors: int = 6


def _split_hist(
    hist: dict[int, list[tuple[int, int, bool]]],
    severed: set,
) -> dict[int, list[tuple[int, int, bool]]]:
    """Split per-track point histories where confirmation severed a link.

    A consecutive step whose node pair is in ``severed`` starts a new track
    id; shared flags ride along per point. Gap steps are never severed.
    """
    new_hist: dict[int, list[tuple[int, int, bool]]] = {}
    nxt = max(hist) + 1 if hist else 0
    for _tid, pts in hist.items():
        cur = []
        prev = None
        for f, r, s in pts:
            if prev is not None and f - prev[0] == 1 and (prev, (f, r)) in severed:
                if cur:
                    new_hist[nxt] = cur
                    nxt += 1
                cur = []
            cur.append((f, r, s))
            prev = (f, r)
        if cur:
            new_hist[nxt] = cur
            nxt += 1
    return new_hist


def _merge_bidirectional_links(
    fwd_links: list[tuple[int, int, int, int]],
    bwd_links: list[tuple[int, int, int, int]],
    frame_particles: list[np.ndarray],
) -> list[tuple[int, int, int, int]]:
    """Merge forward and backward two-phase links (Forward-First policy).

    Forward links take precedence: they represent causal forward motion with
    accumulated velocity estimates. Backward links from reversed tracking recover
    dropped links, terminal ends, and gaps that forward missed, added greedily by
    3D distance provided both endpoints remain unlinked in the forward set
    (strictly 1-to-1 matching; never steals from or degrades forward links).
    """
    fwd_out = {(t0, r0): (t1, r1) for (t0, r0, t1, r1) in fwd_links}
    fwd_in = {(t1, r1): (t0, r0) for (t0, r0, t1, r1) in fwd_links}
    resolved = set(fwd_links)

    fwd_set = set(fwd_links)
    bwd_cands = [L for L in bwd_links if L not in fwd_set]
    fp = [np.asarray(p, dtype=np.float64) for p in frame_particles]
    bwd_cands.sort(key=lambda x: np.linalg.norm(fp[x[2]][x[3]] - fp[x[0]][x[1]]))

    for t0, r0, t1, r1 in bwd_cands:
        if (t0, r0) not in fwd_out and (t1, r1) not in fwd_in:
            resolved.add((t0, r0, t1, r1))
            fwd_out[(t0, r0)] = (t1, r1)
            fwd_in[(t1, r1)] = (t0, r0)

    return sorted(list(resolved))


def _links_to_linkage(
    links: list[tuple[int, int, int, int]],
    frames: list[int],
    sizes: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-frame (prev, next) row arrays for the ptv_is linkage.

    ptv_is ``next``/``prev`` index rows of the ADJACENT frame, so only links
    between consecutive frame numbers are written. A gap link (t -> t+2 over
    a missed detection) cannot be expressed there: written as-is, its row
    index lands on an unrelated particle of frame t+1, and two gap links with
    coinciding row indices form a reciprocal phantom link (seen live: an
    82 mm one-frame "step"). Gap links are left unlinked at both ends
    instead; downstream gap repair rejoins them.
    """
    out = [
        (np.full(n, -1, dtype=np.int32), np.full(n, -1, dtype=np.int32)) for n in sizes
    ]
    for t0, p0, t1, p1 in links:
        if frames[t1] != frames[t0] + 1 or p0 >= sizes[t0] or p1 >= sizes[t1]:
            continue
        out[t0][1][p0] = p1
        out[t1][0][p1] = p0
    return out


def _chains_from_links(
    links: list[tuple[int, int, int, int]],
    frame_particles: list[np.ndarray],
) -> list[dict]:
    """Assemble trajectory chains from a 1-to-1 link list."""
    nxt = {(t0, r0): (t1, r1) for (t0, r0, t1, r1) in links}
    tgt = set(nxt.values())
    fp = [np.asarray(p, dtype=np.float64) for p in frame_particles]
    chains = []
    tid = 0
    visited = set()
    for s in sorted(set(nxt) - tgt):
        c = [s]
        k = s
        while k in nxt and nxt[k] not in visited:
            visited.add(k)
            k = nxt[k]
            c.append(k)
        fr = [t for (t, r) in c]
        ps = np.array([fp[t][r] for (t, r) in c])
        chains.append(
            {
                "tid": tid,
                "frames": fr,
                "pos": ps,
                "shared": [False] * len(fr),
            }
        )
        tid += 1
    return chains


def fit_local_affine_divergence_free(
    rel_pos: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit local affine velocity field v(r) = v0 + A @ r with tr(A) = 0.

    Parameters
    ----------
    rel_pos : (K, 3) float array
        Neighbor positions relative to the query point (x_k - x_q).
    velocities : (K, 3) float array
        Neighbor velocities.

    Returns
    -------
    v0 : (3,) float array
        Velocity evaluated at query point r = 0.
    residuals : (K,) float array
        Norm of residual vector for each neighbor.
    """
    k = len(rel_pos)
    m_mat = np.zeros((3 * k, 11), dtype=np.float64)
    b_vec = velocities.ravel()

    idx = np.arange(k)
    m_mat[3 * idx, 0] = 1.0
    m_mat[3 * idx, 3:6] = rel_pos
    m_mat[3 * idx + 1, 1] = 1.0
    m_mat[3 * idx + 1, 6:9] = rel_pos
    m_mat[3 * idx + 2, 2] = 1.0
    m_mat[3 * idx + 2, 3] = -rel_pos[:, 2]
    m_mat[3 * idx + 2, 7] = -rel_pos[:, 2]
    m_mat[3 * idx + 2, 9:11] = rel_pos[:, :2]

    try:
        # The normal equations are used instead of np.linalg.lstsq: the
        # system has 11 unknowns and only a few dozen rows, and the LAPACK
        # workspace query of lstsq costs ~240 us per call, which dominates
        # the tracker when the fit runs for every point of every frame.
        ata = m_mat.T @ m_mat
        atb = m_mat.T @ b_vec
        try:
            p = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            p, _, _, _ = np.linalg.lstsq(m_mat, b_vec, rcond=None)
        v0 = p[:3]
        a_mat = np.array(
            [
                [p[3], p[4], p[5]],
                [p[6], p[7], p[8]],
                [p[9], p[10], -p[3] - p[7]],
            ]
        )
        v_pred = v0 + rel_pos @ a_mat.T
        res = np.linalg.norm(velocities - v_pred, axis=1)
        return v0, res
    except Exception:
        med_v = np.median(velocities, axis=0)
        res = np.linalg.norm(velocities - med_v, axis=1)
        return med_v, res


def compute_bootstrap_velocity_field(
    pts0: np.ndarray,
    pts1: np.ndarray,
    v_max: float,
    dt: float = 1.0,
    grid_size: float = 20.0,
    min_pairs: int = 4,
) -> np.ndarray:
    """Compute coarse initial velocity field for frame 0 / cold start particles.

    Partitions the domain into spatial cells of size `grid_size`. For each cell,
    computes candidate displacement vectors to particles within `v_max` in frame 1,
    and extracts the dominant displacement mode.

    Parameters
    ----------
    pts0 : (N, 3) float array
        Particles at frame t.
    pts1 : (M, 3) float array
        Particles at frame t+1.
    v_max : float
        Maximum velocity search radius.
    dt : float
        Time step between frames.
    grid_size : float
        Spatial binning size in mm.
    min_pairs : int
        Minimum candidate pairs required in a cell to form a mode.

    Returns
    -------
    boot_v : (N, 3) float array
        Estimated initial velocity for each particle in `pts0`.
    """
    n0 = len(pts0)
    boot_v = np.zeros((n0, 3), dtype=np.float64)
    if n0 == 0 or len(pts1) == 0:
        return boot_v

    tree0 = cKDTree(pts0)
    tree1 = cKDTree(pts1)
    cand_pairs = tree0.query_ball_tree(tree1, r=v_max)

    min_b = np.min(pts0, axis=0) - 1.0
    max_b = np.max(pts0, axis=0) + 1.0
    bins = [np.arange(min_b[i], max_b[i] + grid_size, grid_size) for i in range(3)]
    for i in range(3):
        if len(bins[i]) < 2:
            bins[i] = np.array([min_b[i], min_b[i] + grid_size])

    ix = np.clip(np.digitize(pts0[:, 0], bins[0]) - 1, 0, len(bins[0]) - 2)
    iy = np.clip(np.digitize(pts0[:, 1], bins[1]) - 1, 0, len(bins[1]) - 2)
    iz = np.clip(np.digitize(pts0[:, 2], bins[2]) - 1, 0, len(bins[2]) - 2)
    cell_ids = ix * 10000 + iy * 100 + iz

    from collections import defaultdict

    cell_disps = defaultdict(list)
    for i, cands in enumerate(cand_pairs):
        cid = cell_ids[i]
        p0 = pts0[i]
        for j in cands:
            cell_disps[cid].append((pts1[j] - p0) / dt)

    cell_vel = {}
    h_range = [(-v_max / dt, v_max / dt)] * 3
    for cid, disps in cell_disps.items():
        if len(disps) >= min_pairs:
            d = np.array(disps, dtype=np.float64)
            h, edges = np.histogramdd(d, bins=8, range=h_range)
            if h.sum() > 0:
                idx = np.unravel_index(np.argmax(h), h.shape)
                in_bin = (
                    (d[:, 0] >= edges[0][idx[0]])
                    & (d[:, 0] <= edges[0][idx[0] + 1])
                    & (d[:, 1] >= edges[1][idx[1]])
                    & (d[:, 1] <= edges[1][idx[1] + 1])
                    & (d[:, 2] >= edges[2][idx[2]])
                    & (d[:, 2] <= edges[2][idx[2] + 1])
                )
                if in_bin.sum() > 0:
                    cell_vel[cid] = np.median(d[in_bin], axis=0)
                else:
                    cell_vel[cid] = np.array(
                        [
                            0.5 * (edges[dim][idx[dim]] + edges[dim][idx[dim] + 1])
                            for dim in range(3)
                        ]
                    )

    if not cell_vel:
        return boot_v

    unassigned = []
    for i in range(n0):
        cid = cell_ids[i]
        if cid in cell_vel:
            boot_v[i] = cell_vel[cid]
        else:
            unassigned.append(i)

    if unassigned and len(cell_vel) > 0:
        assigned_mask = np.ones(n0, dtype=bool)
        assigned_mask[unassigned] = False
        if assigned_mask.any():
            nn_tree = cKDTree(pts0[assigned_mask])
            _, nn_idx = nn_tree.query(pts0[unassigned], k=1)
            boot_v[unassigned] = boot_v[assigned_mask][nn_idx]

    return boot_v


def fit_local_affine_divergence_free_batch(
    rel_list: list[np.ndarray],
    vel_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Batched version of `fit_local_affine_divergence_free`.

    The scalar fit costs ~100 us per point, which dominates the tracker when
    it runs for every point of every frame.  This version builds the design
    matrices of all points at once and solves all the normal equations with
    one batched call.

    Parameters
    ----------
    rel_list : list of (K_i, 3) float arrays
        Neighbour positions relative to each query point.
    vel_list : list of (K_i, 3) float arrays
        Neighbour velocities.

    Returns
    -------
    v0 : (N, 3) float array
        Velocity at each query point.
    disp : (N,) float array
        Median residual of the fit at each query point.
    """
    n = len(rel_list)
    v0 = np.zeros((n, 3), dtype=np.float64)
    disp = np.zeros(n, dtype=np.float64)
    if n == 0:
        return v0, disp

    counts = np.array([len(r) for r in rel_list], dtype=np.int64)
    if counts.max(initial=0) == 0:
        return v0, disp

    # The points are fitted in groups of a similar neighbour count.  A point
    # where many tracks pile up has far more neighbours than the rest, and
    # padding every point to the largest count would need tens of gigabytes.
    for group in _groups_by_count(counts):
        idx = list(group)
        sub_rel = [rel_list[i] for i in idx]
        sub_vel = [vel_list[i] for i in idx]
        g_v0, g_disp = _affine_fit_group(sub_rel, sub_vel)
        v0[group] = g_v0
        disp[group] = g_disp
    return v0, disp


def _affine_fit_group(
    rel_list: list[np.ndarray],
    vel_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """The divergence-free affine fit of one group of points (see above)."""
    n = len(rel_list)
    v0 = np.zeros((n, 3), dtype=np.float64)
    disp = np.zeros(n, dtype=np.float64)
    k_max = max((len(r) for r in rel_list), default=0)
    if k_max == 0:
        return v0, disp

    rel = np.zeros((n, k_max, 3), dtype=np.float64)
    vel = np.zeros((n, k_max, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for i, (r, v) in enumerate(zip(rel_list, vel_list)):
        k = len(r)
        counts[i] = k
        if k:
            rel[i, :k] = r
            vel[i, :k] = v

    # Design matrices: one row per neighbour and per component.
    des_x = np.zeros((n, k_max, 11), dtype=np.float64)
    des_x[:, :, 0] = 1.0
    des_x[:, :, 3:6] = rel
    des_y = np.zeros((n, k_max, 11), dtype=np.float64)
    des_y[:, :, 1] = 1.0
    des_y[:, :, 6:9] = rel
    des_z = np.zeros((n, k_max, 11), dtype=np.float64)
    des_z[:, :, 2] = 1.0
    des_z[:, :, 3] = -rel[:, :, 2]
    des_z[:, :, 7] = -rel[:, :, 2]
    des_z[:, :, 9:11] = rel[:, :, :2]

    # Row order: neighbour 0 (x, y, z), neighbour 1 (x, y, z), ...
    des = np.stack([des_x, des_y, des_z], axis=2).reshape(n, 3 * k_max, 11)
    b_vec = vel.reshape(n, 3 * k_max)

    # The padded neighbours must not enter the fit: zero their rows.
    padded = np.arange(k_max)[None, :] >= counts[:, None]
    if padded.any():
        des[padded.repeat(3, axis=1)] = 0.0

    des_t = des.transpose(0, 2, 1)
    ata = np.matmul(des_t, des)
    atb = np.matmul(des_t, b_vec[:, :, None])[:, :, 0]

    # A tiny ridge keeps the system solvable when the neighbours do not span
    # three dimensions.  It is far below the noise of the data.
    scale = np.maximum(np.einsum("naa->n", ata) / 11.0, 1e-30)
    ata = ata + (1e-12 * scale)[:, None, None] * np.eye(11)
    try:
        sol = np.linalg.solve(ata, atb[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.zeros((n, 11), dtype=np.float64)
        for i in range(n):
            try:
                sol[i] = np.linalg.solve(ata[i], atb[i])
            except np.linalg.LinAlgError:
                sol[i] = np.linalg.lstsq(des[i], b_vec[i], rcond=None)[0]

    v0 = sol[:, :3]
    a_mat = np.empty((n, 3, 3), dtype=np.float64)
    a_mat[:, 0, 0] = sol[:, 3]
    a_mat[:, 0, 1] = sol[:, 4]
    a_mat[:, 0, 2] = sol[:, 5]
    a_mat[:, 1, 0] = sol[:, 6]
    a_mat[:, 1, 1] = sol[:, 7]
    a_mat[:, 1, 2] = sol[:, 8]
    a_mat[:, 2, 0] = sol[:, 9]
    a_mat[:, 2, 1] = sol[:, 10]
    a_mat[:, 2, 2] = -sol[:, 3] - sol[:, 7]

    pred = np.matmul(rel, a_mat.transpose(0, 2, 1)) + v0[:, None, :]
    res = np.linalg.norm(vel - pred, axis=2)
    mask = np.arange(k_max)[None, :] < counts[:, None]
    for i in range(n):
        vals = res[i][mask[i]]
        disp[i] = float(np.median(vals)) if len(vals) else 0.0
    return v0, disp


def compute_local_consensus_field(
    positions: np.ndarray,
    velocities: np.ndarray,
    query_positions: np.ndarray,
    radius: float,
    min_neighbors: int = 3,
    use_affine: bool = False,
    affine_min_neighbors: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local consensus velocity for a set of query positions.

    Parameters
    ----------
    positions : (N, 3) float array
        Established reference positions (e.g. tracks with length >= 2).
    velocities : (N, 3) float array
        Established velocities corresponding to `positions`.
    query_positions : (M, 3) float array
        Query positions where local consensus should be evaluated.
    radius : float
        Spatial neighborhood radius in mm.
    min_neighbors : int
        Minimum neighbors required to establish a valid consensus.
    use_affine : bool
        Whether to fit a divergence-free local affine velocity field tr(A)=0.
    affine_min_neighbors : int
        Minimum neighbors required for the affine fit (default 6).

    Returns
    -------
    consensus_v : (M, 3) float array
        Local consensus velocity (NaN where invalid).
    consensus_disp : (M,) float array
        Local dispersion (MAD or affine residual).
    consensus_valid : (M,) bool array
        True where consensus could be formed (>= min_neighbors).
    """
    m = len(query_positions)
    consensus_v = np.full((m, 3), np.nan, dtype=np.float64)
    consensus_disp = np.zeros(m, dtype=np.float64)
    consensus_valid = np.zeros(m, dtype=bool)

    if len(positions) < min_neighbors or m == 0:
        return consensus_v, consensus_disp, consensus_valid

    tree = cKDTree(positions)
    nbrs_list = tree.query_ball_point(query_positions, r=radius)

    if use_affine:
        # The affine fits are done in one batch.
        fit_idx = [
            ki for ki, nbrs in enumerate(nbrs_list) if len(nbrs) >= affine_min_neighbors
        ]
        if fit_idx:
            rel_list = [
                positions[nbrs_list[ki]] - query_positions[ki] for ki in fit_idx
            ]
            vel_list = [velocities[nbrs_list[ki]] for ki in fit_idx]
            v0, disp = fit_local_affine_divergence_free_batch(rel_list, vel_list)
            for pos, ki in enumerate(fit_idx):
                consensus_v[ki] = v0[pos]
                consensus_disp[ki] = disp[pos]
                consensus_valid[ki] = True
        # The points with too few neighbours for the fit use the median.
        med_idx = [
            ki
            for ki, nbrs in enumerate(nbrs_list)
            if not consensus_valid[ki] and len(nbrs) >= min_neighbors
        ]
        if med_idx:
            sub = [nbrs_list[ki] for ki in med_idx]
            med_v, mad = _neighbourhood_median_and_mad(sub, velocities)
            for pos, ki in enumerate(med_idx):
                consensus_v[ki] = med_v[pos]
                consensus_disp[ki] = mad[pos]
                consensus_valid[ki] = True
        return consensus_v, consensus_disp, consensus_valid

    med_idx = [ki for ki, nbrs in enumerate(nbrs_list) if len(nbrs) >= min_neighbors]
    if med_idx:
        sub = [nbrs_list[ki] for ki in med_idx]
        med_v, mad = _neighbourhood_median_and_mad(sub, velocities)
        for pos, ki in enumerate(med_idx):
            consensus_v[ki] = med_v[pos]
            consensus_disp[ki] = mad[pos]
            consensus_valid[ki] = True

    return consensus_v, consensus_disp, consensus_valid


def _groups_by_count(counts: np.ndarray) -> list[np.ndarray]:
    """Indices grouped by the number of neighbours.

    The neighbour counts of a frame can differ by orders of magnitude (a
    point where many tracks pile up has far more neighbours than the rest).
    Padding every point to the largest count would then need tens of
    gigabytes.  The groups keep the count inside a factor of two, so the
    padded arrays stay small and the work stays close to the useful work.
    """
    order = np.argsort(counts, kind="stable")
    groups: list[np.ndarray] = []
    start = 0
    n = len(order)
    for i in range(1, n + 1):
        if i == n or counts[order[i]] > 2 * max(int(counts[order[start]]), 1):
            groups.append(order[start:i])
            start = i
    return groups


def _padded_neighbour_values(
    nbrs_list: list[list[int]],
    values: np.ndarray,
) -> np.ndarray:
    """Gather the values of the neighbours into a padded array (NaN in the padding).

    The neighbour lists have different lengths.  Padding them lets the median
    and the MAD of every neighbourhood be computed in one vectorised call
    instead of one call per link (that was the slowest part of the tracker
    on dense data).
    """
    n = len(nbrs_list)
    k_max = max((len(x) for x in nbrs_list), default=0)
    out = np.full((n, k_max) + values.shape[1:], np.nan, dtype=np.float64)
    for i, nbrs in enumerate(nbrs_list):
        if len(nbrs):
            out[i, : len(nbrs)] = values[nbrs]
    return out


def _nan_median(values: np.ndarray, axis: int) -> np.ndarray:
    """Median along an axis, ignoring the NaN padding."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _neighbourhood_median_and_mad(
    nbrs_list: list[list[int]],
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD of the neighbourhood of every point (vectorised)."""
    n = len(nbrs_list)
    counts = np.array([len(x) for x in nbrs_list], dtype=np.int64)
    med = np.zeros((n, values.shape[1]), dtype=np.float64)
    mad = np.zeros(n, dtype=np.float64)
    for group in _groups_by_count(counts):
        sub = [nbrs_list[i] for i in group]
        padded = _padded_neighbour_values(sub, values)
        med_g = _nan_median(padded, axis=1)
        dev = np.linalg.norm(padded - med_g[:, None, :], axis=2)
        med[group] = med_g
        mad[group] = _nan_median(dev, axis=1)
    return med, mad


def filter_links_by_spatial_consensus(
    links: set[tuple[int, int]],
    tids: list[int],
    tracks: dict[int, dict],
    pts1: np.ndarray,
    dt: float,
    current_t: int,
    radius: float,
    min_neighbors: int = 3,
    post_k: float = 3.0,
    min_tol: float = 0.5,
) -> tuple[set[tuple[int, int]], set[int]]:
    """Filter tentative frame matches by spatial neighborhood consensus (Option B / Fast-VFC).

    Parameters
    ----------
    links : set of (ai, det) tuples
        Tentative matches from Hungarian assignment.
    tids : list of int
        Active track IDs corresponding to predictor indices `ai`.
    tracks : dict of int -> dict
        Active track dictionary containing 'pos' and 'last_t'.
    pts1 : (M, 3) float array
        Candidate positions at frame t+1.
    dt : float
        Time step.
    current_t : int
        Current time index (step t+1).
    radius : float
        Neighborhood radius in mm.
    min_neighbors : int
        Minimum neighbors required to enforce consensus.
    post_k : float
        Multiplier over local MAD dispersion.
    min_tol : float
        Absolute residual floor in mm/frame.

    Returns
    -------
    kept_links : set of (ai, det) tuples
    rejected_dets : set of int
        Detection indices rejected as spatial outliers.
    """
    if len(links) < min_neighbors:
        return links, set()

    link_list = list(links)
    pos0 = np.array(
        [tracks[tids[int(ai)]]["pos"] for ai, _ in link_list], dtype=np.float64
    )
    dt_steps = np.array(
        [(current_t - tracks[tids[int(ai)]]["last_t"]) * dt for ai, _ in link_list],
        dtype=np.float64,
    )
    dt_steps = np.maximum(dt_steps, 1e-6)
    cands = np.array([pts1[int(det)] for _, det in link_list], dtype=np.float64)
    disps = (cands - pos0) / dt_steps[:, None]

    tree = cKDTree(pos0)
    nbrs_list = tree.query_ball_point(pos0, r=radius)

    # One vectorised median per neighbourhood instead of one call per link.
    med_disp, mad = _neighbourhood_median_and_mad(nbrs_list, disps)
    counts = np.array([len(x) for x in nbrs_list])

    rejected_pairs = set()
    rejected_dets = set()
    res = np.linalg.norm(disps - med_disp, axis=1)
    tol = np.maximum(post_k * mad, min_tol)
    for ki, (ai, det) in enumerate(link_list):
        if counts[ki] >= min_neighbors and res[ki] > tol[ki]:
            rejected_pairs.add((ai, det))
            rejected_dets.add(int(det))

    kept_links = {pair for pair in links if pair not in rejected_pairs}
    return kept_links, rejected_dets


def _match_two_phase_frame(
    pts0: np.ndarray,
    pts1: np.ndarray,
    xy0: np.ndarray,
    xy1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    leaf_weight: float = 1.0,
    cost_mode: str = "projected",
    allow_shared: bool = False,
    share_tol: float | None = None,
    max_group_size: int = 128,
    seen0: np.ndarray | None = None,
    seen1: np.ndarray | None = None,
    ghost1: np.ndarray | None = None,
    q_weight: float = 0.0,
    logb0: np.ndarray | None = None,
    logb1: np.ndarray | None = None,
    blob_gate: float | None = None,
    young0: np.ndarray | None = None,
    q_block: float = 1.0,
    consensus_v: np.ndarray | None = None,
    consensus_disp: np.ndarray | None = None,
    consensus_valid: np.ndarray | None = None,
    vfc_gate: float | None = None,
    vfc_weight: float = 0.0,
    pts0_orig: np.ndarray | None = None,
    dt_steps: np.ndarray | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Two-phase frame-to-frame matching: 3D search + 2D ranking.

    Parameters
    ----------
    pts0, pts1 : (N, 3) and (M, 3) — 3D positions in mm. With velocity
        prediction enabled these are PREDICTED positions; pass the
        re-projected pixel positions as xy0 so costs are evaluated at the
        prediction (stale appearance nulls the prediction -- the bounce
        bias returns).
    xy0, xy1 : (N, D) and (M, D) — flattened 2D leaf features
    p0, p1 : particle IDs for frame 0 and 1
    radius : float — 3D search radius in mm
    leaf_weight : float — weight for 2D distances in cost matrix
    cost_mode : str — "projected" (2D leaf costs) or "3d" (3D distance
        between pts0 and candidates; needs no calibration).
    allow_shared : bool — prototype: in components with more predictors
        than candidates, unassigned predictors share their best candidate
        instead of going unmatched.
    share_tol : float | None — maximum edge cost for a shared claim
        (mutual-good-prediction gate). None disables the gate.
    seen0, seen1 : (N, C) and (M, C) bool arrays — per-camera "really seen".
        Missing cameras must be marked here, NOT as NaN inside xy0/xy1 (any
        later nan_to_num would silently un-mark them, which is bug 2). None
        falls back to the legacy NaN check.

    Returns
    -------
    links : set of (pid0, pid1) pairs
    shared : set of (pid0, pid1) pairs, subset of links, observed jointly
        (empty unless allow_shared)
    """
    n_pred = len(pts0)
    n_cand = len(pts1)
    if n_pred == 0 or n_cand == 0:
        return set(), set()

    # Phase 1: 3D KD-tree candidate search
    tree3d = cKDTree(pts1)
    neighbours = tree3d.query_ball_point(pts0, r=radius)

    if young0 is not None and ghost1 is not None:
        g1 = np.asarray(ghost1)
        neighbours = [
            [c for c in lst if g1[c] <= q_block] if young0[i] else lst
            for i, lst in enumerate(neighbours)
        ]

    # Build edge list with 2D costs
    use_leaves = cost_mode == "projected" and leaf_weight > 0 and xy0.shape[1] > 0
    C = xy0.shape[1] // 2 if use_leaves else 0

    def _resolve_seen(seen, xy, n, name):
        if not C:
            return np.zeros((n, 0), dtype=bool)  # 2D leaves unused
        if seen is None:
            # Legacy: NaN marks a missing camera (only valid if the caller
            # never ran nan_to_num on the leaves).
            return ~np.isnan(xy.reshape(n, -1, 2)[:, :, 0])
        seen = np.asarray(seen, dtype=bool)
        if seen.shape != (n, C):
            # A wrong mask must never fall back to the NaN check: after
            # nan_to_num that silently re-creates the missing-camera bug.
            raise ValueError(f"{name} has shape {seen.shape}, expected {(n, C)}")
        return seen

    seen0 = _resolve_seen(seen0, xy0, n_pred, "seen0")
    seen1 = _resolve_seen(seen1, xy1, n_cand, "seen1")
    lens = np.fromiter((len(c) for c in neighbours), dtype=np.intp, count=n_pred)
    n_edges = int(lens.sum())
    if n_edges == 0:
        return set(), set()
    row_arr = np.repeat(np.arange(n_pred), lens)
    col_arr = np.fromiter(
        (c for lst in neighbours for c in lst), dtype=np.intp, count=n_edges
    )
    if use_leaves:
        # 2D cost: mean Euclidean distance per camera, weighted by overlap count
        valid = seen0[row_arr] & seen1[col_arr]
        n_valid = valid.sum(axis=1)
        with np.errstate(invalid="ignore"):
            cam_dists = np.linalg.norm(
                xy1[col_arr].reshape(n_edges, C, 2)
                - xy0[row_arr].reshape(n_edges, C, 2),
                axis=2,
            )
        # Unseen cameras may hold NaN: mask them out, never multiply by 0
        sum_dists = np.where(valid, cam_dists, 0.0).sum(axis=1)
        nv = np.maximum(n_valid, 1)
        # Weight: more shared cameras = more reliable distance
        d2d = np.where(n_valid > 0, sum_dists / nv * (C / nv), 1e6)
        cost_arr = d2d * leaf_weight
    else:
        # Fallback: 3D distance
        cost_arr = np.linalg.norm(pts0[row_arr] - pts1[col_arr], axis=1)
    if logb0 is not None and logb1 is not None and blob_gate:
        with np.errstate(all="ignore"):
            delta = np.nanmean(np.abs(logb1[col_arr] - logb0[row_arr]), axis=1)
        ok = ~(delta > blob_gate)  # NaN (no common camera) stays a candidate
        if not ok.all():
            row_arr, col_arr, cost_arr = row_arr[ok], col_arr[ok], cost_arr[ok]
            if len(row_arr) == 0:
                return set(), set()
    if ghost1 is not None and q_weight > 0:
        cost_arr = cost_arr * (1.0 + q_weight * np.asarray(ghost1)[col_arr])

    # Spatial Neighborhood Consensus (Option A: Pre-Hungarian Gate / Cost Penalty)
    if consensus_valid is not None and (vfc_gate is not None or vfc_weight > 0):
        c_mask = consensus_valid[row_arr]
        if c_mask.any():
            r_idx = row_arr[c_mask]
            c_idx = col_arr[c_mask]
            p_orig = pts0[r_idx] if pts0_orig is None else pts0_orig[r_idx]
            dt_s = (
                np.ones((len(r_idx), 1), dtype=np.float64)
                if dt_steps is None
                else np.asarray(dt_steps[r_idx, None], dtype=np.float64)
            )
            dt_s = np.maximum(dt_s, 1e-6)
            disp = (pts1[c_idx] - p_orig) / dt_s
            delta_v = np.linalg.norm(disp - consensus_v[r_idx], axis=1)

            if vfc_gate is not None and vfc_gate > 0:
                bad_edges = delta_v > vfc_gate
                if bad_edges.any():
                    keep = np.ones(len(row_arr), dtype=bool)
                    idx_in_row = np.flatnonzero(c_mask)
                    keep[idx_in_row[bad_edges]] = False
                    row_arr = row_arr[keep]
                    col_arr = col_arr[keep]
                    cost_arr = cost_arr[keep]
                    if len(row_arr) == 0:
                        return set(), set()
                    c_mask = consensus_valid[row_arr]
                    if c_mask.any() and vfc_weight > 0:
                        r_idx = row_arr[c_mask]
                        c_idx = col_arr[c_mask]
                        p_orig = pts0[r_idx] if pts0_orig is None else pts0_orig[r_idx]
                        dt_s = (
                            np.ones((len(r_idx), 1), dtype=np.float64)
                            if dt_steps is None
                            else np.asarray(dt_steps[r_idx, None], dtype=np.float64)
                        )
                        dt_s = np.maximum(dt_s, 1e-6)
                        disp = (pts1[c_idx] - p_orig) / dt_s
                        delta_v = np.linalg.norm(disp - consensus_v[r_idx], axis=1)

            if vfc_weight > 0 and c_mask.any():
                idx_in_row = np.flatnonzero(c_mask)
                disp_scale = (
                    np.ones(len(r_idx))
                    if consensus_disp is None
                    else np.maximum(consensus_disp[r_idx], 1e-4)
                )
                penalty = 1.0 + vfc_weight * (delta_v / disp_scale)
                cost_arr[idx_in_row] *= penalty

    # Phase 2: Hungarian via connected components
    n_nodes = n_pred + n_cand
    graph = coo_matrix(
        (np.ones(len(row_arr), dtype=np.int8), (row_arr, col_arr + n_pred)),
        shape=(n_nodes, n_nodes),
    )
    n_comp, labels = connected_components(graph, directed=False)

    links: set[tuple[int, int]] = set()
    shared: set[tuple[int, int]] = set()
    edge_comp = labels[row_arr]
    comp_edges = np.bincount(edge_comp, minlength=n_comp)

    # Trivial components: accept directly
    trivial = comp_edges[edge_comp] == 1
    for r, c in zip(row_arr[trivial], col_arr[trivial]):
        links.add((int(p0[r]), int(p1[c])))

    # Non-trivial: small dense Hungarian per component
    rest = np.flatnonzero(~trivial)
    assigned_rows: set[int] = set()
    assigned_cols: set[int] = set()
    if len(rest):
        rest = rest[np.argsort(edge_comp[rest], kind="stable")]
        splits = np.flatnonzero(np.diff(edge_comp[rest])) + 1
        for group in np.split(rest, splits):
            c_rows = row_arr[group].tolist()
            c_cols = col_arr[group].tolist()
            c_costs = cost_arr[group].tolist()
            uniq_r = sorted(set(c_rows))
            uniq_c = sorted(set(c_cols))
            if len(uniq_r) + len(uniq_c) > max_group_size:
                # Production-density percolation: cubic Hungarian would stall
                # the frame. Greedy inside the group, sharing still applies.
                order = np.argsort(np.array(c_costs), kind="stable")
                for k in order.tolist():
                    r, c = c_rows[k], c_cols[k]
                    if r not in assigned_rows and c not in assigned_cols:
                        links.add((int(p0[r]), int(p1[c])))
                        assigned_rows.add(r)
                        assigned_cols.add(c)
                if allow_shared:
                    for r in uniq_r:
                        if r in assigned_rows:
                            continue
                        best_c, best_d = None, np.inf
                        for k, (rr, cc) in enumerate(zip(c_rows, c_cols)):
                            if rr == r and c_costs[k] < best_d:
                                best_d, best_c = c_costs[k], cc
                        if best_c is not None and (
                            share_tol is None or best_d < share_tol
                        ):
                            links.add((int(p0[r]), int(p1[best_c])))
                            shared.add((int(p0[r]), int(p1[best_c])))
                continue
            r_local = {v: i for i, v in enumerate(uniq_r)}
            c_local = {v: i for i, v in enumerate(uniq_c)}
            max_cost = max(c_costs) if c_costs else 1.0
            sentinel = max_cost * len(c_costs) + 1.0
            sub = np.full((len(uniq_r), len(uniq_c)), sentinel, dtype=np.float64)
            for rr, cc, dd in zip(c_rows, c_cols, c_costs):
                sub[r_local[rr], c_local[cc]] = dd
            r_ind, c_ind = linear_sum_assignment(sub)
            real = sub[r_ind, c_ind] < sentinel
            group_winners = []
            for r_i, c_i in zip(r_ind[real], c_ind[real]):
                links.add((int(p0[uniq_r[r_i]]), int(p1[uniq_c[c_i]])))
                assigned_rows.add(int(uniq_r[r_i]))
                assigned_cols.add(int(uniq_c[c_i]))
                group_winners.append((int(p0[uniq_r[r_i]]), int(p1[uniq_c[c_i]])))
            # Prototype shared-observation: more predictors than candidates
            # = detector undercount (occlusion). Unassigned predictors share
            # their best candidate instead of dying. Tracked by the caller
            # via the streak cap.
            group_shared = []
            if allow_shared and len(uniq_r) > len(uniq_c):
                for r_i, r in enumerate(uniq_r):
                    if r in assigned_rows:
                        continue
                    best_c, best_d = None, sentinel
                    for c_i, c in enumerate(uniq_c):
                        if sub[r_i, c_i] < best_d:
                            best_d, best_c = sub[r_i, c_i], c
                    if (
                        best_c is not None
                        and best_d < sentinel
                        and (share_tol is None or best_d < share_tol)
                    ):
                        links.add((int(p0[r]), int(p1[best_c])))
                        group_shared.append((int(p0[r]), int(p1[best_c])))
            if group_shared:
                # The winners' points in a sharing group are joint evidence
                # too: nobody updates velocity from a merged point, or the
                # winner predicts from poisoned history at separation.
                shared.update(group_winners)
                shared.update(group_shared)

    return links, shared


class TwoPhaseTracker:
    """Frame-to-frame tracker using 3D search + 2D leaf ranking.

    Parameters
    ----------
    config : TwoPhaseTrackerConfig
        Tracker configuration.
    """

    def __init__(self, config: TwoPhaseTrackerConfig | None = None):
        self.cfg = config or TwoPhaseTrackerConfig()

    def track_frames(
        self,
        frame_particles: list[np.ndarray],
        frame_leaves: list[np.ndarray] | None = None,
        project_fn: Any = None,
        return_chains: bool = False,
        frame_seen: list[np.ndarray] | None = None,
        frame_ghost: list[np.ndarray] | None = None,
        frame_logb: list[np.ndarray] | None = None,
    ) -> (
        list[tuple[int, int, int, int]]
        | tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]
    ):
        """Track particles across frames using two-phase matching.

        Stateful: every live track carries a constant-velocity estimate.
        Each step matches velocity PREDICTIONS (not current positions)
        against new detections, so steady crossings resolve correctly.
        New detections spawn zero-velocity tracks (cold start); tracks
        unmatched for more than ``max_gap`` frames retire, so a particle
        occluded for a frame is re-caught by gap-spanning prediction.

        Parameters
        ----------
        frame_particles : list of (N_i, 3) arrays
            3D positions per frame.
        frame_leaves : list of (N_i, D) arrays, optional
            Flattened 2D leaf features per frame. If None, falls back to
            pure 3D matching.
        project_fn : callable, optional
            ``(N, 3) -> (N, D)`` mapping predicted 3D positions to leaf
            features (re-projection through the camera models). Required
            for ``cost_mode="projected"``; without it costs fall back to
            3D distance (see ``cost_mode``).
        frame_seen : list of (N_i, C) bool arrays, optional
            Per-camera "really seen" masks matching frame_leaves. Predictions
            are always fully seen (projection is defined everywhere); only
            the detection side uses these. None = legacy NaN check.

        Returns
        -------
        links : list of (t0, pid0, t1, pid1) tuples
            Frame-to-frame particle links (0-based time indices, row ids
            within each frame's arrays). Gap-spanning links reference the
            track's last seen frame/row.
        (if return_chains) chains : list of dicts with keys tid, frames,
            rows, pos, shared — per-track point histories; shared flags
            mark jointly-observed points. Row-links alone cannot represent
            sharing (one node, two owners), hence chains.
        """
        num_frames = len(frame_particles)
        if num_frames < 2:
            return ([], []) if return_chains else []

        if not self.cfg.bidirectional:
            return self._track_unidirectional(
                frame_particles,
                frame_leaves,
                project_fn,
                return_chains,
                frame_seen=frame_seen,
                frame_ghost=frame_ghost,
                frame_logb=frame_logb,
            )

        # Bidirectional tracking: forward + backward on reversed frames.
        # The two passes share no state, so they can run concurrently.
        bwd_vmax = (
            self.cfg.bwd_v_max if self.cfg.bwd_v_max is not None else self.cfg.v_max
        )

        def _rev(x):
            return x[::-1] if x is not None else None

        fwd_kw = dict(
            return_chains=False,
            frame_seen=frame_seen,
            frame_ghost=frame_ghost,
            frame_logb=frame_logb,
        )
        bwd_kw = dict(
            return_chains=False,
            v_max_override=bwd_vmax,
            frame_seen=_rev(frame_seen),
            frame_ghost=_rev(frame_ghost),
            frame_logb=_rev(frame_logb),
        )

        def run_fwd():
            return self._track_unidirectional(
                frame_particles, frame_leaves, project_fn, **fwd_kw
            )

        def run_bwd():
            return self._track_unidirectional(
                frame_particles[::-1], _rev(frame_leaves), project_fn, **bwd_kw
            )

        if self.cfg.parallel_directions:
            # Threads do not help (the matching loop is GIL-bound); fork a
            # child for the backward pass. It inherits the frame arrays and
            # closures copy-on-write; only the link list is sent back.
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            recv, send = ctx.Pipe(duplex=False)

            def _child():
                try:
                    send.send(("ok", run_bwd()))
                except BaseException as exc:  # report, never hang the parent
                    send.send(("err", repr(exc)))
                finally:
                    send.close()

            proc = ctx.Process(target=_child, daemon=True)
            proc.start()
            send.close()
            fwd_links = cast(list[tuple[int, int, int, int]], run_fwd())
            status, payload = recv.recv()
            proc.join()
            if status != "ok":
                raise RuntimeError(f"backward tracking process failed: {payload}")
            bwd_links_raw = cast(list[tuple[int, int, int, int]], payload)
        else:
            fwd_links = cast(list[tuple[int, int, int, int]], run_fwd())
            bwd_links_raw = cast(list[tuple[int, int, int, int]], run_bwd())
        bwd_links = [
            (num_frames - 1 - rt1, rr1, num_frames - 1 - rt0, rr0)
            for (rt0, rr0, rt1, rr1) in bwd_links_raw
        ]

        merged_links = _merge_bidirectional_links(fwd_links, bwd_links, frame_particles)
        if not return_chains:
            return merged_links

        chains = _chains_from_links(merged_links, frame_particles)
        return merged_links, chains

    def _track_unidirectional(
        self,
        frame_particles: list[np.ndarray],
        frame_leaves: list[np.ndarray] | None = None,
        project_fn: Any = None,
        return_chains: bool = False,
        v_max_override: float | None = None,
        frame_seen: list[np.ndarray] | None = None,
        frame_ghost: list[np.ndarray] | None = None,
        frame_logb: list[np.ndarray] | None = None,
    ) -> (
        list[tuple[int, int, int, int]]
        | tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]
    ):
        num_frames = len(frame_particles)
        if num_frames < 2:
            return ([], []) if return_chains else []

        eff_vmax = v_max_override if v_max_override is not None else self.cfg.v_max

        init_vels = None
        if self.cfg.vfc_bootstrap:
            if self.cfg.vfc_bootstrap_passes >= 2:
                # Two passes: the field for the second pass is built from the
                # established tracks of the first one.
                first = self._run_pass(
                    frame_particles,
                    frame_leaves,
                    project_fn,
                    return_chains,
                    eff_vmax,
                    frame_seen,
                    frame_ghost,
                    frame_logb,
                    init_vels=None,
                )
                links = first[0] if isinstance(first, tuple) else first
                field = self._bootstrap_field_from_links(links, frame_particles)
                if field is not None and self._bootstrap_field_is_trustworthy(
                    field, frame_particles, links
                ):
                    return self._run_pass(
                        frame_particles,
                        frame_leaves,
                        project_fn,
                        return_chains,
                        eff_vmax,
                        frame_seen,
                        frame_ghost,
                        frame_logb,
                        init_vels=field,
                    )
                # The first pass is not reliable enough -> single pass with
                # the displacement-mode field.
                init_vels = self._mode_bootstrap_field(frame_particles, eff_vmax)
            else:
                init_vels = self._mode_bootstrap_field(frame_particles, eff_vmax)

        return self._run_pass(
            frame_particles,
            frame_leaves,
            project_fn,
            return_chains,
            eff_vmax,
            frame_seen,
            frame_ghost,
            frame_logb,
            init_vels=init_vels,
        )

    def _mode_bootstrap_field(
        self, frame_particles: list[np.ndarray], eff_vmax: float
    ) -> np.ndarray:
        """Coarse initial velocity field from the displacement mode (one pass)."""
        pts0 = np.asarray(frame_particles[0], dtype=np.float64)
        pts1 = np.asarray(frame_particles[1], dtype=np.float64)
        return compute_bootstrap_velocity_field(
            pts0,
            pts1,
            v_max=eff_vmax,
            dt=self.cfg.dt,
            grid_size=self.cfg.vfc_bootstrap_grid_size,
        )

    def _bootstrap_field_from_links(
        self,
        links: list[tuple[int, int, int, int]],
        frame_particles: list[np.ndarray],
    ) -> np.ndarray | None:
        """Frame-0 velocity field built from the tracks of a first pass.

        The links of a completed pass carry the velocity of every tracked
        step.  Their local consensus (median, or the divergence-free affine
        fit of Phase 2B) is a *clean* field: it is free of the candidate
        clutter that makes the displacement-mode extraction unreliable in a
        shear, and it reproduces a linear shear exactly.

        The cold-start step (t0 = 0) is left out: it is the one link made
        without a velocity and therefore the least trustworthy.

        Only the nearest samples are used for every point.  The whole
        sequence gives one sample per link, and the particles pass the same
        region again and again, so the number of samples inside the radius
        grows with the length of the sequence (a hundred thousand on a long
        recording).  The nearest ones are the relevant ones.
        """
        if not links:
            return None
        n_links = len(links)
        pos = np.empty((n_links, 3), dtype=np.float64)
        vel = np.empty((n_links, 3), dtype=np.float64)
        m = 0
        for t0, r0, t1, r1 in links:
            if t0 == 0:
                continue
            step = t1 - t0
            if step <= 0:
                continue
            p0 = np.asarray(frame_particles[t0][r0], dtype=np.float64)
            p1 = np.asarray(frame_particles[t1][r1], dtype=np.float64)
            pos[m] = 0.5 * (p0 + p1)
            vel[m] = (p1 - p0) / (step * self.cfg.dt)
            m += 1
        if m < self.cfg.vfc_min_neighbors:
            return None
        pos = pos[:m]
        vel = vel[:m]

        query = np.asarray(frame_particles[0], dtype=np.float64)
        k = int(min(m, self.cfg.vfc_bootstrap_max_neighbors))
        tree = cKDTree(pos)
        dist, idx = tree.query(query, k=k)
        idx = np.atleast_2d(idx)
        dist = np.atleast_2d(dist)
        valid_nbrs = np.isfinite(dist)

        # The points with enough neighbours, and their neighbours.
        q_idx: list[int] = []
        nbr_idx: list[np.ndarray] = []
        rel_list: list[np.ndarray] = []
        vel_list: list[np.ndarray] = []
        for i in range(len(query)):
            n_ok = int(valid_nbrs[i].sum())
            if n_ok < self.cfg.vfc_min_neighbors:
                continue
            nb = idx[i, :n_ok]
            q_idx.append(i)
            nbr_idx.append(nb)
            rel_list.append(pos[nb] - query[i])
            vel_list.append(vel[nb])
        if len(rel_list) < self.cfg.vfc_min_neighbors:
            return None

        field = np.full((len(query), 3), np.nan, dtype=np.float64)
        if self.cfg.vfc_affine:
            fit_pos = [
                j
                for j, r in enumerate(rel_list)
                if len(r) >= self.cfg.vfc_affine_min_neighbors
            ]
            if fit_pos:
                v0, _disp = fit_local_affine_divergence_free_batch(
                    [rel_list[j] for j in fit_pos], [vel_list[j] for j in fit_pos]
                )
                for j, point in enumerate(fit_pos):
                    field[q_idx[point]] = v0[j]

        # The points with too few neighbours for the fit use the median.
        med_pos = [
            j
            for j, r in enumerate(rel_list)
            if len(r) >= self.cfg.vfc_min_neighbors
            and not np.isfinite(field[q_idx[j]]).all()
        ]
        if med_pos:
            med_v, _mad = _neighbourhood_median_and_mad(
                [nbr_idx[j] for j in med_pos], vel
            )
            for j, point in enumerate(med_pos):
                field[q_idx[point]] = med_v[j]

        valid = np.isfinite(field).all(axis=1)
        if not valid.any():
            return None
        if not valid.all():
            # Too few neighbours here: inherit the nearest valid value.
            valid_idx = np.flatnonzero(valid)
            _, nn = cKDTree(query[valid_idx]).query(query[~valid], k=1)
            field[~valid] = field[valid_idx][nn]
        return field

    def _bootstrap_field_is_trustworthy(
        self,
        field: np.ndarray,
        frame_particles: list[np.ndarray],
        links: list[tuple[int, int, int, int]],
    ) -> bool:
        """Is the first-pass field worth using as the initial velocity?

        The field is built from the tracks of the first pass, and the first
        link of a track is made without a velocity.  It is only trustworthy
        when the flow moves the particles by clearly less than their spacing
        -- otherwise the first link is essentially a random neighbour and the
        field is scrambled.  The test is the displacement of the first-step
        links against the local particle spacing.
        """
        ratio = self.cfg.vfc_bootstrap_shift_ratio
        if ratio <= 0:
            return True
        pts = np.asarray(frame_particles[0], dtype=np.float64)
        if len(pts) < 2:
            return False
        tree = cKDTree(pts)
        dist, _ = tree.query(pts, k=2)
        spacing = float(np.median(dist[:, 1]))
        if not np.isfinite(spacing) or spacing <= 0:
            return False

        shifts = []
        for t0, r0, t1, r1 in links:
            if t0 != 0:
                continue
            p0 = np.asarray(frame_particles[t0][r0], dtype=np.float64)
            p1 = np.asarray(frame_particles[t1][r1], dtype=np.float64)
            shifts.append(float(np.linalg.norm(p1 - p0)))
        if not shifts:
            return False
        shift = float(np.median(shifts))
        return shift <= ratio * spacing

    def _run_pass(
        self,
        frame_particles: list[np.ndarray],
        frame_leaves: list[np.ndarray] | None = None,
        project_fn: Any = None,
        return_chains: bool = False,
        eff_vmax: float | None = None,
        frame_seen: list[np.ndarray] | None = None,
        frame_ghost: list[np.ndarray] | None = None,
        frame_logb: list[np.ndarray] | None = None,
        init_vels: np.ndarray | None = None,
    ) -> (
        list[tuple[int, int, int, int]]
        | tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]
    ):
        num_frames = len(frame_particles)
        if num_frames < 2:
            return ([], []) if return_chains else []

        if eff_vmax is None:
            eff_vmax = self.cfg.v_max

        if frame_leaves is None:
            frame_leaves = [np.zeros((len(p), 0)) for p in frame_particles]

        cost_mode = self.cfg.cost_mode
        if cost_mode == "projected" and project_fn is None:
            cost_mode = "3d"

        next_tid = 0
        # tid -> dict(pos, vel, last_t, last_row, misses, shared_streak)
        # shared_streak counts consecutive shared observations; velocity is
        # NEVER updated from a shared point (each track's speed comes only
        # from its own points).
        tracks: dict[int, dict] = {}
        # tid -> list of (frame, row, is_shared): full point history, kept
        # for retired tracks too (row-links cannot represent sharing).
        hist: dict[int, list[tuple[int, int, bool]]] = {}
        # (frame_idx, row) -> tid, for emitting row-based links
        loc2tid: dict[tuple[int, int], int] = {}
        pts0_init = np.asarray(frame_particles[0], dtype=np.float64)
        has_init_vel = init_vels is not None
        if init_vels is None:
            init_vels = np.zeros((len(pts0_init), 3), dtype=np.float64)
        else:
            init_vels = np.asarray(init_vels, dtype=np.float64)
            if init_vels.shape != (len(pts0_init), 3):
                init_vels = np.zeros((len(pts0_init), 3), dtype=np.float64)
            has_init_vel = bool(np.any(init_vels != 0.0))

        for i, p in enumerate(pts0_init):
            tracks[next_tid] = {
                "pos": p.copy(),
                "vel": init_vels[i].copy(),
                "last_t": 0,
                "last_row": i,
                "misses": 0,
                "shared_streak": 0,
            }
            loc2tid[(0, i)] = next_tid
            hist[next_tid] = [(0, i, False)]
            next_tid += 1

        all_links = []
        for t in range(num_frames - 1):
            pts1 = np.asarray(frame_particles[t + 1], dtype=np.float64)
            lf1 = frame_leaves[t + 1]
            n1 = len(pts1)

            # Active tracks: seen within max_gap frames.
            active = [
                tid
                for tid, tr in tracks.items()
                if t + 1 - tr["last_t"] <= self.cfg.max_gap
            ]
            if not active or n1 == 0:
                pred_pts = np.zeros((0, 3))
                pred_xy = np.zeros((0, lf1.shape[1] if n1 else 0))
                tids: list[int] = []
            else:
                steps = np.array(
                    [t + 1 - tracks[tid]["last_t"] for tid in active], dtype=np.float64
                )
                if self.cfg.use_velocity:
                    pred_pts = np.array(
                        [
                            tracks[tid]["pos"]
                            + tracks[tid]["vel"] * steps[k] * self.cfg.dt
                            for k, tid in enumerate(active)
                        ]
                    )
                else:
                    pred_pts = np.array([tracks[tid]["pos"] for tid in active])
                if cost_mode == "projected":
                    pred_xy = np.asarray(project_fn(pred_pts))
                else:
                    pred_xy = np.zeros((len(active), 0))
                tids = active

            # Predictions are re-projected everywhere: fully "seen". Only the
            # detection side needs a real mask (bug 2: NaN no longer marks it).
            n_leaf_cams = np.asarray(lf1).shape[1] // 2 if n1 else 0
            pred_seen = np.ones((len(tids), n_leaf_cams), dtype=bool)
            det_seen = None
            if frame_seen is not None and frame_seen[t + 1] is not None:
                det_seen = np.asarray(frame_seen[t + 1], dtype=bool)

            # Option A: Compute spatial consensus velocity field if enabled
            consensus_v = None
            consensus_disp = None
            consensus_valid = None
            orig_pts = None
            dt_steps_arr = None
            if self.cfg.vfc_mode in ("pre", "both") and tids:
                orig_pts = np.array(
                    [tracks[tid]["pos"] for tid in tids], dtype=np.float64
                )
                dt_steps_arr = np.array(
                    [(t + 1 - tracks[tid]["last_t"]) * self.cfg.dt for tid in tids],
                    dtype=np.float64,
                )
                # A frame-0 track can serve as a consensus reference only
                # when it really carries a velocity (bootstrapped), not when
                # it still sits at rest.
                min_len = 1 if has_init_vel else 2
                est_tids = [tid for tid in active if len(hist[tid]) >= min_len]
                if len(est_tids) >= self.cfg.vfc_min_neighbors:
                    est_pos = np.array(
                        [tracks[tid]["pos"] for tid in est_tids], dtype=np.float64
                    )
                    est_vel = np.array(
                        [tracks[tid]["vel"] for tid in est_tids], dtype=np.float64
                    )
                    consensus_v, consensus_disp, consensus_valid = (
                        compute_local_consensus_field(
                            est_pos,
                            est_vel,
                            orig_pts,
                            radius=self.cfg.vfc_radius,
                            min_neighbors=self.cfg.vfc_min_neighbors,
                            use_affine=self.cfg.vfc_affine,
                            affine_min_neighbors=self.cfg.vfc_affine_min_neighbors,
                        )
                    )

            got, got_shared = _match_two_phase_frame(
                np.asarray(pred_pts, dtype=np.float64),
                np.asarray(pts1, dtype=np.float64),
                np.asarray(pred_xy, dtype=np.float64),
                np.asarray(lf1, dtype=np.float64),
                np.arange(len(tids), dtype=np.int32),
                np.arange(n1, dtype=np.int32),
                eff_vmax,
                self.cfg.leaf_weight,
                cost_mode=cost_mode,
                allow_shared=self.cfg.allow_shared,
                share_tol=self.cfg.share_tol,
                seen0=pred_seen,
                seen1=det_seen,
                ghost1=None if frame_ghost is None else frame_ghost[t + 1],
                q_weight=self.cfg.q_weight,
                young0=(
                    np.array([len(hist[tid]) < self.cfg.q_young for tid in tids])
                    if frame_ghost is not None and self.cfg.q_young > 0 and tids
                    else None
                ),
                q_block=1.0 if self.cfg.q_seed is None else self.cfg.q_seed,
                logb0=(
                    np.array(
                        [
                            frame_logb[tracks[tid]["last_t"]][tracks[tid]["last_row"]]
                            for tid in tids
                        ]
                    )
                    if frame_logb is not None and self.cfg.blob_gate and tids
                    else None
                ),
                logb1=frame_logb[t + 1] if frame_logb is not None else None,
                blob_gate=self.cfg.blob_gate,
                consensus_v=consensus_v,
                consensus_disp=consensus_disp,
                consensus_valid=consensus_valid,
                vfc_gate=self.cfg.vfc_gate,
                vfc_weight=self.cfg.vfc_weight,
                pts0_orig=orig_pts,
                dt_steps=dt_steps_arr,
            )

            # Option B (Post-Assignment Fast-VFC outlier rejection)
            rejected_dets: set[int] = set()
            if (
                self.cfg.vfc_mode in ("post", "both")
                and len(got) >= self.cfg.vfc_min_neighbors
            ):
                got, rejected_dets = filter_links_by_spatial_consensus(
                    got,
                    tids,
                    tracks,
                    pts1,
                    self.cfg.dt,
                    t + 1,
                    radius=self.cfg.vfc_radius,
                    min_neighbors=self.cfg.vfc_min_neighbors,
                    post_k=self.cfg.vfc_post_k,
                    min_tol=self.cfg.vfc_post_min_tol,
                )
                if got_shared:
                    got_shared = {pair for pair in got_shared if pair in got}

            matched_det = set()
            for ai, det in got:
                tid = tids[int(ai)]
                tr = tracks[tid]
                gap = t + 1 - tr["last_t"]
                old_pos = tr["pos"]
                if (int(ai), int(det)) in got_shared and tr.get(
                    "shared_streak", 0
                ) < self.cfg.max_shared:
                    # Shared observation: follow the point, keep own speed.
                    tr["pos"] = pts1[det].copy()
                    tr["shared_streak"] = tr.get("shared_streak", 0) + 1
                    is_shared = True
                else:
                    tr["vel"] = (pts1[det] - old_pos) / (gap * self.cfg.dt)
                    tr["pos"] = pts1[det].copy()
                    tr["shared_streak"] = 0
                    is_shared = False
                all_links.append((tr["last_t"], tr["last_row"], t + 1, det))
                tr["last_t"] = t + 1
                tr["last_row"] = int(det)
                tr["misses"] = 0
                hist[tid].append((t + 1, int(det), is_shared))
                loc2tid[(t + 1, int(det))] = tid
                matched_det.add(int(det))

            # Age every unseen track (including ones already outside the
            # active window) and retire the exhausted.
            for tid in list(tracks.keys()):
                if tracks[tid]["last_t"] <= t:
                    tracks[tid]["misses"] += 1
                    if tracks[tid]["misses"] > self.cfg.max_gap:
                        del tracks[tid]

            # Cold start: unmatched detections become zero-velocity tracks.
            for det in range(n1):
                if det not in matched_det:
                    if (
                        self.cfg.vfc_mode in ("post", "both")
                        and self.cfg.vfc_post_reject_cold_start
                        and det in rejected_dets
                    ):
                        continue
                    if (
                        frame_ghost is not None
                        and self.cfg.q_seed is not None
                        and frame_ghost[t + 1][det] > self.cfg.q_seed
                    ):
                        continue  # doubtful point: may not start a trajectory
                    tracks[next_tid] = {
                        "pos": pts1[det].copy(),
                        "vel": np.zeros(3),
                        "last_t": t + 1,
                        "last_row": det,
                        "misses": 0,
                        "shared_streak": 0,
                    }
                    loc2tid[(t + 1, det)] = next_tid
                    hist[next_tid] = [(t + 1, det, False)]
                    next_tid += 1

        confirm_tol = self.cfg.confirm_tol
        if self.cfg.confirm_auto:
            st = _kink_statistics(
                all_links, frame_particles, radius=self.cfg.confirm_auto_radius
            )
            ok = (
                np.isfinite(st["median_kink"])
                and st["neighbours"] <= self.cfg.confirm_auto_max_neighbours
                and st["ratio"] >= self.cfg.confirm_auto_min_ratio
            )
            self.last_confirm_info = {**st, "auto": bool(ok)}
            if ok:
                confirm_tol = self.cfg.confirm_auto * st["median_kink"]
        self.last_confirm_tol = confirm_tol
        if not return_chains:
            if confirm_tol is not None:
                all_links, _ = _confirm_links(
                    all_links,
                    frame_particles,
                    confirm_tol,
                    self.cfg.confirm_ends,
                )
            return all_links
        if confirm_tol is not None:
            all_links, _sev = _confirm_links(
                all_links, frame_particles, confirm_tol, self.cfg.confirm_ends
            )
            hist = _split_hist(hist, _sev)
        chains = []
        fp_arr = [np.asarray(p, dtype=np.float64) for p in frame_particles]
        for tid, pts in hist.items():
            if len(pts) == 0:
                continue
            chains.append(
                {
                    "tid": tid,
                    "frames": [f for f, _, _ in pts],
                    "pos": np.array([fp_arr[f][r] for f, r, _ in pts]),
                    "shared": [s for _, _, s in pts],
                }
            )
        return all_links, chains


def build_project_fn(cals, cpar):
    """Re-project predicted 3D positions to leaf pixels through the camera models.

    Returns None when calibrations are unavailable; the tracker then
    falls back to 3D-distance costs (see ``cost_mode``).
    """
    try:
        cals = list(cals or [])
        if not cals or cpar is None:
            return None
        mm = cpar.mm
        imx, imy = float(cpar.imx), float(cpar.imy)
        pix_x, pix_y = float(cpar.pix_x), float(cpar.pix_y)

        from openptv2.algorithms.imgcoord import img_coord_batch

        def project_fn(pred):
            # Batched: one img_coord_batch call per camera for all points
            # (was a Python loop over points x cameras).
            pred = np.asarray(pred, dtype=np.float64)
            n = len(pred)
            nc = len(cals)
            xy = np.full((n, nc * 2), np.nan)
            if n == 0:
                return xy
            for ci, cal in enumerate(cals):
                m = np.asarray(img_coord_batch(pred, cal, mm))
                xy[:, 2 * ci] = m[:, 0] / pix_x + imx / 2
                xy[:, 2 * ci + 1] = imy / 2 - m[:, 1] / pix_y
            return np.nan_to_num(xy)

        # Smoke-test on one point so a broken model fails here, not
        # mid-run.
        project_fn(np.zeros((1, 3)))
        return project_fn
    except Exception as exc:
        print(f"TwoPhaseTracker: no projection ({exc}); falling back to 3D costs.")
        return None


def ghost_probabilities(frame_particles, rcm, n_seen, logb, q_model=DEFAULT_Q_MODEL):
    """Per-point ghost probability (A10) from per-frame ray miss distances, camera
    counts and (for the ``rcm_blob`` model) log blob brightness."""
    from openptv2.point_quality import (
        brightness_spread,
        fit_scale,
        ghost_probability,
    )

    scale = fit_scale(
        np.vstack(frame_particles), np.concatenate(rcm), np.concatenate(n_seen)
    )
    spreads = [None] * len(frame_particles)
    if q_model == "rcm_blob" and logb is not None:
        allb = np.vstack(logb)
        with np.errstate(all="ignore"):
            offsets = np.nanmedian(allb, axis=0)  # per-camera gain
            varies = float(np.nanstd(allb)) > 1e-6
        if varies:
            spreads = [brightness_spread(lb, offsets) for lb in logb]
        else:
            print(
                "TwoPhaseTracker: blob brightness is constant in this store; "
                "ghost marks use ray convergence only."
            )
    return [
        ghost_probability(p, r, n, scale, sp)
        for p, r, n, sp in zip(frame_particles, rcm, n_seen, spreads)
    ]


def q_model_of(track_cfg: dict) -> str:
    """The ghost model named by the ``track:`` section (validated)."""
    q_model = str(track_cfg.get("q_model", DEFAULT_Q_MODEL))
    if q_model not in ("rcm", "rcm_blob"):
        raise ValueError(f"q_model must be rcm or rcm_blob, got {q_model!r}")
    return q_model


def track_store(
    store: Any,
    frames: list[int],
    track_cfg: dict,
    cals: Any = None,
    cpar: Any = None,
    *,
    linkage_name: str = "ptv_is",
) -> list[tuple[int, int, int, int]]:
    """Two-phase tracking of the correspondences of ``store`` over ``frames``.

    Everything is read from the store (3D points, camera target ids, target
    rows) and every setting from ``track_cfg`` (the ``track:`` section of the
    parameter file). Writes the ghost probabilities (when the point-quality
    rules are on) and the linkage ``linkage_name`` back into the store, and
    returns the links. The two_phase plugin calls this; so can a script that
    re-tracks a store it built itself.
    """
    leaf_weight = float(track_cfg.get("leaf_weight", 1.0))
    v_max = float(track_cfg.get("v_max", track_cfg.get("dvxmax", 15.5)))
    use_velocity = bool(track_cfg.get("use_velocity", True))
    cost_mode = str(track_cfg.get("cost_mode", "projected"))
    max_gap = int(track_cfg.get("max_gap", 2))
    allow_shared = bool(track_cfg.get("allow_shared", False))
    max_shared = int(track_cfg.get("max_shared", 2))
    share_tol_raw = track_cfg.get("share_tol", 1.0)
    share_tol = None if share_tol_raw is None else float(share_tol_raw)
    max_group_size = int(track_cfg.get("max_group_size", 128))
    confirm_tol, confirm_ends, confirm_note = resolve_confirm(track_cfg, v_max)
    confirm_auto = resolve_confirm_auto(track_cfg)
    confirm_guard = {
        "confirm_auto_max_neighbours": float(
            track_cfg.get("confirm_auto_max_neighbours", AUTO_MAX_NEIGHBOURS)
        ),
        "confirm_auto_min_ratio": float(
            track_cfg.get("confirm_auto_min_ratio", AUTO_MIN_KINK_RATIO)
        ),
        "confirm_auto_radius": float(track_cfg.get("confirm_auto_radius", 5.0)),
    }
    if confirm_note:
        print(f"TwoPhaseTracker: {confirm_note}")
    bidirectional = bool(track_cfg.get("bidirectional", False))
    # Point-quality rules (A10, ray convergence). ON by default: a point whose
    # camera rays meet badly may not start a trajectory (q_seed) and a young
    # trajectory may not continue onto it (q_young). Switch off with
    # ``q_seed: null`` and ``q_young: 0``. See docs/tracking_quality.md.
    bg_raw = track_cfg.get("blob_gate", None)
    blob_gate = None if bg_raw is None else float(bg_raw)
    q_weight = float(track_cfg.get("q_weight", 0.0))
    q_seed_raw = track_cfg.get("q_seed", DEFAULT_Q_SEED)
    q_seed = None if q_seed_raw is None else float(q_seed_raw)
    q_young = int(track_cfg.get("q_young", DEFAULT_Q_YOUNG))
    q_model = q_model_of(track_cfg)
    bwd_v_max_raw = track_cfg.get("bwd_v_max", None)
    bwd_v_max = None if bwd_v_max_raw is None else float(bwd_v_max_raw)
    vfc_mode = str(track_cfg.get("vfc_mode", "off"))
    vfc_radius = float(track_cfg.get("vfc_radius", 10.0))
    vfc_min_neighbors = int(track_cfg.get("vfc_min_neighbors", 3))
    vfc_gate_raw = track_cfg.get("vfc_gate", None)
    vfc_gate = None if vfc_gate_raw is None else float(vfc_gate_raw)
    vfc_weight = float(track_cfg.get("vfc_weight", 0.0))
    vfc_post_k = float(track_cfg.get("vfc_post_k", 3.0))
    vfc_post_min_tol = float(track_cfg.get("vfc_post_min_tol", 0.5))
    vfc_post_reject_cold_start = bool(track_cfg.get("vfc_post_reject_cold_start", True))
    vfc_bootstrap = bool(track_cfg.get("vfc_bootstrap", False))
    vfc_bootstrap_grid_size = float(track_cfg.get("vfc_bootstrap_grid_size", 20.0))
    vfc_bootstrap_passes = int(track_cfg.get("vfc_bootstrap_passes", 2))
    vfc_bootstrap_shift_ratio = float(track_cfg.get("vfc_bootstrap_shift_ratio", 0.5))
    vfc_bootstrap_max_neighbors = int(track_cfg.get("vfc_bootstrap_max_neighbors", 32))
    vfc_affine = bool(track_cfg.get("vfc_affine", False))
    vfc_affine_min_neighbors = int(track_cfg.get("vfc_affine_min_neighbors", 6))

    # Infer num_cams from correspondences shape: (N, 3+C) where C = num_cams
    corr_shape = store.root[f"correspondences/frame_{frames[0]:06d}"].shape
    num_cams = corr_shape[1] - 3

    # Point-quality rules need the camera models; without them they are off.
    want_q = q_weight > 0 or q_seed is not None or q_young > 0
    cals = list(cals or [])
    if want_q and not (cals and cpar):
        print(
            "TwoPhaseTracker: no calibrations on the experiment; "
            "point-quality rules (q_seed/q_young) are off."
        )
        q_weight, q_seed, q_young, want_q = 0.0, None, 0, False
    need_logb = bool(blob_gate) or (want_q and q_model == "rcm_blob")

    from openptv2.point_quality import (
        log_brightness_from_arrays,
        rcm_from_arrays,
        read_frame_arrays,
    )
    from openptv2.storage.run_store import seen_mask

    frame_particles = []
    frame_leaves = []
    frame_seen = []
    rcm_list, nseen_list, logb_list = [], [], []
    for f in frames:
        # every array of the frame is read ONCE (zarr reads were half the run time)
        pos_3d, cam_ids, rows = read_frame_arrays(store, f, num_cams)
        pos_3d = np.asarray(pos_3d)
        frame_particles.append(pos_3d)

        n = len(pos_3d)
        xy = np.full((n, num_cams, 2), np.nan)
        for c in range(num_cams):
            if rows[c] is not None:
                valid = cam_ids[:, c] >= 0
                xy[valid, c] = rows[c][cam_ids[valid, c], 1:3]
        frame_leaves.append(np.nan_to_num(xy.reshape(n, -1)))
        # Bug 2 fix: the seen-mask is authoritative; NaN no longer marks
        # "missing" (any nan_to_num would silently un-mark it). Single
        # definition: run_store.seen_mask -- do not re-derive per reader.
        seen = seen_mask(cam_ids)
        frame_seen.append(seen)
        if want_q:
            rcm_list.append(rcm_from_arrays(pos_3d, cam_ids, rows, cals, cpar))
            nseen_list.append(seen.sum(axis=1))
        if need_logb:
            logb_list.append(log_brightness_from_arrays(cam_ids, rows))

    cfg = TwoPhaseTrackerConfig(
        v_max=v_max,
        blob_gate=blob_gate,
        q_weight=q_weight,
        q_seed=q_seed,
        q_young=q_young,
        leaf_weight=leaf_weight,
        use_velocity=use_velocity,
        cost_mode=cost_mode,
        max_gap=max_gap,
        allow_shared=allow_shared,
        max_shared=max_shared,
        share_tol=share_tol,
        max_group_size=max_group_size,
        confirm_tol=confirm_tol,
        confirm_ends=confirm_ends,
        confirm_auto=confirm_auto,
        **confirm_guard,
        bidirectional=bidirectional,
        parallel_directions=bool(track_cfg.get("parallel_directions", False)),
        bwd_v_max=bwd_v_max,
        vfc_mode=vfc_mode,
        vfc_radius=vfc_radius,
        vfc_min_neighbors=vfc_min_neighbors,
        vfc_gate=vfc_gate,
        vfc_weight=vfc_weight,
        vfc_post_k=vfc_post_k,
        vfc_post_min_tol=vfc_post_min_tol,
        vfc_post_reject_cold_start=vfc_post_reject_cold_start,
        vfc_bootstrap=vfc_bootstrap,
        vfc_bootstrap_grid_size=vfc_bootstrap_grid_size,
        vfc_bootstrap_passes=vfc_bootstrap_passes,
        vfc_bootstrap_shift_ratio=vfc_bootstrap_shift_ratio,
        vfc_bootstrap_max_neighbors=vfc_bootstrap_max_neighbors,
        vfc_affine=vfc_affine,
        vfc_affine_min_neighbors=vfc_affine_min_neighbors,
    )
    tracker = TwoPhaseTracker(cfg)
    project_fn = build_project_fn(cals, cpar)
    frame_logb = logb_list if blob_gate else None
    frame_ghost = None
    if want_q:
        frame_ghost = ghost_probabilities(
            frame_particles, rcm_list, nseen_list, logb_list or None, q_model
        )
        store.write_point_quality_many(frames, frame_ghost)
    links = cast(
        list[tuple[int, int, int, int]],
        tracker.track_frames(
            frame_particles,
            frame_leaves,
            project_fn=project_fn,
            frame_seen=frame_seen,
            frame_ghost=frame_ghost,
            frame_logb=frame_logb,
        ),
    )

    # Per-step progress like trackcorr (track3d step: curr/next/links)
    from collections import Counter

    _cnt = Counter(t0 for t0, _, _, _ in links)
    for _i in range(len(frames) - 1):
        _curr = len(frame_particles[_i])
        _nxt = len(frame_particles[_i + 1])
        _links = _cnt.get(_i, 0)
        print(
            f"two_phase step: {frames[_i]}, curr: {_curr}, next: {_nxt}, links: {_links}"
        )
    if frames:
        _avg_parts = sum(len(p) for p in frame_particles) / len(frame_particles)
        _avg_links = len(links) / max(len(frames) - 1, 1)
        print(
            f"Average over sequence, particles: {_avg_parts:.1f}, links: {_avg_links:.1f}, lost: {_avg_parts - _avg_links:.1f}"
        )

    linkage = _links_to_linkage(links, frames, [len(p) for p in frame_particles])
    store.write_linkage_many(frames, linkage, frame_particles, name=linkage_name)

    print(
        f"TwoPhaseTracker: {len(links)} links across {len(frames)} frames "
        f"(leaf_weight={leaf_weight}, v_max={v_max}, "
        f"use_velocity={use_velocity}, cost_mode={cost_mode}, "
        f"max_gap={max_gap}, "
        f"project_fn={'yes' if project_fn is not None else 'no'})"
    )
    return links


class Tracking:
    """Plugin interface for the GUI/batch pipeline.

    Reads ``leaf_weight`` and ``dvxmax`` (as v_max) from the experiment's
    ``track`` parameter section, loads 3D positions + 2D leaves from the
    RunStore, runs two-phase matching, and writes links back
    (:func:`track_store`).
    """

    q_model = DEFAULT_Q_MODEL

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def _ghost_probabilities(self, frame_particles, rcm, n_seen, logb):
        return ghost_probabilities(frame_particles, rcm, n_seen, logb, self.q_model)

    def _build_project_fn(self):
        return build_project_fn(
            getattr(self.exp, "cals", None), getattr(self.exp, "cpar", None)
        )

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)

        track_cfg = pm.parameters.get("track", {}) if pm else {}
        self.q_model = q_model_of(track_cfg)

        store = getattr(self.exp, "_store", None)
        if store is None:
            from openptv2.storage import RunStore
            from openptv2.storage.run_store import find_existing_store

            zarr_path = find_existing_store(Path.cwd())
            if zarr_path is not None:
                store = RunStore(zarr_path, mode="a")
            else:
                print(
                    "TwoPhaseTracker requires a RunStore (zarr) with "
                    "correspondences and targets. Falling back to default tracker."
                )
                tracker = self.ptv.py_trackcorr_init(self.exp)
                self.exp.tracker = tracker
                tracker.full_forward()
                return

        frames = sorted(store.frames())

        # Honor batch requested range if available via exp.spar
        _seq_first = _seq_last = None
        _spar = getattr(self.exp, "spar", None) or getattr(self.exp, "seq_par", None)
        if _spar is not None:
            try:
                _seq_first = int(_spar.get_first())
                _seq_last = int(_spar.get_last())
            except Exception:
                try:
                    _seq_first = int(getattr(_spar, "first"))
                    _seq_last = int(getattr(_spar, "last"))
                except Exception:
                    pass
        if _seq_first is None:
            _seq_cfg = pm.parameters.get("sequence", {}) if pm else {}
            if "first" in _seq_cfg and "last" in _seq_cfg:
                _seq_first, _seq_last = int(_seq_cfg["first"]), int(_seq_cfg["last"])
        if _seq_first is not None and _seq_last is not None:
            frames = [f for f in frames if _seq_first <= f <= _seq_last]

        # Only process the first contiguous block
        if len(frames) > 1:
            contiguous = [frames[0]]
            for i in range(1, len(frames)):
                if frames[i] == contiguous[-1] + 1:
                    contiguous.append(frames[i])
                else:
                    break
            frames = contiguous

        if not frames:
            print(
                "TwoPhaseTracker: no frames to process, falling back to default tracker."
            )
            tracker = self.ptv.py_trackcorr_init(self.exp)
            self.exp.tracker = tracker
            tracker.full_forward()
            return

        track_store(
            store,
            frames,
            track_cfg,
            getattr(self.exp, "cals", None),
            getattr(self.exp, "cpar", None),
        )

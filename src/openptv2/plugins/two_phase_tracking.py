"""Two-phase tracking: 3D search + 2D leaf ranking.

Phase 1: 3D KD-tree finds candidate matches within a search radius.
Phase 2: 2D pixel distances become the cost matrix; Hungarian assignment
         within connected components picks the best match globally.

This exploits the tree-forest architecture: 3D positions are the "trunk"
(structural search), per-camera 2D leaf positions are the "signature"
(disambiguation when 3D is noisy or ambiguous).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


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
    bidirectional : bool
        Run forward tracking, backward tracking on reversed frames, and
        merge the two sets (reciprocal-first core, non-conflicting links
        added greedily by 3D distance). Closes ~75% of the accuracy gap
        to 4-frame trackcorr on dense data in a fraction of the time.
        Default False (unidirectional forward).
    bwd_v_max : float | None
        Optional search radius for the backward pass in bidirectional mode.
        None defaults to v_max. Setting a slightly wider bwd_v_max (e.g. 2.5
        when v_max=2.0) allows backward tracking to reach fast particles that
        forward missed, safely protected by the Forward-First lock.
    """

    v_max: float = 5.0
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
    bidirectional: bool = False
    bwd_v_max: float | None = None


def _confirm_links(
    links: list[tuple[int, int, int, int]],
    frame_particles: list[np.ndarray],
    tol: float | None,
    ends: bool,
) -> tuple[list[tuple[int, int, int, int]], set]:
    """Two-hop confirmation post-pass over consecutive links.

    A consecutive link (t0,r0)->(t1,r1) survives iff a consecutive onward
    link from (t1,r1) continues within velocity kink ``tol`` (mm/frame).
    Links into the last frame cannot be judged and are kept; gap links
    pass through untouched (gap logic already decided them). With ``ends``,
    dead-end consecutive links (no onward link, not last frame) are also
    severed -- a dying track grabbing a stranger. Severing fragments
    chains; rejoining them is gap-relink/repair's job, not assembly's.

    Returns (kept_links, severed) with severed a set of node pairs.
    """
    last_t = len(frame_particles) - 1
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
                if kink <= tol:
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

    # Build edge list with 2D costs
    rows: list[int] = []
    cols: list[int] = []
    costs: list[float] = []
    use_leaves = cost_mode == "projected" and leaf_weight > 0 and xy0.shape[1] > 0
    for pi in range(n_pred):
        cands = neighbours[pi]
        if len(cands) == 0:
            continue
        if use_leaves:
            # 2D cost: mean Euclidean distance per camera, weighted by overlap count
            C = xy0.shape[1] // 2
            xy0_cam = xy0[pi].reshape(C, 2)
            d2d = np.zeros(len(cands))
            for ci_idx, ci in enumerate(cands):
                xy1_cam = xy1[ci].reshape(C, 2)
                valid = ~np.isnan(xy0_cam[:, 0]) & ~np.isnan(xy1_cam[:, 0])
                n_valid = valid.sum()
                if n_valid > 0:
                    cam_dists = np.linalg.norm(xy0_cam[valid] - xy1_cam[valid], axis=1)
                    # Weight: more shared cameras = more reliable distance
                    d2d[ci_idx] = cam_dists.mean() * (C / n_valid)
                else:
                    d2d[ci_idx] = 1e6
            for ci_idx, ci in enumerate(cands):
                rows.append(pi)
                cols.append(ci)
                costs.append(d2d[ci_idx] * leaf_weight)
        else:
            # Fallback: 3D distance
            for ci in cands:
                rows.append(pi)
                cols.append(ci)
                costs.append(np.linalg.norm(pts0[pi] - pts1[ci]))

    if len(rows) == 0:
        return set(), set()

    row_arr = np.array(rows)
    col_arr = np.array(cols)
    cost_arr = np.array(costs)

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
                frame_particles, frame_leaves, project_fn, return_chains
            )

        # Bidirectional tracking: forward + backward on reversed frames
        fwd_links = cast(
            list[tuple[int, int, int, int]],
            self._track_unidirectional(
                frame_particles, frame_leaves, project_fn, return_chains=False
            ),
        )

        rev_particles = frame_particles[::-1]
        rev_leaves = frame_leaves[::-1] if frame_leaves is not None else None
        bwd_vmax = (
            self.cfg.bwd_v_max if self.cfg.bwd_v_max is not None else self.cfg.v_max
        )
        bwd_links_raw = cast(
            list[tuple[int, int, int, int]],
            self._track_unidirectional(
                rev_particles,
                rev_leaves,
                project_fn,
                return_chains=False,
                v_max_override=bwd_vmax,
            ),
        )
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
    ) -> (
        list[tuple[int, int, int, int]]
        | tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]
    ):
        num_frames = len(frame_particles)
        if num_frames < 2:
            return ([], []) if return_chains else []

        eff_vmax = v_max_override if v_max_override is not None else self.cfg.v_max

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
        for i, p in enumerate(frame_particles[0]):
            tracks[next_tid] = {
                "pos": np.asarray(p, dtype=np.float64),
                "vel": np.zeros(3),
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
            )
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

        if not return_chains:
            if self.cfg.confirm_tol is not None:
                all_links, _ = _confirm_links(
                    all_links,
                    frame_particles,
                    self.cfg.confirm_tol,
                    self.cfg.confirm_ends,
                )
            return all_links
        if self.cfg.confirm_tol is not None:
            all_links, _sev = _confirm_links(
                all_links, frame_particles, self.cfg.confirm_tol, self.cfg.confirm_ends
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


class Tracking:
    """Plugin interface for the GUI/batch pipeline.

    Reads ``leaf_weight`` and ``dvxmax`` (as v_max) from the experiment's
    ``track`` parameter section, loads 3D positions + 2D leaves from the
    RunStore, runs two-phase matching, and writes links back.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def _build_project_fn(self):
        """Re-project predicted 3D positions to leaf pixels via exp cals.

        Returns None when calibrations are unavailable; the tracker then
        falls back to 3D-distance costs (see ``cost_mode``).
        """
        try:
            cals = list(getattr(self.exp, "cals", None) or [])
            cpar = getattr(self.exp, "cpar", None)
            if not cals or cpar is None:
                return None
            mm = cpar.mm
            imx, imy = float(cpar.imx), float(cpar.imy)
            pix_x, pix_y = float(cpar.pix_x), float(cpar.pix_y)

            from openptv2.algorithms.imgcoord import img_coord_batch

            def project_fn(pred):
                pred = np.asarray(pred, dtype=np.float64)
                n = len(pred)
                nc = len(cals)
                xy = np.full((n, nc * 2), np.nan)
                for i in range(n):
                    for ci, cal in enumerate(cals):
                        m = img_coord_batch(pred[i : i + 1], cal, mm)[0]
                        xy[i, 2 * ci] = m[0] / pix_x + imx / 2
                        xy[i, 2 * ci + 1] = imy / 2 - m[1] / pix_y
                return np.nan_to_num(xy)

            # Smoke-test on one point so a broken model fails here, not
            # mid-run.
            project_fn(np.zeros((1, 3)))
            return project_fn
        except Exception as exc:
            print(f"TwoPhaseTracker: no projection ({exc}); falling back to 3D costs.")
            return None

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)

        track_cfg = pm.parameters.get("track", {}) if pm else {}
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
        confirm_raw = track_cfg.get("confirm_tol", None)
        confirm_tol = None if confirm_raw is None else float(confirm_raw)
        confirm_ends = bool(track_cfg.get("confirm_ends", False))
        bidirectional = bool(track_cfg.get("bidirectional", False))
        bwd_v_max_raw = track_cfg.get("bwd_v_max", None)
        bwd_v_max = None if bwd_v_max_raw is None else float(bwd_v_max_raw)

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
        # Infer num_cams from correspondences shape: (N, 3+C) where C = num_cams
        first_frame = frames[0]
        corr_shape = store.root[f"correspondences/frame_{first_frame:06d}"].shape
        num_cams = corr_shape[1] - 3

        frame_particles = []
        frame_leaves = []
        for f in frames:
            pos_3d, _ = store.read_correspondences(f)
            pos_3d = np.asarray(pos_3d)
            frame_particles.append(pos_3d)

            n = len(pos_3d)
            cam_ids = np.asarray(store.root[f"correspondences/frame_{f:06d}"])[
                :, 3:
            ].astype(int)
            xy = np.full((n, num_cams, 2), np.nan)
            for c in range(num_cams):
                key = f"targets/cam_{c}/frame_{f:06d}"
                if key in store.root:
                    t = np.asarray(store.root[key])
                    valid = cam_ids[:, c] >= 0
                    xy[valid, c] = t[cam_ids[valid, c], 1:3]
            frame_leaves.append(np.nan_to_num(xy.reshape(n, -1)))

        cfg = TwoPhaseTrackerConfig(
            v_max=v_max,
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
            bidirectional=bidirectional,
            bwd_v_max=bwd_v_max,
        )
        tracker = TwoPhaseTracker(cfg)
        project_fn = self._build_project_fn()
        links = cast(
            list[tuple[int, int, int, int]],
            tracker.track_frames(frame_particles, frame_leaves, project_fn=project_fn),
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

        # Build prev/next arrays per frame from links
        from collections import defaultdict

        nxt_map: defaultdict[int, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(lambda: -1)
        )
        prv_map: defaultdict[int, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(lambda: -1)
        )
        for t0, p0, t1, p1 in links:
            f0, f1 = frames[t0], frames[t1]
            nxt_map[f0][p0] = p1
            prv_map[f1][p1] = p0

        for i, f in enumerate(frames):
            n = len(frame_particles[i])
            nxt = np.full(n, -1, dtype=np.int32)
            prv = np.full(n, -1, dtype=np.int32)
            for p0, p1 in nxt_map[f].items():
                if p0 < n and p1 < n:
                    nxt[p0] = p1
            for p1, p0 in prv_map[f].items():
                if p1 < n and p0 < n:
                    prv[p1] = p0
            store.write_linkage(f, prv, nxt, frame_particles[i], name="ptv_is")

        print(
            f"TwoPhaseTracker: {len(links)} links across {len(frames)} frames "
            f"(leaf_weight={leaf_weight}, v_max={v_max}, "
            f"use_velocity={use_velocity}, cost_mode={cost_mode}, "
            f"max_gap={max_gap}, "
            f"project_fn={'yes' if project_fn is not None else 'no'})"
        )

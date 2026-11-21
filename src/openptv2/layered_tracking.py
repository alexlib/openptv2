"""Layered / 2-Pass Clean-and-Claim tracking (Milestone K + B7).

Implements the 2-Pass Clean-and-Claim strategy (``docs/plans/unified-tracker-implementation.md``
§12.5) plus conflict-component Hungarian matching (B7) and K1 quality-layering primitives:

1. **Pass 1 (Tight-Core):** Filter correspondences to a high-purity core
   (high camera support / low ghost probability), run bidirectional tracking
   (``full_forward`` + ``full_backward`` + ``enforce_reciprocity``), and keep
   only verified core tracks (length >= ``min_core_length``).
2. **Deterministic 2D-target pruning (``prune_claimed_correspondences``):**
   Mark all 2D target IDs claimed by Pass-1 tracks and drop any remaining 3D
   correspondence that references a claimed 2D target (O(N) mask, no
   re-triangulation). Ghosts that stole a 2D target from a real particle
   vanish deterministically.
3. **Pass 2 (Residual + ``track_assisted`` + Postprocess):** Track the
   ghost-depleted residual cloud, merge Pass-1 and Pass-2 linkage in
   ``RunStore``, run ``track_assisted.run_corrective_pass`` (3-4 pt polynomial
   predictor + optional shared targets), and run final ``tracking_postprocess``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE
from openptv2.ghost_filter import camera_support
from openptv2.track_assisted import CorrectiveStats, run_corrective_pass
from openptv2.tracker import Tracker, default_naming
from openptv2.tracking_postprocess import (
    count_links,
    enforce_reciprocity,
    link_step,
    prune_kinky_links,
    read_linkage,
    relink_trajectory_gaps,
)

# ---------------------------------------------------------------------------
# K1 quality-layering primitives
# ---------------------------------------------------------------------------


def quality_scores(cam_target_ids: np.ndarray) -> np.ndarray:
    """Per-correspondence quality (higher = more trustworthy).

    Default: camera support count. When ghost probabilities are available,
    pass them in via ``peel_layers``'s ``score_fn`` override.
    """
    return camera_support(cam_target_ids).astype(float)


def assign_layers(
    scores: np.ndarray, n_layers: int = 3, quantiles: list[float] | None = None
) -> np.ndarray:
    """Assign each correspondence to a layer index (0..n_layers-1).

    Layer 0 = highest quality. Quantiles define cumulative cutoffs.
    """
    if quantiles is None:
        quantiles = [i / n_layers for i in range(1, n_layers)] + [1.0]
    if len(quantiles) != n_layers:
        raise ValueError("quantiles length must equal n_layers")
    order = np.argsort(scores)[::-1]  # highest first
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(len(scores))
    cutoffs = (np.array(quantiles) * len(scores)).astype(int)
    layer = np.zeros(len(scores), dtype=np.int32)
    prev = 0
    for i, c in enumerate(cutoffs):
        layer[(ranks >= prev) & (ranks < c)] = i
        prev = c
    layer[ranks >= cutoffs[-1]] = n_layers - 1
    return layer


def peel_layers(
    store: Any,
    frames: list[int],
    n_layers: int = 3,
    quantiles: list[float] | None = None,
    score_fn=None,
) -> list[dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Split each frame's correspondences into ``n_layers`` quality tiers.

    Returns a list (one per layer) of dicts: frame -> (pos, cam_ids) for that
    layer's subset. Layers are ordered best-quality first.
    """
    layers: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [
        {} for _ in range(n_layers)
    ]
    for f in frames:
        if not store.has_correspondences(f):
            for layer_dict in layers:
                layer_dict[f] = (np.empty((0, 3)), np.empty((0, 0), dtype=np.int32))
            continue
        pos, cam = store.read_correspondences(f)
        if len(pos) == 0:
            for layer_dict in layers:
                layer_dict[f] = (pos, cam)
            continue
        scores = score_fn(cam) if score_fn is not None else quality_scores(cam)
        lyr = assign_layers(scores, n_layers, quantiles)
        for i in range(n_layers):
            mask = lyr == i
            layers[i][f] = (pos[mask], cam[mask])
    return layers


# ---------------------------------------------------------------------------
# B7: Conflict-component Hungarian resolution
# ---------------------------------------------------------------------------


def resolve_conflict_components(
    candidate_links: Sequence[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Resolve candidate ``(src_idx, dst_idx, cost)`` links between two frames.

    Keeps unambiguous 1-to-1 matches on the fast path, picks minimum-cost for
    1-to-many / many-to-1 stars, and solves ``scipy.optimize.linear_sum_assignment``
    on connected conflict components (``>= 2`` sources competing for ``>= 2``
    destinations).

    Returns matched ``(src_idx, dst_idx, cost)`` tuples sorted by ``(src_idx, dst_idx)``.
    """
    if not candidate_links:
        return []

    # Deduplicate (src, dst) keeping lowest cost.
    edge_cost: dict[tuple[int, int], float] = {}
    src_adj: dict[int, list[int]] = {}
    dst_adj: dict[int, list[int]] = {}
    for src, dst, cost in candidate_links:
        s = int(src)
        d = int(dst)
        c = float(cost)
        key = (s, d)
        if key not in edge_cost:
            edge_cost[key] = c
            src_adj.setdefault(s, []).append(d)
            dst_adj.setdefault(d, []).append(s)
        elif c < edge_cost[key]:
            edge_cost[key] = c

    visited_src: set[int] = set()
    visited_dst: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for start_src in sorted(src_adj):
        if start_src in visited_src:
            continue

        # BFS over the bipartite graph to isolate one connected component.
        comp_srcs: list[int] = []
        comp_dsts: list[int] = []
        queue: deque[tuple[str, int]] = deque([("s", start_src)])
        visited_src.add(start_src)

        while queue:
            kind, node = queue.popleft()
            if kind == "s":
                comp_srcs.append(node)
                for d in src_adj[node]:
                    if d not in visited_dst:
                        visited_dst.add(d)
                        queue.append(("d", d))
            else:
                comp_dsts.append(node)
                for s in dst_adj[node]:
                    if s not in visited_src:
                        visited_src.add(s)
                        queue.append(("s", s))

        # Fast path: unambiguous 1-to-1 match.
        if len(comp_srcs) == 1 and len(comp_dsts) == 1:
            s, d = comp_srcs[0], comp_dsts[0]
            matches.append((s, d, edge_cost[(s, d)]))
            continue

        # Star: 1 source -> multiple destinations.
        if len(comp_srcs) == 1:
            s = comp_srcs[0]
            best_d = min(comp_dsts, key=lambda d: edge_cost[(s, d)])
            matches.append((s, best_d, edge_cost[(s, best_d)]))
            continue

        # Star: multiple sources -> 1 destination.
        if len(comp_dsts) == 1:
            d = comp_dsts[0]
            best_s = min(comp_srcs, key=lambda s: edge_cost[(s, d)])
            matches.append((best_s, d, edge_cost[(best_s, d)]))
            continue

        # True conflict component (>= 2 sources vs >= 2 destinations): Hungarian.
        comp_srcs.sort()
        comp_dsts.sort()
        total_abs = sum(
            abs(edge_cost[(s, d)])
            for s in comp_srcs
            for d in src_adj[s]
        )
        big = max(1.0, total_abs) * 1e6 + 1e6
        cost_mat = np.full((len(comp_srcs), len(comp_dsts)), big, dtype=np.float64)
        src_pos = {s: i for i, s in enumerate(comp_srcs)}
        dst_pos = {d: j for j, d in enumerate(comp_dsts)}
        for s in comp_srcs:
            i = src_pos[s]
            for d in src_adj[s]:
                cost_mat[i, dst_pos[d]] = edge_cost[(s, d)]

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        for r, c in zip(row_ind, col_ind):
            s = comp_srcs[int(r)]
            d = comp_dsts[int(c)]
            if (s, d) in edge_cost:
                matches.append((s, d, edge_cost[(s, d)]))

    matches.sort(key=lambda item: (item[0], item[1]))
    return matches


# ---------------------------------------------------------------------------
# K2: Deterministic 2D-target pruning
# ---------------------------------------------------------------------------


def prune_frame_correspondences(
    pos: np.ndarray,
    cam_ids: np.ndarray,
    claimed_targets: Sequence[set[int] | np.ndarray] | dict[int, set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop 3D correspondences in one frame that reference any claimed 2D target.

    Returns ``(pruned_pos, pruned_cam_ids, keep_mask)`` in O(N) without
    re-triangulation.
    """
    n = len(pos)
    if n == 0 or cam_ids.size == 0:
        return pos, cam_ids, np.ones(n, dtype=bool)

    n_cams = cam_ids.shape[1]
    conflict = np.zeros(n, dtype=bool)
    for cam in range(n_cams):
        if isinstance(claimed_targets, dict):
            claimed = claimed_targets.get(cam)
        else:
            claimed = claimed_targets[cam] if cam < len(claimed_targets) else None
        if not claimed:
            continue
        col = cam_ids[:, cam]
        valid = col >= 0
        if not np.any(valid):
            continue
        claimed_arr = np.fromiter(claimed, dtype=np.int32)
        conflict[valid] |= np.isin(col[valid], claimed_arr)

    keep = ~conflict
    return pos[keep], cam_ids[keep], keep


def prune_claimed_correspondences(
    store: Any,
    frames: Sequence[int],
    claimed_by_frame: dict[int, Sequence[set[int]] | dict[int, set[int]]],
) -> dict[int, int]:
    """Prune correspondences in ``store`` referencing claimed 2D target IDs.

    Returns ``{frame: n_dropped}`` for each frame in ``frames``.
    """
    dropped: dict[int, int] = {}
    for f in frames:
        if not store.has_correspondences(f):
            dropped[f] = 0
            continue
        pos, cam_ids = store.read_correspondences(f)
        claimed = claimed_by_frame.get(f)
        if len(pos) == 0 or not claimed:
            dropped[f] = 0
            continue
        pruned_pos, pruned_ids, keep = prune_frame_correspondences(
            pos, cam_ids, claimed
        )
        n_drop = int((~keep).sum())
        if n_drop > 0:
            store.write_correspondences(f, pruned_pos, pruned_ids)
        dropped[f] = n_drop
    return dropped


# ---------------------------------------------------------------------------
# 2-Pass Clean-and-Claim orchestrator (K2 + K4 + K5)
# ---------------------------------------------------------------------------


@dataclass
class PeelStats:
    """Diagnostics from :func:`peel_and_track`."""

    min_core_cameras: int = 0
    core_points_total: int = 0
    verified_core_tracks: int = 0
    verified_core_points: int = 0
    pruned_correspondences: int = 0
    residual_points_total: int = 0
    links_after_pass1: int = 0
    links_after_pass2: int = 0
    links_final: int = 0
    corrective_stats: CorrectiveStats | None = None


def _choose_core_min_cameras(
    orig_ids: dict[int, np.ndarray],
    frames: list[int],
    num_cams: int,
    requested: int | None,
) -> int:
    if requested is not None:
        return max(2, int(requested))
    if num_cams <= 2:
        return num_cams
    total = 0
    full_cam = 0
    for f in frames:
        ids = orig_ids[f]
        if len(ids) == 0:
            continue
        total += len(ids)
        full_cam += int((camera_support(ids) >= num_cams).sum())
    if total > 0 and (full_cam / total) >= 0.10:
        return num_cams
    return max(2, num_cams - 1)


def _split_kinky_chain(
    chain: list[tuple[int, int]],
    core_pos: dict[int, np.ndarray],
    max_accel_err: float | None,
) -> list[list[tuple[int, int]]]:
    """Split or trim a reciprocal chain at kinematic velocity-jump kinks."""
    if len(chain) < 3 or max_accel_err is None or max_accel_err <= 0:
        return [chain]

    from openptv2.tracking_report import _is_velocity_jump

    def _triplet_is_kink(idx: int) -> bool:
        f0, r0 = chain[idx]
        f1, r1 = chain[idx + 1]
        f2, r2 = chain[idx + 2]
        p0 = core_pos[f0][r0]
        p1 = core_pos[f1][r1]
        p2 = core_pos[f2][r2]
        dt1 = max(1, f1 - f0)
        dt2 = max(1, f2 - f1)
        v1 = (p1 - p0) / float(dt1)
        v2 = (p2 - p1) / float(dt2)
        acc_err = float(np.linalg.norm(v2 - v1))
        return acc_err > float(max_accel_err) and _is_velocity_jump(
            p0, p1, p2, dt1, dt2, jump_angle_deg=90.0, jump_ratio=1.0
        )

    for idx in range(len(chain) - 2):
        if not _triplet_is_kink(idx):
            continue
        if idx == 0 and len(chain) >= 4 and not _triplet_is_kink(1):
            return _split_kinky_chain(chain[1:], core_pos, max_accel_err)
        if idx == len(chain) - 3 and len(chain) >= 4:
            return _split_kinky_chain(chain[:-1], core_pos, max_accel_err)
        left = _split_kinky_chain(chain[: idx + 1], core_pos, max_accel_err)
        right = _split_kinky_chain(chain[idx + 1 :], core_pos, max_accel_err)
        return left + right

    return [chain]


def _extract_verified_core(
    linkage_name: str,
    first: int,
    last: int,
    store: Any,
    core_pos: dict[int, np.ndarray],
    core_ids: dict[int, np.ndarray],
    num_cams: int,
    min_core_length: int,
    max_accel_err: float | None = None,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, list[set[int]]],
    int,
]:
    """Walk Pass-1 reciprocal chains and compactly extract tracks >= min_core_length."""
    frames = list(range(first, last + 1))
    link_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for f in frames:
        r = read_linkage(linkage_name, f, store=store)
        if r is not None:
            link_map[f] = r

    visited: set[tuple[int, int]] = set()
    tracks: list[list[tuple[int, int]]] = []

    for f in frames:
        if f not in link_map:
            continue
        prv, nxt, _xyz = link_map[f]
        for i in range(len(nxt)):
            if (f, i) in visited or prv[i] >= 0:
                continue
            chain = [(f, i)]
            visited.add((f, i))
            cur_f, cur_i = f, i
            while True:
                if cur_f not in link_map:
                    break
                j = int(link_map[cur_f][1][cur_i])
                if j < 0:
                    break
                step = link_step(
                    lambda m: link_map[m][0] if m in link_map else None,
                    cur_f,
                    cur_i,
                    j,
                )
                if not step:
                    break
                nxt_f = cur_f + step
                if (nxt_f, j) in visited:
                    break
                visited.add((nxt_f, j))
                chain.append((nxt_f, j))
                cur_f, cur_i = nxt_f, j
            if len(chain) >= min_core_length:
                for sub in _split_kinky_chain(chain, core_pos, max_accel_err):
                    if len(sub) >= min_core_length:
                        tracks.append(sub)

    # Assign compact per-frame indices 0..n_p1[f]-1 for verified core points.
    kept_rows: dict[int, list[int]] = {f: [] for f in frames}
    row_remap: dict[tuple[int, int], int] = {}
    for chain in tracks:
        for f, row in chain:
            if (f, row) not in row_remap:
                row_remap[(f, row)] = len(kept_rows[f])
                kept_rows[f].append(row)

    p1_pos: dict[int, np.ndarray] = {}
    p1_ids: dict[int, np.ndarray] = {}
    p1_prev: dict[int, np.ndarray] = {}
    p1_next: dict[int, np.ndarray] = {}
    claimed_by_frame: dict[int, list[set[int]]] = {}

    for f in frames:
        rows = kept_rows[f]
        n = len(rows)
        if n == 0:
            p1_pos[f] = np.empty((0, 3), dtype=np.float64)
            p1_ids[f] = np.empty((0, num_cams), dtype=np.int32)
            p1_prev[f] = np.empty(0, dtype=np.int32)
            p1_next[f] = np.empty(0, dtype=np.int32)
            claimed_by_frame[f] = [set() for _ in range(num_cams)]
            continue
        idx_arr = np.asarray(rows, dtype=np.int32)
        p1_pos[f] = core_pos[f][idx_arr]
        p1_ids[f] = core_ids[f][idx_arr]
        p1_prev[f] = np.full(n, PREV_NONE, dtype=np.int32)
        p1_next[f] = np.full(n, NEXT_NONE, dtype=np.int32)
        claimed_by_frame[f] = [
            {int(v) for v in p1_ids[f][:, c] if v >= 0} for c in range(num_cams)
        ]

    for chain in tracks:
        for k in range(len(chain) - 1):
            f0, r0 = chain[k]
            f1, r1 = chain[k + 1]
            c0 = row_remap[(f0, r0)]
            c1 = row_remap[(f1, r1)]
            p1_next[f0][c0] = c1
            p1_prev[f1][c1] = c0

    return p1_pos, p1_ids, p1_prev, p1_next, claimed_by_frame, len(tracks)


def peel_and_track(
    cpar: Any,
    vpar: Any,
    tpar: Any,
    spar: Any,
    cals: Any,
    store: Any,
    *,
    naming: dict[str, str] | None = None,
    min_core_length: int = 3,
    min_core_cameras: int | None = None,
    rcm_threshold: float | None = None,
    corrective_passes: int = 1,
    allow_shared_targets: bool = False,
    predictor: str = "poly",
    radius_px: float = 3.0,
    engine: str = "3d",
    tiers: int = 2,
) -> PeelStats:
    """Run Clean-and-Claim multi-pass peeling tracking on ``store``.

    1. **Core Pass(es):** Filter correspondences to high-purity core
       (``tiers=2`` runs 1 core pass; ``tiers=3`` on ``>=3``-cam rigs runs
       strict ``num_cams`` core then medium ``num_cams-1`` core), run
       bidirectional ``Tracker`` + ``enforce_reciprocity``, and extract
       kinematically smooth core tracks of length >= ``min_core_length``.
    2. **Prune:** Restore full correspondences and run
       :func:`prune_claimed_correspondences` using claimed 2D targets.
    3. **Residual + Post-Prune + ``track_assisted``:** Track the pruned
       residual cloud, prune unlinked residual rows conflicting with Pass-2
       linked targets, merge into ``store``, run gated
       :func:`openptv2.track_assisted.run_corrective_pass`, and run final
       ``relink_trajectory_gaps`` + ``enforce_reciprocity``.
    """
    if naming is None:
        naming = dict(default_naming)
    linkage_name = naming["linkage"].split("/")[-1]
    first, last = spar.get_first(), spar.get_last()
    frames = list(range(first, last + 1))
    num_cams = int(cpar.get_num_cams())
    eff_min_core_len = (
        min(int(min_core_length), len(frames))
        if len(frames) >= 2
        else int(min_core_length)
    )
    dacc_val = float(tpar.dacc) if hasattr(tpar, "dacc") and float(tpar.dacc) > 0 else None

    # Snapshot original correspondences and optional rcm / ghost probabilities.
    orig_pos: dict[int, np.ndarray] = {}
    orig_ids: dict[int, np.ndarray] = {}
    orig_qual: dict[int, np.ndarray | None] = {}
    for f in frames:
        if store.has_correspondences(f):
            pos, ids = store.read_correspondences(f)
            if len(pos) == 0:
                ids = np.empty((0, num_cams), dtype=np.int32)
        else:
            pos = np.empty((0, 3), dtype=np.float64)
            ids = np.empty((0, num_cams), dtype=np.int32)
        orig_pos[f] = np.asarray(pos, dtype=np.float64)
        orig_ids[f] = np.asarray(ids, dtype=np.int32).reshape(-1, num_cams)
        rcm_arr = store.read_rcm(f) if hasattr(store, "read_rcm") else None
        orig_qual[f] = rcm_arr if rcm_arr is not None else store.read_point_quality(f)

    core_min_cams = _choose_core_min_cameras(
        orig_ids, frames, num_cams, min_core_cameras
    )
    stats = PeelStats(min_core_cameras=core_min_cams)
    use_3d = str(engine).strip().lower() in ("3d", "track3d", "cython_3d", "priority_segment_3d")

    def _run_core_tier(
        src_pos: dict[int, np.ndarray],
        src_ids: dict[int, np.ndarray],
        min_cams_tier: int,
        min_len_tier: int,
        apply_rcm: bool,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, list[set[int]]],
        int,
        int,
    ]:
        c_pos: dict[int, np.ndarray] = {}
        c_ids: dict[int, np.ndarray] = {}
        for f in frames:
            pos, ids = src_pos[f], src_ids[f]
            if len(pos) == 0:
                c_pos[f] = pos
                c_ids[f] = ids
            else:
                mask = camera_support(ids) >= min_cams_tier
                if apply_rcm and rcm_threshold is not None:
                    qual = orig_qual[f]
                    if qual is not None and len(qual) == len(pos):
                        mask &= qual <= float(rcm_threshold)
                c_pos[f] = pos[mask]
                c_ids[f] = ids[mask]
            stats.core_points_total += len(c_pos[f])
            store.write_correspondences(f, c_pos[f], c_ids[f])

        tr = Tracker(cpar, vpar, tpar, spar, cals, naming=naming, store=store)
        if use_3d:
            tr.full_forward_3d()
            enforce_reciprocity(linkage_name, first, last, store=store)
        else:
            tr.full_forward()
            tr.full_backward()
            enforce_reciprocity(linkage_name, first, last, store=store)
            # Re-read correspondences in case full_backward() appended new particles (tpar.add > 0).
            for f in frames:
                up_pos, up_ids = store.read_correspondences(f)
                if len(up_pos) == 0:
                    up_ids = np.empty((0, num_cams), dtype=np.int32)
                c_pos[f] = np.asarray(up_pos, dtype=np.float64)
                c_ids[f] = np.asarray(up_ids, dtype=np.int32).reshape(-1, num_cams)
        n_links = count_links(linkage_name, first, last, store=store)
        extracted = _extract_verified_core(
            linkage_name,
            first,
            last,
            store,
            c_pos,
            c_ids,
            num_cams,
            min_core_length=min_len_tier,
            max_accel_err=dacc_val,
        )
        return (*extracted, n_links)

    def _merge_core_tier(
        a_pos: dict[int, np.ndarray],
        a_ids: dict[int, np.ndarray],
        a_prv: dict[int, np.ndarray],
        a_nxt: dict[int, np.ndarray],
        b_pos: dict[int, np.ndarray],
        b_ids: dict[int, np.ndarray],
        b_prv: dict[int, np.ndarray],
        b_nxt: dict[int, np.ndarray],
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
    ]:
        m_pos_d: dict[int, np.ndarray] = {}
        m_ids_d: dict[int, np.ndarray] = {}
        m_prv_d: dict[int, np.ndarray] = {}
        m_nxt_d: dict[int, np.ndarray] = {}
        for f in frames:
            na = len(a_pos[f])
            nb = len(b_pos[f])
            m_pos_d[f] = np.vstack([a_pos[f], b_pos[f]]) if (na + nb) > 0 else np.empty((0, 3), dtype=np.float64)
            m_ids_d[f] = np.vstack([a_ids[f], b_ids[f]]) if (na + nb) > 0 else np.empty((0, num_cams), dtype=np.int32)
            na_prev = len(a_pos[f - 1]) if (f - 1) in a_pos else 0
            na_next = len(a_pos[f + 1]) if (f + 1) in a_pos else 0
            adj_prv = np.where(b_prv[f] >= 0, b_prv[f] + na_prev, PREV_NONE).astype(np.int32)
            adj_nxt = np.where(b_nxt[f] >= 0, b_nxt[f] + na_next, NEXT_NONE).astype(np.int32)
            m_prv_d[f] = np.concatenate([a_prv[f], adj_prv]) if (na + nb) > 0 else np.empty(0, dtype=np.int32)
            m_nxt_d[f] = np.concatenate([a_nxt[f], adj_nxt]) if (na + nb) > 0 else np.empty(0, dtype=np.int32)
        return m_pos_d, m_ids_d, m_prv_d, m_nxt_d

    if tiers >= 3 and num_cams >= 3:
        # Tier 1: strict full-camera core with longer minimum chain length.
        t1_pos, t1_ids, t1_prv, t1_nxt, claimed_t1, n_t1, links_t1 = _run_core_tier(
            orig_pos,
            orig_ids,
            min_cams_tier=num_cams,
            min_len_tier=min(max(4, eff_min_core_len), len(frames)) if len(frames) >= 2 else eff_min_core_len,
            apply_rcm=True,
        )
        for f in frames:
            store.write_correspondences(f, orig_pos[f], orig_ids[f])
        pruned_t1 = prune_claimed_correspondences(store, frames, claimed_t1)
        stats.pruned_correspondences += sum(pruned_t1.values())

        mid_pos: dict[int, np.ndarray] = {}
        mid_ids: dict[int, np.ndarray] = {}
        for f in frames:
            mp, mi = store.read_correspondences(f)
            if len(mp) == 0:
                mi = np.empty((0, num_cams), dtype=np.int32)
            mid_pos[f] = np.asarray(mp, dtype=np.float64)
            mid_ids[f] = np.asarray(mi, dtype=np.int32).reshape(-1, num_cams)

        # Tier 2: medium core on pruned cloud.
        t2_pos, t2_ids, t2_prv, t2_nxt, claimed_t2, n_t2, links_t2 = _run_core_tier(
            mid_pos,
            mid_ids,
            min_cams_tier=max(2, num_cams - 1),
            min_len_tier=eff_min_core_len,
            apply_rcm=False,
        )
        for f in frames:
            store.write_correspondences(f, mid_pos[f], mid_ids[f])
        pruned_t2 = prune_claimed_correspondences(store, frames, claimed_t2)
        stats.pruned_correspondences += sum(pruned_t2.values())

        p1_pos, p1_ids, p1_prev, p1_next = _merge_core_tier(
            t1_pos, t1_ids, t1_prv, t1_nxt, t2_pos, t2_ids, t2_prv, t2_nxt
        )
        stats.links_after_pass1 = links_t1 + links_t2
        stats.verified_core_tracks = n_t1 + n_t2
        stats.verified_core_points = sum(len(p1_pos[f]) for f in frames)
    else:
        p1_pos, p1_ids, p1_prev, p1_next, claimed_by_frame, n_core_tracks, links_p1 = _run_core_tier(
            orig_pos,
            orig_ids,
            min_cams_tier=core_min_cams,
            min_len_tier=eff_min_core_len,
            apply_rcm=True,
        )
        stats.links_after_pass1 = links_p1
        stats.verified_core_tracks = n_core_tracks
        stats.verified_core_points = sum(len(p1_pos[f]) for f in frames)

        # Restore original cloud and deterministically prune claimed 2D targets.
        for f in frames:
            store.write_correspondences(f, orig_pos[f], orig_ids[f])
        pruned = prune_claimed_correspondences(store, frames, claimed_by_frame)
        stats.pruned_correspondences = sum(pruned.values())

    res_pos: dict[int, np.ndarray] = {}
    res_ids: dict[int, np.ndarray] = {}
    for f in frames:
        r_pos, r_ids = store.read_correspondences(f)
        if len(r_pos) == 0:
            r_ids = np.empty((0, num_cams), dtype=np.int32)
        res_pos[f] = np.asarray(r_pos, dtype=np.float64)
        res_ids[f] = np.asarray(r_ids, dtype=np.int32).reshape(-1, num_cams)
        stats.residual_points_total += len(res_pos[f])

    # Pass 2: track pruned residual cloud.
    tracker_p2 = Tracker(cpar, vpar, tpar, spar, cals, naming=naming, store=store)
    if use_3d:
        tracker_p2.full_forward_3d()
    else:
        tracker_p2.full_forward()
    tracker_p2.full_backward()
    enforce_reciprocity(linkage_name, first, last, store=store)
    stats.links_after_pass2 = count_links(linkage_name, first, last, store=store)

    # Re-read residual correspondences in case full_backward() appended new particles.
    for f in frames:
        up_pos, up_ids = store.read_correspondences(f)
        if len(up_pos) == 0:
            up_ids = np.empty((0, num_cams), dtype=np.int32)
        res_pos[f] = np.asarray(up_pos, dtype=np.float64)
        res_ids[f] = np.asarray(up_ids, dtype=np.int32).reshape(-1, num_cams)

    # Read Pass-2 linkage and post-prune unlinked rows sharing 2D targets with linked Pass-2 tracks.
    p2_prv_map: dict[int, np.ndarray] = {}
    p2_nxt_map: dict[int, np.ndarray] = {}
    keep_masks: dict[int, np.ndarray] = {}
    remap_p2: dict[int, np.ndarray] = {}
    for f in frames:
        n2 = len(res_pos[f])
        r2 = read_linkage(linkage_name, f, store=store)
        if r2 is None:
            p2_prv = np.full(n2, PREV_NONE, dtype=np.int32)
            p2_nxt = np.full(n2, NEXT_NONE, dtype=np.int32)
        else:
            p2_prv, p2_nxt, _ = r2
        p2_prv_map[f] = p2_prv
        p2_nxt_map[f] = p2_nxt
        linked = (p2_prv >= 0) | (p2_nxt >= 0)
        claimed_p2: list[set[int]] = [
            {int(v) for v in res_ids[f][linked, c] if v >= 0}
            for c in range(num_cams)
        ]
        conflict = np.zeros(n2, dtype=bool)
        if n2 > 0 and any(claimed_p2):
            for c in range(num_cams):
                if not claimed_p2[c]:
                    continue
                col = res_ids[f][:, c]
                valid = (col >= 0) & (~linked)
                if np.any(valid):
                    conflict[valid] |= np.isin(col[valid], list(claimed_p2[c]))
        keep = ~conflict
        keep_masks[f] = keep
        new_idx = np.full(n2, -1, dtype=np.int32)
        new_idx[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        remap_p2[f] = new_idx

    # Merge Pass-1 verified core and post-pruned Pass-2 residual into store.
    merged_linkage: list[tuple[np.ndarray, np.ndarray]] = []
    merged_xyz: list[np.ndarray] = []
    for f in frames:
        keep = keep_masks[f]
        res_pos_f = res_pos[f][keep]
        res_ids_f = res_ids[f][keep]
        p2_prv = p2_prv_map[f][keep]
        p2_nxt = p2_nxt_map[f][keep]
        if (f - 1) in remap_p2:
            rm_prev = remap_p2[f - 1]
            if len(rm_prev) > 0:
                p2_prv = np.where(
                    (p2_prv >= 0) & (p2_prv < len(rm_prev)),
                    rm_prev[np.clip(p2_prv, 0, len(rm_prev) - 1)],
                    PREV_NONE,
                ).astype(np.int32)
            else:
                p2_prv = np.full(len(p2_prv), PREV_NONE, dtype=np.int32)
        if (f + 1) in remap_p2:
            rm_next = remap_p2[f + 1]
            if len(rm_next) > 0:
                p2_nxt = np.where(
                    (p2_nxt >= 0) & (p2_nxt < len(rm_next)),
                    rm_next[np.clip(p2_nxt, 0, len(rm_next) - 1)],
                    NEXT_NONE,
                ).astype(np.int32)
            else:
                p2_nxt = np.full(len(p2_nxt), NEXT_NONE, dtype=np.int32)

        n1 = len(p1_pos[f])
        n2 = len(res_pos_f)
        m_pos = (
            np.vstack([p1_pos[f], res_pos_f])
            if (n1 + n2) > 0
            else np.empty((0, 3), dtype=np.float64)
        )
        m_ids = (
            np.vstack([p1_ids[f], res_ids_f])
            if (n1 + n2) > 0
            else np.empty((0, num_cams), dtype=np.int32)
        )
        store.write_correspondences(f, m_pos, m_ids)

        n1_prev_frame = len(p1_pos[f - 1]) if (f - 1) in p1_pos else 0
        n1_next_frame = len(p1_pos[f + 1]) if (f + 1) in p1_pos else 0
        adj_prv = np.where(p2_prv >= 0, p2_prv + n1_prev_frame, PREV_NONE).astype(
            np.int32
        )
        adj_nxt = np.where(p2_nxt >= 0, p2_nxt + n1_next_frame, NEXT_NONE).astype(
            np.int32
        )
        m_prv = np.concatenate([p1_prev[f], adj_prv]) if (n1 + n2) > 0 else np.empty(0, dtype=np.int32)
        m_nxt = np.concatenate([p1_next[f], adj_nxt]) if (n1 + n2) > 0 else np.empty(0, dtype=np.int32)
        merged_linkage.append((m_prv, m_nxt))
        merged_xyz.append(m_pos)

    store.write_linkage_many(frames, merged_linkage, merged_xyz, name=linkage_name)

    dangle_val = float(getattr(tpar, "dangle", 0.0))
    if dacc_val is not None and dacc_val > 0.0:
        prune_kinky_links(
            linkage_name,
            first,
            last,
            max_accel_err=dacc_val,
            max_angle_deg=dangle_val,
            store=store,
        )

    # Upgraded track_assisted corrective pass (3-4 pt poly + ray-miss/kinematic gates).
    if corrective_passes > 0:
        stats.corrective_stats = run_corrective_pass(
            cpar,
            vpar,
            tpar,
            spar,
            cals,
            store,
            linkage_name=linkage_name,
            max_passes=corrective_passes,
            release_ghosts=True,
            forward=True,
            predictor=predictor,
            allow_shared_targets=allow_shared_targets,
            radius_px=radius_px,
            max_ray_miss_mm=rcm_threshold,
            max_pos_err_mm=dacc_val,
            gap_accel_err=0.25 * dacc_val if dacc_val is not None else None,
            require_bidir_vel=True,
            uniqueness_ratio=1.5,
        )

    # Final postprocess: tight gap relinking + reciprocity guard.
    relink_trajectory_gaps(
        linkage_name,
        first,
        last,
        max_gap=2,
        max_accel_err=0.25 * float(tpar.dacc),
        store=store,
        require_bidir_vel=True,
        uniqueness_ratio=1.5,
    )
    enforce_reciprocity(linkage_name, first, last, store=store)
    if dacc_val is not None and dacc_val > 0.0:
        prune_kinky_links(
            linkage_name,
            first,
            last,
            max_accel_err=dacc_val,
            max_angle_deg=dangle_val,
            store=store,
        )
    stats.links_final = count_links(linkage_name, first, last, store=store)
    return stats


__all__ = [
    "PeelStats",
    "assign_layers",
    "peel_and_track",
    "peel_layers",
    "prune_claimed_correspondences",
    "prune_frame_correspondences",
    "quality_scores",
    "resolve_conflict_components",
]

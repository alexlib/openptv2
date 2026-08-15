"""Corrective backward pass with track-assisted re-correspondence (Stage 2,
docs/plans/2026-08-15-tracking-quality-overhaul.md).

Walks backward frame t+1 -> t using the tracks a forward run already
established. For every track that STARTS at t+1 going forward (prev == -1,
i.e. the combinatorial correspondence step at frame t either missed the
particle entirely or a 2-camera-only ghost pushed it out) and has a known
forward velocity, this predicts where it should be at frame t, projects
that prediction into every camera, searches each camera's UNCLAIMED 2D
targets (the ones the forward correspondence pass never used), and
triangulates when >= 2 cameras agree -- recovering a link the frozen,
one-way detect -> correspond -> track pipeline structurally cannot recover
on its own. Reuses tracking_postprocess.relink_trajectory_gaps and
enforce_reciprocity (already proven, unchanged) for gap-bridging and
disagreement pruning rather than reimplementing either.

Scope cuts from the original Stage 2 spec (deliberate, each because the
primitive it needs doesn't exist as reusable plain-Python code -- see
docs/plans/2026-08-15-tracking-quality-overhaul.md's Stage 2 section for the
verification trail):
  - No GMM long-history predictor (plugins/proptv/prediction.py) or STB
    shake refinement (plugins/stb_4d_refinement.py, needs real per-camera
    image arrays this pass doesn't have) -- linear 2-point backward
    extrapolation only.
  - No re-ranking of DISAGREEING existing links against the fixed
    trackback_loop_fast decision logic (track_kernels_corr.py) -- that
    logic lives only inside the compiled nogil kernel with no plain-Python
    equivalent (assess_new_position/assess_new_position_fast are the same
    story: unused elsewhere in the codebase, and unusable outside the full
    fast-kernel TrackingRun/FrameBuf setup). enforce_reciprocity's existing
    forward/backward veto is the disagreement signal used instead.
  - No per-track running cam-count ghost decay (a track-claimed particle
    sustained by only 2 cameras for >= 3 consecutive frames) -- claimed
    rows are tagged 2-cam vs 3+-cam so a decay rule can be added without
    touching the claim mechanism, but nothing prunes on it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openptv2.algorithms.constants import COORD_UNUSED, PT_UNUSED, TR_UNUSED
from openptv2.algorithms.orientation import point_position
from openptv2.algorithms.track import candsearch_in_pix_rest, point_to_pixel
from openptv2.algorithms.tracking_frame_buf import Target
from openptv2.correspondences import MatchedCoords
from openptv2.tracking_postprocess import (
    count_links,
    enforce_reciprocity,
    read_linkage,
    relink_trajectory_gaps,
    write_linkage,
)

ADD_RADIUS_PX = 3.0  # matches ADD_PART's search radius convention elsewhere


@dataclass
class CorrectiveStats:
    passes_run: int = 0
    claimed_total: int = 0
    claimed_2cam: int = 0
    links_before: int = 0
    links_after: int = 0
    gaps_bridged: dict[str, dict] = field(default_factory=dict)
    reciprocity: dict[str, dict] = field(default_factory=dict)


def _claimed_indices(cam_target_ids: np.ndarray, cam: int) -> set[int]:
    if cam_target_ids.shape[0] == 0:
        return set()
    return {int(v) for v in cam_target_ids[:, cam] if v >= 0}


def _unclaimed_target_list(store, cam: int, frame: int, claimed: set[int]) -> list[Target]:
    """Plain, property-based Target objects (what candsearch_in_pix_rest
    needs), preserving read_targets' stored order (already y-sorted --
    every writer in this codebase sorts before storing). tnr encodes
    claimed/unclaimed since that's the field candsearch_in_pix_rest gates
    on."""
    tarr = store.read_targets(cam, frame)
    out = []
    for i in range(len(tarr)):
        t = tarr[i]
        x, y = t.pos()
        tnr = TR_UNUSED if i not in claimed else TR_UNUSED - 1
        out.append(Target(pnr=i, x=x, y=y, n=0, nx=0, ny=0, sumg=0, tnr=tnr))
    return out


def _claim_particle(predicted_pos, cals, cpar, store, frame: int, radius_px: float = ADD_RADIUS_PX):
    """Project predicted_pos into every camera, search unclaimed targets,
    triangulate when >= 2 cameras hit. Returns (pos_3d, cam_target_ids,
    n_cams) or None."""
    num_cams = cpar.num_cams
    if store.has_correspondences(frame):
        _pos, existing_ids = store.read_correspondences(frame)
    else:
        existing_ids = np.empty((0, num_cams), dtype=np.int32)

    targets_metric = np.full((num_cams, 2), COORD_UNUSED)
    cam_ids = np.full(num_cams, -1, dtype=np.int32)
    n_hits = 0
    for cam in range(num_cams):
        if not store.has_targets(cam, frame):
            continue
        claimed = _claimed_indices(existing_ids, cam)
        targets = _unclaimed_target_list(store, cam, frame, claimed)
        if not targets:
            continue
        px, py = point_to_pixel(predicted_pos, cals[cam], cpar)
        idx_out = [PT_UNUSED]
        counter = candsearch_in_pix_rest(
            targets, len(targets), px, py, radius_px, radius_px, radius_px, radius_px,
            idx_out, cpar,
        )
        if counter > 0 and idx_out[0] != PT_UNUSED:
            hit = targets[idx_out[0]]
            flat = MatchedCoords([hit], cpar, cals[cam])
            targets_metric[cam] = (flat[0].x, flat[0].y)
            cam_ids[cam] = idx_out[0]
            n_hits += 1

    if n_hits < 2:
        return None
    pos, _dist = point_position(targets_metric, num_cams, cpar.mm, cals)
    return pos, cam_ids, n_hits


def _append_correspondence(store, frame: int, pos, cam_ids) -> int:
    num_cams = len(cam_ids)
    if store.has_correspondences(frame):
        old_pos, old_ids = store.read_correspondences(frame)
    else:
        old_pos = np.empty((0, 3))
        old_ids = np.empty((0, num_cams), dtype=np.int32)
    new_pos = np.vstack([old_pos, np.asarray(pos, dtype=np.float64)[None, :]])
    new_ids = np.vstack([old_ids, np.asarray(cam_ids, dtype=np.int32)[None, :]])
    store.write_correspondences(frame, new_pos, new_ids)
    return new_pos.shape[0] - 1


def _empty_linkage():
    return (
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 3)),
    )


def _backward_walk(cpar, cals, store, linkage_name: str, first: int, last: int) -> tuple[int, int]:
    """One backward sweep, frame last-1 -> first: for every track head at
    t+1 with a known forward velocity, try to claim a particle at t.
    Returns (claimed_total, claimed_2cam)."""
    claimed_total = claimed_2cam = 0

    for t in range(last - 1, first - 1, -1):
        r_here = read_linkage(linkage_name, t + 1, store=store)
        if r_here is None:
            continue
        prev_here, next_here, xyz_here = r_here
        r_next2 = read_linkage(linkage_name, t + 2, store=store) if t + 2 <= last else None
        xyz_next2 = r_next2[2] if r_next2 is not None else None

        r_t = read_linkage(linkage_name, t, store=store)
        prev_t, next_t, xyz_t = r_t if r_t is not None else _empty_linkage()
        dirty_here = dirty_t = False

        for i in range(len(prev_here)):
            if prev_here[i] >= 0 or next_here[i] < 0:
                continue  # already has a predecessor, or no forward velocity to extrapolate from
            j = int(next_here[i])
            if xyz_next2 is None or j >= len(xyz_next2):
                continue
            velocity = xyz_next2[j] - xyz_here[i]
            predicted = xyz_here[i] - velocity

            claimed = _claim_particle(predicted, cals, cpar, store, t)
            if claimed is None:
                continue
            pos, cam_ids, n_hits = claimed
            row = _append_correspondence(store, t, pos, cam_ids)

            prev_t = np.append(prev_t, -1)
            next_t = np.append(next_t, i)
            xyz_t = np.vstack([xyz_t, np.asarray(pos, dtype=np.float64)[None, :]])
            dirty_t = True

            prev_here[i] = row
            dirty_here = True
            claimed_total += 1
            if n_hits == 2:
                claimed_2cam += 1

        if dirty_here:
            write_linkage(linkage_name, t + 1, prev_here, next_here, xyz_here, store=store)
        if dirty_t:
            write_linkage(linkage_name, t, prev_t, next_t, xyz_t, store=store)

    return claimed_total, claimed_2cam


def run_corrective_pass(
    cpar, vpar, tpar, spar, cals, store,
    linkage_name: str = "ptv_is", max_passes: int = 2, min_change_frac: float = 0.01,
) -> CorrectiveStats:
    """The Stage-2 corrective pass. Iterates the backward walk +
    gap-bridging + reciprocity while the sequence's total link count keeps
    changing by more than min_change_frac; stops after max_passes
    regardless."""
    stats = CorrectiveStats()
    first, last = spar.first, spar.last
    stats.links_before = count_links(linkage_name, first, last, store=store)

    for p in range(1, max_passes + 1):
        stats.passes_run = p
        links_before_pass = count_links(linkage_name, first, last, store=store)

        claimed, claimed_2cam = _backward_walk(cpar, cals, store, linkage_name, first, last)
        stats.claimed_total += claimed
        stats.claimed_2cam += claimed_2cam

        stats.gaps_bridged[f"pass{p}"] = relink_trajectory_gaps(
            linkage_name, first, last, max_gap=2,
            max_velocity_err=float(tpar.dvxmax), store=store,
        )
        stats.reciprocity[f"pass{p}"] = enforce_reciprocity(
            linkage_name, first, last, store=store,
        )

        links_after_pass = count_links(linkage_name, first, last, store=store)
        change = abs(links_after_pass - links_before_pass) / max(links_before_pass, 1)
        if change < min_change_frac:
            break

    stats.links_after = count_links(linkage_name, first, last, store=store)
    return stats


__all__ = ["CorrectiveStats", "run_corrective_pass"]

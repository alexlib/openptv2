"""Corrective pass with track-assisted re-correspondence (Stage 2,
docs/plans/2026-08-15-tracking-quality-overhaul.md).

Runs after a forward tracking run, on the RunStore it wrote. Every track
that STARTS at frame t+1 (no predecessor) and already has a velocity is
extrapolated back to frame t; with ``forward=True`` every track that ENDS
at t-1 is also extrapolated forward to t. The predicted 3D point is
projected into every camera and the nearest FREE 2D target within
``radius_px`` is taken in each; with >= 2 cameras the particle is
triangulated (least squares), appended to frame t and linked. New particles
extend their track further on, in the same sweep.

A target is free when no particle uses it. With ``release_ghosts=True``
targets used only by unlinked particles -- points no trajectory accepted,
most of them ghosts -- are free too: a track can take them back. A released
ghost that loses a target to a new particle is removed.

Then tracking_postprocess.relink_trajectory_gaps and enforce_reciprocity
run, as before. Every frame is read once per pass (the earlier version
re-read the frame's targets and correspondences for every predicted
particle, ~40x slower).

test_cavity (5 frames, 2026-10-24): trajectories >= 4 frames that stay within
0.2 mm of a smooth path 258 -> 441 with both options; claims from
predictions moved 2-5 mm off: ~5 % of that, i.e. the added particles are
real detections, not chance hits.

Not done: a long-history predictor (linear 2-point extrapolation only), and
re-ranking of disagreeing existing links (enforce_reciprocity's veto is the
disagreement signal used instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openptv2.algorithms.constants import COORD_UNUSED
from openptv2.algorithms.orientation import point_position
from openptv2.algorithms.track import point_to_pixel
from openptv2.tracking_postprocess import (
    count_links,
    enforce_reciprocity,
    read_linkage,
    relink_trajectory_gaps,
    write_linkage,
)

ADD_RADIUS_PX = 3.0  # matches ADD_PART's search radius convention elsewhere
PRIO_ADDED = 4  # the tracker's priority code for a particle it added
NEXT_NONE = -2
PREV_NONE = -1


@dataclass
class CorrectiveStats:
    passes_run: int = 0
    claimed_total: int = 0
    claimed_2cam: int = 0
    ghosts_removed: int = 0
    links_before: int = 0
    links_after: int = 0
    gaps_bridged: dict[str, dict] = field(default_factory=dict)
    reciprocity: dict[str, dict] = field(default_factory=dict)


class _Frame:
    """One frame's particles, links and targets, held in memory for a pass."""

    def __init__(self, store: Any, linkage_name: str, frame: int, num_cams: int):
        from scipy.spatial import cKDTree

        self.frame = frame
        if store.has_correspondences(frame):
            _pos, ids = store.read_correspondences(frame)
        else:
            ids = np.empty((0, num_cams), dtype=np.int32)
        link = read_linkage(linkage_name, frame, store=store)
        if link is None:
            n = len(ids)
            prev = np.full(n, PREV_NONE, np.int32)
            nxt = np.full(n, NEXT_NONE, np.int32)
            xyz = np.empty((0, 3)) if n == 0 else _pos
        else:
            prev, nxt, xyz = link
        self.n_initial = len(prev)
        # Per-row priority columns that must keep the rows' order and count:
        # the linkage group's own prio, and a legacy-imported "added" group.
        self.prio = {}
        self.link_name = Path(linkage_name).name
        for name in (self.link_name, "added"):
            if not store.has_linkage(frame, name):
                continue
            col = store.read_prio(frame, name)
            if col is not None and len(col) == self.n_initial:
                self.prio[name] = list(np.asarray(col, dtype=np.int32))
        self.prev = list(prev)
        self.next = list(nxt)
        self.xyz = list(np.asarray(xyz, dtype=np.float64))
        self.ids = list(np.asarray(ids, dtype=np.int32).reshape(-1, num_cams))
        self.targets = []
        self.trees = []
        for cam in range(num_cams):
            if store.has_targets(cam, frame):
                ta = store.read_targets(cam, frame)
                xy = np.array([ta[i].pos() for i in range(len(ta))]).reshape(-1, 2)
            else:
                xy = np.empty((0, 2))
            self.targets.append(xy)
            self.trees.append(cKDTree(xy) if len(xy) else None)
        self.dirty = False

    def unlinked(self) -> np.ndarray:
        prev = np.asarray(self.prev[: self.n_initial])
        nxt = np.asarray(self.next[: self.n_initial])
        return (prev < 0) & (nxt < 0)

    def claimed_sets(self, release_ghosts: bool) -> list[set[int]]:
        ids = np.asarray(self.ids[: self.n_initial]).reshape(-1, len(self.targets))
        keep = ~self.unlinked() if release_ghosts else np.ones(len(ids), bool)
        return [
            {int(v) for v in ids[keep, cam] if v >= 0}
            for cam in range(len(self.targets))
        ]

    def append(self, xyz, ids, prev, nxt) -> int:
        self.xyz.append(np.asarray(xyz, dtype=np.float64))
        self.ids.append(np.asarray(ids, dtype=np.int32))
        self.prev.append(prev)
        self.next.append(nxt)
        for col in self.prio.values():
            col.append(PRIO_ADDED)
        self.dirty = True
        return len(self.xyz) - 1


def _claim_particle(predicted_pos, fr: _Frame, claimed, cals, cpar, radius_px):
    """Nearest free target within ``radius_px`` of the projected prediction in
    every camera; triangulate when >= 2 cameras hit. Returns (pos, ids, n) or
    None. Marks the used targets as claimed."""
    from openptv2.transforms import convert_arr_pixel_to_metric, distorted_to_flat

    num_cams = cpar.num_cams
    targets_metric = np.full((num_cams, 2), COORD_UNUSED)
    cam_ids = np.full(num_cams, -1, dtype=np.int32)
    for cam in range(num_cams):
        tree = fr.trees[cam]
        if tree is None:
            continue
        px = point_to_pixel(predicted_pos, cals[cam], cpar)
        k = min(4, len(fr.targets[cam]))
        dist, idx = tree.query(px, k=k, distance_upper_bound=radius_px)
        for d, j in zip(np.atleast_1d(dist), np.atleast_1d(idx)):
            if np.isfinite(d) and int(j) not in claimed[cam]:
                metric = convert_arr_pixel_to_metric(fr.targets[cam][j][None], cpar)
                targets_metric[cam] = distorted_to_flat(metric, cals[cam])[0]
                cam_ids[cam] = int(j)
                break
    n_hits = int((cam_ids >= 0).sum())
    if n_hits < 2:
        return None
    pos, _dist = point_position(targets_metric, num_cams, cpar.mm, cals, method="lsq")
    for cam in range(num_cams):
        if cam_ids[cam] >= 0:
            claimed[cam].add(int(cam_ids[cam]))
    return pos, cam_ids, n_hits


def _extend_tracks(
    cpar: Any,
    cals: Any,
    store: Any,
    linkage_name: str,
    first: int,
    last: int,
    *,
    release_ghosts: bool = True,
    forward: bool = True,
    radius_px: float = ADD_RADIUS_PX,
) -> tuple[int, int, int]:
    """One backward (and optionally forward) sweep. Returns
    (claimed_total, claimed_2cam, ghosts_removed)."""
    num_cams = cpar.num_cams
    frames = {
        f: _Frame(store, linkage_name, f, num_cams) for f in range(first, last + 1)
    }
    claimed = {f: fr.claimed_sets(release_ghosts) for f, fr in frames.items()}
    claimed_total = claimed_2cam = 0

    def extend(t: int, here: _Frame, beyond: _Frame, backward: bool) -> None:
        nonlocal claimed_total, claimed_2cam
        target = frames[t]
        for i in range(len(here.xyz)):
            has_new_side = here.prev[i] >= 0 if backward else here.next[i] >= 0
            j = here.next[i] if backward else here.prev[i]
            if has_new_side or j < 0 or j >= len(beyond.xyz):
                continue
            predicted = 2.0 * here.xyz[i] - beyond.xyz[j]
            got = _claim_particle(predicted, target, claimed[t], cals, cpar, radius_px)
            if got is None:
                continue
            pos, ids, n_hits = got
            if backward:
                row = target.append(pos, ids, PREV_NONE, i)
                here.prev[i] = row
            else:
                row = target.append(pos, ids, i, NEXT_NONE)
                here.next[i] = row
            here.dirty = True
            claimed_total += 1
            claimed_2cam += n_hits == 2

    for t in range(last - 2, first - 1, -1):  # heads at t+1, velocity from t+2
        extend(t, frames[t + 1], frames[t + 2], backward=True)
    if forward:
        for t in range(first + 2, last + 1):  # tails at t-1, velocity from t-2
            extend(t, frames[t - 1], frames[t - 2], backward=False)

    ghosts_removed = _drop_reclaimed_ghosts(frames, num_cams) if release_ghosts else 0

    for f, fr in frames.items():
        if not fr.dirty:
            continue
        xyz = np.asarray(fr.xyz, dtype=np.float64).reshape(-1, 3)
        ids = np.asarray(fr.ids, dtype=np.int32).reshape(-1, num_cams)
        store.write_correspondences(f, xyz, ids)
        prev = np.asarray(fr.prev, np.int32)
        nxt = np.asarray(fr.next, np.int32)
        write_linkage(linkage_name, f, prev, nxt, xyz, store=store)
        for name, col in fr.prio.items():
            store.write_linkage(f, prev, nxt, xyz, name=name, prio=np.asarray(col))
        _write_target_rows(store, f, ids)
    return claimed_total, claimed_2cam, ghosts_removed


def _drop_reclaimed_ghosts(frames: dict[int, _Frame], num_cams: int) -> int:
    """Remove unlinked particles that share a target with a newly added one,
    renumbering the links of the neighbouring frames."""
    removed = 0
    keep_by_frame = {}
    for f, fr in frames.items():
        ids = np.asarray(fr.ids, dtype=np.int32).reshape(-1, num_cams)
        new = ids[fr.n_initial :]
        drop = np.zeros(len(ids), bool)
        if len(new):
            taken = [set(new[:, c][new[:, c] >= 0].tolist()) for c in range(num_cams)]
            for r in np.flatnonzero(fr.unlinked()):
                drop[r] = any(
                    ids[r, c] >= 0 and ids[r, c] in taken[c] for c in range(num_cams)
                )
        keep_by_frame[f] = ~drop
        removed += int(drop.sum())
    if removed == 0:
        return 0
    for f, keep in keep_by_frame.items():
        if keep.all():
            continue
        new_index = np.cumsum(keep) - 1
        for g, attr in ((f - 1, "next"), (f + 1, "prev")):
            if g in frames:
                links = getattr(frames[g], attr)
                for k, v in enumerate(links):
                    if v >= 0:
                        links[k] = int(new_index[v])
                frames[g].dirty = True
        fr = frames[f]
        for attr in ("xyz", "ids", "prev", "next"):
            setattr(fr, attr, [v for v, k in zip(getattr(fr, attr), keep) if k])
        fr.prio = {n: [v for v, k in zip(c, keep) if k] for n, c in fr.prio.items()}
        fr.dirty = True
    return removed


def _write_target_rows(store: Any, frame: int, ids: np.ndarray) -> None:
    """Point every target's tnr at the particle row that now uses it."""
    for cam in range(ids.shape[1]):
        if not store.has_targets(cam, frame):
            continue
        ta = store.read_targets(cam, frame)
        tnr = np.full(len(ta), -1, dtype=np.int64)
        for row, j in enumerate(ids[:, cam]):
            if 0 <= j < len(ta):
                tnr[j] = row
        for i in range(len(ta)):
            ta[i].set_tnr(int(tnr[i]))
        store.write_targets(cam, frame, ta)


def run_corrective_pass(
    cpar: Any,
    vpar: Any,
    tpar: Any,
    spar: Any,
    cals: Any,
    store: Any,
    linkage_name: str = "ptv_is",
    max_passes: int = 2,
    min_change_frac: float = 0.01,
    release_ghosts: bool = True,
    forward: bool = True,
) -> CorrectiveStats:
    """The Stage-2 corrective pass. Iterates track extension + gap-bridging +
    reciprocity while the sequence's total link count keeps changing by more
    than min_change_frac; stops after max_passes regardless."""
    stats = CorrectiveStats()
    first, last = spar.first, spar.last
    stats.links_before = count_links(linkage_name, first, last, store=store)

    for p in range(1, max_passes + 1):
        stats.passes_run = p
        links_before_pass = count_links(linkage_name, first, last, store=store)

        claimed, claimed_2cam, ghosts = _extend_tracks(
            cpar,
            cals,
            store,
            linkage_name,
            first,
            last,
            release_ghosts=release_ghosts,
            forward=forward,
        )
        stats.claimed_total += claimed
        stats.claimed_2cam += claimed_2cam
        stats.ghosts_removed += ghosts

        stats.gaps_bridged[f"pass{p}"] = relink_trajectory_gaps(
            linkage_name,
            first,
            last,
            max_gap=2,
            max_accel_err=float(tpar.dacc),
            store=store,
        )
        stats.reciprocity[f"pass{p}"] = enforce_reciprocity(
            linkage_name,
            first,
            last,
            store=store,
        )

        links_after_pass = count_links(linkage_name, first, last, store=store)
        change = abs(links_after_pass - links_before_pass) / max(links_before_pass, 1)
        if change < min_change_frac:
            break

    stats.links_after = count_links(linkage_name, first, last, store=store)
    return stats


def run_configured_corrective_pass(exp: Any, tracker: Any, track_cfg: dict) -> None:
    """Run ``track.corrective_passes`` passes after a tracking plugin's own run
    (0 = off), on the RunStore the tracker wrote to. ``track.corrective_release_ghosts``
    and ``track.corrective_forward`` (both default true) select the options."""
    corrective_passes = int(track_cfg.get("corrective_passes", 0))
    if corrective_passes <= 0:
        return
    store = getattr(tracker, "_store", None)
    if store is None:
        print(
            "Corrective pass requested (track.corrective_passes) but the "
            "tracker has no RunStore attached -- skipping (needs "
            "per-camera 2D targets and correspondences, ASCII-only runs "
            "cannot provide those to this pass)."
        )
        return
    stats = run_corrective_pass(
        exp.cpar,
        exp.vpar,
        exp.track_par,
        exp.spar,
        exp.cals,
        store,
        linkage_name=Path(tracker._naming["linkage"]).name,
        max_passes=corrective_passes,
        release_ghosts=bool(track_cfg.get("corrective_release_ghosts", True)),
        forward=bool(track_cfg.get("corrective_forward", True)),
    )
    print(
        f"Corrective pass: claimed {stats.claimed_total} particle(s) "
        f"({stats.claimed_2cam} 2-camera-only), removed {stats.ghosts_removed} "
        f"ghost(s) over {stats.passes_run} pass(es); links "
        f"{stats.links_before} -> {stats.links_after}"
    )


__all__ = ["CorrectiveStats", "run_configured_corrective_pass", "run_corrective_pass"]

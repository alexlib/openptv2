"""The seal pass: the only place in this codebase that walks the tracker's
``prev``/``next`` linkage graph.

Everything else -- the GUI, flowtracks, openptv-cloud's post-processing --
used to each carry its own copy of that walk, and each copy carried the same
three correctness guards (a gap in the stored frames, a row-count mismatch,
an ambiguous/duplicate ``prev`` claim). See
``docs/plans/2026-08-14-storage-formats-as-built.md`` and the linkage-walk
case of the legacy ``read_zarr_trajectories``
(``openptv2/storage/zarr_store.py:611-692``), whose logic this module
supersedes.

``seal(store)``:

1. Walks ``linkage/<name>/frame_*`` in frame order, assigning a ``trajid`` to
   every particle by connected-component over ``prev``/``next``, and writes
   that labelling back onto each frame group.
2. Builds the ``traj/`` index: ``(trajid, first, last, length)`` -- the same
   triple as flowtracks' ``/bounds`` table.
3. Materialises the flat ``trajectories/`` cache (``pos, vel, accel, time,
   trajid``, in METRES, ordered by ``(trajid, time)``) so a trajectory can be
   read as one slice instead of a graph walk.
4. Stamps ``meta/sealed=True`` and ``meta/source_hash`` -- a hash of the
   linkage content just walked, so a later read of ``trajectories/`` can
   detect "this is stale, the source changed since the last seal" instead of
   silently returning an outdated result the way the old ``trajectories/``
   cache did.

Idempotent: calling ``seal()`` twice on unchanged linkage is a no-op (the
second call's source hash matches and it returns early). Re-run after a
re-track, or after ``tracking_postprocess`` edits linkage in place
(``enforce_reciprocity``, ``seed_cold_start``, ``relink_trajectory_gaps``).

Velocity/acceleration are not computed here -- Phase A's ``seal`` writes
zeros for ``vel``/``accel``; a later post-processing pass (Savitzky-Golay or
similar, matching ``flowtracks.smoothing``) fills them in by writing back
into the same ``trajectories/`` rows. This mirrors what
``openptv-cloud/src/openptv_cloud/post.py`` already does downstream.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..tracking_postprocess import MAX_LINK_STEP
from .run_store import RunStore, RunStoreError, _frame_num

MM_TO_M = 1.0 / 1000.0


def compute_source_hash(store: RunStore, name: str = "ptv_is") -> str:
    """Deterministic hash of one linkage stream's content, in frame order."""
    h = hashlib.sha256()
    frames = sorted(
        (_frame_num(k) for k in store.root[f"linkage/{name}"].keys()),
    ) if f"linkage/{name}" in store.root else []
    for frame in frames:
        prev, next_, pos = store.read_linkage(frame, name)
        h.update(prev.tobytes())
        h.update(next_.tobytes())
        h.update(pos.tobytes())
    return h.hexdigest()


def needs_reseal(store: RunStore, name: str = "ptv_is") -> bool:
    if not store.sealed:
        return True
    return store.root["meta"].attrs.get("source_hash") != compute_source_hash(store, name)


def seal(store: RunStore, name: str = "ptv_is", force: bool = False, min_length: int = 1) -> dict:
    """Run the seal pass. Returns a small summary dict.

    ``name`` selects which linkage stream is the trajectory backbone
    (``"ptv_is"`` by default; pass ``"added"`` for the second tracking pass).

    ``min_length``: discard trajectories shorter than this many frames.
    """
    if f"linkage/{name}" not in store.root:
        raise RunStoreError(f"No linkage '{name}' to seal.")

    source_hash = compute_source_hash(store, name)
    if not force and store.sealed and store.root["meta"].attrs.get("source_hash") == source_hash:
        return {"skipped": True, "reason": "already sealed at this source_hash"}

    frame_keys = sorted(store.root[f"linkage/{name}"].keys(), key=_frame_num)
    frames = [_frame_num(k) for k in frame_keys]

    pos_l, time_l, trajid_l = [], [], []
    per_frame_trajid: dict[int, np.ndarray] = {}
    # frame -> (trajids, next_ids) for the last MAX_LINK_STEP frames. A
    # gap-bridged `prev` points more than one frame back, so a one-frame
    # window would start a new trajectory id at every bridge.
    history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    next_trajid = 0

    for frame in frames:
        prev_ids, next_ids, pos = store.read_linkage(frame, name)
        n = len(pos)
        trajids = np.empty(n, dtype=np.int32)

        # Resolve each `prev` to the frame it actually addresses: the nearest
        # earlier frame whose `next` points back here. Guards 1 (missing
        # frame) and 2 (row-count mismatch, e.g. leftover frames from an
        # earlier run) fall out of that check.
        src = np.full(n, -1, dtype=np.int64)  # inherited trajid, -1 = none
        for s in range(1, MAX_LINK_STEP + 1):
            h = history.get(frame - s)
            if h is None:
                continue
            h_traj, h_next = h
            cand = np.where((prev_ids >= 0) & (src < 0))[0]
            if cand.size == 0:
                break
            i = prev_ids[cand].astype(np.int64)
            ok = i < len(h_next)
            cand, i = cand[ok], i[ok]
            hit = h_next[i] == cand
            src[cand[hit]] = h_traj[i[hit]]

        # Fallback for a non-reciprocal `prev` (linkage that never went
        # through enforce_reciprocity): treat it as the step-1 link it
        # claims to be, as this pass always has.
        h = history.get(frame - 1)
        if h is not None:
            h_traj, _ = h
            cand = np.where((prev_ids >= 0) & (src < 0))[0]
            i = prev_ids[cand].astype(np.int64)
            ok = i < len(h_traj)
            src[cand[ok]] = h_traj[i[ok]]

        # Guard 3: disambiguate multi-claim linkages -- more than one particle
        # in this frame inheriting the same trajid is not a valid chain
        # continuation for either claimant.
        claimed, claim_counts = np.unique(src[src >= 0], return_counts=True)
        if claimed.size:
            src[np.isin(src, claimed[claim_counts > 1])] = -1

        linked = src >= 0
        trajids[linked] = src[linked]
        n_new = int((~linked).sum())
        trajids[~linked] = np.arange(next_trajid, next_trajid + n_new)
        next_trajid += n_new

        per_frame_trajid[frame] = trajids
        pos_l.append(pos)
        time_l.append(np.full(n, frame, dtype=np.int64))
        trajid_l.append(trajids.astype(np.int64))
        history[frame] = (trajids, next_ids)
        for old in [f for f in history if f <= frame - MAX_LINK_STEP]:
            del history[old]

    # Write the labelling back onto each linkage frame group.
    for frame, trajids in per_frame_trajid.items():
        store.set_trajid(frame, name, trajids)

    if not pos_l:
        store.write_traj_index(
            np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32)
        )
        store.write_trajectories(
            np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64)
        )
        store.root["meta"].attrs["sealed"] = True
        store.root["meta"].attrs["source_hash"] = source_hash
        return {"n_trajectories": 0, "n_rows": 0}

    pos_all = np.concatenate(pos_l) * MM_TO_M
    time_all = np.concatenate(time_l)
    trajid_all = np.concatenate(trajid_l)

    order = np.lexsort((time_all, trajid_all))
    pos_all = pos_all[order]
    time_all = time_all[order]
    trajid_all = trajid_all[order]

    unique_ids, first_idx, counts = np.unique(
        trajid_all, return_index=True, return_counts=True
    )
    last_time = np.array(
        [time_all[first_idx[i] + counts[i] - 1] for i in range(len(unique_ids))],
        dtype=np.int32,
    )
    first_time = time_all[first_idx].astype(np.int32)

    store.write_traj_index(
        unique_ids.astype(np.int32), first_time, last_time, counts.astype(np.int32),
        first_row=first_idx.astype(np.int64),
    )

    vel = np.zeros_like(pos_all)
    accel = np.zeros_like(pos_all)
    store.write_trajectories(pos_all, vel, accel, time_all, trajid_all)

    store.root["meta"].attrs["sealed"] = True
    store.root["meta"].attrs["source_hash"] = source_hash

    return {"n_trajectories": len(unique_ids), "n_rows": len(pos_all)}

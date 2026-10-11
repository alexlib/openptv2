"""Flat (trajid, time, pos) arrays of a store's tracker linkage.

The tracker writes, per frame, every 3D point with ``prev`` / ``next``
pointers. Post-processing (reconnect, repair, smoothing, stitching in
flowtracks) wants one row per point with a trajectory id; this walks the
pointers once and returns exactly that.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np


def _get_array(group: Any, name: str) -> np.ndarray | None:
    """group[name] as ndarray, or None when the member is absent (no separate
    membership query -- each one is a round-trip through zarr's async layer)."""
    try:
        return np.asarray(group[name])
    except KeyError:
        return None


def read_linkage_arrays(
    store_path: Any,
    first: int | None = None,
    last: int | None = None,
    workers: int = 16,
    name: str = "ptv_is",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(trajid, time, pos) of every point on the linkage path, pos in metres.

    Frames are read concurrently (I/O and decompression, not Python, dominate);
    trajectory ids then follow the prev/next pointers frame by frame: a link
    counts when ``prev`` points back at a point whose ``next`` points here and
    no other point claims the same ``prev``. Positions are scaled to metres
    from the store's ``meta.raw_units`` (default mm). Reads the block layout
    (openptv2 >= 0.5.15) and per-frame groups alike. Returns None if the store
    has no linkage.
    """
    import zarr

    from openptv2.storage.linkage_blocks import BlockSet, committed_gen

    root: Any = zarr.open_group(str(store_path), mode="r")
    if "linkage" not in root:
        return None
    link_root: Any = root["linkage"]
    names = list(link_root.keys())
    if not names:
        return None
    group: Any = link_root[name if name in names else names[0]]
    blocks = BlockSet(group) if committed_gen(group) else None
    block_frames = set(blocks.frames()) if blocks is not None else set()
    keys = sorted(
        (
            k
            for k in group.keys()
            if k.startswith("frame_") and int(k.split("_")[1]) not in block_frames
        ),
        key=lambda k: int(k.split("_")[1]),
    )
    if first is not None:
        keys = [k for k in keys if int(k.split("_")[1]) >= first]
    if last is not None:
        keys = [k for k in keys if int(k.split("_")[1]) <= last]

    raw_units = "mm"
    if "meta" in root and "raw_units" in root["meta"].attrs:
        raw_units = root["meta"].attrs["raw_units"]
    scale = 0.001 if raw_units == "mm" else 1.0

    def load(key: str) -> tuple[int, np.ndarray, np.ndarray, Any]:
        fg = group[key]
        pos = _get_array(fg, "pos")
        pos = np.empty((0, 3)) if pos is None else pos.astype(np.float64) * scale
        if pos.ndim == 1:
            pos = pos.reshape(-1, 3) if pos.size % 3 == 0 else pos.reshape(-1, 1)
        prev = _get_array(fg, "prev")
        return (
            int(key.split("_")[1]),
            pos,
            np.empty(0, dtype=np.int64) if prev is None else prev.astype(np.int64),
            _get_array(fg, "next"),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(load, keys))
    for f in sorted(block_frames):
        if (first is not None and f < first) or (last is not None and f > last):
            continue
        cols: Any = blocks.read(f) if blocks is not None else None
        if cols is None:
            continue
        pos = np.asarray(cols["pos"], dtype=np.float64).reshape(-1, 3) * scale
        frames.append(
            (f, pos, np.asarray(cols["prev"], dtype=np.int64), np.asarray(cols["next"]))
        )
    frames.sort(key=lambda t: t[0])

    pos_l, time_l, trajid_l = [], [], []
    prev_trajids, prev_next_ids, next_trajid = None, None, 0
    for frame_num, pos, prev_ids, curr_next_ids in frames:
        n = len(pos)
        if n == 0:
            prev_trajids, prev_next_ids = np.empty(0, dtype=np.int64), curr_next_ids
            continue
        trajids = np.empty(n, dtype=np.int64)
        if prev_trajids is None or len(prev_trajids) == 0 or len(prev_ids) != n:
            trajids[:] = np.arange(next_trajid, next_trajid + n)
            next_trajid += n
        else:
            reciprocal = prev_ids >= 0
            if prev_next_ids is not None:
                in_range = reciprocal & (prev_ids < len(prev_next_ids))
                back = np.zeros(n, dtype=bool)
                back[in_range] = (
                    np.asarray(prev_next_ids)[prev_ids[in_range]]
                    == np.flatnonzero(in_range)
                )
                reciprocal = back
            claimed, counts = np.unique(prev_ids[reciprocal], return_counts=True)
            linked = reciprocal & ~np.isin(prev_ids, claimed[counts > 1])
            linked &= prev_ids < len(prev_trajids)
            trajids[linked] = prev_trajids[prev_ids[linked]]
            n_new = int((~linked).sum())
            trajids[~linked] = np.arange(next_trajid, next_trajid + n_new)
            next_trajid += n_new
        pos_l.append(pos)
        time_l.append(np.full(n, frame_num, dtype=np.int64))
        trajid_l.append(trajids)
        prev_trajids, prev_next_ids = trajids, curr_next_ids

    if not pos_l:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty((0, 3)))
    return np.concatenate(trajid_l), np.concatenate(time_l), np.concatenate(pos_l)

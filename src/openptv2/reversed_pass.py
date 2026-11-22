"""Backward tracking as "the same tracker on reversed frame numbers" (docs/trackers.md, direction: both).

``two_phase`` always did it this way. For ``fast_3d`` and ``trackcorr`` this module does the same: a second
:class:`~openptv2.tracker.Tracker` runs forward over a :class:`ReversedStore`, a view of the run store whose frame
``f`` is the real frame ``first + last - f`` and whose writes stay in memory, so the real store is never touched.
Its links are mapped back to real frames and merged with the forward links by the Forward-First rule: forward links
stay, backward links are added only where both ends are still free.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import numpy as np


class ReversedStore:
    """Read view of ``inner`` with frame numbers mirrored; all writes go to memory."""

    def __init__(self, inner: Any, first: int, last: int):
        self._inner = inner
        self._sum = int(first) + int(last)
        self._corr: dict[int, tuple] = {}
        self._targets: dict[tuple[int, int], Any] = {}
        self._link: dict[str, dict[int, dict[str, np.ndarray]]] = {}

    def _m(self, frame: int) -> int:
        return self._sum - int(frame)

    # -- correspondences and targets: read through, writes held ----------------
    def read_correspondences(self, frame):
        held = self._corr.get(int(frame))
        return held if held is not None else self._inner.read_correspondences(self._m(frame))

    def has_correspondences(self, frame):
        return int(frame) in self._corr or self._inner.has_correspondences(self._m(frame))

    def write_correspondences(self, frame, pos_3d, cam_target_ids):
        pos = np.asarray(pos_3d, dtype=np.float64).copy()
        ids = np.asarray(cam_target_ids, dtype=np.int32).copy()
        self._corr[int(frame)] = (pos, ids)

    def read_targets(self, cam, frame):
        held = self._targets.get((int(cam), int(frame)))
        return held if held is not None else self._inner.read_targets(cam, self._m(frame))

    def has_targets(self, cam, frame):
        return (int(cam), int(frame)) in self._targets or self._inner.has_targets(cam, self._m(frame))

    def write_targets(self, cam, frame, targets):
        self._targets[(int(cam), int(frame))] = targets

    def read_rcm(self, frame):
        return self._inner.read_rcm(self._m(frame))

    # -- linkage: memory only -----------------------------------------------------
    def write_linkage(self, frame, prev_ids, next_ids, pos_3d, name="ptv_is", prio=None):
        held = {
            "prev": np.array(prev_ids, dtype=np.int32),
            "next": np.array(next_ids, dtype=np.int32),
            "pos": np.array(pos_3d, dtype=np.float64),
        }
        if prio is not None:
            held["prio"] = np.array(prio, dtype=np.int32)
        self._link.setdefault(name, {})[int(frame)] = held

    def read_linkage(self, frame, name="ptv_is"):
        held = self._link.get(name, {}).get(int(frame))
        if held is None:
            raise KeyError(f"no linkage {name!r} for reversed frame {frame}")
        return held["prev"], held["next"], held["pos"]

    def read_prio(self, frame, name="ptv_is"):
        held = self._link.get(name, {}).get(int(frame))
        return None if held is None else held.get("prio")

    def has_linkage(self, frame, name="ptv_is"):
        return int(frame) in self._link.get(name, {})

    def clear_linkage(self, name="ptv_is"):
        self._link.pop(name, None)

    def deferred_linkage(self):
        return nullcontext()

    def __getattr__(self, item):        # mmlut cache and the like: the real store
        return getattr(self._inner, item)

    def held_links(self, name: str = "ptv_is") -> dict[int, np.ndarray]:
        """Reversed frame -> ``next`` column of the backward pass."""
        return {f: c["next"] for f, c in self._link.get(name, {}).items()}


def _links_of(store: Any, first: int, last: int, name: str) -> list[tuple[int, int, int, int]]:
    """Forward links (t0, row0, t1, row1) of the stored linkage, t counted from 0."""
    out = []
    for f in range(first, last):
        if not store.has_linkage(f, name):
            continue
        _prev, nxt, _pos = store.read_linkage(f, name)
        for r in np.flatnonzero(np.asarray(nxt) >= 0):
            out.append((f - first, int(r), f - first + 1, int(nxt[r])))
    return out


def merge_reversed_pass(
    store: Any,
    first: int,
    last: int,
    run_backward: Callable[[ReversedStore], None],
    name: str = "ptv_is",
) -> dict:
    """Run ``run_backward(reversed_store)`` (a tracker pass over the mirrored frames), map its links back and
    merge them with the stored forward links (Forward-First). Rewrites linkage ``name`` in ``store``."""
    from openptv2.plugins.two_phase_tracking import (
        _links_to_linkage,
        _merge_bidirectional_links,
    )

    frames = list(range(first, last + 1))
    sizes, particles = [], []
    for f in frames:
        pos, _ids = store.read_correspondences(f)
        particles.append(np.asarray(pos, dtype=np.float64))
        sizes.append(len(pos))

    fwd = _links_of(store, first, last, name)

    rev = ReversedStore(store, first, last)
    run_backward(rev)
    bwd = []
    for g, nxt in rev.held_links(name).items():
        real_to = last - (g - first)            # real frame of the reversed frame g
        t1 = real_to - first
        t0 = t1 - 1
        if t0 < 0 or t1 >= len(frames):
            continue
        for i in np.flatnonzero(np.asarray(nxt) >= 0):
            j = int(nxt[i])                     # row in reversed frame g+1 = real frame real_to-1
            if i < sizes[t1] and j < sizes[t0]:
                bwd.append((t0, j, t1, int(i)))

    merged = _merge_bidirectional_links(fwd, bwd, particles)
    linkage = _links_to_linkage(merged, frames, sizes)
    store.write_linkage_many(
        frames,
        [(prv, np.where(nxt < 0, -2, nxt).astype(np.int32)) for prv, nxt in linkage],
        particles,
        name=name,
    )
    return {"forward": len(fwd), "backward": len(bwd), "merged": len(merged)}

"""Block layout for tracker output: a few big arrays instead of 3-6 per frame.

A per-frame zarr array costs ~1.2 ms of CPU locally and one request round trip
on an object store, so a 1000-frame linkage is thousands of tiny objects. The
tracker writes its whole result at once anyway, so it stores frames in blocks::

    <group>/block_<gen>_<first>_<last>/
        frames   (F,)   int32   frame numbers in this block (not necessarily contiguous)
        off      (F+1,) int64   row offsets of each frame in the flat arrays
        <name>   (T, ...)       one flat array per column (prev, next, pos, ...)

Commit and crash safety. A write creates blocks of a new generation ``gen`` and
then publishes them with ONE attribute write, ``<group>.attrs["blocks_gen"] =
gen``. A block counts only if its generation is <= ``blocks_gen``; blocks of a
halted write are invisible and are removed by the next write. For a frame
covered by several committed blocks the newest generation wins, so a later
partial rewrite (e.g. a post-processing pass touching a few frames) simply
shadows the old rows. There is no delete-then-write window.

Legacy per-frame groups (``frame_NNNNNN``) stay readable: a frame found in no
block falls back to its per-frame group. Once a name has blocks, new writes go
to blocks (a single-frame write is a one-frame block).

One writer per group at a time (the commit pointer is a single counter).
Per-frame groups remain the layout for many concurrent writers.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from typing import Any, Iterator, Optional

import numpy as np

BLOCK_FRAMES = 256  # frames per block written in one call
_CACHE_BLOCKS = 8
_BLOCK_RE = re.compile(r"^block_(\d+)_(\d+)_(\d+)$")


def _retry(fn, attempts: int = 10):
    """Same retry as RunStore's metadata writes: parallel readers/writers can
    race the rename-into-place of a group's zarr.json (seen on Windows)."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


def parse_block_key(key: str) -> Optional[tuple[int, int, int]]:
    m = _BLOCK_RE.match(key)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def committed_gen(group: Any) -> int:
    return int(group.attrs.get("blocks_gen", 0) or 0)


def _block_keys(group: Any) -> list[str]:
    return [k for k in group.keys() if parse_block_key(k)]


class BlockSet:
    """Reader/writer of the blocks inside one zarr group (linkage/<name>, quality)."""

    def __init__(self, group: Any):
        self.group = group
        self._index: Optional[dict[int, tuple[str, int]]] = None
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def invalidate(self) -> None:
        self._index = None
        self._cache.clear()

    # -- reading --------------------------------------------------------

    def _build_index(self) -> dict[int, tuple[str, int]]:
        gen_ok = committed_gen(self.group)
        if gen_ok == 0:  # legacy per-frame group: nothing to list
            return {}
        keyed = []
        for k in _block_keys(self.group):
            gen, _, _ = parse_block_key(k)  # type: ignore[misc]
            if gen <= gen_ok:
                keyed.append((gen, k))
        index: dict[int, tuple[str, int]] = {}
        for _, k in sorted(keyed):  # oldest first; newer overwrite
            for i, f in enumerate(np.asarray(self.group[k]["frames"]).tolist()):
                index[int(f)] = (k, i)
        return index

    @property
    def index(self) -> dict[int, tuple[str, int]]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def frames(self) -> list[int]:
        return sorted(self.index)

    def __contains__(self, frame: int) -> bool:
        return frame in self.index

    def _load(self, key: str) -> dict[str, np.ndarray]:
        blk = self._cache.get(key)
        if blk is None:
            g = self.group[key]
            blk = {n: np.asarray(g[n]) for n in g.keys()}
            self._cache[key] = blk
            while len(self._cache) > _CACHE_BLOCKS:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return blk

    def read(self, frame: int) -> Optional[dict[str, np.ndarray]]:
        """Columns of ``frame`` (fresh copies, safe to mutate) or None."""
        hit = self.index.get(frame)
        if hit is None:
            return None
        key, i = hit
        blk = self._load(key)
        a, b = int(blk["off"][i]), int(blk["off"][i + 1])
        return {
            n: v[a:b].copy() for n, v in blk.items() if n not in ("frames", "off")
        }

    # -- writing --------------------------------------------------------

    def _next_gen(self) -> int:
        gens = [parse_block_key(k)[0] for k in _block_keys(self.group)]  # type: ignore[index]
        return max(gens + [committed_gen(self.group)]) + 1

    def discard_uncommitted(self) -> int:
        """Remove blocks of halted writes (generation above the commit pointer)."""
        gen_ok = committed_gen(self.group)
        n = 0
        for k in _block_keys(self.group):
            if parse_block_key(k)[0] > gen_ok:  # type: ignore[index]
                del self.group[k]
                n += 1
        return n

    def write(
        self,
        frames: list[int],
        columns: dict[str, list[np.ndarray]],
        block_frames: int = BLOCK_FRAMES,
        commit: bool = True,
    ) -> int:
        """Write ``frames`` as new blocks and (unless ``commit`` is False --
        only for testing a halted write) publish them. Returns the generation."""
        if not frames:
            return committed_gen(self.group)
        self.discard_uncommitted()
        gen = self._next_gen()
        names = list(columns)
        for s in range(0, len(frames), block_frames):
            sel = slice(s, s + block_frames)
            fs = frames[sel]
            lens = np.array([len(columns[names[0]][i]) for i in range(s, s + len(fs))])
            blk = self.group.create_group(f"block_{gen:04d}_{fs[0]:06d}_{fs[-1]:06d}")
            blk.create_array("frames", data=np.asarray(fs, dtype=np.int32))
            blk.create_array(
                "off", data=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            )
            for n in names:
                parts = columns[n][sel]
                blk.create_array(n, data=np.concatenate(parts, axis=0))
        if commit:
            _retry(lambda: self.group.attrs.__setitem__("blocks_gen", gen))
            self._gc()
        self.invalidate()
        return gen

    def _gc(self) -> None:
        """Delete committed blocks that no frame points to any more."""
        self._index = None
        live = {k for k, _ in self.index.values()}
        for k in _block_keys(self.group):
            if k not in live and parse_block_key(k)[0] <= committed_gen(self.group):  # type: ignore[index]
                del self.group[k]

    def set_column(self, name: str, per_frame: dict[int, np.ndarray]) -> list[int]:
        """Add/replace column ``name`` for frames living in blocks (e.g. trajid
        written back by seal). Returns the frames NOT found in any block."""
        by_block: dict[str, dict[int, np.ndarray]] = {}
        missing = []
        for f, arr in per_frame.items():
            hit = self.index.get(f)
            if hit is None:
                missing.append(f)
            else:
                by_block.setdefault(hit[0], {})[hit[1]] = arr
        for key, rows in by_block.items():
            blk = self._load(key)
            off = blk["off"]
            ref = np.asarray(next(v for n, v in blk.items() if n not in ("frames", "off")))
            if name in blk:
                col = blk[name].copy()
            else:
                col = np.full(ref.shape[0], -1, dtype=np.int32)
            for i, arr in rows.items():
                col[int(off[i]) : int(off[i + 1])] = arr
            self.group[key].create_array(name, data=col, overwrite=True)
            self._cache.pop(key, None)
        return missing


def frame_number(key: str) -> Optional[int]:
    if key.startswith("frame_"):
        try:
            return int(key.split("_", 1)[1])
        except ValueError:
            return None
    return None


def iter_linkage(group: Any) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """Every frame of a linkage group, ascending: blocks first, per-frame groups
    for frames no block covers. For readers that hold a raw zarr group (cloud
    post-processing, ``read_zarr_trajectories``) instead of a ``RunStore``."""
    blocks = BlockSet(group)
    per_frame = {}
    for k in group.keys():
        f = frame_number(k)
        if f is not None:
            per_frame[f] = k
    for f in sorted(set(blocks.frames()) | set(per_frame)):
        cols = blocks.read(f)
        if cols is None:
            fg = group[per_frame[f]]
            cols = {n: np.asarray(fg[n]) for n in fg.keys()}
        yield f, cols


def read_all_quality(group: Any) -> dict[int, np.ndarray]:
    """Ghost probability of every frame in a ``quality`` group, either layout
    (blocks win over per-frame arrays)."""
    blocks = BlockSet(group)
    out: dict[int, np.ndarray] = {}
    for k in group.keys():
        f = frame_number(k)
        if f is not None:
            out[f] = np.asarray(group[k])
    for f in blocks.frames():
        out[f] = blocks.read(f)["ghost"]  # type: ignore[index]
    return out

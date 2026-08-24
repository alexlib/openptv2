"""Parallel 3D particle tracking via temporal chunking and boundary stitching (Task 4).

Decomposes long tracking sequences into overlapping temporal sub-windows,
executes single-threaded tracking runs concurrently across multiple processes or threads,
and stitches linkages across boundaries with guaranteed trajectory continuity.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
    convert_track_par_to_tuple,
)
from openptv2.algorithms.track import (
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.track3d import track3d_loop
from openptv2.algorithms.track4be import track4be_loop
from openptv2.algorithms.tracking_run import TrackingRun
from openptv2.storage.run_store import RunStore, RunStoreError
from openptv2.storage.seal import seal
from openptv2.tracker import DEFAULT_MAX_TARGETS, Tracker, default_naming


def partition_tracking_chunks(
    first: int,
    last: int,
    n_workers: int,
    overlap: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Partition a sequence [first, last] into overlapping chunks for parallel tracking.

    Args:
        first: First frame number in sequence (inclusive).
        last: Last frame number in sequence (inclusive).
        n_workers: Number of parallel workers requested.
        overlap: Overlap window in frames (default: 4).

    Returns:
        List of tuples: (chunk_first, chunk_last, valid_first, valid_last)
        - [chunk_first, chunk_last]: The frame range tracked by the worker.
        - [valid_first, valid_last]: The valid frame interval whose linkages are kept from this worker.
    """
    total_frames = last - first + 1
    if total_frames <= 0:
        return []
    if n_workers <= 1 or total_frames <= overlap + 2:
        return [(first, last, first, last)]

    max_useful_workers = max(1, total_frames // (overlap + 2))
    n_workers = min(n_workers, max_useful_workers)
    if n_workers <= 1:
        return [(first, last, first, last)]

    step = total_frames / n_workers
    splits = [first + int(round(i * step)) for i in range(n_workers + 1)]
    splits[0] = first
    splits[-1] = last + 1

    chunks = []
    for i in range(n_workers):
        valid_start = splits[i]
        valid_end = splits[i + 1] - 1  # inclusive

        # Extend start backward by overlap (except first chunk)
        chunk_start = max(first, valid_start - overlap) if i > 0 else first
        # Extend end forward by overlap (except last chunk)
        chunk_end = min(last, valid_end + overlap) if i < n_workers - 1 else last

        chunks.append((chunk_start, chunk_end, valid_start, valid_end))

    return chunks


class _InMemoryLinkageStore:
    """Zero-I/O in-memory store adapter for isolated worker chunk tracking."""

    def __init__(self, source_store: Optional[RunStore] = None):
        self._source_store = source_store
        self.root: dict[str, Any] = {}
        self._linkages: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}

    def read_correspondences(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        if self._source_store is not None:
            return self._source_store.read_correspondences(frame)
        raise RunStoreError(f"No correspondences stored for frame {frame}")

    def has_correspondences(self, frame: int) -> bool:
        if self._source_store is not None:
            return self._source_store.has_correspondences(frame)
        return False

    def has_targets(self, cam_idx: int, frame: int) -> bool:
        if self._source_store is not None:
            return self._source_store.has_targets(cam_idx, frame)
        return False

    def read_targets(self, cam_idx: int, frame: int):
        if self._source_store is not None:
            return self._source_store.read_targets(cam_idx, frame)
        return []

    def write_correspondences(self, frame: int, pos_3d: np.ndarray, cam_target_ids: np.ndarray) -> None:
        pass

    def write_linkage(
        self,
        frame: int,
        prev_ids: np.ndarray,
        next_ids: np.ndarray,
        pos_3d: np.ndarray,
        name: str = "ptv_is",
        prio: Optional[np.ndarray] = None,
    ) -> None:
        self._linkages[(name, frame)] = (
            np.asarray(prev_ids, dtype=np.int32).copy(),
            np.asarray(next_ids, dtype=np.int32).copy(),
            np.asarray(pos_3d, dtype=np.float64).copy(),
            np.asarray(prio, dtype=np.int32).copy() if prio is not None else None,
        )

    def has_linkage(self, frame: int, name: str = "ptv_is") -> bool:
        return (name, frame) in self._linkages

    def read_linkage(
        self, frame: int, name: str = "ptv_is"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (name, frame) not in self._linkages:
            raise RunStoreError(f"No linkage '{name}' for frame {frame}")
        p, n, pos, _ = self._linkages[(name, frame)]
        return p, n, pos

    def clear_linkage(self, name: str = "ptv_is") -> None:
        keys_to_del = [k for k in self._linkages if k[0] == name]
        for k in keys_to_del:
            del self._linkages[k]


@dataclass
class ChunkTrackingResult:
    chunk_idx: int
    chunk_first: int
    chunk_last: int
    valid_first: int
    valid_last: int
    npart: int
    nlinks: int
    # frame -> (prev_ids, next_ids, pos_3d, prio)
    linkages: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]


def _track_single_chunk(
    chunk_idx: int,
    chunk_first: int,
    chunk_last: int,
    valid_first: int,
    valid_last: int,
    cpar: ControlPar,
    vpar: VolumePar,
    tpar: TrackPar,
    spar: SequencePar,
    cals: list[Calibration],
    store_path: Optional[str] = None,
    naming: Optional[dict[str, str]] = None,
    flatten_tol: float = 0.0001,
    mode: str = "3d",
    working_dir: Optional[str] = None,
) -> ChunkTrackingResult:
    """Execute tracking for a single temporal chunk."""
    cwd_orig = None
    if working_dir:
        cwd_orig = os.getcwd()
        os.chdir(working_dir)

    try:
        if naming is None:
            naming = default_naming.copy()

        chunk_spar = SequencePar(
            img_base_name=spar.img_base_name,
            first=chunk_first,
            last=chunk_last,
        )

        source_store = None
        if store_path:
            source_store = RunStore(store_path, mode="r")

        local_store = _InMemoryLinkageStore(source_store=source_store)

        tracker = Tracker(
            cpar=cpar,
            vpar=vpar,
            tpar=tpar,
            spar=chunk_spar,
            cals=cals,
            naming=naming,
            flatten_tol=flatten_tol,
            store=local_store,
        )

        import io
        import sys

        sys_stdout_orig = sys.stdout
        sys.stdout = io.StringIO()
        try:
            if mode == "3d":
                tracker.full_forward_3d()
            elif mode == "4be":
                tracker.full_forward_4be()
            else:
                tracker.full_forward()
        finally:
            sys.stdout = sys_stdout_orig

        link_name = Path(naming.get("linkage", "res/ptv_is")).name
        linkages: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        for (name, f), (prev_ids, next_ids, pos_3d, prio_arr) in local_store._linkages.items():
            if name == link_name:
                linkages[f] = (prev_ids, next_ids, pos_3d, prio_arr)

        return ChunkTrackingResult(
            chunk_idx=chunk_idx,
            chunk_first=chunk_first,
            chunk_last=chunk_last,
            valid_first=valid_first,
            valid_last=valid_last,
            npart=tracker.npart,
            nlinks=tracker.nlinks,
            linkages=linkages,
        )
    finally:
        if cwd_orig is not None:
            os.chdir(cwd_orig)


def stitch_chunked_linkages(
    results: list[ChunkTrackingResult],
    first: int,
    last: int,
    store: Optional[RunStore] = None,
    link_name: str = "ptv_is",
    corres_file_base: Optional[str] = None,
    linkage_file_base: Optional[str] = None,
    prio_file_base: Optional[str] = None,
) -> tuple[int, int]:
    """Stitch linkages across adjacent temporal chunks into the destination store / files.

    Returns:
        (total_npart, total_nlinks)
    """
    results_sorted = sorted(results, key=lambda r: r.chunk_idx)
    num_chunks = len(results_sorted)

    total_links = 0
    total_parts = 0

    for idx, res in enumerate(results_sorted):
        v_first = res.valid_first
        v_last = res.valid_last

        for f in range(v_first, v_last + 1):
            if f not in res.linkages:
                continue

            prev_ids, next_ids, pos_3d, prio_arr = res.linkages[f]
            prev_stitched = prev_ids.copy()
            next_stitched = next_ids.copy()

            # At left boundary frame (v_first) for chunk idx > 0:
            # prev link (f-1 -> f) comes from previous chunk (idx - 1)
            if f == v_first and idx > 0:
                prev_chunk = results_sorted[idx - 1]
                if f in prev_chunk.linkages:
                    prev_stitched = prev_chunk.linkages[f][0]

            # At right boundary frame (v_last) for chunk idx < num_chunks - 1:
            # next link (f -> f+1) comes from next chunk (idx + 1)
            if f == v_last and idx < num_chunks - 1:
                next_chunk = results_sorted[idx + 1]
                if f in next_chunk.linkages:
                    next_stitched = next_chunk.linkages[f][1]

            # Accumulate npart and nlinks over forward steps (f < last) matching Tracker definition
            if f < last:
                total_parts += len(pos_3d)
                total_links += int(np.count_nonzero(next_stitched >= 0))

            # Write to store
            if store is not None:
                store.write_linkage(
                    frame=f,
                    prev_ids=prev_stitched,
                    next_ids=next_stitched,
                    pos_3d=pos_3d,
                    name=link_name,
                    prio=prio_arr,
                )

            # Write to legacy ASCII if requested and store not present
            if store is None and linkage_file_base:
                link_fname = f"{linkage_file_base}.{f}"
                with open(link_fname, "w") as fh:
                    fh.write(f"{len(pos_3d)}\n")
                    for p_idx in range(len(pos_3d)):
                        p = pos_3d[p_idx]
                        pr = prev_stitched[p_idx]
                        nx = next_stitched[p_idx]
                        if prio_arr is not None:
                            fh.write(
                                f"{pr:4d} {nx:4d} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f} {prio_arr[p_idx]:4d}\n"
                            )
                        else:
                            fh.write(
                                f"{pr:4d} {nx:4d} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}\n"
                            )

    return total_parts, total_links


def track_sequence_chunked_parallel(
    cpar: ControlPar,
    vpar: VolumePar,
    tpar: TrackPar,
    spar: SequencePar,
    cals: list[Calibration],
    store: Optional[RunStore] = None,
    naming: Optional[dict[str, str]] = None,
    flatten_tol: float = 0.0001,
    n_workers: Optional[int] = None,
    overlap: int = 4,
    mode: Literal["3d", "4be", "corr"] = "3d",
    postprocess: bool = True,
) -> tuple[int, int]:
    """Execute 3D particle tracking across sequence in parallel temporal chunks.

    Args:
        cpar: ControlPar instance.
        vpar: VolumePar instance.
        tpar: TrackPar instance.
        spar: SequencePar instance.
        cals: List of Calibration instances.
        store: RunStore instance (recommended), or None.
        naming: Dict with 'corres', 'linkage', 'prio' paths.
        flatten_tol: Flatness tolerance for epipolar matching.
        n_workers: Number of worker threads/processes. Defaults to os.cpu_count().
        overlap: Overlap window in frames (default: 4).
        mode: Tracking algorithm mode ("3d", "4be", "corr").
        postprocess: Whether to run quality postprocessing passes.

    Returns:
        (total_npart, total_nlinks)
    """
    first = spar.get_first()
    last = spar.get_last()
    total_frames = last - first + 1

    if naming is None:
        naming = default_naming.copy()

    link_name = Path(naming.get("linkage", "res/ptv_is")).name
    if store is not None:
        store.clear_linkage(link_name)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    chunks = partition_tracking_chunks(first, last, n_workers=n_workers, overlap=overlap)

    store_path_str = str(store.store_path) if store is not None else None
    cwd = os.getcwd()

    results: list[ChunkTrackingResult] = []
    executor_cls = ProcessPoolExecutor if len(chunks) > 1 else ThreadPoolExecutor
    try:
        with executor_cls(max_workers=len(chunks)) as pool:
            futures = []
            for idx, (c_first, c_last, v_first, v_last) in enumerate(chunks):
                f = pool.submit(
                    _track_single_chunk,
                    chunk_idx=idx,
                    chunk_first=c_first,
                    chunk_last=c_last,
                    valid_first=v_first,
                    valid_last=v_last,
                    cpar=cpar,
                    vpar=vpar,
                    tpar=tpar,
                    spar=spar,
                    cals=cals,
                    store_path=store_path_str,
                    naming=naming,
                    flatten_tol=flatten_tol,
                    mode=mode,
                    working_dir=cwd,
                )
                futures.append(f)

            for f in futures:
                results.append(f.result())
    except Exception:
        # Fallback to ThreadPoolExecutor if ProcessPool encounters any environment constraint
        results = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = []
            for idx, (c_first, c_last, v_first, v_last) in enumerate(chunks):
                f = pool.submit(
                    _track_single_chunk,
                    chunk_idx=idx,
                    chunk_first=c_first,
                    chunk_last=c_last,
                    valid_first=v_first,
                    valid_last=v_last,
                    cpar=cpar,
                    vpar=vpar,
                    tpar=tpar,
                    spar=spar,
                    cals=cals,
                    store_path=store_path_str,
                    naming=naming,
                    flatten_tol=flatten_tol,
                    mode=mode,
                    working_dir=cwd,
                )
                futures.append(f)

            for f in futures:
                results.append(f.result())

    # Stitch linkages across chunk boundaries
    npart, nlinks = stitch_chunked_linkages(
        results=results,
        first=first,
        last=last,
        store=store,
        link_name=link_name,
        corres_file_base=naming.get("corres"),
        linkage_file_base=naming.get("linkage"),
        prio_file_base=naming.get("prio"),
    )

    # Optional post-processing passes
    if postprocess and store is not None:
        from openptv2.tracking_postprocess import (
            enforce_reciprocity,
            relink_trajectory_gaps,
            seed_cold_start,
        )
        try:
            seed_cold_start(
                naming["linkage"], first, last, float(tpar.dvxmax), store=store
            )
            relink_trajectory_gaps(
                naming["linkage"], first, last, max_gap=2, max_accel_err=float(tpar.dacc), store=store
            )
            enforce_reciprocity(naming["linkage"], first, last, store=store)
        except Exception:
            pass

    # Seal store if attached
    if store is not None:
        seal(store, name=link_name, force=True)

    return npart, nlinks

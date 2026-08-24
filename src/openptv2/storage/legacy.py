"""Import an existing ASCII run into a :class:`RunStore`, and export one back
out byte-compatibly.

This is the backward-compatibility boundary for Phase A: nothing else reads
or writes ASCII through this module's callers, but any external tool that
still expects ``ptv_is.*``/``rt_is.*``/``*_targets`` can get them via
``export_run``. The exact format strings reproduced here are recorded in
``docs/plans/2026-08-14-storage-formats-as-built.md`` and were taken directly
from ``algorithms/tracking_frame_buf.py`` (targets) and ``gui/ptv.py`` /
``tracking_frame_buf.write_path_frame`` (rt_is / ptv_is / added).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from openptv2.algorithms.tracking_frame_buf import read_targets as _read_targets_ascii

from .run_store import RunStore, RunStoreError, resolve_store_path

_TARGET_RE = re.compile(r"cam(\d+)\.(\d+)_targets$")
_RT_IS_RE = re.compile(r"rt_is\.(\d+)$")


def _discover_cams(img_dir: Path) -> list[int]:
    """1-based on-disk camera numbers, e.g. cam1.. -> [1, 2, 3, 4]."""
    nums = set()
    for p in img_dir.glob("cam*_targets"):
        m = _TARGET_RE.search(p.name)
        if m:
            nums.add(int(m.group(1)))
    return sorted(nums)


def _discover_frames(res_dir: Path) -> list[int]:
    nums = set()
    for p in res_dir.glob("rt_is.*"):
        m = _RT_IS_RE.search(p.name)
        if m:
            nums.add(int(m.group(1)))
    return sorted(nums)


def _discover_target_frames(img_dir: Path) -> list[int]:
    """Frame numbers discovered from cam*_targets files."""
    nums = set()
    for p in img_dir.glob("cam*_targets"):
        m = _TARGET_RE.search(p.name)
        if m:
            nums.add(int(m.group(2)))
    return sorted(nums)


def _load_rt_is(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 3)), np.zeros((0, 0), dtype=np.int32)
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if data.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 0), dtype=np.int32)
    pos = data[:, 1:4]
    cam_ids = data[:, 4:].astype(np.int32)
    return pos, cam_ids


def _load_linkage(
    path: Path,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Parse a ``prev next x y z [prio]`` file. The 6th ``prio`` column is
    present on the tracker's prio output (conventionally ``res/added.*`` --
    see :meth:`RunStore.write_linkage`), absent on plain linkage (``ptv_is``).
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if data.size == 0:
        return None
    prev = data[:, 0].astype(np.int32)
    nxt = data[:, 1].astype(np.int32)
    pos = data[:, 2:5]
    prio = data[:, 5].astype(np.int32) if data.shape[1] >= 6 else None
    return prev, nxt, pos, prio


def import_run(
    experiment_root: Union[str, Path],
    store_path: Optional[Union[str, Path]] = None,
    remove_ascii: bool = False,
) -> RunStore:
    """Ingest an existing ASCII run (``img/*_targets``, ``res/rt_is.*``,
    ``res/ptv_is.*``, ``res/added.*``) into a new/updated RunStore.

    Frame keys are normalised to the store's fixed-width convention as part
    of the import, fixing the inconsistent padding the legacy Zarr mirror had
    (unpadded for targets/correspondences, ``:05d`` for linkage).

    Args:
        experiment_root: Path to the experiment directory containing img/ and/or res/
        store_path: Optional output path for the Zarr store (default: res/run.zarr)
        remove_ascii: If True, delete imported ASCII files after successful ingestion
    """
    root = Path(experiment_root)
    img_dir = root / "img"
    res_dir = root / "res"
    if not img_dir.is_dir() and not res_dir.is_dir():
        raise RunStoreError(f"{root} does not look like a PTV run (missing img/ and res/)")

    cams = _discover_cams(img_dir) if img_dir.is_dir() else []
    frames_rt = _discover_frames(res_dir) if res_dir.is_dir() else []
    frames_targ = _discover_target_frames(img_dir) if img_dir.is_dir() else []
    all_frames = sorted(set(frames_rt) | set(frames_targ))

    if not all_frames:
        raise RunStoreError(f"No targets or rt_is.* files found under {root}")

    res_dir.mkdir(parents=True, exist_ok=True)
    store = RunStore(store_path or resolve_store_path(root), mode="a")
    files_to_remove = []

    for frame in all_frames:
        targets_per_cam = []
        for cam in cams:
            target_file = img_dir / f"cam{cam}.{frame:04d}_targets"
            if not target_file.exists():
                target_file = img_dir / f"cam{cam}.{frame}_targets"
            if target_file.exists():
                file_base = str(img_dir / f"cam{cam}.")
                targets = _read_targets_ascii(file_base, frame, cam_idx=cam - 1)
                store.write_targets(cam - 1, frame, targets)
                targets_per_cam.append(len(targets))
                files_to_remove.append(target_file)
            else:
                targets_per_cam.append(0)

        rt_path = res_dir / f"rt_is.{frame}"
        has_corres = rt_path.exists()
        pos, cam_ids = _load_rt_is(rt_path)
        if has_corres:
            store.write_correspondences(frame, pos, cam_ids)
            files_to_remove.append(rt_path)

        ptv_is_path = res_dir / f"ptv_is.{frame}"
        ptv_is = _load_linkage(ptv_is_path)
        if ptv_is is not None:
            p_prev, p_next, p_pos, p_prio = ptv_is
            store.write_linkage(frame, p_prev, p_next, p_pos, name="ptv_is", prio=p_prio)
            files_to_remove.append(ptv_is_path)

        added_path = res_dir / f"added.{frame}"
        added = _load_linkage(added_path)
        if added is not None:
            a_prev, a_next, a_pos, a_prio = added
            store.write_linkage(frame, a_prev, a_next, a_pos, name="added", prio=a_prio)
            files_to_remove.append(added_path)

        n_targets = np.array(targets_per_cam, dtype=np.int32)
        clique_size = (cam_ids != -1).sum(axis=1) if cam_ids.size else np.zeros(0, dtype=np.int32)
        n_quads = int((clique_size == 4).sum())
        n_trips = int((clique_size == 3).sum())
        n_pairs = int((clique_size == 2).sum())
        cam_seen = (
            (cam_ids != -1).sum(axis=0).astype(np.int32)
            if cam_ids.size
            else np.zeros(len(cams), dtype=np.int32)
        )
        n_links = int((ptv_is[1] >= 0).sum()) if ptv_is is not None else 0
        store.write_stats(
            frame,
            n_targets=n_targets,
            cam_seen=cam_seen,
            n_quads=n_quads,
            n_trips=n_trips,
            n_pairs=n_pairs,
            n_corres=len(pos),
            n_links=n_links,
        )

    if remove_ascii:
        for f in files_to_remove:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

    return store


convert_ascii_to_zarr = import_run


def _write_targets_ascii(path: Path, arr: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"{len(arr)}\n")
        for row in arr:
            f.write(
                f"{int(row[0]):4d} {row[1]:9.4f} {row[2]:9.4f} "
                f"{int(row[3]):5d} {int(row[4]):5d} {int(row[5]):5d} "
                f"{int(row[6]):5d} {int(row[7]):5d}\n"
            )


def _write_rt_is_ascii(path: Path, pos: np.ndarray, cam_ids: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        for i, (p, ids) in enumerate(zip(pos, cam_ids)):
            id_str = " ".join(f"{int(c):4d}" for c in ids)
            f.write(f"{i + 1:4d} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f} {id_str}\n")


def _write_linkage_ascii(
    path: Path,
    prev: np.ndarray,
    nxt: np.ndarray,
    pos: np.ndarray,
    prio: Optional[np.ndarray] = None,
) -> None:
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        if prio is None:
            for p_i, n_i, p in zip(prev, nxt, pos):
                f.write(f"{int(p_i):4d} {int(n_i):4d} {p[0]:10.3f} {p[1]:10.3f} {p[2]:10.3f}\n")
        else:
            for p_i, n_i, p, pr in zip(prev, nxt, pos, prio):
                f.write(
                    f"{int(p_i):4d} {int(n_i):4d} {p[0]:10.3f} {p[1]:10.3f} {p[2]:10.3f} {int(pr):d}\n"
                )


def export_run(store: RunStore, experiment_root: Union[str, Path]) -> None:
    """Regenerate ``img/*_targets``, ``res/rt_is.*``, ``res/ptv_is.*``,
    ``res/added.*`` from a RunStore, byte-compatible with the legacy writers.
    """
    root = Path(experiment_root)
    img_dir = root / "img"
    res_dir = root / "res"
    img_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    cams = store.target_cameras()
    frames = store.frames()

    for frame in frames:
        for cam in cams:
            targets = store.read_targets(cam, frame)
            arr = np.array(
                [[t.pnr(), t.x(), t.y(), t.n, t.nx, t.ny, t.sumg, t.tnr()] for t in targets],
                dtype=np.float64,
            )
            _write_targets_ascii(img_dir / f"cam{cam + 1}.{frame}_targets", arr)

        if store.has_correspondences(frame):
            pos, cam_ids = store.read_correspondences(frame)
            _write_rt_is_ascii(res_dir / f"rt_is.{frame}", pos, cam_ids)

        if store.has_linkage(frame, "ptv_is"):
            prev, nxt, pos = store.read_linkage(frame, "ptv_is")
            # ptv_is.<frame> never carries a prio column in the legacy
            # format, even though the live pipeline's own "ptv_is" store
            # group does (write_path_frame embeds prio there directly,
            # since added.* is just that same data plus a prio column --
            # see write_linkage's docstring).
            _write_linkage_ascii(res_dir / f"ptv_is.{frame}", prev, nxt, pos, prio=None)

            if store.has_linkage(frame, "added"):
                a_prev, a_nxt, a_pos = store.read_linkage(frame, "added")
                a_prio = store.read_prio(frame, "added")
                _write_linkage_ascii(res_dir / f"added.{frame}", a_prev, a_nxt, a_pos, a_prio)
            else:
                prio = store.read_prio(frame, "ptv_is")
                if prio is not None:
                    _write_linkage_ascii(res_dir / f"added.{frame}", prev, nxt, pos, prio)


def main(argv=None) -> int:
    """CLI entrypoint to convert legacy ASCII files to a unified Zarr store."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="openptv2-convert-legacy",
        description="Convert legacy OpenPTV ASCII run files (*_targets, rt_is.*, ptv_is.*, added.*) to a unified Zarr store.",
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Experiment folder containing img/ and/or res/ directories.",
    )
    parser.add_argument(
        "--store",
        "-s",
        type=str,
        default=None,
        help="Optional destination path for the .zarr store (defaults to <folder>/res/run.zarr).",
    )
    parser.add_argument(
        "--remove-ascii",
        "-r",
        action="store_true",
        help="Delete legacy ASCII files after successful conversion to Zarr.",
    )

    args = parser.parse_args(argv)
    folder_path = Path(args.folder).resolve()

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Directory '{folder_path}' does not exist.", file=sys.stderr)
        return 1

    try:
        store = import_run(folder_path, store_path=args.store, remove_ascii=args.remove_ascii)
        n_frames = len(store.frames())
        print(f"Successfully converted {n_frames} frames to Zarr store: {store.store_path}")
        if args.remove_ascii:
            print("Legacy ASCII files removed successfully.")
        return 0
    except Exception as exc:
        print(f"Error during conversion: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

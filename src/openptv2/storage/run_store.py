"""Unified per-run Zarr store -- Phase A of the storage redesign.

See ``docs/plans/2026-08-14-storage-formats-as-built.md`` for the legacy
per-frame ASCII formats (``*_targets``, ``rt_is.*``, ``ptv_is.*``,
``added.*``) this is designed to sit alongside, and the defects recorded
there that this module fixes:

- No bare ``except`` around a write. A failed write raises ``RunStoreError``
  instead of silently dropping a frame.
- One store-path resolver (:func:`resolve_store_path`) instead of the four
  divergent candidate-path lists the project had grown, and no cwd-relative
  paths.
- Frame keys are zero-padded consistently (``frame_000000``) so lexical and
  numeric sort agree past 5-digit frame numbers.

Per-frame groups (``targets/``, ``correspondences/``, ``linkage/``) are the
permanent format: each frame's arrays are independent Zarr arrays, so
parallel workers can write distinct frames with no locking. ``seal()`` (see
``seal.py``) is the only code path in the project that walks the
``prev``/``next`` linkage graph; everything else reads one frame directly, or
reads the sealed ``trajectories/`` / ``traj/`` groups it produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr

from openptv2.algorithms.tracking_frame_buf import Target, TargetArray

STORE_DIRNAME = "run.zarr"
FRAME_KEY_WIDTH = 6  # frame_000000 .. frame_999999


def _frame_key(frame: int) -> str:
    return f"frame_{frame:0{FRAME_KEY_WIDTH}d}"


def _frame_num(key: str) -> int:
    return int(key.split("_", 1)[1])


def resolve_store_path(experiment_root: Union[str, Path]) -> Path:
    """Compute the ``run.zarr`` path for an experiment, the one place this
    project should do so.

    Accepts the experiment root, its ``res/`` directory, or the store path
    itself, and always returns an absolute-relative-to-input path -- never a
    cwd-relative ``Path("res/run.zarr")``.
    """
    root = Path(experiment_root)
    if root.name == STORE_DIRNAME:
        return root
    if root.name == "res":
        return root / STORE_DIRNAME
    return root / "res" / STORE_DIRNAME


class RunStoreError(RuntimeError):
    """A RunStore operation failed.

    Raised, never swallowed: the legacy dual-write path silently dropped
    Zarr writes on any exception (``tracking_frame_buf.py`` ``write_path_frame``
    wraps its Zarr mirror in a bare ``except: pass``), which let a run lose
    frames while reporting success. RunStore does not repeat that mistake.
    """


class RunStore:
    """Single Zarr store for one PTV run.

    Holds detected targets, 3D correspondences, tracking linkage, and (once
    :func:`openptv2.storage.seal.seal` has run) the trajectory index and the
    flat trajectory cache. Also holds a per-frame statistics manifest so
    "how many tracers", "which camera drops out of quadruplets", etc. can be
    answered without reading bulk data.
    """

    _GROUPS = (
        "targets",
        "correspondences",
        "linkage",
        "traj",
        "trajectories",
        "stats",
        "meta",
    )

    def __init__(self, store_path: Union[str, Path], mode: str = "a"):
        self.store_path = Path(store_path)
        try:
            self.root = zarr.open_group(str(self.store_path), mode=mode)
        except Exception as exc:  # zarr's own open-mode/group-exists errors
            raise RunStoreError(
                f"Failed to open run store at {self.store_path!s} (mode={mode!r}): {exc}"
            ) from exc

        if mode in ("w", "w-", "a", "r+"):
            for name in self._GROUPS:
                if name not in self.root:
                    self.root.create_group(name)
            meta = self.root["meta"]
            if "schema_version" not in meta.attrs:
                meta.attrs["schema_version"] = 1
                meta.attrs["sealed"] = False
                meta.attrs["source_hash"] = None
                meta.attrs["raw_units"] = "mm"  # matches legacy ASCII (rt_is/ptv_is)
                meta.attrs["trajectories_units"] = "m"  # matches flowtracks

    @classmethod
    def open(cls, experiment_root: Union[str, Path], mode: str = "a") -> "RunStore":
        return cls(resolve_store_path(experiment_root), mode=mode)

    # -- meta -----------------------------------------------------------

    @property
    def meta(self) -> dict:
        return dict(self.root["meta"].attrs)

    @property
    def sealed(self) -> bool:
        return bool(self.root["meta"].attrs.get("sealed", False))

    def _mark_unsealed(self) -> None:
        """Any write to targets/correspondences/linkage invalidates the seal."""
        if self.root["meta"].attrs.get("sealed"):
            self.root["meta"].attrs["sealed"] = False

    # -- targets ----------------------------------------------------------

    @staticmethod
    def _targets_to_array(targets) -> np.ndarray:
        if isinstance(targets, np.ndarray):
            return np.asarray(targets, dtype=np.float64)
        arr = np.zeros((len(targets), 8), dtype=np.float64)
        for i, t in enumerate(targets):
            if not isinstance(t, Target):
                raise RunStoreError(
                    f"write_targets expects Target instances or an (N,8) ndarray, "
                    f"got {type(t)!r} at index {i}"
                )
            arr[i] = [t.pnr(), t.x(), t.y(), t.n, t.nx, t.ny, t.sumg, t.tnr()]
        return arr

    def write_targets(self, cam: int, frame: int, targets) -> None:
        """Write detected 2D targets for one camera/frame.

        ``targets``: a ``TargetArray``/list of ``Target``, or an ``(N, 8)``
        ndarray of ``[pnr, x, y, n, nx, ny, sumg, tnr]``.
        """
        arr = self._targets_to_array(targets)
        cam_group = self.root["targets"].require_group(f"cam_{cam}")
        try:
            cam_group.create_array(_frame_key(frame), data=arr, overwrite=True)
        except Exception as exc:
            raise RunStoreError(
                f"Failed to write targets for cam={cam} frame={frame}: {exc}"
            ) from exc
        self._mark_unsealed()

    def read_targets(self, cam: int, frame: int) -> TargetArray:
        key = f"targets/cam_{cam}/{_frame_key(frame)}"
        if key not in self.root:
            raise RunStoreError(f"No targets stored for cam={cam} frame={frame}")
        raw = np.asarray(self.root[key])
        tarr = TargetArray(len(raw))
        for i, row in enumerate(raw):
            tarr[i].set_pnr(int(row[0]))
            tarr[i].set_pos((row[1], row[2]))
            tarr[i].set_pixel_counts(int(row[3]), int(row[4]), int(row[5]))
            tarr[i].set_sum_grey_value(int(row[6]))
            tarr[i].set_tnr(int(row[7]))
        return tarr

    def has_targets(self, cam: int, frame: int) -> bool:
        return f"targets/cam_{cam}/{_frame_key(frame)}" in self.root

    def target_cameras(self) -> list[int]:
        grp = self.root["targets"]
        return sorted(int(k.split("_", 1)[1]) for k in grp.keys() if k.startswith("cam_"))

    # -- correspondences ----------------------------------------------------

    def write_correspondences(
        self, frame: int, pos_3d: np.ndarray, cam_target_ids: np.ndarray
    ) -> None:
        """``pos_3d``: (N,3) mm. ``cam_target_ids``: (N,C) int, -1 = not seen."""
        pos_3d = np.asarray(pos_3d, dtype=np.float64)
        cam_target_ids = np.asarray(cam_target_ids, dtype=np.int32)
        if pos_3d.shape[0] != cam_target_ids.shape[0]:
            raise RunStoreError(
                f"correspondences row-count mismatch for frame {frame}: "
                f"pos has {pos_3d.shape[0]}, cam_target_ids has {cam_target_ids.shape[0]}"
            )
        combined = np.hstack([pos_3d, cam_target_ids.astype(np.float64)])
        try:
            self.root["correspondences"].create_array(
                _frame_key(frame), data=combined, overwrite=True
            )
        except Exception as exc:
            raise RunStoreError(
                f"Failed to write correspondences for frame {frame}: {exc}"
            ) from exc
        self._mark_unsealed()

    def read_correspondences(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        key = f"correspondences/{_frame_key(frame)}"
        if key not in self.root:
            raise RunStoreError(f"No correspondences stored for frame {frame}")
        data = np.asarray(self.root[key])
        return data[:, :3], data[:, 3:].astype(np.int32)

    def has_correspondences(self, frame: int) -> bool:
        return f"correspondences/{_frame_key(frame)}" in self.root

    # -- linkage --------------------------------------------------------

    def write_linkage(
        self,
        frame: int,
        prev_ids: np.ndarray,
        next_ids: np.ndarray,
        pos_3d: np.ndarray,
        name: str = "ptv_is",
        prio: Optional[np.ndarray] = None,
    ) -> None:
        """``prio``: optional per-particle priority column. The legacy
        ``res/added.*`` stream is actually the tracker's *prio* output
        (``default_naming['prio'] = 'res/added'``, ``tracker.py:14-18``), a
        6-column file (``prev next x y z prio``), not a second 5-column
        linkage pass -- confirmed by round-tripping ``test_data/test_cavity``.
        """
        prev_ids = np.asarray(prev_ids, dtype=np.int32)
        next_ids = np.asarray(next_ids, dtype=np.int32)
        pos_3d = np.asarray(pos_3d, dtype=np.float64)
        n = pos_3d.shape[0]
        if prev_ids.shape[0] != n or next_ids.shape[0] != n:
            raise RunStoreError(
                f"linkage row-count mismatch for frame {frame} ({name}): "
                f"prev={prev_ids.shape[0]} next={next_ids.shape[0]} pos={n}"
            )
        try:
            fg = self.root["linkage"].require_group(name).require_group(_frame_key(frame))
            fg.create_array("prev", data=prev_ids, overwrite=True)
            fg.create_array("next", data=next_ids, overwrite=True)
            fg.create_array("pos", data=pos_3d, overwrite=True)
            if prio is not None:
                fg.create_array(
                    "prio", data=np.asarray(prio, dtype=np.int32), overwrite=True
                )
        except Exception as exc:
            raise RunStoreError(
                f"Failed to write linkage '{name}' for frame {frame}: {exc}"
            ) from exc
        self._mark_unsealed()

    def read_linkage(
        self, frame: int, name: str = "ptv_is"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = f"linkage/{name}/{_frame_key(frame)}"
        if key not in self.root:
            raise RunStoreError(f"No linkage '{name}' stored for frame {frame}")
        fg = self.root[key]
        return np.asarray(fg["prev"]), np.asarray(fg["next"]), np.asarray(fg["pos"])

    def read_prio(self, frame: int, name: str = "ptv_is") -> Optional[np.ndarray]:
        key = f"linkage/{name}/{_frame_key(frame)}"
        if key not in self.root or "prio" not in self.root[key]:
            return None
        return np.asarray(self.root[key]["prio"])

    def has_linkage(self, frame: int, name: str = "ptv_is") -> bool:
        return f"linkage/{name}/{_frame_key(frame)}" in self.root

    def set_trajid(self, frame: int, name: str, trajid: np.ndarray) -> None:
        """Write the trajid labelling back into a linkage frame group.
        Called only by :func:`openptv2.storage.seal.seal`."""
        key = f"linkage/{name}/{_frame_key(frame)}"
        if key not in self.root:
            raise RunStoreError(f"No linkage '{name}' stored for frame {frame}")
        self.root[key].create_array(
            "trajid", data=np.asarray(trajid, dtype=np.int32), overwrite=True
        )

    def linkage_names(self) -> list[str]:
        return sorted(self.root["linkage"].keys())

    def frames(self, source: Optional[str] = None) -> list[int]:
        """Sorted frame numbers present in the store.

        Replaces filesystem globbing for frame discovery (e.g. ``rt_is.*``),
        which silently finds zero frames against a store. Reads
        correspondences by default, falling back to the ``ptv_is`` linkage,
        then to any camera's targets.
        """
        candidates = []
        if source is not None:
            candidates = [("correspondences" if source == "correspondences" else None)]
        for grp_path in (
            [source] if source else ["correspondences", "linkage/ptv_is"]
        ):
            if grp_path not in self.root:
                continue
            keys = [k for k in self.root[grp_path].keys() if k.startswith("frame_")]
            if keys:
                return sorted(_frame_num(k) for k in keys)
        cams = self.target_cameras()
        if cams:
            keys = [
                k
                for k in self.root[f"targets/cam_{cams[0]}"].keys()
                if k.startswith("frame_")
            ]
            return sorted(_frame_num(k) for k in keys)
        return []

    # -- trajectory index (written by seal) ------------------------------

    def write_traj_index(
        self,
        trajid: np.ndarray,
        first: np.ndarray,
        last: np.ndarray,
        length: np.ndarray,
    ) -> None:
        """Note: unlike the legacy readers (``storage/zarr_store.py``'s
        ``read_zarr_trajectories``, and flowtracks itself), this index labels
        *every* particle with a trajid, including length-1 singletons that
        never linked to another frame. Nothing is silently dropped from the
        store. A caller that wants flowtracks' contract (trajectories of at
        least 2 points) filters on ``length >= 2`` -- see
        ``test_traj_index_matches_legacy_reader_after_singleton_filter`` in
        ``tests/unit/test_run_store.py`` for the exact equivalence."""
        grp = self.root["traj"]
        grp.create_array("trajid", data=np.asarray(trajid, dtype=np.int32), overwrite=True)
        grp.create_array("first", data=np.asarray(first, dtype=np.int32), overwrite=True)
        grp.create_array("last", data=np.asarray(last, dtype=np.int32), overwrite=True)
        grp.create_array("length", data=np.asarray(length, dtype=np.int32), overwrite=True)

    def traj_index(self) -> dict[str, np.ndarray]:
        grp = self.root["traj"]
        if "trajid" not in grp:
            raise RunStoreError("No trajectory index -- run seal() first.")
        return {k: np.asarray(grp[k]) for k in ("trajid", "first", "last", "length")}

    # -- trajectories (derived flat cache, written by seal) --------------

    def write_trajectories(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        accel: np.ndarray,
        time: np.ndarray,
        trajid: np.ndarray,
    ) -> None:
        """Flat particle-observation table, one row per (trajectory, frame),
        sorted by ``(trajid, time)``. Positions/velocities in METRES -- this
        is the array set and unit convention flowtracks and openptv-cloud's
        post-processing already use, so downstream tools read it with no
        conversion."""
        grp = self.root["trajectories"]
        grp.create_array("pos", data=np.asarray(pos, dtype=np.float64), overwrite=True)
        grp.create_array("vel", data=np.asarray(vel, dtype=np.float64), overwrite=True)
        grp.create_array("accel", data=np.asarray(accel, dtype=np.float64), overwrite=True)
        grp.create_array("time", data=np.asarray(time, dtype=np.int64), overwrite=True)
        grp.create_array("trajid", data=np.asarray(trajid, dtype=np.int64), overwrite=True)

    def trajectory(self, trajid: int) -> dict[str, np.ndarray]:
        """One trajectory's rows from the sealed ``trajectories/`` cache.

        Raises if the store has never been sealed, or if targets/
        correspondences/linkage have changed since the last seal (checked by
        the caller via :meth:`needs_reseal`, since only ``seal()`` knows how
        to recompute the hash cheaply).
        """
        if not self.sealed:
            raise RunStoreError(
                "trajectories/ is stale or absent -- run "
                "openptv2.storage.seal.seal(store) first."
            )
        grp = self.root["trajectories"]
        tids = np.asarray(grp["trajid"])
        # trajectories/ is written sorted by (trajid, time), so each
        # trajectory occupies one contiguous run -- searchsorted instead of
        # a boolean mask over the whole array.
        lo = int(np.searchsorted(tids, trajid, side="left"))
        hi = int(np.searchsorted(tids, trajid, side="right"))
        if lo == hi:
            raise RunStoreError(f"No trajectory with id {trajid}")
        sl = slice(lo, hi)
        return {
            "pos": np.asarray(grp["pos"][sl]),
            "vel": np.asarray(grp["vel"][sl]),
            "accel": np.asarray(grp["accel"][sl]),
            "time": np.asarray(grp["time"][sl]),
            "trajid": np.asarray(grp["trajid"][sl]),
        }

    def trajectories(self) -> dict[str, np.ndarray]:
        if not self.sealed:
            raise RunStoreError(
                "trajectories/ is stale or absent -- run "
                "openptv2.storage.seal.seal(store) first."
            )
        grp = self.root["trajectories"]
        return {k: np.asarray(grp[k]) for k in ("pos", "vel", "accel", "time", "trajid")}

    # -- stats ------------------------------------------------------------

    def write_stats(
        self,
        frame: int,
        n_targets: np.ndarray,
        cam_seen: np.ndarray,
        n_quads: int,
        n_trips: int,
        n_pairs: int,
        n_corres: int,
        n_links: int,
        wall_ms: Optional[float] = None,
    ) -> None:
        fg = self.root["stats"].require_group(_frame_key(frame))
        fg.create_array(
            "n_targets", data=np.asarray(n_targets, dtype=np.int32), overwrite=True
        )
        fg.create_array(
            "cam_seen", data=np.asarray(cam_seen, dtype=np.int32), overwrite=True
        )
        fg.attrs["n_quads"] = int(n_quads)
        fg.attrs["n_trips"] = int(n_trips)
        fg.attrs["n_pairs"] = int(n_pairs)
        fg.attrs["n_corres"] = int(n_corres)
        fg.attrs["n_links"] = int(n_links)
        if wall_ms is not None:
            fg.attrs["wall_ms"] = float(wall_ms)

    def stats(self) -> list[dict]:
        """Per-frame stats records, sorted by frame number."""
        grp = self.root["stats"]
        rows = []
        for key in sorted(grp.keys(), key=_frame_num):
            fg = grp[key]
            rows.append(
                {
                    "frame": _frame_num(key),
                    "n_targets": np.asarray(fg["n_targets"]) if "n_targets" in fg else None,
                    "cam_seen": np.asarray(fg["cam_seen"]) if "cam_seen" in fg else None,
                    "n_quads": fg.attrs.get("n_quads"),
                    "n_trips": fg.attrs.get("n_trips"),
                    "n_pairs": fg.attrs.get("n_pairs"),
                    "n_corres": fg.attrs.get("n_corres"),
                    "n_links": fg.attrs.get("n_links"),
                    "wall_ms": fg.attrs.get("wall_ms"),
                }
            )
        return rows

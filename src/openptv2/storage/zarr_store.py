"""Zarr storage engine for OpenPTV2 target detection, 3D correspondences, and trajectories.

Replaces legacy per-frame ASCII text files (*_targets, rt_is.*, ptv_is.*) with a
cloud-native, lock-free, chunked Zarr directory structure.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import h5py
import numpy as np
import zarr

from openptv2.algorithms.tracking_frame_buf import Target, TargetArray


import time

def _get_or_create_group(parent: Any, name: str) -> Any:
    """Safely get or create a subgroup in a Zarr store handling concurrent workers."""
    for attempt in range(10):
        try:
            if name in parent:
                return parent[name]
            return parent.create_group(name)
        except (PermissionError, OSError, Exception):
            if attempt == 9:
                try:
                    return parent[name]
                except Exception:
                    raise
            time.sleep(0.02 * (attempt + 1))


class ZarrFrameStore:
    """Cloud-native chunked Zarr store for OpenPTV2 experimental data."""

    def __init__(self, store_path: Union[str, Path], mode: str = "a"):
        """Initialize ZarrFrameStore.

        Args:
            store_path: Path to the .zarr directory or S3 key.
            mode: Storage mode ('r', 'r+', 'w', 'w-', 'a').
        """
        self.store_path = Path(store_path)
        try:
            self.root = zarr.open_group(str(self.store_path), mode=mode)
        except (zarr.errors.ContainsGroupError, Exception):
            if mode in ("a", "w", "r+"):
                self.root = zarr.open_group(str(self.store_path), mode="r+")
            else:
                self.root = zarr.open_group(str(self.store_path), mode="r")

        # Initialize sub-groups if creating or appending
        if mode in ("w", "w-", "a", "r+"):
            _get_or_create_group(self.root, "targets")
            _get_or_create_group(self.root, "correspondences")
            _get_or_create_group(self.root, "trajectories")
            _get_or_create_group(self.root, "metadata")

    # -------------------------------------------------------------------------
    # Target Operations (img/camX.YYYY_targets replacement)
    # -------------------------------------------------------------------------

    def write_targets(
        self,
        cam_idx: int,
        frame: int,
        targets: Union[TargetArray, List[Target], np.ndarray],
    ) -> None:
        """Write detected targets for a given camera and frame directly to Zarr.

        Args:
            cam_idx: Camera index (0-based or 1-based).
            frame: Frame index (e.g. 10000).
            targets: TargetArray, list of Target objects, or (N, 8) numpy array.
        """
        targets_group = _get_or_create_group(self.root, "targets")
        cam_group = _get_or_create_group(targets_group, f"cam_{cam_idx}")

        if isinstance(targets, np.ndarray):
            arr_data = targets.astype(np.float64)
        else:
            # Extract Target fields into contiguous NumPy 2D array [N, 8]:
            # [pnr, x, y, n, nx, ny, sumg, tnr]
            count = len(targets)
            arr_data = np.zeros((count, 8), dtype=np.float64)
            for i, t in enumerate(targets):
                if hasattr(t, "pnr"):
                    arr_data[i] = [
                        t.pnr(),
                        t.pos()[0],
                        t.pos()[1],
                        t.count_pixels()[0],
                        t.count_pixels()[1],
                        t.count_pixels()[2],
                        t.sum_grey_value(),
                        t.tnr(),
                    ]
                else:
                    arr_data[i] = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]]

        # Store array for specific frame
        for attempt in range(10):
            try:
                cam_group.create_array(
                    name=f"frame_{frame}",
                    data=arr_data,
                    overwrite=True,
                )
                break
            except (PermissionError, OSError):
                if attempt == 9:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def read_targets(self, cam_idx: int, frame: int) -> TargetArray:
        """Read targets for a given camera and frame from Zarr as TargetArray.

        Args:
            cam_idx: Camera index.
            frame: Frame index.

        Returns:
            TargetArray containing Target objects.
        """
        key = f"targets/cam_{cam_idx}/frame_{frame}"
        if key not in self.root:
            raise FileNotFoundError(f"No targets found in Zarr store for {key}")

        raw_data = self.root[key][:]
        tarr = TargetArray(len(raw_data))
        for i, row in enumerate(raw_data):
            tarr[i].set_pnr(int(row[0]))
            tarr[i].set_pos((row[1], row[2]))
            tarr[i].set_pixel_counts(int(row[3]), int(row[4]), int(row[5]))
            tarr[i].set_sum_grey_value(int(row[6]))
            tarr[i].set_tnr(int(row[7]))

        return tarr

    def has_targets(self, cam_idx: int, frame: int) -> bool:
        """Check if targets exist for a given camera and frame."""
        return f"targets/cam_{cam_idx}/frame_{frame}" in self.root

    # -------------------------------------------------------------------------
    # Correspondences Operations (res/rt_is.YYYY replacement)
    # -------------------------------------------------------------------------

    def write_correspondences(
        self, frame: int, pos_3d: np.ndarray, cam_target_ids: np.ndarray
    ) -> None:
        """Write matched 3D coordinates and camera target IDs for a frame.

        Args:
            frame: Frame index.
            pos_3d: Array of shape (N, 3) [X, Y, Z] in mm.
            cam_target_ids: Array of shape (N, num_cams) with target IDs per camera.
        """
        corr_group = _get_or_create_group(self.root, "correspondences")
        pos_3d = np.asarray(pos_3d, dtype=np.float64)
        cam_target_ids = np.asarray(cam_target_ids, dtype=np.int32)

        combined = np.hstack([pos_3d, cam_target_ids.astype(np.float64)])
        corr_group.create_array(
            name=f"frame_{frame}",
            data=combined,
            overwrite=True,
        )

    def read_correspondences(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read 3D coordinates and camera target IDs for a frame.

        Returns:
            Tuple of (pos_3d (N, 3), cam_target_ids (N, num_cams))
        """
        key = f"correspondences/frame_{frame}"
        if key not in self.root:
            raise FileNotFoundError(
                f"No correspondences found in Zarr store for frame {frame}"
            )

        data = self.root[key][:]
        pos_3d = data[:, :3]
        cam_target_ids = data[:, 3:].astype(np.int32)
        return pos_3d, cam_target_ids

    # -------------------------------------------------------------------------
    # Trajectories Operations (res/ptv_is.YYYY & Flowtracks replacement)
    # -------------------------------------------------------------------------

    def write_trajectories(
        self,
        pos: np.ndarray,
        vel: Optional[np.ndarray] = None,
        acc: Optional[np.ndarray] = None,
        frames: Optional[np.ndarray] = None,
        traj_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Write full particle trajectory dataset.

        Args:
            pos: Array of shape (N, 3) with positions in mm.
            vel: Array of shape (N, 3) with velocities in mm/s.
            acc: Array of shape (N, 3) with accelerations in mm/s^2.
            frames: Array of shape (N,) with frame numbers.
            traj_ids: Array of shape (N,) with trajectory IDs.
        """
        traj_group = _get_or_create_group(self.root, "trajectories")
        traj_group.create_array(
            "pos", data=np.asarray(pos, dtype=np.float64), overwrite=True
        )

        if vel is not None:
            traj_group.create_array(
                "vel", data=np.asarray(vel, dtype=np.float64), overwrite=True
            )
        if acc is not None:
            traj_group.create_array(
                "acc", data=np.asarray(acc, dtype=np.float64), overwrite=True
            )
        if frames is not None:
            traj_group.create_array(
                "frame", data=np.asarray(frames, dtype=np.int32), overwrite=True
            )
        if traj_ids is not None:
            traj_group.create_array(
                "trajid", data=np.asarray(traj_ids, dtype=np.int32), overwrite=True
            )

    # -------------------------------------------------------------------------
    # Tracking Linkage Operations (res/ptv_is.YYYY, res/added.YYYY)
    # -------------------------------------------------------------------------

    def write_linkage(
        self,
        frame: int,
        prev_ids: np.ndarray,
        next_ids: np.ndarray,
        pos_3d: np.ndarray,
        linkage_name: str = "ptv_is",
    ) -> None:
        """Write frame linkage data (ptv_is / added) for tracking.

        Args:
            frame: Frame number.
            prev_ids: Array of shape (N,) with previous frame particle indices.
            next_ids: Array of shape (N,) with next frame particle indices.
            pos_3d: Array of shape (N, 3) with 3D positions in mm.
            linkage_name: 'ptv_is' or 'added'.
        """
        link_group = _get_or_create_group(self.root, f"linkage/{linkage_name}")
        frame_group = _get_or_create_group(link_group, f"frame_{frame:05d}")

        frame_group.create_array(
            "prev", data=np.asarray(prev_ids, dtype=np.int32), overwrite=True
        )
        frame_group.create_array(
            "next", data=np.asarray(next_ids, dtype=np.int32), overwrite=True
        )
        frame_group.create_array(
            "pos", data=np.asarray(pos_3d, dtype=np.float64), overwrite=True
        )

    def read_linkage(
        self, frame: int, linkage_name: str = "ptv_is"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read frame linkage data (prev_ids, next_ids, pos_3d)."""
        frame_path = f"linkage/{linkage_name}/frame_{frame:05d}"
        if frame_path not in self.root:
            raise KeyError(f"No linkage '{linkage_name}' for frame {frame}")
        fg = self.root[frame_path]
        return fg["prev"][:], fg["next"][:], fg["pos"][:]

    def has_linkage(self, frame: int, linkage_name: str = "ptv_is") -> bool:
        """Check if linkage dataset exists for a frame."""
        return f"linkage/{linkage_name}/frame_{frame:05d}" in self.root

    def to_flowtracks_h5(self, h5_path: Union[str, Path]) -> None:
        """Export Zarr trajectories to Flowtracks-compliant HDF5 file.

        Args: h5_path: Output .h5 file path.
        """
        h5_path = Path(h5_path)
        traj_group = self.root["trajectories"]

        if "pos" not in traj_group:
            raise ValueError("No trajectory data available in Zarr store to export")

        pos_m = traj_group["pos"][:] / 1000.0  # Convert mm to meters for Flowtracks

        with h5py.File(str(h5_path), "w") as h5f:
            h5f.create_dataset("pos", data=pos_m, compression="gzip")

            if "vel" in traj_group:
                vel_m = traj_group["vel"][:] / 1000.0
                h5f.create_dataset("vel", data=vel_m, compression="gzip")
            if "acc" in traj_group:
                acc_m = traj_group["acc"][:] / 1000.0
                h5f.create_dataset("acc", data=acc_m, compression="gzip")
            if "frame" in traj_group:
                h5f.create_dataset(
                    "frame", data=traj_group["frame"][:], compression="gzip"
                )
            if "trajid" in traj_group:
                h5f.create_dataset(
                    "trajid", data=traj_group["trajid"][:], compression="gzip"
                )

    # ----------------─────────────────────────────────────────────────────────
    # Human-Readable Text Inspection Methods
    # ----------------─────────────────────────────────────────────────────────

    def export_frame_text(
        self, frame: int, dataset_type: str = "targets", cam_idx: int = 0
    ) -> str:
        """Format binary frame data as traditional ASCII text string.

        Args:
            frame: Frame number.
            dataset_type: 'targets' (camX.YYYY_targets format),
                          'rt_is' (res/rt_is.YYYY 3D correspondences format), or
                          'ptv_is' (res/ptv_is.YYYY trajectory links format).
            cam_idx: Camera index (used when dataset_type='targets').

        Returns:
            Formatted ASCII text string as if opening the legacy text file.
        """
        lines = []

        if dataset_type.lower() == "targets":
            key = f"targets/cam_{cam_idx}/frame_{frame}"
            if key not in self.root:
                raise FileNotFoundError(f"No targets found in store for {key}")
            raw = self.root[key][:]
            lines.append(f"{len(raw)}")
            for row in raw:
                # pnr, x, y, n, nx, ny, sumg, tnr
                lines.append(
                    f"{int(row[0]):4d} {row[1]:9.4f} {row[2]:9.4f} "
                    f"{int(row[3]):5d} {int(row[4]):5d} {int(row[5]):5d} "
                    f"{int(row[6]):5d} {int(row[7]):5d}"
                )

        elif dataset_type.lower() in ("rt_is", "correspondences"):
            pos_3d, cam_ids = self.read_correspondences(frame)
            lines.append(f"{len(pos_3d)}")
            for idx, (p, c) in enumerate(zip(pos_3d, cam_ids)):
                cam_str = " ".join(f"{cid:4d}" for cid in c)
                lines.append(f"{idx:4d} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f} {cam_str}")

        elif dataset_type.lower() in ("ptv_is", "added", "trajectories"):
            link_type = dataset_type.lower()
            if link_type == "trajectories":
                link_type = "ptv_is"
            if self.has_linkage(frame, link_type):
                prev_ids, next_ids, pos = self.read_linkage(frame, link_type)
                lines.append(f"{len(pos)}")
                for prev_i, next_i, p in zip(prev_ids, next_ids, pos):
                    lines.append(
                        f"{int(prev_i):4d} {int(next_i):4d} {p[0]:10.3f} {p[1]:10.3f} {p[2]:10.3f}"
                    )
            elif "trajectories" in self.root:
                traj_group = self.root["trajectories"]
                if "frame" in traj_group and "pos" in traj_group:
                    frames = traj_group["frame"][:]
                    mask = frames == frame
                    pos = traj_group["pos"][mask]
                    ids = (
                        traj_group["trajid"][mask]
                        if "trajid" in traj_group
                        else np.arange(len(pos))
                    )
                    lines.append(f"{len(pos)}")
                    for pnr, p in zip(ids, pos):
                        lines.append(
                            f"{int(pnr):4d} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}"
                        )
            else:
                raise FileNotFoundError(
                    f"No '{dataset_type}' linkage data stored for frame {frame}"
                )

        else:
            raise ValueError(
                f"Unknown dataset_type '{dataset_type}'. Use 'targets', 'rt_is', or 'ptv_is'."
            )

        return "\n".join(lines) + "\n"

    def dump_frame_text(
        self, frame: int, dataset_type: str = "targets", cam_idx: int = 0
    ) -> None:
        """Print formatted frame text directly to stdout for human inspection."""
        print(
            self.export_frame_text(
                frame=frame, dataset_type=dataset_type, cam_idx=cam_idx
            )
        )


def inspect_zarr_store(zarr_path: Union[str, Path]) -> str:
    """Inspect a Zarr store dataset across all pipeline stages and return a human-readable report."""
    import zarr
    import numpy as np

    root = zarr.open_group(str(zarr_path), mode="r")
    lines = []
    lines.append("=" * 60)
    lines.append(f"[INSPECT] Zarr Dataset Inspection: {zarr_path}")
    lines.append("=" * 60)

    # 1. Targets
    if "targets" in root:
        tgt_grp = root["targets"]
        cams = sorted([k for k in tgt_grp.keys() if k.startswith("cam_")])
        lines.append(f"[Targets]: {len(cams)} camera groups found ({', '.join(cams)})")
        for cam in cams:
            f_keys = sorted([k for k in tgt_grp[cam].keys() if k.startswith("frame_")])
            if f_keys:
                f_min = f_keys[0].split("_")[1]
                f_max = f_keys[-1].split("_")[1]
                lines.append(f"   - {cam}: {len(f_keys)} frames (Range: {f_min} .. {f_max})")
    else:
        lines.append("[Targets]: None")

    # 2. Correspondences
    if "correspondences" in root:
        corr_grp = root["correspondences"]
        f_keys = sorted([k for k in corr_grp.keys() if k.startswith("frame_")])
        if f_keys:
            f_min = f_keys[0].split("_")[1]
            f_max = f_keys[-1].split("_")[1]
            first_arr = np.asarray(corr_grp[f_keys[0]])
            lines.append(f"[Correspondences]: {len(f_keys)} frames (Range: {f_min} .. {f_max}), ~{len(first_arr)} matches/frame")
        else:
            lines.append("[Correspondences]: Present (0 frames)")
    else:
        lines.append("[Correspondences]: None")

    # 3. Trajectories
    if "trajectories" in root:
        traj_grp = root["trajectories"]
        if "trajid" in traj_grp:
            trids = np.asarray(traj_grp["trajid"])
            times = np.asarray(traj_grp["time"])
            pos = np.asarray(traj_grp["pos"])
            u_ids = len(np.unique(trids))
            lines.append(f"[Trajectories]: {u_ids} unique trajectories, {len(pos)} total points (Time: {times.min()} .. {times.max()})")
            if "vel" in traj_grp:
                vel = np.asarray(traj_grp["vel"])
                v_mag = np.linalg.norm(vel, axis=1)
                lines.append(f"   - Velocities present: min={v_mag.min():.4f}, max={v_mag.max():.4f}, mean={v_mag.mean():.4f} m/s")
        else:
            lines.append(f"[Trajectories]: Group present ({list(traj_grp.keys())})")
    else:
        lines.append("[Trajectories]: None")

    # 4. Eulerian fields
    if "eulerian" in root:
        eul_grp = root["eulerian"]
        vars_found = list(eul_grp.keys())
        lines.append(f"[Eulerian Fields]: {len(vars_found)} variables ({', '.join(vars_found[:5])}...)")
    else:
        lines.append("[Eulerian Fields]: None")

    lines.append("=" * 60)
    report = "\n".join(lines)
    return report


def main_cli():
    """Command-line tool to inspect ZarrFrameStore datasets as human-readable text."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect ZarrFrameStore binary data as human-readable legacy ASCII text."
    )
    parser.add_argument("store_path", help="Path to .zarr directory")
    parser.add_argument("--frame", "-f", type=int, default=None, help="Frame index for text dump")
    parser.add_argument(
        "--type",
        "-t",
        choices=["targets", "rt_is", "ptv_is"],
        default="targets",
        help="Data type to display",
    )
    parser.add_argument(
        "--cam", "-c", type=int, default=0, help="Camera index for targets"
    )

    args = parser.parse_args()
    if args.frame is None:
        print(inspect_zarr_store(args.store_path))
    else:
        store = ZarrFrameStore(args.store_path, mode="r")
        store.dump_frame_text(frame=args.frame, dataset_type=args.type, cam_idx=args.cam)


if __name__ == "__main__":
    main_cli()

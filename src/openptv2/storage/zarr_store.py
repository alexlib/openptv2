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


def _get_or_create_group(parent: Any, name: str) -> Any:
    """Safely get or create a subgroup in a Zarr store handling concurrent workers."""
    if name in parent:
        return parent[name]
    try:
        return parent.create_group(name)
    except Exception:
        return parent[name]


class ZarrFrameStore:
    """Cloud-native chunked Zarr store for OpenPTV2 experimental data."""

    def __init__(self, store_path: Union[str, Path], mode: str = "a"):
        """Initialize ZarrFrameStore.

        Args:
            store_path: Path to the .zarr directory or S3 key.
            mode: Storage mode ('r', 'r+', 'w', 'w-', 'a').
        """
        self.store_path = Path(store_path)
        self.root = zarr.open_group(str(self.store_path), mode=mode)

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
        cam_group.create_array(
            name=f"frame_{frame}",
            data=arr_data,
            overwrite=True,
        )

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

        elif dataset_type.lower() in ("ptv_is", "trajectories"):
            traj_group = self.root["trajectories"]
            if "frame" not in traj_group or "pos" not in traj_group:
                raise FileNotFoundError("No trajectory data stored")
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
                lines.append(f"{int(pnr):4d} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f}")

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


def main_cli():
    """Command-line tool to inspect ZarrFrameStore datasets as human-readable text."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect ZarrFrameStore binary data as human-readable legacy ASCII text."
    )
    parser.add_argument("store_path", help="Path to .zarr directory")
    parser.add_argument("--frame", "-f", type=int, required=True, help="Frame index")
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
    store = ZarrFrameStore(args.store_path, mode="r")
    store.dump_frame_text(frame=args.frame, dataset_type=args.type, cam_idx=args.cam)


if __name__ == "__main__":
    main_cli()

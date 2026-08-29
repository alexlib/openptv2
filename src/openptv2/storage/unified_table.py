"""Unified particle table: 3D positions with per-camera 2D leaves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UnifiedParticleTable:
    """One row per particle per frame, carrying 3D position + per-camera 2D leaves.

    Attributes
    ----------
    time : (N,) int32 — frame number
    pid : (N,) int32 — unique particle ID within the frame
    xyz : (N, 3) float64 — [X, Y, Z] in mm
    xy_cam : (N, C, 2) float64 — per-camera [x, y] in pixels, NaN where absent
    num_cams : int — number of cameras
    """

    time: np.ndarray
    pid: np.ndarray
    xyz: np.ndarray
    xy_cam: np.ndarray
    num_cams: int

    @classmethod
    def from_correspondences_and_targets(
        cls,
        frames: list[int],
        correspondences: dict[int, np.ndarray],
        targets: dict[int, list[np.ndarray]],
        cam_ids: Optional[dict[int, np.ndarray]] = None,
    ) -> "UnifiedParticleTable":
        """Build unified table from existing correspondences + per-camera targets.

        Parameters
        ----------
        frames : list of frame numbers
        correspondences : {frame: (N, 3+C) array} from RunStore.read_correspondences
        targets : {frame: list of (M, 8) arrays per camera} from RunStore.read_targets
        cam_ids : optional {frame: (N, C) int array} of per-camera target indices
        """
        rows_time = []
        rows_pid = []
        rows_xyz = []
        rows_xy = []

        for f in frames:
            corr = np.asarray(correspondences[f])
            n = corr.shape[0]
            xyz = corr[:, :3]
            rows_time.append(np.full(n, f, dtype=np.int32))
            rows_pid.append(np.arange(n, dtype=np.int32))
            rows_xyz.append(xyz)

            num_cams = len(targets[f])
            xy = np.full((n, num_cams, 2), np.nan, dtype=np.float64)

            if cam_ids is not None and f in cam_ids:
                cam_id_arr = np.asarray(cam_ids[f])
                for c in range(num_cams):
                    valid = cam_id_arr[:, c] >= 0
                    idx = cam_id_arr[valid, c].astype(int)
                    if idx.size > 0 and targets[f][c].shape[0] > 0:
                        xy[valid, c, 0] = targets[f][c][idx, 1]  # x
                        xy[valid, c, 1] = targets[f][c][idx, 2]  # y

            rows_xy.append(xy)

        return cls(
            time=np.concatenate(rows_time),
            pid=np.concatenate(rows_pid),
            xyz=np.vstack(rows_xyz),
            xy_cam=np.concatenate(rows_xy, axis=0),
            num_cams=num_cams,
        )

    def frame_mask(self, frame: int) -> np.ndarray:
        """Boolean mask for rows belonging to a given frame."""
        return self.time == frame

    def frame_slice(self, frame: int) -> "UnifiedParticleTable":
        """Return a new table with only rows from the given frame."""
        m = self.frame_mask(frame)
        return UnifiedParticleTable(
            time=self.time[m],
            pid=self.pid[m],
            xyz=self.xyz[m],
            xy_cam=self.xy_cam[m],
            num_cams=self.num_cams,
        )

    def frames_in_range(self, t0: int, t1: int) -> "UnifiedParticleTable":
        """Return rows with time in [t0, t1]."""
        m = (self.time >= t0) & (self.time <= t1)
        return UnifiedParticleTable(
            time=self.time[m],
            pid=self.pid[m],
            xyz=self.xyz[m],
            xy_cam=self.xy_cam[m],
            num_cams=self.num_cams,
        )

    @property
    def num_particles(self) -> int:
        return self.xyz.shape[0]

    def ndim_features(self, alpha: float = 1.0) -> np.ndarray:
        """Build N-dimensional feature array: [X, Y, Z, alpha*x0, alpha*y0, ...].

        NaN values are replaced with 0 (the KD-tree can handle this since
        NaN particles are simply far from everything in practice).

        Parameters
        ----------
        alpha : float
            Weight for 2D camera dimensions relative to 3D.
        """
        n = self.xyz.shape[0]
        C = self.num_cams
        features = np.zeros((n, 3 + C * 2), dtype=np.float64)
        features[:, :3] = self.xyz
        for c in range(C):
            features[:, 3 + c * 2] = self.xy_cam[:, c, 0] * alpha
            features[:, 3 + c * 2 + 1] = self.xy_cam[:, c, 1] * alpha
        # NaN → 0 ( KD-tree treats these as "no information")
        np.nan_to_num(features, copy=False)
        return features

    def valid_cam_mask(self) -> np.ndarray:
        """(N, C) boolean — True where camera c has a detection for this particle."""
        return ~np.isnan(self.xy_cam[:, :, 0])

    def to_dict(self) -> dict:
        """Serialize to a dict for zarr storage."""
        return {
            "time": self.time,
            "pid": self.pid,
            "xyz": self.xyz,
            "xy_cam": self.xy_cam,
            "num_cams": np.int32(self.num_cams),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedParticleTable":
        """Deserialize from a dict loaded from zarr."""
        return cls(
            time=d["time"],
            pid=d["pid"],
            xyz=d["xyz"],
            xy_cam=d["xy_cam"],
            num_cams=int(d["num_cams"]),
        )

    def __len__(self) -> int:
        return self.num_particles

    def __repr__(self) -> str:
        return (
            f"UnifiedParticleTable(n={self.num_particles}, "
            f"frames={self.time.min()}..{self.time.max()}, "
            f"cams={self.num_cams})"
        )

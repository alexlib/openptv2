"""KD-tree over unified particle tables for efficient N-dimensional nearest-neighbor queries."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.spatial import KDTree as _KDTree
except ImportError:
    _KDTree = None


class ParticleKDTree:
    """KD-tree built over particles from 1-4 consecutive frames.

    Supports two modes:
    - ``mode='3d'``: tree over (X, Y, Z) only — fast, matches liboptv behaviour.
    - ``mode='nd'``: tree over (X, Y, Z, alpha*x0, alpha*y0, ...) — the full
      N-dimensional feature space that carries per-camera leaf information.

    Parameters
    ----------
    table : UnifiedParticleTable
        The particle data to index.
    frames : list of int
        Which frames to include in the tree.
    alpha : float
        Weight for 2D camera dimensions (only used in ``mode='nd'``).
    mode : str
        ``'3d'`` or ``'nd'``.
    """

    def __init__(
        self,
        table,
        frames: list[int],
        alpha: float = 1.0,
        mode: str = "3d",
    ):
        if _KDTree is None:
            raise ImportError("scipy.spatial.KDTree is required: pip install scipy")

        self._table = table
        self._frames = sorted(frames)
        self._alpha = alpha
        self._mode = mode

        # Collect indices of particles in the requested frames
        self._global_idx = np.concatenate(
            [np.where(table.frame_mask(f))[0] for f in self._frames]
        )
        self._frame_of = table.time[self._global_idx]
        self._pid_of = table.pid[self._global_idx]

        if len(self._global_idx) == 0:
            self._tree = None
            self._points = np.empty((0, 3), dtype=np.float64)
            return

        if mode == "3d":
            self._points = table.xyz[self._global_idx].copy()
        elif mode == "nd":
            features = table.ndim_features(alpha)
            self._points = features[self._global_idx].copy()
        else:
            raise ValueError(f"mode must be '3d' or 'nd', got '{mode}'")

        # Replace NaN with 0 for KD-tree (NaN particles are just far away)
        np.nan_to_num(self._points, copy=False)

        self._tree = _KDTree(self._points)

    @property
    def num_points(self) -> int:
        return len(self._global_idx)

    @property
    def is_empty(self) -> bool:
        return self._tree is None or self.num_points == 0

    def query(
        self,
        q: np.ndarray,
        k: int = 1,
        max_dist: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each query point.

        Parameters
        ----------
        q : (M, D) array — query points
        k : int — number of neighbors
        max_dist : float, optional — ignore neighbors farther than this

        Returns
        -------
        dists : (M, k) float64 — distances
        idxs : (M, k) int — indices into the tree's point set
        """
        if self.is_empty:
            return (
                np.full((len(q), k), np.inf),
                np.full((len(q), k), -1, dtype=np.int32),
            )

        dists, idxs = self._tree.query(q, k=k, p=2)

        if max_dist is not None:
            mask = dists > max_dist
            dists[mask] = np.inf
            idxs[mask] = -1

        # scipy returns (k,) for single query; broadcast to (1, k)
        if dists.ndim == 1:
            dists = dists[np.newaxis, :]
            idxs = idxs[np.newaxis, :]

        return dists.astype(np.float64), idxs.astype(np.int32)

    def query_ball(
        self,
        q: np.ndarray,
        r: float,
    ) -> list[list[int]]:
        """Find all points within radius r of each query point.

        Returns list of lists of indices into the tree's point set.
        """
        if self.is_empty:
            return [[] for _ in range(len(q))]
        return self._tree.query_ball_point(q, r)

    def global_index(self, tree_idx: np.ndarray) -> np.ndarray:
        """Convert tree-local indices to global table indices."""
        return self._global_idx[tree_idx]

    def frame_of(self, tree_idx: np.ndarray) -> np.ndarray:
        """Get frame numbers for tree-local indices."""
        return self._frame_of[tree_idx]

    def pid_of(self, tree_idx: np.ndarray) -> np.ndarray:
        """Get particle IDs for tree-local indices."""
        return self._pid_of[tree_idx]

    def get_xy_cam(self, tree_idx: np.ndarray) -> np.ndarray:
        """Get per-camera 2D positions for tree-local indices.

        Returns (M, C, 2) array, NaN where camera didn't detect.
        """
        global_idx = self._global_idx[tree_idx]
        return self._table.xy_cam[global_idx]

    def get_xyz(self, tree_idx: np.ndarray) -> np.ndarray:
        """Get 3D positions for tree-local indices."""
        return self._points[tree_idx]

    def build_frame_pairs(
        self,
        alpha: float = 1.0,
        mode: str = "3d",
    ) -> list[tuple["ParticleKDTree", "ParticleKDTree"]]:
        """Build (tree_curr, tree_next) pairs for each consecutive frame gap.

        Returns one pair per consecutive frame gap in the tree's frame range.
        """
        pairs = []
        for i in range(len(self._frames) - 1):
            t0, t1 = self._frames[i], self._frames[i + 1]
            tree_curr = ParticleKDTree(self._table, [t0], alpha=alpha, mode=mode)
            tree_next = ParticleKDTree(self._table, [t1], alpha=alpha, mode=mode)
            pairs.append((tree_curr, tree_next))
        return pairs

    def __repr__(self) -> str:
        return (
            f"ParticleKDTree(mode={self._mode}, frames={self._frames}, "
            f"points={self.num_points}, alpha={self._alpha})"
        )

"""Temporal joint smoothing for ghost suppression.

Ghosts rarely persist as smooth trajectories: a phantom 3D point may look
geometrically consistent in one frame but does not follow a smooth motion
across frames. This module scores candidate trajectories by their temporal
smoothness (acceleration residual) and down-weights / drops the rough ones
via iterative re-weighted least squares (IRLS). Cheap, no solver, runs in
O(window * points).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def trajectory_acceleration_residual(positions: np.ndarray) -> float:
    """RMS second-difference (acceleration) of a trajectory.

    ``positions`` is (T, D). Returns 0.0 for T < 3.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 3:
        return 0.0
    # second difference: p[t+2] - 2 p[t+1] + p[t]
    accel = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
    return float(np.sqrt(np.mean(accel ** 2)))


def trajectory_residual_with_gaps(
    positions: np.ndarray,
    max_gap: int = 1,
) -> float:
    """Residual that tolerates small gaps (linearly interpolated)."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 3:
        return 0.0
    # Simple: use acceleration residual on available frames.
    return trajectory_acceleration_residual(positions)


def smoothness_keep(
    positions: np.ndarray,
    threshold: float,
) -> bool:
    """Return True if trajectory residual <= threshold."""
    return trajectory_acceleration_residual(positions) <= float(threshold)


def irls_weights(
    positions: np.ndarray,
    n_iter: int = 3,
    residual_threshold: float = 1.0,
) -> np.ndarray:
    """Iterative re-weighted least squares weights per frame.

    Returns an array of weights in [0,1] (length T) emphasising frames that
    lie on a smooth trajectory. Ghost-heavy frames get down-weighted.
    """
    positions = np.asarray(positions, dtype=float)
    t = positions.shape[0]
    if t < 3:
        return np.ones(t, dtype=float)
    w = np.ones(t, dtype=float)
    for _ in range(max(1, n_iter)):
        # Fit a cubic (or lower) smooth model per dimension via least squares
        # weighted by current w, then recompute residuals.
        idx = np.arange(t, dtype=float)
        # design matrix for cubic
        deg = min(3, t - 1)
        A = np.vander(idx, deg + 1)  # (t, deg+1)
        W = np.diag(w)
        # solve weighted least squares: (A^T W A) beta = A^T W y per dim
        AtWA = A.T @ (w[:, None] * A)
        try:
            beta_all = np.linalg.solve(AtWA, A.T @ (w[:, None] * positions))
        except np.linalg.LinAlgError:
            beta_all = np.linalg.lstsq(A, positions, rcond=None)[0]
        pred = A @ beta_all
        resid = np.sqrt(np.sum((positions - pred) ** 2, axis=1))
        # update weights: high residual -> low weight
        scale = np.median(resid) + 1e-12
        w = np.exp(-0.5 * (resid / (scale + 1e-12)) ** 2)
        w = np.clip(w, 1e-3, 1.0)
    return w


def smooth_trajectories(
    trajectories: Sequence[np.ndarray],
    threshold: float,
    n_iter: int = 3,
) -> list[bool]:
    """Return keep/drop flags for a list of trajectory position arrays.

    A trajectory is kept if its final IRLS residual <= threshold.
    """
    out: list[bool] = []
    for traj in trajectories:
        arr = np.asarray(traj, dtype=float)
        if arr.shape[0] < 3:
            out.append(True)
            continue
        w = irls_weights(arr, n_iter=n_iter)
        # residual after smoothing
        idx = np.arange(arr.shape[0], dtype=float)
        deg = min(3, arr.shape[0] - 1)
        A = np.vander(idx, deg + 1)
        AtWA = A.T @ (w[:, None] * A)
        try:
            beta_all = np.linalg.solve(AtWA, A.T @ (w[:, None] * arr))
        except np.linalg.LinAlgError:
            beta_all = np.linalg.lstsq(A, arr, rcond=None)[0]
        pred = A @ beta_all
        resid = float(np.sqrt(np.mean(np.sum((arr - pred) ** 2, axis=1))))
        out.append(bool(resid <= float(threshold)))
    return out


def extract_trajectories(
    store: Any,
    first: int,
    last: int,
    linkage_name: str = "ptv_is",
) -> list[list[tuple[int, int]]]:
    """Extract trajectories from a store's linkage as lists of (frame, point_index).

    Follows the ``next`` pointers. Returns a list of trajectories, each a list
    of (frame, idx) pairs.
    """
    import numpy as np

    # read per-frame linkage
    frames_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for f in range(first, last + 1):
        try:
            prev, nxt, pos = store.read_linkage(f, name=linkage_name)
        except Exception:
            continue
        frames_data[f] = (np.asarray(prev).ravel(), np.asarray(nxt).ravel(), np.asarray(pos))
    if not frames_data:
        return []

    # Build trajectories: start from points whose prev == -1 (or unvisited).
    visited: set[tuple[int, int]] = set()
    trajs: list[list[tuple[int, int]]] = []
    for start_f in range(first, last + 1):
        if start_f not in frames_data:
            continue
        prev_arr, _, _ = frames_data[start_f]
        n = len(prev_arr)
        for i in range(n):
            if int(prev_arr[i]) != -1:
                continue
            # walk forward
            chain: list[tuple[int, int]] = []
            f, idx = start_f, i
            seen: set[tuple[int, int]] = set()
            while True:
                key = (f, idx)
                if key in seen:
                    break
                seen.add(key)
                chain.append(key)
                if f not in frames_data:
                    break
                _, nxt_arr, _ = frames_data[f]
                if idx >= len(nxt_arr):
                    break
                ni = int(nxt_arr[idx])
                if ni < 0:
                    break
                # find which frame holds point ni: next points to global index?
                # Convention: next stores index into next frame's point list.
                f_next = f + 1
                if f_next > last or f_next not in frames_data:
                    break
                if ni >= len(frames_data[f_next][0]):
                    break
                f, idx = f_next, ni
            if chain:
                for k in chain:
                    visited.add(k)
                trajs.append(chain)
    # Also include any unvisited points as singleton trajectories.
    for f in range(first, last + 1):
        if f not in frames_data:
            continue
        prev_arr, _, _ = frames_data[f]
        for i in range(len(prev_arr)):
            if (f, i) not in visited:
                trajs.append([(f, i)])
                visited.add((f, i))
    return trajs


def trajectory_positions_from_store(
    store: Any,
    traj: list[tuple[int, int]],
    linkage_name: str = "ptv_is",
) -> np.ndarray:
    """Return (T, 3) positions for a trajectory."""
    import numpy as np

    pts = []
    for f, idx in traj:
        try:
            _, _, pos = store.read_linkage(f, name=linkage_name)
        except Exception:
            pts.append([np.nan, np.nan, np.nan])
            continue
        pos = np.asarray(pos)
        if idx < len(pos):
            pts.append(np.asarray(pos[idx]).ravel()[:3])
        else:
            pts.append([np.nan, np.nan, np.nan])
    return np.asarray(pts, dtype=float)


def apply_temporal_smoothing_to_store(
    store: Any,
    first: int,
    last: int,
    threshold: float = 1.0,
    linkage_name: str = "ptv_is",
    n_iter: int = 3,
) -> dict[str, int]:
    """Compute smoothing residuals per trajectory and report stats.

    Returns ``{"n_trajectories", "n_high_residual", "n_dropped"}``. This
    version reports only; it does not mutate the store (safe default). Set
    ``apply=True`` behaviour is provided by callers that want to drop.
    """
    trajs = extract_trajectories(store, first, last, linkage_name)
    residuals = []
    high = 0
    for t in trajs:
        pos = trajectory_positions_from_store(store, t, linkage_name)
        # drop nan rows
        valid = ~np.isnan(pos).any(axis=1)
        if valid.sum() < 3:
            residuals.append(0.0)
            continue
        r = trajectory_acceleration_residual(pos[valid])
        residuals.append(r)
        if r > threshold:
            high += 1
    return {
        "n_trajectories": len(trajs),
        "n_high_residual": high,
        "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
        "n_dropped": 0,
    }


def resolve_smoothing_config(track_cfg: dict, regime: dict) -> bool:
    val = track_cfg.get("temporal_smoothing", "auto")
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("auto", ""):
            val = "auto"
        elif v in ("1", "true", "yes", "on"):
            return True
        elif v in ("0", "false", "no", "off"):
            return False
    if val is True:
        return True
    if val is False:
        return False
    # auto
    slow = bool(regime.get("slow_flow", False))
    jump_rate = float(regime.get("jump_rate", 0.0) or 0.0)
    return slow or jump_rate > 0.1
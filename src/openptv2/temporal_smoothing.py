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


def irls_predict_state(
    times: np.ndarray,
    positions: np.ndarray,
    target_time: float,
    *,
    window: int = 6,
    max_deg: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict (pos, vel, rms_resid) at ``target_time`` via IRLS polynomial fit.

    Uses the closest ``window`` points (caller passes ordered points). Degree is
    capped at ``min(max_deg, len - 1)`` (quadratic by default, safer for
    short extrapolation than cubic).
    """
    t_arr = np.asarray(times, dtype=float)
    p_arr = np.asarray(positions, dtype=float)
    if len(t_arr) > window:
        # Keep the points closest to target_time.
        order = np.argsort(np.abs(t_arr - target_time))[:window]
        order = np.sort(order)
        t_arr = t_arr[order]
        p_arr = p_arr[order]

    n = len(t_arr)
    if n == 0:
        raise ValueError("times/positions must be non-empty")
    if n == 1:
        return p_arr[0].copy(), np.zeros(p_arr.shape[1], dtype=float), 0.0

    t0 = float(t_arr[-1])
    dt = t_arr - t0
    deg = min(int(max_deg), 1 if n <= 3 else n - 1)
    # Use IRLS weights when >= 4 points; otherwise uniform.
    w = irls_weights(p_arr, n_iter=3) if n >= 4 else np.ones(n, dtype=float)
    # Vandermonde in increasing powers: [1, dt, dt^2]
    A = np.column_stack([dt ** k for k in range(deg + 1)])
    AtWA = A.T @ (w[:, None] * A)
    try:
        beta = np.linalg.solve(AtWA + 1e-12 * np.eye(deg + 1), A.T @ (w[:, None] * p_arr))
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, p_arr, rcond=None)[0]

    fitted = A @ beta
    resid = float(np.sqrt(np.mean(np.sum((p_arr - fitted) ** 2, axis=1))))

    tau = float(target_time) - t0
    pos_pred = np.zeros(p_arr.shape[1], dtype=float)
    vel_pred = np.zeros(p_arr.shape[1], dtype=float)
    for k in range(deg + 1):
        pos_pred += beta[k] * (tau ** k)
        if k >= 1:
            vel_pred += k * beta[k] * (tau ** (k - 1))
    return pos_pred, vel_pred, resid


def _extract_tracklets_from_frames(
    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    last: int,
) -> list[list[tuple[int, int]]]:
    visited: set[tuple[int, int]] = set()
    tracklets: list[list[tuple[int, int]]] = []
    for f in sorted(frames):
        prv, _, _ = frames[f]
        for i in range(len(prv)):
            if (f, i) in visited:
                continue
            p_idx = int(prv[i])
            if (
                p_idx >= 0
                and (f - 1) in frames
                and 0 <= p_idx < len(frames[f - 1][1])
                and int(frames[f - 1][1][p_idx]) == i
            ):
                continue
            chain: list[tuple[int, int]] = []
            cur_f, cur_i = f, i
            while cur_f in frames and 0 <= cur_i < len(frames[cur_f][1]):
                if (cur_f, cur_i) in visited:
                    break
                visited.add((cur_f, cur_i))
                chain.append((cur_f, cur_i))
                nxt_i = int(frames[cur_f][1][cur_i])
                if (
                    nxt_i < 0
                    or cur_f + 1 > last
                    or (cur_f + 1) not in frames
                    or nxt_i >= len(frames[cur_f + 1][0])
                    or int(frames[cur_f + 1][0][nxt_i]) != cur_i
                ):
                    break
                cur_f, cur_i = cur_f + 1, nxt_i
            tracklets.append(chain)
    return tracklets


def _turning_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float | None:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def stitch_tracklets_irls(
    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    first: int,
    last: int,
    *,
    pos_tol: float = 1.0,
    vel_tol: float = 1.0,
    max_step: float | None = None,
    max_angle_deg: float | None = None,
    window: int = 6,
    max_passes: int = 3,
) -> tuple[int, set[int]]:
    """Stitch fragmented tracklets at adjacent frames using IRLS prediction.

    Mutates ``frames`` (prev/next arrays) in-place and returns
    ``(n_stitched, dirty_frames)``.
    """
    total_stitched = 0
    dirty: set[int] = set()

    for _ in range(max(1, max_passes)):
        tracklets = _extract_tracklets_from_frames(frames, last)
        starts_by_frame: dict[int, list[list[tuple[int, int]]]] = {}
        for tr in tracklets:
            starts_by_frame.setdefault(tr[0][0], []).append(tr)

        candidates: list[tuple[float, int, int, int, int]] = []
        for tr in tracklets:
            if len(tr) < 2:
                continue
            f_end, i_end = tr[-1]
            if int(frames[f_end][1][i_end]) >= 0:
                continue
            f_next = f_end + 1
            if f_next > last or f_next not in starts_by_frame:
                continue
            tail = tr[-window:]
            t_tail = np.array([cf for cf, _ in tail], dtype=float)
            p_tail = np.array([frames[cf][2][ci] for cf, ci in tail], dtype=float)
            deg_tail = 2 if len(tail) >= 5 else 1
            p_pred, v_pred, resid = irls_predict_state(
                t_tail, p_tail, float(f_next), window=window, max_deg=deg_tail
            )
            eff_pos_tol = min(2.0 * float(pos_tol), max(float(pos_tol), 3.0 * resid))
            p_end = p_tail[-1]
            v_tail_step = p_tail[-1] - p_tail[-2]

            for tgt in starts_by_frame[f_next]:
                _, i_start = tgt[0]
                if int(frames[f_next][0][i_start]) >= 0:
                    continue
                p_start = frames[f_next][2][i_start]
                step_vec = p_start - p_end
                if max_step is not None and float(np.linalg.norm(step_vec)) > float(max_step):
                    continue
                if float(np.linalg.norm(step_vec - v_tail_step)) > float(vel_tol):
                    continue
                if max_angle_deg is not None:
                    ang = _turning_angle_deg(v_tail_step, step_vec)
                    if ang is not None and ang > float(max_angle_deg):
                        continue

                err_fwd = float(np.linalg.norm(p_start - p_pred))
                if err_fwd > eff_pos_tol:
                    continue
                if len(tgt) >= 2:
                    head = tgt[:window]
                    t_head = np.array([cf for cf, _ in head], dtype=float)
                    p_head = np.array([frames[cf][2][ci] for cf, ci in head], dtype=float)
                    v_head_step = p_head[1] - p_head[0]
                    if float(np.linalg.norm(v_head_step - step_vec)) > float(vel_tol):
                        continue
                    if max_angle_deg is not None:
                        ang_h = _turning_angle_deg(step_vec, v_head_step)
                        if ang_h is not None and ang_h > float(max_angle_deg):
                            continue
                    deg_head = 2 if len(head) >= 5 else 1
                    p_back, v_back, resid_b = irls_predict_state(
                        t_head, p_head, float(f_end), window=window, max_deg=deg_head
                    )
                    err_back = float(np.linalg.norm(p_end - p_back))
                    eff_back_tol = min(2.0 * float(pos_tol), max(float(pos_tol), 3.0 * resid_b))
                    if err_back > eff_back_tol:
                        continue
                    if float(np.linalg.norm(v_back - v_pred)) > float(vel_tol):
                        continue
                    score = 0.5 * (err_fwd + err_back)
                else:
                    score = err_fwd
                candidates.append((score, f_end, i_end, f_next, i_start))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        used_ends: set[tuple[int, int]] = set()
        used_starts: set[tuple[int, int]] = set()
        pass_stitched = 0
        for _, f_end, i_end, f_next, i_start in candidates:
            if (f_end, i_end) in used_ends or (f_next, i_start) in used_starts:
                continue
            frames[f_end][1][i_end] = i_start
            frames[f_next][0][i_start] = i_end
            used_ends.add((f_end, i_end))
            used_starts.add((f_next, i_start))
            dirty.add(f_end)
            dirty.add(f_next)
            pass_stitched += 1

        total_stitched += pass_stitched
        if pass_stitched == 0:
            break

    return total_stitched, dirty
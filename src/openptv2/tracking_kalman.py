"""3D Constant-Acceleration Kalman Filter predictor for OpenPTV2 tracking.

Provides per-track 9D kinematic state tracking ([x, y, z, vx, vy, vz, ax, ay, az]),
O(1) prediction and measurement updates, and dynamic innovation ellipsoid gating
S = H P H^T + R that automatically adapts search volumes based on trajectory
uncertainty and missing-frame gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class KalmanTrackState:
    """Kinematic state container for a single 3D particle trajectory."""

    track_id: int
    state: np.ndarray  # (9,) float64: [x, y, z, vx, vy, vz, ax, ay, az]
    cov: np.ndarray  # (9, 9) float64: Covariance matrix P
    history_len: int = 1
    last_frame: int = 0
    history_positions: List[np.ndarray] = field(default_factory=list)
    history_times: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state.shape != (9,):
            raise ValueError(f"State must be shape (9,), got {self.state.shape}")
        if self.cov.shape != (9, 9):
            raise ValueError(f"Covariance must be shape (9, 9), got {self.cov.shape}")


class ConstantAccelerationKF3D:
    """3D Constant-Acceleration Kalman Filter with dynamic innovation gating."""

    def __init__(
        self,
        process_noise_acc: float = 1.0,
        measurement_noise: float = 0.05,
        v_max: float = 15.0,
        a_max: float = 10.0,
        gate_chi2: float = 11.34,
    ) -> None:
        self.process_noise_acc = float(process_noise_acc)
        self.measurement_noise = float(measurement_noise)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.gate_chi2 = float(gate_chi2)

        # Measurement matrix H: 3x9 mapping state -> position [x, y, z]
        self.H = np.zeros((3, 9), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise covariance R: 3x3
        self.R = (self.measurement_noise**2) * np.eye(3, dtype=np.float64)

    def init_state(
        self, track_id: int, pos: np.ndarray, frame_idx: int = 0
    ) -> KalmanTrackState:
        """Initialize a new track state from a single initial 3D position."""
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        state = np.zeros(9, dtype=np.float64)
        state[0:3] = pos

        cov = np.zeros((9, 9), dtype=np.float64)
        cov[0:3, 0:3] = (self.measurement_noise**2) * np.eye(3)
        cov[3:6, 3:6] = (self.v_max**2) * np.eye(3)
        cov[6:9, 6:9] = (self.a_max**2) * np.eye(3)

        return KalmanTrackState(
            track_id=track_id,
            state=state,
            cov=cov,
            history_len=1,
            last_frame=frame_idx,
            history_positions=[pos.copy()],
            history_times=[int(frame_idx)],
        )

    def compute_F(self, dt: float = 1.0) -> np.ndarray:
        """Construct 9x9 state transition matrix F for step dt."""
        F = np.eye(9, dtype=np.float64)
        dt2 = 0.5 * (dt**2)

        # Position update from velocity and acceleration
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[0, 6] = dt2
        F[1, 7] = dt2
        F[2, 8] = dt2

        # Velocity update from acceleration
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        return F

    def compute_Q(self, dt: float = 1.0) -> np.ndarray:
        """Construct 9x9 process noise covariance Q for step dt."""
        dt2 = 0.5 * (dt**2)
        dt3 = (dt**3) / 3.0
        dt4 = (dt**4) / 4.0

        q_var = self.process_noise_acc**2
        q3 = np.eye(3, dtype=np.float64)

        Q = np.zeros((9, 9), dtype=np.float64)
        # Position-Position block
        Q[0:3, 0:3] = dt4 * q3
        # Position-Velocity block
        Q[0:3, 3:6] = dt3 * q3
        Q[3:6, 0:3] = dt3 * q3
        # Position-Acceleration block
        Q[0:3, 6:9] = dt2 * q3
        Q[6:9, 0:3] = dt2 * q3
        # Velocity-Velocity block
        Q[3:6, 3:6] = (dt**2) * q3
        # Velocity-Acceleration block
        Q[3:6, 6:9] = dt * q3
        Q[6:9, 3:6] = dt * q3
        # Acceleration-Acceleration block
        Q[6:9, 6:9] = q3

        return Q * q_var

    def predict(
        self, track_state: KalmanTrackState, dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict state, covariance, position, and innovation covariance S.

        Returns
        -------
        pred_state : (9,) ndarray
        pred_cov : (9, 9) ndarray
        pred_pos : (3,) ndarray
        S : (3, 3) ndarray
            Spatial innovation covariance matrix for spatial gating.
        """
        F = self.compute_F(dt)
        Q = self.compute_Q(dt)

        pred_state = F @ track_state.state
        pred_cov = F @ track_state.cov @ F.T + Q

        pred_pos = pred_state[0:3]
        S = pred_cov[0:3, 0:3] + self.R

        return pred_state, pred_cov, pred_pos, S

    def mahalanobis_distance_sq(
        self, candidates: np.ndarray, pred_pos: np.ndarray, S: np.ndarray
    ) -> np.ndarray:
        """Compute Mahalanobis distance squared for N candidate positions.

        Parameters
        ----------
        candidates : (N, 3) ndarray
        pred_pos : (3,) ndarray
        S : (3, 3) ndarray

        Returns
        -------
        dist_sq : (N,) ndarray
        """
        cands = np.asarray(candidates, dtype=np.float64)
        if len(cands) == 0:
            return np.empty(0, dtype=np.float64)

        diff = cands - pred_pos  # (N, 3)
        S_inv = np.linalg.inv(S)

        # (N, 3) @ (3, 3) -> (N, 3)
        diff_Sinv = diff @ S_inv
        # Row-wise dot product
        dist_sq = np.einsum("ij,ij->i", diff_Sinv, diff)
        return dist_sq

    def in_gate(
        self,
        candidates: np.ndarray,
        pred_pos: np.ndarray,
        S: np.ndarray,
        gate_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Return boolean mask of candidate points lying inside the innovation ellipsoid gate."""
        thresh = self.gate_chi2 if gate_threshold is None else gate_threshold
        d_sq = self.mahalanobis_distance_sq(candidates, pred_pos, S)
        return d_sq <= thresh

    def update(
        self,
        track_state: KalmanTrackState,
        measurement: np.ndarray,
        dt: float = 1.0,
        frame_idx: Optional[int] = None,
    ) -> KalmanTrackState:
        """Perform Kalman prediction and measurement update with a new position observation."""
        meas = np.asarray(measurement, dtype=np.float64).reshape(3)
        pred_state, pred_cov, pred_pos, S = self.predict(track_state, dt)

        # Innovation residual y
        y = meas - pred_pos

        # Kalman Gain K = P_pred @ H^T @ S^-1
        S_inv = np.linalg.inv(S)
        K = pred_cov @ self.H.T @ S_inv  # (9, 3)

        # Updated state and covariance
        updated_state = pred_state + K @ y
        I9 = np.eye(9, dtype=np.float64)
        # Joseph form for numerical stability: (I - K H) P (I - K H)^T + K R K^T
        IKH = I9 - K @ self.H
        updated_cov = IKH @ pred_cov @ IKH.T + K @ self.R @ K.T

        next_frame = track_state.last_frame + int(dt) if frame_idx is None else frame_idx
        history_positions = track_state.history_positions + [meas.copy()]
        history_times = track_state.history_times + [int(next_frame)]

        return KalmanTrackState(
            track_id=track_state.track_id,
            state=updated_state,
            cov=updated_cov,
            history_len=track_state.history_len + 1,
            last_frame=next_frame,
            history_positions=history_positions,
            history_times=history_times,
        )

    def batch_predict(
        self, track_states: List[KalmanTrackState], dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized prediction across M active track states.

        Returns
        -------
        pred_states : (M, 9) ndarray
        pred_covs : (M, 9, 9) ndarray
        pred_positions : (M, 3) ndarray
        S_matrices : (M, 3, 3) ndarray
        """
        M = len(track_states)
        if M == 0:
            return (
                np.empty((0, 9)),
                np.empty((0, 9, 9)),
                np.empty((0, 3)),
                np.empty((0, 3, 3)),
            )

        states = np.array([ts.state for ts in track_states], dtype=np.float64)  # (M, 9)
        covs = np.array([ts.cov for ts in track_states], dtype=np.float64)  # (M, 9, 9)

        F = self.compute_F(dt)  # (9, 9)
        Q = self.compute_Q(dt)  # (9, 9)

        # F @ states.T -> (9, M) -> transpose to (M, 9)
        pred_states = (F @ states.T).T  # (M, 9)

        # F @ covs @ F^T + Q
        # (M, 9, 9)
        pred_covs = np.matmul(np.matmul(F, covs), F.T) + Q

        pred_positions = pred_states[:, 0:3]
        S_matrices = pred_covs[:, 0:3, 0:3] + self.R  # (M, 3, 3)

        return pred_states, pred_covs, pred_positions, S_matrices

"""
Modular Tracking Cost Functions and Adaptive Search Volume utilities.

Provides multi-term cost matrix calculation (distance, velocity error, acceleration error,
intensity error) and velocity-aligned adaptive search ellipsoids for advanced 2D/3D tracking.
Inspired by Matlab PTV, proPTV, MyPTV, and OpenLPT.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CostWeights:
    """Weights for multi-term tracking cost matrix calculation."""

    w_distance: float = 1.0
    w_velocity: float = 0.5
    w_acceleration: float = 0.2
    w_intensity: float = 0.1

    def normalize(self) -> "CostWeights":
        """Return normalized weights summing to 1.0."""
        total = self.w_distance + self.w_velocity + self.w_acceleration + self.w_intensity
        if total <= 0:
            return CostWeights(1.0, 0.0, 0.0, 0.0)
        return CostWeights(
            w_distance=self.w_distance / total,
            w_velocity=self.w_velocity / total,
            w_acceleration=self.w_acceleration / total,
            w_intensity=self.w_intensity / total,
        )


def compute_velocity_aligned_search_radius(
    velocities: np.ndarray,
    v_max: float,
    a_max: float,
    aspect_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adaptive, velocity-aligned anisotropic search radii along track direction.

    Args:
        velocities: (N, 3) velocity vectors of active tracks
        v_max: Maximum baseline velocity search radius
        a_max: Maximum acceleration search radius for seeded tracks
        aspect_ratio: Ratio of longitudinal (along velocity) to transverse search radius

    Returns:
        r_long: (N,) Longitudinal search radius
        r_trans: (N,) Transverse search radius
    """
    speeds = np.linalg.norm(velocities, axis=1)
    has_speed = speeds > 1e-6

    r_long = np.where(has_speed, a_max * aspect_ratio, v_max)
    r_trans = np.where(has_speed, a_max, v_max)

    return r_long, r_trans


def compute_multi_term_cost_matrix(
    pred_pos: np.ndarray,
    cand_pos: np.ndarray,
    pred_vel: Optional[np.ndarray] = None,
    pred_acc: Optional[np.ndarray] = None,
    pred_intensity: Optional[np.ndarray] = None,
    cand_intensity: Optional[np.ndarray] = None,
    weights: Optional[CostWeights] = None,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compute multi-term cost matrix between predicted track states and candidates.

    Args:
        pred_pos: (N_pred, 3) predicted positions
        cand_pos: (N_cand, 3) candidate positions
        pred_vel: Optional (N_pred, 3) estimated velocities of active tracks
        pred_acc: Optional (N_pred, 3) estimated accelerations
        pred_intensity: Optional (N_pred,) particle intensities/sizes
        cand_intensity: Optional (N_cand,) candidate intensities/sizes
        weights: CostWeights dataclass
        dt: Time step duration

    Returns:
        cost_matrix: (N_pred, N_cand) float64 array of total weighted candidate costs
    """
    if weights is None:
        weights = CostWeights()
    w = weights.normalize()

    n_pred = len(pred_pos)
    n_cand = len(cand_pos)

    if n_pred == 0 or n_cand == 0:
        return np.zeros((n_pred, n_cand), dtype=np.float64)

    from scipy.spatial.distance import cdist

    # 1. Spatial distance cost C_d = ||pos_cand - pos_pred||
    dist = cdist(pred_pos, cand_pos)  # (N_pred, N_cand) fast C implementation
    cost = w.w_distance * dist

    # 2. Velocity continuity cost C_v = || (pos_cand - pos_last)/dt - v_pred ||
    if w.w_velocity > 0 and pred_vel is not None and len(pred_vel) == n_pred:
        # Distance from projected position to candidate position
        last_pos = pred_pos - pred_vel * dt
        implied_dist = cdist(last_pos, cand_pos)
        vel_diff = implied_dist / max(dt, 1e-6)
        cost += w.w_velocity * vel_diff

    # 3. Acceleration cost C_a = || (v_link - v_pred)/dt - a_pred ||
    if w.w_acceleration > 0 and pred_acc is not None and len(pred_acc) == n_pred:
        last_pos_acc = pred_pos - pred_vel * dt - 0.5 * pred_acc * (dt**2)
        implied_dist_acc = cdist(last_pos_acc, cand_pos)
        acc_diff = implied_dist_acc / max(dt**2, 1e-6)
        cost += w.w_acceleration * acc_diff

    # 4. Intensity / blob size similarity cost C_i = |I_cand - I_pred|
    if (
        w.w_intensity > 0
        and pred_intensity is not None
        and cand_intensity is not None
        and len(pred_intensity) == n_pred
        and len(cand_intensity) == n_cand
    ):
        intensity_diff = np.abs(pred_intensity[:, None] - cand_intensity[None, :])
        cost += w.w_intensity * intensity_diff

    return cost


__all__ = [
    "CostWeights",
    "compute_velocity_aligned_search_radius",
    "compute_multi_term_cost_matrix",
]

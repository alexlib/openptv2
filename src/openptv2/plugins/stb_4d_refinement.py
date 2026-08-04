"""
4D Shake-The-Box (STB) Particle Position Refinement Prototype for OpenPTV2.

Implements particle position 'shaking' (3D coordinate optimization via multi-camera
projection residual minimization) inspired by OpenLPT / Shake-The-Box.
"""

from typing import List, Tuple, Any
import numpy as np


def shake_particle_position_3d(
    pos_3d: np.ndarray,
    cals: List[Any],
    cpar: Any,
    image_crops: List[np.ndarray],
    step_size: float = 0.005,
    max_iterations: int = 5,
) -> np.ndarray:
    """
    Refine a 3D particle position by 'shaking' (small 3D coordinate perturbations)
    to maximize reprojected pixel intensity match across multi-camera views.

    Args:
        pos_3d: (3,) float array of initial 3D position [X, Y, Z]
        cals: List of Calibration instances for all active cameras
        cpar: ControlParams instance
        image_crops: List of camera intensity image arrays
        step_size: Spatial step size in mm for numerical gradient calculation
        max_iterations: Maximum optimization iterations per particle

    Returns:
        refined_pos_3d: (3,) float array of optimized 3D position
    """
    from openptv2.algorithms.track import point_to_pixel

    refined = np.array(pos_3d, dtype=np.float64)

    for _ in range(max_iterations):
        grad = np.zeros(3, dtype=np.float64)

        for dim in range(3):
            pos_plus = refined.copy()
            pos_minus = refined.copy()
            pos_plus[dim] += step_size
            pos_minus[dim] -= step_size

            score_plus = 0.0
            score_minus = 0.0

            for cam_idx, (cal, img) in enumerate(zip(cals, image_crops)):
                if img is None or img.size == 0:
                    continue

                # Reproject pos_plus and pos_minus
                try:
                    px_p, py_p = point_to_pixel(pos_plus, cal, cpar)
                    px_m, py_m = point_to_pixel(pos_minus, cal, cpar)

                    ix_p, iy_p = int(round(px_p)), int(round(py_p))
                    ix_m, iy_m = int(round(px_m)), int(round(py_m))

                    if 0 <= iy_p < img.shape[0] and 0 <= ix_p < img.shape[1]:
                        score_plus += float(img[iy_p, ix_p])
                    if 0 <= iy_m < img.shape[0] and 0 <= ix_m < img.shape[1]:
                        score_minus += float(img[iy_m, ix_m])
                except Exception:
                    continue

            grad[dim] = (score_plus - score_minus) / (2.0 * step_size)

        # Update position along positive intensity gradient
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-6:
            break

        refined += (grad / norm_grad) * (step_size * 0.5)

    return refined


__all__ = ["shake_particle_position_3d"]

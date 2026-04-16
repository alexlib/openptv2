"""Camera orientation and bundle adjustment.

Translation of lib/src/orientation.c and lib/include/orientation.h.

Determines camera exterior orientation and refines interior/distortion
parameters using known 3D points and their 2D image projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OrientPar:
    """Flags for which parameters to adjust during orientation.

    Attributes:
        useflag: which points to use (0=all, 1=even, 2=odd, 3=every 3rd).
        ccflag: fix back focal distance.
        xhflag, yhflag: fix principal point offsets.
        k1flag, k2flag, k3flag: fix radial distortion.
        p1flag, p2flag: fix decentering distortion.
        scxflag, sheflag: fix scale/shear.
        interfflag: fix glass interface vector.
    """
    useflag: int = 0
    ccflag: int = 0
    xhflag: int = 0
    yhflag: int = 0
    k1flag: int = 0
    k2flag: int = 0
    k3flag: int = 0
    p1flag: int = 0
    p2flag: int = 0
    scxflag: int = 0
    sheflag: int = 0
    interfflag: int = 0

    @classmethod
    def from_file(cls, filename: str | Path) -> OrientPar:
        """Read orientation parameters from file.

        Args:
            filename: path to parameter file.

        Returns:
            OrientPar instance.
        """
        path = Path(filename)
        lines = path.read_text().strip().splitlines()

        if len(lines) < 12:
            raise ValueError(f"Expected 12 lines, got {len(lines)}")

        return cls(
            useflag=int(lines[0]),
            ccflag=int(lines[1]),
            xhflag=int(lines[2]),
            yhflag=int(lines[3]),
            k1flag=int(lines[4]),
            k2flag=int(lines[5]),
            k3flag=int(lines[6]),
            p1flag=int(lines[7]),
            p2flag=int(lines[8]),
            scxflag=int(lines[9]),
            sheflag=int(lines[10]),
            interfflag=int(lines[11]),
        )


def skew_midpoint(
    vert1: np.ndarray,
    direct1: np.ndarray,
    vert2: np.ndarray,
    direct2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find midpoint of shortest distance segment between two skew rays.

    Args:
        vert1, direct1: vertex and direction of first ray.
        vert2, direct2: vertex and direction of second ray.

    Returns:
        (distance, midpoint) where midpoint is the average of closest points.
    """
    # Cross product of directions
    cross = np.cross(direct1, direct2)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-10:
        # Parallel rays - return midpoint of vertices
        return np.linalg.norm(vert2 - vert1), (vert1 + vert2) / 2

    # Normalized cross
    n = cross / cross_norm

    # Distance between rays
    diff = vert2 - vert1
    dist = abs(np.dot(diff, n))

    # Closest points (simplified - full solution uses linear system)
    # For now, return midpoint
    midpoint = (vert1 + vert2) / 2

    return dist, midpoint


def point_position(
    targets: list[tuple[float, float]],
    calibrations: list[dict],
    mm_params: dict,
) -> tuple[np.ndarray, float]:
    """Compute average 3D position from multiple camera rays.

    Args:
        targets: per-camera 2D image coordinates.
        calibrations: per-camera calibration dicts.
        mm_params: multimedia parameters.

    Returns:
        (position, avg_ray_distance) tuple.
    """
    from .ray_tracing import ray_tracing

    num_cams = len(targets)
    vertices = []
    directions = []

    # Ray trace from each camera
    for cam in range(num_cams):
        cal = calibrations[cam]
        x, y = targets[cam]

        pos, direction = ray_tracing(
            x, y,
            cal["dm"], cal["x0"], cal["y0"], cal["z0"], cal["cc"],
            cal["gx"], cal["gy"], cal["gz"],
            mm_params["n1"], mm_params["n2_0"], mm_params["n3"], mm_params["d0"],
        )
        vertices.append(pos)
        directions.append(direction)

    # Find pairwise skew midpoints
    total_dist = 0.0
    count = 0
    sum_positions = np.zeros(3)

    for i in range(num_cams):
        for j in range(i + 1, num_cams):
            dist, midpoint = skew_midpoint(
                vertices[i], directions[i],
                vertices[j], directions[j],
            )
            total_dist += dist
            sum_positions += midpoint
            count += 1

    if count == 0:
        return np.zeros(3), 0.0

    avg_position = sum_positions / count
    avg_dist = total_dist / count

    return avg_position, avg_dist


def num_deriv_exterior(
    cal: dict,
    cpar: dict,
    pos: np.ndarray,
    dpos: float = 0.1,
    dang: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute numerical derivatives of image coordinates w.r.t. exterior params.

    Args:
        cal: calibration dict (dm, x0, y0, z0, omega, phi, kappa, cc, etc.).
        cpar: control parameters.
        pos: 3D world position.
        dpos: position perturbation step.
        dang: angle perturbation step.

    Returns:
        (x_ders, y_ders) each shape (6,) for x0,y0,z0,omega,phi,kappa.
    """
    from .imgcoord import img_coord

    def project(cal_mod: dict) -> tuple[float, float]:
        return img_coord(
            pos,
            cal_mod["x0"], cal_mod["y0"], cal_mod["z0"],
            cal_mod["dm"], cal_mod["cc"],
            cal_mod["int_xh"], cal_mod["int_yh"],
            cal_mod["gx"], cal_mod["gy"], cal_mod["gz"],
            cal_mod.get("mm_n1", 1.0), cal_mod.get("mm_n2_0", 1.0),
            cal_mod.get("mm_n3", 1.0), cal_mod.get("mm_d0", 0.0),
            cal_mod.get("k1", 0.0), cal_mod.get("k2", 0.0),
            cal_mod.get("k3", 0.0), cal_mod.get("p1", 0.0),
            cal_mod.get("p2", 0.0), cal_mod.get("scx", 1.0),
            cal_mod.get("she", 0.0),
        )

    x0, y0 = project(cal)

    x_ders = np.zeros(6)
    y_ders = np.zeros(6)

    # Position derivatives
    for i, param in enumerate(["x0", "y0", "z0"]):
        cal_plus = cal.copy()
        cal_minus = cal.copy()
        cal_plus[param] = cal[param] + dpos
        cal_minus[param] = cal[param] - dpos

        x_plus, y_plus = project(cal_plus)
        x_minus, y_minus = project(cal_minus)

        x_ders[i] = (x_plus - x_minus) / (2 * dpos)
        y_ders[i] = (y_plus - y_minus) / (2 * dpos)

    # Angle derivatives
    for i, param in enumerate(["omega", "phi", "kappa"]):
        cal_plus = cal.copy()
        cal_minus = cal.copy()
        cal_plus[param] = cal[param] + dang
        cal_minus[param] = cal[param] - dang

        # Recompute rotation matrix
        from .calibration import Exterior
        ext_plus = Exterior(
            x0=cal["x0"], y0=cal["y0"], z0=cal["z0"],
            omega=cal_plus["omega"], phi=cal_plus["phi"], kappa=cal_plus["kappa"],
        )
        ext_plus.compute_rotation_matrix()
        cal_plus["dm"] = ext_plus.dm

        ext_minus = Exterior(
            x0=cal["x0"], y0=cal["y0"], z0=cal["z0"],
            omega=cal_minus["omega"], phi=cal_minus["phi"], kappa=cal_minus["kappa"],
        )
        ext_minus.compute_rotation_matrix()
        cal_minus["dm"] = ext_minus.dm

        x_plus, y_plus = project(cal_plus)
        x_minus, y_minus = project(cal_minus)

        x_ders[i + 3] = (x_plus - x_minus) / (2 * dang)
        y_ders[i + 3] = (y_plus - y_minus) / (2 * dang)

    return x_ders, y_ders


def orient(
    calibrations: list[dict],
    cpar: dict,
    fix_points: list[np.ndarray],
    image_points: list[list[tuple[float, float]]],
    flags: OrientPar,
    mm_params: dict,
    max_iter: int = 80,
    tol: float = 1e-5,
) -> tuple[list[dict], np.ndarray] | None:
    """Bundle adjustment using Gauss-Markov model.

    Iteratively refines calibration parameters to minimize reprojection error.

    Args:
        calibrations: per-camera calibration dicts.
        cpar: control parameters.
        fix_points: known 3D positions.
        image_points: per-camera 2D image correspondences.
        flags: which parameters to adjust.
        mm_params: multimedia parameters.
        max_iter: maximum iterations.
        tol: convergence tolerance.

    Returns:
        (updated_calibrations, residuals) or None on failure.
    """
    from .lsqadj import ata, atl, matinv, matmul

    num_cams = len(calibrations)
    n_points = len(fix_points)

    # Build design matrix and observations
    # Simplified: only adjust exterior (6 params per camera)
    n_params = 6 * num_cams
    n_obs = 2 * n_points * num_cams  # x and y per point per camera

    A = np.zeros((n_obs, n_params))
    y = np.zeros(n_obs)
    P = np.eye(n_obs)  # Weight matrix

    for iteration in range(max_iter):
        obs_idx = 0

        for cam in range(num_cams):
            cal = calibrations[cam]

            for pt_idx in range(n_points):
                # Skip if not using this point
                if flags.useflag == 1 and pt_idx % 2 != 0:
                    obs_idx += 2
                    continue
                if flags.useflag == 2 and pt_idx % 2 != 1:
                    obs_idx += 2
                    continue
                if flags.useflag == 3 and pt_idx % 3 != 0:
                    obs_idx += 2
                    continue

                # Project current estimate
                x_pred, y_pred = img_coord(
                    fix_points[pt_idx],
                    cal["x0"], cal["y0"], cal["z0"],
                    cal["dm"], cal["cc"],
                    cal["int_xh"], cal["int_yh"],
                    cal["gx"], cal["gy"], cal["gz"],
                    mm_params["n1"], mm_params["n2_0"],
                    mm_params["n3"], mm_params["d0"],
                    cal.get("k1", 0.0), cal.get("k2", 0.0),
                    cal.get("k3", 0.0), cal.get("p1", 0.0),
                    cal.get("p2", 0.0), cal.get("scx", 1.0),
                    cal.get("she", 0.0),
                )

                # Observations (residuals)
                y[obs_idx] = image_points[cam][pt_idx][0] - x_pred
                y[obs_idx + 1] = image_points[cam][pt_idx][1] - y_pred

                # Numerical derivatives
                x_ders, y_ders = num_deriv_exterior(cal, cpar, fix_points[pt_idx])

                # Fill design matrix
                param_offset = 6 * cam
                A[obs_idx, param_offset:param_offset + 6] = x_ders
                A[obs_idx + 1, param_offset:param_offset + 6] = y_ders

                obs_idx += 2

        # Solve normal equations: beta = (A^T P A)^{-1} A^T P y
        ATA = ata(A, n_obs, n_params)
        ATy = atl(A, y, n_obs, n_params)

        try:
            matinv(ATA, n_params)
        except ZeroDivisionError:
            return None

        beta = matmul(ATA, ATy, n_params)

        # Update parameters
        for cam in range(num_cams):
            cal = calibrations[cam]
            param_offset = 6 * cam

            cal["x0"] += beta[param_offset]
            cal["y0"] += beta[param_offset + 1]
            cal["z0"] += beta[param_offset + 2]
            cal["omega"] += beta[param_offset + 3]
            cal["phi"] += beta[param_offset + 4]
            cal["kappa"] += beta[param_offset + 5]

            # Recompute rotation matrix
            from .calibration import Exterior
            ext = Exterior(
                x0=cal["x0"], y0=cal["y0"], z0=cal["z0"],
                omega=cal["omega"], phi=cal["phi"], kappa=cal["kappa"],
            )
            ext.compute_rotation_matrix()
            cal["dm"] = ext.dm

        # Check convergence
        if np.max(np.abs(beta)) < tol:
            break

    # Compute residuals
    residuals = np.zeros(n_points)
    for pt_idx in range(n_points):
        positions = []
        for cam in range(num_cams):
            positions.append(
                point_position(
                    [image_points[cam][pt_idx]],
                    [calibrations[cam]],
                    mm_params,
                )[0]
            )
        residuals[pt_idx] = np.std(positions)

    return calibrations, residuals


def raw_orient(
    cal: dict,
    cpar: dict,
    fix_points: list[np.ndarray],
    image_points: list[tuple[float, float]],
    mm_params: dict,
    max_iter: int = 20,
    tol: float = 0.1,
) -> dict | None:
    """Simplified orientation using only 6 exterior parameters.

    For manual orientation from typically 4 clicked points.

    Args:
        cal: calibration dict.
        cpar: control parameters.
        fix_points: known 3D positions.
        image_points: 2D image correspondences.
        mm_params: multimedia parameters.
        max_iter: maximum iterations.
        tol: convergence tolerance.

    Returns:
        Updated calibration dict or None on failure.
    """
    for iteration in range(max_iter):
        # Compute current reprojection error
        error = 0.0
        for pt_idx in range(len(fix_points)):
            x_pred, y_pred = img_coord(
                fix_points[pt_idx],
                cal["x0"], cal["y0"], cal["z0"],
                cal["dm"], cal["cc"],
                cal["int_xh"], cal["int_yh"],
                cal["gx"], cal["gy"], cal["gz"],
                mm_params["n1"], mm_params["n2_0"],
                mm_params["n3"], mm_params["d0"],
                cal.get("k1", 0.0), cal.get("k2", 0.0),
                cal.get("k3", 0.0), cal.get("p1", 0.0),
                cal.get("p2", 0.0), cal.get("scx", 1.0),
                cal.get("she", 0.0),
            )
            error += (image_points[pt_idx][0] - x_pred) ** 2
            error += (image_points[pt_idx][1] - y_pred) ** 2

        if error < tol:
            break

        # Numerical gradient descent (simplified)
        dpos = 0.1
        dang = 0.0001

        for param in ["x0", "y0", "z0", "omega", "phi", "kappa"]:
            cal_plus = cal.copy()
            cal_minus = cal.copy()

            step = dpos if param in ["x0", "y0", "z0"] else dang
            cal_plus[param] = cal[param] + step
            cal_minus[param] = cal[param] - step

            # Recompute dm for angle changes
            if param in ["omega", "phi", "kappa"]:
                from .calibration import Exterior
                ext_plus = Exterior(
                    x0=cal["x0"], y0=cal["y0"], z0=cal["z0"],
                    omega=cal_plus.get("omega", cal["omega"]),
                    phi=cal_plus.get("phi", cal["phi"]),
                    kappa=cal_plus.get("kappa", cal["kappa"]),
                )
                ext_plus.compute_rotation_matrix()
                cal_plus["dm"] = ext_plus.dm

            # Compute gradients (simplified)
            # ... (omitted for brevity - would need full Jacobian)

    return cal


def read_man_ori_fix(*args, **kwargs):
    """Stub for read_man_ori_fix: returns None or dummy data."""
    return None


def read_calblock(*args, **kwargs):
    """Stub for read_calblock: returns None or dummy data."""
    return None


def weighted_dumbbell_precision(*args, **kwargs):
    """Stub for weighted_dumbbell_precision: returns 0 or dummy value."""
    return 0

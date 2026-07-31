"""
Calibration functions for PyPTV.

Contains all camera calibration routines including:
- Scipy-based full calibration (interior + exterior parameters)
- Dumbbell target calibration with bundle adjustment
- Particle-based calibration

Extracted from ptv.py to reduce god-module coupling.
"""

import os
from typing import List

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, minimize

from openptv2.calibration import Calibration
from openptv2.parameters import ControlParams
from openptv2.tracking_framebuf import TargetArray
from openptv2.transforms import convert_arr_pixel_to_metric

# Calibration parameter names
NAMES = ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2", "scale", "shear", "interf"]


def _read_calibrations(cpar: ControlParams, num_cams: int) -> List[Calibration]:
    """Read calibration files for all cameras.

    Returns empty/default calibrations if files don't exist, which is normal
    for the calibration GUI before calibrations have been created.
    """
    cals = []
    for i_cam in range(num_cams):
        cal = Calibration()
        base_name = cpar.get_cal_img_base_name(i_cam)

        if not base_name:
            print(
                f"Calibration base name missing for camera {i_cam + 1} - using defaults"
            )
            cals.append(cal)
            continue

        ori_file = base_name + ".ori"
        addpar_file = base_name + ".addpar"

        # Check if calibration files exist and are readable
        ori_exists = os.path.isfile(ori_file) and os.access(ori_file, os.R_OK)
        addpar_exists = os.path.isfile(addpar_file) and os.access(addpar_file, os.R_OK)

        if ori_exists and addpar_exists:
            # Both files exist, load them
            cal.from_file(ori_file, addpar_file)
            print(f"Loaded calibration for camera {i_cam + 1} from {ori_file}")
        else:
            # Files don't exist yet - this is normal for calibration GUI
            # Create default/empty calibration
            print(
                f"Calibration files not found for camera {i_cam + 1} - using defaults"
            )
            print(
                f"  Missing: {ori_file if not ori_exists else ''} {addpar_file if not addpar_exists else ''}"
            )

        cals.append(cal)

    return cals


def full_scipy_calibration(
    cal: Calibration, XYZ: np.ndarray, targs: TargetArray, cpar: ControlParams, flags=[]
):
    """Full calibration using scipy.optimize"""
    from openptv2.imgcoord import image_coordinates
    from openptv2.transforms import convert_arr_metric_to_pixel

    def _residuals_k(x, cal, XYZ, xy, cpar):
        cal.set_radial_distortion(x)
        targets = convert_arr_metric_to_pixel(
            image_coordinates(XYZ, cal, cpar.get_multimedia_params()), cpar
        )
        xyt = np.array([t.pos() if t.pnr() != -999 else [np.nan, np.nan] for t in xy])
        residuals = np.nan_to_num(xyt - targets)
        return np.sum(residuals**2)

    def _residuals_p(x, cal, XYZ, xy, cpar):
        cal.set_decentering(x)
        targets = convert_arr_metric_to_pixel(
            image_coordinates(XYZ, cal, cpar.get_multimedia_params()), cpar
        )
        xyt = np.array([t.pos() if t.pnr() != -999 else [np.nan, np.nan] for t in xy])
        residuals = np.nan_to_num(xyt - targets)
        return np.sum(residuals**2)

    def _residuals_s(x, cal, XYZ, xy, cpar):
        cal.set_affine_trans(x)
        targets = convert_arr_metric_to_pixel(
            image_coordinates(XYZ, cal, cpar.get_multimedia_params()), cpar
        )
        xyt = np.array([t.pos() if t.pnr() != -999 else [np.nan, np.nan] for t in xy])
        residuals = np.nan_to_num(xyt - targets)
        return np.sum(residuals**2)

    def _residuals_combined(x, cal, XYZ, xy, cpar):
        cal.set_radial_distortion(x[:3])
        cal.set_decentering(x[3:5])
        cal.set_affine_trans(x[5:])

        targets = convert_arr_metric_to_pixel(
            image_coordinates(XYZ, cal, cpar.get_multimedia_params()), cpar
        )
        xyt = np.array([t.pos() if t.pnr() != -999 else [np.nan, np.nan] for t in xy])
        residuals = np.nan_to_num(xyt - targets)
        return residuals

    if any(flag in flags for flag in ["k1", "k2", "k3"]):
        sol = minimize(
            _residuals_k,
            cal.get_radial_distortion(),
            args=(cal, XYZ, targs, cpar),
            method="Nelder-Mead",
            tol=1e-11,
            options={"disp": True},
        )
        radial = sol.x
        cal.set_radial_distortion(radial)
    else:
        radial = cal.get_radial_distortion()

    if any(flag in flags for flag in ["p1", "p2"]):
        sol = minimize(
            _residuals_p,
            cal.get_decentering(),
            args=(cal, XYZ, targs, cpar),
            method="Nelder-Mead",
            tol=1e-11,
            options={"disp": True},
        )
        decentering = sol.x
        cal.set_decentering(decentering)
    else:
        decentering = cal.get_decentering()

    if any(flag in flags for flag in ["scale", "shear"]):
        sol = minimize(
            _residuals_s,
            cal.get_affine(),
            args=(cal, XYZ, targs, cpar),
            method="Nelder-Mead",
            tol=1e-11,
            options={"disp": True},
        )
        affine = sol.x
        cal.set_affine_trans(affine)

    else:
        affine = cal.get_affine()

    residuals = _residuals_combined(
        np.r_[radial, decentering, affine], cal, XYZ, targs, cpar
    )

    residuals /= 100

    return residuals


def dumbbell_target_func(targets, cpar, calibs, db_length, db_weight):
    """
    Calculate the ray convergence error for a set of targets and calibrations.

    Arguments:
    targets : np.ndarray
        Array of shape (num_cams, num_targets, 2), where num_cams is the number of cameras,
        num_targets is the total number of dumbbell endpoints (should be even, typically 2 per frame),
        and 2 corresponds to the (x, y) metric coordinates for each target in each camera.
    cpar : ControlParams
        A ControlParams object describing the overall setting.
    calibs : list of Calibration
        An array of per-camera Calibration objects.
    db_length : float
        Expected distance between two dumbbell points.
    db_weight : float
        Weight of the distance error in the target function.

    Returns:
    float
        The weighted ray convergence + length error measure.
    """
    from openptv2.orientation import multi_cam_point_positions

    num_cams = cpar.get_num_cams()
    num_targs = targets.shape[1]
    multimed_pars = cpar.get_multimedia_params()

    # Prepare the result arrays
    res = [np.zeros((num_cams, 3)) for _ in range(2)]
    res_current = None
    dtot = 0.0
    len_err_tot = 0.0
    dist = 0.0

    # Iterate over pairs of targets
    if num_targs % 2 != 0:
        raise ValueError("Number of targets must be even for dumbbell calibration")

    # Process each target pair
    for pt in range(0, num_targs, 2):
        # For each pair of targets (dumbbell ends)
        # Get their 2D positions in all cameras for this pair
        pair_targets = targets[:, pt : pt + 2, :]  # shape: (num_cams, 2, pos)
        # Compute their 3D positions using all cameras
        # Each column: [cam1_t1, cam2_t1, ..., camN_t1], [cam1_t2, ..., camN_t2]
        # So we need to transpose to (2, num_cams, pos)
        pair_targets = pair_targets.transpose(1, 0, 2)  # shape: (2, num_cams, pos)
        # Get 3D positions for each end
        xyz1, err1 = multi_cam_point_positions(
            pair_targets[0, np.newaxis], cpar, calibs
        )
        xyz2, err2 = multi_cam_point_positions(
            pair_targets[1, np.newaxis], cpar, calibs
        )
        # xyz1, xyz2 are (1, 3) arrays (single point)
        # Compute the distance between the two ends
        dist = np.linalg.norm(xyz1[0] - xyz2[0])
        # Accumulate squared length error for smooth objective
        len_err_tot += (dist - db_length) ** 2
        # Accumulate the ray convergence error (sum of distances from rays to intersection)
        # Use the error returned by point_positions
        dtot += err1 + err2

    # Calculate the total error
    len_err_tot /= 2.0  # since we counted pairs, divide by 2

    # Calculate the total error as a weighted sum of ray convergence and length error
    dtot /= num_targs / 2.0  # average over pairs
    if db_length <= 0:
        raise ValueError("Dumbbell length must be positive")

    if db_weight < 0:
        raise ValueError("Dumbbell weight must be non-negative")

    # Return the total error
    return dtot + db_weight * len_err_tot / (num_targs / 2.0)


def dumbbell_target_residuals(targets, cpar, calibs, db_length, db_weight):
    """Return residuals per target pair for least-squares optimization."""
    from openptv2.orientation import multi_cam_point_positions

    num_targs = targets.shape[1]
    if num_targs % 2 != 0:
        raise ValueError("Number of targets must be even for dumbbell calibration")
    if db_length <= 0:
        raise ValueError("Dumbbell length must be positive")
    if db_weight < 0:
        raise ValueError("Dumbbell weight must be non-negative")

    residuals = []
    for pt in range(0, num_targs, 2):
        pair_targets = targets[:, pt : pt + 2, :].transpose(1, 0, 2)
        xyz1, err1 = multi_cam_point_positions(
            pair_targets[0, np.newaxis], cpar, calibs
        )
        xyz2, err2 = multi_cam_point_positions(
            pair_targets[1, np.newaxis], cpar, calibs
        )
        dist = np.linalg.norm(xyz1[0] - xyz2[0])
        residuals.append(float(err1))
        residuals.append(float(err2))
        if db_weight > 0:
            residuals.append(np.sqrt(db_weight) * (dist - db_length))

    residuals = np.asarray(residuals, dtype=float)
    return np.nan_to_num(residuals, nan=1e6, posinf=1e6, neginf=-1e6)


def dumbbell_ba_residuals(
    calib_vec,
    targets,
    cpar,
    calibs,
    active_cams,
    db_length,
    db_weight,
    pos_scale=1.0,
):
    """Bundle adjustment residuals for dumbbell calibration.

    calib_vec packs active camera extrinsics and per-frame 3D endpoints.
    targets is shaped (num_cams, num_frames, 2, 2) in metric coordinates.
    """
    from openptv2.imgcoord import image_coordinates

    if db_length <= 0:
        raise ValueError("Dumbbell length must be positive")
    if db_weight < 0:
        raise ValueError("Dumbbell weight must be non-negative")

    num_cams, num_frames, num_pts, _ = targets.shape
    if num_pts != 2:
        raise ValueError("Targets must contain exactly 2 points per frame")

    active_cams = np.asarray(active_cams, dtype=bool)
    num_active = int(np.sum(active_cams))
    cam_params_len = num_active * 6
    if calib_vec.shape[0] < cam_params_len:
        raise ValueError("calib_vec too short for active camera parameters")

    calib_pars = calib_vec[:cam_params_len].reshape(-1, 2, 3)

    ptr = 0
    for cam, cal in enumerate(calibs):
        if not active_cams[cam]:
            continue
        pars = calib_pars[ptr]
        cal.set_pos(pars[0] * pos_scale)
        cal.set_angles(pars[1])
        ptr += 1

    points = calib_vec[cam_params_len:]
    expected_len = num_frames * 2 * 3
    if points.shape[0] != expected_len:
        raise ValueError(
            f"Expected {expected_len} point parameters, got {points.shape[0]}"
        )
    points = points.reshape(num_frames, 2, 3)

    mm_params = cpar.get_multimedia_params()
    residuals = []

    for frame_idx in range(num_frames):
        xyz = points[frame_idx]
        for cam in range(num_cams):
            proj = image_coordinates(xyz, calibs[cam], mm_params)
            diff = targets[cam, frame_idx] - proj
            residuals.extend(diff.ravel())

        if db_weight > 0:
            length_err = np.linalg.norm(xyz[0] - xyz[1]) - db_length
            residuals.append(np.sqrt(db_weight) * length_err)

    residuals = np.asarray(residuals, dtype=float)
    return np.nan_to_num(residuals, nan=1e6, posinf=1e6, neginf=-1e6)


def dumbbell_ba_jac_sparsity(
    targets: np.ndarray,
    active_cams: np.ndarray,
    db_weight: float,
) -> sparse.csr_matrix:
    """Return Jacobian sparsity pattern for dumbbell bundle adjustment."""
    num_cams, num_frames, num_pts, _ = targets.shape
    if num_pts != 2:
        raise ValueError("Targets must contain exactly 2 points per frame")

    active_cams = np.asarray(active_cams, dtype=bool)
    num_active = int(np.sum(active_cams))
    cam_params_len = num_active * 6

    per_frame_cam_residuals = num_cams * 4
    per_frame_len_residuals = 1 if db_weight > 0 else 0
    residuals_per_frame = per_frame_cam_residuals + per_frame_len_residuals
    total_residuals = num_frames * residuals_per_frame
    total_params = cam_params_len + num_frames * 6

    pattern = sparse.lil_matrix((total_residuals, total_params), dtype=bool)

    active_map = {}
    active_idx = 0
    for cam_idx, is_active in enumerate(active_cams):
        if is_active:
            active_map[cam_idx] = active_idx
            active_idx += 1

    row = 0
    for frame_idx in range(num_frames):
        point_base = cam_params_len + frame_idx * 6
        point_cols = list(range(point_base, point_base + 6))

        for cam_idx in range(num_cams):
            cam_cols = []
            if cam_idx in active_map:
                cam_base = active_map[cam_idx] * 6
                cam_cols = list(range(cam_base, cam_base + 6))

            for _ in range(4):
                if cam_cols:
                    pattern[row, cam_cols] = True
                pattern[row, point_cols] = True
                row += 1

        if db_weight > 0:
            pattern[row, point_cols] = True
            row += 1

    return pattern.tocsr()


def calib_convergence(
    calib_vec, targets, calibs, active_cams, cpar, db_length, db_weight, pos_scale=1.0
):
    """
    Mediated the ray_convergence function and the parameter format used by
    SciPy optimization routines, by taking a vector of variable calibration
    parameters and pouring it into the Calibration objects understood by
    OpenPTV.

    Arguments:
    calib_vec - 1D array. 3 elements: camera 1 position, 3 element: camera 1
        angles, next 6 for camera 2 etc.
    targets - a (c,t,2) array, for t target metric positions in each of c
        cameras.
    calibs - an array of per-camera Calibration objects. The permanent fields
        are retained, the variable fields get overwritten.
    active_cams - a sequence of True/False values stating whether the
        corresponding camera is free to move or just a parameter.
    cpar - a ControlParams object describing the overall setting.
    db_length - expected distance between two dumbbell points.
    db_weight - weight of the distance error in the target function.

    Returns:
    The weighted ray convergence + length error measure.
    """
    calib_pars = calib_vec.reshape(-1, 2, 3)

    for cam, cal in enumerate(calibs):
        if not active_cams[cam]:
            continue

        # Pop a parameters line:
        pars = calib_pars[0]
        calib_pars = calib_pars[1:]

        cal.set_pos(pars[0] * pos_scale)
        cal.set_angles(pars[1])

    return dumbbell_target_func(targets, cpar, calibs, db_length, db_weight)


def calib_convergence_residuals(
    calib_vec, targets, calibs, active_cams, cpar, db_length, db_weight, pos_scale=1.0
):
    """Return residual vector for least-squares optimization."""
    calib_pars = calib_vec.reshape(-1, 2, 3)

    for cam, cal in enumerate(calibs):
        if not active_cams[cam]:
            continue

        pars = calib_pars[0]
        calib_pars = calib_pars[1:]

        cal.set_pos(pars[0] * pos_scale)
        cal.set_angles(pars[1])

    return dumbbell_target_residuals(targets, cpar, calibs, db_length, db_weight)


def calib_dumbbell(cal_gui) -> None:
    """Calibration with dumbbell targets.

    Args:
        cal_gui: Calibration GUI object with experiment.pm attribute
    """
    # Import here to avoid circular dependency
    from .ptv import py_start_proc_c, read_targets

    pm = cal_gui.experiment.pm
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    num_cams = cpar.get_num_cams()
    target_filenames = pm.get_target_filenames()

    # Get dumbbell length from parameters (or set default)
    dumbbell_params = pm.get_parameter("dumbbell") or {}
    db_length = dumbbell_params.get("dumbbell_scale")
    db_weight = dumbbell_params.get("dumbbell_penalty_weight")
    db_eps = float(dumbbell_params.get("dumbbell_eps") or 0.0)
    fixed_cam_param = int(dumbbell_params.get("dumbbell_fixed_camera") or 0)
    if db_length is None or float(db_length) <= 0:
        raise ValueError("dumbbell.dumbbell_scale must be > 0")
    if db_weight is None or float(db_weight) < 0:
        raise ValueError("dumbbell.dumbbell_penalty_weight must be >= 0")

    # Get frame range
    first_frame = spar.get_first()
    last_frame = spar.get_last()

    num_frames = last_frame - first_frame + 1
    all_targs = []
    coverage = np.zeros(num_cams, dtype=int)
    for frame in range(num_frames):
        frame_targets = []
        valid = True
        for cam in range(num_cams):
            targs = read_targets(target_filenames[cam], first_frame + frame)
            if len(targs) == 2:
                coverage[cam] += 1
            else:
                valid = False
            if len(targs) != 2:
                valid = False
                break
            frame_targets.append([targ.pos() for targ in targs])
        if valid:
            all_targs.append(frame_targets)

    if len(all_targs) == 0:
        raise ValueError(
            "No frames with two targets per camera found for dumbbell calibration"
        )

    all_targs = np.array(all_targs)
    assert all_targs.shape[1] == num_cams and all_targs.shape[2] == 2
    num_frames, n_cams, num_targs, num_pos = all_targs.shape

    metric_by_cam = []
    for cam in range(num_cams):
        cam_pixels = all_targs[:, cam, :, :].reshape(num_frames * num_targs, num_pos)
        cam_metric = convert_arr_pixel_to_metric(cam_pixels, cpar)
        metric_by_cam.append(cam_metric.reshape(num_frames, num_targs, num_pos))
    metric_by_cam = np.array(metric_by_cam)

    if db_eps > 0:
        from openptv2.orientation import multi_cam_point_positions

        keep_mask = np.ones(num_frames, dtype=bool)
        removed = 0
        for frame_idx in range(num_frames):
            frame_targets = metric_by_cam[:, frame_idx, :, :]
            xyz1, _err1 = multi_cam_point_positions(
                frame_targets[:, 0, :][np.newaxis], cpar, cals
            )
            xyz2, _err2 = multi_cam_point_positions(
                frame_targets[:, 1, :][np.newaxis], cpar, cals
            )
            dist = np.linalg.norm(xyz1[0] - xyz2[0])
            if abs(dist - db_length) > db_eps:
                keep_mask[frame_idx] = False
                removed += 1
        if removed > 0:
            print(f"Filtered {removed} frame(s) by dumbbell length eps {db_eps}")
        metric_by_cam = metric_by_cam[:, keep_mask, :, :]
        num_frames = metric_by_cam.shape[1]
        if num_frames == 0:
            raise ValueError("All frames filtered by dumbbell length eps")

    def _print_camera_residuals(label: str, metric_targets: np.ndarray) -> None:
        from openptv2.imgcoord import image_coordinates
        from openptv2.orientation import multi_cam_point_positions

        num_cams_local, num_frames_local, num_targs_local, _ = metric_targets.shape
        sums = np.zeros(num_cams_local, dtype=float)
        counts = np.zeros(num_cams_local, dtype=int)
        mm_params = cpar.get_multimedia_params()

        for frame_idx in range(num_frames_local):
            frame_targets = metric_targets[:, frame_idx, :, :]
            xyz1, _err1 = multi_cam_point_positions(
                frame_targets[:, 0, :][np.newaxis], cpar, cals
            )
            xyz2, _err2 = multi_cam_point_positions(
                frame_targets[:, 1, :][np.newaxis], cpar, cals
            )
            xyz = np.vstack([xyz1, xyz2])

            for cam in range(num_cams_local):
                proj = image_coordinates(xyz, cals[cam], mm_params)
                diff = frame_targets[cam] - proj
                mask = np.isfinite(diff).all(axis=1)
                if np.any(mask):
                    sums[cam] += float(np.sum(diff[mask] ** 2))
                    counts[cam] += int(np.sum(mask))

        rms = np.sqrt(sums / np.maximum(counts, 1))
        print(f"{label} per-camera RMS (metric): {rms.tolist()}")

    print(f"Using {num_frames} frame(s) for dumbbell calibration")
    per_frame_metric = metric_by_cam

    # Generate initial guess vector and bounds for optimization:
    if 1 <= fixed_cam_param <= num_cams:
        fixed_cam = fixed_cam_param - 1
        print(
            f"Fixing camera {fixed_cam + 1} from parameter dumbbell_fixed_camera; "
            f"coverage counts: {coverage.tolist()}"
        )
    else:
        fixed_cam = int(np.argmax(coverage)) if num_cams > 0 else 0
        print(
            f"Fixing camera {fixed_cam + 1} based on coverage counts: {coverage.tolist()}"
        )

    active = np.ones(num_cams)
    active[fixed_cam] = 0
    num_active = int(np.sum(active))
    if num_active == 0:
        raise ValueError("All cameras fixed; need at least one active camera")

    pos_scale = 1.0
    print(f"Position scale set to {pos_scale} (optimize in millimeters)")
    calib_vec = np.empty((num_active, 2, 3))
    active_ptr = 0
    for cam in range(num_cams):
        if active[cam]:
            calib_vec[active_ptr, 0] = cals[cam].get_pos() / pos_scale
            calib_vec[active_ptr, 1] = cals[cam].get_angles()
            active_ptr += 1

    calib_vec = calib_vec.flatten()

    def _init_dumbbell_points(metric_targets: np.ndarray) -> np.ndarray:
        from openptv2.orientation import multi_cam_point_positions

        num_cams_local, num_frames_local, _, _ = metric_targets.shape
        points = np.zeros((num_frames_local, 2, 3), dtype=float)

        for frame_idx in range(num_frames_local):
            frame_targets = metric_targets[:, frame_idx, :, :]
            xyz1, _err1 = multi_cam_point_positions(
                frame_targets[:, 0, :][np.newaxis], cpar, cals
            )
            xyz2, _err2 = multi_cam_point_positions(
                frame_targets[:, 1, :][np.newaxis], cpar, cals
            )
            points[frame_idx, 0] = xyz1[0]
            points[frame_idx, 1] = xyz2[0]

        return points

    points_init = _init_dumbbell_points(per_frame_metric)
    x0 = np.concatenate([calib_vec, points_init.reshape(-1)])

    # Test optimizer-ready target function:
    print("Initial values (1 row per active camera, scaled pos, then angle):")
    print(calib_vec.reshape(num_active, -1))
    print("Current target function (sum of squared residuals):", end=" ")
    init_residuals = dumbbell_ba_residuals(
        x0, per_frame_metric, cpar, cals, active, db_length, db_weight, pos_scale
    )
    print(np.sum(init_residuals**2))
    _print_camera_residuals("Initial", per_frame_metric)

    # Optimization:
    method = "trf"
    loss = "soft_l1"
    print(f"Using least_squares method={method} loss={loss}")
    tol_steps = [
        (1e-6, 1e-6, 1e-5),
        (1e-5, 1e-5, 1e-4),
        (1e-4, 1e-4, 1e-3),
    ]
    max_rounds = 3
    nfev_per_round = 200
    min_improvement = 1e-3

    best_x = x0
    best_fun = float(np.sum(init_residuals**2))
    res = None

    jac_sparsity = dumbbell_ba_jac_sparsity(per_frame_metric, active, db_weight)

    for idx in range(max_rounds):
        xtol, ftol, gtol = tol_steps[min(idx, len(tol_steps) - 1)]
        res = least_squares(
            dumbbell_ba_residuals,
            best_x,
            args=(
                per_frame_metric,
                cpar,
                cals,
                active,
                db_length,
                db_weight,
                pos_scale,
            ),
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
            jac_sparsity=jac_sparsity,
            x_scale="jac",
            max_nfev=nfev_per_round,
            loss=loss,
            f_scale=1.0,
            verbose=2,
            method=method,
        )
        new_fun = float(np.sum(res.fun**2))
        improvement = (best_fun - new_fun) / max(best_fun, 1e-12)
        print(
            f"Adaptive round {idx + 1}: fun={new_fun:.6g} improvement={improvement:.3g} "
            f"xtol={xtol} ftol={ftol} gtol={gtol}"
        )
        best_x = res.x
        best_fun = new_fun
        if improvement < min_improvement:
            break

    if res is None:
        raise RuntimeError("Adaptive least_squares did not run")

    print("Result of dumbbell calibration")
    cam_params_len = num_active * 6
    print(best_x[:cam_params_len].reshape(num_active, -1))
    print("Success:", res.success, res.message)
    print("Final target function (sum of squared residuals):", end=" ")
    final_residuals = dumbbell_ba_residuals(
        best_x, per_frame_metric, cpar, cals, active, db_length, db_weight, pos_scale
    )
    print(np.sum(final_residuals**2))

    # convert calib_vec back to Calibration objects:
    cam_params_len = num_active * 6
    calib_pars = best_x[:cam_params_len].reshape(-1, 2, 3)

    for cam, cal in enumerate(cals):
        if not active[cam]:
            continue

        # Pop a parameters line:
        pars = calib_pars[0]
        calib_pars = calib_pars[1:]

        cal.set_pos(pars[0] * pos_scale)
        cal.set_angles(pars[1])

        _print_camera_residuals("Final", per_frame_metric)

        # Write the calibration results to files:
        ori_filename = cpar.get_cal_img_base_name(cam)
        addpar_filename = ori_filename + ".addpar"
        ori_filename = ori_filename + ".ori"
        cal.write(ori_filename.encode("utf-8"), addpar_filename.encode("utf-8"))


def calib_particles(exp):
    """Calibration with particles."""
    from openptv2.tracking_framebuf import Frame

    # Handle both Experiment objects and MainGUI objects
    if hasattr(exp, "pm"):
        # Traditional experiment object
        pm = exp.pm
        num_cams = pm.num_cams
        cpar = exp.cpar
        spar = exp.spar
        vpar = exp.vpar
        tpar = exp.tpar
        cals = exp.cals
    elif hasattr(exp, "exp1") and hasattr(exp.exp1, "pm"):
        # MainGUI object - ensure parameter objects are initialized
        pm = exp.exp1.pm
        num_cams = exp.num_cams
        cpar = exp.cpar
        spar = exp.spar
        vpar = exp.vpar
        tpar = exp.tpar
        cals = exp.cals
    else:
        raise ValueError("Object must have either pm or exp1.pm attribute")

    num_cams = cpar.get_num_cams()
    calibs = _read_calibrations(cpar, num_cams)

    targ_files = [
        spar.get_img_base_name(c).split("%d")[0].encode("utf-8")
        for c in range(num_cams)
    ]

    orient_params = pm.get_parameter("orient")
    shaking_params = pm.get_parameter("shaking")

    flags = [name for name in NAMES if orient_params.get(name) == 1]
    all_known = []
    all_detected = [[] for c in range(num_cams)]

    for frm_num in range(
        shaking_params["shaking_first_frame"], shaking_params["shaking_last_frame"] + 1
    ):
        frame = Frame(
            cpar.get_num_cams(),
            corres_file_base=("res/rt_is").encode("utf-8"),
            linkage_file_base=("res/ptv_is").encode("utf-8"),
            target_file_base=targ_files,
            frame_num=frm_num,
        )

        all_known.append(frame.positions())
        for cam in range(num_cams):
            all_detected[cam].append(frame.target_positions_for_camera(cam))

    all_known = np.vstack(all_known)

    targ_ix_all = []
    residuals_all = []
    targs_all = []
    for cam in range(num_cams):
        detects = np.vstack(all_detected[cam])
        assert detects.shape[0] == all_known.shape[0]

        have_targets = ~np.isnan(detects[:, 0])
        used_detects = detects[have_targets, :]
        used_known = all_known[have_targets, :]

        targs = TargetArray(len(used_detects))

        for tix, detect in enumerate(used_detects):
            targ = targs[tix]
            targ.set_pnr(tix)
            targ.set_pos(detect)

        residuals = full_scipy_calibration(
            calibs[cam], used_known, targs, exp.cpar, flags=flags
        )
        print(f"After scipy full calibration, {np.sum(residuals**2)}")

        print(f"Camera {cam + 1}")
        print((calibs[cam].get_pos()))
        print((calibs[cam].get_angles()))

        ori_filename = exp.cpar.get_cal_img_base_name(cam)
        addpar_filename = ori_filename + ".addpar"
        ori_filename = ori_filename + ".ori"
        calibs[cam].write(ori_filename.encode("utf-8"), addpar_filename.encode("utf-8"))

        targ_ix = [t.pnr() for t in targs if t.pnr() != -999]

        targs_all.append(targs)
        targ_ix_all.append(targ_ix)
        residuals_all.append(residuals)

    print("End calibration with particles")
    return targs_all, targ_ix_all, residuals_all


def clone_calibration(calibration_obj):
    """Return a copy of a Calibration object using all get/set methods."""
    new_cal = Calibration()
    new_cal.set_pos(np.array(calibration_obj.get_pos()))
    new_cal.set_angles(np.array(calibration_obj.get_angles()))
    new_cal.set_primary_point(np.array(calibration_obj.get_primary_point()))
    new_cal.set_radial_distortion(np.array(calibration_obj.get_radial_distortion()))
    new_cal.set_decentering(np.array(calibration_obj.get_decentering()))
    new_cal.set_affine_trans(np.array(calibration_obj.get_affine()))
    new_cal.set_glass_vec(np.array(calibration_obj.get_glass_vec()))
    return new_cal

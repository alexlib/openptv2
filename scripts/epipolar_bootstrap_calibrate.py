"""Epipolar bootstrapping for poorly-calibrated cameras.

Uses well-calibrated cameras (cam3/cam4) to triangulate calblock dots in 3D,
then projects into poorly-calibrated cameras (cam1/cam2) and refines via
epipolar search along curves from cam3/cam4, finally re-calibrating with
dozens of seed points instead of 4.

Usage:
    uv run python scripts/epipolar_bootstrap_calibrate.py <dataset_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import (
    external_calibration,
    read_calblock,
)
from openptv2.algorithms.tracking_frame_buf import read_targets
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.trafo import metric_to_pixel, pixel_to_metric
from openptv2.autocalibration import _load_dataset_params, resolve_calblock, cam_files
from openptv2.epipolar import epipolar_curve


def load_cals(base: Path, num_cams: int) -> list[Calibration]:
    """Load calibrations for all cameras."""
    return [
        Calibration.from_file(*[str(p) for p in cam_files(base, c)[1:]])
        for c in range(num_cams)
    ]


def epipolar_match_cam34(
    c3_pix: np.ndarray,
    c4_pix: np.ndarray,
    cal3: Calibration,
    cal4: Calibration,
    cpar,
    vpar,
    tol_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match dots between cam3 and cam4 via epipolar search.

    For each detection in cam3, project the epipolar curve into cam4 and
    find the nearest cam4 detection within ``tol_px``.

    Returns:
        (matched_c3_pix, matched_c4_pix) arrays of shape (n_matches, 2).
    """
    c3_matched: list[np.ndarray] = []
    c4_matched: list[np.ndarray] = []

    c4_points = c4_pix  # (n4, 2) pixel coords

    for p in c3_pix:
        # Project epipolar curve from cam3 into cam4
        curve = epipolar_curve(tuple(p), cal3, cal4, num_points=50, cpar=cpar, vpar=vpar)

        # Find nearest cam4 detection to any point on the curve
        best_dist = float("inf")
        best_j = -1
        for j, q in enumerate(c4_points):
            d = np.sqrt(((curve[:, 0] - q[0]) ** 2 + (curve[:, 1] - q[1]) ** 2)).min()
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_dist <= tol_px and best_j >= 0:
            c3_matched.append(p)
            c4_matched.append(c4_points[best_j])

    if not c3_matched:
        return np.empty((0, 2)), np.empty((0, 2))

    return np.asarray(c3_matched), np.asarray(c4_matched)


def triangulate_matches(
    c3_pix: np.ndarray,
    c4_pix: np.ndarray,
    cal3: Calibration,
    cal4: Calibration,
    cpar,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate matched cam3/cam4 dot pairs into 3D world coordinates.

    Returns:
        (points_3d, rcm) arrays.
    """
    n = len(c3_pix)
    targets = np.empty((n, 2, 2), dtype=np.float64)

    for i in range(n):
        xm3, ym3 = pixel_to_metric(c3_pix[i, 0], c3_pix[i, 1], cpar)
        xm4, ym4 = pixel_to_metric(c4_pix[i, 0], c4_pix[i, 1], cpar)
        targets[i, 0, 0] = xm3
        targets[i, 0, 1] = ym3
        targets[i, 1, 0] = xm4
        targets[i, 1, 1] = ym4

    from openptv2.orientation import multi_cam_point_positions

    cals = [cal3, cal4]
    positions, rcm = multi_cam_point_positions(targets, cpar, cals, method="pairs")
    return positions, rcm


def refine_positions_via_epipolar(
    points_3d: np.ndarray,
    c3_match_pix: np.ndarray,
    origin_cal: Calibration,   # cam3 calibration
    target_cal: Calibration,   # cam1 or cam2 calibration (to be refined)
    cpar,
    vpar,
    search_tol_px: float = 15.0,
) -> np.ndarray:
    """Refine 2D positions in target_cam by searching along epipolar curves.

    For each 3D point, we know its detection in cam3 (c3_match_pix).
    Project the epipolar curve from cam3 into target_cam (cam1/cam2).
    The actual dot in target_cam must lie on this curve.
    Also project the 3D point into target_cam using current (bad) calibration
    to get a rough position, then find the point on the epipolar curve nearest
    to that rough position.

    Returns:
        Refined pixel coordinates in target_cam (n, 2).
    """
    n = len(points_3d)
    refined = np.empty((n, 2), dtype=np.float64)

    # Rough projection using current (bad) calibration
    rough_proj = np.empty((n, 2), dtype=np.float64)
    for i, p in enumerate(points_3d):
        xm, ym = img_coord(p, target_cal, cpar.mm)
        px, py = metric_to_pixel(xm, ym, cpar)
        rough_proj[i] = [px, py]

    for i in range(n):
        # Epipolar curve from cam3 detection into target cam
        curve = epipolar_curve(
            tuple(c3_match_pix[i]), origin_cal, target_cal, num_points=100,
            cpar=cpar, vpar=vpar,
        )  # (100, 2) pixel coords in target cam

        rough = rough_proj[i]

        # Find point on curve nearest to rough projection
        d = np.sqrt(((curve[:, 0] - rough[0]) ** 2 + (curve[:, 1] - rough[1]) ** 2))
        best_idx = int(d.argmin())
        best_dist = float(d[best_idx])

        if best_dist <= search_tol_px:
            refined[i] = curve[best_idx]
        else:
            # Curve doesn't pass near rough projection; fall back to rough
            refined[i] = rough

    return refined


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dataset_dir>")
        sys.exit(1)

    base = Path(sys.argv[1]).resolve()
    if not base.exists():
        print(f"Dataset dir not found: {base}")
        sys.exit(1)

    ds = _load_dataset_params(base, resolve_calblock(base))
    cpar = ds.cpar
    num_cams = ds.num_cams
    fix_all, num_fix = read_calblock(str(resolve_calblock(base)))
    fix_all = np.asarray(fix_all, float)

    print(f"Dataset: {base}")
    print(f"  num_cams: {num_cams}, calblock points: {num_fix}")

    # Load calibrations
    cals = load_cals(base, num_cams)

    # Load targets for cam3 and cam4 (well calibrated)
    c3_pix = np.array([[t.x, t.y] for t in read_targets(str(base / "cal" / "cam3.tif"), 0)])
    c4_pix = np.array([[t.x, t.y] for t in read_targets(str(base / "cal" / "cam4.tif"), 0)])
    print(f"  cam3 detections: {len(c3_pix)}, cam4 detections: {len(c4_pix)}")

    # Volume parameters for epipolar curve Z-range
    from openptv2.algorithms.parameters import VolumePar

    vpar = VolumePar(
        eps0=0.05,
        X_lay=[-40.0, 40.0],
        Zmin_lay=[-40.0, -40.0],
        Zmax_lay=[40.0, 40.0],
        cn=0.02,
        cnx=0.02,
        cny=0.02,
        csumg=0.02,
        corrmin=33.0,
    )

    # Step 1: Match cam3↔cam4 dots via epipolar search
    print("\n=== Step 1: Epipolar match cam3↔cam4 ===")
    c3_match, c4_match = epipolar_match_cam34(
        c3_pix, c4_pix, cals[2], cals[3], cpar, vpar, tol_px=3.0
    )
    print(f"  Matched pairs: {len(c3_match)}")

    if len(c3_match) < 10:
        print("  ERROR: Too few epipolar matches between cam3 and cam4")
        sys.exit(1)

    # Step 2: Triangulate to 3D
    print("\n=== Step 2: Triangulate matched pairs to 3D ===")
    points_3d, rcm = triangulate_matches(c3_match, c4_match, cals[2], cals[3], cpar)
    good = rcm < 5.0
    points_3d = points_3d[good]
    c3_match = c3_match[good]
    c4_match = c4_match[good]
    print(f"  Good triangulations (rcm<5): {len(points_3d)}")

    # Step 3-4: For cam1/cam2, refine positions via epipolar search then recalibrate
    print("\n=== Step 3-4: Epipolar refine + recalibrate cam1/cam2 ===")
    for cam_idx in [0, 1]:  # cam1, cam2
        cal_cam = cals[cam_idx]

        # Refine 2D positions via epipolar search from cam3
        refined_px = refine_positions_via_epipolar(
            points_3d, c3_match, cals[2], cal_cam, cpar, vpar, search_tol_px=15.0
        )

        # Check how many refined positions are close to actual detections
        pix_det = np.array(
            [[t.x, t.y] for t in read_targets(str(base / "cal" / f"cam{cam_idx + 1}.tif"), 0)]
        )
        dists = np.sqrt(((refined_px[:, None, :] - pix_det[None, :, :]) ** 2).sum(axis=2))
        nearest = dists.min(axis=1)
        n_close = (nearest < 3.0).sum()
        print(
            f"  cam{cam_idx + 1}: {n_close}/{len(points_3d)} refined positions within 3px of a detection "
            f"(mean err={nearest.mean():.2f}px)"
        )

        # Filter to only good correspondences (< 3px from a detection)
        good = nearest < 3.0
        if good.sum() < 10:
            print(f"  ERROR: Too few good correspondences ({good.sum()}) for cam{cam_idx + 1}")
            continue

        ref_pts = points_3d[good]
        img_pts = refined_px[good]

        # Step 4: Re-calibrate with bootstrapped correspondences
        print(f"  Re-calibrating cam{cam_idx + 1} with {len(ref_pts)} seed points...")
        new_cal = Calibration.from_file(
            str(cam_files(base, cam_idx)[1]),
            str(cam_files(base, cam_idx)[2]) if Path(cam_files(base, cam_idx)[2]).exists() else "",
        )
        ok = external_calibration(new_cal, ref_pts, img_pts, cpar)
        if not ok:
            print(f"  WARNING: external_calibration did not converge for cam{cam_idx + 1}")

        # Check reprojection with new calibration
        proj_px_new = np.empty((len(points_3d), 2))
        for i, p in enumerate(points_3d):
            xm, ym = img_coord(p, new_cal, cpar.mm)
            px, py = metric_to_pixel(xm, ym, cpar)
            proj_px_new[i] = [px, py]
        dists_new = np.sqrt(((proj_px_new[:, None, :] - pix_det[None, :, :]) ** 2).sum(axis=2))
        nearest_new = dists_new.min(axis=1)
        n_close_new = (nearest_new < 3.0).sum()
        print(
            f"  cam{cam_idx + 1} AFTER recalibration: {n_close_new}/{len(points_3d)} within 3px "
            f"(mean err={nearest_new.mean():.2f}px)"
        )

        # Write updated calibration
        ori_path, add_path = cam_files(base, cam_idx)[1], cam_files(base, cam_idx)[2]
        new_cal.to_file(str(ori_path), str(add_path) if Path(add_path).exists() else "")
        print(f"  Wrote {ori_path.name}")


if __name__ == "__main__":
    main()

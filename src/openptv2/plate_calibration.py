"""Multi-plane plate → OpenCV intrinsics/extrinsics → solved Z_per_plane.

Reproduces the loop previewed in
``file:///C:/Users/alex/Dropbox/3DPTV_Illmenau/Multiview-Calibration/manual_openptv_orientation_from_opencv_pipeline.html``
(``multiview_calibration.py:87-118``): flat-Z0 ``calibrateCamera`` per cam →
``stereoCalibrate`` plane 0 → 4-cam DLT tri → recalibrate, before the hub's
algebraic OpenCV→``.ori`` tail.  Kept ``cv2``-free in spirit — the ``cv2``
branch is only imported when ``method='opencv'`` is explicitly requested.

A pure-Python ``method='dlt'`` via :func:`openptv2.calibration_seed.seed_from_dlt`
is also available when the caller already holds labelled ``(X,Y,Z)↔(x,y)`` with
known ``Z`` per plane (or the Zs will be triangulated iteratively the same way
the Multiview loop does).
"""

from __future__ import annotations

import numpy as np


def solve_opencv_multiview(
    views_per_cam: list[list[np.ndarray]],
    refs_per_plane: list[np.ndarray],
    image_size: tuple[int, int],
    *,
    flags: int | None = None,
):
    """Multiview solver calling ``cv2.calibrateCamera`` when available.

    Parameters
    ----------
    views_per_cam:
        ``views_per_cam[cam][plane]`` is ``(n,2)`` image points, already
        labelled and ordered matching ``refs_per_plane[plane]`` with
        ``Z=0``.  ``plane`` index is the hand-held position.
    refs_per_plane:
        ``refs_per_plane[plane]`` is ``(n,3)`` world points with ``Z=0`` for
        plane ``plane``.  Hand-held ``Z`` is solved iteratively.
    image_size:
        ``(imx, imy)``.

    Returns
    -------
    ``(K_list, dist_list, rvec_list, tvec_list, P_planes, Zs)``
    where ``P_planes[plane]`` is the triangulated ``(n,3)`` world points with
    solved ``Z`` and ``Zs`` is the per-plane ``Z`` offset (``Zs[0]==0``).
    """
    try:
        import cv2 as _cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "solve_opencv_multiview needs OpenCV (pip install opencv-python or "
            "uv add --optional opencv).  Use method='dlt' for the pure-Python "
            "path."
        ) from exc

    num_cams = len(views_per_cam)
    num_planes = len(refs_per_plane)
    if num_planes == 0:
        raise ValueError("need ≥1 plane")

    # Phase 1: per-cam flat-Z0 calibration (hub: straight-ray on plane 0 only in air)
    # Build XYZ with Z=0 for every plane (unknown Zs, initial guess)
    imx, imy = image_size
    XYZ0 = [np.asarray(r, dtype=np.float32) for r in refs_per_plane]
    Ms, dists, rvecs, tvecs = [], [], [], []
    for cam in range(num_cams):
        img_pts = [
            np.asarray(views_per_cam[cam][p], dtype=np.float32)
            for p in range(num_planes)
        ]
        ret, M, dist, rvec, tvec = _cv2.calibrateCamera(
            XYZ0, img_pts, (imx, imy), None, None
        )
        Ms.append(M)
        dists.append(dist)
        rvecs.append(rvec)
        tvecs.append(tvec)

    # Build projection matrices relative to cam0 via stereoCalibrate on plane 0
    # (matches multiview_calibration.py P_c construction)
    from scipy.spatial.transform import Rotation as _R

    Rc = []
    posc = []
    Pc = []
    for cam in range(num_cams):
        R0 = _R.from_rotvec(rvecs[cam][0].ravel()).as_matrix()
        Rc.append(R0)
        t0 = tvecs[cam][0].ravel()
        posc.append(-R0.T @ t0)
        if cam == 0:
            RT = np.concatenate([R0, t0[:, None]], axis=1)
            Pc.append(Ms[0] @ RT)
        else:
            ret, CM0, d0, CM1, d1, R, T, *_ = _cv2.stereoCalibrate(
                XYZ0[:1],
                [views_per_cam[0][0].astype(np.float32)],
                [views_per_cam[cam][0].astype(np.float32)],
                Ms[0],
                dists[0],
                Ms[cam],
                dists[cam],
                (imx, imy),
                flags=_cv2.CALIB_FIX_INTRINSIC,
            )
            RT = np.concatenate([R @ Rc[0], (R @ tvecs[0][0] + T)], axis=1)
            Pc.append(Ms[cam] @ RT)

    # DLT tri per plane (4-cam) to solve Zs, then recalibrate
    def _dlt(P_list, xys):
        # 8×4 A as in multiview DLT
        A = []
        for P, xy in zip(P_list, xys):
            x, y = float(xy[0]), float(xy[1])
            A.append(y * P[2, :] - P[1, :])
            A.append(P[0, :] - x * P[2, :])
        A = np.vstack(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]

    P_planes = []
    for p in range(num_planes):
        # triangulate each marker across available cameras (≥2).  The previous
        # code padded to 4 by duplicating the last view, which makes A
        # rank-deficient (duplicate rows) and can bias/unstabilize the SVD —
        # use only the available views.
        n = refs_per_plane[p].shape[0]
        plane_3d = []
        for k in range(n):
            xys = [views_per_cam[c][p][k] for c in range(min(4, num_cams))]
            Ps = Pc[: len(xys)]
            plane_3d.append(_dlt(Ps, xys))
        P_planes.append(np.asarray(plane_3d, dtype=np.float32))
    # Keep plane 0 at Z=0 as reference (hub: P[0]=XYZ0)
    P_planes[0] = XYZ0[0]

    # Recalibrate with solved P
    for cam in range(num_cams):
        ret, M, dist, rvec, tvec = _cv2.calibrateCamera(
            P_planes,
            [
                np.asarray(views_per_cam[cam][p], dtype=np.float32)
                for p in range(num_planes)
            ],
            (imx, imy),
            Ms[cam],
            dists[cam],
            flags=_cv2.CALIB_USE_INTRINSIC_GUESS,
        )
        Ms[cam], dists[cam], rvecs[cam], tvecs[cam] = M, dist, rvec, tvec

    Zs = np.array([float(p[:, 2].mean()) for p in P_planes])
    return Ms, dists, rvecs, tvecs, P_planes, Zs


def solve_dlt_per_cam(
    ref_pts: np.ndarray,
    img_pts: np.ndarray,
    cpar,
):
    """Single-cam DLT helper — thin wrapper over :func:`seed_from_dlt`.

    Useful when the hand-held ``Z`` per plane is already known or triangulated
    elsewhere and you simply want a pose+``cc``.
    """
    from openptv2.calibration_seed import seed_from_dlt

    cal = seed_from_dlt(ref_pts, img_pts, cpar)
    return cal

"""Bootstrap openptv2 .ori/.addpar calibration for proPTV's camera rig from
known 3D<->2D correspondences (no calibration-target photos exist for
proPTV, see docs/plans/2026-08-17-lagrangian-accuracy-program.md Phase 2's
trackcorr scoping note).

Two-stage approach: openptv2's own exterior solver (`external_calibration`
== `raw_orient`) is an iterative Gauss-Newton refinement that needs a decent
starting guess -- openptv2 ships no from-scratch resection. Classic DLT
camera resection (Abdel-Aziz & Karara 1971) supplies that starting guess
from the same correspondences (`origin_*.txt`'s known 3D position + observed
per-camera pixel projection), no initial guess needed. Then openptv2's own
`external_calibration` + `full_calibration` refine it with its own pinhole
model, matching the plan's "use openptv2's own machinery" decision.

Projection convention (verified empirically, not assumed -- see the plan's
next-steps item 1): metric = -cc * (R @ (X_world - C))_{x,y} / (R @ (X_world
- C))_z, where R is `Calibration.ext_par.dm` (world-to-camera rotation) and
C = (x0,y0,z0) the camera center. So the DLT projection matrix decomposes as
P = diag(-cc,-cc,1) @ [R | -R@C], not the textbook diag(f,f,1) form -- the
sign is folded into cc's sign, handled below by picking the decomposition
with positive depth (Z_cam > 0) for all points, which pins the sign
unambiguously.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import rq

sys.path.insert(0, "src")


def dlt_resection(ref_pts: np.ndarray, metric_pts: np.ndarray):
    """Classic DLT camera resection.

    ref_pts: (N,3) known 3D world points.
    metric_pts: (N,2) observed metric image coordinates (already centered
        via pixel_to_metric -- NOT raw pixel coordinates).

    Returns (R, C, cc): world-to-camera rotation (3,3), camera center (3,),
    camera constant (signed).
    """
    n = ref_pts.shape[0]
    assert n >= 6, "DLT resection needs >= 6 correspondences"

    A = np.zeros((2 * n, 12))
    X = ref_pts
    u = metric_pts[:, 0]
    v = metric_pts[:, 1]
    for i in range(n):
        x, y, z = X[i]
        A[2 * i] = [x, y, z, 1, 0, 0, 0, 0, -u[i] * x, -u[i] * y, -u[i] * z, -u[i]]
        A[2 * i + 1] = [0, 0, 0, 0, x, y, z, 1, -v[i] * x, -v[i] * y, -v[i] * z, -v[i]]

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # DLT's P is defined up to an arbitrary overall scale/sign (the SVD null
    # vector has no preferred sign) -- pin it by requiring positive depth
    # (points in front of the camera) for the majority of points. Without
    # this, RQ's per-diagonal-entry sign fix below can silently accept a
    # decomposition where every point is "behind" the camera; row/column
    # sign flips alone can't fix that, only flipping all of P can.
    depth_sign = P[2, :3] @ X.T + P[2, 3]
    if np.mean(depth_sign > 0) < 0.5:
        P = -P

    M = P[:, :3]
    # RQ decomposition: M = K @ R, K upper-triangular, R orthogonal.
    K, R = rq(M)

    # Fix RQ's sign ambiguity: force K's diagonal to (-|cc|, -|cc|, +1) to
    # match this codebase's metric = -cc*Xcam/Zcam convention, flipping the
    # corresponding rows of R (and K's columns) to compensate so K@R == M
    # still holds up to the DLT's own free overall scale/sign.
    signs = np.sign(np.diag(K))
    target = np.array([-1.0, -1.0, 1.0])
    fix = signs * target  # what to multiply each diagonal entry by to reach target sign
    S = np.diag(fix)
    K = K @ S
    R = S @ R

    # Recover overall scale (DLT's P is only defined up to scale) from K[2,2].
    scale = K[2, 2]
    K = K / scale
    # t (last column of M^-1 @ P[:,3]) scales the same way; recover camera
    # center C from world = -R^T @ (M^-1 @ p4).
    # Simpler: solve M @ C = -p4 for camera center directly (unscaled M, p4).
    C = -np.linalg.solve(P[:, :3], P[:, 3])

    cc = K[0, 0]  # already negative per the target sign convention above

    # Ensure a proper rotation (det=+1); DLT/RQ can return a reflection.
    if np.linalg.det(R) < 0:
        R = -R
        cc = -cc

    # Sign bookkeeping through RQ + two separate flip corrections above is
    # fragile in practice (verified failing on real, non-synthetic data even
    # after the checks above) -- R and C always come out numerically correct
    # regardless, and the model's dependence on cc is a single isolated
    # multiplicative sign (metric = -cc * num/den, R/C unaffected by cc's
    # sign), so settle it by direct numeric check against the actual input
    # data rather than more sign algebra.
    cam = (R @ (ref_pts - C).T).T
    proj = -cc * cam[:, :2] / cam[:, 2:3]
    if np.mean(np.sum((proj - metric_pts) ** 2, axis=1)) > np.mean(
        np.sum((-proj - metric_pts) ** 2, axis=1)
    ):
        cc = -cc

    return R, C, cc


def rotation_matrix_to_angles(dm: np.ndarray) -> tuple[float, float, float]:
    """Invert Calibration.Exterior.compute_rotation_matrix()'s omega/phi/kappa
    convention (see src/openptv2/algorithms/calibration.py).

    Canonical implementation now lives in :mod:`openptv2.calibration_seed`
    (``angles_from_dm``) — this shim is kept so the script remains standalone.
    """
    from openptv2.calibration_seed import angles_from_dm as _afd

    return _afd(dm)


def _self_test():
    """Verify dlt_resection against a KNOWN synthetic camera before trusting
    it on real proPTV data -- exactly the camera used to empirically pin
    down the projection sign convention above."""
    from openptv2.algorithms.calibration import (
        AddedPar,
        Calibration,
        Exterior,
        Glass,
        Interior,
    )
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.parameters import ControlPar, MmNp

    cpar = ControlPar(num_cams=1, imx=800, imy=800, pix_x=1.0, pix_y=1.0, mm=MmNp())
    ext = Exterior(x0=0.3, y0=-0.2, z0=-5.0, omega=0.1, phi=-0.15, kappa=0.2)
    ext.compute_rotation_matrix()
    intp = Interior(xh=0.0, yh=0.0, cc=6.0)
    cal = Calibration(
        ext_par=ext, int_par=intp, glass_par=Glass(), added_par=AddedPar()
    )

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 2, size=(40, 3))
    metric = np.array([img_coord(p, cal, cpar.mm) for p in pts])

    R, C, cc = dlt_resection(pts, metric)
    print("true  C:", ext.x0, ext.y0, ext.z0, "cc:", intp.cc)
    print("dlt   C:", C, "cc:", cc)

    # img_coord consumes ext_par.dm TRANSPOSED as the world-to-camera
    # rotation (verified empirically -- see module docstring), so the
    # dataclass's "dm" field itself is camera-to-world: dm = R.T.
    omega, phi, kappa = rotation_matrix_to_angles(R.T)
    print(
        "recovered angles:",
        omega,
        phi,
        kappa,
        "vs true:",
        ext.omega,
        ext.phi,
        ext.kappa,
    )


def load_camera_correspondences(
    origin_path: Path, cam: int
) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) world XYZ and (N,2) raw pixel xc,yc for one camera from a
    proPTV origin_*.txt frame, dropping rows where that camera didn't see
    the particle (xc/yc is NaN)."""
    xyz, pix = [], []
    for line in origin_path.read_text().strip().splitlines():
        if line.startswith("#"):
            continue
        parts = [float(p) for p in line.split()]
        # columns: ID,X,Y,Z,U,V,W,T,P,xc0,yc0,xc1,yc1,xc2,yc2,xc3,yc3
        x, y, z = parts[1], parts[2], parts[3]
        xc, yc = parts[9 + 2 * cam], parts[10 + 2 * cam]
        if np.isnan(xc) or np.isnan(yc):
            continue
        xyz.append((x, y, z))
        pix.append((xc, yc))
    return np.array(xyz), np.array(pix)


def calibrate_camera_from_scratch(cam: int, origin_path: Path, cpar):
    """DLT bootstrap -> openptv2 external_calibration -> full_calibration.
    Returns (Calibration, rms_px, n_points) or raises on failure."""
    from openptv2.algorithms.calibration import (
        AddedPar,
        Calibration,
        Exterior,
        Glass,
        Interior,
    )
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.trafo import pixel_to_metric_batch
    from openptv2.orientation import external_calibration, full_calibration

    xyz, pix = load_camera_correspondences(origin_path, cam)
    if len(xyz) < 20:
        raise RuntimeError(f"cam{cam}: only {len(xyz)} visible correspondences")

    metric = pixel_to_metric_batch(pix, cpar)
    R, C, cc = dlt_resection(xyz, metric)
    omega, phi, kappa = rotation_matrix_to_angles(R.T)

    ext = Exterior(x0=C[0], y0=C[1], z0=C[2], omega=omega, phi=phi, kappa=kappa)
    ext.compute_rotation_matrix()
    intp = Interior(xh=0.0, yh=0.0, cc=cc)
    cal = Calibration(
        ext_par=ext, int_par=intp, glass_par=Glass(), added_par=AddedPar()
    )

    ok = external_calibration(cal, xyz, pix, cpar)
    if not ok:
        raise RuntimeError(
            f"cam{cam}: external_calibration (raw_orient) did not converge"
        )

    from openptv2.algorithms.tracking_frame_buf import Target

    targets = [Target(pnr=i, x=pix[i, 0], y=pix[i, 1]) for i in range(len(pix))]
    residuals, used, _err = full_calibration(
        cal, xyz, targets, cpar, flags=["cc", "xh", "yh", "k1", "k2", "p1", "p2"]
    )

    reproj = np.array([img_coord(xyz[i], cal, cpar.mm) for i in used])
    obs_metric = metric[used]
    rms = float(np.sqrt(np.mean(np.sum((reproj - obs_metric) ** 2, axis=1))))
    return cal, rms, len(used)


def main():
    from openptv2.algorithms.parameters import ControlPar, MmNp

    proptv_dir = Path(r"C:/Users/alex/Github/proPTV/data/500_30")
    origin_path = proptv_dir / "origin" / "origin_00000.txt"
    out_dir = Path("test_data/proptv_500_30/cal")
    out_dir.mkdir(parents=True, exist_ok=True)

    cpar = ControlPar(num_cams=1, imx=800, imy=800, pix_x=1.0, pix_y=1.0, mm=MmNp())

    for cam in range(4):
        try:
            cal, rms, n = calibrate_camera_from_scratch(cam, origin_path, cpar)
        except Exception as e:
            print(f"cam{cam}: FAILED -- {e}")
            continue
        # openptv2's naming convention: 1-indexed, "camN.tif.ori"/".addpar"
        # (matches what adapt_proptv_dataset.py's cloned scaffold expects).
        ori = out_dir / f"cam{cam + 1}.tif.ori"
        addpar = out_dir / f"cam{cam + 1}.tif.addpar"
        cal.write(str(ori), str(addpar))
        print(f"cam{cam}: rms={rms:.4f} px  n={n}  -> {ori}")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "selftest":
        _self_test()
    else:
        main()

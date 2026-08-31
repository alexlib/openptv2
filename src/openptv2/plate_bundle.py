"""Joint bundle adjustment over many hand-held plate positions.

Solving every camera's pose on a single reference frame makes the model exact on
that plane by construction and lets its error grow linearly away from it.  On
the Illmenau 4-camera rig that cost 0.58 % of the plate's distance from the
anchor plane in ray-convergence miss -- sub-millimetre at the plane, ~18 mm at
3-4 m -- while per-camera reprojection RMS sat at a healthy 0.5 px throughout.
This module removes the anchoring: the unknowns become every camera pose *and*
one rigid plate pose per frame, solved together.

What is deliberately NOT fitted here:

``cc`` and the distortion
    Focal length is exactly degenerate on a single plane and only weakly
    determined even over many, so it is an input -- fit it separately (see
    ``docs/illmenau-4cam-calibration.md``) and pass the value you verified.
    Distortion fitted from few planes trades against pose and produces a
    polynomial that diverges outside the fitted points.

the reference frame's plate pose
    Held at identity.  This is the gauge: the world stays pinned to the physical
    dot that defines it, so an existing calibration block, datum record, or
    manual check of that frame all remain valid.  With the gauge fixed there is
    no free similarity, so scale cannot drift even though ``cc`` is not fitted.

The module is deliberately free of any OpenCV dependency -- it takes initial
poses from the caller (who may well have used ``cv2.solvePnP`` to get them) and
uses only numpy and scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector -> 3x3 rotation matrix (``cv2.Rodrigues`` without cv2)."""
    r = np.asarray(rvec, float).ravel()
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector."""
    R = np.asarray(R, float)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-12:
        return np.zeros(3)
    if theta > np.pi - 1e-6:            # near-180 deg: read the axis off R + I
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        if k[i] > 1e-9:
            k = A[:, i] / k[i]
        return theta * k / max(float(np.linalg.norm(k)), 1e-12)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * axis / (2.0 * np.sin(theta))


def tilt_off_vertical_deg(R: np.ndarray, up_axis: int = 1) -> float:
    """How far a plate pose departs from a pure rotation about the world up axis.

    For a plate held vertical -- its own up along world up, free only to yaw
    about it -- the two off-yaw degrees of freedom are the world-up components
    of the plate's in-plane horizontal axis and of its normal.  Both vanish for
    a pure yaw, so the larger of them is a single number for "how non-vertical".

    Useful twice over: as an outlier test (a grossly mislabelled view yields a
    plate pose tens of degrees off vertical while still fitting its own points),
    and as the residual of a soft prior in the bundle.
    """
    R = np.asarray(R, float)
    others = [c for c in range(3) if c != up_axis]
    return float(np.degrees(np.arcsin(np.clip(
        max(abs(R[up_axis, others[0]]), abs(R[up_axis, others[1]])), 0.0, 1.0))))


@dataclass
class PlateObservations:
    """Flattened dot observations feeding the bundle.

    ``cam`` and ``frame`` index into the camera and plate-pose arrays; a
    ``frame`` of ``-1`` marks the reference frame, whose plate pose is the fixed
    gauge and therefore not an unknown.
    """

    cam: np.ndarray                 # (n,) int
    frame: np.ndarray               # (n,) int, -1 = reference frame
    obj: np.ndarray                 # (n,3) plate coordinates
    pix: np.ndarray                 # (n,2) observed pixels

    def __post_init__(self):
        n = len(self.cam)
        if not (len(self.frame) == len(self.obj) == len(self.pix) == n):
            raise ValueError("cam / frame / obj / pix must have equal length")


@dataclass
class BundleResult:
    cam_rvec: np.ndarray            # (ncam,3) world -> camera
    cam_tvec: np.ndarray            # (ncam,3)
    plate_rvec: np.ndarray          # (nframe,3) plate -> world
    plate_tvec: np.ndarray          # (nframe,3)
    keep: np.ndarray                # (n,) bool, dots surviving the trim
    residual_px: np.ndarray         # (n,) reprojection error of every dot
    trim_history: list = field(default_factory=list)

    def camera_centre(self, ci: int) -> np.ndarray:
        """Projection centre of camera ``ci`` in world coordinates."""
        return -rodrigues(self.cam_rvec[ci]).T @ self.cam_tvec[ci]


def _pack(cam_rvec, cam_tvec, plate_rvec, plate_tvec):
    return np.concatenate([np.ravel(cam_rvec), np.ravel(cam_tvec),
                           np.ravel(np.column_stack([plate_rvec, plate_tvec]))])


def _unpack(p, ncam, nframe):
    cam_rvec = p[:3 * ncam].reshape(ncam, 3)
    cam_tvec = p[3 * ncam:6 * ncam].reshape(ncam, 3)
    rest = p[6 * ncam:].reshape(nframe, 6) if nframe else np.zeros((0, 6))
    return cam_rvec, cam_tvec, rest[:, :3], rest[:, 3:]


def project(p, obs: PlateObservations, K, ncam, nframe):
    """Project every observation's plate point into its camera, in pixels."""
    cam_rvec, cam_tvec, plate_rvec, plate_tvec = _unpack(p, ncam, nframe)
    Rc = np.array([rodrigues(r) for r in cam_rvec])
    # row 0 is the reference frame's fixed identity gauge; frame -1 maps to it
    Rf = np.concatenate([np.eye(3)[None],
                         np.array([rodrigues(r) for r in plate_rvec])
                         if nframe else np.zeros((0, 3, 3))])
    tf = np.concatenate([np.zeros((1, 3)), plate_tvec])
    fi = obs.frame + 1
    Xw = np.einsum("nij,nj->ni", Rf[fi], obs.obj) + tf[fi]
    Xc = np.einsum("nij,nj->ni", Rc[obs.cam], Xw) + cam_tvec[obs.cam]
    z = Xc[:, 2]
    return np.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2],
                     K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1)


def bundle_plate_poses(
    obs: PlateObservations,
    cam_rvec0: np.ndarray,
    cam_tvec0: np.ndarray,
    plate_rvec0: np.ndarray,
    plate_tvec0: np.ndarray,
    K: np.ndarray,
    *,
    vertical_px: float = 0.0,
    vertical_sigma_deg: float = 1.0,
    up_axis: int = 1,
    trim_rounds: int = 6,
    trim_mad: float = 3.0,
    trim_floor_px: float = 1.0,
    max_nfev: int = 300,
) -> BundleResult:
    """Solve camera poses and per-frame plate poses together.

    ``vertical_px`` turns on a soft prior that each plate pose be a pure yaw
    about the world up axis, expressed as the pixel cost of one
    ``vertical_sigma_deg`` of tilt; 0 disables it.  Soft rather than hard on
    purpose -- a hand-held plate departs from vertical by a degree or so, and
    forcing that to zero biases the far corners of a large plate by more than
    the accuracy being chased.

    Outliers are trimmed on the bundle's own residuals, since a labelling that
    is wrong but internally self-consistent cannot be seen any other way.  The
    reference frame is never trimmed: it is the gauge.  Gate obviously-bad views
    out *before* calling this -- a robust loss still lets them drag the early
    iterations.
    """
    from scipy.optimize import least_squares

    ncam = len(cam_rvec0)
    nframe = len(plate_rvec0)
    K = np.asarray(K, float)
    x = _pack(cam_rvec0, cam_tvec0, plate_rvec0, plate_tvec0)

    def vertical_residual(p):
        if nframe == 0 or vertical_px <= 0.0:
            return np.zeros(0)
        w = vertical_px / np.sin(np.radians(vertical_sigma_deg))
        _, _, prv, _ = _unpack(p, ncam, nframe)
        others = [c for c in range(3) if c != up_axis]
        R = np.array([rodrigues(r) for r in prv])
        return w * np.stack([R[:, up_axis, others[0]],
                             R[:, up_axis, others[1]]], 1).ravel()

    keep = np.ones(len(obs.cam), bool)
    history = []
    for _ in range(max(1, trim_rounds)):
        sel = keep

        def fun(p, sel=sel):
            return np.concatenate([(project(p, obs, K, ncam, nframe) - obs.pix)[sel].ravel(),
                                   vertical_residual(p)])

        x = least_squares(fun, x, method="trf", loss="soft_l1", f_scale=1.0,
                          xtol=1e-12, ftol=1e-12, max_nfev=max_nfev).x
        err = np.linalg.norm(project(x, obs, K, ncam, nframe) - obs.pix, axis=1)
        thr = max(trim_floor_px, trim_mad * float(np.median(err[keep])))
        nxt = (err < thr) | (obs.frame < 0)
        history.append((int(keep.sum()), float(np.sqrt(np.mean(err[keep] ** 2))), thr))
        if nxt.sum() == keep.sum():
            keep = nxt
            break
        keep = nxt

    cam_rvec, cam_tvec, plate_rvec, plate_tvec = _unpack(x, ncam, nframe)
    err = np.linalg.norm(project(x, obs, K, ncam, nframe) - obs.pix, axis=1)
    return BundleResult(cam_rvec, cam_tvec, plate_rvec, plate_tvec, keep, err, history)


def agreeing_views(dots_per_view: dict, tol_mm: float) -> list:
    """Largest subset of views whose implied dot positions all agree.

    ``dots_per_view`` maps a view key to an ``(m,3)`` array of where that view
    says the plate's dots are in world coordinates.  Correct labellings agree;
    a mislabelled one lands elsewhere.

    Compared **per dot**, not per plate centre: a scrambled labelling can leave
    the centroid roughly where it belongs while the pattern around it is wrong,
    so a centre-only test passes frames that are visibly broken.
    """
    from itertools import combinations

    keys = list(dots_per_view)
    for size in range(len(keys), 1, -1):
        for sub in combinations(keys, size):
            if all(np.linalg.norm(dots_per_view[a] - dots_per_view[b], axis=1).max() < tol_mm
                   for a, b in combinations(sub, 2)):
                return list(sub)
    return []

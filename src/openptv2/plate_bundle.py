"""Joint bundle adjustment over many hand-held plate positions.

Solving every camera's pose on a single reference frame makes the model exact on
that plane by construction and lets its error grow linearly away from it.  On
the Illmenau 4-camera rig that cost 0.58 % of the plate's distance from the
anchor plane in ray-convergence miss -- sub-millimetre at the plane, ~18 mm at
3-4 m -- while per-camera reprojection RMS sat at a healthy 0.5 px throughout.
This module removes the anchoring: the unknowns become every camera pose *and*
one rigid plate pose per frame, solved together.

The camera interior, by default fixed:

``cc``, the principal point and the distortion
    By default the bundle is a pure pinhole with the camera matrix ``K`` held
    fixed: focal length is exactly degenerate on a single plane, and distortion
    fitted from few planes trades against pose and produces a polynomial that
    diverges outside the fitted points.  Over MANY plate poses spanning depth and
    yaw they are determined, and ``free_intrinsics`` fits them per camera in
    openptv's own interior model (``cc``, ``xh``, ``yh``, Brown ``k1..k3``,
    ``p1``, ``p2``), so the result writes to ``.ori``/``.addpar`` exactly.  On
    Illmenau that took the ray-convergence miss on held-out plates from 3.8 to
    0.3 mm.  Judge it on plates the fit did not see -- in-sample RMS always
    improves with more parameters.  Air only: the projection has no
    refraction, like the pinhole path.

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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: openptv interior parameters, in the column order of an ``intrinsics`` array:
#: ``cc``/``xh``/``yh`` (mm) from ``.ori``, ``k1..p2`` from ``.addpar``.
INTRINSICS = ("cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2")


@dataclass(frozen=True)
class Sensor:
    """Image size [px] and pixel pitch [mm]: openptv metric <-> pixels."""

    imx: int
    imy: int
    pix_x: float
    pix_y: float | None = None

    @property
    def pitch_y(self) -> float:
        return self.pix_x if self.pix_y is None else self.pix_y


def pinhole_intrinsics(K: np.ndarray, sensor: Sensor) -> np.ndarray:
    """openptv interior (:data:`INTRINSICS`) of an OpenCV pinhole ``K``."""
    K = np.asarray(K, float)
    return np.array(
        [
            K[0, 0] * sensor.pix_x,
            (K[0, 2] - sensor.imx / 2) * sensor.pix_x,
            (sensor.imy / 2 - K[1, 2]) * sensor.pitch_y,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )


def openptv_pixels(
    xn: np.ndarray, yn: np.ndarray, intr: np.ndarray, sensor: Sensor
) -> np.ndarray:
    """Pixels of normalised camera coordinates through openptv's interior.

    ``xn = Xc/Zc``, ``yn = Yc/Zc`` in the OpenCV camera frame (y down, z
    forward).  openptv's flat image coordinates are ``(cc*xn + xh, -cc*yn +
    yh)``; the Brown distortion is then applied about the sensor centre exactly
    as ``trafo.flat_to_dist`` does (``scx = 1``, ``she = 0``), and metric goes
    to pixels as ``trafo.metric_to_pixel`` (no interlace).  ``intr`` holds
    :data:`INTRINSICS` rows, one per point.
    """
    cc, xh, yh, k1, k2, k3, p1, p2 = np.moveaxis(np.asarray(intr, float), -1, 0)
    x = cc * xn + xh
    y = -cc * yn + yh
    r2 = x * x + y * y
    rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * rad + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
    yd = y * rad + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
    return np.stack(
        [xd / sensor.pix_x + sensor.imx * 0.5, sensor.imy * 0.5 - yd / sensor.pitch_y],
        -1,
    )


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector -> 3x3 rotation matrix (``cv2.Rodrigues`` without cv2)."""
    r = np.asarray(rvec, float).ravel()
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.asarray(
        np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K),
        dtype=np.float64,
    )


def rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector."""
    R = np.asarray(R, float)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-12:
        return np.zeros(3)
    if theta > np.pi - 1e-6:  # near-180 deg: read the axis off R + I
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        if k[i] > 1e-9:
            k = A[:, i] / k[i]
        return np.asarray(
            theta * k / max(float(np.linalg.norm(k)), 1e-12), dtype=np.float64
        )
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.asarray(theta * axis / (2.0 * np.sin(theta)), dtype=np.float64)


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
    return float(
        np.degrees(
            np.arcsin(
                np.clip(
                    max(abs(R[up_axis, others[0]]), abs(R[up_axis, others[1]])),
                    0.0,
                    1.0,
                )
            )
        )
    )


@dataclass
class PlateObservations:
    """Flattened dot observations feeding the bundle.

    ``cam`` and ``frame`` index into the camera and plate-pose arrays; a
    ``frame`` of ``-1`` marks the reference frame, whose plate pose is the fixed
    gauge and therefore not an unknown.
    """

    cam: np.ndarray  # (n,) int
    frame: np.ndarray  # (n,) int, -1 = reference frame
    obj: np.ndarray  # (n,3) plate coordinates
    pix: np.ndarray  # (n,2) observed pixels

    def __post_init__(self):
        n = len(self.cam)
        if not (len(self.frame) == len(self.obj) == len(self.pix) == n):
            raise ValueError("cam / frame / obj / pix must have equal length")


@dataclass
class BundleResult:
    cam_rvec: np.ndarray  # (ncam,3) world -> camera
    cam_tvec: np.ndarray  # (ncam,3)
    plate_rvec: np.ndarray  # (nframe,3) plate -> world
    plate_tvec: np.ndarray  # (nframe,3)
    keep: np.ndarray  # (n,) bool, dots surviving the trim
    residual_px: np.ndarray  # (n,) reprojection error of every dot
    trim_history: list = field(default_factory=list)
    intrinsics: np.ndarray | None = None  # (ncam, 8) INTRINSICS, if fitted

    def camera_centre(self, ci: int) -> np.ndarray:
        """Projection centre of camera ``ci`` in world coordinates."""
        return np.asarray(
            -rodrigues(self.cam_rvec[ci]).T @ self.cam_tvec[ci],
            dtype=np.float64,
        )


def _pack(cam_rvec, cam_tvec, plate_rvec, plate_tvec):
    return np.concatenate(
        [
            np.ravel(cam_rvec),
            np.ravel(cam_tvec),
            np.ravel(np.column_stack([plate_rvec, plate_tvec])),
        ]
    )


def _unpack(p, ncam, nframe):
    cam_rvec = p[: 3 * ncam].reshape(ncam, 3)
    cam_tvec = p[3 * ncam : 6 * ncam].reshape(ncam, 3)
    rest = p[6 * ncam :].reshape(nframe, 6) if nframe else np.zeros((0, 6))
    return cam_rvec, cam_tvec, rest[:, :3], rest[:, 3:]


def _camera_coords(
    p: np.ndarray, obs: PlateObservations, ncam: int, nframe: int
) -> np.ndarray:
    """Every observation's plate point in its camera's (OpenCV) frame."""
    cam_rvec, cam_tvec, plate_rvec, plate_tvec = _unpack(p, ncam, nframe)
    Rc = np.array([rodrigues(r) for r in cam_rvec])
    # row 0 is the reference frame's fixed identity gauge; frame -1 maps to it
    Rf = np.concatenate(
        [
            np.eye(3)[None],
            np.array([rodrigues(r) for r in plate_rvec])
            if nframe
            else np.zeros((0, 3, 3)),
        ]
    )
    tf = np.concatenate([np.zeros((1, 3)), plate_tvec])
    fi = obs.frame + 1
    Xw = np.einsum("nij,nj->ni", Rf[fi], obs.obj) + tf[fi]
    Xc: np.ndarray = np.einsum("nij,nj->ni", Rc[obs.cam], Xw) + cam_tvec[obs.cam]
    return Xc


def project(
    p: np.ndarray, obs: PlateObservations, K: np.ndarray, ncam: int, nframe: int
) -> np.ndarray:
    """Project every observation's plate point into its camera, in pixels."""
    Xc = _camera_coords(p, obs, ncam, nframe)
    z = Xc[:, 2]
    return np.stack(
        [K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1
    )


def project_intrinsics(
    p: np.ndarray,
    obs: PlateObservations,
    ncam: int,
    nframe: int,
    intrinsics: np.ndarray,
    sensor: Sensor,
) -> np.ndarray:
    """:func:`project` with a per-camera openptv interior (``(ncam, 8)``)."""
    Xc = _camera_coords(p, obs, ncam, nframe)
    return openptv_pixels(
        Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2], intrinsics[obs.cam], sensor
    )


def _jac_sparsity(
    obs: PlateObservations,
    sel: np.ndarray,
    ncam: int,
    nframe: int,
    nfree: int,
    n_vertical_rows: int,
) -> Any:  # scipy.sparse.coo_matrix
    """Which parameters each residual depends on: its camera, its frame."""
    from scipy.sparse import coo_matrix

    cam, frame = obs.cam[sel], obs.frame[sel]
    n, nb = len(cam), 6 * (ncam + nframe)
    blocks = [
        cam[:, None] * 3 + np.arange(3),
        3 * ncam + cam[:, None] * 3 + np.arange(3),
    ]
    if nfree:
        blocks.append(nb + cam[:, None] * nfree + np.arange(nfree))
    cols = np.concatenate(blocks, 1)
    rows = np.repeat(np.arange(n), cols.shape[1])
    cols = cols.ravel()
    has = frame >= 0
    fcols = (6 * ncam + 6 * frame[has])[:, None] + np.arange(6)
    rows = np.concatenate([rows, np.repeat(np.flatnonzero(has), 6)])
    cols = np.concatenate([cols, fcols.ravel()])
    rows = np.concatenate([2 * rows, 2 * rows + 1])
    cols = np.concatenate([cols, cols])
    if n_vertical_rows:
        j = np.arange(nframe)
        vr = 2 * n + np.repeat(np.stack([2 * j, 2 * j + 1], 1).ravel(), 3)
        vc = np.repeat(6 * ncam + 6 * j, 2)[:, None] + np.arange(3)
        rows = np.concatenate([rows, vr])
        cols = np.concatenate([cols, vc.ravel()])
    return coo_matrix(
        (np.ones(len(rows), int), (rows, cols)),
        shape=(2 * n + n_vertical_rows, nb + ncam * nfree),
    )


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
    sensor: Sensor | None = None,
    intrinsics0: np.ndarray | None = None,
    free_intrinsics: Sequence[str] = (),
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

    With a ``sensor`` the cameras use openptv's interior model instead of ``K``:
    ``intrinsics0`` (``(ncam, 8)``, columns :data:`INTRINSICS`; default: ``K``
    as a pinhole for every camera) is the start, and the names in
    ``free_intrinsics`` are fitted per camera.  The result then carries the
    fitted ``intrinsics``.
    """
    from scipy.optimize import least_squares

    ncam = len(cam_rvec0)
    nframe = len(plate_rvec0)
    K = np.asarray(K, float)
    if vertical_px > 0 and vertical_sigma_deg <= 0:
        raise ValueError(
            "vertical_sigma_deg must be >0 when vertical_px>0 (got 0 or negative, would divide by sin(0))"
        )
    x = _pack(cam_rvec0, cam_tvec0, plate_rvec0, plate_tvec0)
    nb = len(x)
    if sensor is None and (intrinsics0 is not None or free_intrinsics):
        raise ValueError("intrinsics0 / free_intrinsics need a sensor")
    unknown = [n for n in free_intrinsics if n not in INTRINSICS]
    if unknown:
        raise ValueError(f"unknown intrinsics {unknown}; choose from {INTRINSICS}")
    free_idx = [INTRINSICS.index(n) for n in free_intrinsics]
    intr0 = np.zeros((ncam, len(INTRINSICS)))
    if sensor is not None:
        intr0 = (
            np.tile(pinhole_intrinsics(K, sensor), (ncam, 1))
            if intrinsics0 is None
            else np.array(intrinsics0, float).reshape(ncam, len(INTRINSICS))
        )
        x = np.concatenate([x, intr0[:, free_idx].ravel()])

    def intrinsics_of(p: np.ndarray) -> np.ndarray:
        intr = intr0.copy()
        intr[:, free_idx] = p[nb:].reshape(ncam, len(free_idx))
        return intr

    def proj(p: np.ndarray) -> np.ndarray:
        if sensor is None:
            return project(p, obs, K, ncam, nframe)
        return project_intrinsics(p[:nb], obs, ncam, nframe, intrinsics_of(p), sensor)

    def vertical_residual(p):
        if nframe == 0 or vertical_px <= 0.0:
            return np.zeros(0)
        w = vertical_px / np.sin(np.radians(vertical_sigma_deg))
        _, _, prv, _ = _unpack(p[:nb], ncam, nframe)
        others = [c for c in range(3) if c != up_axis]
        R = np.array([rodrigues(r) for r in prv])
        return (
            w
            * np.stack([R[:, up_axis, others[0]], R[:, up_axis, others[1]]], 1).ravel()
        )

    keep = np.ones(len(obs.cam), bool)
    history = []
    for _ in range(max(1, trim_rounds)):
        sel = keep

        def fun(p, sel=sel):
            return np.concatenate(
                [(proj(p) - obs.pix)[sel].ravel(), vertical_residual(p)]
            )

        extra: dict = {}
        if sensor is not None:
            nv = 2 * nframe if (nframe and vertical_px > 0.0) else 0
            extra = dict(
                jac_sparsity=_jac_sparsity(obs, sel, ncam, nframe, len(free_idx), nv),
                x_scale="jac",
            )
        x = least_squares(
            fun,
            x,
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            xtol=1e-12,
            ftol=1e-12,
            max_nfev=max_nfev,
            **extra,
        ).x
        err = np.linalg.norm(proj(x) - obs.pix, axis=1)
        thr = max(trim_floor_px, trim_mad * float(np.median(err[keep])))
        nxt = (err < thr) | (obs.frame < 0)
        history.append((int(keep.sum()), float(np.sqrt(np.mean(err[keep] ** 2))), thr))
        if nxt.sum() == keep.sum():
            keep = nxt
            break
        keep = nxt

    cam_rvec, cam_tvec, plate_rvec, plate_tvec = _unpack(x[:nb], ncam, nframe)
    err = np.linalg.norm(proj(x) - obs.pix, axis=1)
    return BundleResult(
        cam_rvec,
        cam_tvec,
        plate_rvec,
        plate_tvec,
        keep,
        err,
        history,
        intrinsics_of(x) if sensor is not None else None,
    )


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
            if all(
                np.linalg.norm(dots_per_view[a] - dots_per_view[b], axis=1).max()
                < tol_mm
                for a, b in combinations(sub, 2)
            ):
                return list(sub)
    return []

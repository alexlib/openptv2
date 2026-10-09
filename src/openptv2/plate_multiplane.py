"""Multi-plane plate calibration: the steps around detection and the bundle.

A rigid dot plate is held at many unknown positions and every camera of a group
photographs it.  This module holds the parts of that procedure that do not
depend on a particular dataset:

* :class:`PlateGrid` -- the printed lattice and its datum dot: point ids <->
  plate coordinates, and the calibration block written from them.
* :func:`detect_coded_plate` / :func:`label_plate_view` -- detection with a
  search for the coded-dot threshold, then coded-L labelling anchored on the
  datum; :func:`datum_index_from_complete_view` reads the datum off the data.
* :func:`save_plate_views` / :func:`load_plate_views` -- the detection cache
  that every later step reads, so all of them use one labelling.
* :func:`fit_shared_cc` -- the one shared focal length from multi-plane
  consistency (``cc`` is exactly degenerate on a single plane).
* :func:`pnp_pose` / :func:`pinhole_calibration` -- a pure-pinhole camera from
  one plate view.
* :func:`prepare_bundle` -- the three view gates and the initial poses for
  :func:`openptv2.plate_bundle.bundle_plate_poses`.
* checks that fit nothing: :func:`triangulate_plate` +
  :func:`plate_frame_metrics` (planarity, pitch, deviation from the rigid
  grid, ray-convergence miss), :func:`epipolar_curve_misses` and
  :func:`epipolar_horizon_z`.

The procedure, the order of the steps and what each check catches are in
``.claude/skills/openptv-multiplane-calibrate/SKILL.md``; the numbers on a real
rig are in ``docs/illmenau-4cam-calibration.md``.  Thin drivers for that rig
live in ``scripts/illmenau/``.

OpenCV is needed only for the PnP steps and is imported where they run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.orientation import COORD_UNUSED
from openptv2.algorithms.ray_tracing import ray_tracing
from openptv2.algorithms.trafo import dist_to_flat, metric_to_pixel, pixel_to_metric
from openptv2.plate_bundle import (
    PlateObservations,
    agreeing_views,
    rodrigues,
    rotvec,
    tilt_off_vertical_deg,
)

if TYPE_CHECKING:
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.parameters import ControlPar, TargetPar
    from openptv2.detect_plate import PlateDetectionResult


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the PnP steps of the plate calibration need OpenCV: "
            "uv pip install opencv-python-headless"
        ) from e
    return cv2


# --------------------------------------------------------------------------
# the plate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlateGrid:
    """A regular ``nx`` x ``ny`` dot lattice and its datum dot.

    Point ids are row-major and 1-based from the bottom-left node,
    ``id = iy*nx + ix + 1``.  The datum is the grid index of the coded L corner;
    that dot is the world origin, so the world is pinned to a physical piece of
    the plate rather than to whatever a view happened to see.
    """

    nx: int
    ny: int
    pitch_x: float
    pitch_y: float
    datum_ix: int = 0
    datum_iy: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> PlateGrid:
        """Read the ``plate:`` block of a ``plate.yaml``."""
        import yaml

        p = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})["plate"]
        d = p.get("datum") or {}
        return cls(
            nx=int(p["nx"]),
            ny=int(p["ny"]),
            pitch_x=float(p["pitch_x"]),
            pitch_y=float(p.get("pitch_y", p["pitch_x"])),
            datum_ix=int(d.get("ix", 0)),
            datum_iy=int(d.get("iy", 0)),
        )

    @property
    def n_points(self) -> int:
        return self.nx * self.ny

    @property
    def datum_id(self) -> int:
        return self.datum_iy * self.nx + self.datum_ix + 1

    def ids_from_index(self, ix: Any, iy: Any) -> np.ndarray:
        return np.asarray(iy) * self.nx + np.asarray(ix) + 1

    def object_points(self, ids: Any) -> np.ndarray:
        """Plate coordinates of point ids: datum dot at the origin, plate in z=0."""
        ids = np.asarray(ids)
        ix, iy = (ids - 1) % self.nx, (ids - 1) // self.nx
        return np.stack(
            [
                (ix - self.datum_ix) * self.pitch_x,
                (iy - self.datum_iy) * self.pitch_y,
                np.zeros(len(ix)),
            ],
            1,
        ).astype(float)

    def calibration_block_lines(self) -> list[str]:
        """``id X Y Z`` lines of ``cal/calibration_block.txt`` for this plate."""
        ids = np.arange(1, self.n_points + 1)
        return [
            f"{i} {p[0]:.1f} {p[1]:.1f} {p[2]:.1f}"
            for i, p in zip(ids, self.object_points(ids))
        ]

    def neighbour_pitches(
        self, pos: np.ndarray, ids: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Distances between triangulated neighbours along X and along Y."""
        idx = {p: k for k, p in enumerate(ids)}
        dx = [
            np.linalg.norm(pos[idx[p]] - pos[idx[p + 1]])
            for p in ids
            if p + 1 in idx and ((p - 1) % self.nx) < self.nx - 1
        ]
        dy = [
            np.linalg.norm(pos[idx[p]] - pos[idx[p + self.nx]])
            for p in ids
            if p + self.nx in idx
        ]
        return np.asarray(dx, float), np.asarray(dy, float)


# --------------------------------------------------------------------------
# detection, labelling and the detection cache
# --------------------------------------------------------------------------


def detect_coded_plate(
    image: np.ndarray,
    tpar: TargetPar,
    cpar: ControlPar,
    cam: int,
    *,
    coded_thresholds: Sequence[float] = (30, 25, 20, 15, 10),
    n_coded: int = 3,
) -> PlateDetectionResult | None:
    """Detect the plate dots, searching for the threshold that finds the code.

    The number of coded dots is known a priori, so instead of fixing one
    ``coded_thr`` this returns the first detection that finds exactly
    ``n_coded`` of them, or ``None``.  A single fixed threshold missed the code
    entirely on some views, leaving the labeller nothing to anchor on.
    """
    from openptv2.detect_plate import detect_plate_targets

    for thr in coded_thresholds:
        r = detect_plate_targets(image, tpar, cpar, cam=cam, coded_thr=float(thr))
        if int(r.coded_mask.sum()) == n_coded:
            return r
    return None


def label_plate_view(
    detection: PlateDetectionResult,
    grid: PlateGrid,
    *,
    cal: Calibration | None = None,
    cpar: ControlPar | None = None,
    y_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Label one detection: ``(ids, pixels)`` on the datum-anchored grid.

    The labeller is given the datum (``corner_index``) so a view that misses a
    column or row is not shifted one pitch, and -- when a calibration ``cal`` is
    passed -- the image direction of world +Y (``up_hint``), which settles the
    coded L under strong perspective for a plate held vertical.
    """
    from openptv2.plate_labeler import image_up_direction, label_plate

    hint = None
    if cal is not None and len(detection.centroids):
        hint = image_up_direction(cal, cpar, np.mean(detection.centroids, axis=0))
    ip, rp, _ = label_plate(
        detection.centroids,
        detection.coded_mask,
        pitch_x=grid.pitch_x,
        pitch_y=grid.pitch_y,
        nx=grid.nx,
        ny=grid.ny,
        y_sign=y_sign,
        corner_index=(grid.datum_ix, grid.datum_iy),
        up_hint=hint,
    )
    ix = np.round(rp[:, 0] / grid.pitch_x).astype(int)
    iy = np.round(rp[:, 1] / grid.pitch_y).astype(int)
    return grid.ids_from_index(ix, iy), ip


def datum_index_from_complete_view(
    detection: PlateDetectionResult, grid: PlateGrid, *, y_sign: int = 1
) -> tuple[int, int] | None:
    """Grid index ``(ix, iy)`` of the coded L corner, read off one view.

    Only a view that sees the complete lattice can answer: labelled without a
    ``corner_index``, the grid anchors on the smallest detected index, which is
    a true 0 only when nothing is missing.  Returns ``None`` for any other view.
    """
    from openptv2.plate_labeler import label_plate

    if len(detection.centroids) < grid.n_points:
        return None
    ip, _, idx = label_plate(
        detection.centroids,
        detection.coded_mask,
        pitch_x=grid.pitch_x,
        pitch_y=grid.pitch_y,
        nx=grid.nx,
        ny=grid.ny,
        y_sign=y_sign,
    )
    if len(ip) < grid.n_points:
        return None
    # the corner is the coded dot whose two partners sit at 1 and 2 pitch
    coded = detection.centroids[detection.coded_mask]
    d = np.linalg.norm(coded[:, None, :] - coded[None, :, :], axis=2)
    corner_xy = coded[int(np.argmin(np.sort(d, axis=1)[:, 1:].sum(1)))]
    k = int(np.argmin(np.linalg.norm(ip - corner_xy, axis=1)))
    return int(idx[k, 0]), int(idx[k, 1])


def save_plate_views(path: str | Path, views: dict) -> None:
    """Write ``{(cam_index, frame): (ids, pixels)}`` as the detection cache."""
    store = {}
    for (ci, fr), (ids, px) in views.items():
        store[f"c{ci}_{fr}_ids"] = ids
        store[f"c{ci}_{fr}_px"] = px
    np.savez_compressed(path, **store)


def load_plate_views(path: str | Path) -> dict:
    """``{(cam_index, frame): (ids, pixels)}`` from the detection cache."""
    d = np.load(path)
    views = {}
    for k in d.files:
        if k.endswith("_ids"):
            c, fr, _ = k.split("_")
            views[(int(c[1:]), fr)] = (d[k], d[f"{c}_{fr}_px"])
    return views


# --------------------------------------------------------------------------
# pinhole poses and the shared focal length
# --------------------------------------------------------------------------


def pinhole_K(cc_mm: float, pix_mm: float, imx: int, imy: int) -> np.ndarray:
    """Camera matrix of a pinhole with its principal point at the sensor centre."""
    return np.array(
        [[cc_mm / pix_mm, 0, imx / 2], [0, cc_mm / pix_mm, imy / 2], [0, 0, 1.0]]
    )


def pnp_pose(
    obj: np.ndarray, px: np.ndarray, K: np.ndarray, *, min_points: int = 6
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """``(rvec, tvec, rms_px)`` of the plate in one view, or ``None``.

    ``cv2.solvePnP`` refined by ``solvePnPRefineLM``, zero distortion.  The
    residual uses no other camera, so a large one means the view is mislabelled
    rather than that the pose is hard to fit.
    """
    cv2 = _cv2()
    if len(obj) < min_points:
        return None
    d0 = np.zeros(5)
    ok, rv, tv = cv2.solvePnP(obj, px.astype(np.float64), K, d0)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, px.astype(np.float64), K, d0, rv, tv)
    rep, _ = cv2.projectPoints(obj, rv, tv, K, d0)
    rms = float(np.sqrt(np.mean(np.sum((rep.reshape(-1, 2) - px) ** 2, 1))))
    return rv.ravel(), tv.ravel(), rms


def pinhole_calibration(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    *,
    imx: int,
    imy: int,
    pix_mm: float,
) -> Calibration:
    """openptv2 :class:`Calibration` of a pure pinhole: ``.addpar`` all zero."""
    from openptv2.calibration_import import calibration_from_opencv

    cal, _ = calibration_from_opencv(
        K,
        np.zeros(5),
        rvec,
        tvec,
        imx=imx,
        imy=imy,
        pix_x=pix_mm,
        pixel_origin="corner",
    )
    return cal


@dataclass
class SharedCCFit:
    cc: float  # best shared focal length [mm]
    spread_mm: float  # median cross-camera spread of the plate centre at cc
    rows: list  # (frame, distance from the reference plane, spread) at cc
    frames: list  # frames every camera saw well enough to be used
    scan: list = field(default_factory=list)  # (cc, n frames, spread) per trial


def plate_spread(
    cc: float,
    views: dict,
    grid: PlateGrid,
    ref: str,
    frames: list,
    ncam: int,
    *,
    imx: int,
    imy: int,
    pix_mm: float,
    max_view_rms: float = 1.5,
) -> tuple[float, list]:
    """Median cross-camera spread of the plate centre for a trial ``cc``.

    Each camera's pose is fitted on the reference frame, then for every other
    frame each camera says on its own where the plate is.  With the right
    ``cc`` the answers coincide; with a wrong one every camera's world sits at
    the wrong distance and they spread apart, the more the further the plate is
    from the reference plane.
    """
    K = pinhole_K(cc, pix_mm, imx, imy)
    ref_pose = {}
    for ci in range(ncam):
        p = pnp_pose(grid.object_points(views[(ci, ref)][0]), views[(ci, ref)][1], K)
        if p is None:
            return np.inf, []
        ref_pose[ci] = p
    rows = []
    for fr in frames:
        if fr == ref:
            continue
        ts: list[np.ndarray] = []
        complete = True
        for ci in range(ncam):
            ids, px = views[(ci, fr)]
            p = pnp_pose(grid.object_points(ids), px, K)
            if p is None or p[2] > max_view_rms:
                complete = False
                break
            R0, _ = _cv2().Rodrigues(ref_pose[ci][0])
            ts.append(R0.T @ (p[1] - ref_pose[ci][1]))
        if not complete or len(ts) < ncam:
            continue
        t = np.array(ts)
        rows.append(
            (
                fr,
                np.linalg.norm(t.mean(0)),
                float(np.max(np.linalg.norm(t - t.mean(0), axis=1))),
            )
        )
    if not rows:
        return np.inf, []
    return float(np.median([r[2] for r in rows])), rows


def fit_shared_cc(
    views: dict,
    grid: PlateGrid,
    ref: str,
    ncam: int,
    cc_scan: Any,
    *,
    imx: int,
    imy: int,
    pix_mm: float,
    bracket: float | None = None,
    refine_iter: int = 30,
    min_dots: int = 12,
    max_view_rms: float = 1.5,
) -> SharedCCFit:
    """One shared ``cc`` minimising :func:`plate_spread`.

    Scans ``cc_scan`` (bracket it around the nominal lens, e.g. +-25 %), then
    refines by ternary search within ``bracket`` (default: the scan step) of the
    best trial.  Uses frames where every camera sees at least ``min_dots``.
    """
    frames = [
        f
        for f in sorted({f for _, f in views})
        if all(
            (ci, f) in views and len(views[(ci, f)][0]) >= min_dots
            for ci in range(ncam)
        )
    ]

    def spread(cc: float) -> tuple[float, list]:
        return plate_spread(
            cc,
            views,
            grid,
            ref,
            frames,
            ncam,
            imx=imx,
            imy=imy,
            pix_mm=pix_mm,
            max_view_rms=max_view_rms,
        )

    cc_scan = np.asarray(cc_scan, float)
    if bracket is None:
        bracket = float(cc_scan[1] - cc_scan[0])
    scan: list = []
    best: tuple[float, float] | None = None
    for cc in cc_scan:
        s, rows = spread(float(cc))
        scan.append((float(cc), len(rows), s))
        if best is None or s < best[0]:
            best = (s, float(cc))
    assert best is not None, "cc_scan is empty"
    lo, hi = best[1] - bracket, best[1] + bracket
    for _ in range(refine_iter):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if spread(m1)[0] < spread(m2)[0]:
            hi = m2
        else:
            lo = m1
    cc = 0.5 * (lo + hi)
    s, rows = spread(cc)
    return SharedCCFit(cc=cc, spread_mm=s, rows=rows, frames=frames, scan=scan)


# --------------------------------------------------------------------------
# joint bundle: gates and initial poses
# --------------------------------------------------------------------------


@dataclass
class BundleSetup:
    """Everything :func:`openptv2.plate_bundle.bundle_plate_poses` needs."""

    obs: PlateObservations
    cam_rvec0: np.ndarray  # (ncam,3) poses on the reference frame
    cam_tvec0: np.ndarray
    plate_rvec0: np.ndarray  # (len(free),3) plate -> world
    plate_tvec0: np.ndarray
    frames: list  # frames kept by the gates (reference included)
    free: list  # frames whose plate pose is an unknown (all but the reference)
    poses: dict  # (cam, frame) -> (rvec, tvec, rms) of every view kept
    n_pnp: int  # views passing gate 1
    tilt_rejects: list  # (frame, cam, tilt deg) rejected by gate 2
    dropped: list  # (frame, cams, worst per-dot disagreement mm) by gate 3


def prepare_bundle(
    views: dict,
    grid: PlateGrid,
    K: np.ndarray,
    ncam: int,
    ref: str,
    *,
    min_dots: int = 12,
    view_gate_px: float = 1.0,
    tilt_gate_deg: float = 5.0,
    agree_mm: float = 100.0,
) -> BundleSetup:
    """Gate the views and build the bundle's observations and initial poses.

    Three gates run BEFORE the bundle, each catching what the previous cannot;
    a robust loss and residual trimming still let a bad view drag the early
    iterations.

    1. per-camera PnP residual < ``view_gate_px`` -- uses no cross-camera
       information, so it is a pure labelling test for one view;
    2. plate upright and within ``tilt_gate_deg`` of vertical (its +Y along
       world +Y) -- for a plate held vertical, a pose tens of degrees off is a
       mislabelling however well it fits its own points.  A plate is never
       upside down either, so a pose with its +Y pointing down counts as
       ``180 - lean`` degrees off: that is a grid numbered from the wrong
       corner (a 180 deg relabelling), which is still exactly vertical and
       would otherwise pass;
    3. per-dot cross-camera agreement < ``agree_mm`` (:func:`agreeing_views`).

    The reference frame must survive gate 1 in every camera; its plate pose is
    the gauge.
    """
    frames_all = sorted({f for _, f in views})
    good = {}
    for fr in frames_all:
        for ci in range(ncam):
            if (ci, fr) not in views or len(views[(ci, fr)][0]) < min_dots:
                continue
            ids, px = views[(ci, fr)]
            p = pnp_pose(grid.object_points(ids), px, K)
            if p is not None and p[2] < view_gate_px:
                good[(ci, fr)] = p
    if any((ci, ref) not in good for ci in range(ncam)):
        raise ValueError(f"reference frame {ref} is not clean in all {ncam} cameras")
    n_pnp = len(good)

    ref_R = {ci: rodrigues(good[(ci, ref)][0]) for ci in range(ncam)}
    ref_t = {ci: good[(ci, ref)][1] for ci in range(ncam)}
    all_dots = grid.object_points(np.arange(1, grid.n_points + 1))

    def dots_in_world(ci: int, fr: str) -> np.ndarray:
        rv, tv, _ = good[(ci, fr)]
        dots: np.ndarray = (all_dots @ rodrigues(rv).T + tv - ref_t[ci]) @ ref_R[ci]
        return dots

    tilt_rejects = []
    for key in list(good):
        ci, fr = key
        if fr == ref:
            continue
        R = ref_R[ci].T @ rodrigues(good[key][0])
        t = tilt_off_vertical_deg(R)
        if R[1, 1] < 0:  # plate +Y points down: upside down
            t = 180.0 - t
        if t > tilt_gate_deg:
            tilt_rejects.append((fr, ci, t))
            del good[key]

    kept, dropped = {}, []
    for fr in frames_all:
        vs = [ci for ci in range(ncam) if (ci, fr) in good]
        if len(vs) < 2:
            continue
        per = {ci: dots_in_world(ci, fr) for ci in vs}
        best = agreeing_views(per, agree_mm)
        worst = max(
            np.linalg.norm(per[a] - per[b], axis=1).max()
            for a in vs
            for b in vs
            if a < b
        )
        if not best:
            dropped.append((fr, vs, worst))
            continue
        if len(best) < len(vs):
            dropped.append((fr, [c for c in vs if c not in best], worst))
        kept[fr] = best
    good = {(ci, fr): good[(ci, fr)] for fr, cs in kept.items() for ci in cs}
    frames = sorted(kept)
    free = [fr for fr in frames if fr != ref]

    fidx = {fr: k for k, fr in enumerate(free)}
    cam_i, frm_i, objp, pixp = [], [], [], []
    for fr in frames:
        for ci in range(ncam):
            if (ci, fr) not in good:
                continue
            ids, px = views[(ci, fr)]
            o = grid.object_points(ids)
            cam_i.append(np.full(len(o), ci))
            frm_i.append(np.full(len(o), fidx.get(fr, -1)))
            objp.append(o)
            pixp.append(px.astype(float))
    obs = PlateObservations(
        np.concatenate(cam_i),
        np.concatenate(frm_i),
        np.concatenate(objp),
        np.concatenate(pixp),
    )

    cam_rvec0 = np.array([good[(ci, ref)][0] for ci in range(ncam)])
    cam_tvec0 = np.array([good[(ci, ref)][1] for ci in range(ncam)])
    prv0, ptv0 = [], []
    for fr in free:
        ci = next(c for c in range(ncam) if (c, fr) in good)
        rv, tv, _ = good[(ci, fr)]
        prv0.append(rotvec(ref_R[ci].T @ rodrigues(rv)))
        ptv0.append(ref_R[ci].T @ (tv - ref_t[ci]))
    return BundleSetup(
        obs=obs,
        cam_rvec0=cam_rvec0,
        cam_tvec0=cam_tvec0,
        plate_rvec0=np.array(prv0).reshape(-1, 3),
        plate_tvec0=np.array(ptv0).reshape(-1, 3),
        frames=frames,
        free=free,
        poses=good,
        n_pnp=n_pnp,
        tilt_rejects=tilt_rejects,
        dropped=dropped,
    )


# --------------------------------------------------------------------------
# checks: triangulation and epipolar geometry (they fit nothing)
# --------------------------------------------------------------------------


def pixel_to_flat(
    pixel: Sequence[float], cal: Calibration, cpar: ControlPar
) -> tuple[float, float]:
    """Distortion-free metric image coordinates of a pixel in camera ``cal``."""
    mx, my = pixel_to_metric(pixel[0], pixel[1], cpar)
    a = cal.added_par
    return dist_to_flat(
        mx,
        my,
        cal.int_par.xh,
        cal.int_par.yh,
        a.k1,
        a.k2,
        a.k3,
        a.p1,
        a.p2,
        a.scx,
        a.she,
    )


def sight_ray(
    pixel: Sequence[float], cal: Calibration, cpar: ControlPar
) -> tuple[np.ndarray, np.ndarray]:
    """``(vertex, direction)`` of the sight ray through ``pixel``, world frame."""
    xf, yf = pixel_to_flat(pixel, cal, cpar)
    mm: Any = cpar.mm
    pos, v = ray_tracing(
        xf,
        yf,
        cal.ext_par.dm,  # type: ignore[arg-type]  # Cython memoryview
        cal.ext_par.x0,
        cal.ext_par.y0,
        cal.ext_par.z0,
        cal.int_par.cc,
        cal.glass_par.vec_x,
        cal.glass_par.vec_y,
        cal.glass_par.vec_z,
        mm.n1,
        mm.n2[0],
        mm.n3,
        mm.d[0],
    )
    return np.asarray(pos, float), np.asarray(v, float)


@dataclass
class PlateTriangulation:
    pos: np.ndarray  # (n,3) triangulated dots
    ids: np.ndarray  # (n,) point ids
    rcm: np.ndarray  # (n,) ray-convergence miss [mm]
    ncam: np.ndarray  # (n,) cameras that saw each dot


def triangulate_plate(
    per_cam: dict,
    cals: Sequence[Calibration],
    cpar: ControlPar,
    *,
    min_dots: int = 6,
) -> PlateTriangulation | None:
    """Triangulate one frame's labelled dots seen by at least two cameras.

    ``per_cam`` maps a camera index (into ``cals``) to ``(ids, pixels)``.  Dots
    whose position or ray-convergence miss is not finite are dropped; ``None``
    when fewer than ``min_dots`` are left.
    """
    from openptv2.orientation import multi_cam_point_positions

    per = {
        ci: dict(zip(ids.tolist(), px.tolist())) for ci, (ids, px) in per_cam.items()
    }
    ids = [
        i
        for i in sorted({i for m in per.values() for i in m})
        if sum(i in m for m in per.values()) >= 2
    ]
    if len(ids) < min_dots:
        return None
    t = np.full((len(ids), len(cals), 2), COORD_UNUSED)
    ncam = []
    for k, pid in enumerate(ids):
        n = 0
        for ci, m in per.items():
            if pid in m:
                t[k, ci] = pixel_to_flat(m[pid], cals[ci], cpar)
                n += 1
        ncam.append(n)
    pos, rcm = multi_cam_point_positions(t, cpar, cals)
    rcm = np.asarray(rcm, float)
    ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1) & np.isfinite(rcm)
    if ok.sum() < min_dots:
        return None
    return PlateTriangulation(
        pos=pos[ok], ids=np.asarray(ids)[ok], rcm=rcm[ok], ncam=np.asarray(ncam)[ok]
    )


def rigid_fit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``A`` moved rigidly onto ``B`` (Kabsch) -- no scaling, so pitch error shows."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out: np.ndarray = (A - ca) @ R + cb
    return out


@dataclass
class PlateFrameMetrics:
    centre: np.ndarray  # mean of the triangulated dots
    axes: np.ndarray  # (3,3) rows: in-plane e1, in-plane e2, plane normal
    residual: np.ndarray  # (n,) signed distance from the best-fit plane [mm]
    pitch_x: np.ndarray  # neighbour distances along X [mm]
    pitch_y: np.ndarray  # neighbour distances along Y [mm]
    grid_fit: np.ndarray  # (n,3) ideal rigid grid fitted onto the dots
    grid_dev: np.ndarray  # (n,) distance of each dot from its grid node [mm]
    nominal_error: np.ndarray  # (n,3) dots minus their block coordinates

    @property
    def normal(self) -> np.ndarray:
        return np.asarray(self.axes[2])

    @property
    def planarity_rms(self) -> float:
        return float(np.sqrt(np.mean(self.residual**2)))

    @property
    def planarity_max(self) -> float:
        return float(np.abs(self.residual).max())


def plate_frame_metrics(
    pos: np.ndarray, ids: Any, grid: PlateGrid
) -> PlateFrameMetrics:
    """What a triangulated plate says about the calibration and the labels.

    * planarity -- little on its own: a systematic model error distorts a plane
      consistently and it still looks flat;
    * pitch -- the absolute scale;
    * deviation from the rigid grid (Kabsch, no scaling) -- did the pattern come
      back as the pattern; a large value means the labeller mis-assigned dots;
    * error against the block coordinates, no alignment -- meaningful on the
      reference frame, where the world is defined.

    Pair it with the ray-convergence miss from :func:`triangulate_plate`, which
    uses no plate model: a bad miss with a good grid is a calibration error.
    """
    centre = pos.mean(0)
    _, _, Vt = np.linalg.svd(pos - centre)
    nominal = grid.object_points(ids)
    fit = rigid_fit(nominal, pos)
    dx, dy = grid.neighbour_pitches(pos, ids)
    return PlateFrameMetrics(
        centre=centre,
        axes=Vt,
        residual=(pos - centre) @ Vt[2],
        pitch_x=dx,
        pitch_y=dy,
        grid_fit=fit,
        grid_dev=np.linalg.norm(pos - fit, axis=1),
        nominal_error=pos - nominal,
    )


def epipolar_curve_misses(
    det_a: dict,
    det_b: dict,
    cal_a: Calibration,
    cal_b: Calibration,
    cpar: ControlPar,
    z_values: Any,
    *,
    margin_px: float = 200.0,
) -> tuple[list, int]:
    """Closest approach of camera A's epipolar CURVE to B's matching dot.

    For every dot id both cameras labelled (``det_*`` map id -> pixel), the sight
    ray from A is sampled at ``z_values``, projected into B, and only samples
    inside B's sensor (plus ``margin_px``) are kept.  Returns the per-dot misses
    [px] and how many projected rays are monotone (do not double back).

    Do not approximate the curve by the chord between two far endpoints -- that
    produced a spurious 289 px reading once.  A straight 3D ray that doubles
    back in the image means the distortion model is unphysical, which no miss
    distance shows.
    """
    z_values = np.asarray(z_values, float)
    misses, monotone = [], 0
    for pid, pa in det_a.items():
        if pid not in det_b:
            continue
        pos, v = sight_ray(pa, cal_a, cpar)
        P = pos + ((z_values - pos[2]) / v[2])[:, None] * v
        q = np.array([metric_to_pixel(*img_coord(p, cal_b, cpar.mm), cpar) for p in P])
        inside = (
            (q[:, 0] > -margin_px)
            & (q[:, 0] < cpar.imx + margin_px)
            & (q[:, 1] > -margin_px)
            & (q[:, 1] < cpar.imy + margin_px)
        )
        if inside.sum() < 3:
            continue
        qi = q[inside]
        misses.append(float(np.min(np.linalg.norm(qi - np.array(det_b[pid]), axis=1))))
        step = np.diff(qi, axis=0)
        monotone += int(np.all(step @ step[0] > 0))
    return misses, monotone


def epipolar_horizon_z(
    ray_pos: np.ndarray, ray_dir: np.ndarray, cal_b: Calibration
) -> float | None:
    """Z at which a sight ray crosses camera B's principal plane.

    Past it the point is behind B: its projection runs off to infinity and
    returns on the other side of the sensor, so an epipolar segment drawn to a
    ``Zmax_lay`` beyond the horizon is thrown across the image.  ``None`` when
    the ray is parallel to that plane.
    """
    Cb = np.array([cal_b.ext_par.x0, cal_b.ext_par.y0, cal_b.ext_par.z0])
    axis = -np.asarray(cal_b.ext_par.dm)[:, 2]  # the camera looks along -dm[:,2]
    k = (ray_dir @ axis) / ray_dir[2]
    if abs(k) < 1e-12:
        return None
    return float(ray_pos[2] - ((ray_pos - Cb) @ axis) / k)

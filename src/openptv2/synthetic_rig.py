"""A synthetic camera rig with a known, independently computed imaging model, for testing calibration step by step.

The calibration code of openptv2 models each camera as a pinhole with lens distortion behind a stack of flat parallel
layers (air | glass of thickness ``d`` | water), traced by a radial shift. This module computes the same imaging by an
*independent* route (Snell's law in 3D, solved for the ray that really reaches the point), and, unlike the calibration
model, it can break the assumptions one at a time: a tilted or wedge-shaped glass, a wrong thickness or index, a lens
term the model does not have. A rig is therefore the truth; a calibration is a guess about it, and every difference is
known.

    rig = TruthRig.from_cals(cals, cpar)                       # same cameras, ideal flat layers
    px = rig.project(points)                                   # (N, n_cams, 2) pixels, NaN outside the image
    rig2 = rig.with_(glass_thickness=7.0, tilt_front=[(0.01, 0)] * 4)   # a deliberately different truth

``project`` of an ideal rig agrees with ``openptv2.loco.project_pixels`` to ~1e-6 px (a test): that checks both this
tracer and the radial-shift multimedia model of the library.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from openptv2.algorithms.parameters import MmNp


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _refract(u: np.ndarray, n: np.ndarray, eta: float) -> np.ndarray:
    """Vector Snell's law: unit directions ``u`` (N, 3) crossing a plane of unit normal ``n`` (N, 3) from the medium
    of index n_a into n_b, ``eta = n_a / n_b``. Total reflection returns NaN."""
    cosi = -np.sum(n * u, axis=1, keepdims=True)
    n = np.where(cosi < 0, -n, n)
    cosi = np.abs(cosi)
    k = 1.0 - eta**2 * (1.0 - cosi**2)
    cost = np.sqrt(np.where(k >= 0, k, np.nan))
    return eta * u + (eta * cosi - cost) * n


def _hit(origin: np.ndarray, u: np.ndarray, n: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Point where the rays (origin, u) meet the planes through ``p0`` with normal ``n`` (all (N, 3))."""
    t = np.sum((p0 - origin) * n, axis=1) / np.sum(u * n, axis=1)
    return origin + t[:, None] * u


def _tilted(normal: np.ndarray, tilt: tuple[float, float]) -> np.ndarray:
    """``normal`` (3,) rotated by small angles ``tilt`` (rad) about two axes perpendicular to it."""
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(n, a))
    e2 = np.cross(n, e1)
    return _unit(n + tilt[0] * e1 + tilt[1] * e2)


def air_direction(
    cam: np.ndarray,
    pts: np.ndarray,
    ghat: np.ndarray,
    s_front: float,
    thickness: float,
    n_air: float,
    n_glass: float,
    n_water: float,
    normal_back: np.ndarray,
    normal_front: np.ndarray,
    n_iter: int = 12,
) -> np.ndarray:
    """Unit direction, in the air at the camera, of the ray that reaches each point of ``pts`` (N, 3) through the glass.

    The glass lies between the planes through ``s_front * ghat`` (water side, normal ``normal_front``) and
    ``(s_front + thickness) * ghat`` (air side, normal ``normal_back``). The direction is found by Gauss-Newton on the
    perpendicular miss of the refracted ray (2 unknowns per point)."""
    pts = np.asarray(pts, float)
    n_pts = len(pts)
    cam = np.broadcast_to(np.asarray(cam, float), (n_pts, 3))
    a0 = np.broadcast_to((s_front + thickness) * ghat, (n_pts, 3))
    b0 = np.broadcast_to(s_front * ghat, (n_pts, 3))
    nb = np.broadcast_to(normal_back, (n_pts, 3))
    nf = np.broadcast_to(normal_front, (n_pts, 3))

    def miss(u):
        pa = _hit(cam, u, nb, a0)
        ug = _refract(u, nb, n_air / n_glass)
        pb = _hit(pa, ug, nf, b0)
        uw = _refract(ug, nf, n_glass / n_water)
        d = pts - pb
        return d - np.sum(d * uw, axis=1, keepdims=True) * uw

    u0 = _unit(pts - cam)
    # two directions perpendicular to the start direction
    a = np.where(np.abs(u0[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = _unit(np.cross(u0, a))
    e2 = np.cross(u0, e1)
    x = np.zeros((n_pts, 2))
    h = 1e-7

    def direction(x):
        return _unit(u0 + x[:, :1] * e1 + x[:, 1:] * e2)

    for _ in range(n_iter):
        u = direction(x)
        r = miss(u)
        j0 = (miss(direction(x + np.array([h, 0]))) - r) / h
        j1 = (miss(direction(x + np.array([0, h]))) - r) / h
        # least squares of J dx = -r with J = [j0 j1] (3 x 2) per point
        a11 = np.sum(j0 * j0, 1)
        a12 = np.sum(j0 * j1, 1)
        a22 = np.sum(j1 * j1, 1)
        b1 = -np.sum(j0 * r, 1)
        b2 = -np.sum(j1 * r, 1)
        det = a11 * a22 - a12**2
        det = np.where(np.abs(det) < 1e-30, np.nan, det)
        x = x + np.stack([(a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det], axis=1)
    return direction(x)


@dataclass
class TruthRig:
    """The true imaging of a rig: cameras (``cals``: exterior, interior, lens distortion, glass vector) and the layers."""

    cals: list
    cpar: Any
    n_air: float = 1.0
    n_glass: float = 1.46
    n_water: float = 1.33
    glass_thickness: float = 6.0
    #: per camera (rad): rotation of the water-side / air-side glass surface normal away from the glass vector
    tilt_front: list = field(default_factory=list)
    tilt_back: list = field(default_factory=list)
    #: extra lens terms the calibration model does not fit: per camera dict(k3=, scx=, she=, ...) added to ``added_par``
    note: str = "ideal flat layers"
    vpar: Any = None                  # volume / epipolar parameters to match this rig's blobs with (set by ``cavity_rig``)

    @classmethod
    def from_cals(cls, cals: list, cpar: Any) -> "TruthRig":
        """A rig with the cameras and layers of an existing calibration (``cpar.mm`` indices and thickness)."""
        mm = cpar.mm
        n = len(cals)
        return cls(
            [copy.deepcopy(c) for c in cals], cpar, float(mm.n1), float(np.atleast_1d(mm.n2)[0]), float(mm.n3),
            float(np.atleast_1d(mm.d)[0]), [(0.0, 0.0)] * n, [(0.0, 0.0)] * n,
        )

    def with_(self, **kw) -> "TruthRig":
        """A copy with some fields replaced (``cals`` and the tilt lists are copied first)."""
        out = replace(self, cals=[copy.deepcopy(c) for c in self.cals], tilt_front=list(self.tilt_front),
                      tilt_back=list(self.tilt_back))
        for k, v in kw.items():
            setattr(out, k, v)
        return out

    def model_cpar(self) -> Any:
        """A copy of ``cpar`` whose multimedia indices and thickness are this rig's: what a *perfect* calibration uses."""
        cp = copy.copy(self.cpar)
        cp.mm = MmNp(1, self.n_air, [self.n_glass], [self.glass_thickness, 0.0, 0.0], self.n_water)
        return cp

    def _vacuum(self) -> MmNp:
        return MmNp(1, 1.0, [1.0], [0.0, 0.0, 0.0], 1.0)

    def ray_path(self, c: int, pts: np.ndarray) -> np.ndarray:
        """The traced ray of camera ``c`` to each point: (N, 4, 3) = camera centre, point on the air-side glass surface,
        point on the water-side surface, the particle. For drawing and for checking the tracer by eye."""
        cal = self.cals[c]
        pts = np.asarray(pts, float).reshape(-1, 3)
        gvec = np.asarray(cal.get_glass_vec(), float)
        s_front = float(np.linalg.norm(gvec))
        ghat = gvec / s_front
        nf, nb = _tilted(ghat, self.tilt_front[c]), _tilted(ghat, self.tilt_back[c])
        cam = np.asarray(cal.get_pos(), float)
        u = air_direction(cam, pts, ghat, s_front, self.glass_thickness, self.n_air, self.n_glass, self.n_water, nb, nf)
        n = len(pts)
        camb = np.broadcast_to(cam, (n, 3))
        pa = _hit(camb, u, np.broadcast_to(nb, (n, 3)), np.broadcast_to((s_front + self.glass_thickness) * ghat, (n, 3)))
        ug = _refract(u, np.broadcast_to(nb, (n, 3)), self.n_air / self.n_glass)
        pb = _hit(pa, ug, np.broadcast_to(nf, (n, 3)), np.broadcast_to(s_front * ghat, (n, 3)))
        return np.stack([camb, pa, pb, pts], axis=1)

    def project_cam(self, c: int, pts: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) of ``pts`` in camera ``c`` (no image-bounds check)."""
        from openptv2.algorithms.imgcoord import img_coord_batch

        cal = self.cals[c]
        pts = np.asarray(pts, float).reshape(-1, 3)
        gvec = np.asarray(cal.get_glass_vec(), float)
        s_front = float(np.linalg.norm(gvec))
        ghat = gvec / s_front
        nf = _tilted(ghat, self.tilt_front[c])
        nb = _tilted(ghat, self.tilt_back[c])
        cam = np.asarray(cal.get_pos(), float)
        u = air_direction(cam, pts, ghat, s_front, self.glass_thickness, self.n_air, self.n_glass, self.n_water, nb, nf)
        eff = cam + u * np.linalg.norm(pts - cam, axis=1, keepdims=True)
        m = np.asarray(img_coord_batch(np.ascontiguousarray(eff), cal, self._vacuum()), float)
        cp = self.cpar
        return np.column_stack([m[:, 0] / cp.pix_x + cp.imx / 2, cp.imy / 2 - m[:, 1] / cp.pix_y])

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Pixels (N, n_cams, 2) of ``pts`` (mm); NaN where a point falls outside the sensor."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        out = np.stack([self.project_cam(c, pts) for c in range(len(self.cals))], axis=1)
        cp = self.cpar
        bad = (out[..., 0] < 0) | (out[..., 0] > cp.imx) | (out[..., 1] < 0) | (out[..., 1] > cp.imy) | ~np.isfinite(out[..., 0])
        out[bad] = np.nan
        return out

    def observe(self, pts: np.ndarray, noise_px: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        """``project`` plus Gaussian centroid noise (px)."""
        px = self.project(pts)
        if noise_px > 0:
            rng = rng or np.random.default_rng(0)
            px = px + rng.normal(0, noise_px, px.shape)
        return px





# --------------------------------------------------------------------------------------------------------------------
# a cavity-like rig (cameras on both sides of a water cell), rigid transforms of a rig, and synthetic run stores


def _angles_for_matrix(target: np.ndarray, cal: Any) -> np.ndarray:
    """Angles (omega, phi, kappa) of the library's convention whose rotation matrix is ``target`` (multi-start least squares)."""
    from scipy.optimize import least_squares

    trial = copy.deepcopy(cal)
    rng = np.random.default_rng(0)
    best = None
    for k in range(40):
        a0 = np.asarray(cal.get_angles(), float) if k == 0 else rng.uniform(-np.pi, np.pi, 3)

        def resid(a):
            trial.set_angles(a)
            return (trial.get_rotation_matrix() - target).ravel()

        sol = least_squares(resid, a0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < 1e-24:
            break
    return best.x


def transformed(rig: "TruthRig", rot: np.ndarray, shift: np.ndarray) -> "TruthRig":
    """The same physical rig moved rigidly in the laboratory frame: every point ``x`` becomes ``rot @ x + shift``. Cameras and
    glass planes move with it (the glass vector is the foot of the plane from the origin, so its length changes with a shift).
    ``project`` of the new rig at the moved points equals ``project`` of the old rig at the old points."""
    rot, shift = np.asarray(rot, float), np.asarray(shift, float)
    out = rig.with_()
    for cal in out.cals:
        m = np.asarray(cal.get_rotation_matrix(), float)
        pos = np.asarray(cal.get_pos(), float)
        g = np.asarray(cal.get_glass_vec(), float)
        ghat = g / np.linalg.norm(g)
        new_hat = rot @ ghat
        foot = new_hat * float(new_hat @ (rot @ g + shift))
        cal.set_pos(rot @ pos + shift)
        cal.set_angles(_angles_for_matrix(rot @ m, cal))      # the library's matrix has the camera axes as columns
        cal.set_glass_vec(foot)
        cal.invalidate_mmlut()
    out.note = f"{rig.note}; moved"
    return out


def cavity_rig(imx: int = 1280, imy: int = 1024, half_width: float = 125.0, truth_lens: bool = True) -> "TruthRig":
    """A water cell seen through two flat windows by two cameras on each side (geometry of ``test_data/test_cavity_small``): cameras at
    z about -575 with the glass vector (0, 0, -half_width) and at z about +580 with (0, 0, +half_width); glass 6 mm, n = 1 / 1.46 / 1.33;
    a 1280 x 1024 sensor of 12 um pixels, ``cc`` 70 mm. With ``truth_lens`` each camera has its own small principal-point offset and
    lens terms (k1, k2, p1, p2) of the size the real refinement found, so a calibration that leaves them out is wrong in a known way."""
    from pathlib import Path

    from openptv2.recorrespond import load_parameters

    yml = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"
    cpar, _vpar, cals, _ = load_parameters(yml)
    cpar = copy.copy(cpar)
    cpar.imx, cpar.imy = imx, imy
    cals = [copy.deepcopy(c) for c in cals]
    lens = [(-2.5e-4, 3e-6, 1.5e-4, -2.0e-4, 0.8, -0.3), (-1.5e-4, 2e-6, -1.0e-4, -2.5e-4, -0.9, 0.4),
            (-1.0e-4, 2e-6, 2.0e-5, -3.0e-4, 0.4, 0.6), (-2.0e-4, 2e-6, -3.5e-4, -4.0e-4, -0.7, -0.5)]
    for cal, (k1, k2, p1, p2, xh, yh) in zip(cals, lens):
        g = np.asarray(cal.get_glass_vec(), float)
        cal.set_glass_vec(g / np.linalg.norm(g) * half_width)
        if truth_lens:
            cal.added_par.k1, cal.added_par.k2, cal.added_par.p1, cal.added_par.p2 = k1, k2, p1, p2
            cal.int_par.xh, cal.int_par.yh = xh, yh
        cal.invalidate_mmlut()
    rig = TruthRig.from_cals(cals, cpar)
    rig.cpar = cpar
    rig.vpar = load_parameters(yml.parents[1] / "test_cavity" / "parameters_Run1.yaml")[1]       # volume +-40 mm, eps0 0.04
    rig.note = "cavity-like rig, two cameras per side"
    return rig


def synthetic_run(
    rig: "TruthRig",
    path: Any,
    frames: list[int],
    n_particles: int = 900,
    half_size: float = 30.0,
    noise_px: float = 0.15,
    detect_prob: float = 0.8,
    seed: int = 0,
    centre: tuple = (0.0, 0.0, 0.0),
    half_z: float | None = None,
    gauss_xy: float | None = None,
) -> dict:
    """A run store of ``frames`` with blobs of ``n_particles`` random tracers in a cube (``half_size`` mm) seen through ``rig``: each
    camera detects a particle with probability ``detect_prob`` and adds Gaussian centroid noise. ``half_z`` limits the depth separately,
    ``gauss_xy`` (mm) replaces the lateral uniform distribution by a Gaussian. The target number ``pnr`` is the
    particle index, so a matched quadruplet is right when all its blobs share it. Nothing is matched yet.
    Returns ``{frame: positions (n, 3)}`` (the truth)."""
    from openptv2.storage import RunStore

    rng = np.random.default_rng(seed)
    st = RunStore(str(path), mode="a")
    truth = {}
    for f in frames:
        pts = np.asarray(centre) + rng.uniform(-half_size, half_size, (n_particles, 3))
        if half_z is not None:
            pts[:, 2] = centre[2] + rng.uniform(-half_z, half_z, n_particles)
        if gauss_xy is not None:                          # a cloud concentrated at the centre, sparse towards the sensor edges
            pts[:, :2] = np.asarray(centre)[:2] + rng.normal(0, gauss_xy, (n_particles, 2))
        px = rig.observe(pts, noise_px, rng)
        seen = np.isfinite(px[:, :, 0]) & (rng.random((n_particles, len(rig.cals))) < detect_prob)
        for c in range(len(rig.cals)):
            rows = np.flatnonzero(seen[:, c])
            tg = np.zeros((len(rows), 8))
            tg[:, 0] = rows
            tg[:, 1:3] = px[rows, c]
            tg[:, 3], tg[:, 4], tg[:, 5], tg[:, 6], tg[:, 7] = 9, 3, 3, 400, -1
            st.write_targets(c, f, tg[np.argsort(tg[:, 2], kind="stable")])         # sorted by y like the detector's output
        truth[f] = pts
    return truth


def score_matching(store: Any, frames: list[int], truth: dict, n_cams: int = 4) -> dict:
    """Truth-based scores of the correspondences stored for ``frames``: matched points, right quadruplets (all blobs of one
    particle), wrong ones (ghosts), the recall of the particles seen by all cameras, and the median 3D error of the right ones."""
    right = wrong = total = possible = 0
    err = []
    for f in frames:
        pos, ids = store.read_correspondences(f)
        pos, ids = np.asarray(pos, float), np.asarray(ids)[:, :n_cams]
        tg = [np.asarray(store.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(n_cams)]
        pnr = np.full(ids.shape, -1.0)
        for c in range(n_cams):
            m = ids[:, c] >= 0
            pnr[m, c] = tg[c][ids[m, c], 0]
        quad = (ids >= 0).all(axis=1)
        good = quad & (pnr == pnr[:, :1]).all(axis=1)
        total += int(quad.sum())
        right += int(good.sum())
        wrong += int((quad & ~good).sum())
        seen_all = np.ones(len(truth[f]), bool)
        for c in range(n_cams):
            seen_all &= np.isin(np.arange(len(truth[f])), tg[c][:, 0].astype(int))
        possible += int(seen_all.sum())
        if good.any():
            err.append(np.linalg.norm(pos[good] - truth[f][pnr[good, 0].astype(int)], axis=1))
    e = np.concatenate(err) if err else np.zeros(0)
    return dict(quads=total, right=right, ghosts=wrong, possible=possible, recall=right / max(possible, 1),
                ghost_share=wrong / max(total, 1), err_med=float(np.median(e)) if len(e) else float("nan"))


__all__ = ["TruthRig", "air_direction", "cavity_rig", "score_matching", "synthetic_run", "transformed"]

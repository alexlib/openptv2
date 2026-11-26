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


__all__ = ["TruthRig", "air_direction"]

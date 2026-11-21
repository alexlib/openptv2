"""Joint self-calibration from tracer particles: camera parameters and 3D positions fitted together.

A bundle adjustment over many tracked particles: every observation (a blob centre in a camera) is compared with the
projection of the particle's 3D position, and the camera parameters (exterior orientation, principal point, radial and
decentering distortion) and the positions are adjusted together. The gauge (a global shift, rotation and scale of the
whole system) is fixed by holding the exterior of one camera and by soft priors on all parameters, so the world frame
does not drift. Unlike a leave-one-camera-out correction of each camera by its own residual field, no camera is
corrected twice for the same error.

The residual of tracer particles is a smooth, repeatable function of image position and depth (see
``scripts/loco_map.py``): a calibration error. A plate calibration constrains the depth direction weakly; tracers fill
the whole volume.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from openptv2.loco import project_pixels

# parameter groups: name -> (getter, setter, prior sigma)
_EXT_SIGMA = (0.5, 0.5, 0.5, 0.002, 0.002, 0.002)          # mm, rad


def _ext_get(c):
    return np.concatenate([c.get_pos(), c.get_angles()])


def _ext_set(c, v):
    c.set_pos(v[:3])
    c.set_angles(v[3:6])


def _int_get(c, names):
    return np.array([getattr(c.int_par, n) for n in names])


def _int_set(c, v, names):
    for n, x in zip(names, v):
        setattr(c.int_par, n, float(x))


def _add_get(c, names):
    return np.array([getattr(c.added_par, n) for n in names])


def _add_set(c, v, names):
    for n, x in zip(names, v):
        setattr(c.added_par, n, float(x))


@dataclass
class SelfCalResult:
    cals: list
    rms_before_px: float
    rms_after_px: float
    n_points: int
    n_obs: int
    n_params: int
    change: dict
    success: bool
    nfev: int = 0


def joint_self_calibrate(
    obs_px: np.ndarray,
    x0: np.ndarray,
    cals: list,
    cpar: Any,
    *,
    hold_cam: int = 0,
    free_ext: bool = True,
    free_pp: bool = True,
    free_cc: bool = False,
    free_radial: tuple = ("k1", "k2"),
    free_decentering: bool = True,
    pp_sigma_mm: float = 0.05,
    ext_sigma_scale: float = 1.0,
    cc_sigma_mm: float = 0.1,
    k_sigma: float = 0.02,
    p_sigma: float = 0.002,
    huber_px: float = 0.7,
    max_nfev: int = 25,
    verbose: int = 0,
) -> SelfCalResult:
    """Fit camera parameters and 3D positions to ``obs_px`` ``(N, C, 2)`` (NaN where a camera has no blob), starting from
    positions ``x0`` ``(N, 3)`` mm. Returns new calibrations (the inputs are not modified)."""
    obs_px = np.asarray(obs_px, float)
    n, n_cams, _ = obs_px.shape
    cur = [copy.deepcopy(c) for c in cals]
    valid = np.isfinite(obs_px[:, :, 0])

    # parameter layout: per camera a list of (slice, getter, setter, sigma)
    layout = []
    x_init = []
    off = 0
    for c in range(n_cams):
        groups = []
        if free_ext and c != hold_cam:
            groups.append((_ext_get, _ext_set, np.array(_EXT_SIGMA) * ext_sigma_scale))
        if free_pp:
            groups.append((lambda cc: _int_get(cc, ("xh", "yh")), lambda cc, v: _int_set(cc, v, ("xh", "yh")),
                           np.full(2, pp_sigma_mm)))
        if free_cc:
            groups.append((lambda cc: _int_get(cc, ("cc",)), lambda cc, v: _int_set(cc, v, ("cc",)),
                           np.full(1, cc_sigma_mm)))
        if free_radial:
            nm = tuple(free_radial)
            groups.append((lambda cc, nm=nm: _add_get(cc, nm), lambda cc, v, nm=nm: _add_set(cc, v, nm),
                           np.full(len(nm), k_sigma)))
        if free_decentering:
            groups.append((lambda cc: _add_get(cc, ("p1", "p2")), lambda cc, v: _add_set(cc, v, ("p1", "p2")),
                           np.full(2, p_sigma)))
        cam_groups = []
        for get, setf, sig in groups:
            v = get(cur[c])
            cam_groups.append((slice(off, off + len(v)), get, setf, sig))
            x_init.append(v)
            off += len(v)
        layout.append(cam_groups)
    n_par = off
    x_cam0 = np.concatenate(x_init)
    sig_all = np.concatenate([g[3] for cg in layout for g in cg])
    x_start = np.concatenate([x_cam0, np.asarray(x0, float).ravel()])

    rows_of = [np.flatnonzero(valid[:, c]) for c in range(n_cams)]
    n_res_obs = 2 * sum(len(r) for r in rows_of)

    def apply(x):
        for c in range(n_cams):
            for sl, _g, setf, _s in layout[c]:
                setf(cur[c], x[sl])

    def resid(x):
        apply(x)
        pts = x[n_par:].reshape(n, 3)
        out = []
        for c in range(n_cams):
            r = rows_of[c]
            if len(r):
                out.append((project_pixels(pts[r], cur[c], cpar) - obs_px[r, c]).ravel())
        out.append((x[:n_par] - x_cam0) / sig_all)
        return np.nan_to_num(np.concatenate(out), nan=1e3, posinf=1e3, neginf=-1e3)

    # sparsity: an observation row depends on its camera's parameters and its own point
    ri, ci = [], []
    base = 0
    for c in range(n_cams):
        r = rows_of[c]
        cols_cam = np.concatenate([np.arange(g[0].start, g[0].stop) for g in layout[c]]) if layout[c] else np.zeros(0, int)
        for k, pt in enumerate(r):
            rr = base + 2 * k
            for dr in (0, 1):
                ri.extend([rr + dr] * (len(cols_cam) + 3))
                ci.extend(cols_cam.tolist() + [n_par + 3 * pt, n_par + 3 * pt + 1, n_par + 3 * pt + 2])
        base += 2 * len(r)
    ri.extend((n_res_obs + np.arange(n_par)).tolist())
    ci.extend(np.arange(n_par).tolist())
    sp = sparse.coo_matrix((np.ones(len(ri), np.int8), (ri, ci)), shape=(n_res_obs + n_par, n_par + 3 * n)).tocsr()

    r0 = resid(x_start)[:n_res_obs]
    sol = least_squares(resid, x_start, jac_sparsity=sp, method="trf", x_scale="jac", loss="huber", f_scale=huber_px,
                        max_nfev=max_nfev, tr_solver="lsmr", verbose=verbose)
    r1 = resid(sol.x)[:n_res_obs]
    apply(sol.x)
    names = ("ext", "pp", "radial", "decentering")
    change = {}
    for c in range(n_cams):
        for sl, _g, _s, _sg in layout[c]:
            change.setdefault(c + 1, []).append(np.round(sol.x[sl] - x_cam0[sl], 5).tolist())
    del names
    return SelfCalResult(cur, float(np.sqrt(np.mean(r0**2))), float(np.sqrt(np.mean(r1**2))), n, int(valid.sum()),
                         n_par, change, bool(sol.success), int(sol.nfev))

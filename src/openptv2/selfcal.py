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

Tracers constrain consistency, not scale: a uniform stretch of the volume reprojects equally well. Self-calibration is
therefore a tuning step that comes *after* the regular calibration (plate / target body), which supplies the starting
cameras, and it is anchored to metric units by a known object seen by the same cameras: a ``KnownBody`` (points with known
world coordinates, e.g. the calibration target) and/or a ``Dumbbell`` (two points a known distance apart, moved through
the volume). Both enter the same fit; with neither, the result is only defined up to a scale.
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
class KnownBody:
    """Points with known world coordinates (mm) and their pixel observations: ``xyz`` ``(M, 3)``, ``obs_px`` ``(M, C, 2)``."""

    xyz: np.ndarray
    obs_px: np.ndarray


@dataclass
class Dumbbell:
    """Pairs of points a known distance apart: ``obs_px`` ``(K, 2, C, 2)`` pixels (NaN where a camera has no blob),
    ``length_mm``; ``xyz0`` ``(K, 2, 3)`` optional starting positions (else triangulated from the cameras)."""

    obs_px: np.ndarray
    length_mm: float
    xyz0: np.ndarray | None = None


def triangulate_pixels(obs_px: np.ndarray, cals: list, cpar: Any, vpar: Any) -> np.ndarray:
    """Least-squares 3D positions (mm) from pixel observations ``(N, C, 2)`` (NaN = camera unused)."""
    from openptv2.loco import pixels_to_flat
    from openptv2.orientation import point_positions

    flat = np.stack([pixels_to_flat(obs_px[:, c], cals[c], cpar) for c in range(len(cals))], axis=1)
    return np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)


def _detect_calibration_targets(base: Any, cam: int, cpar: Any) -> list:
    """Targets detected in camera ``cam``'s calibration image. A calibration image that already has the camera's size is
    used as it is; a larger (multiplexed splitter) frame is split first."""
    from imageio.v3 import imread
    from skimage.color import rgb2gray
    from skimage.util import img_as_ubyte

    from openptv2.autocalibration import _cam_view, _tpar_from_dataset, cam_files
    from openptv2.image_processing import preprocess_image
    from openptv2.segmentation import target_recognition

    img_path, _ori, _addpar = cam_files(base, cam)
    tpar = _tpar_from_dataset(base)
    if tpar is None or not img_path.exists():
        return []
    raw = imread(img_path)
    if raw.shape[:2] != (cpar.imy, cpar.imx):
        raw = _cam_view(base, cam, raw)
    if raw.ndim == 3:
        raw = rgb2gray(raw)
    hp = preprocess_image(img_as_ubyte(raw), cpar.hp_flag or 1, cpar, 25)
    found = target_recognition(hp, tpar, cam, cpar)
    if not found:
        return []
    if hasattr(found, "sort_y"):
        found.sort_y()
    return list(found)


def known_body_from_calibration(base: Any, cals: list, cpar: Any, *, eps: int = 15, min_cams: int = 2,
                                max_px: float = 3.0) -> KnownBody:
    """The regular calibration's target body as a scale anchor: the calblock's known 3D points matched (with ``sortgrid``
    and the current cameras) to the targets detected in each camera's calibration image. ``base`` is the experiment folder
    holding ``cal/`` and the parameter YAML. Points seen by fewer than ``min_cams`` cameras are dropped."""
    from pathlib import Path

    from openptv2.algorithms.sortgrid import read_calblock, sortgrid
    from openptv2.autocalibration import resolve_calblock

    base = Path(base)
    fix, nfix = read_calblock(str(resolve_calblock(base)))
    fix = np.asarray(fix, float)
    obs = np.full((nfix, len(cals), 2), np.nan)
    for c, cal in enumerate(cals):
        pix = _detect_calibration_targets(base, c, cpar)
        if not pix:
            continue
        sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), max(15, eps), pix)
        for i, t in enumerate(sorted_pix):
            if t.pnr >= 0:
                obs[i, c] = (t.x, t.y)
    for c, cal in enumerate(cals):
        seen = np.isfinite(obs[:, c, 0])
        if seen.any():
            err = np.linalg.norm(project_pixels(fix[:nfix][seen], cal, cpar) - obs[seen, c], axis=1)
            idx = np.flatnonzero(seen)[err > max_px]
            obs[idx, c] = np.nan
    keep = np.isfinite(obs[:, :, 0]).sum(axis=1) >= min_cams
    return KnownBody(fix[:nfix][keep], obs[keep])


def dumbbell_from_store(store: Any, num_cams: int, length_mm: float, *, first: int, last: int, step: int = 1) -> Dumbbell:
    """Dumbbell frames from a run store: every frame with exactly two targets in each camera gives one pair. As in the
    standalone dumbbell calibration, the two targets are assumed to be listed in the same order (end A, end B) in every
    camera."""
    rows = []
    for frame in range(first, last + 1, step):
        per_cam = []
        for cam in range(num_cams):
            try:
                targs = list(store.read_targets(cam, frame))
            except Exception:
                targs = []
            if len(targs) != 2:
                break
            per_cam.append([t.pos() for t in targs])
        else:
            rows.append(np.asarray(per_cam, float).transpose(1, 0, 2))          # (2, C, 2)
    if not rows:
        raise ValueError("no frame with exactly 2 targets in every camera")
    return Dumbbell(np.asarray(rows), float(length_mm))


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
    length_rms_mm_before: float | None = None      # dumbbell length error, if a Dumbbell was given
    length_rms_mm_after: float | None = None
    body_rms_mm_after: float | None = None         # known-body 3D misfit (triangulated vs known), if a KnownBody was given
    multimedia: dict | None = None                 # fitted glass thickness / indices when ``free_multimedia`` was used
    cpar: Any = None                               # ``cpar`` with the fitted multimedia (the input one when none was free)


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
    edge_px: float | None = None,
    free_multimedia: tuple = (),
    free_glass: bool = False,
    mm_sigma: dict | None = None,
    glass_sigma_mm: float = 0.5,
    pp_sigma_mm: float = 0.05,
    ext_sigma_scale: float = 1.0,
    cc_sigma_mm: float = 0.1,
    k_sigma: float = 0.02,
    p_sigma: float = 0.002,
    huber_px: float = 0.7,
    anneal: tuple = (8.0, 3.0, 1.0),
    max_nfev: int = 25,
    verbose: int = 0,
    body: KnownBody | None = None,
    dumbbell: Dumbbell | None = None,
    vpar: Any = None,
    body_sigma_mm: float = 0.02,
    length_sigma_mm: float = 0.05,
) -> SelfCalResult:
    """Fit camera parameters and 3D positions to ``obs_px`` ``(N, C, 2)`` (NaN where a camera has no blob), starting from
    positions ``x0`` ``(N, 3)`` mm. ``body`` / ``dumbbell`` anchor the scale (see the module docstring; ``vpar`` is needed to
    triangulate dumbbell starts when ``Dumbbell.xyz0`` is None). Returns new calibrations (the inputs are not modified)."""
    obs_px = np.asarray(obs_px, float)
    x0 = np.asarray(x0, float)
    n_tr = len(obs_px)
    n_cams = obs_px.shape[1]
    body_idx = db_a = db_b = None
    extra_obs, extra_x0 = [], []
    if body is not None:
        body_idx = n_tr + np.arange(len(body.xyz))
        extra_obs.append(np.asarray(body.obs_px, float))
        extra_x0.append(np.asarray(body.xyz, float))
    if dumbbell is not None:
        d_obs = np.asarray(dumbbell.obs_px, float)                      # (K, 2, C, 2)
        k = len(d_obs)
        flat_obs = d_obs.reshape(2 * k, n_cams, 2)
        xyz0 = dumbbell.xyz0
        if xyz0 is None:
            if vpar is None:
                raise ValueError("vpar is required to triangulate dumbbell starting positions")
            xyz0 = triangulate_pixels(flat_obs, cals, cpar, vpar)
        start = n_tr + (0 if body is None else len(body.xyz))
        db_a = start + 2 * np.arange(k)
        db_b = db_a + 1
        extra_obs.append(flat_obs)
        extra_x0.append(np.asarray(xyz0, float).reshape(2 * k, 3))
    if extra_obs:
        obs_px = np.concatenate([obs_px] + extra_obs)
        x0 = np.concatenate([x0] + extra_x0)
    n = len(obs_px)
    cur = [copy.deepcopy(c) for c in cals]
    valid = np.isfinite(obs_px[:, :, 0])

    # Priors on the lens terms in physical units: the image shift (px) that one standard deviation of each term causes at the sensor
    # corner. The distortion acts on the image-plane radius in mm (k1 ~ r^3, k2 ~ r^5, p ~ r^2), so a fixed number for k2 means
    # thousands of pixels at the periphery where the data are sparse: the fit then sets k2 from the centre and extrapolates.
    if edge_px is not None:
        r_corner = float(np.hypot(cpar.imx * cpar.pix_x / 2.0, cpar.imy * cpar.pix_y / 2.0))
        shift = float(edge_px) * float(cpar.pix_x)
        k_sig_by = {"k1": shift / r_corner**3, "k2": shift / r_corner**5, "k3": shift / r_corner**7}
        p_sigma = shift / r_corner**2
    else:
        k_sig_by = None

    # multimedia parameters shared by all cameras (glass thickness, indices): a private copy of cpar carries them
    import copy as _copy

    from openptv2.algorithms.parameters import MmNp

    cpar = _copy.copy(cpar)
    mm0 = cpar.mm
    cpar.mm = MmNp(int(mm0.nlay), float(mm0.n1), np.array(mm0.n2, float), np.array(mm0.d, float), float(mm0.n3))
    mm_names = tuple(free_multimedia)
    bad = [m for m in mm_names if m not in ("d", "n2", "n3")]
    if bad:
        raise ValueError(f"free_multimedia takes 'd', 'n2', 'n3'; got {bad}")
    mm_sig = {"d": 0.5, "n2": 0.02, "n3": 0.02, **(mm_sigma or {})}

    def _mm_get():
        return np.array([float(np.atleast_1d(cpar.mm.d)[0]) if m == "d" else float(np.atleast_1d(cpar.mm.n2)[0]) if m == "n2"
                         else float(cpar.mm.n3) for m in mm_names])

    def _mm_set(v):
        for m, x in zip(mm_names, v):
            if m == "d":
                cpar.mm.d[0] = float(x)
            elif m == "n2":
                cpar.mm.n2[0] = float(x)
            else:
                cpar.mm.n3 = float(x)

    def _glass_get(cc):
        return np.array([float(np.linalg.norm(cc.get_glass_vec()))])

    def _glass_set(cc, v):
        g = np.asarray(cc.get_glass_vec(), float)
        cc.set_glass_vec(g * float(v[0]) / np.linalg.norm(g))

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
                           np.array([k_sig_by[n] for n in nm]) if k_sig_by else np.full(len(nm), k_sigma)))
        if free_decentering:
            groups.append((lambda cc: _add_get(cc, ("p1", "p2")), lambda cc, v: _add_set(cc, v, ("p1", "p2")),
                           np.full(2, p_sigma)))
        if free_glass:
            groups.append((_glass_get, _glass_set, np.full(1, glass_sigma_mm)))
        cam_groups = []
        for get, setf, sig in groups:
            v = get(cur[c])
            cam_groups.append((slice(off, off + len(v)), get, setf, sig))
            x_init.append(v)
            off += len(v)
        layout.append(cam_groups)
    mm_cols = np.arange(off, off + len(mm_names))
    if mm_names:
        x_init.append(_mm_get())
        off += len(mm_names)
    n_par = off
    x_cam0 = np.concatenate(x_init)
    sig_all = np.concatenate([g[3] for cg in layout for g in cg] + ([np.array([mm_sig[m] for m in mm_names])] if mm_names else []))
    x_start = np.concatenate([x_cam0, np.asarray(x0, float).ravel()])

    rows_of = [np.flatnonzero(valid[:, c]) for c in range(n_cams)]
    n_res_obs = 2 * sum(len(r) for r in rows_of)

    def apply(x):
        for c in range(n_cams):
            for sl, _g, setf, _s in layout[c]:
                setf(cur[c], x[sl])
        if mm_names:
            _mm_set(x[mm_cols])

    def resid(x):
        apply(x)
        pts = x[n_par:].reshape(n, 3)
        out = []
        for c in range(n_cams):
            r = rows_of[c]
            if len(r):
                out.append((project_pixels(pts[r], cur[c], cpar) - obs_px[r, c]).ravel())
        out.append((x[:n_par] - x_cam0) / sig_all)
        if body_idx is not None:
            out.append(((pts[body_idx] - body.xyz) / body_sigma_mm).ravel())
        if db_a is not None:
            out.append((np.linalg.norm(pts[db_a] - pts[db_b], axis=1) - dumbbell.length_mm) / length_sigma_mm)
        return np.nan_to_num(np.concatenate(out), nan=1e3, posinf=1e3, neginf=-1e3)

    # sparsity: an observation row depends on its camera's parameters and its own point
    ri, ci = [], []
    base = 0
    for c in range(n_cams):
        r = rows_of[c]
        cols_cam = np.concatenate([np.arange(g[0].start, g[0].stop) for g in layout[c]]) if layout[c] else np.zeros(0, int)
        cols_cam = np.concatenate([cols_cam, mm_cols])
        for k, pt in enumerate(r):
            rr = base + 2 * k
            for dr in (0, 1):
                ri.extend([rr + dr] * (len(cols_cam) + 3))
                ci.extend(cols_cam.tolist() + [n_par + 3 * pt, n_par + 3 * pt + 1, n_par + 3 * pt + 2])
        base += 2 * len(r)
    ri.extend((n_res_obs + np.arange(n_par)).tolist())
    ci.extend(np.arange(n_par).tolist())
    row = n_res_obs + n_par
    if body_idx is not None:
        for pt in body_idx:
            for d in range(3):
                ri.append(row)
                ci.append(n_par + 3 * int(pt) + d)
                row += 1
    if db_a is not None:
        for a_, b_ in zip(db_a, db_b):
            for d in range(3):
                ri.extend([row, row])
                ci.extend([n_par + 3 * int(a_) + d, n_par + 3 * int(b_) + d])
            row += 1
    sp = sparse.coo_matrix((np.ones(len(ri), np.int8), (ri, ci)), shape=(row, n_par + 3 * n)).tocsr()

    def length_rms(x):
        pts_ = x[n_par:].reshape(n, 3)
        return float(np.sqrt(np.mean((np.linalg.norm(pts_[db_a] - pts_[db_b], axis=1) - dumbbell.length_mm) ** 2)))

    len0 = length_rms(x_start) if db_a is not None else None
    r0 = resid(x_start)[:n_res_obs]
    # coarse to fine: with a large starting error every residual looks like an outlier to a tight Huber loss and the fit
    # stalls; start with a wide one (``anneal`` = multiples of ``huber_px``, the last is the final loss) and narrow it.
    x_cur, nfev_total = x_start, 0
    for k, mult in enumerate(anneal or (1.0,)):
        sol = least_squares(resid, x_cur, jac_sparsity=sp, method="trf", x_scale="jac", loss="huber", f_scale=huber_px * mult,
                            max_nfev=max_nfev, tr_solver="lsmr", verbose=verbose)
        x_cur, nfev_total = sol.x, nfev_total + int(sol.nfev)
    r1 = resid(sol.x)[:n_res_obs]
    apply(sol.x)
    names = ("ext", "pp", "radial", "decentering")
    change = {}
    for c in range(n_cams):
        for sl, _g, _s, _sg in layout[c]:
            change.setdefault(c + 1, []).append(np.round(sol.x[sl] - x_cam0[sl], 5).tolist())
    del names
    len1 = length_rms(sol.x) if db_a is not None else None
    body_rms = None
    if body_idx is not None:
        pts_ = sol.x[n_par:].reshape(n, 3)
        body_rms = float(np.sqrt(np.mean(np.sum((pts_[body_idx] - body.xyz) ** 2, axis=1))))
    return SelfCalResult(cur, float(np.sqrt(np.mean(r0**2))), float(np.sqrt(np.mean(r1**2))), n_tr, int(valid.sum()),
                         n_par, change, bool(sol.success), nfev_total, len0, len1, body_rms,
                         dict(zip(mm_names, [float(v) for v in _mm_get()])) if mm_names else None, cpar)

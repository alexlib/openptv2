"""Joint bundle adjustment of all cameras on calibration dots + real particles.

A calibration body is small and shallow; particles fill the whole measurement
volume. Per-camera fits on the body alone leave the cameras disagreeing far
from it (wide epipolar bands, few quadruplets). This fit couples the cameras:

* calibration dots (known 3D, from the body file) fix scale and position;
* 4-camera particles from a finished sequence run (3D unknown, solved for)
  tie the cameras together over the full volume.

All cameras' exterior, interior (cc, xh, yh) and distortion parameters are
fitted together with a robust loss, so a few ghost quadruplets or mismatched
dots do not pull the solution.

Usage (run the sequence step first so ``res/run.zarr`` holds targets and
correspondences):

    uv run python scripts/particle_bundle_adjust.py <case>/parameters_Run1.yaml \
        --frames 10000 10004 --free ext,cc,xh,yh,k1,k2,p1,p2,gz --max-px 2 --write

``--write`` saves the new ``.ori``/``.addpar`` (old ones backed up as
``*.pbabck``). Repeat sequence -> this script 2-3 times: a better calibration
finds more quadruplets, which constrain the next fit better. ``gz`` frees the
glass-wall distance; ``--prior``/``--tie-glass`` restrain parameters the data
barely separate. If the starting calibration is far off, first fit the body
alone: ``--max-px 0 --free ext --body-radius-px 20 --body-sigma-mm 0.001``.

test_cavity (2026-10-24): ~0.5 px particle reprojection (from ~1.3 px);
quadruplets per frame at eps0 0.04 ~440 (old calibration, same settings ~150).
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.orientation import point_positions
from openptv2.algorithms.trafo import metric_to_pixel_batch, pixel_to_metric_batch
from openptv2.storage import RunStore, find_existing_store
from openptv2.transforms import distorted_to_flat

# Per-camera parameter vector layout.
PARAMS = [
    "x0", "y0", "z0", "omega", "phi", "kappa",
    "cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2", "scx", "she", "gz",
]  # fmt: skip
GROUPS = {"ext": PARAMS[:6]}
# Typical magnitudes, so the solver works on well-scaled unknowns.
SCALE = dict(
    x0=1.0, y0=1.0, z0=1.0, omega=1e-3, phi=1e-3, kappa=1e-3,
    cc=0.1, xh=0.01, yh=0.01, k1=1e-4, k2=1e-6, k3=1e-8, p1=1e-4, p2=1e-4,
    scx=1e-4, she=1e-4, gz=1.0,
)  # fmt: skip


def cal_to_vec(cal: Calibration) -> np.ndarray:
    e, i, a = cal.ext_par, cal.int_par, cal.added_par
    return np.array(
        [e.x0, e.y0, e.z0, e.omega, e.phi, e.kappa, i.cc, i.xh, i.yh,
         a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she, cal.glass_par.vec_z],
        dtype=np.float64,
    )  # fmt: skip


def vec_to_cal(v: np.ndarray, cal: Calibration) -> None:
    e, i, a = cal.ext_par, cal.int_par, cal.added_par
    e.x0, e.y0, e.z0, e.omega, e.phi, e.kappa = (float(x) for x in v[:6])
    e.compute_rotation_matrix()
    i.cc, i.xh, i.yh = float(v[6]), float(v[7]), float(v[8])
    a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she = (float(x) for x in v[9:16])
    cal.glass_par.vec_z = float(v[16])  # glass wall distance along its normal
    cal.invalidate_mmlut()


def project_px(cal, X, cpar) -> np.ndarray:
    return metric_to_pixel_batch(img_coord_batch(X, cal, cpar.mm), cpar)


def load_case(yaml_path: Path):
    """Parameters and fresh calibrations (no multimedia LUT -> exact rays)."""
    from openptv2.batch.pyptv_batch import build_processing_experiment

    pe = build_processing_experiment(yaml_path, 0, 0)
    params = yaml.safe_load(yaml_path.read_text())
    cal_ori = params["cal_ori"]
    cals = []
    for ori in cal_ori["img_ori"]:
        add = ori.replace(".ori", ".addpar")
        cals.append(Calibration.from_file(ori, add))
    return pe.cpar, pe.vpar, cals, cal_ori


def calblock_observations(cal_ori, cals, cpar, radius_px, min_npix):
    """Match each body dot to the nearest calibration-image target.

    Targets smaller than ``min_npix`` pixels are skipped: calibration images
    also hold particles and noise, which are much smaller than the dots."""
    from openptv2.tracking_framebuf import read_targets

    body = np.loadtxt(cal_ori["fixp_name"])
    ids, xyz = body[:, 0].astype(int), body[:, 1:4]
    obs = []  # (cam, dot index, px, py)
    for cam, img in enumerate(cal_ori["img_cal_name"]):
        targs = read_targets(img, 0)
        if len(targs) == 0:
            continue
        det = np.array([[t.x, t.y] for t in targs if t.n >= min_npix]).reshape(-1, 2)
        proj = project_px(cals[cam], xyz, cpar)
        used = set()
        for k, p in enumerate(proj):
            d = np.hypot(*(det - p).T)
            j = int(np.argmin(d))
            if d[j] <= radius_px and j not in used:
                used.add(j)
                obs.append((cam, k, det[j, 0], det[j, 1]))
    return ids, xyz, obs


def particle_observations(store_path, frames, cals, cpar, vpar, max_px, min_cams):
    """Particles seen by >= min_cams cameras whose triangulation is consistent."""
    store = RunStore(store_path, mode="r")
    n_cams = cpar.num_cams
    obs_px, seeds = [], []
    for fr in frames:
        if not store.has_correspondences(fr):
            continue
        _, ids = store.read_correspondences(fr)
        det = []
        for cam in range(n_cams):
            t = store.read_targets(cam, fr)
            det.append(np.array([[t[i].x(), t[i].y()] for i in range(len(t))]))
        for row in ids:
            seen = [c for c in range(n_cams) if row[c] >= 0]
            if len(seen) < min_cams:
                continue
            obs_px.append({c: det[c][row[c]] for c in seen})
    if not obs_px:
        return [], np.empty((0, 3))
    # Triangulate with the current calibration; drop inconsistent points.
    flat = np.full((len(obs_px), n_cams, 2), -1e10)
    for i, o in enumerate(obs_px):
        for c, px in o.items():
            m = pixel_to_metric_batch(np.atleast_2d(px), cpar)
            flat[i, c] = distorted_to_flat(m, cals[c])[0]
    X, _ = point_positions(flat, cpar, cals, vpar)
    keep_obs, keep_X = [], []
    for i, o in enumerate(obs_px):
        err = [np.hypot(*(project_px(cals[c], X[i : i + 1], cpar)[0] - px))
               for c, px in o.items()]  # fmt: skip
        if max(err) <= max_px:
            keep_obs.append(o)
            keep_X.append(X[i])
    return keep_obs, np.array(keep_X).reshape(-1, 3)


def bundle_adjust(
    cals, cpar, body_xyz, body_obs, part_obs, part_X, free, *,
    body_sigma_mm, f_scale_px, max_nfev, verbose, priors=None, tie_glass=False,
):  # fmt: skip
    """``priors``: {param: (sigma, center or None)} -- a soft pull of each
    camera's parameter toward ``center`` (None: its starting value), so
    parameters the data can barely separate (principal point vs. angles, glass
    distance vs. focal length) cannot wander. ``tie_glass``: cameras looking
    through the same wall (same sign of glass z) share one wall distance."""
    priors = {p: v for p, v in (priors or {}).items() if p in free}
    n_cams = len(cals)
    cals = [copy.deepcopy(c) for c in cals]
    free_idx = [PARAMS.index(p) for p in free]
    v0 = [cal_to_vec(c) for c in cals]
    scale = np.array([SCALE[p] for p in free])
    n_cp = len(free_idx)
    n_body, n_part = len(body_xyz), len(part_X)

    # Observation lists per camera: (point index into the joint point array, px)
    # Points 0..n_body-1 are body dots, then particles.
    cam_rows = {c: ([], []) for c in range(n_cams)}
    for cam, k, px, py in body_obs:
        cam_rows[cam][0].append(k)
        cam_rows[cam][1].append((px, py))
    for i, o in enumerate(part_obs):
        for cam, px in o.items():
            cam_rows[cam][0].append(n_body + i)
            cam_rows[cam][1].append(tuple(px))
    cam_rows = {c: (np.array(a, int), np.array(b, float).reshape(-1, 2))
                for c, (a, b) in cam_rows.items()}  # fmt: skip
    n_obs = sum(len(a) for a, _ in cam_rows.values())

    def unpack(z):
        cams = z[: n_cams * n_cp].reshape(n_cams, n_cp) * scale
        pts = z[n_cams * n_cp :].reshape(-1, 3)
        return cams, pts

    def residuals(z):
        dcam, pts = unpack(z)
        res = []
        for c in range(n_cams):
            v = v0[c].copy()
            v[free_idx] += dcam[c]
            vec_to_cal(v, cals[c])
            idx, px = cam_rows[c]
            if len(idx):
                res.append((project_px(cals[c], pts[idx], cpar) - px).ravel())
        # Body dots may move only as far as the body's accuracy allows.
        res.append(((pts[:n_body] - body_xyz) / body_sigma_mm * f_scale_px).ravel())
        for p, (sigma, center) in priors.items():
            j = PARAMS.index(p)
            for c in range(n_cams):
                ref = v0[c][j] if center is None else center
                val = v0[c][j] + dcam[c][free.index(p)]
                res.append(np.atleast_1d((val - ref) / sigma * f_scale_px))
        for group in glass_groups:
            gz = [v0[c][16] + dcam[c][free.index("gz")] for c in group]
            res.append((np.array(gz) - np.mean(gz)) / 0.01 * f_scale_px)
        return np.concatenate(res)

    glass_groups = []
    if tie_glass and "gz" in free:
        for sign in (-1, 1):
            g = [c for c in range(n_cams) if np.sign(v0[c][16]) == sign]
            if len(g) > 1:
                glass_groups.append(g)
    n_extra = len(priors) * n_cams + sum(len(g) for g in glass_groups)

    z0 = np.concatenate([np.zeros(n_cams * n_cp), body_xyz.ravel(), part_X.ravel()])

    sp = lil_matrix((2 * n_obs + 3 * n_body + n_extra, len(z0)), dtype=int)
    r = 0
    for c in range(n_cams):
        idx, _ = cam_rows[c]
        for k in idx:
            sp[r : r + 2, c * n_cp : (c + 1) * n_cp] = 1
            col = n_cams * n_cp + 3 * k
            sp[r : r + 2, col : col + 3] = 1
            r += 2
    for k in range(n_body):
        col = n_cams * n_cp + 3 * k
        for d in range(3):
            sp[r, col + d] = 1
            r += 1
    for p in priors:
        for c in range(n_cams):
            sp[r, c * n_cp + free.index(p)] = 1
            r += 1
    for group in glass_groups:
        for _ in group:
            for c in group:
                sp[r, c * n_cp + free.index("gz")] = 1
            r += 1

    def report(z, tag):
        res = residuals(z)
        rep = res[: 2 * n_obs].reshape(-1, 2)
        err = np.hypot(rep[:, 0], rep[:, 1])
        out, start = {}, 0
        for c in range(n_cams):
            n = len(cam_rows[c][0])
            e = err[start : start + n]
            nb = int(np.sum(cam_rows[c][0] < n_body))
            out[c] = (np.median(e[:nb]) if nb else np.nan,
                      np.median(e[nb:]) if n > nb else np.nan)  # fmt: skip
            start += n
        print(f"[{tag}] median reprojection px (body / particles): "
              + "  ".join(f"cam{c + 1} {b:.2f}/{p:.2f}" for c, (b, p) in out.items()))  # fmt: skip
        return err

    report(z0, "before")
    sol = least_squares(
        residuals, z0, jac_sparsity=sp, loss="soft_l1", f_scale=f_scale_px,
        x_scale="jac", max_nfev=max_nfev, verbose=2 if verbose else 0,
    )  # fmt: skip
    err = report(sol.x, "after ")
    residuals(sol.x)  # leaves the fitted parameters in `cals`
    _, pts = unpack(sol.x)
    moved = np.linalg.norm(pts[:n_body] - body_xyz, axis=1)
    print(
        f"body dots moved: median {np.median(moved):.3f} mm, max {moved.max():.3f} mm"
    )
    return cals, sol, err


def parse_priors(text):
    priors = {}
    for item in filter(None, text.split(",")):
        name, spec = item.split("=")
        sigma, _, center = spec.partition("@")
        priors[name] = (float(sigma), float(center) if center else None)
    return priors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("yaml", type=Path)
    ap.add_argument("--frames", type=int, nargs=2, required=True)
    ap.add_argument(
        "--store", default=None, help="run store (default: found next to the YAML)"
    )
    ap.add_argument("--free", default="ext,cc,xh,yh,k1,k2,p1,p2")
    ap.add_argument("--min-cams", type=int, default=4)
    ap.add_argument("--max-px", type=float, default=3.0,
                    help="drop particles reprojecting worse than this before the fit")  # fmt: skip
    ap.add_argument("--max-particles", type=int, default=3000)
    ap.add_argument("--body-radius-px", type=float, default=6.0)
    ap.add_argument("--body-min-npix", type=int, default=6)
    ap.add_argument("--body-sigma-mm", type=float, default=0.05)
    ap.add_argument("--f-scale-px", type=float, default=0.5)
    ap.add_argument("--max-nfev", type=int, default=50)
    ap.add_argument("--prior", default="",
                    help="soft priors, e.g. xh=0.2@0,yh=0.2@0,gz=20 (sigma[@center]; "
                         "no center = the starting value)")  # fmt: skip
    ap.add_argument("--tie-glass", action="store_true",
                    help="cameras behind the same wall share its distance")  # fmt: skip
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    yaml_path = a.yaml.resolve()
    os.chdir(yaml_path.parent)
    cpar, vpar, cals, cal_ori = load_case(yaml_path)
    free = []
    for f in a.free.split(","):
        free += GROUPS.get(f, [f])

    ids, body_xyz, body_obs = calblock_observations(
        cal_ori, cals, cpar, a.body_radius_px, a.body_min_npix
    )
    print(f"body dots matched per camera: "
          f"{[sum(1 for o in body_obs if o[0] == c) for c in range(len(cals))]}")  # fmt: skip
    frames = range(a.frames[0], a.frames[1] + 1)
    store = a.store or find_existing_store(yaml_path.parent)
    if store is None:
        sys.exit("no run store found: run the sequence step first")
    part_obs, part_X = particle_observations(
        store, frames, cals, cpar, vpar, a.max_px, a.min_cams
    )
    if len(part_obs) > a.max_particles:
        sel = np.linspace(0, len(part_obs) - 1, a.max_particles).astype(int)
        part_obs, part_X = [part_obs[i] for i in sel], part_X[sel]
    print(f"particles used: {len(part_obs)} (>= {a.min_cams} cams, <= {a.max_px} px)")

    new_cals, sol, _ = bundle_adjust(
        cals, cpar, body_xyz, body_obs, part_obs, part_X, free,
        body_sigma_mm=a.body_sigma_mm, f_scale_px=a.f_scale_px,
        max_nfev=a.max_nfev, verbose=a.verbose, priors=parse_priors(a.prior),
        tie_glass=a.tie_glass,
    )  # fmt: skip
    print(f"least_squares: {sol.message} (nfev {sol.nfev}, cost {sol.cost:.1f})")
    for c, cal in enumerate(new_cals):
        e, i = cal.ext_par, cal.int_par
        print(f"cam{c + 1}: pos ({e.x0:.2f}, {e.y0:.2f}, {e.z0:.2f}) "
              f"cc {i.cc:.3f} xh {i.xh:.3f} yh {i.yh:.3f} k1 {cal.added_par.k1:.2e} "
              f"glass z {cal.glass_par.vec_z:.2f}")  # fmt: skip

    if a.write:
        for ori, cal in zip(cal_ori["img_ori"], new_cals):
            add = ori.replace(".ori", ".addpar")
            for f in (ori, add):
                if Path(f).exists():
                    shutil.copy(f, f + ".pbabck")
            cal.to_file(ori, add)
        print("written:", ", ".join(cal_ori["img_ori"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

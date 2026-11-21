"""Self-calibration with a known answer: perturb the calibration of a rendered synthetic recording by a known amount, fit the
joint self-calibration on its detected tracer blobs, and compare 3D positions with the truth (after removing the gauge,
a best-fit affine map) and the held-out leave-one-camera-out residual.

    uv run python scripts/selfcal_synthetic.py [--scale 1.0] [--n-fit 12000]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selfcal_real import loco_rms  # noqa: E402
from synth_bench import _experiment  # noqa: E402

ROOT = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest"


def triangulate(xy, cals, cpar, vpar):
    from openptv2.loco import pixels_to_flat
    from openptv2.orientation import point_positions

    flat = np.stack([pixels_to_flat(xy[:, c], cals[c], cpar) for c in range(len(cals))], axis=1)
    return np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)


def affine_residual(pos, truth):
    A = np.c_[pos, np.ones(len(pos))]
    coef, *_ = np.linalg.lstsq(A, truth, rcond=None)
    return np.linalg.norm(A @ coef - truth, axis=1)


def perturb(cals, scale, rng):
    out = [copy.deepcopy(c) for c in cals]
    for c in out:
        c.set_pos(c.get_pos() + rng.normal(0, 0.10 * scale, 3))
        c.set_angles(c.get_angles() + rng.normal(0, 3e-4 * scale, 3))
        c.int_par.xh += rng.normal(0, 0.01 * scale)
        c.int_par.yh += rng.normal(0, 0.01 * scale)
        c.added_par.k1 += rng.normal(0, 1e-4 * scale)
        c.added_par.k2 += rng.normal(0, 1e-4 * scale)
        c.added_par.p1 += rng.normal(0, 3e-4 * scale)
        c.added_par.p2 += rng.normal(0, 3e-4 * scale)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--n-fit", type=int, default=12000)
    ap.add_argument("--nfev", type=int, default=300)
    ap.add_argument("--frames", type=int, default=400)
    a = ap.parse_args()

    from openptv2.selfcal import joint_self_calibrate
    from openptv2.storage import RunStore

    run = ROOT / "syn1" / "test"
    exp = _experiment(run, run / "parameters_Run1.yaml", 1, a.frames)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    st = RunStore(run / "res" / "run.zarr", mode="r")
    tr = np.load(ROOT / "truth/truth.npz")
    xy_l, tpos_l, fr_l = [], [], []
    for f in sorted(st.frames())[: a.frames]:
        pos, ids = st.read_correspondences(f)
        ids = np.asarray(ids)[:, :n_cams]
        ok = (ids >= 0).all(1)
        if not ok.any():
            continue
        tg = [np.asarray(st.read_targets(c, f), dtype=object) for c in range(n_cams)]
        xy = np.stack([np.array([[tg[c][i].x, tg[c][i].y] for i in ids[ok, c]]) for c in range(n_cams)], axis=1)
        m = tr["frame"] == f
        d, j = cKDTree(tr["pos_mm"][m]).query(np.asarray(pos)[ok], distance_upper_bound=0.4)
        good = np.isfinite(d)
        if not good.any():
            continue
        txy = tr["pixel_xy_per_cam"][m][j[good]].reshape(-1, n_cams, 2)
        # clean blob: detected centre within 0.5 px of the true projection in every camera
        clean = (np.linalg.norm(xy[good] - txy, axis=2) < 0.5).all(1)
        sel = np.flatnonzero(good)[clean]
        xy_l.append(xy[sel])
        tpos_l.append(tr["pos_mm"][m][j[sel]] if False else tr["pos_mm"][m][j[good]][clean])
        fr_l.append(np.full(len(sel), f))
    xy, tpos, fr = np.concatenate(xy_l), np.concatenate(tpos_l), np.concatenate(fr_l)
    print(f"clean 4-camera points: {len(xy)} in {len(np.unique(fr))} frames")
    odd = fr % 2 == 1
    rng = np.random.default_rng(1)
    fit_idx = rng.choice(np.flatnonzero(odd), min(a.n_fit, odd.sum()), replace=False)
    test = ~odd

    def report(name, cs):
        pos = triangulate(xy[test], cs, cpar, vpar)
        raw = np.linalg.norm(pos - tpos[test], axis=1)
        aff = affine_residual(pos, tpos[test])
        lo = loco_rms(xy[test], cs, cpar, vpar)
        print(f"{name:24s} position error vs truth: raw median {np.median(raw):.4f} mm | after affine alignment "
              f"median {np.median(aff):.4f} p90 {np.percentile(aff, 90):.4f} | LOCO rms {np.mean(lo):.4f} px")

    report("true calibration", cals)
    pc = perturb(cals, a.scale, rng)
    report(f"perturbed (x{a.scale})", pc)
    for name, kw in (("self-cal ext", dict(free_pp=False, free_radial=(), free_decentering=False)),
                     ("self-cal all", dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=True))):
        res = joint_self_calibrate(xy[fit_idx], tpos[fit_idx] * 0 + triangulate(xy[fit_idx], pc, cpar, vpar), pc, cpar,
                                   max_nfev=a.nfev, **kw)
        report(name, res.cals)


if __name__ == "__main__":
    main()

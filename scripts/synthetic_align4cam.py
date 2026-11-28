"""Truth test for the 4-camera residual alignment (``merge_refine.fit_4cam_residual_field``).

Points with known 3D positions are projected through the TRUE cameras (``test_data/test_cavity``, noise added),
then processed with SPOILED cameras, as a real calibration error would. Per particle we triangulate with all four
cameras (``x4``) and with one camera left out (``x3``), before and after the alignment, and report against the truth:

* ``err3``  : 3D error of the three-camera position (what the alignment is meant to reduce)
* ``jump``  : |x3 - x4|, the sideways jump when a particle changes between 3 and 4 cameras
* ``err4``  : 3D error of the four-camera position (the alignment must not touch it)

Scenarios cover: no calibration error (must do no harm), two sizes of error, outliers among the 4-camera points,
a small fit sample, and partial points outside the region the field was fitted on (extrapolation).

    uv run python scripts/synthetic_align4cam.py [--n 6000] [--seed 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, VolumePar
from openptv2.loco import pixels_to_flat, project_pixels
from openptv2.merge_refine import (
    eval_4cam_residual_field,
    fit_4cam_residual_field,
    triangulate_weighted,
)
from openptv2.orientation import point_positions

ROOT = Path(__file__).resolve().parents[1]
CAV = ROOT / "test_data" / "test_cavity"


def _cals(kind: str, scale: float, rng):
    out = []
    for c in range(1, 5):
        cal = Calibration.from_file(str(CAV / "cal" / f"cam{c}.tif.ori"), str(CAV / "cal" / f"cam{c}.tif.addpar"))
        if kind == "spoiled" and c > 1:
            cal.set_pos(cal.get_pos() + rng.normal(0, 0.5 * scale, 3))
            cal.set_angles(cal.get_angles() + rng.normal(0, 0.0015 * scale, 3))
        out.append(cal)
    return out


def _points(n, truth, cpar, rng, shell=None):
    """Random points seen by all four cameras; ``shell`` = (r_min, r_max) as fractions of the volume radius."""
    lo, hi = np.array([-20.0, -20.0, -20.0]), np.array([20.0, 20.0, 20.0])
    ctr = np.zeros(3)
    pts = []
    while sum(len(p) for p in pts) < n:
        p = rng.uniform(lo, hi, (4 * n, 3))
        ok = np.ones(len(p), bool)
        for cal in truth:
            px = project_pixels(p, cal, cpar)
            ok &= (px[:, 0] > 20) & (px[:, 0] < cpar.imx - 20) & (px[:, 1] > 20) & (px[:, 1] < cpar.imy - 20)
        p = p[ok]
        r = np.linalg.norm((p - ctr) / 20.0, axis=1) / np.sqrt(3)
        if shell is not None:
            p = p[(r >= shell[0]) & (r < shell[1])]
        pts.append(p)
    return np.vstack(pts)[:n]


def _observe(x, truth, cpar, noise, rng):
    return np.stack([project_pixels(x, cal, cpar) for cal in truth], axis=1) + rng.normal(0, noise, (len(x), 4, 2))


def _tri(ctr, cals, cpar, vpar, drop=None):
    flat = np.full((len(ctr), 4, 2), -1e10)
    for c in range(4):
        if drop is not None:
            use = drop != c
        else:
            use = np.ones(len(ctr), bool)
        flat[use, c] = pixels_to_flat(ctr[use, c], cals[c], cpar)
    return np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)


def scenario(name, seed, n=6000, scale=1.0, kind="spoiled", noise=0.1, n_fit=None, outliers=0.0, fit_shell=None,
             eval_shell=None):
    rng = np.random.default_rng(seed)
    cpar, vpar = ControlPar.from_yaml(str(CAV / "parameters_Run1.yaml")), VolumePar.from_yaml(str(CAV / "parameters_Run1.yaml"))
    truth, proc = _cals("truth", 0, rng), _cals(kind, scale, rng)
    xf = _points(n_fit or n, truth, cpar, rng, fit_shell)
    cf = _observe(xf, truth, cpar, noise, rng)
    if outliers:
        bad = rng.random(len(cf)) < outliers
        cf[bad, rng.integers(0, 4, bad.sum())] += rng.normal(0, 3.0, (bad.sum(), 2))
    pf = _tri(cf, proc, cpar, vpar)
    x4 = triangulate_weighted(cf, np.full((len(cf), 4), 0.01), cpar, proc, pf, n_iter=3, frozen_jacobian=True)
    ok = np.isfinite(x4[:, 0])
    field = fit_4cam_residual_field(x4[ok], cf[ok], proc, cpar)

    xe = _points(n, truth, cpar, rng, eval_shell)
    ce = _observe(xe, truth, cpar, noise, rng)
    x4e = _tri(ce, proc, cpar, vpar)
    drop = rng.integers(0, 4, len(xe))
    x3 = _tri(ce, proc, cpar, vpar, drop)
    d = eval_4cam_residual_field(x3, field)
    c_al = ce - d
    c_al[np.arange(len(xe)), drop] = ce[np.arange(len(xe)), drop]
    x3a = _tri(c_al, proc, cpar, vpar, drop)
    # the same call as the production code: delta evaluated at the three-camera position, all cameras shifted
    x3b = _tri(ce - d, proc, cpar, vpar, drop)

    def rms(a, b):
        e = np.linalg.norm(a - b, axis=1)
        return float(np.sqrt(np.nanmean(e[np.isfinite(e)] ** 2)))

    row = dict(name=name, err3_raw=rms(x3, xe), err3_al=rms(x3b, xe), jump_raw=rms(x3, x4e), jump_al=rms(x3b, x4e),
               err4=rms(x4e, xe))
    print(f"{name:34s} err3 {row['err3_raw']:.4f} -> {row['err3_al']:.4f}   jump {row['jump_raw']:.4f} -> {row['jump_al']:.4f}"
          f"   err4 (untouched) {row['err4']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    s, n = a.seed, a.n
    print("RMS 3D error in mm against the truth; jump = |x3 - x4|")
    scenario("A  no calibration error", s, n, kind="truth")
    scenario("B  calibration error x1", s, n, scale=1.0)
    scenario("C  calibration error x3", s, n, scale=3.0)
    scenario("D  noise 0.3 px, error x1", s, n, noise=0.3)
    scenario("E  5% gross outliers in fit set", s, n, outliers=0.05)
    scenario("F  only 150 fit points", s, n, n_fit=150)
    scenario("G  fit inner 40%, evaluate outer", s, n, fit_shell=(0.0, 0.4), eval_shell=(0.4, 1.0))
    scenario("H  fit inner 40%, evaluate inner", s, n, fit_shell=(0.0, 0.4), eval_shell=(0.0, 0.4))


if __name__ == "__main__":
    main()

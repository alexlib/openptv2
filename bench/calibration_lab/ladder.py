# ruff: noqa
"""Calibration ladder: break one assumption of the imaging model at a time and see what a standard calibration does with it.

    uv run python bench/calibration_lab/ladder.py [--scenarios a,b,...] [--seed 1]

Truth = ``TruthRig`` (independent ray tracing). Model = the library (pinhole + distortion + flat layers with *nominal*
glass/indices). Per scenario the truth differs from the nominal in ONE thing:

  none     truth equals the nominal model (control: nothing should be left over)
  d+1      glass thickness 1 mm thicker than the model assumes
  n3+.01   index of the fluid 0.01 higher
  g+1      glass plane 1 mm farther from the origin (glass vector length)
  tiltF    water-side glass surface tilted by 0.3 deg
  tiltB    air-side glass surface tilted by 0.3 deg (a wedge together with the water side)

Stages:  A  plate calibration (multi-plane dots, known positions): each camera alone, exterior + interior + distortion
         B  A, then tracer self-calibration (joint bundle adjustment, plate points as anchor)
         C  B with the glass thickness and the fluid index as additional free parameters (joint over all cameras)
Scores on held-out tracer points (noise-free observations: only systematic error is measured):
  pos   3D position error vs truth (mm): median / p90, and the part along z
  loco  leave-one-camera-out reprojection rms (px)
  jump  systematic position difference between 3-camera and 4-camera triangulation, per missing camera (um)
"""
import argparse, copy, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from openptv2.loco import loco_residuals, pixels_to_flat, project_pixels
from openptv2.orientation import point_positions
from openptv2.recorrespond import load_parameters
from openptv2.selfcal import KnownBody, joint_self_calibrate
from openptv2.synthetic_rig import TruthRig

SYN1 = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest/syn1/test/parameters_Run1.yaml"
BOX = np.array([[-12.0, -8.0, -18.0], [60.0, 66.0, 19.0]])     # the tracer volume of syn1 (mm)


def scenario(rig, name, mag=1.0):
    if name == "none":
        return rig
    if name == "d+1":
        return rig.with_(glass_thickness=rig.glass_thickness + 1.0 * mag)
    if name == "n3+.01":
        return rig.with_(n_water=rig.n_water + 0.01 * mag)
    if name == "g+1":
        out = rig.with_()
        for c in out.cals:
            g = np.asarray(c.get_glass_vec(), float)
            c.set_glass_vec(g * (np.linalg.norm(g) + 1.0 * mag) / np.linalg.norm(g))
        return out
    if name == "tiltF":
        return rig.with_(tilt_front=[(np.radians(0.3 * mag), 0.0)] * len(rig.cals))
    if name.startswith("drift"):            # the cameras moved after the plate was taken: tracer-time truth differs
        out = rig.with_()
        k = float(name[5:] or 1.0)
        out.cals = perturb(out.cals, np.random.default_rng(77), 0.3 * k)
        return out
    if name == "tiltB":
        return rig.with_(tilt_back=[(0.0, np.radians(0.3 * mag))] * len(rig.cals))
    raise ValueError(name)


def plate_points(n=13, planes=5, depth_frac=1.0):
    zc = 0.5 * (BOX[0, 2] + BOX[1, 2]); zh = 0.5 * depth_frac * (BOX[1, 2] - BOX[0, 2])
    zs = np.linspace(zc - zh, zc + zh, planes)
    gx = np.linspace(BOX[0, 0], BOX[1, 0], n)
    gy = np.linspace(BOX[0, 1], BOX[1, 1], n)
    return np.concatenate([np.array([[x, y, z] for x in gx for y in gy]) for z in zs])


def perturb(cals, rng, s):
    out = [copy.deepcopy(c) for c in cals]
    for c in out:
        c.set_pos(c.get_pos() + rng.normal(0, 0.3 * s, 3))
        c.set_angles(c.get_angles() + rng.normal(0, 1e-3 * s, 3))
        c.int_par.xh += rng.normal(0, 0.01 * s)
        c.int_par.yh += rng.normal(0, 0.01 * s)
    return out


def fit_camera(cal0, pts, px, cpar, free_cc=True):
    """Plate calibration of one camera: exterior, principal point, (cc), k1 k2 p1 p2 from known points and their pixels."""
    cal = copy.deepcopy(cal0)
    ok = np.isfinite(px[:, 0])
    p0 = np.concatenate([cal.get_pos(), cal.get_angles() * 1e3, [cal.int_par.xh, cal.int_par.yh, cal.int_par.cc],
                         [cal.added_par.k1 * 1e3, cal.added_par.k2 * 1e3, cal.added_par.p1 * 1e3, cal.added_par.p2 * 1e3]])
    mask = np.ones(len(p0), bool)
    if not free_cc:
        mask[8] = False

    def apply(x):
        v = p0.copy(); v[mask] = x
        cal.set_pos(v[:3]); cal.set_angles(v[3:6] * 1e-3)
        cal.int_par.xh, cal.int_par.yh, cal.int_par.cc = v[6], v[7], v[8]
        cal.added_par.k1, cal.added_par.k2, cal.added_par.p1, cal.added_par.p2 = v[9] * 1e-3, v[10] * 1e-3, v[11] * 1e-3, v[12] * 1e-3

    def resid(x):
        apply(x)
        return (project_pixels(pts[ok], cal, cpar) - px[ok]).ravel()

    scale = np.array([1, 1, 1, 1, 1, 1, .01, .01, .1, .1, .1, .1, .1])[mask]
    sol = least_squares(resid, p0[mask], x_scale=scale, method="trf")
    apply(sol.x)
    return cal, float(np.sqrt(np.mean(sol.fun**2)))


def triangulate(px, cals, cpar, vpar):
    flat = np.stack([pixels_to_flat(px[:, c], cals[c], cpar) for c in range(len(cals))], axis=1)
    return np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)


def score(px, truth_pts, cals, cpar, vpar):
    n = len(cals)
    pos = triangulate(px, cals, cpar, vpar)
    e = pos - truth_pts
    lo = loco_residuals(px, cpar, cals, vpar)
    jumps = []
    for m in range(n):
        sub = [c for c in range(n) if c != m]
        p3 = triangulate(px[:, sub], [cals[c] for c in sub], cpar, vpar)
        jumps.append(np.linalg.norm(np.nanmean(p3 - pos, axis=0)) * 1e3)
    A_ = np.c_[pos, np.ones(len(pos))]
    coef, *_ = np.linalg.lstsq(A_[np.isfinite(pos).all(1)], truth_pts[np.isfinite(pos).all(1)], rcond=None)
    aff = np.linalg.norm(A_ @ coef - truth_pts, axis=1)
    return dict(pos_aff=float(np.nanmedian(aff)), pos_med=float(np.nanmedian(np.linalg.norm(e, axis=1))), pos_p90=float(np.nanpercentile(np.linalg.norm(e, axis=1), 90)),
                z_med=float(np.nanmedian(np.abs(e[:, 2]))), loco=float(np.nanmean(np.abs(lo))) if lo.ndim else float(lo),
                jump_um=jumps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="none,d+1,n3+.01,g+1,tiltF,tiltB")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-trace", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--stages", default="A,B,C")
    ap.add_argument("--nfev", type=int, default=200)
    ap.add_argument("--ext-sigma", type=float, default=20.0)
    ap.add_argument("--mag", type=float, default=1.0, help="multiply the size of every mismatch")
    ap.add_argument("--plate-depth", type=float, default=1.0, help="fraction of the tracer depth the plate planes span")
    a = ap.parse_args()
    cpar, vpar, cals_nom, _ = load_parameters(SYN1)
    base = TruthRig.from_cals(cals_nom, cpar)
    rng = np.random.default_rng(a.seed)
    plate = plate_points(depth_frac=a.plate_depth)
    fit_tr = rng.uniform(BOX[0], BOX[1], (a.n_trace, 3))
    test_tr = rng.uniform(BOX[0], BOX[1], (2000, 3))
    print(f"plate {len(plate)} points on 5 planes spanning {a.plate_depth:.0%} of the depth, mismatch x{a.mag:g}, {a.n_trace} tracers to fit, 2000 held out; centroid noise {a.noise} px; model = nominal glass {base.glass_thickness} mm, n {base.n_glass}/{base.n_water}")
    hdr = f"{'scenario':8} {'stage':6} {'plate rms':>9} {'pos med':>8} {'p90':>7} {'z med':>7} {'aff':>7} {'loco':>6}   jump um per missing camera"
    print(hdr)
    for name in a.scenarios.split(","):
        rig = scenario(base, name, a.mag)
        rig_plate = base if name.startswith("drift") else rig         # the plate was taken before the cameras drifted
        px_plate = rig_plate.observe(plate, a.noise, rng)
        px_fit = rig.observe(fit_tr, a.noise, rng)
        px_test = rig.project(test_tr)
        ok_t = np.isfinite(px_test).all(axis=(1, 2))
        start = perturb(cals_nom, rng, 1.0)
        # A: plate calibration, camera by camera (model: nominal layers)
        calA, rms = [], []
        for c in range(len(start)):
            cc_, r = fit_camera(start[c], plate, px_plate[:, c], cpar)
            calA.append(cc_); rms.append(r)
        def show(stage, cs, plate_rms):
            s = score(px_test[ok_t], test_tr[ok_t], cs, cpar, vpar)
            print(f"{name:8} {stage:6} {plate_rms:9.3f} {s['pos_med']:8.4f} {s['pos_p90']:7.4f} {s['z_med']:7.4f} {s['pos_aff']:7.4f} {s['loco']:6.3f}   " + " ".join(f"{j:6.1f}" for j in s["jump_um"]), flush=True)
        show("A", calA, float(np.mean(rms)))
        if "B" in a.stages:
            ok_f = np.isfinite(px_fit).all(axis=(1, 2))
            x0 = triangulate(px_fit[ok_f], calA, cpar, vpar)
            ok_p = np.isfinite(px_plate).all(axis=(1, 2))
            body = None if name.startswith("drift") else KnownBody(plate[ok_p], px_plate[ok_p])
            t0 = time.perf_counter()
            res = joint_self_calibrate(px_fit[ok_f], x0, calA, cpar, body=body, max_nfev=a.nfev, ext_sigma_scale=a.ext_sigma, free_radial=("k1", "k2"))
            show("B", res.cals, res.rms_after_px)
            print(f"{'':8} {'':6} (B took {time.perf_counter() - t0:.0f}s, {res.nfev} evaluations)")
        if "C" in a.stages:
            t0 = time.perf_counter()
            resC = joint_self_calibrate(px_fit[ok_f], x0, calA, cpar, body=body, max_nfev=a.nfev, ext_sigma_scale=a.ext_sigma,
                                        free_radial=("k1", "k2"), free_multimedia=("d", "n3"), free_glass=True)
            cparC = resC.cpar
            s = score(px_test[ok_t], test_tr[ok_t], resC.cals, cparC, vpar)
            print(f"{name:8} {'C':6} {resC.rms_after_px:9.3f} {s['pos_med']:8.4f} {s['pos_p90']:7.4f} {s['z_med']:7.4f} {s['pos_aff']:7.4f} {s['loco']:6.3f}   " + " ".join(f"{j:6.1f}" for j in s["jump_um"]))
            print(f"{'':8} {'':6} (C took {time.perf_counter() - t0:.0f}s; fitted d {resC.multimedia['d']:.3f} (model {base.glass_thickness}, truth {rig.glass_thickness}), n3 {resC.multimedia['n3']:.4f} (model {base.n_water}, truth {rig.n_water}))")


if __name__ == "__main__":
    main()

# ruff: noqa
"""Deep tests of self-calibration on a synthetic cavity (cameras on both sides of a water cell), everything scored against truth.

    uv run python bench/calibration_lab/cavity_synth_lab.py <scratch> [experiments comma list | all] [--frames 6] [--n 900] [--noise 0.15]

Truth: ``cavity_rig()`` (independent Snell ray tracer; cameras at -z and +z, glass vectors of both signs, per-camera lens terms), blobs of
random tracers with detection probability 0.8. Model: the library with the calibration the experiment gives it (a perturbed copy of the
truth, a calibration that left the lens out, wrong glass, ...). Per experiment, the model's matching is scored, the cameras are
re-fitted from the quadruplets that matching produced (``self_calibrate_from_store``, optionally with other free parameters), the blobs are
matched again with the new cameras and scored: recall of the particles seen by all four cameras, share of wrong quadruplets, 3D error,
image-space calibration error (rms px difference between the model's and the truth's image of random volume points), and the
truth-referenced principal stretches / shift of the volume (the gauge: not measurable on real data).

Invariance experiments move the whole rig (rotate 180 deg about y so that +z cameras become -z, rotate 90 deg about x, translate, reverse the
camera order) with identical blobs: the answers must not change.
"""
import argparse, copy, shutil, sys, time, traceback
from pathlib import Path

import numpy as np

from openptv2.calibration_consistency import consistency_report, self_calibrate_from_store
from openptv2.correspondences import match_correspondences_batch_parallel
from openptv2.loco import project_pixels
from openptv2.storage import RunStore
from openptv2.synthetic_rig import cavity_rig, score_matching, synthetic_run, transformed

ap = argparse.ArgumentParser()
ap.add_argument("work"); ap.add_argument("exps", nargs="?", default="all")
ap.add_argument("--frames", type=int, default=6); ap.add_argument("--n", type=int, default=900); ap.add_argument("--noise", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=1)
A = ap.parse_args()
WORK = Path(A.work); WORK.mkdir(parents=True, exist_ok=True)


def rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# ---- perturbations of a list of calibrations ----------------------------------------------------------------------------
def drift(cals, scale=1.0, which=None, seed=3, pos=0.15, ang=2e-4, pp=0.003):
    rng = np.random.default_rng(seed)
    out = [copy.deepcopy(c) for c in cals]
    for i, c in enumerate(out):
        r = [rng.normal(0, pos * scale, 3), rng.normal(0, ang * scale, 3), rng.normal(0, pp * scale, 2)]
        if which is None or i in which:
            c.set_pos(c.get_pos() + r[0]); c.set_angles(c.get_angles() + r[1]); c.int_par.xh += r[2][0]; c.int_par.yh += r[2][1]
    return out


def lens_scaled(cals, f):
    """The lens terms (k1 k2 p1 p2) of the model are only the fraction ``f`` of the truth's; principal point and the rest unchanged."""
    out = [copy.deepcopy(c) for c in cals]
    for c in out:
        c.added_par.k1 *= f; c.added_par.k2 *= f; c.added_par.p1 *= f; c.added_par.p2 *= f
    return out


def glass_shift(cals, delta, which):
    out = [copy.deepcopy(c) for c in cals]
    for i in which:
        g = np.asarray(out[i].get_glass_vec(), float); out[i].set_glass_vec(g / np.linalg.norm(g) * (np.linalg.norm(g) + delta)); out[i].invalidate_mmlut()
    return out


def move_side(cals, which, vec):
    out = [copy.deepcopy(c) for c in cals]
    for i in which:
        out[i].set_pos(out[i].get_pos() + np.asarray(vec, float)); g = np.asarray(out[i].get_glass_vec(), float)
        out[i].invalidate_mmlut()
    return out


NEG, POS_ = [0, 1], [2, 3]          # cameras at -z / at +z
# name: (truth modifier of the rig used for blobs, model modifier (cals -> cals), model cpar modifier or None, notes)
EXPS = {
    "E00_control": (None, lambda c: c, None),
    "E01_drift_small": (None, lambda c: drift(c, 0.15), None),
    "E02_drift": (None, lambda c: drift(c, 0.35), None),
    "E03_drift_cam1_negz": (None, lambda c: drift(c, 0.5, which=[0]), None),
    "E04_drift_cam3_posz": (None, lambda c: drift(c, 0.5, which=[2]), None),
    "E05_lens_85pct": (None, lambda c: lens_scaled(c, 0.85), None),
    "E06_glass_plus1_all": (None, lambda c: glass_shift(c, 1.0, [0, 1, 2, 3]), None),
    "E07_glass_plus1_negz": (None, lambda c: glass_shift(c, 1.0, NEG), None),
    "E08_glass_plus1_posz": (None, lambda c: glass_shift(c, 1.0, POS_), None),
    "E09_glass_minus1_negz": (None, lambda c: glass_shift(c, -1.0, NEG), None),
    "E10_thickness_plus2": ("thick", lambda c: c, None),
    "E11_n3_plus_0.005": ("n3", lambda c: c, None),
    "E12_tilt_negz_windows": ("tilt", lambda c: c, None),
    "E13_side_offset_z": (None, lambda c: move_side(c, POS_, [0.0, 0.0, 0.3]), None),
}
TRUTH_MOD = {
    "thick": lambda r: r.with_(glass_thickness=r.glass_thickness + 2.0),
    "n3": lambda r: r.with_(n_water=r.n_water + 0.005),
    "tilt": lambda r: r.with_(tilt_front=[(np.radians(0.3), 0.0)] * 2 + [(0.0, 0.0)] * 2),
}
# invariance experiments: the same as E02 but the whole rig is moved (blobs identical by construction)
MOVES = {
    "V1_rot180_about_y": (rot((0, 1, 0), np.pi), np.zeros(3), None),
    "V2_rot90_about_x": (rot((1, 0, 0), np.pi / 2), np.zeros(3), None),
    "V3_translate": (np.eye(3), np.array([6.0, -12.0, 5.0]), None),
    "V4_reverse_cameras": (np.eye(3), np.zeros(3), [3, 2, 1, 0]),
    "V5_rot_general": (rot((1, 2, 3), 0.9), np.array([3.0, 4.0, -5.0]), None),
}


def match_store(src, dst, frames, cals, cpar, vpar):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    match_correspondences_batch_parallel(frames=frames, cpar=cpar, cals=list(cals), vpar=vpar, zarr_store_path=str(dst), n_workers=1, write_to_store=True)
    return RunStore(str(dst), mode="r")


def volume_vs_truth(st, frames, truth, n=4):
    """Principal stretches and shift of the best affine map from the stored positions of the RIGHT quadruplets to the truth."""
    P, T = [], []
    for f in frames:
        pos, ids = st.read_correspondences(f); pos, ids = np.asarray(pos, float), np.asarray(ids)[:, :n]
        tg = [np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(n)]
        quad = (ids >= 0).all(1)
        pnr = np.stack([tg[c][ids[quad, c], 0] for c in range(n)], 1)
        good = (pnr == pnr[:, :1]).all(1)
        P.append(pos[quad][good]); T.append(truth[f][pnr[good, 0].astype(int)])
    P, T = np.concatenate(P), np.concatenate(T)
    a = np.c_[P, np.ones(len(P))]
    coef = np.linalg.lstsq(a, T, rcond=None)[0]
    sv = np.linalg.svd(coef[:3], compute_uv=False)
    mean_shift = np.linalg.norm(np.mean(P - T, axis=0))
    return np.sort(sv), float(np.linalg.norm(coef[3])), float(mean_shift), float(np.median(np.linalg.norm(P - T, axis=1)))


def mx(v):
    return max(v) if len(v) else float('nan')


def glass_err(model_cals, truth_cals):
    """|glass vector| of the model minus the truth, per camera (mm)."""
    return [round(float(np.linalg.norm(m.get_glass_vec()) - np.linalg.norm(t.get_glass_vec())), 3) for m, t in zip(model_cals, truth_cals)]


def px_error(model_cals, cpar_model, truth_rig, box_centre, seed=11):
    pts = np.asarray(box_centre) + np.random.default_rng(seed).uniform(-30, 30, (1500, 3))
    err = []
    for c in range(len(model_cals)):
        a = project_pixels(pts, model_cals[c], cpar_model); b = truth_rig.project_cam(c, pts)
        err.append(np.linalg.norm(a - b, axis=1))
    e = np.stack(err, 1)
    return float(np.sqrt(np.nanmean(e**2))), float(np.nanmax(e))


def run(name, truth_mod, model_fn, move=None, variants=("default",), frames=None, noise=None, n=None, tag=""):
    frames = frames or A.frames; noise = A.noise if noise is None else noise; n = n or A.n
    F = list(range(1, frames + 1))
    base = cavity_rig()
    blob_rig = TUR(base, truth_mod)
    R, t, perm = (move if move else (np.eye(3), np.zeros(3), None))
    # blobs are made in the base frame; the rig is then moved (images identical), model cameras are moved with it
    d = WORK / f"{name}{tag}"; d.mkdir(parents=True, exist_ok=True)
    src = d / "blobs.zarr"
    if src.exists():
        shutil.rmtree(src)
    truth = synthetic_run(blob_rig, src, F, n_particles=n, noise_px=noise, seed=A.seed)
    truth = {f: p @ R.T + t for f, p in truth.items()}
    truth_rig = transformed(blob_rig, R, t)
    model = model_fn([copy.deepcopy(c) for c in base.cals])
    cpar, vpar = base.cpar, base.vpar
    model_rig = TruthRig_from(model, cpar)
    model = transformed(model_rig, R, t).cals
    if perm is not None:                       # reverse the camera order everywhere: blobs, truth cameras, model cameras
        for f in F:
            for c in range(4):
                pass
        reorder_store(src, F, perm)
        truth_rig.cals = [truth_rig.cals[i] for i in perm]; truth_rig.tilt_front = [truth_rig.tilt_front[i] for i in perm]; truth_rig.tilt_back = [truth_rig.tilt_back[i] for i in perm]
        model = [model[i] for i in perm]
    centre = t
    out = []
    st0 = match_store(src, d / "m0.zarr", F, model, cpar, vpar)
    s0 = score_matching(st0, F, truth)
    sv0, sh0, ms0, e0 = volume_vs_truth(st0, F, truth)
    pe0 = px_error(model, cpar, truth_rig, centre)
    rep0 = consistency_report(st0, F, model, cpar, vpar)
    head = f"{name}{tag:12} before : recall {s0['recall']:.3f} ghosts {s0['ghost_share']:.3f} 3D err {s0['err_med'] * 1e3:6.1f} um | calib px err rms {pe0[0]:.3f} max {pe0[1]:.2f} | jump {mx(rep0.jump_um):6.1f} um [{rep0.verdict}]"
    print(head + f" | glass |g| err per camera {glass_err(model, truth_rig.cals)}", flush=True)
    for var in variants:
        kw = dict(default={}, mm=dict(free_multimedia=("d", "n3"), free_glass=True), glass=dict(free_glass=True), hold0=dict(hold_cam=0), hold3=dict(hold_cam=3), free_all=dict(hold_cam=-1), noprior=dict(edge_px=None),
                  tight=dict(edge_px=0.25))[var]
        t1 = time.perf_counter()
        res = self_calibrate_from_store(st0, F, model, cpar, vpar, min_yield=0, max_nfev=200, **kw)
        cp2 = res.cpar if res.cpar is not None else cpar
        st1 = match_store(src, d / f"m1_{var}.zarr", F, res.cals, cp2, vpar)
        s1 = score_matching(st1, F, truth)
        sv1, sh1, ms1, e1 = volume_vs_truth(st1, F, truth)
        pe1 = px_error(res.cals, cp2, truth_rig, centre)
        mmtxt = "" if not res.multimedia else " mm " + ",".join(f"{k}={v:.3f}" for k, v in res.multimedia.items())
        print(f"{'':{len(name) + len(tag)}}   {var:8} after  : recall {s1['recall']:.3f} ghosts {s1['ghost_share']:.3f} 3D err {s1['err_med'] * 1e3:6.1f} um | calib px err rms {pe1[0]:.3f} max {pe1[1]:.2f} | jump {mx(res.after.jump_um):6.1f} um | "
              f"hold {res.hold_cam} accepted {res.accepted} | glass err {glass_err(res.cals, truth_rig.cals)} | stretches vs truth {np.round(sv0, 4)} -> {np.round(sv1, 4)}, shift {sh0:.2f}->{sh1:.2f} mm{mmtxt} ({time.perf_counter() - t1:.0f}s)", flush=True)
        out.append(dict(name=name + tag, var=var, before=(s0, sv0, pe0), after=(s1, sv1, pe1)))
    return out


def TUR(rig, mod):
    return TRUTH_MOD[mod](rig) if mod else rig


def TruthRig_from(cals, cpar):
    from openptv2.synthetic_rig import TruthRig
    return TruthRig.from_cals(cals, cpar)


def reorder_store(path, frames, perm):
    st = RunStore(str(path), mode="a")
    for f in frames:
        arrs = [np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(4)]
        for c in range(4):
            st.write_targets(c, f, arrs[perm[c]])


VARIANTS = {
    "E02_drift": ("default", "hold0", "free_all"),
    "E03_drift_cam1_negz": ("default", "hold0", "free_all"),
    "E04_drift_cam3_posz": ("default", "hold0", "free_all"),
    "E01_drift_small": ("default", "hold0", "free_all"),
    "E05_lens_85pct": ("default", "hold0", "free_all"),
    "E06_glass_plus1_all": ("default", "hold0", "free_all"),
    "E07_glass_plus1_negz": ("default", "hold0", "free_all"),
    "E08_glass_plus1_posz": ("default", "hold0", "free_all"),
    "E09_glass_minus1_negz": ("default", "hold0", "free_all"),
    "E10_thickness_plus2": ("default", "hold0", "free_all"),
    "E11_n3_plus_0.005": ("default", "hold0", "free_all", "mm"),
    "E12_tilt_negz_windows": ("default", "hold0", "free_all"),
    "E13_side_offset_z": ("default", "hold0", "free_all"),
}

def recall_by_radius(st, frames, truth, cpar, bins=(0, 2, 3.5, 5, 6.5, 8, 10)):
    """Recall of the particles seen by all cameras, by the largest sensor radius (mm) of the particle's blobs, from the stored matches."""
    right = np.zeros(len(bins) - 1); possible = np.zeros(len(bins) - 1)
    cx, cy = cpar.imx / 2, cpar.imy / 2
    for f in frames:
        pos, ids = st.read_correspondences(f); ids = np.asarray(ids)[:, :4]
        tg = [np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(4)]
        rad = [np.hypot((t[:, 1] - cx) * cpar.pix_x, (t[:, 2] - cy) * cpar.pix_y) for t in tg]
        n_p = len(truth[f]); rmax = np.full(n_p, -1.0); seen = np.ones(n_p, bool)
        r_by = np.full((n_p, 4), np.nan)
        for c in range(4):
            r_by[tg[c][:, 0].astype(int), c] = rad[c]
        seen = np.isfinite(r_by).all(1); rmax = np.nanmax(r_by, 1)
        quad = (ids >= 0).all(1)
        pnr = np.stack([tg[c][ids[quad, c], 0] for c in range(4)], 1)
        good = (pnr == pnr[:, :1]).all(1)
        got = np.zeros(n_p, bool); got[pnr[good, 0].astype(int)] = True
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            m = seen & (rmax >= lo) & (rmax < hi)
            possible[i] += m.sum(); right[i] += (m & got).sum()
    return right, possible


def run_central(f_lens=0.5, frames=4, n=900, tag=""):
    """The lv mechanism on truth-known data: lens terms of the model wrong, fit data concentrated at the centre, matching judged on a cloud
    that fills the whole sensor. Compares the old (loose) lens priors with the physical ones."""
    import copy as cp
    base = cavity_rig(); cpar = base.cpar
    vpar = cp.copy(base.vpar)
    try:
        vpar.X_lay = np.array([-90.0, 90.0])
    except Exception:
        pass
    F = list(range(1, frames + 1))
    d = WORK / f"central{tag}"; d.mkdir(parents=True, exist_ok=True)
    for nm in ("fit.zarr", "eval.zarr"):
        if (d / nm).exists(): shutil.rmtree(d / nm)
    truth_fit = synthetic_run(base, d / "fit.zarr", F, n_particles=n, noise_px=A.noise, seed=5, gauss_xy=9.0, half_z=30.0)
    truth_eval = synthetic_run(base, d / "eval.zarr", F, n_particles=2 * n, noise_px=A.noise, seed=6, half_size=70.0, half_z=30.0)
    model = lens_scaled([cp.deepcopy(c) for c in base.cals], f_lens)
    stf = match_store(d / "fit.zarr", d / "fit_m.zarr", F, model, cpar, vpar)
    s0 = recall_by_radius(match_store(d / "eval.zarr", d / "eval_m0.zarr", F, model, cpar, vpar), F, truth_eval, cpar)
    s_true = recall_by_radius(match_store(d / "eval.zarr", d / "eval_mt.zarr", F, base.cals, cpar, vpar), F, truth_eval, cpar)
    print(f"[central cloud, model lens = {f_lens:.2f} x truth] fit data: {sum(len(v) for v in truth_fit.values())} particles within a few mm of the axis; px error of the model {px_error(model, cpar, base, (0, 0, 0))[0]:.2f}")
    bins = (0, 2, 3.5, 5, 6.5, 8, 10)
    print(f"  {'sensor radius (mm)':>20} " + " ".join(f"{b0:g}-{b1:g}".rjust(9) for b0, b1 in zip(bins[:-1], bins[1:])))
    pr = lambda lab, v: print(f"  {lab:>20} " + " ".join((f"{100 * r / p:8.1f}%" if p > 20 else "       --") for r, p in zip(*v)))
    print(f"  {'possible (all four)':>20} " + " ".join(f"{int(p):9d}" for p in s0[1]))
    pr("true cameras", s_true); pr("model (before)", s0)
    for name, kw in (("loose lens priors", dict(edge_px=None)), ("physical, 1 px", dict(edge_px=1.0)), ("physical, 0.25 px", dict(edge_px=0.25))):
        res = self_calibrate_from_store(stf, F, model, cpar, vpar, min_yield=0, max_nfev=200, **kw)
        s1 = recall_by_radius(match_store(d / "eval.zarr", d / f"eval_m_{name[:5]}.zarr", F, res.cals, cpar, vpar), F, truth_eval, cpar)
        pr(name, s1)
        print(f"{'':>22}calibration px error {px_error(model, cpar, base, (0, 0, 0))[0]:.2f} -> {px_error(res.cals, res.cpar or cpar, base, (0, 0, 0))[0]:.2f}; k1 change (max over cameras) {max(abs(a.added_par.k1 - b.added_par.k1) for a, b in zip(res.cals, model)):.2e}, k2 {max(abs(a.added_par.k2 - b.added_par.k2) for a, b in zip(res.cals, model)):.1e}")


if __name__ == "__main__":
    if A.exps.startswith("central"):
        run_central(f_lens=float(A.exps.split("@")[1]) if "@" in A.exps else 0.5); sys.exit(0)
    names = list(EXPS) + list(MOVES) if A.exps == "all" else A.exps.split(",")
    for nm in names:
        try:
            if nm in EXPS:
                tm, mf, _ = EXPS[nm]
                var = VARIANTS.get(nm, ("default",))
                run(nm, tm, mf, variants=var)
            elif nm in MOVES:
                run(nm, None, lambda c: drift(c, 0.35), move=MOVES[nm], variants=('default', 'free_all'))
            elif nm.startswith("N_"):                                    # N_noise0.3 / N_frames3
                if "noise" in nm: run("E02_drift", None, lambda c: drift(c, 0.35), noise=float(nm.split("noise")[1]), tag="@" + nm)
                else: run("E02_drift", None, lambda c: drift(c, 0.35), frames=int(nm.split("frames")[1]), tag="@" + nm)
        except Exception:
            print(f"!! {nm} failed:"); traceback.print_exc(limit=4)

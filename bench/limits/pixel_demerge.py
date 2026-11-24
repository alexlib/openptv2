# ruff: noqa
"""Go/no-go for pixel-level de-merging (plan 2026-11-21-tracker-setup.md, section 17.3 #1).

Rendered case with truth: Synthetic-CompleteTest/syn1 (images + truth.npz). For each detected blob that holds >= 2 true
particles (truth projected into the camera), fit one Gaussian PSF per member to the high-passed pixels, seeded at
truth + N(0, seed_px) (the track-prediction error), and measure the error against truth. Compared with:
  centroid : the blob centre itself (what a merged observation gives today)
  seed     : the prediction alone (what the tracker already knows)
  fit      : least squares, free positions
  map      : least squares with a Gaussian prior on the seed (sigma seed_px)
Reads images only.

    uv run python bench/limits/pixel_demerge.py RUN_DIR TRUTH_NPZ [--frames 3 40 90] [--seed-px 0.28]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def model(shape_xy, params, sigma, n):
    xx, yy = shape_xy
    out = np.zeros_like(xx, dtype=float)
    for j in range(n):
        x, y, a = params[3 * j: 3 * j + 3]
        out += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return out


def fit_window(win, xx, yy, seeds, sigma, prior_px, free_sigma=False, plane=False, prior_w=10.0):
    """Fit len(seeds) Gaussians (shared sigma). prior_px None: free; else MAP with that prior on positions."""
    n = len(seeds)
    p0 = []
    for sx, sy in seeds:
        ix = int(np.clip(np.floor(sx - xx[0, 0] + 0.5), 0, xx.shape[1] - 1))
        iy = int(np.clip(np.floor(sy - yy[0, 0] + 0.5), 0, xx.shape[0] - 1))
        p0 += [sx, sy, max(float(win[iy, ix]), 5.0)]
    p0 = np.array(p0)
    lo = np.tile([-np.inf, -np.inf, 0.0], n)
    hi = np.tile([np.inf, np.inf, 1e4], n)
    extra0, elo, ehi = [], [], []
    if free_sigma:
        extra0.append(sigma); elo.append(0.5); ehi.append(1.1)
    if plane:
        extra0 += [float(np.median(win)), 0.0, 0.0]; elo += [-np.inf] * 3; ehi += [np.inf] * 3
    p0f = np.concatenate([p0, extra0])
    lo = np.concatenate([lo, elo]); hi = np.concatenate([hi, ehi])
    cx0, cy0 = xx.mean(), yy.mean()

    def res(pf):
        p = pf[:3 * n]
        sg = pf[3 * n] if free_sigma else sigma
        m = model((xx, yy), p, sg, n)
        if plane:
            q = pf[3 * n + (1 if free_sigma else 0):]
            m = m + q[0] + q[1] * (xx - cx0) + q[2] * (yy - cy0)
        r = (m - win).ravel()
        if prior_px is not None:
            r = np.concatenate([r, [(p[3 * j + k] - p0[3 * j + k]) / prior_px * prior_w for j in range(n) for k in (0, 1)]])
        return r

    sol = least_squares(res, p0f, bounds=(lo, hi), method="trf", max_nfev=60)
    return sol.x[:3 * n].reshape(n, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("truth")
    ap.add_argument("--frames", type=int, nargs="+", default=[3, 40, 90])
    ap.add_argument("--seed-px", type=float, default=0.28)
    ap.add_argument("--assoc", type=float, default=2.5, help="truth within this of a blob centre belongs to it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image", choices=["hp", "raw"], default="hp", help="fit the high-passed or the raw image (raw: + local plane)")
    ap.add_argument("--free-sigma", action="store_true", help="shared sigma per blob is free (0.5..1.1)")
    ap.add_argument("--prior-w", type=float, default=10.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    run = Path(a.run_dir).resolve()
    os.chdir(run)
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import DEFAULT_HIGHPASS_FILTER_SIZE, py_start_proc_c, read_frame_images, simple_highpass
    from openptv2.segmentation import target_recognition

    t = np.load(a.truth)
    tf = t["frame"]
    print("truth frames", tf.min(), tf.max(), "pixel_xy", t["pixel_xy_per_cam"].shape)
    pm = ParameterManager()
    pm.from_yaml(run / "parameters_Run1.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    ncam = len(cals)
    base = pm.get_parameter("sequence")["base_name"]
    ptv = pm.get_parameter("ptv") or {}
    f0 = int(tf.min())

    rows = []          # (kind, n_members, err_centroid, err_seed, err_fit, err_map) per member
    singles = []
    offs = {c: [] for c in range(ncam)}
    store = []
    for f in a.frames:
        raw = read_frame_images(pm, base, ncam, f)
        sel = tf == f0 + (f - 1) if False else None
        # find truth frame matching image index: try f-1+f0 first
        tfr = f0 + (f - 1)
        m = tf == tfr
        pxy = t["pixel_xy_per_cam"][m]
        for c in range(ncam):
            hp = simple_highpass(raw[c], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))
            ts = target_recognition(hp, tpar, c, cpar)
            det = np.array([[x.pos()[0], x.pos()[1], x.count_pixels()[1], x.count_pixels()[2], x.count_pixels()[0],
                             x.sum_grey_value()] for x in ts])
            tr = pxy[:, 2 * c: 2 * c + 2]
            ok = np.isfinite(tr).all(axis=1)
            tr = tr[ok]
            store.append((f, c, (hp if a.image == 'hp' else raw[c].astype(float)), det, tr))
    # global offset between detection and truth per camera, from isolated blobs
    off = {}
    for f, c, hp, det, tr in store:
        if len(det) < 3 or len(tr) < 3:
            continue
        d, i = cKDTree(tr).query(det[:, :2])
        dn, _ = cKDTree(det[:, :2]).query(det[:, :2], k=2)
        good = (d < 1.5) & (dn[:, 1] > 8)
        offs[c].append(np.median(det[good, :2] - tr[i[good]], axis=0) if good.any() else [np.nan, np.nan])
    for c in range(ncam):
        off[c] = np.nanmedian(np.array(offs[c]), axis=0)
    print("detection - truth offset per cam (px):", {c: np.round(off[c], 3).tolist() for c in off})

    # isolated-blob PSF sigma per camera
    sig = {}
    for c in range(ncam):
        sg = []
        for f, cc, hp, det, tr in store:
            if cc != c or len(det) < 3:
                continue
            dn, _ = cKDTree(det[:, :2]).query(det[:, :2], k=2)
            for (x, y, nx, ny, n, s), d in zip(det, dn[:, 1]):
                if d < 8 or n < 5:
                    continue
                r = 4
                x0, y0 = int(round(x)) - r, int(round(y)) - r
                w = hp[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(float)
                if w.shape != (2 * r + 1, 2 * r + 1) or w.sum() <= 0:
                    continue
                w = np.clip(w, 0, None)
                yy, xx = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
                cx, cy = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
                sg.append(np.sqrt(0.5 * ((w * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / w.sum())))
        sig[c] = float(np.median(sg)) if sg else 0.8
    print("isolated-blob PSF sigma per cam:", {c: round(v, 3) for c, v in sig.items()})

    res = {"merged": [], "single": []}
    nb = 0
    for f, c, hp, det, tr in store:
        if len(det) < 3 or len(tr) < 3:
            continue
        trs = tr + off[c]
        tt = cKDTree(trs)
        for (x, y, nx, ny, n, s) in det:
            nb += 1
            members = tt.query_ball_point([x, y], a.assoc)
            if not members:
                continue
            r = int(max(nx, ny) // 2 + 4)
            x0, y0 = int(round(x)) - r, int(round(y)) - r
            if x0 < 0 or y0 < 0 or x0 + 2 * r + 1 > hp.shape[1] or y0 + 2 * r + 1 > hp.shape[0]:
                continue
            win = hp[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].astype(float)
            yy, xx = np.mgrid[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1] + 0.5   # pixel centre = index + 0.5 (targ_rec adds 0.5)
            # every truth particle whose centre lies in the window (members of this or the neighbouring blobs)
            inwin = tt.query_ball_point([x, y], r + 1.5)
            inwin = [j for j in inwin if x0 - 1 <= trs[j, 0] <= x0 + 2 * r + 2 and y0 - 1 <= trs[j, 1] <= y0 + 2 * r + 2]
            seeds_true = trs[inwin]
            seeds = seeds_true + rng.normal(0, a.seed_px, seeds_true.shape)
            kind = "merged" if len(members) >= 2 else "single"
            out = {}
            for name, prior in (("fit", None), ("map", a.seed_px)):
                try:
                    out[name] = fit_window(win, xx, yy, [tuple(sd) for sd in seeds], sig[c], prior, a.free_sigma, a.image == 'raw', a.prior_w)
                except Exception:
                    out[name] = None
            if out["fit"] is None:
                continue
            mem_k = [k for k, j in enumerate(inwin) if j in members]
            for k, j in enumerate(inwin):
                if j not in members:
                    continue
                others = [trs[inwin[q]] for q in mem_k if q != k]
                sep = min((np.hypot(*(trs[j] - o)) for o in others), default=np.nan)
                # baseline: brightness-weighted centre split (the shared_correction idea) with the fitted amplitudes
                amps = np.array([out["map"][q, 2] if out["map"] is not None else 1.0 for q in mem_k])
                w = amps / max(amps.sum(), 1e-9)
                pm_ = np.array([seeds[q] for q in mem_k])
                rr = np.array([x, y]) - (w[:, None] * pm_).sum(axis=0)
                kk = mem_k.index(k)
                e_split = np.hypot(*(seeds[k] + w[kk] * rr / (w ** 2).sum() - trs[j])) if len(mem_k) > 1 else np.nan
                e_cent = np.hypot(*(np.array([x, y]) - trs[j]))
                e_seed = np.hypot(*(seeds[k] - trs[j]))
                e_fit = np.hypot(*(out["fit"][k, :2] - trs[j]))
                e_map = np.hypot(*(out["map"][k, :2] - trs[j])) if out["map"] is not None else np.nan
                res[kind].append((len(members), e_cent, e_seed, e_fit, e_map, sep, e_split))
    np.savez(os.environ.get("DEMERGE_OUT", "/tmp/demerge_res.npz"), merged=np.array(res["merged"]), single=np.array(res["single"]))
    print(f"blobs {nb}  merged members {len(res['merged'])}  single {len(res['single'])}")
    for kind in ("single", "merged"):
        v = np.array(res[kind])
        if len(v) == 0:
            continue
        print(f"\n{kind}: {len(v)} members")
        print("  method     median   mean    p90    share>1px  share>0.5px")
        for name, col in (("centroid", 1), ("seed", 2), ("split", 6), ("fit", 3), ("map", 4)):
            e = v[:, col]
            e = e[np.isfinite(e)]
            if len(e) == 0:
                continue
            print(f"  {name:9s} {np.median(e):7.3f} {e.mean():7.3f} {np.percentile(e, 90):7.3f}   {np.mean(e > 1):7.3f}   {np.mean(e > 0.5):7.3f}")
        if kind == "merged":
            print("  by separation to the nearest co-member (map | split | seed):")
            for lo_, hi_ in ((0, 0.75), (0.75, 1.25), (1.25, 2.0), (2.0, 99)):
                w = v[(v[:, 5] >= lo_) & (v[:, 5] < hi_)]
                if len(w):
                    print(f"    sep {lo_:4.2f}-{hi_:4.2f}px  n={len(w):5d}  median map {np.median(w[:,4]):.3f}  split {np.nanmedian(w[:,6]):.3f}"
                          f"  | >1px map {np.mean(w[:,4]>1):.3f} split {np.nanmean(w[:,6]>1):.3f} seed {np.mean(w[:,2]>1):.3f}")
            for nm in (2, 3):
                w = v[v[:, 0] == nm] if nm == 2 else v[v[:, 0] >= 3]
                if len(w):
                    print(f"  members={'2' if nm == 2 else '>=3'}: n={len(w)}  median fit {np.median(w[:,3]):.3f}  map {np.nanmedian(w[:,4]):.3f}  share>1px map {np.mean(w[:,4] > 1):.3f}")


if __name__ == "__main__":
    main()

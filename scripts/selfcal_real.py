"""Joint self-calibration on the tracers of a finished run; scored by leave-one-camera-out on frames the fit never saw.

    uv run python scripts/selfcal_real.py <run folder> [--last 1072] [--every 3] [--n-fit 12000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import _experiment  # noqa: E402


def loco_rms(xy, cals, cpar, vpar):
    from openptv2.algorithms.constants import COORD_UNUSED
    from openptv2.loco import pixels_to_flat, project_pixels
    from openptv2.orientation import point_positions

    n_cams = len(cals)
    flat = np.stack([pixels_to_flat(xy[:, c], cals[c], cpar) for c in range(n_cams)], axis=1)
    out = []
    for c in range(n_cams):
        held = flat.copy()
        held[:, c] = COORD_UNUSED
        pos, _ = point_positions(held, cpar, cals, vpar, method="lsq")
        r = xy[:, c] - project_pixels(np.asarray(pos, float), cals[c], cpar)
        r = r[np.all(np.isfinite(r), axis=1)]
        out.append(float(np.sqrt(np.mean(r[np.linalg.norm(r, axis=1) < 3.0] ** 2))))
    return out


def tracks_compare(a, fin, obs, cals, new_cals, cpar, vpar, n_cams):
    import realism_metrics as rm
    from holistic_real import chains_of

    from openptv2.algorithms.constants import COORD_UNUSED
    from openptv2.loco import pixels_to_flat
    from openptv2.orientation import point_positions

    xy = obs[:, :, :2].astype(float)
    poss = []
    for name, cs in (("old calibration", cals), ("new calibration", new_cals)):
        flat = np.full((len(xy), n_cams, 2), COORD_UNUSED)
        for c in range(n_cams):
            ok = np.isfinite(xy[:, c, 0])
            flat[ok, c] = pixels_to_flat(xy[ok, c], cs[c], cpar)
        pos = np.empty((len(xy), 3))
        for lo in range(0, len(flat), 200_000):
            pos[lo: lo + 200_000] = np.asarray(point_positions(flat[lo: lo + 200_000], cpar, cs, vpar,
                                                               method="lsq")[0], float)
        poss.append(pos)
        ch = chains_of(fin["tid"], fin["frame"], pos, (fin["blob"] >= 0).sum(1))
        m = rm.metrics(ch, a.last)
        print(f"{name:16s}", {k: m[k] for k in ("mean_len", "jump_share", "jitter_median_xyz",
                                                 "median_abs_2nd_diff_xyz", "jitter_z_p90")})
    d = poss[1] - poss[0]
    ok = np.all(np.isfinite(d), axis=1)
    A = np.c_[poss[0][ok], np.ones(ok.sum())]
    coef, *_ = np.linalg.lstsq(A, d[ok], rcond=None)
    nonaff = np.linalg.norm(d[ok] - A @ coef, axis=1)
    print(f"position shift old -> new (mm): median {np.median(np.linalg.norm(d[ok], axis=1)):.4f} "
          f"p99 {np.percentile(np.linalg.norm(d[ok], axis=1), 99):.4f}; after removing a global affine map "
          f"(gauge): median {np.median(nonaff):.4f} p99 {np.percentile(nonaff, 99):.4f}; "
          f"affine part: max |linear coefficient| {np.abs(coef[:3]).max():.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--last", type=int, default=1072)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--n-fit", type=int, default=12000)
    ap.add_argument("--variants", nargs="*", default=["ext", "ext+pp", "ext+pp+k", "all"])
    ap.add_argument("--nfev", type=int, default=25)
    ap.add_argument("--ext-sigma-scale", type=float, default=1.0,
                    help="scale of the soft prior on the exterior parameters (smaller = the cameras move less)")
    ap.add_argument("--tracks", action="store_true",
                    help="re-triangulate every observation of the run with the old and with the new calibration of "
                         "the last variant and compare the track metrics (jitter, jump share)")
    ap.add_argument("--save", default="", help="write the new calibration of the last variant to this folder (.ori/.addpar)")
    a = ap.parse_args()

    from openptv2.merge_post import arrays_from_store
    from openptv2.selfcal import joint_self_calibrate
    from openptv2.storage import RunStore

    run = Path(a.run)
    exp = _experiment(run, run / "parameters_Run1.yaml", 1, a.last)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    st = RunStore(run / "res" / "run.zarr", mode="r")
    fin = arrays_from_store(st, "ptv_is", 1, a.last, n_cams)
    frames = fin["frame"]
    obs = np.vstack([o for o in (st.read_observations(f) for f in range(1, a.last + 1)) if o is not None])
    clean = (fin["blob"] >= 0).all(1) & (obs[:, :, 3] == 1).all(1) & (frames % a.every == 0)
    xy, pos0, fr = fin["centre"][clean], fin["pos"][clean], frames[clean]
    odd = (fr // a.every) % 2 == 1
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(np.flatnonzero(odd), min(a.n_fit, odd.sum()), replace=False)
    test = ~odd
    print(f"clean 4-camera points {clean.sum()}; fit on {len(fit_idx)} (odd frames), test on {test.sum()} (even frames)")
    base = loco_rms(xy[test], cals, cpar, vpar)
    print("baseline LOCO rms per camera (px, held-out):", np.round(base, 4).tolist(), "mean", round(float(np.mean(base)), 4))
    kinds = {
        "ext": dict(free_pp=False, free_radial=(), free_decentering=False),
        "ext+pp": dict(free_pp=True, free_radial=(), free_decentering=False),
        "ext+pp+k": dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=False),
        "all": dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=True),
        "all+cc": dict(free_pp=True, free_cc=True, free_radial=("k1", "k2"), free_decentering=True),
        "all+k3": dict(free_pp=True, free_cc=True, free_radial=("k1", "k2", "k3"), free_decentering=True),
        "ext+pp+p": dict(free_pp=True, free_radial=(), free_decentering=True),
        "pp+p": dict(free_ext=False, free_pp=True, free_radial=(), free_decentering=True),
        "pp+k+p": dict(free_ext=False, free_pp=True, free_radial=("k1", "k2"), free_decentering=True),
    }
    for name in a.variants:
        t0 = time.time()
        res = joint_self_calibrate(xy[fit_idx], pos0[fit_idx], cals, cpar, max_nfev=a.nfev, ext_sigma_scale=a.ext_sigma_scale, **kinds[name])
        after = loco_rms(xy[test], res.cals, cpar, vpar)
        print(f"{name:10s} params {res.n_params:3d}  fit rms {res.rms_before_px:.4f} -> {res.rms_after_px:.4f} px | "
              f"nfev {res.nfev} ok={res.success} | held-out LOCO per camera {np.round(after, 4).tolist()} mean {np.mean(after):.4f} "
              f"({100 * (np.mean(after) / np.mean(base) - 1):+.1f} %)  [{time.time() - t0:.0f} s]")
        if name == a.variants[-1]:
            for c, ch in res.change.items():
                print("  camera", c, "parameter changes:", ch)
            last_cals = res.cals
    if a.tracks:
        tracks_compare(a, fin, obs, cals, last_cals, cpar, vpar, n_cams)
    if a.save:
        out = Path(a.save)
        out.mkdir(parents=True, exist_ok=True)
        for c, cal in enumerate(last_cals):
            cal.write(str(out / f"cam_{c + 1}.tif.ori"), str(out / f"cam_{c + 1}.tif.addpar"))
        print("calibration written to", out)


if __name__ == "__main__":
    main()

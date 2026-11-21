"""N4 feasibility: is the leave-one-camera-out residual systematic (a calibration error) or random (detection noise)?

Clean 4-camera points (no shared blob) of a finished run: each camera in turn is left out, the point is triangulated from
the other three and projected into that camera; the residual vector (observed - projected, px) is collected per camera.
If a smooth function of the image position (and depth) explains a good part of the residual variance, on frames the fit
did not see, the calibration can be improved from the tracks; if not, the floor is detection noise.

    uv run python scripts/loco_map.py <run folder> [--last 1072] [--every 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import _experiment  # noqa: E402


def design(u, v, z, deg):
    cols = [np.ones_like(u)]
    for i in range(1, deg + 1):
        for j in range(i + 1):
            cols.append(u ** (i - j) * v**j)
    if z is not None:
        cols += [z, z * u, z * v]
    return np.stack(cols, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--last", type=int, default=1072)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--apply", action="store_true",
                    help="correct every observation with the field fitted on odd frames, re-triangulate, compare the "
                         "track metrics before and after")
    a = ap.parse_args()

    from openptv2.algorithms.constants import COORD_UNUSED
    from openptv2.loco import pixels_to_flat, project_pixels
    from openptv2.merge_post import arrays_from_store
    from openptv2.orientation import point_positions
    from openptv2.storage import RunStore

    run = Path(a.run)
    exp = _experiment(run, run / "parameters_Run1.yaml", 1, a.last)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    st = RunStore(run / "res" / "run.zarr", mode="r")
    fin = arrays_from_store(st, "ptv_is", 1, a.last, n_cams)
    frames = fin["frame"]
    obs = np.vstack([o for o in (st.read_observations(f) for f in range(1, a.last + 1)) if o is not None])
    assert len(obs) == len(frames)
    blob = fin["blob"]
    clean = (blob >= 0).all(1) & (obs[:, :, 3] == 1).all(1) & (frames % a.every == 0)
    xy = fin["centre"][clean]
    pos0 = fin["pos"][clean]
    fr = frames[clean]
    print(f"clean 4-camera points used: {clean.sum()} of {len(frames)}")
    flat = np.stack([pixels_to_flat(xy[:, c], cals[c], cpar) for c in range(n_cams)], axis=1)
    res = {}
    for c in range(n_cams):
        held = flat.copy()
        held[:, c] = COORD_UNUSED
        pos, _ = point_positions(held, cpar, cals, vpar, method="lsq")
        proj = project_pixels(np.asarray(pos, float), cals[c], cpar)
        r = xy[:, c] - proj
        res[c] = (r, xy[:, c], np.asarray(pos, float))
    print(f"{'cam':3s} {'rms px':>7s} {'|mean| px':>9s}  field fit (train odd / test even frames): variance explained"
          f"  rms after  (image position only | + depth)")
    odd = (fr // a.every) % 2 == 1
    fields: list = []
    for c in range(n_cams):
        r, p, x3 = res[c]
        ok = np.all(np.isfinite(r), axis=1) & (np.linalg.norm(r, axis=1) < 3.0)
        u = (p[:, 0] - cpar.imx / 2) / (cpar.imx / 2)
        v = (p[:, 1] - cpar.imy / 2) / (cpar.imy / 2)
        z = (x3[:, 2] - np.nanmean(x3[:, 2])) / np.nanstd(x3[:, 2])
        out = []
        for zz in (None, z):
            A = design(u, v, zz, a.deg)
            ve = []
            after = []
            for ax in range(2):
                coef, *_ = np.linalg.lstsq(A[ok & odd], r[ok & odd, ax], rcond=None)
                test = ok & ~odd
                pred = A[test] @ coef
                ve.append(1 - np.var(r[test, ax] - pred) / np.var(r[test, ax]))
                after.append(r[test, ax] - pred)
            after = np.sqrt(np.mean(np.concatenate(after) ** 2))
            out.append((np.mean(ve), after))
        fields.append((np.linalg.lstsq(design(u, v, z, a.deg)[ok & odd], r[ok & odd], rcond=None)[0],
                       np.nanmean(x3[:, 2]), np.nanstd(x3[:, 2])))
        rms = np.sqrt(np.mean(r[ok] ** 2))
        mean_v = np.linalg.norm(r[ok].mean(0))
        print(f"{c + 1:3d} {rms:7.3f} {mean_v:9.3f}  x,y mean over axes: {out[0][0]:6.3f} -> rms {out[0][1]:.3f}"
              f"  |  with depth {out[1][0]:6.3f} -> rms {out[1][1]:.3f}")
    if a.apply:
        apply_and_compare(a, fields, fin, obs, cpar, cals, vpar, n_cams)




def apply_and_compare(a, fields, fin, obs, cpar, cals, vpar, n_cams):
    """Shift every observation by the fitted field, re-triangulate, and score the tracks before and after."""
    import realism_metrics as rm
    from holistic_real import chains_of

    from openptv2.algorithms.constants import COORD_UNUSED
    from openptv2.loco import pixels_to_flat
    from openptv2.orientation import point_positions

    xy = obs[:, :, :2].astype(float)
    pos_now = fin["pos"]
    z3 = pos_now[:, 2]
    flat0 = np.full((len(xy), n_cams, 2), COORD_UNUSED)
    flat1 = flat0.copy()
    for c in range(n_cams):
        ok = np.isfinite(xy[:, c, 0])
        coef, zm, zs = fields[c]
        u = (xy[ok, c, 0] - cpar.imx / 2) / (cpar.imx / 2)
        v = (xy[ok, c, 1] - cpar.imy / 2) / (cpar.imy / 2)
        corr = design(u, v, (z3[ok] - zm) / zs, a.deg) @ coef
        flat0[ok, c] = pixels_to_flat(xy[ok, c], cals[c], cpar)
        flat1[ok, c] = pixels_to_flat(xy[ok, c] + corr, cals[c], cpar)
    out = {}
    for name, flat in (("before", flat0), ("after", flat1)):
        pos = np.empty_like(pos_now)
        for lo in range(0, len(flat), 200_000):
            pos[lo: lo + 200_000] = np.asarray(point_positions(flat[lo: lo + 200_000], cpar, cals, vpar,
                                                               method="lsq")[0], float)
        ch = chains_of(fin["tid"], fin["frame"], pos, (fin["blob"] >= 0).sum(1))
        out[name] = rm.metrics(ch, a.last)
        print(name, {k: out[name][k] for k in ("mean_len", "jump_share", "jitter_median_xyz",
                                                "median_abs_2nd_diff_xyz", "jitter_z_p90")})
    return out


if __name__ == "__main__":
    main()

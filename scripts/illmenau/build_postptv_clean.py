"""Post-PTV clean chain for Ilmenau rigs (reconnect -> repair -> kink-split -> SG -> stitch -> SG).

Reads raw linkage from a run.zarr, applies the cloud-post reconnect+repair,
splits violent fast kinks into FRESH trajectories (never reuse parent ids),
then Savitzky-Golay smooth -> conservative stitch -> smooth, and saves a
 standalone store. Replaces the cloud stitch step (which does not scale to
dense 1-4 linkage: 45+ min stuck) with flowtracks' reference stitcher, and
adds the kink-split that cloud post lacks.

Every setting comes from the `postptv_clean:` section of the rig's parameter
file (RIGDIR/parameters_Run1.yaml, or --params); a command-line flag overrides
that, and a setting missing from both falls back to the built-in default below.
The effective values and their source are printed at the start.

    postptv_clean:
      fps: 10.0                 # frame rate [Hz]: velocities/accelerations and
                                # the stitch velocity limits are in m/s
      reconnect_max_gap: 10     # reconnect: max frame gap
      reconnect_tol_sigmas: 4.0 # reconnect: tolerance in sigmas
      sg_window: 7              # Savitzky-Golay window / order
      sg_order: 2
      kink_deg: 90.0            # kink split: turn angle ...
      kink_step_mm: 25.0        # ... on steps longer than this
      gap1: 12                  # first stitch: max gap [frames],
      dist1: 0.05               #   max distance [m],
      vd1: 0.5                  #   max velocity difference [m/s]
      gap2: 8                   # second (tight) stitch on clean ends
      dist2: 0.025
      vd2: 0.35
      win_long: 0               # >0: final smoothing (win_long, order_long)
      order_long: 3             #   for tracks with >= len_long points
      len_long: 25

Usage:
    uv run --project /Users/alex/Documents/Github/openptv-cloud python \
        scripts/illmenau/build_postptv_clean.py RIGDIR [--first F] [--last L] \
        [--store run_fulldiam.zarr] [--out run_fulldiam_clean.zarr] [--fps 10 ...]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/alex/Documents/Github/postptv")

from flowtracks.io import save_zarr_trajectories
from flowtracks.repair import repair_arrays
from flowtracks.smoothing import savitzky_golay
from flowtracks.stitching import stitch_trajectories
from flowtracks.trajectory import Trajectory
from openptv_cloud import experiment as E
from openptv_cloud import post as P

from openptv2.reconnect import reconnect_pieces


def build_trajs(trajid_all, time_all, pos_all):
    order = np.argsort(trajid_all, kind="stable")
    tj, tm, pp = trajid_all[order], time_all[order], pos_all[order]
    bounds = np.flatnonzero(np.diff(tj)) + 1
    starts = np.r_[0, bounds]
    out = []
    for i, s in enumerate(starts):
        e = int(bounds[i]) if i < len(bounds) else len(tj)
        if e - s >= 2:
            o = np.argsort(tm[s:e], kind="stable")
            out.append(Trajectory(pp[s:e][o], np.zeros_like(pp[s:e][o]),
                                  tm[s:e][o], int(tj[s])))
    return out


def kink_split(trajs, max_deg, step_min, fresh_start=10_000_000):
    """Split fast violent reversals; every cut piece gets a FRESH trajid so
    trajid-grouped readers (plots, Scene) cannot recombine the chimera."""
    out, ncuts, nxt = [], 0, fresh_start
    for tr in trajs:
        v = tr.velocity()
        n = np.linalg.norm(v, axis=1)
        t, p = np.asarray(tr.time()), tr.pos()
        st = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ok = (n[:-1] > 1e-12) & (n[1:] > 1e-12)
        cosang = np.ones(len(v) - 1)
        cosang[ok] = np.clip((v[:-1][ok] * v[1:][ok]).sum(1) / (n[:-1][ok] * n[1:][ok]), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        instep = np.concatenate([st[:1], st[:-1]])
        bad = np.flatnonzero((ang > max_deg) & (instep > step_min) & (st > step_min)) + 1
        ncuts += len(bad)
        segs = np.split(np.arange(len(t)), bad)
        if len(segs) == 1:
            out.append(tr)
            continue
        for seg in segs:
            if len(seg) >= 2:
                out.append(Trajectory(p[seg], v[seg], t[seg], nxt))
                nxt += 1
    return out, ncuts


# built-in defaults: the values this script used before they moved into the
# parameter file (fps 50 is NOT the Ilmenau frame rate -- set it in the file)
DEFAULTS = dict(
    fps=50.0, reconnect_max_gap=10, reconnect_tol_sigmas=4.0, sg_window=7,
    sg_order=2, kink_deg=90.0, kink_step_mm=25.0, gap1=12, dist1=0.05, vd1=0.5,
    gap2=8, dist2=0.025, vd2=0.35, win_long=0, order_long=3, len_long=25,
)


def resolve_settings(rigdir, params, cli):
    """DEFAULTS <- parameter file `postptv_clean:` <- command line."""
    from pathlib import Path

    import yaml

    path = Path(params) if params else Path(rigdir) / "parameters_Run1.yaml"
    section = {}
    if path.exists():
        section = (yaml.safe_load(path.read_text()) or {}).get("postptv_clean") or {}
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise SystemExit(f"unknown postptv_clean keys in {path}: {unknown}")
    cfg, source = {}, {}
    for k, v in DEFAULTS.items():
        if cli.get(k) is not None:
            cfg[k], source[k] = cli[k], "command line"
        elif k in section:
            cfg[k], source[k] = type(v)(section[k]), path.name
        else:
            cfg[k], source[k] = v, "built-in default"
    if not section:
        print(f"WARNING: no postptv_clean section in {path}; built-in defaults "
              f"(fps {DEFAULTS['fps']:g}) unless given on the command line", flush=True)
    print("postptv_clean settings:", flush=True)
    for k in DEFAULTS:
        print(f"  {k:22s} {cfg[k]!s:>8s}   ({source[k]})", flush=True)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rigdir")
    ap.add_argument("--params", default=None,
                    help="parameter file (default RIGDIR/parameters_Run1.yaml)")
    ap.add_argument("--first", type=int, default=10001)
    ap.add_argument("--last", type=int, default=10500)
    ap.add_argument("--store", default="run.zarr",
                    help="run store filename inside rigdir/res")
    ap.add_argument("--out", default="run_postptv_clean.zarr")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v), default=None,
                        help=f"override the parameter file (built-in default {v})")
    args = ap.parse_args()
    c = resolve_settings(args.rigdir, args.params, vars(args))
    sgw, sgo = c["sg_window"], c["sg_order"]

    t0 = time.perf_counter()
    store = args.rigdir.rstrip("/") + "/res/" + args.store
    trajid0, time0, pos0 = P.read_zarr_linkage_arrays(store, first=args.first, last=args.last)
    print(f"linkage rows: {len(trajid0)}", flush=True)
    tj = reconnect_pieces(trajid0, time0, pos0, max_gap=c["reconnect_max_gap"],
                          tol_sigmas=c["reconnect_tol_sigmas"])
    tj, rep = repair_arrays(tj, time0, pos0, **P.repair_kwargs(E.trajectory_repair(args.rigdir)))
    print(f"reconnect+repair: cut={rep.get('links_cut')} joins={rep.get('joins')}", flush=True)
    s1 = savitzky_golay(build_trajs(tj, time0, pos0), c["fps"], sgw, sgo, min_window=5)
    split, ncuts = kink_split(s1, c["kink_deg"], c["kink_step_mm"] / 1000.0)
    print(f"kinksplit(>{c['kink_deg']}deg, steps>{c['kink_step_mm']}mm): "
          f"{len(s1)} -> {len(split)} objs, {ncuts} cuts", flush=True)
    s2 = savitzky_golay(split, c["fps"], sgw, sgo, min_window=5)
    st = stitch_trajectories(s2, fps=c["fps"], max_gap=c["gap1"],
                             max_distance=c["dist1"], max_vel_diff=c["vd1"])
    s2b = savitzky_golay(st, c["fps"], sgw, sgo, min_window=5)
    split2, ncuts2 = kink_split(s2b, c["kink_deg"], c["kink_step_mm"] / 1000.0,
                                fresh_start=20_000_000)
    s2c = savitzky_golay(split2, c["fps"], sgw, sgo, min_window=5)
    st2 = stitch_trajectories(s2c, fps=c["fps"], max_gap=c["gap2"],
                              max_distance=c["dist2"], max_vel_diff=c["vd2"])
    if c["win_long"] > 0:
        short = [t for t in st2 if len(t) < c["len_long"]]
        long = [t for t in st2 if len(t) >= c["len_long"]]
        s_short = savitzky_golay(short, c["fps"], sgw, sgo, min_window=5)
        s_long = savitzky_golay(long, c["fps"], c["win_long"], c["order_long"],
                                min_window=7)
        s3 = s_short + s_long
        print(f"two-tier final smooth: {len(s_short)}x(7,2) + {len(s_long)}x"
              f"({c['win_long']},{c['order_long']})", flush=True)
    else:
        s3 = savitzky_golay(st2, c["fps"], sgw, sgo, min_window=5)
    lens = np.array([len(t) for t in s3])
    print(f"final: n={len(s3)} joined1={len(s2) - len(st)} joined2={len(s2c) - len(st2)} "
          f"med={np.median(lens):.0f} max={lens.max()} rows={lens.sum()} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    save_zarr_trajectories(s3, args.rigdir.rstrip("/") + "/res/" + args.out, overwrite=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

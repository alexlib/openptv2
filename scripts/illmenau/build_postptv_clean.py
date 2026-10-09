"""Post-PTV clean chain for Ilmenau rigs (reconnect -> repair -> kink-split -> SG -> stitch -> SG).

Reads raw linkage from a run.zarr, applies the cloud-post reconnect+repair,
splits violent fast kinks into FRESH trajectories (never reuse parent ids),
then Savitzky-Golay smooth -> conservative stitch -> smooth, and saves a
 standalone store. Replaces the cloud stitch step (which does not scale to
dense 1-4 linkage: 45+ min stuck) with flowtracks' reference stitcher, and
adds the kink-split that cloud post lacks.

Usage:
    uv run --project /Users/alex/Documents/Github/openptv-cloud python \
        scripts/illmenau/build_postptv_clean.py RIGDIR [--first F] [--last L] \
        [--out run_postptv_clean.zarr] [--kink-deg 90] [--kink-step-mm 25]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rigdir")
    ap.add_argument("--first", type=int, default=10001)
    ap.add_argument("--last", type=int, default=10500)
    ap.add_argument("--store", default="run.zarr",
                    help="run store filename inside rigdir/res")
    ap.add_argument("--out", default="run_postptv_clean.zarr")
    ap.add_argument("--kink-deg", type=float, default=90.0)
    ap.add_argument("--kink-step-mm", type=float, default=25.0)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--win-long", type=int, default=0,
                    help="if >0, final smoothing uses (win-long, order-long) for "
                         "tracks with >= len-long points, (7,2) otherwise")
    ap.add_argument("--order-long", type=int, default=3)
    ap.add_argument("--len-long", type=int, default=25)
    ap.add_argument("--gap1", type=int, default=12,
                    help="first (aggressive) stitch max_gap (kink-split guards it)")
    ap.add_argument("--dist1", type=float, default=0.05)
    ap.add_argument("--vd1", type=float, default=0.5)
    ap.add_argument("--gap2", type=int, default=8,
                    help="second (tight) stitch max_gap on clean ends")
    ap.add_argument("--dist2", type=float, default=0.025)
    ap.add_argument("--vd2", type=float, default=0.35)
    args = ap.parse_args()

    t0 = time.perf_counter()
    store = args.rigdir.rstrip("/") + "/res/" + args.store
    trajid0, time0, pos0 = P.read_zarr_linkage_arrays(store, first=args.first, last=args.last)
    print(f"linkage rows: {len(trajid0)}", flush=True)
    tj = reconnect_pieces(trajid0, time0, pos0, max_gap=10, tol_sigmas=4.0)
    tj, rep = repair_arrays(tj, time0, pos0, **P.repair_kwargs(E.trajectory_repair(args.rigdir)))
    print(f"reconnect+repair: cut={rep.get('links_cut')} joins={rep.get('joins')}", flush=True)
    s1 = savitzky_golay(build_trajs(tj, time0, pos0), args.fps, 7, 2, min_window=5)
    split, ncuts = kink_split(s1, args.kink_deg, args.kink_step_mm / 1000.0)
    print(f"kinksplit(>{args.kink_deg}deg, steps>{args.kink_step_mm}mm): "
          f"{len(s1)} -> {len(split)} objs, {ncuts} cuts", flush=True)
    s2 = savitzky_golay(split, args.fps, 7, 2, min_window=5)
    st = stitch_trajectories(s2, fps=args.fps, max_gap=args.gap1,
                             max_distance=args.dist1, max_vel_diff=args.vd1)
    s2b = savitzky_golay(st, args.fps, 7, 2, min_window=5)
    split2, ncuts2 = kink_split(s2b, args.kink_deg, args.kink_step_mm / 1000.0,
                                fresh_start=20_000_000)
    s2c = savitzky_golay(split2, args.fps, 7, 2, min_window=5)
    st2 = stitch_trajectories(s2c, fps=args.fps, max_gap=args.gap2,
                              max_distance=args.dist2, max_vel_diff=args.vd2)
    if args.win_long > 0:
        short = [t for t in st2 if len(t) < args.len_long]
        long = [t for t in st2 if len(t) >= args.len_long]
        s_short = savitzky_golay(short, args.fps, 7, 2, min_window=5)
        s_long = savitzky_golay(long, args.fps, args.win_long, args.order_long,
                                min_window=7)
        s3 = s_short + s_long
        print(f"two-tier final smooth: {len(s_short)}x(7,2) + {len(s_long)}x"
              f"({args.win_long},{args.order_long})", flush=True)
    else:
        s3 = savitzky_golay(st2, args.fps, 7, 2, min_window=5)
    lens = np.array([len(t) for t in s3])
    print(f"final: n={len(s3)} joined1={len(s2) - len(st)} joined2={len(s2c) - len(st2)} "
          f"med={np.median(lens):.0f} max={lens.max()} rows={lens.sum()} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    save_zarr_trajectories(s3, args.rigdir.rstrip("/") + "/res/" + args.out, overwrite=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

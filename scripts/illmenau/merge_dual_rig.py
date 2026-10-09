"""Merge the two rigs' cleaned trajectories into ONE barrel-frame dataset.

Identical treatment: both inputs must be built by build_postptv_clean.py with
the same args from run_fulldiam.zarr stores. This script only:
  1. converts both to millimetres in the shared barrel frame
     (barrel = rig + [0,-1175,0]; inter-rig yaw assumed 0, see below),
  2. drops points outside the barrel cylinder (r>3575 or |Y|>1790),
  3. de-duplicates the centre overlap slab: same frame + closer than
     --dup-mm keeps ONE copy — the longer trajectory wins (ties: rig 1-4),
  4. writes merged.zarr with the same trajectories schema (time/pos/trajid/
     vel/accel), rig origin kept per trajectory (trajid namespace per rig).

The residual inter-rig alignment (yaw about Y + ~2-4 mm anchor offsets) is NOT
solved here; --dup-mm (default 20 mm) comfortably exceeds it while staying an
order below the ~300 mm inter-particle spacing. A future joint alignment will
only shrink the duplicates' pair distances.

Usage:
    uv run python scripts/illmenau/merge_dual_rig.py [--dup-mm 20]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

RAW = Path("/Users/alex/Downloads/Ilmenau")
# SAME convention as run_postptv_clean.zarr: metres, barrel frame.
# (linkage stores are mm; cleaned trajectories are m.)
Y_SHIFT_M = -1.175
R_WALL_M, H_BARREL_M = 3.575, 3.58


def load_clean(rig, name):
    import zarr

    g = zarr.open(str(RAW / rig / "res" / name), mode="r")
    t = g["trajectories"]
    out = {k: np.asarray(t[k]) for k in
           ("time", "pos", "trajid", "vel", "accel")}
    return out


def to_barrel_m(d):
    """Cleaned stores are metres in rig frame; shift Y into barrel frame."""
    pos = np.asarray(d["pos"], float)
    assert np.abs(pos).max() < 50, "expected metres"
    pos = pos.copy()
    pos[:, 1] += Y_SHIFT_M
    d["pos"] = pos
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup-mm", type=float, default=20.0,
                    help="same-frame duplicate distance [mm]")
    ap.add_argument("--a", default="openptv_illmenau_4cam")
    ap.add_argument("--b", default="openptv_illmenau_5678")
    ap.add_argument("--clean", default="run_fulldiam_clean.zarr")
    ap.add_argument("--out", default="merged_barrel.zarr")
    ap.add_argument("--tag-a", type=int, default=1_000_000)
    ap.add_argument("--tag-b", type=int, default=2_000_000)
    a = ap.parse_args()
    t0 = time.time()

    from scipy.spatial import cKDTree

    sys.path.insert(0, "/Users/alex/Documents/Github/postptv")
    from flowtracks.io import save_zarr_trajectories
    from flowtracks.trajectory import Trajectory

    da = to_barrel_m(load_clean(a.a, a.clean))
    db = to_barrel_m(load_clean(a.b, a.clean))
    print(f"loaded: 1-4 rows={len(da['time'])} 5-8 rows={len(db['time'])}",
          flush=True)

    def inside(p):
        return ((np.hypot(p[:, 0], p[:, 2]) <= R_WALL_M)
                & (np.abs(p[:, 1]) <= H_BARREL_M / 2))

    for d in (da, db):
        m = inside(d["pos"])
        print(f"inside walls: {m.sum()}/{len(m)}", flush=True)
        for k in d:
            d[k] = d[k][m]

    # per-trajectory rows, with rig-tagged ids
    def split(d, tag, rig):
        tids = np.asarray(d["trajid"])
        order = np.argsort(tids, kind="stable")
        out = {}
        for tid in np.unique(tids):
            m = tids == tid
            o = np.argsort(np.asarray(d["time"])[m], kind="stable")
            idx = np.where(m)[0][o]
            out[tag + int(tid)] = dict(
                idx=idx, time=np.asarray(d["time"])[idx],
                pos=np.asarray(d["pos"])[idx],
                vel=np.asarray(d["vel"])[idx],
                accel=np.asarray(d["accel"])[idx], rig=rig)
        return out

    ta = split(da, a.tag_a, "14")
    tb = split(db, a.tag_b, "58")
    print(f"trajectories: 1-4={len(ta)} 5-8={len(tb)}", flush=True)

    # index B rows per frame for duplicate search
    bframe = {}
    for tid, t in tb.items():
        for f, p in zip(t["time"], t["pos"]):
            bframe.setdefault(int(f), []).append((tid, p))
    btree = {f: (cKDTree(np.array([p for _, p in v])), v)
             for f, v in bframe.items()}
    dup_m = a.dup_mm / 1000.0

    drop_b = set()  # losing trajectory ids (whole track dropped on any dup)
    lendiff = []
    for tid, t in ta.items():
        for f, p in zip(t["time"], t["pos"]):
            if int(f) not in btree:
                continue
            tree, v = btree[int(f)]
            dist, j = tree.query(p, k=1)
            if dist < dup_m:
                btid = v[j][0]
                la, lb = len(t["time"]), len(tb[btid]["time"])
                lendiff.append(la - lb)
                if la >= lb:
                    drop_b.add(btid)
                # TA always survives (tie rule); B drops only when shorter.
                # A longer B duplicate is kept too (both survive) — counted.
    kept_b = [t for tid, t in tb.items() if tid not in drop_b]
    print(f"duplicates: {len(drop_b)} B-tracks dropped "
          f"(median len diff A-B: {np.median(lendiff):.0f})", flush=True)

    trajs = []
    for k, t in enumerate(list(ta.values()) + kept_b):
        trajs.append(Trajectory(t["pos"], t["vel"], t["time"], 1000 + k,
                                accel=t["accel"]))
    dst = str(RAW / a.out)
    save_zarr_trajectories(trajs, dst, overwrite=True)
    lens = np.array([len(t) for t in trajs])
    print(f"merged: n={len(trajs)} med={np.median(lens):.0f} max={lens.max()} "
          f"rows={lens.sum()} -> {dst} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

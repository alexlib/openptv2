"""Merge the two rigs' cleaned trajectories into ONE barrel-frame dataset.

Identical treatment: both inputs must be built by build_postptv_clean.py with
the same args from run_fulldiam.zarr stores. This script:
  1. MEASURES how rig 5-8's world sits in rig 1-4's, from the bubbles both
     rigs see in the same frame (the readme: one calibration image defines
     both worlds; cams 5-8 fire 3 ms after cams 1-4, so a bubble moves < 3 mm
     between them).  A 0 and a 180 deg turn about the vertical are tried; the
     one whose rig-to-rig offsets form a sharp cluster wins, and the offset is
     its median.  No sharp cluster -> refuse to merge.  Which turn applies
     depends on how the back face of the plate was labelled when rig 5-8 was
     calibrated, so it is measured, never assumed,
  2. maps rig 5-8 into rig 1-4 (positions, and velocity/acceleration rotated),
     then both into the barrel frame (barrel = rig 1-4 + [0,-1175,0] mm: the
     datum dot is on the axis, 615 mm above the heating plate),
  3. drops points outside the barrel cylinder (r>3575 or |Y|>1790),
  4. de-duplicates the overlap: same frame + closer than --dup-mm keeps ONE
     copy — the longer trajectory wins (ties: rig 1-4).  --keep-duplicates
     keeps every track and only marks them (track_info/duplicate),
  5. writes the store with the same trajectories schema (time/pos/trajid/
     vel/accel), plus track_info/{trajid, rig, partner, shared_points,
     duplicate}.  Rig 1-4 trajectories come first (trajid < 1000 + n_rig14);
     the measured rig transform is stored in the attributes.

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


def estimate_rig_b_in_a(dir_a, dir_b, store="run_fulldiam.zarr", step=10):
    """Rigid map rig B -> rig A from bubbles both rigs triangulate.

    Returns (M, off_mm, n_pairs, spread_mm) with p_A = M @ p_B - off (mm).
    """
    from scipy.spatial import cKDTree

    from openptv2.storage import RunStore

    sa = RunStore(str(RAW / dir_a / "res" / store), mode="r")
    sb = RunStore(str(RAW / dir_b / "res" / store), mode="r")
    frames = sorted(set(sa.frames(source="correspondences"))
                    & set(sb.frames(source="correspondences")))[::step]

    def four_cam(st, f):
        p, i = st.read_correspondences(f)
        return p[(i >= 0).sum(1) == 4]

    pts = [(four_cam(sa, f), four_cam(sb, f)) for f in frames]
    best = None
    for yaw in (0, 180):
        c = 1.0 if yaw == 0 else -1.0
        M = np.diag([c, 1.0, c])
        diffs = []
        for pa, pb in pts:
            if not len(pa) or not len(pb):
                continue
            pbm = pb @ M.T
            for k, js in enumerate(cKDTree(pbm).query_ball_point(pa, 150.0)):
                if js:
                    diffs.append(pbm[js] - pa[k])
        if not diffs:
            continue
        d = np.vstack(diffs)
        H, edges = np.histogramdd(d, bins=50, range=[(-150, 150)] * 3)
        i = np.unravel_index(H.argmax(), H.shape)
        peak = np.array([edges[k][i[k]] + 3.0 for k in range(3)])
        score = H.max() / max(H.mean(), 1e-9)
        print(f"  rig transform: yaw {yaw:3d} deg -> densest 6 mm bin {int(H.max())} "
              f"({score:.0f}x the mean)", flush=True)
        if best is None or H.max() > best[1]:
            best = (M, H.max(), score, peak, yaw)
    if best is None or best[1] < 50 or best[2] < 100:
        raise SystemExit("no common bubbles between the rigs: refusing to merge")
    M, _, _, off, yaw = best
    for _ in range(2):  # refine the offset on tight pairs
        dd = []
        for pa, pb in pts:
            if not len(pa) or not len(pb):
                continue
            q = pb @ M.T - off
            dist, j = cKDTree(q).query(pa, distance_upper_bound=8.0)
            ok = np.isfinite(dist)
            dd.append(pb[j[ok]] @ M.T - pa[ok])
        dd = np.vstack(dd)
        off = np.median(dd, axis=0)
    spread = np.median(np.abs(dd - off), axis=0)
    print(f"  rig 5-8 -> rig 1-4: yaw {yaw} deg, offset {np.round(off, 2)} mm, "
          f"{len(dd)} pairs, spread (MAD) {np.round(spread, 2)} mm", flush=True)
    return M, off, len(dd), spread


def to_rig_a(d, M, off_mm):
    """Rig B trajectories (metres, rig B frame) into rig A's frame."""
    d["pos"] = np.asarray(d["pos"], float) @ M.T - off_mm / 1000.0
    d["vel"] = np.asarray(d["vel"], float) @ M.T
    d["accel"] = np.asarray(d["accel"], float) @ M.T
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup-mm", type=float, default=20.0,
                    help="same-frame duplicate distance [mm]")
    ap.add_argument("--a", default="openptv_illmenau_4cam")
    ap.add_argument("--b", default="openptv_illmenau_5678")
    ap.add_argument("--clean", default="run_fulldiam_clean.zarr")
    ap.add_argument("--out", default="merged_barrel.zarr")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep rig 5-8 duplicates, marked in track_info")
    ap.add_argument("--tag-a", type=int, default=1_000_000)
    ap.add_argument("--tag-b", type=int, default=2_000_000)
    a = ap.parse_args()
    t0 = time.time()

    from scipy.spatial import cKDTree

    sys.path.insert(0, "/Users/alex/Documents/Github/postptv")
    from flowtracks.io import save_zarr_trajectories
    from flowtracks.trajectory import Trajectory

    M, off_mm, n_pairs, spread = estimate_rig_b_in_a(a.a, a.b)
    da = to_barrel_m(load_clean(a.a, a.clean))
    db = to_barrel_m(to_rig_a(load_clean(a.b, a.clean), M, off_mm))
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

    # every A point against the B points of the same frame: count, per
    # (A track, B track), the points closer than --dup-mm
    shared = {}
    for tid, t in ta.items():
        for f, p in zip(t["time"], t["pos"]):
            if int(f) not in btree:
                continue
            tree, v = btree[int(f)]
            dist, j = tree.query(p, k=1)
            if dist < dup_m:
                key = (tid, v[j][0])
                shared[key] = shared.get(key, 0) + 1
    # best partner of each track = the one it shares most points with
    partner_a, partner_b = {}, {}
    for (atid, btid), n in shared.items():
        if n > partner_a.get(atid, (None, 0))[1]:
            partner_a[atid] = (btid, n)
        if n > partner_b.get(btid, (None, 0))[1]:
            partner_b[btid] = (atid, n)
    # the de-duplication rule: a B track sharing any point with a longer (or
    # equal) A track is a duplicate; A always survives
    drop_b = {btid for (atid, btid) in shared
              if len(ta[atid]["time"]) >= len(tb[btid]["time"])}
    lendiff = [len(ta[atid]["time"]) - len(tb[btid]["time"]) for (atid, btid) in shared]
    kept_b = {tid: t for tid, t in tb.items()
              if a.keep_duplicates or tid not in drop_b}
    print(f"duplicates: {len(drop_b)} B-tracks share points with a longer A track "
          f"(median len diff A-B: {np.median(lendiff) if lendiff else 0:.0f}); "
          f"{'KEPT and marked' if a.keep_duplicates else 'dropped'}", flush=True)

    order = [("14", tid) for tid in ta] + [("58", tid) for tid in kept_b]
    out_id = {key: 1000 + k for k, key in enumerate(order)}
    trajs, info = [], {k: [] for k in ("trajid", "rig", "partner", "shared_points",
                                       "duplicate")}
    for (rig, tid), oid in out_id.items():
        t = ta[tid] if rig == "14" else kept_b[tid]
        trajs.append(Trajectory(t["pos"], t["vel"], t["time"], oid, accel=t["accel"]))
        if rig == "14":
            other, n = partner_a.get(tid, (None, 0))
            pid = out_id.get(("58", other), -1) if other is not None else -1
            dup = False
        else:
            other, n = partner_b.get(tid, (None, 0))
            pid = out_id.get(("14", other), -1) if other is not None else -1
            dup = tid in drop_b
        info["trajid"].append(oid)
        info["rig"].append(0 if rig == "14" else 1)
        info["partner"].append(pid)
        info["shared_points"].append(n)
        info["duplicate"].append(dup)
    dst = str(RAW / a.out)
    save_zarr_trajectories(trajs, dst, overwrite=True)
    import zarr

    root = zarr.open_group(dst, mode="a")
    ti = root.require_group("track_info")
    for k, v in info.items():
        ti.create_array(k, data=np.asarray(v, dtype=bool if k == "duplicate" else np.int64),
                        overwrite=True)
    root.attrs.update(
        n_rig14=len(ta),
        duplicates_kept=bool(a.keep_duplicates),
        dup_mm=a.dup_mm,
        track_info=("per trajectory: rig 0 = 1-4 / 1 = 5-8; partner = trajid in the "
                    "other rig sharing most same-frame points closer than dup_mm "
                    "(-1: none); duplicate = rig 5-8 track the de-duplication rule "
                    "drops (shares points with a longer rig 1-4 track)"),
        rig58_to_rig14=dict(matrix=M.tolist(), offset_mm=off_mm.tolist(),
                            n_pairs=int(n_pairs), spread_mm=spread.tolist()),
        barrel_from_rig14_mm=[0.0, Y_SHIFT_M * 1000.0, 0.0],
        units="m, barrel frame (origin on the axis at mid-height, +Y up)",
    )
    lens = np.array([len(t) for t in trajs])
    print(f"merged: n={len(trajs)} med={np.median(lens):.0f} max={lens.max()} "
          f"rows={lens.sum()} -> {dst} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

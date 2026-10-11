"""Merge the two Illmenau rigs' cleaned trajectories into ONE barrel-frame dataset.

The generic steps are flowtracks.merge (postptv); this script supplies the
Illmenau specifics: the rig folders, the two candidate rotations, the barrel
frame and walls, and the output layout.

  1. MEASURES how rig 5-8's world sits in rig 1-4's from the bubbles both
     rigs see with all four cameras in the same frame
     (flowtracks.merge.estimate_rigid_map).  One calibration image defines
     both worlds and cams 5-8 fire 3 ms after cams 1-4, so a bubble moves
     < 3 mm between them.  A 0 and a 180 deg turn about the vertical are
     tried -- which applies depends on how the back face of the plate was
     labelled when rig 5-8 was calibrated, so it is measured, never assumed.
     No sharp cluster -> refuse to merge,
  2. maps rig 5-8 into rig 1-4 (positions; velocity/acceleration rotated),
     then both into the barrel frame (barrel = rig 1-4 + [0,-1175,0] mm: the
     datum dot is on the axis, 615 mm above the heating plate),
  3. drops points outside the barrel cylinder (r>3575 or |Y|>1790 mm),
  4. de-duplicates the overlap (flowtracks.merge.overlap_counts, duplicates):
     same frame + closer than --dup-mm keeps ONE copy -- the longer
     trajectory wins (ties: rig 1-4).  --keep-duplicates keeps every track and
     only marks them (track_info/duplicate),
  5. writes the store with the same trajectories schema (time/pos/trajid/
     vel/accel), plus track_info/{trajid, rig, partner, shared_points,
     duplicate}.  Rig 1-4 trajectories come first (trajid < 1000 + n_rig14);
     the measured rig transform is stored in the attributes.

Usage (flowtracks >= 1.7.0, postptv):
    uv run python \
        scripts/illmenau/merge_dual_rig.py [--dup-mm 20] [--keep-duplicates]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import zarr
from flowtracks.io import save_zarr_trajectories
from flowtracks.merge import (
    apply_rigid,
    best_partners,
    duplicates,
    estimate_rigid_map,
    overlap_counts,
)
from flowtracks.trajectory import Trajectory

from openptv2.storage import RunStore

RAW = Path("/Users/alex/Downloads/Ilmenau")
# SAME convention as the cleaned stores: metres, barrel frame.
# (linkage stores are mm; cleaned trajectories are m.)
Y_SHIFT_M = -1.175
R_WALL_M, H_BARREL_M = 3.575, 3.58
ROTATIONS = [("yaw 0 deg", np.diag([1.0, 1.0, 1.0])),
             ("yaw 180 deg", np.diag([-1.0, 1.0, -1.0]))]


def load_clean(rig, name):
    t = zarr.open(str(RAW / rig / "res" / name), mode="r")["trajectories"]
    return {k: np.asarray(t[k]) for k in ("time", "pos", "trajid", "vel", "accel")}


def four_camera_pairs(dir_a, dir_b, store="run_fulldiam.zarr", step=10):
    """Same-frame (rig A, rig B) four-camera points [mm], every step-th frame."""
    sa = RunStore(str(RAW / dir_a / "res" / store), mode="r")
    sb = RunStore(str(RAW / dir_b / "res" / store), mode="r")
    frames = sorted(set(sa.frames(source="correspondences"))
                    & set(sb.frames(source="correspondences")))[::step]

    def four_cam(st, f):
        p, i = st.read_correspondences(f)
        return p[(i >= 0).sum(1) == 4]

    return [(four_cam(sa, f), four_cam(sb, f)) for f in frames]


def split(d, tag, rig):
    """Per-trajectory rows, ids tagged per rig."""
    tids = np.asarray(d["trajid"])
    out = {}
    for tid in np.unique(tids):
        m = tids == tid
        o = np.argsort(np.asarray(d["time"])[m], kind="stable")
        idx = np.where(m)[0][o]
        out[tag + int(tid)] = dict(
            idx=idx, time=np.asarray(d["time"])[idx], pos=np.asarray(d["pos"])[idx],
            vel=np.asarray(d["vel"])[idx], accel=np.asarray(d["accel"])[idx], rig=rig)
    return out


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

    try:
        rig = estimate_rigid_map(four_camera_pairs(a.a, a.b), ROTATIONS,
                                 log=lambda s: print(s, flush=True))
    except ValueError as exc:
        raise SystemExit(f"{exc}: refusing to merge") from None
    M, off_mm = rig["matrix"], rig["offset"]

    da, db = load_clean(a.a, a.clean), load_clean(a.b, a.clean)
    db["pos"] = apply_rigid(db["pos"], M, off_mm / 1000.0)
    db["vel"], db["accel"] = apply_rigid(db["vel"], M), apply_rigid(db["accel"], M)
    for d in (da, db):  # rig 1-4 frame -> barrel frame
        assert np.abs(d["pos"]).max() < 50, "expected metres"
        d["pos"] = d["pos"].astype(float, copy=True)
        d["pos"][:, 1] += Y_SHIFT_M
    print(f"loaded: 1-4 rows={len(da['time'])} 5-8 rows={len(db['time'])}", flush=True)

    for d in (da, db):
        p = d["pos"]
        m = (np.hypot(p[:, 0], p[:, 2]) <= R_WALL_M) & (np.abs(p[:, 1]) <= H_BARREL_M / 2)
        print(f"inside walls: {m.sum()}/{len(m)}", flush=True)
        for k in d:
            d[k] = d[k][m]

    ta = split(da, a.tag_a, "14")
    tb = split(db, a.tag_b, "58")
    print(f"trajectories: 1-4={len(ta)} 5-8={len(tb)}", flush=True)

    shared = overlap_counts(ta, tb, a.dup_mm / 1000.0)
    partner_a, partner_b = best_partners(shared)
    drop_b = duplicates(shared, {k: len(t["time"]) for k, t in ta.items()},
                        {k: len(t["time"]) for k, t in tb.items()})
    lendiff = [len(ta[atid]["time"]) - len(tb[btid]["time"]) for (atid, btid) in shared]
    kept_b = {tid: t for tid, t in tb.items() if a.keep_duplicates or tid not in drop_b}
    print(f"duplicates: {len(drop_b)} B-tracks share points with a longer A track "
          f"(median len diff A-B: {np.median(lendiff) if lendiff else 0:.0f}); "
          f"{'KEPT and marked' if a.keep_duplicates else 'dropped'}", flush=True)

    order = [("14", tid) for tid in ta] + [("58", tid) for tid in kept_b]
    out_id = {key: 1000 + k for k, key in enumerate(order)}
    trajs, info = [], {k: [] for k in ("trajid", "rig", "partner", "shared_points",
                                       "duplicate")}
    for (rig_name, tid), oid in out_id.items():
        t = ta[tid] if rig_name == "14" else kept_b[tid]
        trajs.append(Trajectory(t["pos"], t["vel"], t["time"], oid, accel=t["accel"]))
        if rig_name == "14":
            other, n = partner_a.get(tid, (None, 0))
            pid = out_id.get(("58", other), -1) if other is not None else -1
            dup = False
        else:
            other, n = partner_b.get(tid, (None, 0))
            pid = out_id.get(("14", other), -1) if other is not None else -1
            dup = tid in drop_b
        info["trajid"].append(oid)
        info["rig"].append(0 if rig_name == "14" else 1)
        info["partner"].append(pid)
        info["shared_points"].append(n)
        info["duplicate"].append(dup)
    dst = str(RAW / a.out)
    save_zarr_trajectories(trajs, dst, overwrite=True)
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
                            n_pairs=int(rig["n_pairs"]), spread_mm=rig["spread"].tolist()),
        barrel_from_rig14_mm=[0.0, Y_SHIFT_M * 1000.0, 0.0],
        units="m, barrel frame (origin on the axis at mid-height, +Y up)",
    )
    lens = np.array([len(t) for t in trajs])
    print(f"merged: n={len(trajs)} med={np.median(lens):.0f} max={lens.max()} "
          f"rows={lens.sum()} -> {dst} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

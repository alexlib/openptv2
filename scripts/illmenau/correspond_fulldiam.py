"""Full-diameter correspondence for one Illmenau rig, one shard of frames.

Reuses stored detections (no images needed), triangulates with the CURRENT
cal/camN.tif.ori/.addpar and the widened criteria box from parameters_Run1.yaml,
and writes per-frame (pos, cam_target_ids) into a partial npz for later merge.

Usage:
    uv run python scripts/illmenau/correspond_fulldiam.py --rig 14 --shard 0 \
        --nshards 6 --first 10001 --last 10500
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _config as CFG  # noqa: E402


def build_volume(crit):
    from openptv2.algorithms.parameters import VolumePar

    return VolumePar(
        X_lay=np.array(crit["X_lay"], float),
        Zmin_lay=np.array(crit["Zmin_lay"], float),
        Zmax_lay=np.array(crit["Zmax_lay"], float),
        cn=float(crit.get("cn", 0)),
        cnx=float(crit.get("cnx", 0)),
        cny=float(crit.get("cny", 0)),
        csumg=float(crit.get("csumg", 0)),
        corrmin=float(crit.get("corrmin", 0)),
        eps0=float(crit.get("eps0", 0.05)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", choices=["14", "58"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--first", type=int, default=10001)
    ap.add_argument("--last", type=int, default=10500)
    ap.add_argument("--min-cams", type=int, default=0,
                    help="keep only 3D points seen by at least this many cameras "
                         "(default: keep all the matcher returns)")
    a = ap.parse_args()
    want = [1, 2, 3, 4] if a.rig == "14" else [5, 6, 7, 8]
    if CFG.CAMS != want:
        raise SystemExit(
            f"camera group mismatch: ILLMENAU_CAMS={CFG.CAMS}, --rig {a.rig} "
            f"needs {want}. Set ILLMENAU_DIR and ILLMENAU_CAMS together.")
    print(f"[{a.rig}/{a.shard}] {CFG.banner()}", flush=True)

    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.correspondences import match_frame_correspondences
    from openptv2.storage import RunStore

    cpar = ControlPar(
        num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
        img_base_name=[""] * 4, cal_img_base_name=[""] * 4)
    cals = CFG.load_calibrations()
    y = yaml.safe_load((CFG.DIR / "parameters_Run1.yaml").read_text())
    vp = build_volume(y["criteria"])
    print(f"[{a.rig}/{a.shard}] box X {y['criteria']['X_lay']} "
          f"Z {y['criteria']['Zmin_lay']}..{y['criteria']['Zmax_lay']}",
          flush=True)

    store = RunStore(str(CFG.DIR / "res" / "run.zarr"), mode="r")
    frames = [f for f in sorted(store.frames())
              if a.first <= f <= a.last
              and all(store.has_targets(c, f) for c in range(4))]
    mine = [f for i, f in enumerate(frames) if i % a.nshards == a.shard]
    print(f"[{a.rig}/{a.shard}] {len(mine)}/{len(frames)} frames", flush=True)

    out = {}
    t0 = time.time()
    for k, f in enumerate(mine):
        try:
            det = [store.read_targets(c, f) for c in range(4)]
            pos, ids = match_frame_correspondences(det, cpar, cals, vp)
            pos, ids = np.asarray(pos, float), np.asarray(ids, np.int32)
            if a.min_cams and len(ids):
                keep = (ids >= 0).sum(1) >= a.min_cams
                pos, ids = pos[keep], ids[keep]
            out[f"{f}/pos"] = pos
            out[f"{f}/ids"] = ids
        except Exception as exc:
            print(f"[{a.rig}/{a.shard}] frame {f} FAILED: {exc!r}", flush=True)
            out[f"{f}/pos"] = np.empty((0, 3))
            out[f"{f}/ids"] = np.empty((0, 4), np.int32)
        if (k + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"[{a.rig}/{a.shard}] {k + 1}/{len(mine)} "
                  f"({dt / (k + 1):.1f}s/frame)", flush=True)
    dst = CFG.DIR / "cal" / f"fulldiam_part_{a.rig}_s{a.shard}.npz"
    np.savez_compressed(dst, **out)
    print(f"[{a.rig}/{a.shard}] wrote {dst} "
          f"({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()

"""Full-diameter correspondence for one Illmenau rig, one shard of frames.

Re-matches the detections stored in res/run.zarr with the CURRENT calibration
and the parameter file's settings -- the generic step is
openptv2.recorrespond -- and writes this shard's per-frame (pos,
cam_target_ids) into cal/fulldiam_part_<rig>_s<shard>.npz for
assemble_track.py.  The Illmenau part is only the sharding (one process per
shard) and the rig/folder bookkeeping.

Usage (ILLMENAU_DIR / ILLMENAU_CAMS select the rig, see _config.py):
    uv run python scripts/illmenau/correspond_fulldiam.py --rig 14 --shard 0 \
        --nshards 6 --first 10001 --last 10500
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _config as CFG  # noqa: E402


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

    from openptv2.recorrespond import correspond_frame, load_parameters, stored_frames
    from openptv2.storage import RunStore

    cpar, vpar, cals, params = load_parameters(CFG.DIR / "parameters_Run1.yaml")
    crit = params["criteria"]
    print(f"[{a.rig}/{a.shard}] box X {crit['X_lay']} "
          f"Z {crit['Zmin_lay']}..{crit['Zmax_lay']} eps0 {crit['eps0']}", flush=True)

    store = RunStore(str(CFG.DIR / "res" / "run.zarr"), mode="r")
    frames = stored_frames(store, len(cals), a.first, a.last)
    mine = [f for i, f in enumerate(frames) if i % a.nshards == a.shard]
    print(f"[{a.rig}/{a.shard}] {len(mine)}/{len(frames)} frames", flush=True)

    out = {}
    t0 = time.time()
    for k, f in enumerate(mine):
        try:
            pos, ids = correspond_frame(store, f, cpar, cals, vpar, a.min_cams)
        except Exception as exc:
            print(f"[{a.rig}/{a.shard}] frame {f} FAILED: {exc!r}", flush=True)
            pos, ids = np.empty((0, 3)), np.empty((0, len(cals)), np.int32)
        out[f"{f}/pos"] = pos
        out[f"{f}/ids"] = ids
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

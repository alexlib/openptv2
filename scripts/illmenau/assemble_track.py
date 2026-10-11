"""Assemble full-diameter correspondences and track one Illmenau rig.

Merges the shards of correspond_fulldiam.py into a new store
res/run_fulldiam.zarr (with the detections of res/run.zarr), then tracks it
with the production two_phase tracker and the parameter file's `track:`
section -- the generic steps are openptv2.recorrespond.write_store and
openptv2.plugins.two_phase_tracking.track_store.  The Illmenau part is only
the shard files and the rig folder.

Usage (ONE rig at a time; env selects it):
    ILLMENAU_DIR=.../openptv_illmenau_4cam ILLMENAU_CAMS=1,2,3,4 \
    uv run python scripts/illmenau/assemble_track.py [--first F --last L]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _config as CFG  # noqa: E402


def read_shards(cal_dir):
    """{frame: (pos, cam_target_ids)} from every fulldiam_part_*.npz shard."""
    parts = sorted(Path(cal_dir).glob("fulldiam_part_*.npz"))
    if not parts:
        raise SystemExit("no fulldiam_part_*.npz shards: run correspond_fulldiam first")
    merged = {}
    for p in parts:
        with np.load(str(p), allow_pickle=True) as z:
            for k in z.files:
                if k.endswith("/pos"):
                    f = int(k.split("/")[0])
                    merged[f] = (np.asarray(z[k], float),
                                 np.asarray(z[f"{f}/ids"], np.int32))
    return merged, len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--out", default="run_fulldiam.zarr",
                    help="store name inside res/ (or an absolute path)")
    a = ap.parse_args()
    t0 = time.time()

    from openptv2.plugins.two_phase_tracking import track_store
    from openptv2.recorrespond import load_parameters, write_store
    from openptv2.storage import RunStore

    cpar, _vpar, cals, params = load_parameters(CFG.DIR / "parameters_Run1.yaml")
    track_cfg = dict(params.get("track", {}))
    print(f"cams {CFG.CAMS}: track {track_cfg}", flush=True)

    merged, nparts = read_shards(CFG.DIR / "cal")
    frames = sorted(merged)
    if a.first is not None:
        frames = [f for f in frames if a.first <= f <= (a.last or a.first)]
    merged = {f: merged[f] for f in frames}
    print(f"merged {nparts} shards, {len(frames)} frames {frames[0]}..{frames[-1]}",
          flush=True)

    out = CFG.DIR / "res" / a.out
    new = RunStore(str(out), mode="w")
    write_store(RunStore(str(CFG.DIR / "res" / "run.zarr"), mode="r"), new, merged)
    links = track_store(new, frames, track_cfg, cals, cpar)
    print(f"tracked: {len(links)} links; wrote {out} ({time.time() - t0:.0f}s total)",
          flush=True)


if __name__ == "__main__":
    main()

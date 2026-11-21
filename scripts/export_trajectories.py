"""Write the tracks of a finished run into ``res/run.zarr/trajectories`` so openptv-cloud's ``03_results`` notebook
(3D tracks coloured by speed) and flowtracks can read them.  The holistic loop does this itself; use this script for
runs made by another tracker or edited afterwards.

    uv run python scripts/export_trajectories.py <run folder or run.zarr> [more ...] [--linkage ptv_is] [--min-len 2]

Positions go in metres, velocity and acceleration per frame, sorted by (trajid, time).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from openptv2.holistic import write_trajectories_cache
    from openptv2.storage import RunStore

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--linkage", default="ptv_is")
    ap.add_argument("--min-len", type=int, default=2)
    a = ap.parse_args()
    for p in a.paths:
        path = Path(p)
        if path.name != "run.zarr":
            path = path / "res" / "run.zarr"
        store = RunStore(path, mode="a")
        frames = sorted(store.frames())
        n_cams = len([k for k in store.root["targets"].keys() if k.startswith("cam_")])
        n = write_trajectories_cache(store, frames, n_cams, a.linkage, a.min_len)
        print(f"{path}: {n} tracks written to trajectories/")


if __name__ == "__main__":
    main()

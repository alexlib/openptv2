"""Profile trackcorr forward: kernel vs Python orchestration vs IO split.

Run from repo root:
    uv run python -u scripts/prof_trackcorr.py
"""

import cProfile
import pstats
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_same_trajectories import DS, REF  # noqa: E402

WORK = (Path(__file__).resolve().parent.parent / "scratch" / "_wp1_ab").resolve()


def main():
    import os

    for f in range(100001, 100011):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
        for cam in range(1, 5):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                WORK / "img_3dptv" / f"Cam{cam}.{f}_targets")
    os.chdir(WORK)
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    pm = ParameterManager()
    pm.from_yaml(WORK / "parameters_Run_dacc19.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                 loser_retry=0, cold_start_neighbour=0)
    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    tr.full_forward()
    pr.disable()
    wall = time.perf_counter() - t0
    print(f"wall forward: {wall:.1f}s", flush=True)
    st = pstats.Stats(pr)
    print(f"nfuncs={len(st.stats)}", flush=True)
    rows = []
    for (fn, _ln, name), (_cc, nc, tt, ct, _callers) in st.stats.items():
        low = (fn + "#" + name).lower()
        if "openptv2" in low and (
                "track" in low or "frame" in low or "sync" in low
                or "path" in low or "read" in low or "write" in low):
            rows.append((ct, tt, nc, name))
    print(f"matched={len(rows)}", flush=True)
    for ct, tt, nc, name in sorted(rows, reverse=True)[:16]:
        print(f"cum={ct:.3f} self={tt:.3f} nc={nc} {name}", flush=True)


if __name__ == "__main__":
    main()

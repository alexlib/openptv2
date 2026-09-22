"""wp1 validation for the adaptive scheduler (parity base, app=0).

Restores pristine inputs, runs adaptive forward + stock backward, scores
vs the 3dptv reference. Must not regress vs fixed-gate 98.16%.

Run from repo root (detached):
    uv run python -u scripts/verify_adaptive_wp1.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_same_trajectories import (  # noqa: E402
    DS,
    REF,
    compare_links,
    compare_tracks,
)

WORK = (Path(__file__).resolve().parent.parent / "scratch" / "_wp1_ab"
        ).resolve()
FIRST, LAST = 100001, 100010


def main():
    from phase_scheduler import run_adaptive

    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
        for cam in range(1, 5):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                WORK / "img_3dptv" / f"Cam{cam}.{f}_targets")
    print("inputs restored; running adaptive forward", flush=True)
    run_adaptive(WORK, 0, 0, first=FIRST, last=LAST, verbose=True)
    compare_links(WORK, 100003, 100010, "adaptive-fwd")
    compare_tracks(WORK)


if __name__ == "__main__":
    main()

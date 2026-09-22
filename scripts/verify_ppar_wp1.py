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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=int, default=0)
    ap.add_argument("--hi", type=float, default=1.5)
    ap.add_argument("--app", type=float, default=0.0)
    ap.add_argument("--mode", type=str, default="vel")
    ap.add_argument("--yaml", type=str, default="parameters_Run_dacc19.yaml")
    args = ap.parse_args()
    from phase_scheduler import run_adaptive_pp

    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
        for cam in range(1, 5):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                WORK / "img_3dptv" / f"Cam{cam}.{f}_targets")
    print(f"inputs restored; running per-particle adaptive forward "
          f"(lr={args.lr} hi={args.hi} app={args.app} mode={args.mode} "
          f"yaml={args.yaml})", flush=True)
    run_adaptive_pp(WORK, args.lr, 0, hi=args.hi, mode=args.mode,
                    app=args.app, first=FIRST, last=LAST, verbose=True,
                    yaml=args.yaml)
    compare_links(WORK, 100003, 100010, "adaptive-fwd")
    compare_tracks(WORK)


if __name__ == "__main__":
    main()

"""Run one forward tracking step and report links vs a reference run.

Used to compare the COMPILED kernels against the pure-Python fallback on the
same input: run it once normally, once with the .pyd/.so files moved aside.

    uv run python scripts/_run_one_step.py --work <dir> --ref <ref res dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def read_links(res: Path, frame: int):
    p = res / f"ptv_is.{frame}"
    lines = p.read_text().strip().splitlines()
    n = int(lines[0])
    return [int(line.split()[0]) for line in lines[1 : n + 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=100002)
    args = ap.parse_args()

    work = args.work.resolve()
    ref = args.ref.resolve()

    import openptv2.algorithms.track as trk

    compiled = trk.is_compiled()
    print(f"compiled kernels: {compiled}")

    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / "parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for i, s in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(i, str(Path(s).resolve()) + ".")
        Tracker(cpar, vpar, track_par, spar, cals, default_naming).full_forward()
    finally:
        os.chdir(old)

    r = read_links(ref, args.frame)
    t = read_links(work / "res", args.frame)
    refn = sum(1 for p in r if p >= 0)
    testn = sum(1 for p in t if p >= 0)
    match = sum(1 for i in range(min(len(r), len(t))) if r[i] >= 0 and t[i] == r[i])
    print(f"\nframe {args.frame}")
    print(f"  reference (3dptv) links : {refn}")
    print(f"  this run links          : {testn}")
    print(f"  reproduced exactly      : {match}  ({match / refn:.1%} of reference)")


if __name__ == "__main__":
    main()

"""Decisive T0.1 follow-up: run openptv2 with dacc=1.9 (the 3dptv value).

The work folder YAML sets dacc=0.8 while track.par / 3dptv use dacc=1.9.
17.9% of reference links have 3-point acc >= 0.8 vs a 16.3% exact-link gap.
If this run reproduces ~100% of reference links, the gap is config.
"""

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DATASET / "res_ground_truth_backup"
FIRST, LAST = 100001, 100010
SCORE_FIRST, SCORE_LAST = 100003, 100010


def main():
    work = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
        REPO / "scratch" / "_wp1_ab").resolve()
    dacc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.9

    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", work / "res" / f"rt_is.{f}")
        for cam in range(1, 5):
            shutil.copyfile(
                DATASET / "img_3dptv" / f"Cam{cam}.{f}_targets",
                work / "img_3dptv" / f"Cam{cam}.{f}_targets",
            )
    print(f"inputs restored; running dacc={dacc}", flush=True)

    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / "parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        print(f"effective dacc before override: {track_par.dacc}", flush=True)
        track_par.dacc = dacc
        Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                loser_retry=0, cold_start_neighbour=0).full_forward()
    finally:
        os.chdir(old)

    def read_links(res: Path, frame: int):
        lines = (res / f"ptv_is.{frame}").read_text().strip().splitlines()
        n = int(lines[0])
        return [int(line.split()[0]) for line in lines[1: n + 1]]

    ref_total = test_total = match = 0
    for f in range(SCORE_FIRST, SCORE_LAST + 1):
        r = read_links(REF, f)
        t = read_links(work / "res", f)
        ref_total += sum(1 for p in r if p >= 0)
        test_total += sum(1 for p in t if p >= 0)
        match += sum(1 for i in range(min(len(r), len(t)))
                     if r[i] >= 0 and t[i] == r[i])
    print(f"dacc={dacc}: ref={ref_total} test={test_total} "
          f"exact={match} recall={match / ref_total:.2%}", flush=True)


if __name__ == "__main__":
    main()

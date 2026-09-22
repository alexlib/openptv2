"""T2.1 — parameter sensitivity sweep: which gate binds the missing links?

Runs full forward tracking on the wp1 working folder from PRISTINE inputs
(restored before every run: rt_is.* from the 3dptv reference backup,
Cam*_targets from the dataset) while widening one track.par parameter at a
time. Scores exact-link reproduction vs the 3dptv reference on clean frames.

Methodology note: a tracking run REWRITES res/rt_is.* (added particles) and
updates target-file tnr columns in place, so re-running in the same folder
without restoring inputs silently changes the inputs. Every run here starts
from byte-identical inputs (verified by hash before the run).

Run from repo root:
    uv run python scripts/sweep_track_params.py

Baseline reproduces the plan's 59.0% openptv2 link rate; each row shows which
parameter closes the gap toward 3dptv's fixed 69.5%.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DATASET / "res_ground_truth_backup"
FIRST, LAST = 100001, 100010
SCORE_FIRST, SCORE_LAST = 100003, 100010  # clean frames (100001 is corrupt)


def restore_inputs(work: Path) -> None:
    """Copy pristine rt_is + target files into the work folder."""
    res = work / "res"
    img = work / "img_3dptv"
    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", res / f"rt_is.{f}")
        for cam in range(1, 5):
            shutil.copyfile(
                DATASET / "img_3dptv" / f"Cam{cam}.{f}_targets",
                img / f"Cam{cam}.{f}_targets",
            )


def hash_inputs(work: Path) -> str:
    h = hashlib.sha256()
    for f in range(FIRST, LAST + 1):
        h.update((work / "res" / f"rt_is.{f}").read_bytes())
        for cam in range(1, 5):
            h.update(
                (work / "img_3dptv" / f"Cam{cam}.{f}_targets").read_bytes()
            )
    return h.hexdigest()[:16]


def run_once(work: Path, overrides: dict, loser_retry: int,
             cold_start_neighbour: int) -> None:
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
        for key, val in overrides.items():
            setattr(track_par, key, val)
        t = Tracker(
            cpar, vpar, track_par, spar, cals, default_naming,
            loser_retry=loser_retry,
            cold_start_neighbour=cold_start_neighbour,
        )
        t.full_forward()
    finally:
        os.chdir(old)


def read_links(res: Path, frame: int):
    p = res / f"ptv_is.{frame}"
    lines = p.read_text().strip().splitlines()
    n = int(lines[0])
    return [int(line.split()[0]) for line in lines[1: n + 1]]


def score(work: Path):
    ref_total = test_total = match = 0
    for f in range(SCORE_FIRST, SCORE_LAST + 1):
        r = read_links(REF, f)
        t = read_links(work / "res", f)
        ref_total += sum(1 for p in r if p >= 0)
        test_total += sum(1 for p in t if p >= 0)
        for i in range(min(len(r), len(t))):
            if r[i] >= 0 and t[i] == r[i]:
                match += 1
    return ref_total, test_total, match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path,
                    default=REPO / "scratch" / "_wp1_ab")
    args = ap.parse_args()
    work = args.work.resolve()

    print(f"reference dir: {REF}")
    print(f"scoring frames {SCORE_FIRST}..{SCORE_LAST} (clean frames)")

    grid = [
        ("baseline (track.c parity)", {}, 0, 0),
        ("baseline (openptv2 defaults)", {}, 1, 1),
        ("dv xmax 1.9 -> 2.5",
         {"dvxmin": -2.5, "dvxmax": 2.5, "dvymin": -2.5, "dvymax": 2.5,
          "dvzmin": -2.5, "dvzmax": 2.5}, 0, 0),
        ("dv xmax 1.9 -> 4.0",
         {"dvxmin": -4.0, "dvxmax": 4.0, "dvymin": -4.0, "dvymax": 4.0,
          "dvzmin": -4.0, "dvzmax": 4.0}, 0, 0),
        ("dacc 1.9 -> 3.0", {"dacc": 3.0}, 0, 0),
        ("dacc 1.9 -> 6.0", {"dacc": 6.0}, 0, 0),
        ("dangle 270 -> 400", {"dangle": 400.0}, 0, 0),
        ("add 1 -> 0", {"add": 0}, 0, 0),
    ]

    print(f"\n{'config':32}{'ref':>7}{'test':>7}{'exact':>7}"
          f"{'recall':>9}{'rate-vs-ref':>12}")
    results = []
    for label, overrides, lr, csn in grid:
        restore_inputs(work)
        h = hash_inputs(work)
        run_once(work, overrides, lr, csn)
        ref_total, test_total, match = score(work)
        recall = match / ref_total if ref_total else 0.0
        results.append((label, ref_total, test_total, match, recall, h))
        print(f"{label:32}{ref_total:>7}{test_total:>7}{match:>7}"
              f"{recall:>8.1%}  ({match}/{ref_total}, inputs {h})")

    print("\nDecision: whichever parameter moves recall toward ~100% of the "
          "reference link count names the binding constraint. If none moves "
          "it, the loss is structural (see T1.1 census).")


if __name__ == "__main__":
    main()

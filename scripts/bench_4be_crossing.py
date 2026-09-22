"""Can 4BE (four-frame best estimate) solve the hard crossings?

4BE scores a candidate by whether it predicts a REAL particle two frames
ahead -- longer temporal context than trackcorr's one-frame lookahead plus
averaged gate. Same S-scenes, same seeds, same scoring as synth_crossing.

Engines: trackcorr-parity (lr0), trackcorr-default (lr1), 4be (paper
defaults: STRICT_SUPPORT=0, GREEDY_CONFLICTS=0).

Run from repo root:
    uv run python -u scripts/bench_4be_crossing.py [--reps N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from synth_crossing import (  # noqa: E402
    REPO,
    load_optics,
    make_truth,
    score,
    setup_work,
    write_scene,
)


def run_4be(work: Path):
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
        Tracker(cpar, vpar, track_par, spar, cals,
                default_naming).full_forward_4be()
    finally:
        os.chdir(old)


def run_corr(work: Path, lr: int, cs: int):
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
        Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                loser_retry=lr, cold_start_neighbour=cs).full_forward()
    finally:
        os.chdir(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    cpar, cals = load_optics()
    scenarios = [
        ("S1 head-on", {"v": 0.5}),
        ("S8 kick", {"v": 0.5, "maneuver": "kick"}),
        ("S11 kick+noise", {"v": 0.5, "maneuver": "kick", "noise": 0.07}),
        ("S12 convoy+noise", {"v": 0.5, "convoy": 0.6, "noise": 0.07}),
        ("S4 head-on+noise", {"v": 0.5, "noise": 0.07}),
    ]
    engines = [
        ("corr-parity", lambda w: run_corr(w, 0, 0)),
        ("corr-default", lambda w: run_corr(w, 1, 1)),
        ("4be", run_4be),
    ]
    print(f"{'scenario':18}{'engine':14}{'recall':>9}{'cross-ok':>10}",
          flush=True)
    for sname, kw in scenarios:
        noisy = kw.get("noise", 0.0) > 0
        reps = args.reps if noisy else 3
        for ename, run in engines:
            recs, crs = [], []
            for rep in range(reps):
                work = REPO / "scratch" / "_synth_4be"
                setup_work(work)
                truth = make_truth(seed=1000 + rep, **kw)
                rows_pf = write_scene(work, truth, cpar, cals)
                run(work)
                r, c = score(work, rows_pf)
                recs.append(r)
                crs.append(c)
            print(f"{sname:18}{ename:14}{np.mean(recs):>8.1%}  "
                  f"{sum(crs)}/{len(crs)}", flush=True)


if __name__ == "__main__":
    main()

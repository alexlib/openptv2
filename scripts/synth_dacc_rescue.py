"""Does widening dacc rescue the noisy-maneuver crossing (S11)?

Distinguishes gate-binding (fixable by parameter) from contest loss
(structural). Reuses synth_crossing builders.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth_crossing as SC  # noqa: E402
from synth_crossing import (  # noqa: E402
    REPO,
    load_optics,
    make_truth,
    score,
    setup_work,
    write_scene,
)
from synth_crossing import (
    run_tracker as _run,
)


def run_tracker_dacc(work: Path, lr: int, cs: int, dacc: float):
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
        track_par.dacc = dacc
        Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                loser_retry=lr, cold_start_neighbour=cs).full_forward()
    finally:
        os.chdir(old)


def main():
    cpar, cals = load_optics()
    print(f"{'dacc':>6}{'lr':>4}{'recall':>9}{'cross-ok':>10}", flush=True)
    for dacc in (1.9, 2.5, 3.5):
        for lr, cs in ((0, 0), (1, 1)):
            recs, crs = [], []
            for rep in range(8):
                work = REPO / "scratch" / "_synth_d"
                setup_work(work)
                truth = make_truth(v=0.5, maneuver="kick", noise=0.07,
                                   seed=1000 + rep)
                rows_pf = write_scene(work, truth, cpar, cals)
                run_tracker_dacc(work, lr, cs, dacc)
                r, c = score(work, rows_pf)
                recs.append(r)
                crs.append(c)
            import numpy as np
            print(f"{dacc:>6}{lr:>4}{np.mean(recs):>8.1%}  "
                  f"{sum(crs)}/{len(crs)}", flush=True)


if __name__ == "__main__":
    main()

"""Prototype: global (Hungarian) conflict resolution for trackcorr.

Replaces the kernel's greedy Phase-2 (particle order, evict-on-cheaper)
with a minimum-total-cost assignment over each step's decis matrix --
without touching the kernel: after each stock trackcorr_c_loop call, read
back inlist/decis/linkdecis, resolve globally, patch bufs + files, so the
cascade consumes the patched history. End-to-end honest test of the idea.

Judged on S11 (kick+noise): if cross-ok beats parity's 2/8 with no
regressions elsewhere, the idea earns a kernel implementation (as a flag).

Run from repo root:
    uv run python -u scripts/proto_global_resolve.py [--reps N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from synth_crossing import (  # noqa: E402
    FIRST,
    NF,
    REPO,
    load_optics,
    make_truth,
    score,
    setup_work,
    write_scene,
)


def run_global(work: Path, cs: int):
    """Forward tracking with Hungarian resolution per step."""
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming
    from openptv2.algorithms.track import trackcorr_c_loop, _sync_soa_to_aos
    from openptv2.algorithms.track import trackcorr_c_finish

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / "parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=cs)
        tr.restart()
        run = tr._run
        last = FIRST + NF - 1
        for step in range(FIRST, last):
            # Stock kernel call advances bufs + files with greedy
            # resolution; post-rotation the processed frame s is buf[0]
            # and s+1 is buf[1]. Re-resolve globally, then patch bufs
            # (next of s; prev of s+1, which this step alone writes) and
            # rewrite the step's files, so the cascade consumes patched
            # history. decis state is read, never modified.
            trackcorr_c_loop(run, step)
            done = run.fb.buf[0]
            after = run.fb.buf[1]
            n1 = done.num_parts
            inlist = np.asarray(done.path_inlist[:n1])
            decis = np.asarray(done.path_decis[:n1])
            linkd = np.asarray(done.path_linkdecis[:n1])
            cand_set = sorted({int(c) for h in range(n1)
                               for c in linkd[h][:int(inlist[h])]})
            if cand_set:
                cpos = {c: k for k, c in enumerate(cand_set)}
                sub = np.full((n1, len(cand_set)), np.inf)
                for h in range(n1):
                    for k in range(int(inlist[h])):
                        sub[h, cpos[int(linkd[h][k])]] = float(decis[h][k])
                ri, ci = linear_sum_assignment(sub)
                new_next = np.full(n1, -2, dtype=np.int32)
                for r, c in zip(ri, ci):
                    if np.isfinite(sub[r, c]):
                        new_next[r] = cand_set[c]
            else:
                new_next = np.full(n1, -2, dtype=np.int32)
            np.asarray(done.path_next[:n1])[:] = new_next
            ap = np.asarray(after.path_prev)
            ap[:] = -1
            for h in range(n1):
                if 0 <= new_next[h] < len(ap):
                    ap[new_next[h]] = h
            _sync_soa_to_aos(done)
            _sync_soa_to_aos(after)
            run.fb.write_frame_from_start(step)
        trackcorr_c_finish(run, last)
    finally:
        os.chdir(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    cpar, cals = load_optics()
    scenarios = [
        ("S1 head-on", {"v": 0.5}),
        ("S8 kick", {"v": 0.5, "maneuver": "kick"}),
        ("S11 kick+noise", {"v": 0.5, "maneuver": "kick", "noise": 0.07}),
        ("S4 head-on+noise", {"v": 0.5, "noise": 0.07}),
        ("S12 convoy+noise", {"v": 0.5, "convoy": 0.6, "noise": 0.07}),
    ]
    print(f"{'scenario':16}{'engine':14}{'recall':>9}{'cross-ok':>10}",
          flush=True)
    for sname, kw in scenarios:
        noisy = kw.get("noise", 0.0) > 0
        reps = args.reps if noisy else 3
        for ename, fn in (("parity", None), ("default", None),
                           ("global", run_global)):
            recs, crs = [], []
            for rep in range(reps):
                work = REPO / "scratch" / "_synth_g"
                setup_work(work)
                truth = make_truth(seed=1000 + rep, **kw)
                rows_pf = write_scene(work, truth, cpar, cals)
                if fn is None:
                    from synth_crossing import run_tracker
                    run_tracker(work, 1 if ename == "default" else 0, 0)
                else:
                    fn(work, 0)
                r, c = score(work, rows_pf)
                recs.append(r)
                crs.append(c)
            print(f"{sname:16}{ename:14}{np.mean(recs):>8.1%}  "
                  f"{sum(crs)}/{len(crs)}", flush=True)


if __name__ == "__main__":
    main()

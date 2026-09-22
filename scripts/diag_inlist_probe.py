"""Dump the tracker's real candidate list (inlist/linkdecis) for one step.

The autopsy replays the candidate stage with the REFERENCE implementations in
track.py. This probe instead reads what the COMPILED kernel actually produced,
so any divergence between the two shows up directly.

For every frame-1 particle that the reference run linked forward but the run
under test did not, it prints how many candidates the kernel actually
registered (inlist) and whether the reference's chosen partner is among them.

    inlist == 0            the kernel found NO admissible candidate at all
    partner in linkdecis   the kernel saw it, so the loss is in resolution
    partner NOT in list    the kernel never registered it -> search/gate bug
                           (and the reference replay disagrees with the kernel)

Run from repo root:
    uv run python scripts/diag_inlist_probe.py --work scratch/_wp1_ab \
        --ref <folder>/res_ground_truth_backup --step 100001
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def read_ptv_is(path: Path):
    lines = path.read_text().strip().splitlines()
    n = int(lines[0])
    return np.array([int(line.split()[0]) for line in lines[1 : n + 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--step", type=int, default=100001)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--dacc", type=float, default=None,
                    help="override track_par.dacc (e.g. 1.9 to match 3dptv)")
    ap.add_argument("--dv", type=float, default=None,
                    help="override all dv* bounds symmetrically")
    args = ap.parse_args()

    work = args.work.resolve()
    ref = args.ref.resolve()

    from openptv2.algorithms.track import trackcorr_c_loop
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / "parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")

        if args.dacc is not None:
            track_par.dacc = args.dacc
        if args.dv is not None:
            track_par.dvxmin = track_par.dvymin = track_par.dvzmin = -args.dv
            track_par.dvxmax = track_par.dvymax = track_par.dvzmax = args.dv

        from openptv2.tracker import Tracker

        tracker = Tracker(
            cpar, vpar, track_par, spar, cals, default_naming, loser_retry=0,
            cold_start_neighbour=0,
        )
        tracker.restart()  # builds TrackingRun + track_forward_start
        run = tracker._run

        # Advance the buffer to the requested step first: after restart()
        # buf[1] holds seq_par.first, so stepping s = first..args.step - 1
        # brings buf[1] to args.step. Calling trackcorr_c_loop(run, s) with
        # any other s processes the buffered frames but labels/writes them
        # as s -- the old code did exactly that for --step != first and
        # compared unrelated frames (plus mislabeled output files).
        first = run.seq_par.first
        if args.step < first:
            raise SystemExit(f"--step {args.step} is before first={first}")
        for s in range(first, args.step):
            trackcorr_c_loop(run, s)

        # Snapshot the frame-1 arrays the loop is about to fill.
        curr = run.fb.buf[1]
        n1 = curr.num_parts
        trackcorr_c_loop(run, args.step)

        # buf rotated; the processed frame is now buf[0]
        done = run.fb.buf[0]
        inlist = np.asarray(done.path_inlist[:n1]).copy()
        linkdecis = np.asarray(done.path_linkdecis[:n1]).copy()
        nxt = np.asarray(done.path_next[:n1]).copy()
    finally:
        os.chdir(old)

    # Reference forward links out of the same frame.
    ref_next = {}
    rp = read_ptv_is(ref / f"ptv_is.{args.step + 1}")
    for j, pj in enumerate(rp):
        if pj >= 0:
            ref_next[pj] = j

    tot = miss = seen = empty = notseen = 0
    examples = []
    for h in range(n1):
        if h not in ref_next:
            continue
        tot += 1
        if nxt[h] == ref_next[h]:
            continue
        miss += 1
        want = ref_next[h]
        il = int(inlist[h])
        cands = [int(c) for c in linkdecis[h][:il]] if il > 0 else []
        if il == 0:
            empty += 1
            tag = "inlist EMPTY"
        elif want in cands:
            seen += 1
            tag = f"partner PRESENT (rank {cands.index(want)}/{il})"
        else:
            notseen += 1
            tag = f"partner ABSENT (inlist={il})"
        if len(examples) < args.show:
            examples.append((h, want, int(nxt[h]), il, tag))

    print(f"\nstep {args.step}: {n1} particles in frame, "
          f"{tot} have a reference forward link")
    print(f"openptv2 reproduced {tot - miss}, missed {miss}\n")
    print(f"  inlist EMPTY (no candidate registered at all) {empty:>6}"
          f"  {empty / miss * 100 if miss else 0:>5.1f}%")
    print(f"  partner PRESENT but not chosen                {seen:>6}"
          f"  {seen / miss * 100 if miss else 0:>5.1f}%")
    print(f"  partner ABSENT from a non-empty inlist        {notseen:>6}"
          f"  {notseen / miss * 100 if miss else 0:>5.1f}%")

    if examples:
        print(f"\n{'particle':>9}{'want':>7}{'got':>7}{'inlist':>8}   verdict")
        for h, want, got, il, tag in examples:
            print(f"{h:>9}{want:>7}{got:>7}{il:>8}   {tag}")


if __name__ == "__main__":
    main()

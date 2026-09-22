"""Drill: is the S4-global failure a decision artifact or history corruption?

Finds an S4 seed where global fails the cross, runs parity on the same
seed, and diffs histories (frames 1..3 must be identical for a clean
resolution-level verdict at frame 4).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proto_global_resolve import run_global
from synth_crossing import (
    FIRST,
    NF,
    REPO,
    load_optics,
    make_truth,
    run_tracker,
    setup_work,
    write_scene,
)


def prev_of(work):
    out = {}
    for f in range(NF):
        lines = (work / "res" / f"ptv_is.{FIRST + f}").read_text().splitlines()
        out[f] = [int(l.split()[0]) for l in lines[1: int(lines[0]) + 1]]
    return out


def main():
    cpar, cals = load_optics()
    for rep in range(20):
        work = REPO / "scratch" / "_synth_drill"
        setup_work(work)
        truth = make_truth(v=0.5, noise=0.07, seed=1000 + rep)
        write_scene(work, truth, cpar, cals)
        run_global(work, 0)
        lines = (work / "res" / f"ptv_is.{FIRST + 4}").read_text().splitlines()
        prev = [int(l.split()[0]) for l in lines[1: int(lines[0]) + 1]]
        if prev != [0, 1]:
            print(f"global fails cross at seed {1000 + rep}: "
                  f"frame4 prev={prev}", flush=True)
            g = prev_of(work)
            setup_work(work)
            write_scene(work, truth, cpar, cals)
            run_tracker(work, 0, 0)
            p = prev_of(work)
            for f in (1, 2, 3):
                mark = "SAME " if g[f] == p[f] else "DIFF "
                print(f"  {mark} frame{f}: global={g[f]} parity={p[f]}",
                      flush=True)
            lines = (work / "res" / f"ptv_is.{FIRST + 4}"
                     ).read_text().splitlines()
            pp = [int(l.split()[0]) for l in lines[1: int(lines[0]) + 1]]
            print(f"  parity frame4 prev={pp}", flush=True)
            return
    print("no failing seed found", flush=True)


if __name__ == "__main__":
    main()

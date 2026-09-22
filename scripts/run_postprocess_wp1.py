"""Willneff-pipeline completion: forward + backward + postprocess on wp1.

The standard openptv2 plugin flow runs tracker.postprocess() (cold-start
seeding, gap relinking, reciprocity enforcement) after tracking -- our
verification runs skipped it. This script measures exactly what it adds:
link efficiency, chain census, and exact reproduction vs the 3dptv reference.

Run from repo root:
    uv run python -u scripts/run_postprocess_wp1.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
WORK = (REPO / "scratch" / "_wp1_ab").resolve()
FIRST, LAST = 100001, 100010
NCAMS = 4


def load_all(res: Path):
    prev, nxt = {}, {}
    for f in range(FIRST, LAST + 1):
        lines = (res / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        p, x = [], []
        for line in lines[1: n + 1]:
            s = line.split()
            p.append(int(s[0]))
            x.append(int(s[1]))
        prev[f], nxt[f] = p, x
    return prev, nxt


def census(prev, nxt, tag: str):
    eff = np.mean([sum(1 for v in nxt[f] if v >= 0) / max(len(nxt[f]), 1)
                   for f in range(FIRST, LAST)])
    links = np.mean([sum(1 for v in nxt[f] if v >= 0)
                     for f in range(FIRST, LAST)])
    chains = []
    for f in range(FIRST, LAST + 1):
        for i in range(len(prev[f])):
            if prev[f][i] < 0:
                ch = [(f, i)]
                cf, ci = f, i
                while cf <= LAST and ci < len(nxt[cf]) and nxt[cf][ci] >= 0:
                    ci = nxt[cf][ci]
                    cf += 1
                    ch.append((cf, ci))
                chains.append(ch)
    L = np.array([len(c) for c in chains])
    import collections
    h = collections.Counter(len(c) for c in chains)
    print(f"{tag}: eff={eff:.1%} links/step={links:.0f} chains={len(chains)} "
          f"meanlen={L.mean():.2f} fullspan={h[LAST - FIRST + 1]} "
          f"len>=5={sum(v for k, v in h.items() if k >= 5)} "
          f"len>=8={sum(v for k, v in h.items() if k >= 8)}", flush=True)
    return chains


def score_links(res: Path, first: int, last: int):
    tot = match = 0
    for f in range(first, last + 1):
        lines = (res / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        t = [int(l.split()[0]) for l in lines[1: n + 1]]
        rlines = (REF / f"ptv_is.{f}").read_text().splitlines()
        rn = int(rlines[0])
        r = [int(l.split()[0]) for l in rlines[1: rn + 1]]
        tot += sum(1 for p in r if p >= 0)
        match += sum(1 for i in range(min(n, rn)) if r[i] >= 0 and t[i] == r[i])
    return tot, match


def main():
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
        for cam in range(1, NCAMS + 1):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                WORK / "img_3dptv" / f"Cam{cam}.{f}_targets")

    old = os.getcwd()
    os.chdir(WORK)
    try:
        pm = ParameterManager()
        pm.from_yaml(WORK / "parameters_Run_dacc19.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=0)
        tr.full_forward()
        tr.full_backward()
        p0, x0 = load_all(WORK / "res")
        census(p0, x0, "fwd+bwd        ")
        t, m = score_links(WORK / "res", 100003, 100010)
        print(f"fwd+bwd exact: {m}/{t} = {m / t:.2%}", flush=True)
        stats = tr.postprocess()
        print(f"postprocess stats: {stats}", flush=True)
        p1, x1 = load_all(WORK / "res")
        census(p1, x1, "fwd+bwd+post   ")
        t, m = score_links(WORK / "res", 100003, 100010)
        print(f"+post exact: {m}/{t} = {m / t:.2%}", flush=True)
    finally:
        os.chdir(old)


if __name__ == "__main__":
    main()

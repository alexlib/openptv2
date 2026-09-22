"""Narrow replay of the candidate SEARCH stage (searchquader + per-cam
4-nearest + freq merge), validated against the kernel's own inlist arrays
on control particles before trusting it on refusal cases.

Question: do small-step refusals (e.g. 100005:1242 -> 100006:535, step
0.73, all gates pass, targets present with correct tnr) die because the
true target falls out of the per-cam top-4 in crowded regions
(prediction offset 1.3mm + 5 neighbours)?

Run from repo root:
    uv run python -u scripts/probe_search.py --work scratch/_wp1_ab
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


def load_targets(work: Path, frame: int, ncams: int):
    per_cam = []
    base = work / "img_3dptv"
    for cam in range(1, ncams + 1):
        # target base names come from the yaml; resolve the same way
        cands = sorted(base.glob(f"Cam{cam}.{frame}_targets"))
        if not cands:
            cands = sorted(base.glob(f"cam{cam}.{frame}_targets"))
        tl = cands[0].read_text().splitlines()
        xs, ys, tnrs = [], [], []
        for line in tl[1:]:
            s = line.split()
            xs.append(float(s[1]))
            ys.append(float(s[2]))
            tnrs.append(int(s[7]))
        per_cam.append((np.array(xs), np.array(ys), np.array(tnrs)))
    return per_cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--cases", type=str, default="100005:1242:535",
                    help="f1:h:want,row triples (frame,row-in-f1,true partner)")
    ap.add_argument("--n-controls", type=int, default=60)
    args = ap.parse_args()
    work = args.work.resolve()

    from openptv2.algorithms.track import (
        point_to_pixel,
        searchquader,
        trackcorr_c_loop,
    )
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
        track_par.dacc = 1.9
        ncams = cpar.num_cams

        DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
        REF = DS / "res_ground_truth_backup"
        # pristine inputs: the kernel rewrites rt/tnr in place
        for f in range(100001, 100007):
            shutil.copyfile(REF / f"rt_is.{f}", work / "res" / f"rt_is.{f}")
            for cam in range(1, ncams + 1):
                shutil.copyfile(
                    DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                    work / "img_3dptv" / f"Cam{cam}.{f}_targets")

        def rt(f):
            lines = (REF / f"rt_is.{f}").read_text().splitlines()
            n = int(lines[0])
            return np.array([[float(v) for v in l.split()[1:4]]
                             for l in lines[1: n + 1]])

        def replay_freq(X2, f2, topk):
            """freq dict {particle_row: cams} for frame-f2 search."""
            xr, xl, yd, yu = searchquader(X2, track_par, cpar, cals)
            per_cam = load_targets(work, f2, ncams)
            freq: dict = {}
            for cam in range(ncams):
                cx, cy = point_to_pixel(X2, cals[cam], cpar)
                xmin, xmax = cx - xl[cam], cx + xr[cam]
                ymin, ymax = cy - yu[cam], cy + yd[cam]
                xs, ys, tnrs = per_cam[cam]
                inside = [(np.hypot(xs[j] - cx, ys[j] - cy), int(tnrs[j]))
                          for j in range(len(xs))
                          if xmin < xs[j] < xmax and ymin < ys[j] < ymax
                          and tnrs[j] != -1]
                inside.sort()
                for _, t in inside[:topk]:
                    freq[t] = freq.get(t, 0) + 1
            return freq

        # kernel truth for controls: run to step, read inlist/linkdecis
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=0)
        tr.restart()
        run = tr._run
        for s in range(100001, 100005):
            trackcorr_c_loop(run, s)
        done = run.fb.buf[0]  # processed frame 100004... we need 100005
        # one more step to fill inlist for frame 100005
        trackcorr_c_loop(run, 100005)
        done = run.fb.buf[0]
        n1 = done.num_parts
        kinlist = np.asarray(done.path_inlist[:n1]).copy()
        klink = np.asarray(done.path_linkdecis[:n1]).copy()
        # TEST's own history (not the reference's -- cascade may differ)
        tpl = (work / "res" / "ptv_is.100005").read_text().splitlines()
        tprev5 = [int(l.split()[0]) for l in tpl[1: int(tpl[0]) + 1]]

        # controls: particles the kernel linked (inlist>0), sample randomly
        rng = np.random.default_rng(0)
        cands = [h for h in range(n1) if int(kinlist[h]) > 0]
        controls = rng.choice(cands, size=min(args.n_controls, len(cands)),
                              replace=False)
        X1 = rt(100005)
        X0 = rt(100004)
        agree = tot = 0
        for h in controls:
            p = tprev5[h] if h < len(tprev5) else -1
            if p < 0 or p >= len(X0):
                continue
            X2 = 2 * X1[h] - X0[p]
            fq = replay_freq(X2, 100006, 4)
            tot += 1
            # kernel's best candidate must be among replay's freq set
            best = int(klink[h][0])
            agree += 1 if best in fq else 0
        print(f"control validation: kernel-best candidate in replay freq "
              f"set on {agree}/{tot}", flush=True)

        for spec in args.cases.split(","):
            f1s, hs, ws = spec.split(":")
            f1, h, want = int(f1s), int(hs), int(ws)
            f2 = f1 + 1
            p = tprev5[h]
            X2 = 2 * X1[h] - X0[p]
            print(f"\ncase frame {f1} row {h} want {want}: |X3-X2|="
                  f"{np.linalg.norm(rt(f2)[want] - X2):.3f}mm", flush=True)
            for topk in (4, 8):
                fq = replay_freq(X2, f2, topk)
                got = fq.get(want, 0)
                print(f"  top-{topk}: true partner in {got} cams "
                      f"(need>=2); total cands={len(fq)}", flush=True)
            print(f"  kernel inlist[{h}]={int(kinlist[h])} "
                  f"linkdecis={list(map(int, klink[h][:int(kinlist[h])]))}",
                  flush=True)
    finally:
        os.chdir(old)


if __name__ == "__main__":
    main()

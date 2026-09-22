"""Full verification on the wp1 dataset folder: same parameters => same trajectories?

Runs openptv2 tracking IN the dataset work folder
(C:\\Users\\alex\\Downloads\\HiDImaging\\wp1_10_images) with dacc=1.9 to match
the 3dptv reference, then compares against res_ground_truth_backup at two
levels: per-link exact reproduction and full trajectory-chain reproduction.

Safety: inputs (rt_is.*, Cam*_targets) are restored from pristine sources
before the run. Outputs (res/ptv_is.*, res/added.*, res/rt_is.*) are the
comparison subject. Target-file tnr columns dirtied by the run are restored
afterwards by scripts/restore_dataset.py.

Run from repo root:
    uv run python -u scripts/verify_same_trajectories.py [--backward]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
FIRST, LAST = 100001, 100010
SCORE_FIRST, SCORE_LAST = 100003, 100010


def restore_inputs(work: Path) -> None:
    res = work / "res"
    res.mkdir(exist_ok=True)
    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", res / f"rt_is.{f}")
        for cam in range(1, 5):
            src = DS / "img_3dptv" / f"Cam{cam}.{f}_targets"
            dst = work / "img_3dptv" / f"Cam{cam}.{f}_targets"
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)


def check_consistency(work: Path) -> int:
    """rt_is p-columns must be the inverse of target-file tnr columns.

    Returns mismatches on CLEAN frames only; frame 100001 is a known
    corrupt frame (rt p-columns disagree with targets) and is reported
    separately without aborting.
    """
    bad_clean = bad_100001 = 0
    for f in range(FIRST, LAST + 1):
        rt = (REF / f"rt_is.{f}").read_text().strip().splitlines()
        n = int(rt[0])
        tnr = {}
        for cam in range(1, 5):
            tl = (work / "img_3dptv" / f"Cam{cam}.{f}_targets"
                  ).read_text().strip().splitlines()
            tnr[cam] = {int(x.split()[0]): int(x.split()[7]) for x in tl[1:]}
        bad = 0
        for line in rt[1: n + 1]:
            p = line.split()
            row = int(p[0])
            for cam in range(1, 5):
                j = int(p[3 + cam])
                t = tnr[cam].get(j, -1)
                if j >= 0 and t not in (row - 1, row):
                    bad += 1
        if f == 100001:
            bad_100001 = bad
        else:
            bad_clean += bad
    print(f"consistency: clean-frame mismatches={bad_clean}, "
          f"frame-100001 mismatches={bad_100001} (known corrupt)", flush=True)
    return bad_clean


def run_tracking(work: Path, do_backward: bool, params_file: str,
                 dacc: float | None, app: float = 0.0) -> None:
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / params_file)
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        if dacc is not None:
            track_par.dacc = dacc
        print(f"effective params from {params_file}"
              f"{' + dacc override' if dacc is not None else ''}: "
              f"dv={track_par.dvxmax} dacc={track_par.dacc} "
              f"dangle={track_par.dangle} add={track_par.add}", flush=True)
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=0, app_weight=app)
        tr.full_forward()
        if do_backward:
            tr.full_backward()
    finally:
        os.chdir(old)


def read_pn(res: Path, frame: int):
    """prev/next columns of ptv_is.<frame>."""
    lines = (res / f"ptv_is.{frame}").read_text().strip().splitlines()
    n = int(lines[0])
    prev, nxt = [], []
    for line in lines[1: n + 1]:
        p = line.split()
        prev.append(int(p[0]))
        nxt.append(int(p[1]))
    return prev, nxt


def compare_links(work: Path, first: int, last: int, tag: str) -> tuple:
    ref_total = test_total = match = mis = 0
    for f in range(first, last + 1):
        r, _ = read_pn(REF, f)
        t, _ = read_pn(work / "res", f)
        ref_total += sum(1 for p in r if p >= 0)
        test_total += sum(1 for p in t if p >= 0)
        for i in range(min(len(r), len(t))):
            if r[i] >= 0:
                if t[i] == r[i]:
                    match += 1
                else:
                    mis += 1
    print(f"[{tag}] links frames {first}..{last}: ref={ref_total} "
          f"test={test_total} exact={match} miss={mis} "
          f"recall={match / ref_total:.2%}", flush=True)
    return ref_total, test_total, match


def compare_tracks(work: Path) -> None:
    """Chain-level comparison over rows shared with the reference.

    Added particles append rows, so reference rows keep their indices in the
    test output; chains are comparable as (frame, row) node sequences.
    """
    # next pointers per frame for both runs
    ref_nxt, tst_nxt, ref_prev, tst_prev, ref_n = {}, {}, {}, {}, {}
    for f in range(FIRST, LAST + 1):
        rp, rn = read_pn(REF, f)
        tp, tn = read_pn(work / "res", f)
        ref_prev[f], ref_nxt[f] = rp, rn
        tst_prev[f], tst_nxt[f] = tp, tn
        ref_n[f] = len(rp)

    def chains(prev, nxt, nrows, first, last):
        out = []
        for f in range(first, last + 1):
            for i in range(min(nrows[f], len(prev[f]))):
                if prev[f][i] < 0:
                    ch = [(f, i)]
                    cf, ci = f, i
                    while cf <= last and ci < len(nxt[cf]) and nxt[cf][ci] >= 0:
                        ci = nxt[cf][ci]
                        cf += 1
                        ch.append((cf, ci))
                    out.append(tuple(ch))
        return out

    ref_chains = chains(ref_prev, ref_nxt, ref_n, FIRST, LAST)
    tst_chains = chains(tst_prev, tst_nxt, ref_n, FIRST, LAST)
    tst_set = set(tst_chains)
    exact = sum(1 for c in ref_chains if c in tst_set)
    print(f"[tracks] reference chains: {len(ref_chains)}, test chains "
          f"(on shared rows): {len(tst_chains)}, exactly reproduced "
          f"end-to-end: {exact} ({exact / len(ref_chains):.2%})", flush=True)

    import numpy as np
    rl = np.array([len(c) for c in ref_chains])
    tl = np.array([len(c) for c in tst_chains])
    print(f"[tracks] ref length: mean {rl.mean():.2f} median "
          f"{np.median(rl):.0f} max {rl.max()}; test length: mean "
          f"{tl.mean():.2f} median {np.median(tl):.0f} max {tl.max()}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=DS)
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--params", default="parameters_Run1.yaml",
                    help="yaml file in work dir (use a parity yaml to test "
                    "config-level equality with track.par)")
    ap.add_argument("--dacc", type=float, default=None,
                    help="in-memory dacc override; omit to use the yaml value")
    ap.add_argument("--app", type=float, default=0.0,
                    help="appearance weight (requires rebuilt kernels)")
    ap.add_argument("--skip-run", action="store_true",
                    help="only compare existing outputs")
    args = ap.parse_args()
    work = args.work.resolve()

    if not args.skip_run:
        restore_inputs(work)
        print("inputs restored from pristine sources", flush=True)
        bad = check_consistency(work)
        if bad:
            sys.exit(f"ABORT: {bad} inconsistent correspondences")
        run_tracking(work, args.backward, args.params, args.dacc, args.app)

    compare_links(work, SCORE_FIRST, SCORE_LAST, "clean")
    compare_links(work, FIRST + 1, LAST, "all")
    compare_tracks(work)


if __name__ == "__main__":
    main()

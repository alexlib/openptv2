"""Synthetic crossing laboratory: which decision is correct in hard cases?

Two (or three) particles cross between frames with KNOWN ground truth, so
every contested decision is scored instead of debated. Answers the case-2
question empirically: violent-vs-smooth -- who is right, and under which
conditions does openptv2 choose wrong?

Scenes use the real wp1 calibration (4 cams) + real position noise
(sigma=0.07 mm). Work dir is a scratch copy of _wp1_ab with the dacc=1.9
parity yaml, so optics/params match the verification runs.

Scenarios (A/B cross between frames 3->4 of 0..6, speed v mm/frame):
  S1 head-on     : miss distance eps=0
  S2 near-miss   : eps=0.15 (below noise floor)
  S3 near-miss   : eps=0.40
  S4 head-on+noise
  S5 fast        : v=1.2, eps=0
  S6 three-way   : C crosses through the intersection at the cross frame
  S7 asymmetric  : vA=0.7, vB=0.35

Run from repo root:
    uv run python -u scripts/synth_crossing.py [--reps N]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.parameters import ControlPar

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scratch" / "_wp1_ab"
NCAMS = 4
FIRST = 200001
NF = 10
X0, Y0, Z0 = 39.0, 43.0, -40.0  # real-data-like, verified visible below


def setup_work(work: Path) -> None:
    if work.exists():
        shutil.rmtree(work)
    (work / "img_3dptv").mkdir(parents=True)
    (work / "res").mkdir(parents=True)
    shutil.copytree(SRC / "cal", work / "cal")
    shutil.copytree(SRC / "parameters", work / "parameters")
    shutil.copy(SRC / "parameters_Run_dacc19.yaml",
                work / "parameters_Run1.yaml")
    y = yaml.safe_load((work / "parameters_Run1.yaml").read_text())
    for key in ("sequence", "Sequence", "sequence_par"):
        if key in y:
            y[key]["first"] = FIRST
            y[key]["last"] = FIRST + NF - 1
    (work / "parameters_Run1.yaml").write_text(yaml.safe_dump(y))


def load_optics():
    cpar = ControlPar.from_file(str(SRC / "parameters" / "ptv.par"))
    cals = [
        Calibration.from_file(
            str(SRC / "cal" / f"cam_{c}.tif.ori"),
            str(SRC / "cal" / f"cam_{c}.tif.addpar"),
        )
        for c in range(1, NCAMS + 1)
    ]
    return cpar, cals


def project(cpar, cals, x, y, z):
    out = []
    for ci in range(NCAMS):
        xy = img_coord_batch(np.array([[x, y, z]], dtype=np.float64),
                             cals[ci], cpar.mm)[0]
        px = xy[0] / cpar.pix_x + cpar.imx / 2
        py = cpar.imy / 2 - xy[1] / cpar.pix_y
        out.append((px, py))
    return out


def write_scene(work: Path, truth, cpar, cals, sumg_fn=None,
                sumg_jitter: float = 0.0, seed: int = 0):
    """truth: {pid: {frame: (x,y,z)}} (a pid may skip frames = occlusion).

    Rows in pid order per frame; targets y-sorted. Returns per-frame
    row maps {frame: {pid: row}} (rows shift when a pid is absent, so
    scoring must use these, not a static map).

    sumg_fn(pid, frame) -> base grey sum for that particle (default 1000);
    per-target detection jitter ~ N(0, sumg_jitter * base) models
    illumination/view variation.
    """
    import math as _math

    rng = np.random.default_rng(seed)
    if sumg_fn is None:
        sumg_fn = lambda pid, frame: 1000.0  # noqa: E731
    pids = sorted(truth)
    rows_pf = {}
    for f in range(NF):
        frame = FIRST + f
        present = [pid for pid in pids if frame in truth[pid]]
        rows_pf[frame] = {pid: k for k, pid in enumerate(present)}
        det = {c: [] for c in range(NCAMS)}  # (px, py, pid)
        for pid in present:
            x, y, z = truth[pid][frame]
            for ci, (px, py) in enumerate(project(cpar, cals, x, y, z)):
                assert 0 <= px < cpar.imx and 0 <= py < cpar.imy, \
                    f"pid {pid} frame {frame} cam {ci} out of view " \
                    f"({px:.1f},{py:.1f})"
                det[ci].append((px, py, pid))
        per_cam, pnrs_of = {}, {}
        for ci in range(NCAMS):
            det[ci].sort(key=lambda t: t[1])  # y-sort for candsearch
            per_cam[ci] = []
            for idx, (px, py, pid) in enumerate(det[ci]):
                base = float(sumg_fn(pid, frame))
                sg = base + (rng.normal(0, sumg_jitter * base)
                             if sumg_jitter else 0.0)
                per_cam[ci].append((idx, px, py, 50, 5, 5, max(sg, 1.0),
                                    rows_pf[frame][pid]))
                pnrs_of[(pid, ci)] = idx
        rt_rows = []
        for pid in present:
            x, y, z = truth[pid][frame]
            pnrs = [pnrs_of[(pid, ci)] for ci in range(NCAMS)]
            rt_rows.append((len(rt_rows) + 1, x, y, z, *pnrs))
        with open(work / "res" / f"rt_is.{frame}", "w") as fh:
            fh.write(f"{len(rt_rows)}\n")
            for r in rt_rows:
                fh.write(f"{r[0]:4d}  {r[1]:12.6f}  {r[2]:12.6f}  "
                         f"{r[3]:12.6f}  {r[4]:5d}  {r[5]:5d}  "
                         f"{r[6]:5d}  {r[7]:5d}\n")
        for ci in range(NCAMS):
            with open(work / "img_3dptv" / f"Cam{ci + 1}.{frame}_targets",
                      "w") as fh:
                fh.write(f"{len(per_cam[ci])}\n")
                for t in per_cam[ci]:
                    fh.write(f"{t[0]:4d}  {t[1]:10.4f}  {t[2]:10.4f}  "
                             f"{t[3]:6d}  {t[4]:4d}  {t[5]:4d}  "
                             f"{int(round(t[6])):6d}  {t[7]:4d}\n")
    return rows_pf


def run_tracker(work: Path, loser_retry: int, cold_start: int,
                app_weight: float = 0.0):
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
                loser_retry=loser_retry, cold_start_neighbour=cold_start,
                app_weight=app_weight).full_forward()
    finally:
        os.chdir(old)


def score(work: Path, rows_pf):
    """Fraction of true links recovered + the crossing decision (frame 4).

    rows_pf: {frame: {pid: row}}. A pid absent in frame f-1 but present in
    f with a link to its last-seen row counts as recovered (gap link).
    """
    n_ok = n_tot = 0
    cross_ok = None
    last_row = dict(rows_pf[FIRST])
    for f in range(1, NF):
        frame = FIRST + f
        lines = (work / "res" / f"ptv_is.{frame}").read_text().splitlines()
        n = int(lines[0])
        prev = [int(l.split()[0]) for l in lines[1: n + 1]]
        for pid, row_now in rows_pf[frame].items():
            if pid not in last_row:
                continue  # track head (first appearance), no link expected
            n_tot += 1
            ok = row_now < len(prev) and prev[row_now] == last_row[pid]
            n_ok += ok
            if f == 4:
                cross_ok = (cross_ok is not False) and bool(ok)
                if not ok:
                    cross_ok = False
        for pid, row_now in rows_pf[frame].items():
            last_row[pid] = row_now
    return n_ok / n_tot if n_tot else 1.0, \
        (cross_ok if cross_ok is not None else True)


def make_truth(v, eps=0.0, noise=0.0, seed=0, three_way=False, asym=None,
               maneuver=None, convoy=None, dropout=False, pulsatile=None):
    rng = np.random.default_rng(seed)
    if asym is not None:
        va, vb = asym
    else:
        va = vb = v

    def jit(x, y, z):
        if noise:
            return (x + rng.normal(0, noise), y + rng.normal(0, noise),
                    z + rng.normal(0, noise))
        return (x, y, z)

    truth = {0: {}, 1: {}}
    for f in range(NF):
        if maneuver == "kick":
            # A crawls in at 0.15, then jumps 1.0 through the crossing
            # (the case-2 shape: sudden acceleration AT the crossing).
            xa = [X0 - 0.45, X0 - 0.30, X0 - 0.15, X0,
                  X0 + 1.0, X0 + 2.0, X0 + 3.0, X0 + 4.0, X0 + 5.0,
                  X0 + 6.0][f]
            xb = X0 - 0.5 * (f - 3.5)
        elif maneuver == "brake":
            # A rushes in at 1.0, then nearly stops after crossing.
            xa = [X0 - 3.0, X0 - 2.0, X0 - 1.0, X0,
                  X0 + 0.15, X0 + 0.30, X0 + 0.45, X0 + 0.60, X0 + 0.75,
                  X0 + 0.90][f]
            xb = X0 - 0.5 * (f - 3.5)
        else:
            xa = X0 + va * (f - 3.5)
            xb = X0 - vb * (f - 3.5)
        truth[0][FIRST + f] = jit(xa, Y0, Z0)
        truth[1][FIRST + f] = jit(xb, Y0, Z0 + eps)
    if three_way:
        # C crosses along z through the intersection at the cross frame
        truth[2] = {}
        for f in range(NF):
            truth[2][FIRST + f] = jit(X0, Y0, Z0 - 1.0 * (f - 3.5))
    if convoy:
        # second head-on pair, offset in y by dy, same timing
        dy = convoy
        truth[2], truth[3] = {}, {}
        for f in range(NF):
            truth[2][FIRST + f] = jit(X0 + 0.5 * (f - 3.5), Y0 + dy, Z0)
            truth[3][FIRST + f] = jit(X0 - 0.5 * (f - 3.5), Y0 + dy, Z0)
    if dropout:
        # third particle, occluded exactly at frame FIRST+3 (gap of 1).
        # Kept as the LAST pid so surviving rows never shift.
        truth[2] = {}
        for f in range(NF):
            if f == 3:
                continue
            truth[2][FIRST + f] = jit(X0 + 3.0 + 0.1 * f, Y0 + 2.0, Z0)
    if pulsatile:
        # aorta-like plug: x(t) = X0 + A*sin(2*pi*(f-3.5)/T + phase).
        # Anti-phase pair crosses WHILE reversing (worst case); in-phase
        # pair reverses without crossing (isolates reversal cost).
        import math as _math
        vmax = pulsatile.get("vmax", 1.5)
        T = float(pulsatile.get("period", NF))
        A = vmax * T / (2 * _math.pi)
        modes = pulsatile.get("modes", ("anti",))
        truth = {}
        pid = 0
        if "anti" in modes:
            for ph in (0.0, _math.pi):
                truth[pid] = {}
                for f in range(NF):
                    x = X0 + A * _math.sin(2 * _math.pi * (f - 3.5) / T + ph)
                    truth[pid][FIRST + f] = jit(x, Y0, Z0)
                pid += 1
        if "plug" in modes:
            for dy in (-1.5, 1.5):
                truth[pid] = {}
                for f in range(NF):
                    x = X0 + A * _math.sin(2 * _math.pi * (f - 3.5) / T)
                    truth[pid][FIRST + f] = jit(x, Y0 + dy, Z0)
                pid += 1
        if "solo" in modes:
            # one oscillator (vmax may exceed the dv box) + a steady
            # neighbour 1.0 mm away in y: diastole swap-bait that tight
            # gates must reject and systole must not lose.
            truth[pid] = {}
            for f in range(NF):
                x = X0 + A * _math.sin(2 * _math.pi * (f - 3.5) / T)
                truth[pid][FIRST + f] = jit(x, Y0, Z0)
            pid += 1
            truth[pid] = {}
            for f in range(NF):
                truth[pid][FIRST + f] = jit(X0 + 1.0, Y0 + 1.0, Z0)
            pid += 1
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--app", type=float, default=0.0,
                    help="appearance weight for every run in the sweep")
    args = ap.parse_args()

    cpar, cals = load_optics()
    # visibility sanity at the crossing point, all cams
    for ci, (px, py) in enumerate(project(cpar, cals, X0, Y0, Z0)):
        assert 0 <= px < cpar.imx and 0 <= py < cpar.imy, (ci, px, py)
    print(f"imx={cpar.imx} imy={cpar.imy} pix={cpar.pix_x},{cpar.pix_y} "
          f"crossing point visible in all {NCAMS} cams")

    scenarios = [
        ("S1 head-on v=0.5", {"v": 0.5}),
        ("S2 near-miss e=0.15", {"v": 0.5, "eps": 0.15}),
        ("S3 near-miss e=0.40", {"v": 0.5, "eps": 0.40}),
        ("S4 head-on+noise", {"v": 0.5, "noise": 0.07}),
        ("S5 fast v=1.2", {"v": 1.2}),
        ("S6 three-way", {"v": 0.5, "three_way": True}),
        ("S7 asymmetric", {"v": 0.5, "asym": (0.7, 0.35)}),
        ("S8 kick-at-cross", {"v": 0.5, "maneuver": "kick"}),
        ("S9 brake-at-cross", {"v": 0.5, "maneuver": "brake"}),
        ("S10 convoy dy=0.6", {"v": 0.5, "convoy": 0.6}),
        ("S11 kick+noise", {"v": 0.5, "maneuver": "kick", "noise": 0.07}),
        ("S12 convoy+noise", {"v": 0.5, "convoy": 0.6, "noise": 0.07}),
        ("S14 anti-phase", {"v": 0.5,
                            "pulsatile": {"vmax": 1.5, "modes": ("anti",)}}),
        ("S14b plug-reverse", {"v": 0.5,
                               "pulsatile": {"vmax": 1.5,
                                             "modes": ("plug",)}}),
        ("S15 bright-kick", {"v": 0.5, "maneuver": "kick", "noise": 0.07,
                             "_sumg": {0: 2000.0, 1: 500.0},
                             "_sj": 0.10}),
    ]
    variants = [("parity(lr0)", 0, 0), ("default(lr1)", 1, 1)]

    print(f"\n{'scenario':22}{'variant':14}{'recall':>9}{'cross-ok':>10}")
    for sname, kw in scenarios:
        kw = dict(kw)
        sumg_map = kw.pop("_sumg", None)
        sumg_jitter = kw.pop("_sj", 0.0)
        noisy = kw.get("noise", 0.0) > 0 or sumg_jitter > 0
        reps = args.reps if (noisy or kw.get("three_way")
                             or kw.get("convoy") or kw.get("pulsatile")
                             or sumg_map) else 1
        for vname, lr, cs in variants:
            recs, crs = [], []
            for rep in range(reps):
                work = REPO / "scratch" / "_synth_x"
                setup_work(work)
                truth = make_truth(seed=1000 + rep, **kw)
                if sumg_map:
                    sfn = lambda pid, frame, _m=sumg_map: _m.get(pid, 1000.0)  # noqa: E731
                else:
                    sfn = None
                rows_pf = write_scene(work, truth, cpar, cals,
                                      sumg_fn=sfn, sumg_jitter=sumg_jitter,
                                      seed=2000 + rep)
                run_tracker(work, lr, cs, app_weight=args.app)
                r, c = score(work, rows_pf)
                recs.append(r)
                crs.append(c)
            print(f"{sname:22}{vname:14}{np.mean(recs):>8.1%}  "
                  f"{sum(crs)}/{len(crs)}")


if __name__ == "__main__":
    main()

"""Phase-aware scheduler for pulsatile flow (aorta case).

Problem: global dv/dacc gates fit exactly one regime. In pulsatile flow,
systole needs wide gates (fast + accelerating), diastole needs tight ones
(slow + reversal-prone, swaps lurk). One fixed setting loses one phase.

Scheduler (no kernel change -- pure Python around trackcorr_c_loop):
per step, measure the previous step's median link step M (robust) and its
spread. Scale = clamp(M / anchor, lo, hi) where anchor = running median
of past step-medians (self-calibrating, no nominal needed). dv bounds and
dacc scale together (ratio preserved). Reversal/strain detector:
IQR/M > rev_ratio (mixed fast/slow = turning field) boosts dacc only,
because reversal breaks the predictor while positions stay catchable.

First two steps run at scale 1 (no statistics yet).

Run from repo root:
    uv run python -u scripts/phase_scheduler.py [--reps N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
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


def run_adaptive(work: Path, lr: int, cs: int, app: float = 0.0,
                 lo: float = 0.5, hi: float = 2.0, rev_boost: float = 1.5,
                 rev_ratio: float = 2.5, verbose: bool = False,
                 first: int | None = None, last: int | None = None,
                 scale_dv: bool = True, scale_dacc: bool = True,
                 yaml: str = "parameters_Run1.yaml"):
    from openptv2.algorithms.track import trackcorr_c_finish, trackcorr_c_loop
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / yaml)
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=lr, cold_start_neighbour=cs, app_weight=app)
        tr.restart()
        run = tr._run
        base = run.tpar
        if first is None:
            first = FIRST
        if last is None:
            last = FIRST + NF - 1
        meds: list[float] = []
        scales = []
        last_iqr = (0.0, 0.0)
        scale = 1.0
        prev_lost = 0.0
        losts: list[float] = []
        for step in range(first, last):
            # dacc gets the reversal boost on top of the dv scale, both
            # from PREVIOUS-step statistics (causal: nothing from the
            # future is used).
            dscale = scale
            iqr, med = last_iqr
            if med > 0 and iqr / med > rev_ratio:
                dscale = min(scale * rev_boost, 3.0)
            sv = scale if scale_dv else 1.0
            sa = dscale if scale_dacc else 1.0
            run.tpar = base._replace(
                dvxmin=base.dvxmin * sv, dvxmax=base.dvxmax * sv,
                dvymin=base.dvymin * sv, dvymax=base.dvymax * sv,
                dvzmin=base.dvzmin * sv, dvzmax=base.dvzmax * sv,
                dacc=base.dacc * sa)
            scales.append(round(scale, 2))
            trackcorr_c_loop(run, step)
            # post-rotation: buf[0] = processed frame, buf[1] = next
            b0, b1 = run.fb.buf[0], run.fb.buf[1]
            nx = np.asarray(b0.path_next[:b0.num_parts])
            pv = np.asarray(b0.path_prev[:b0.num_parts])
            # Starvation among particles WITH history only: cold-start
            # losses (prev<0, early steps) are not systole and must not
            # trigger widening (that swap-bombs dense data).
            has_hist = pv >= 0
            if has_hist.sum() >= 1:
                lost = 1.0 - (nx[has_hist] >= 0).sum() / has_hist.sum()
            else:
                lost = 0.0
            if len(losts) >= 2 and lost > float(np.median(losts)) + 0.10:
                scale = min(hi, scale * 1.3)
            else:
                scale = max(1.0, scale * 0.9)
            losts.append(lost)
            prev_lost = lost
            ok = nx >= 0
            if ok.sum() >= 3:
                st = np.linalg.norm(
                    np.asarray(b1.path_x)[nx[ok]] -
                    np.asarray(b0.path_x)[:b0.num_parts][ok], axis=1)
                meds.append(float(np.median(st)))
                q75, q25 = np.percentile(st, [75, 25])
                last_iqr = (float(q75 - q25), float(np.median(st)))
        if verbose:
            print(f"  scales: {scales}", flush=True)
        trackcorr_c_finish(run, last)
    finally:
        os.chdir(old)


def run_adaptive_pp(work: Path, lr: int, cs: int, app: float = 0.0,
                    hi: float = 2.5, verbose: bool = False,
                 first: int | None = None, last: int | None = None,
                 mode: str = "vel", yaml: str = "parameters_Run1.yaml",
                 backward: bool = False):
    """Per-particle gate freedom (safe for dense data).

    mode="vel": scale = clamp(|v|/v_typical, 1, hi). Too hot: steady-fast
        particles get max scale and swap-bomb dense data (wp1 88%).
    mode="acc" (default): scale = clamp(1 + |a|/dacc, 1, hi) where a is
        the particle's last acceleration magnitude. Steady motion
        (fast OR slow) keeps tight gates -- prediction is trustworthy;
        only unpredictable (accelerating) particles get room. Noise
        (~0.3) maps to a gentle ~1.16.
    Density-gated crowd veto (automatic regime split): the swap-bomb
        mechanism is density-dependent (lies need a crowd of smooth
        rivals). In dense data (num_parts >= 100) freedom is vetoed
        wherever another particle sits near the predicted position
        (judged at PREDICTION, radius = own widened box + margin, self
        counts: only the truly lonely are freed). In sparse data the veto
        is skipped -- a lone bystander loses to truth on cost, and vetoing
        it would kill genuine systole rescues (S14c). Dense+systole
        combined remains an open case (needs a dense-systole truth scene).
    Communicated via run.gate_scale (no tpar mutation).
    """
    from openptv2.algorithms.track import trackcorr_c_finish, trackcorr_c_loop
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / yaml)
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=lr, cold_start_neighbour=cs, app_weight=app)
        tr.restart()
        run = tr._run
        if first is None:
            first = FIRST
        if last is None:
            last = FIRST + NF - 1
        dacc0 = float(run.tpar.dacc)
        # (frame, row) -> last velocity vector (for acceleration magnitude)
        vel: dict = {}
        run_adaptive_pp._speeds = []
        _sparse = int(run.fb.buf[1].num_parts) < 100
        if verbose:
            print(f"  [pp] density gate: sparse={_sparse} "
                  f"(n={int(run.fb.buf[1].num_parts)})", flush=True)
        for step in range(first, last):
            b1 = run.fb.buf[1]
            maxp = b1.path_x.shape[0]
            gs = np.ones(maxp, dtype=np.float64)
            if mode == "acc":
                pv = np.asarray(b1.path_prev[:b1.num_parts])
                px0 = np.asarray(run.fb.buf[0].path_x)
                px1 = np.asarray(b1.path_x)
                for j in range(b1.num_parts):
                    p = int(pv[j])
                    if p < 0:
                        continue
                    v_new = px1[j] - px0[p]
                    # velocity that arrived at (frame step-1, row p), keyed
                    # by time index t = frame - first
                    v_old = vel.get((step - first, p))
                    if v_old is not None:
                        a = float(np.linalg.norm(v_new - v_old))
                        gs[j] = min(1.0 + a / dacc0, hi)
            else:  # mode == "vel" (legacy experiment, too hot)
                speeds = getattr(run_adaptive_pp, "_speeds", [])
                if speeds:
                    vtyp = float(np.median(speeds))
                    if vtyp > 0 and step > first:
                        pv = np.asarray(b1.path_prev[:b1.num_parts])
                        px0 = np.asarray(run.fb.buf[0].path_x)
                        px1 = np.asarray(b1.path_x)
                        for j in range(b1.num_parts):
                            p = int(pv[j])
                            if 0 <= p:
                                v = float(np.linalg.norm(px1[j] - px0[p]))
                                gs[j] = min(max(v / vtyp, 1.0), hi)
            # Crowd gate (dense data only; see docstring): veto freedom
            # wherever another particle sits near the predicted position.
            if not _sparse and float(gs.max()) > 1.0:
                from scipy.spatial import cKDTree
                _tp = run.tpar
                _dv = max(abs(_tp.dvxmin), abs(_tp.dvxmax),
                          abs(_tp.dvymin), abs(_tp.dvymax),
                          abs(_tp.dvzmin), abs(_tp.dvzmax))
                _px1 = np.asarray(b1.path_x)
                _tree = cKDTree(_px1[:b1.num_parts])
                _pv = np.asarray(b1.path_prev[:b1.num_parts])
                _px0 = np.asarray(run.fb.buf[0].path_x)
                _npre, _nveto = int((gs > 1.0).sum()), 0
                for j in range(b1.num_parts):
                    if gs[j] <= 1.0:
                        continue
                    p = int(_pv[j])
                    if p < 0:
                        continue
                    _vj = _px1[j] - _px0[p]
                    _pred = _px1[j] + _vj
                    _r = _dv * float(gs[j]) + 1.0
                    if len(_tree.query_ball_point(_pred, r=_r)) > 0:
                        gs[j] = 1.0
                        _nveto += 1
                if verbose and _npre:
                    print(f"  [pp] step {step}: pre={_npre} vetoed={_nveto} "
                          f"free={_npre - _nveto}", flush=True)
            run.gate_scale = gs
            trackcorr_c_loop(run, step)
            # harvest velocities keyed by (new frame, row)
            c0, c1 = run.fb.buf[0], run.fb.buf[1]
            nx = np.asarray(c0.path_next[:c0.num_parts])
            ok = nx >= 0
            if ok.sum():
                X0 = np.asarray(c0.path_x)[:c0.num_parts]
                X1 = np.asarray(c1.path_x)
                for h in range(c0.num_parts):
                    if nx[h] >= 0:
                        vel[(step + 1 - first, int(nx[h]))] = \
                            X1[int(nx[h])] - X0[h]
            if mode == "vel":
                if ok.sum():
                    st = np.linalg.norm(
                        np.asarray(c1.path_x)[nx[ok]] -
                        np.asarray(c0.path_x)[:c0.num_parts][ok], axis=1)
                    run_adaptive_pp._speeds = \
                        getattr(run_adaptive_pp, "_speeds", []) + \
                        [float(v) for v in st]
        if verbose:
            print(f"  mode={mode}", flush=True)
        trackcorr_c_finish(run, last)
        if backward:
            from openptv2.algorithms.track import trackback_c
            trackback_c(run)
    finally:
        os.chdir(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    cpar, cals = load_optics()
    scenarios = [
        ("S1 head-on", {"v": 0.5}),
        ("S14-hard p4", {"v": 0.5, "pulsatile": {"vmax": 1.5, "period": 4,
                                                 "modes": ("anti",)}}),
        ("S14b-hard p4", {"v": 0.5, "pulsatile": {"vmax": 1.5, "period": 4,
                                                  "modes": ("plug",)}}),
        ("S14c systole2.5", {"v": 0.5, "pulsatile": {"vmax": 2.5, "period": 10,
                                                     "modes": ("solo",)}}),
        ("S11 kick+noise", {"v": 0.5, "maneuver": "kick", "noise": 0.07}),
    ]
    channels = [("both", True, True), ("dv-only", True, False),
                ("dacc-only", False, True)]
    print(f"{'scenario':16}{'engine':14}{'recall':>9}{'cross-ok':>10}",
          flush=True)
    for sname, kw in scenarios:
        noisy = kw.get("noise", 0.0) > 0
        reps = args.reps if noisy else 3
        rows = [("fixed", None, None, None)]
        if "S14c" in sname or "S11" in sname:
            rows += [(f"adapt-{c}", dv, da, None) for c, dv, da in channels]
            rows += [("ppar", True, True, 2.5), ("ppar15", True, True, 1.5)]
        else:
            rows += [("adaptive", True, True, None), ("ppar", True, True, 2.5)]
        for ename, dv, da, hi in rows:
            recs, crs = [], []
            for rep in range(reps):
                work = REPO / "scratch" / "_synth_ph"
                setup_work(work)
                truth = make_truth(seed=1000 + rep, **kw)
                rows_pf = write_scene(work, truth, cpar, cals)
                if ename == "fixed":
                    from synth_crossing import run_tracker
                    run_tracker(work, 0, 0)
                elif ename.startswith("ppar"):
                    run_adaptive_pp(work, 0, 0, hi=hi, mode="vel")
                else:
                    run_adaptive(work, 0, 0, scale_dv=dv, scale_dacc=da)
                r, c = score(work, rows_pf)
                recs.append(r)
                crs.append(c)
            print(f"{sname:16}{ename:14}{np.mean(recs):>8.1%}  "
                  f"{sum(crs)}/{len(crs)}", flush=True)


if __name__ == "__main__":
    main()

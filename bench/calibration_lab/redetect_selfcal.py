# ruff: noqa
"""Real images without a stored blob table (test_cavity, lv_multi): detect, match, then can tracer self-calibration improve it?

    uv run python bench/calibration_lab/redetect_selfcal.py <scratch> [<root> <run dir in root> <yaml> <first> <last>]

Defaults: test_data/test_cavity (cameras on opposite sides, 5 frames). lv_multi: root ~/Documents/Github/openptv-cloud/examples/lv-multi,
run dir wp5 (or wp4), yaml parameters_wp5.yaml, frames 1 20. The root is copied without its results (never modified), all frames are
detected and matched with the production sequence loop (so the stored targets carry correct tnr), then: consistency report on the shipped calibration, self-calibration on the odd frames judged on the
even ones, a new matching with the new cameras, and tracking with the trackers of the study. Quality measure of the project memory:
trajectories of >= 4 frames whose quadratic fit leaves an rms below 0.2 mm ("good"), plus 3-frame-sensible counts.
"""
import copy, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ARGS = sys.argv[2:]
ANCHOR = os.environ.get("ANCHOR") == "1"       # anchor the fit with the calibration target body (known_body_from_calibration)
SRC = Path(ARGS[0]).expanduser() if ARGS else Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"
RUN = ARGS[1] if ARGS else "."
YAML = ARGS[2] if ARGS else "parameters_Run1.yaml"
FR = list(range(int(ARGS[3]), int(ARGS[4]) + 1)) if ARGS else list(range(10000, 10005))
work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)


def make(name, cal_from=None):
    """A copy of the root; the run dir inside it is returned. ``cal_from``: a dict {relative ori path: Calibration} to write."""
    root = work / name
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(SRC, root, ignore=shutil.ignore_patterns("res", "ensemble", "*.html", "phase_binned.zarr"), symlinks=True)
    d = root / RUN
    y = yaml.safe_load((d / YAML).read_text())
    y["sequence"]["first"], y["sequence"]["last"] = FR[0], FR[-1]
    (d / YAML).write_text(yaml.safe_dump(y, sort_keys=False))
    if cal_from:
        for rel, cal in zip(y["cal_ori"]["img_ori"], cal_from):
            ori = (d / rel).resolve()
            cal.write(str(ori), str(ori).replace(".ori", ".addpar"))
    return d


def sequence(d):
    from openptv2.gui import ptv
    from openptv2.gui.experiment import Experiment  # noqa
    cwd = os.getcwd(); os.chdir(d)
    try:
        e = Experiment(); e.pm.from_yaml(d / YAML); e.target_filenames = e.pm.get_target_filenames()
        cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(e.pm)
        exp = SimpleNamespace(pm=e.pm, num_cams=e.pm.num_cams, cpar=cpar, spar=spar, vpar=vpar, track_par=track_par, tpar=tpar, cals=cals, target_filenames=e.target_filenames)
        ptv.py_sequence_loop(exp)
    finally:
        os.chdir(cwd)


def good_tracks(d, tracker):
    from openptv2.merge_post import arrays_from_store
    from openptv2.plugins import run_tracking_plugin
    from openptv2.storage import RunStore
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import synth_bench as sb
    exp = sb._experiment(d, d / YAML, FR[0], FR[-1])
    cwd = os.getcwd(); os.chdir(d)
    try:
        run_tracking_plugin(tracker, exp)
    finally:
        os.chdir(cwd)
    a = arrays_from_store(RunStore(str(d / "res" / "run.zarr"), mode="r"), "ptv_is", FR[0], FR[-1], 4)
    tid, fr, pos = a["tid"], a["frame"], np.asarray(a["pos"], float)
    n_good = n_len = 0
    for t in np.unique(tid):
        m = tid == t
        if m.sum() < 4:
            continue
        n_len += 1
        o = np.argsort(fr[m]); x, p = fr[m][o].astype(float), pos[m][o]
        rms = np.sqrt(np.mean(np.sum((p - np.stack([np.polyval(np.polyfit(x - x.mean(), p[:, k], 2), x - x.mean()) for k in range(3)], 1)) ** 2, 1)))
        n_good += rms < 0.2
    return len(np.unique(tid)), n_len, n_good


if __name__ == "__main__":
    from openptv2.calibration_consistency import consistency_report, self_calibrate_from_store
    from openptv2.correspondences import match_correspondences_batch_parallel
    from openptv2.recorrespond import load_parameters
    from openptv2.storage import RunStore

    d0 = make("orig"); sequence(d0)
    cpar, vpar, cals, _ = load_parameters(d0 / YAML); cals = list(cals)
    st = RunStore(str(d0 / "res" / "run.zarr"), mode="r")
    pts = [len(st.read_correspondences(f)[0]) for f in FR]
    print("correspondences per frame", pts)
    print("\n" + consistency_report(st, FR, cals, cpar, vpar).text())
    kw = {}
    if ANCHOR:
        from openptv2.selfcal import known_body_from_calibration
        if YAML != "parameters_Run1.yaml" and not (d0 / "parameters_Run1.yaml").exists():
            os.symlink(YAML, d0 / "parameters_Run1.yaml")
        kw["body"] = known_body_from_calibration(d0, cals, cpar)
        print(f"anchor: {len(kw['body'].xyz)} calibration-target points")
    if os.environ.get('HOLD'):
        kw['hold_cam'] = os.environ['HOLD']
    res = self_calibrate_from_store(st, FR, cals, cpar, vpar, **kw)
    print("\n" + res.text())
    d1 = make("selfcal", cal_from=res.cals); shutil.copytree(d0 / "res", d1 / "res")
    cpar1, vpar1, cals1, _ = load_parameters(d1 / YAML)
    match_correspondences_batch_parallel(frames=FR, cpar=cpar1, cals=list(cals1), vpar=vpar1, zarr_store_path=str(d1 / "res" / "run.zarr"),
                                         n_workers=1, write_to_store=True)
    st1 = RunStore(str(d1 / "res" / "run.zarr"), mode="r")
    print("\nnew cameras, all frames, re-matched: quads per frame", [int(((np.asarray(st1.read_correspondences(f)[1])[:, :4] >= 0).all(1)).sum()) for f in FR],
          "(original:", [int(((np.asarray(st.read_correspondences(f)[1])[:, :4] >= 0).all(1)).sum()) for f in FR], ")")
    print(consistency_report(st1, FR, list(cals1), cpar1, vpar1).text())
    print("\ntrajectories (all, >=4 frames, good = quadratic-fit rms < 0.2 mm)")
    for trk in ("two_phase", "trackcorr", "fast_3d"):
        for lab, d in (("orig", d0), ("selfcal", d1)):
            root_d = d if RUN == "." else Path(str(d)[: -len(RUN) - 1])
            dd_root = work / f"{lab}_{trk}"
            if dd_root.exists(): shutil.rmtree(dd_root)
            shutil.copytree(root_d, dd_root, symlinks=True)
            dd = dd_root / RUN
            print(f"  {trk:10} {lab:8} all {{}} / >=4 frames {{}} / good {{}}".format(*good_tracks(dd, trk)), flush=True)

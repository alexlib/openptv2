"""The holistic loop on a REAL run (no truth): scratch copy, two_phase, then rounds of
refine -> births -> relink, then refine + recovery. Truth-free scores before and after.

    uv run python scripts/holistic_real.py LABEL LAST [--rounds 2] [--src RUN_FOLDER]

Default source: CompleteTest wp2 (never written to; the copy goes to /private/tmp/claude-501/real_runs/LABEL).
Scores: scripts/realism_metrics.py (tracks, lengths, jitter, jumps, never-4-camera share), leave-one-camera-out
consistency of raw vs corrected observations (openptv2.loco), births and their track lengths.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realism_metrics as rm  # noqa: E402
from synth_bench import _experiment  # noqa: E402

DEFAULT_SRC = Path.home() / "Downloads/HiDImaging/CompleteTest-e2e-local/wp2/test"


def chains_of(tid, frame, pos, ncam):
    o = np.lexsort((frame, tid))
    tid, frame, pos, ncam = tid[o], frame[o], pos[o], ncam[o]
    cuts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1], True])
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        br = np.flatnonzero(np.diff(frame[a:b]) != 1) + 1
        for seg_p, seg_k in zip(np.split(pos[a:b], br), np.split(ncam[a:b], br)):
            out.append((seg_p, seg_k))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("last", type=int)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--min-resid", type=float, default=0.25)
    a = ap.parse_args()
    from openptv2.loco import loco_residuals, project_pixels, summarize
    from openptv2.merge_post import (
        arrays_from_store,
        birth_points,
        birth_targets,
        extend_membership,
        fill_gaps,
        row_signatures,
        stamp_tnr,
    )
    from openptv2.merge_refine import refine_merged
    from openptv2.plugins import run_tracking_plugin
    from openptv2.reconnect import reconnect_pieces
    from openptv2.storage import RunStore

    src = Path(a.src)
    out = Path("/private/tmp/claude-501/real_runs") / a.label
    if out.exists():
        shutil.rmtree(out)
    (out / "res").mkdir(parents=True)
    shutil.copytree(src / "cal", out / "cal")
    shutil.copytree(src / "res" / "run.zarr", out / "res" / "run.zarr")
    (out / "img").mkdir()
    raw = yaml.safe_load(next(src.glob("parameters_*.yaml")).read_text())
    raw["sequence"]["last"] = a.last
    first = int(raw["sequence"].get("first", 1))
    (out / "parameters_Run1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    exp = _experiment(out, out / "parameters_Run1.yaml", first, a.last)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    frames = range(first, a.last + 1)
    store_path = out / "res" / "run.zarr"
    report: dict = {"label": a.label, "frames": [first, a.last]}

    def track():
        cwd = os.getcwd()
        os.chdir(out)
        try:
            t0 = time.perf_counter()
            run_tracking_plugin("two_phase", exp)
            return round(time.perf_counter() - t0, 1)
        finally:
            os.chdir(cwd)

    def score(tag, tid, frame, pos, blob):
        ch = chains_of(tid, frame, pos, (blob >= 0).sum(1))
        report[tag] = rm.metrics(ch, len(frames))
        print(tag, json.dumps(report[tag]))

    from openptv2.holistic import run_holistic

    st = RunStore(store_path, mode="a")
    report["track_time_s"] = [track()]
    base = arrays_from_store(st, "ptv_is", first, a.last, n_cams)
    score("base_two_phase", base["tid"], base["frame"], base["pos"], base["blob"])
    t0 = time.perf_counter()
    stats = run_holistic(st, list(frames), cpar, cals, vpar, lambda: report["track_time_s"].append(track()),
                         rounds=a.rounds, min_resid=a.min_resid, initial_link=False)
    report["loop_time_s"] = round(time.perf_counter() - t0, 1)
    report["births"] = stats["rounds"]
    final = arrays_from_store(st, "ptv_is", first, a.last, n_cams)
    score("final", final["tid"], final["frame"], final["pos"], final["blob"])
    # leave-one-camera-out on 4-camera points with a shared blob: raw blob centres vs the observations the loop used
    rows_obs = [st.read_observations(f) for f in frames]
    obs = np.vstack([o for o in rows_obs if o is not None])
    blob = final["blob"]
    four = (blob >= 0).sum(1) == n_cams
    if len(obs) == len(blob):
        shared = (obs[:, :, 3] >= 2).any(1) & four
        raw_xy = np.where((blob >= 0)[:, :, None], final["centre"], np.nan)
        cor_xy = obs[:, :, :2].astype(float)
        both = four & np.all(np.isfinite(cor_xy[:, :, 0]), axis=1)
        for name, m in (("all_4cam", both), ("with_shared_blob", both & shared)):
            lr = loco_residuals(raw_xy[m], cpar, cals, vpar)
            lc = loco_residuals(cor_xy[m], cpar, cals, vpar)
            report[f"loco_{name}"] = {"raw": summarize(lr), "corrected": summarize(lc)}
            print("LOCO", name, json.dumps(report[f"loco_{name}"]))
    fin = final["pos"]
    arr3 = final
    np.savez_compressed(out / "final_pred.npz", trajid=arr3["tid"], frame=arr3["frame"], pos=fin)
    (out / "holistic_report.json").write_text(json.dumps(report, indent=1))
    print("REPORT", out / "holistic_report.json")


if __name__ == "__main__":
    main()

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

    report["track_time_s"] = [track()]
    st = RunStore(store_path, mode="r")
    base = arrays_from_store(st, "ptv_is", first, a.last, n_cams)
    score("base_two_phase", base["tid"], base["frame"], base["pos"], base["blob"])
    report["births"] = []
    for rnd in range(1, a.rounds + 1):
        st = RunStore(store_path, mode="r")
        arr = arrays_from_store(st, "ptv_is", first, a.last, n_cams)
        blob, centre, sumg, n_att = extend_membership(arr["frame"], arr["pos"], arr["blob"], arr["centre"],
                                                      arr["sumg"], arr["blobs"], cpar, cals)
        res = refine_merged(arr["frame"], arr["tid"], blob, centre, sumg, cpar, cals, vpar)
        sig = row_signatures(arr["tid"], arr["frame"], blob, sumg)
        counts = {(f, c): np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"])[:, 3:6].astype(int)
                  for f in frames for c in range(n_cams)}
        fallback = np.full(res.pred_px.shape, np.nan)
        okx = np.isfinite(res.x[:, 0])
        for c in range(n_cams):
            m = okx & (blob[:, c] >= 0)
            fallback[m, c] = project_pixels(res.x[m], cals[c], cpar)
        targets = birth_targets(arr["frame"], blob, res.pred_px, sig, arr["blobs"], counts,
                                min_resid=a.min_resid, fallback_px=fallback)
        births = birth_points(targets, frames, cpar, cals, vpar)
        new_pos = np.where(np.isfinite(res.x[:, :1]), res.x, arr["pos"])
        stw = RunStore(store_path, mode="a")
        off = 0
        for f in frames:
            pos0, ids0 = stw.read_correspondences(f)
            n = len(pos0)
            assert (arr["frame"][off: off + n] == f).all(), "row alignment lost"
            seg = new_pos[off: off + n]
            off += n
            b = births["frame"] == f
            ids_b = np.full((int(b.sum()), np.asarray(ids0).shape[1]), -1, dtype=np.int64)
            ids_b[:, :n_cams] = births["blob"][b]
            stw.write_correspondences(f, np.vstack([seg, births["pos"][b]]),
                                      np.vstack([np.asarray(ids0, dtype=np.int64), ids_b]))
        stamp_tnr(stw, frames, n_cams)
        report["births"].append({"round": rnd, "attachments": int(n_att), "births": int(len(births["frame"])),
                                 "residual_targets": int(sum(int((t["kind"] == 1).sum()) for t in targets.values())),
                                 "refine": res.info[-1]})
        print("round", rnd, report["births"][-1])
        report["track_time_s"].append(track())
    # final: refine, rejoin pieces on the refined path, recover gaps (mass-checked), refine again
    st = RunStore(store_path, mode="r")
    arr = arrays_from_store(st, "ptv_is", first, a.last, n_cams)
    score("loop_two_phase_raw", arr["tid"], arr["frame"], arr["pos"], arr["blob"])
    blob, centre, sumg, _ = extend_membership(arr["frame"], arr["pos"], arr["blob"], arr["centre"], arr["sumg"],
                                              arr["blobs"], cpar, cals)
    res = refine_merged(arr["frame"], arr["tid"], blob, centre, sumg, cpar, cals, vpar)
    path = np.where(np.isfinite(res.x[:, :1]), res.x, arr["pos"])
    arr2 = {**arr, "tid": reconnect_pieces(arr["tid"], arr["frame"], path, max_gap=13, min_len=4)}
    new = fill_gaps(arr2, path, cpar, cals, radius=1.5, max_gap=12, interp="hermite", radius_growth=0.25,
                    mass_range=(0.7, 1.4))
    keys = ("frame", "tid", "pos", "blob", "centre", "sumg")
    arr3 = {**arr2, **{k: np.concatenate([arr2[k], new[k]]) for k in keys}}
    b3, c3, s3, _ = extend_membership(arr3["frame"], arr3["pos"], arr3["blob"], arr3["centre"], arr3["sumg"],
                                      arr3["blobs"], cpar, cals)
    res3 = refine_merged(arr3["frame"], arr3["tid"], b3, c3, s3, cpar, cals, vpar)
    fin = np.where(np.isfinite(res3.x[:, :1]), res3.x, arr3["pos"])
    score("final", arr3["tid"], arr3["frame"], fin, b3)
    report["gap_rows_added"] = int(len(new["frame"]))
    # leave-one-camera-out: raw blob centres vs the observations the refinement used, on 4-camera points
    four = (b3 >= 0).sum(1) == n_cams
    shared = (res3.n_share >= 2).any(1) & four
    raw_xy = np.where((b3 >= 0)[:, :, None], c3, np.nan)
    cor_xy = res3.obs_px
    both = four & np.all(np.isfinite(cor_xy[:, :, 0]), axis=1)
    for name, m in (("all_4cam", both), ("with_shared_blob", both & shared)):
        lr = loco_residuals(raw_xy[m], cpar, cals, vpar)
        lc = loco_residuals(cor_xy[m], cpar, cals, vpar)
        report[f"loco_{name}"] = {"raw": summarize(lr), "corrected": summarize(lc)}
        print("LOCO", name, json.dumps(report[f"loco_{name}"]))
    np.savez_compressed(out / "final_pred.npz", trajid=arr3["tid"], frame=arr3["frame"], pos=fin)
    (out / "holistic_report.json").write_text(json.dumps(report, indent=1))
    print("REPORT", out / "holistic_report.json")


if __name__ == "__main__":
    main()

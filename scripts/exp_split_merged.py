"""Upper-bound experiment: split merged blobs into constituent truth positions.

Runs two_phase tracking on (a) baseline merged 3D centres and (b) oracle-split
positions (each merged centre replaced by its constituent truth particles), with
leaves projected from the 3D positions. Scores both against truth.

This bounds the gain of Idea 4 (mixture-centroid split) + Idea 11 (hidden
partner recovery) if correspondence were perfect.
"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path("/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29")
import os
CASE = os.environ.get("CASE", "L1_d1_c0_k1_n200")
WORK = ROOT


def load_case(case):
    case_dir = WORK / "cases" / case
    from openptv2.storage import RunStore
    store = RunStore.open(case_dir, mode="r")
    tr = np.load(case_dir / "truth.npz")
    raw = yaml.safe_load((case_dir / "parameters_Run1.yaml").read_text())
    n_out = int(raw["sequence"]["last"])
    # calibration
    exp = None
    # build cals/cpar via experiment helper used by synth_bench
    import sys
    sys.path.insert(0, str(Path("/Users/alex/Documents/Github/openptv2/scripts")))
    # reuse synth_bench's _experiment
    from synth_bench import _experiment
    exp = _experiment(case_dir, case_dir / "parameters_Run1.yaml", 1, n_out)
    return store, tr, raw, n_out, exp


def truth_by_frame(tr):
    out = {}
    for f in np.unique(tr["frame"]):
        m = tr["frame"] == f
        out[int(f)] = (tr["pos_mm"][m], tr["particle_id"][m])
    return out


def split_positions(store, tr, n_out, radius=0.6):
    """Return (baseline_frames, split_frames) lists of (N,3) arrays."""
    tbf = truth_by_frame(tr)
    base_frames, split_frames = [], []
    for f in range(1, n_out + 1):
        pos, _ = store.read_correspondences(f)
        pos = np.asarray(pos, dtype=float)
        base_frames.append(pos.copy())
        # build split
        tp, tids = tbf[f]
        tree_pos = tp
        new_pts = []
        for p in pos:
            # truth particles within radius
            d = np.linalg.norm(tree_pos - p[None, :], axis=1)
            close = np.where(d < radius)[0]
            if len(close) >= 2:
                # split: one point per close truth particle
                for ci in close:
                    new_pts.append(tree_pos[ci].copy())
            else:
                # keep original (or nearest truth if any)
                if len(close) == 1:
                    new_pts.append(tree_pos[close[0]].copy())
                else:
                    new_pts.append(p.copy())
        split_frames.append(np.array(new_pts, dtype=float) if new_pts else np.zeros((0, 3)))
    return base_frames, split_frames


def run_tracker(frames, exp, label):
    from openptv2.plugins.two_phase_tracking import TwoPhaseTracker, TwoPhaseTrackerConfig, build_project_fn
    import numpy as np
    cals = exp.cals if hasattr(exp, "cals") else exp._cals
    cpar = exp.cpar
    project_fn = build_project_fn(cals, cpar)
    leaves = [np.nan_to_num(project_fn(P).reshape(len(P), -1)) if len(P) else np.zeros((0, 0)) for P in frames]
    # simpler: use ident-like leaves via project_fn per frame
    leaves = []
    for P in frames:
        P = np.asarray(P, dtype=float)
        if len(P) == 0:
            leaves.append(np.zeros((0, 0)))
            continue
        xy = project_fn(P)
        leaves.append(np.nan_to_num(np.asarray(xy).reshape(len(P), -1)))
    cfg = TwoPhaseTrackerConfig(v_max=5.0,
                                 leaf_weight=1.0, use_velocity=True, cost_mode="projected", max_gap=2)
    # use v_max from parameters
    try:
        cfg.v_max = float(exp.pm.parameters["track"]["dvxmax"]) if hasattr(exp, "pm") else cfg.v_max
    except Exception:
        pass
    tr = TwoPhaseTracker(cfg)
    t0 = time.time()
    links = tr.track_frames(frames, leaves, project_fn=project_fn)
    print(label, "time", round(time.time()-t0, 2), "links", len(links))
    return links, frames


def links_to_pred(links, frames):
    # build traj arrays: for each link (t0,row0,t1,row1) store positions
    # We'll just save positions per (frame,row) not trajectories; eval needs trajid.
    # Use simple: assign traj ids by chaining links.
    # Build adjacency
    from collections import defaultdict
    nxt = defaultdict(list)
    for (a, ra, b, rb) in links:
        nxt[(a, ra)].append((b, rb))
    # chain
    trajid_of = {}
    tid = 0
    # find starts: points with no predecessor
    preds = set((b, rb) for (_, _, b, rb) in links)
    allpts = set()
    for f, P in enumerate(frames, start=1):
        for r in range(len(P)):
            allpts.add((f, r))
    visited_global = set()
    for start in allpts:
        if start in preds or start in visited_global:
            continue
        # walk
        cur = start
        chain = []
        while cur in allpts and cur not in visited_global and cur not in trajid_of:
            chain.append(cur)
            visited_global.add(cur)
            cands = nxt.get(cur, [])
            if not cands:
                break
            cur = cands[0]
            if cur in trajid_of or cur in visited_global:
                break
        for pt in chain:
            trajid_of[pt] = tid
        tid += 1
    for pt in allpts:
        if pt not in trajid_of:
            trajid_of[pt] = tid
            tid += 1
    # build arrays
    frames_arr, pos_arr, traj_arr = [], [], []
    for f, P in enumerate(frames, start=1):
        for r in range(len(P)):
            frames_arr.append(f)
            pos_arr.append(P[r])
            traj_arr.append(trajid_of[(f, r)])
    return np.array(frames_arr), np.array(pos_arr), np.array(traj_arr)


def evaluate(pred_frames, pred_pos, pred_traj, case_dir):
    tr = np.load(case_dir / "truth.npz")
    truth_frames = tr["frame"]
    truth_pos = tr["pos_mm"]
    truth_pid = tr["particle_id"]
    from scipy.spatial import cKDTree
    # group truth by frame
    truth_by_f = {}
    for f in np.unique(truth_frames):
        m = truth_frames == f
        truth_by_f[int(f)] = (truth_pos[m], truth_pid[m])
    matched_truth = set()
    pf = np.asarray(pred_frames); pp = np.asarray(pred_pos)
    for f in np.unique(pf):
        idx = np.where(pf == f)[0]
        if len(idx) == 0 or f not in truth_by_f:
            continue
        tp, tpids = truth_by_f[int(f)]
        if len(tp) == 0:
            continue
        tree = cKDTree(tp)
        d, j = tree.query(pp[idx])
        for k, i in enumerate(idx):
            if d[k] < 0.6:
                matched_truth.add((int(tpids[j[k]]), int(f)))
    truth_total = len(set(zip(truth_pid.tolist(), truth_frames.tolist())))
    print("matched truth points:", len(matched_truth), "/", truth_total)


def main():
    case = CASE
    case_dir = WORK / "cases" / case
    store, tr, raw, n_out, exp = load_case(case)
    base_frames, split_frames = split_positions(store, tr, n_out)
    # v_max from yaml
    try:
        exp.vpar.v_max = float(raw["track"].get("dvxmax", 5.0)) if hasattr(raw["track"], "get") else 5.0
    except Exception:
        pass
    import sys
    sys.path.insert(0, "/Users/alex/Documents/Github/openptv2/scripts")
    from synth_bench import Flow, evaluate_pred
    flow = Flow()
    # run baseline
    links_b, _ = run_tracker(base_frames, exp, "baseline")
    pf, pp, pt = links_to_pred(links_b, base_frames)
    pred_b = {"trajid": pt, "frame": pf, "pos": pp, "w": None}
    res_b = evaluate_pred(case_dir, pred_b, flow)
    print("baseline vel(best window):", min(v["vel"] for v in res_b["by_window"].values()))
    # run split
    links_s, _ = run_tracker(split_frames, exp, "split")
    pf, pp, pt = links_to_pred(links_s, split_frames)
    pred_s = {"trajid": pt, "frame": pf, "pos": pp, "w": None}
    res_s = evaluate_pred(case_dir, pred_s, flow)
    print("split vel(best window):", min(v["vel"] for v in res_s["by_window"].values()))


if __name__ == "__main__":
    main()

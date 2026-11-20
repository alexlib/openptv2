import os, sys, time
import numpy as np
sys.path.insert(0, "/Users/alex/Documents/Github/openptv2/scripts")
sys.path.insert(0, "/Users/alex/.local/share/opencode/worktree/d28922/merged-blob-experiments/scripts")
CASE = os.environ.get("CASE", "L1_d1_c0_k1_n200")
WORK = __import__("pathlib").Path("/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29")
case_dir = WORK / "cases" / CASE
from exp_split_merged import load_case
from synth_bench import Flow, evaluate_pred
from openptv2.blob_recovery import recover
from openptv2.plugins.two_phase_tracking import build_project_fn
from openptv2.point_quality import read_frame_arrays

store, tr, raw, n_out, exp = load_case(CASE)
base_frames = [np.asarray(store.read_correspondences(f)[0], dtype=float) for f in range(1, n_out+1)]
frame_leaves = []
frame_brightness = []
for f in range(1, n_out+1):
    pos_3d, cam_ids, rows = read_frame_arrays(store, f, 4)
    n = len(pos_3d)
    xy = np.full((n, 4, 2), np.nan)
    bri = np.zeros(n)
    for c in range(4):
        if rows[c] is not None:
            valid = cam_ids[:, c] >= 0
            xy[valid, c] = rows[c][cam_ids[valid, c], 1:3]
            bri[valid] += rows[c][cam_ids[valid, c], 6]
    frame_leaves.append(np.nan_to_num(xy.reshape(n, -1)))
    frame_brightness.append(bri)
cals = exp.cals; cpar = exp.cpar
project_fn = build_project_fn(cals, cpar)

# without brightness
p1, l1, m1 = recover(range(1,n_out+1), base_frames, frame_leaves, project_fn, cals, cpar)
# with brightness
p2, l2, m2 = recover(range(1,n_out+1), base_frames, frame_leaves, project_fn, cals, cpar, frame_brightness=frame_brightness)

for label, frames, leaves, merged in [("no-bri", p1, l1, m1), ("bri", p2, l2, m2)]:
    from openptv2.plugins.two_phase_tracking import TwoPhaseTracker, TwoPhaseTrackerConfig
    cfg = TwoPhaseTrackerConfig(v_max=5.0, leaf_weight=1.0, use_velocity=True, cost_mode="projected", max_gap=2)
    trk = TwoPhaseTracker(cfg)
    links = trk.track_frames(frames, leaves, project_fn=project_fn, frame_merged=merged)
    from exp_split_merged import links_to_pred
    pf, pp, pt = links_to_pred(links, frames)
    pred = {"trajid": pt, "frame": pf, "pos": pp, "w": None}
    flow = Flow()
    res = evaluate_pred(case_dir, pred, flow)
    v = min(vv["vel"] for vv in res["by_window"].values())
    nsplit = sum(int(m.sum()) for m in merged)
    print(f"{label}: splits={nsplit} vel={v:.4f}")
print("baseline: see synth_bench")

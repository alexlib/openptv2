"""Predictor-based membership two-pass experiment (Idea 11 + 4).

Pass 1: run two_phase on baseline merged 3D centres -> tracks.
Pass 2: identify merged blobs as points claimed by >=2 tracks; split each via
        merged_split.split_merged_blob using the tracks' positions as predictions;
        re-run two_phase on split points; evaluate vErr.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, "/Users/alex/Documents/Github/openptv2/scripts")
sys.path.insert(0, "/Users/alex/.local/share/opencode/worktree/d28922/merged-blob-experiments/scripts")

CASE = os.environ.get("CASE", "L1_d1_c0_k1_n200")
WORK = __import__("pathlib").Path("/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29")
case_dir = WORK / "cases" / CASE

from exp_split_merged import load_case, run_tracker, links_to_pred
from synth_bench import Flow, evaluate_pred
from openptv2.plugins.two_phase_tracking import build_project_fn, TwoPhaseTracker, TwoPhaseTrackerConfig
from openptv2 import merged_split

store, tr, raw, n_out, exp = load_case(CASE)
# baseline frames
base_frames = []
for f in range(1, n_out + 1):
    pos, _ = store.read_correspondences(f)
    base_frames.append(np.asarray(pos, dtype=float))

# leaves from store
frame_leaves = []
for f in range(1, n_out + 1):
    from openptv2.point_quality import read_frame_arrays
    pos_3d, cam_ids, rows = read_frame_arrays(store, f, 4)
    n = len(pos_3d)
    xy = np.full((n, 4, 2), np.nan)
    for c in range(4):
        if rows[c] is not None:
            valid = cam_ids[:, c] >= 0
            xy[valid, c] = rows[c][cam_ids[valid, c], 1:3]
    frame_leaves.append(np.nan_to_num(xy.reshape(n, -1)))

cals = exp.cals
cpar = exp.cpar
project_fn = build_project_fn(cals, cpar)

# Pass 1
links1, _ = run_tracker(base_frames, exp, "pass1")
pf1, pp1, pt1 = links_to_pred(links1, base_frames)
# build track positions: map (frame,row)->trajid
traj_of = {}
for f, p, t in zip(pf1, pp1, pt1):
    traj_of.setdefault(int(t), {}).setdefault(int(f), p)

# group tracks by frame for nearest lookup
from collections import defaultdict
tracks_by_frame = defaultdict(list)
for tid, fmap in traj_of.items():
    for f, p in fmap.items():
        tracks_by_frame[f].append((tid, p))

# Pass 2: build split frames
split_frames = []
n_split_attempted = 0
n_split_accepted = 0
# per-point brightness from store
frame_brightness = []
for f in range(1, n_out + 1):
    pos_3d, cam_ids, rows = __import__("openptv2.point_quality", fromlist=["read_frame_arrays"]).read_frame_arrays(store, f, 4)
    n = len(pos_3d)
    bri = np.zeros(n)
    for i in range(n):
        for c in range(4):
            if rows[c] is not None and cam_ids[i, c] >= 0:
                bri[i] += rows[c][cam_ids[i, c], 6]
    frame_brightness.append(bri)
# median single-particle brightness: points claimed by exactly 1 track
single_bris = []
for f in range(1, n_out + 1):
    tracks_here = tracks_by_frame.get(f-1, [])
    # map point->num tracks via KDTree already; approximate: all points
    pass
# compute median brightness of all points as baseline (single particles dominate)
all_bri = np.concatenate(frame_brightness)
med_bri = np.median(all_bri[all_bri > 0])
bri_threshold = 1.5 * med_bri
print("median brightness:", med_bri, "threshold:", bri_threshold)

for f in range(1, n_out + 1):
    pos = base_frames[f - 1].copy()
    # leaves for this frame's original points
    lf = frame_leaves[f - 1]
    new_pts = []
    new_rows_map = []  # not used
    tracks_here = tracks_by_frame.get(f, [])
    from scipy.spatial import cKDTree
    if tracks_here:
        tpos = np.array([tp for (_, tp) in tracks_here])
        tree = cKDTree(tpos)
        tids_list = [tid for (tid, _) in tracks_here]
    else:
        tree = None
    for i, p in enumerate(pos):
        close_tracks = []
        if tree is not None:
            idxs = tree.query_ball_point(p, 0.6)
            close_tracks = [(tids_list[j], tpos[j]) for j in idxs]
        if len(close_tracks) >= 2:
            # merged: split into constituents using their positions as predictions
            preds = np.array([tp for (_, tp) in close_tracks])
            intensities = np.ones(len(close_tracks))
            # merged px: project p
            mp_all = project_fn(p.reshape(1, 3)).reshape(-1)  # (n_cams*2,)
            n_cams = len(cals)
            merged_px = mp_all.reshape(n_cams, 2)
            seen = np.ones(n_cams, dtype=bool)
            split_pos, _, init_r, _ = merged_split.split_merged_blob(merged_px, seen, intensities, preds, cals, cpar)
            n_split_attempted += 1
            if init_r > 2.0:
                split_pos = None
            else:
                n_split_accepted += 1
            if split_pos is not None:
                for sp in split_pos:
                    new_pts.append(sp)
            else:
                new_pts.append(p)
        else:
            new_pts.append(p)
    split_frames.append(np.array(new_pts, dtype=float) if new_pts else np.zeros((0, 3)))

# leaves for split frames via project_fn
split_leaves = []
for P in split_frames:
    if len(P) == 0:
        split_leaves.append(np.zeros((0, 0)))
        continue
    xy = project_fn(P)
    split_leaves.append(np.nan_to_num(np.asarray(xy).reshape(len(P), -1)))

# run tracker on split arrays directly
cfg = TwoPhaseTrackerConfig(v_max=float(raw["track"].get("dvxmax", 5.0)), leaf_weight=1.0, use_velocity=True, cost_mode="projected", max_gap=2)
tracker = TwoPhaseTracker(cfg)
t0 = time.time()
links2 = tracker.track_frames(split_frames, split_leaves, project_fn=project_fn)
print("pass2 time", round(time.time()-t0, 2), "links", len(links2))
pf2, pp2, pt2 = links_to_pred(links2, split_frames)
pred2 = {"trajid": pt2, "frame": pf2, "pos": pp2, "w": None}
flow = Flow()
res2 = evaluate_pred(case_dir, pred2, flow)
print("split attempted/accepted:", n_split_attempted, n_split_accepted)
print("predictor-split vel(best):", min(v["vel"] for v in res2["by_window"].values()))

# baseline vErr
pred1 = {"trajid": pt1, "frame": pf1, "pos": pp1, "w": None}
res1 = evaluate_pred(case_dir, pred1, flow)
print("baseline vel(best):", min(v["vel"] for v in res1["by_window"].values()))

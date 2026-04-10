"""Diagnostic: Run correspondences on one frame of test_cavity and show breakdown."""
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Run from repo root
os.chdir(Path(__file__).parent / "test_data" / "test_cavity")

from algorithms.batch import (
    _build_control_par, _build_sequence_par, _build_volume_par,
    _build_target_par, _read_calibrations_py,
)
from algorithms.correspondences import (
    correspondences, match_pairs, four_camera_matching, three_camera_matching,
    consistent_pair_matching, take_best_candidates, safely_allocate_target_usage_marks,
    safely_allocate_adjacency_lists, n_tupel_dtype, MatchedCoords,
)
from algorithms.constants import NMAX
from algorithms.tracking_frame_buf import Frame, read_targets
from algorithms.parameters import ControlPar

YAML_FILE = Path("parameters_Run1.yaml")
FRAME = 10001

with open(YAML_FILE) as f:
    params = yaml.safe_load(f)

num_cams = params["num_cams"]
cpar = _build_control_par(params["ptv"], num_cams)
spar = _build_sequence_par(params["sequence"], num_cams)
vpar = _build_volume_par(params["criteria"])
tpar_detect = _build_target_par(params["targ_rec"], num_cams)
cals = _read_calibrations_py(params["cal_ori"], num_cams)

# Load pre-existing targets
from algorithms.batch import _target_file_bases
short_file_bases = _target_file_bases(spar.img_base_name, num_cams)

detections = []
corrected = []
for i_cam in range(num_cams):
    targs = read_targets(short_file_bases[i_cam], FRAME)
    if isinstance(targs, list):
        targs.sort(key=lambda t: t.y)
    detections.append(targs)
    mc = MatchedCoords(targs, cpar, cals[i_cam])
    corrected.append(mc)

print(f"Targets per cam: {[len(d) for d in detections]}")

# Build Frame
frm = Frame(num_cams=num_cams)
for i_cam in range(num_cams):
    n = len(detections[i_cam])
    frm.num_targets[i_cam] = n
    for tnum in range(n):
        t = detections[i_cam][tnum]
        frm.targets[i_cam][tnum].pnr = getattr(t, "pnr", tnum)
        frm.targets[i_cam][tnum].tnr = -1
        frm.targets[i_cam][tnum].x = getattr(t, "x", 0)
        frm.targets[i_cam][tnum].y = getattr(t, "y", 0)
        frm.targets[i_cam][tnum].n = getattr(t, "n", 0)
        frm.targets[i_cam][tnum].nx = getattr(t, "nx", 0)
        frm.targets[i_cam][tnum].ny = getattr(t, "ny", 0)
        frm.targets[i_cam][tnum].sumg = getattr(t, "sumg", 0)

print(f"NMAX={NMAX}, all_cam_flag={cpar.all_cam_flag}")
print(f"corrmin={vpar.corrmin}, cn={vpar.cn}")

# Run correspondences step by step
nmax = NMAX

con0 = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
con0.p = 0
con0.corr = 0.0

con = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
con.p = 0
con.corr = 0.0

tim = safely_allocate_target_usage_marks(num_cams, nmax)
corr_list = safely_allocate_adjacency_lists(num_cams, frm.num_targets)

print("\nRunning match_pairs...", flush=True)
match_pairs(corr_list, corrected, frm, vpar, cpar, cals)

# Count non-empty adjacency entries
for i1 in range(num_cams):
    for i2 in range(i1+1, num_cams):
        total = sum(1 for k in range(frm.num_targets[i1]) if corr_list[i1][i2][k].n > 0)
        print(f"  cam{i1+1}->cam{i2+1}: {total} targets with candidates")

# Quadruplets
match_counts = [0, 0, 0, 0]
matched_4 = four_camera_matching(corr_list, frm.num_targets[0], vpar.corrmin, con0, 4*nmax)
print(f"\nQuadruplet candidates: {matched_4}")

match_counts[0] = take_best_candidates(con0[:matched_4], con, num_cams, tim)
match_counts[3] += match_counts[0]
print(f"Quadruplets taken (after dedup): {match_counts[0]}")

# Triplets
con0.p = 0
con0.corr = 0.0
matched_3 = three_camera_matching(corr_list, num_cams, frm.num_targets, vpar.corrmin, con0, 4*nmax, tim)
print(f"\nTriplet candidates: {matched_3}")

match_counts[1] = take_best_candidates(con0[:matched_3], con[match_counts[3]:].view(np.recarray), num_cams, tim)
match_counts[3] += match_counts[1]
print(f"Triplets taken (after dedup): {match_counts[1]}")

# Pairs
con0.p = 0
con0.corr = 0.0
matched_2 = consistent_pair_matching(corr_list, num_cams, frm.num_targets, vpar.corrmin, con0, 4*nmax, tim)
print(f"\nPair candidates: {matched_2}")

match_counts[2] = take_best_candidates(con0[:matched_2], con[match_counts[3]:].view(np.recarray), num_cams, tim)
match_counts[3] += match_counts[2]
print(f"Pairs taken (after dedup): {match_counts[2]}")

print(f"\n=== Total: {match_counts[3]} correspondences ===")
print(f"  Quadruplets: {match_counts[0]}")
print(f"  Triplets: {match_counts[1]}")
print(f"  Pairs: {match_counts[2]}")
print(f"  Reference: 672")

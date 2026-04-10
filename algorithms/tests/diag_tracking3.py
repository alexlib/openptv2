"""Diagnostic: trace candsearch_in_pix and sort_candidates for first particle."""
import os
import shutil
from pathlib import Path
import tempfile
import numpy as np
import yaml

os.chdir("/home/user/Documents/GitHub/openptv2")

from algorithms.batch import (
    _build_control_par,
    _build_sequence_par,
    _build_track_par,
    _build_volume_par,
    _read_calibrations_py,
)
from algorithms.track import (
    default_naming, point_to_pixel, searchquader,
    candsearch_in_pix, register_closest_neighbs,
    sort_candidates_by_freq,
    Foundpix_dtype, reset_foundpix_array,
)
from algorithms.tracking_run import TrackingRun
from algorithms.tracking_frame_buf import FrameBuf
from algorithms.parameters import convert_track_par_to_tuple
from algorithms.constants import TR_BUFSPACE, MAX_TARGETS, MAX_CANDS, TR_MAX_CAMS, TR_UNUSED, CORRES_NONE

TEST_DATA_DIR = Path("test_data/test_cavity")
YAML_FILE = TEST_DATA_DIR / "parameters_Run1.yaml"

with open(YAML_FILE) as f:
    params = yaml.safe_load(f)

num_cams = params["num_cams"]
cpar = _build_control_par(params["ptv"], num_cams)
spar = _build_sequence_par(params["sequence"], num_cams)
vpar = _build_volume_par(params["criteria"])
tpar = _build_track_par(params["track"])

work_dir = Path(tempfile.mkdtemp())
for name in ("img", "img_orig"):
    src = TEST_DATA_DIR / name
    if src.exists():
        link = work_dir / name
        if not link.exists():
            link.symlink_to(src.resolve())
for item in TEST_DATA_DIR.iterdir():
    if item.is_dir() and item.name not in ("img", "img_orig", "res", "res_orig", "__pycache__"):
        shutil.copytree(item, work_dir / item.name, dirs_exist_ok=True)
    elif item.is_file():
        shutil.copy2(item, work_dir / item.name)
res = work_dir / "res"
res.mkdir(exist_ok=True)
for f in (TEST_DATA_DIR / "res_orig").iterdir():
    shutil.copy2(f, res / f.name)

os.chdir(work_dir)
cals = _read_calibrations_py(params["cal_ori"], num_cams)
spar.first = 10001
spar.last = 10004

run = TrackingRun(spar, tpar, vpar, cpar, TR_BUFSPACE, MAX_TARGETS,
                  "res/rt_is", "res/ptv_is", "res/added", cals, 0.0001)

from algorithms.track import track_forward_start
track_forward_start(run)

fb = run.fb
tpar_tuple = convert_track_par_to_tuple(run.tpar)

# Trace first particle
h = 0
curr_path_inf = fb.buf[1].path_info[h]
curr_corres = fb.buf[1].correspond[h]
curr_targets = fb.buf[1].targets
X1 = curr_path_inf.x.copy()
X2 = X1.copy()

v1 = np.zeros((num_cams, 2))
for j in range(num_cams):
    if curr_corres.p[j] == CORRES_NONE or curr_corres.p[j] >= len(curr_targets[j]):
        v1[j] = point_to_pixel(X2, cals[j], cpar)
    else:
        _ix = curr_corres.p[j]
        v1[j] = np.r_[curr_targets[j][_ix].x, curr_targets[j][_ix].y]

right, left, down, up = searchquader(X2, tpar_tuple, cpar, cals)

print(f"Particle 0: X1={X1}")
print(f"v1={v1}")

# Initialize points array like sorted_candidates_in_volume does
frm = fb.buf[2]  # next frame
points = np.array(
    [(TR_UNUSED, 0, [0] * TR_MAX_CAMS)] * (frm.num_cams * MAX_CANDS),
    dtype=Foundpix_dtype,
).view(np.recarray)
reset_foundpix_array(points, frm.num_cams * MAX_CANDS, frm.num_cams)

# Call candsearch_in_pix for each camera
for cam in range(frm.num_cams):
    all_cands = candsearch_in_pix(
        frm.targets[cam], frm.num_targets[cam], 
        v1[cam][0], v1[cam][1],
        left[cam], right[cam], up[cam], down[cam],
        cpar,
    )
    print(f"\nCam {cam}: candsearch_in_pix returned: {all_cands}")
    for idx, c in enumerate(all_cands):
        if c >= 0 and c < frm.num_targets[cam]:
            t = frm.targets[cam][c]
            print(f"  cand[{idx}]: target_idx={c}, tnr={t.tnr}, x={t.x:.2f}, y={t.y:.2f}")

    # Register like register_closest_neighbs does
    register_closest_neighbs(
        frm.targets[cam], frm.num_targets[cam], cam,
        v1[cam][0], v1[cam][1],
        left[cam], right[cam], up[cam], down[cam],
        points[cam * MAX_CANDS:], cpar,
    )

print(f"\n--- After register_closest_neighbs ---")
for i in range(frm.num_cams * MAX_CANDS):
    if points[i].ftnr != TR_UNUSED:
        print(f"  points[{i}]: ftnr={points[i].ftnr}, freq={points[i].freq}, whichcam={list(points[i].whichcam)}")

num_cands = sort_candidates_by_freq(points, frm.num_cams)
print(f"\n--- After sort_candidates_by_freq ---")
print(f"num_cands: {num_cands}")
for i in range(min(10, frm.num_cams * MAX_CANDS)):
    if points[i].ftnr != TR_UNUSED:
        print(f"  points[{i}]: ftnr={points[i].ftnr}, freq={points[i].freq}")

# After finding candidates, trace into the tracking loop
if num_cands > 0:
    print(f"\n--- Following first candidate ---")
    mm = 0
    while mm < len(points) and points[mm].ftnr != TR_UNUSED:
        ref_idx = points[mm].ftnr
        print(f"  mm={mm}: ftnr={ref_idx}")
        if ref_idx >= 0 and ref_idx < len(frm.path_info):
            ref_pi = frm.path_info[ref_idx]
            print(f"    path_info[{ref_idx}].x = {ref_pi.x}")
        else:
            print(f"    ERROR: ftnr={ref_idx} out of range, num_parts={frm.num_parts}")
        mm += 1
        if mm > 5:
            break
else:
    print("\nNO CANDIDATES FOUND!")

os.chdir("/home/user/Documents/GitHub/openptv2")
shutil.rmtree(work_dir)

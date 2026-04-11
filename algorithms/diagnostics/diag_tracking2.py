"""Diagnostic: trace why trackcorr_c_loop finds no candidates on cavity data."""
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
    register_closest_neighbs, sort_candidates_by_freq,
    Foundpix_dtype, reset_foundpix_array, TR_UNUSED,
)
from algorithms.tracking_run import TrackingRun
from algorithms.tracking_frame_buf import FrameBuf
from algorithms.parameters import convert_track_par_to_tuple
from algorithms.constants import TR_BUFSPACE, MAX_TARGETS, MAX_CANDS, TR_MAX_CAMS

TEST_DATA_DIR = Path("test_data/test_cavity")
YAML_FILE = TEST_DATA_DIR / "parameters_Run1.yaml"

with open(YAML_FILE) as f:
    params = yaml.safe_load(f)

num_cams = params["num_cams"]
cpar = _build_control_par(params["ptv"], num_cams)
spar = _build_sequence_par(params["sequence"], num_cams)
vpar = _build_volume_par(params["criteria"])
tpar = _build_track_par(params["track"])

# Set up work dir
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

run = TrackingRun(
    spar, tpar, vpar, cpar,
    TR_BUFSPACE, MAX_TARGETS,
    "res/rt_is", "res/ptv_is", "res/added",
    cals, 0.0001,
)

# Prime the buffer like track_forward_start
from algorithms.track import track_forward_start
track_forward_start(run)

fb = run.fb
tpar_tuple = convert_track_par_to_tuple(run.tpar)

print(f"Buffer positions:")
for i in range(4):
    print(f"  buf[{i}]: num_parts={fb.buf[i].num_parts}, "
          f"num_targets={[fb.buf[i].num_targets[j] for j in range(num_cams)]}")
    if fb.buf[i].num_parts > 0:
        pi = fb.buf[i].path_info[0]
        print(f"    first particle pos: {pi.x}")
        print(f"    first particle prev_frame: {pi.prev_frame}, next_frame: {pi.next_frame}")

print(f"\nbuf[1] is 'current', buf[2] is 'next frame'")
print(f"current frame particles: {fb.buf[1].num_parts}")
print(f"next frame particles: {fb.buf[2].num_parts}")

# Trace first particle
h = 0
curr_path_inf = fb.buf[1].path_info[h]
curr_corres = fb.buf[1].correspond[h]
curr_targets = fb.buf[1].targets

X1 = curr_path_inf.x.copy()
print(f"\nParticle 0 position (X[1]): {X1}")
print(f"Particle 0 prev_frame: {curr_path_inf.prev_frame}")
print(f"Particle 0 correspond: p={[curr_corres.p[j] for j in range(num_cams)]}")

# No previous frame, so X[2] = X[1]
X2 = X1.copy()

# Compute v1 (projected positions)
v1 = np.zeros((num_cams, 2))
from algorithms.constants import CORRES_NONE
for j in range(num_cams):
    if curr_corres.p[j] == CORRES_NONE or curr_corres.p[j] >= len(curr_targets[j]):
        v1[j] = point_to_pixel(X2, cals[j], cpar)
    else:
        _ix = curr_corres.p[j]
        v1[j] = np.r_[curr_targets[j][_ix].x, curr_targets[j][_ix].y]

print(f"Projected positions v1: {v1}")

# Search limits
right, left, down, up = searchquader(X2, tpar_tuple, cpar, cals)
print(f"Search quader:")
for j in range(num_cams):
    print(f"  cam{j}: left={left[j]:.1f}, right={right[j]:.1f}, up={up[j]:.1f}, down={down[j]:.1f}")

# Now try to find candidates in buf[2] (next frame)
frm = fb.buf[2]
print(f"\nNext frame (buf[2]): num_parts={frm.num_parts}")
print(f"Next frame targets: {[frm.num_targets[j] for j in range(num_cams)]}")

# Do the candidate search manually for cam 0
for cam in range(num_cams):
    print(f"\n  --- Camera {cam} ---")
    print(f"  center_proj: ({v1[cam][0]:.2f}, {v1[cam][1]:.2f})")
    print(f"  search rect: left={left[cam]:.1f}, right={right[cam]:.1f}, up={up[cam]:.1f}, down={down[cam]:.1f}")
    print(f"  num targets in next frame: {frm.num_targets[cam]}")
    
    if frm.num_targets[cam] > 0:
        # Show a few targets from the next frame
        for t_idx in range(min(5, frm.num_targets[cam])):
            tgt = frm.targets[cam][t_idx]
            print(f"    target[{t_idx}]: x={tgt.x:.2f}, y={tgt.y:.2f}, pnr={tgt.pnr}, tnr={tgt.tnr}")
        
        # Check if any target is within the search rect
        count_in_rect = 0
        for t_idx in range(frm.num_targets[cam]):
            tgt = frm.targets[cam][t_idx]
            if (v1[cam][0] - left[cam] <= tgt.x <= v1[cam][0] + right[cam] and
                v1[cam][1] - up[cam] <= tgt.y <= v1[cam][1] + down[cam]):
                count_in_rect += 1
                if count_in_rect <= 3:
                    print(f"    IN RECT: target[{t_idx}]: x={tgt.x:.2f}, y={tgt.y:.2f}")
        print(f"  Targets in search rect: {count_in_rect}")

# Cleanup
os.chdir("/home/user/Documents/GitHub/openptv2")
shutil.rmtree(work_dir)

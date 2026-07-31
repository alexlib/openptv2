#!/usr/bin/env python3
"""Debug why 2D tracking (trackcorr_c_loop) finds 0 links with generated data."""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.chdir("test_data/track")

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import read_control_par
from openptv2.algorithms.track import (
    track_forward_start,
    trackcorr_c_loop,
)
from openptv2.algorithms.tracking_run import tr_new


def read_all_calibration(num_cams, base_path="."):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


# Setup
if os.path.exists("res"):
    shutil.rmtree("res")
if os.path.exists("img"):
    shutil.rmtree("img")
shutil.copytree("res_orig", "res")
shutil.copytree("img_orig", "img")

cpar = read_control_par("parameters/ptv.par")
calib = read_all_calibration(cpar.num_cams, base_path=".")

run = tr_new(
    "parameters/sequence.par",
    "parameters/track.par",
    "parameters/criteria.par",
    "parameters/ptv.par",
    4,
    20000,
    "res/rt_is",
    "res/ptv_is",
    "res/added",
    calib,
    0.0001,
)
run.tpar = run.tpar._replace(add=0)

# Check what's in the buffer after track_forward_start
track_forward_start(run)

fb = run.fb
print("\nBuffer after track_forward_start:")
for i in range(4):
    f = fb.buf[i]
    print(f"  buf[{i}]: num_parts={f.num_parts}, num_targets={list(f.num_targets[:2])}")

# For each particle in buf[1], check pixel projection
print("\nParticle details in buf[1]:")
for h in range(fb.buf[1].num_parts):
    print(f"  Particle {h}:")
    print(f"    path_x = {fb.buf[1].path_x[h]}")
    print(f"    path_prev = {fb.buf[1].path_prev[h]}")
    print(f"    corres_p = {fb.buf[1].corres_p[h]}")
    for j in range(2):
        ix = fb.buf[1].corres_p[h, j]
        if ix >= 0:
            print(
                f"    cam{j}: target index {ix} at ({fb.buf[1].targ_x[j][ix]:.1f}, {fb.buf[1].targ_y[j][ix]:.1f})"
            )

# Check what targets exist in buf[2] (next frame)
print("\nTargets in buf[2]:")
for j in range(2):
    nt = fb.buf[2].num_targets[j]
    print(f"  cam{j}: {nt} targets")
    for t in range(min(nt, 5)):
        print(
            f"    target[{t}]: x={fb.buf[2].targ_x[j][t]:.1f}, y={fb.buf[2].targ_y[j][t]:.1f}, tnr={fb.buf[2].targ_tnr[j][t]}"
        )

# Now try a single tracking step
trackcorr_c_loop(run, run.seq_par.first)

print(f"\nAfter trackcorr_c_loop for frame {run.seq_par.first}:")
print(f"  npart = {run.npart}")
print(f"  nlinks = {run.nlinks}")

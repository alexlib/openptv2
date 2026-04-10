"""Diagnostic: Run Cython correspondences on same frame for comparison."""
import os
import sys
import yaml
import numpy as np
from pathlib import Path

os.chdir(Path(__file__).parent / "test_data" / "test_cavity")

try:
    from optv.correspondences import correspondences as cy_correspondences
    from optv.correspondences import MatchedCoords as cy_MatchedCoords
    from optv.calibration import Calibration as cy_Calibration
    from optv.parameters import (
        ControlParams as cy_ControlParams,
        VolumeParams as cy_VolumeParams,
    )
    from optv.tracking_framebuf import read_targets as cy_read_targets
except ImportError:
    print("Cython optv not available, skipping")
    sys.exit(0)

YAML_FILE = Path("parameters_Run1.yaml")
FRAME = 10001

with open(YAML_FILE) as f:
    params = yaml.safe_load(f)

num_cams = params["num_cams"]
ptv = params["ptv"]
seq = params["sequence"]
crit = params["criteria"]

# Build Cython control params
# Read them from files like C does
cpar = cy_ControlParams(num_cams)
cpar.set_image_size((ptv["imx"], ptv["imy"]))
cpar.set_pixel_size((ptv["pix_x"], ptv["pix_y"]))
cpar.set_mixed_corresp_flag(0)
cpar.set_allCam_flag(int(ptv.get("allcam_flag", False)))

# multimedia
mm = cpar.get_multimedia_params()

# Read cal files
cal_ori = params["cal_ori"]
cals = []
for i in range(num_cams):
    cal = cy_Calibration()
    ori_file = cal_ori[f"cam_{i+1}"]["ori_file"]
    addpar_file = cal_ori[f"cam_{i+1}"]["addpar_file"]
    cal.from_file(ori_file.encode(), addpar_file.encode())
    cals.append(cal)

vpar = cy_VolumeParams()
# Use file-based approach
# Let's just read the image base names and load targets
bases = seq["base_name"]

# Load targets
from optv.tracking_framebuf import Target, TargetArray

detections = []
corrected = []
for i_cam in range(num_cams):
    base = bases[i_cam]
    # cy_read_targets needs file base 
    # Read the target file manually
    targs = cy_read_targets(base.encode(), FRAME)
    detections.append(targs)
    mc = cy_MatchedCoords(targs, cpar, cals[i_cam])
    corrected.append(mc)

tgt_counts = [len(d) for d in detections]
print(f"Cython targets per cam: {tgt_counts}")

# Run correspondences
sets, _, num_targs = cy_correspondences(
    detections, corrected, cals, vpar, cpar
)

print(f"Cython results:")
for i, s in enumerate(sets):
    print(f"  Clique size {4-i}: {len(s)} matches")
total = sum(len(s) for s in sets)
print(f"  Total: {total}")

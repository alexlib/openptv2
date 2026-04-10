"""Diagnostic: compare trackcorr_c_loop vs track3d_loop on cavity data."""
import os
import shutil
from pathlib import Path

import yaml

os.chdir("/home/user/Documents/GitHub/openptv2")

from algorithms.batch import (
    _build_control_par,
    _build_sequence_par,
    _build_track_par,
    _build_volume_par,
    _read_calibrations_py,
)
from algorithms.track import Tracker, default_naming

TEST_DATA_DIR = Path("test_data/test_cavity")
YAML_FILE = TEST_DATA_DIR / "parameters_Run1.yaml"

with open(YAML_FILE) as f:
    params = yaml.safe_load(f)

num_cams = params["num_cams"]
cpar = _build_control_par(params["ptv"], num_cams)
spar = _build_sequence_par(params["sequence"], num_cams)
vpar = _build_volume_par(params["criteria"])
tpar = _build_track_par(params["track"])

# We need to chdir to the test data directory for file reading
import tempfile
work_dir = Path(tempfile.mkdtemp())

# Copy test env
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
src_res = TEST_DATA_DIR / "res_orig"
for f in src_res.iterdir():
    shutil.copy2(f, res / f.name)

os.chdir(work_dir)
cals = _read_calibrations_py(params["cal_ori"], num_cams)

print("=== Test 1: trackcorr_c_loop (full_forward) ===")
spar1 = _build_sequence_par(params["sequence"], num_cams)
spar1.first = 10001
spar1.last = 10004
tracker1 = Tracker(cpar, vpar, tpar, spar1, cals, default_naming)
tracker1.full_forward()
print(f"npart={tracker1.run_info.npart}, nlinks={tracker1.run_info.nlinks}")

# Reset res
for f in res.iterdir():
    f.unlink()
for f in src_res.iterdir():
    shutil.copy2(f, res / f.name)

print("\n=== Test 2: track3d_loop (full_forward_3d) ===")
spar2 = _build_sequence_par(params["sequence"], num_cams)
spar2.first = 10001
spar2.last = 10004
tracker2 = Tracker(cpar, vpar, tpar, spar2, cals, default_naming)
tracker2.full_forward_3d()
print(f"npart={tracker2.run_info.npart}, nlinks={tracker2.run_info.nlinks}")

# Cleanup
shutil.rmtree(work_dir)

import shutil
from pathlib import Path

import yaml
from algorithms.track import Tracker

CAVITY_DATA = Path("test_data/test_cavity")
YAML_FILE = CAVITY_DATA / "parameters_Run1.yaml"

with open(YAML_FILE) as f:
    params = yaml.safe_load(f)

num_cams = params["num_cams"]

# Setup work dir
import tempfile

work_dir = Path(tempfile.mkdtemp())
print(f"Work dir: {work_dir}")
res = work_dir / "res"
res.mkdir()

# Copy original correspondences to res/
res_orig = CAVITY_DATA / "res_orig"
for f in res_orig.iterdir():
    if f.name.startswith("rt_is."):
        shutil.copy2(f, res / f.name)

# Symlink img/
(work_dir / "img").symlink_to(CAVITY_DATA.resolve() / "img", target_is_directory=True)

from algorithms.calibration import read_calibration
from algorithms.parameter_converters import (
    get_control_par,
    get_sequence_par,
    get_track_par_tuple,
    get_volume_par,
)

cpar = get_control_par(params)
vpar = get_volume_par(params)
tpar = get_track_par_tuple(params)
spar = get_sequence_par(params)

# Adjust sequence base names for work_dir
spar.img_base_name = [str(work_dir / "img" / f"cam{i + 1}") for i in range(num_cams)]
spar.first = 10001
spar.last = 10004

cals = []
for i in range(num_cams):
    ori_file = CAVITY_DATA / params["cal_ori"]["img_ori"][i]
    addpar_file = ori_file.with_suffix(".addpar")
    cals.append(
        read_calibration(ori_file, addpar_file if addpar_file.exists() else None)
    )

naming = {
    "corres": str(res / "rt_is"),
    "linkage": str(res / "ptv_is"),
    "prio": str(res / "added"),
}

tracker = Tracker(cpar, vpar, tpar, spar, cals, naming)
tracker.full_forward()

print("Results for cavity dataset:")
counts = []
for f in sorted(res.iterdir()):
    if f.name.startswith("ptv_is."):
        with open(f) as fh:
            line = fh.readline().strip()
            if line:
                n = int(line)
                counts.append(n)
                print(f"  {f.name}: {n} links")

# Check linkage files
total_links = sum(counts)
print(f"Total links found: {total_links}")

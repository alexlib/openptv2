# ruff: noqa: E402
from pathlib import Path

import yaml
from algorithms.track import Tracker

TRACK_DATA = Path("test_data/track")
CONF_YAML = TRACK_DATA / "conf.yaml"

with open(CONF_YAML) as f:
    conf = yaml.safe_load(f)

# Mock some paths
import tempfile

work_dir = Path(tempfile.mkdtemp())
res_orig = TRACK_DATA / "res_orig"
res = work_dir / "res"
res.mkdir()
for f in res_orig.iterdir():
    import shutil

    shutil.copy2(f, res / f.name)

# Symlink newpart
(work_dir / "newpart").symlink_to(TRACK_DATA / "newpart", target_is_directory=True)
(work_dir / "cal").symlink_to(TRACK_DATA / "cal", target_is_directory=True)

from algorithms.calibration import read_calibration
from algorithms.parameters import (
    ControlPar,
    MultimediaPar,
    SequencePar,
    TrackParTuple,
    VolumePar,
)

scene = conf["scene"]
corresp = conf["correspondences"]
tracking = conf["tracking"]
vel = tracking["velocity_lims"]

cals = [read_calibration(c["ori_file"], c.get("addpar_file")) for c in conf["cameras"]]

mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
cpar = ControlPar(
    num_cams=4,
    imx=scene["image_size"][0],
    imy=scene["image_size"][1],
    pix_x=scene["pixel_size"][0],
    pix_y=scene["pixel_size"][1],
    mm=mm,
)
vpar = VolumePar(
    x_lay=corresp["x_span"],
    z_min_lay=[corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))],
    z_max_lay=[corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))],
    cn=corresp.get("pixels_tot", 0),
    cnx=corresp.get("pixels_x", 0),
    cny=corresp.get("pixels_y", 0),
    csumg=corresp.get("ref_gray", 0),
    eps0=corresp.get("epipolar_band", 0),
    corrmin=corresp.get("min_correlation", 0),
)
tpar = TrackParTuple(
    dvxmin=vel[0][0],
    dvxmax=vel[0][1],
    dvymin=vel[1][0],
    dvymax=vel[1][1],
    dvzmin=vel[2][0],
    dvzmax=vel[2][1],
    dangle=tracking["angle_lim"],
    dacc=tracking["accel_lim"],
    add=tracking.get("add_particle", 0),
    dsumg=0.0,
    dn=0.0,
    dnx=0.0,
    dny=0.0,
)
seq = conf["sequence"]
img_base = [str(work_dir / "newpart" / f"cam{i + 1}.") for i in range(4)]
spar = SequencePar(img_base_name=img_base, first=seq["first"], last=seq["last"])

naming = {
    "corres": str(res / "particles"),
    "linkage": str(res / "linkage"),
    "prio": str(res / "whatever"),
}

tracker = Tracker(cpar, vpar, tpar, spar, cals, naming)
tracker.full_forward()

print(f"Results in {res}:")
for f in sorted(res.iterdir()):
    if f.is_file():
        with open(f) as fh:
            first = fh.readline().strip()
        print(f"  {f.name}: {first} particles/links")

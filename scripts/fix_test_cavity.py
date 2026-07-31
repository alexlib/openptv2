import os
from pathlib import Path

import numpy as np
import yaml

os.chdir("/home/user/Documents/GitHub/openptv2")

from algorithms.batch import (
    _build_control_par,
    _build_sequence_par,
    _build_volume_par,
    _read_calibrations_py,
    _target_file_bases,
)
from algorithms.correspondences import MatchedCoords
from algorithms.correspondences import correspondences as run_corres
from algorithms.tracking_frame_buf import Frame, read_targets, write_path_frame

TEST_DATA_DIR = Path("test_data/test_cavity")
YAML_FILE = TEST_DATA_DIR / "parameters_Run1.yaml"
RES_DIR = TEST_DATA_DIR / "res"
RES_DIR.mkdir(exist_ok=True)

with open(YAML_FILE) as f:
    params = yaml.safe_load(f)

num_cams = params["num_cams"]
cpar = _build_control_par(params["ptv"], num_cams)
spar = _build_sequence_par(params["sequence"], num_cams)
vpar = _build_volume_par(params["criteria"])

# Fix calibration paths to be relative to TEST_DATA_DIR
cal_ori_params = params["cal_ori"].copy()
cal_ori_params["img_cal_name"] = [str(TEST_DATA_DIR / name) for name in cal_ori_params["img_cal_name"]]
cal_ori_params["img_ori"] = [str(TEST_DATA_DIR / name) for name in cal_ori_params["img_ori"]]
cals = _read_calibrations_py(cal_ori_params, num_cams)

short_file_bases = [str(TEST_DATA_DIR / b) for b in _target_file_bases(spar.img_base_name, num_cams)]

for frame_num in range(spar.first, spar.last + 1):
    print(f"Processing frame {frame_num}...")
    frm = Frame(num_cams=num_cams)
    detections = []
    corrected = []
    for i_cam in range(num_cams):
        targs = read_targets(short_file_bases[i_cam], frame_num)
        # Ensure targets are sorted by Y as expected by many parts of the system
        targs.sort(key=lambda t: t.y)
        for tnum, t in enumerate(targs):
            t.pnr = tnum

        frm.num_targets[i_cam] = len(targs)
        frm.targets[i_cam][:len(targs)] = targs
        detections.append(targs)

        mc = MatchedCoords(targs, cpar, cals[i_cam])
        corrected.append(mc)

    # Run correspondences
    match_counts = [0] * (num_cams + 1)
    run_corres(frm, corrected, vpar, cpar, cals, match_counts)

    print(f"  Found {frm.num_parts} particles")

    # Calculate 3D positions
    from algorithms.track import fast_point_position
    # Prepare parameters for fast_point_position
    cal_ex_pos = np.array([[c.ext_par.x0, c.ext_par.y0, c.ext_par.z0] for c in cals])
    cal_ex_dm = np.array([c.ext_par.dm for c in cals])
    cal_int_cc = np.array([c.int_par.cc for c in cals])
    cal_glass_par = np.array([c.glass_par for c in cals])

    # We need per-camera mm parameters stacked
    # Using global cpar.mm for all cameras
    mm_d_stack = np.array([[cpar.mm.d[0]] for _ in range(num_cams)])
    # SWAP TEST: n2 <-> n3
    mm_n2_stack = np.array([[cpar.mm.n3] for _ in range(num_cams)]) # Using n3 as n2
    mm_n1_stack = np.array([cpar.mm.n1 for _ in range(num_cams)])
    mm_n3_stack = np.array([cpar.mm.n2[0] for _ in range(num_cams)]) # Using n2 as n3

    for i in range(frm.num_parts):
        t_pos = np.full((num_cams, 2), -1.0e10)
        for cam in range(num_cams):
            t_idx = frm.corres_p[i, cam]
            if t_idx >= 0:
                xm, ym = corrected[cam][t_idx].x, corrected[cam][t_idx].y
                t_pos[cam, 0] = xm
                t_pos[cam, 1] = ym

        dist, pos3d = fast_point_position(t_pos, num_cams, cal_ex_pos, cal_ex_dm, cal_int_cc, cal_glass_par, mm_d_stack, mm_n1_stack, mm_n2_stack, mm_n3_stack)
        frm.path_info[i].x = pos3d

    # Write rt_is
    write_path_frame(frm.corres_nr, frm.corres_p, frm.path_info, frm.num_parts,
                     str(RES_DIR / "rt_is"), str(RES_DIR / "ptv_is"), "", frame_num)

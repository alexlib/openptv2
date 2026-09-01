"""Write a runnable openptv2 experiment folder from a rig + dataset.

Builds the on-disk layout expected by the batch pipeline / plugin runner::

    <dir>/cal/   camN.tif.ori, camN.tif.addpar
    <dir>/res/   rt_is.#, ptv_is.#, added.#, origin_#.txt
    <dir>/img/   camN<frame>_targets
    <dir>/parameters_Run1.yaml

The YAML points at these calibration and target files so that
``build_processing_experiment`` + ``run_tracking_plugin`` can process the
dataset headlessly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from openptv2.benchmarking.camera_rig import CameraRig
from openptv2.benchmarking.datawriter import DatasetSpec, write_dataset

# Realistic refractive indices (YAML convention: n2 = water, n3 = glass).
_N1 = 1.0
_N2_WATER = 1.33
_N3_GLASS = 1.46


def write_experiment(
    rig: CameraRig,
    frame_gt: dict[int, list[tuple[int, float, float, float]]],
    out_dir: str | Path,
    first_frame: int = 10001,
    volume: tuple[float, float, float] = (100.0, 100.0, 100.0),
) -> Path:
    """Write a complete, runnable experiment folder.

    Parameters
    ----------
    rig : CameraRig
    frame_gt : dict[int, list[(pid, x, y, z)]]
        Per-frame ground truth from ``generate_scenario``.
    out_dir : str | Path
        Where to write the experiment.
    first_frame : int
        First frame number (also the offset applied to ground-truth frames).
    volume : tuple[float, float, float]
        Measurement volume size, used only for the YAML criteria section.

    Returns
    -------
    Path
        The parameters YAML path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = out_dir / "cal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Write per-camera calibrations.
    for i, cal in enumerate(rig.cals):
        cal.to_file(
            str(cal_dir / f"cam{i + 1}.tif.ori"),
            str(cal_dir / f"cam{i + 1}.tif.addpar"),
        )

    # Write the dataset (rt_is / ptv_is / added / targets / origin).
    write_dataset(
        rig,
        frame_gt,
        DatasetSpec(
            dir=out_dir,
            res_sub="res",
            img_sub="img",
            first_frame=first_frame,
            num_cams=len(rig.cals),
        ),
    )

    num_cams = len(rig.cals)
    last_frame = first_frame + len(frame_gt) - 1
    imx, imy = rig.cpar.imx, rig.cpar.imy

    misc = num_cams == 4 and rig.refract
    n2 = _N2_WATER if misc else 1.0
    n3 = _N3_GLASS if misc else 1.0
    d = rig.cpar.mm.d[0] if rig.refract else 0.0

    half = [float(v) for v in (np.array(volume) / 2.0)]

    img_cal = [f"cal/cam{i + 1}.tif" for i in range(num_cams)]
    img_ori = [f"cal/cam{i + 1}.tif.ori" for i in range(num_cams)]
    img_name = [f"img/cam{i + 1}.{first_frame}" for i in range(num_cams)]
    base_name = [f"img/cam{i + 1}.%d" for i in range(num_cams)]

    yaml_data = {
        "num_cams": num_cams,
        "plugins": {
            "available_tracking": ["default"],
            "available_sequence": ["default"],
            "selected_tracking": "default",
            "selected_sequence": "default",
        },
        "cal_ori": {
            "chfield": 0,
            "fixp_name": "cal/calibration_target.txt",
            "img_cal_name": img_cal,
            "img_ori": img_ori,
            "pair_flag": True,
            "tiff_flag": True,
            "cal_splitter": False,
        },
        "criteria": {
            "X_lay": [-half[0], half[0]],
            "Zmax_lay": [half[2], half[2]],
            "Zmin_lay": [-half[2], -half[2]],
            "cn": 0.2,
            "cnx": 0.2,
            "cny": 0.2,
            "corrmin": 50.0,
            "csumg": 0.2,
            "eps0": 0.1,
        },
        "detect_plate": {
            "gvth_1": 10,
            "gvth_2": 10,
            "gvth_3": 10,
            "gvth_4": 10,
            "max_npix": 400,
            "max_npix_x": 50,
            "max_npix_y": 50,
            "min_npix": 25,
            "min_npix_x": 5,
            "min_npix_y": 5,
            "size_cross": 3,
            "sum_grey": 100,
            "tol_dis": 500,
        },
        "dumbbell": {
            "dumbbell_eps": 3.0,
            "dumbbell_gradient_descent": 0.05,
            "dumbbell_niter": 500,
            "dumbbell_penalty_weight": 1.0,
            "dumbbell_scale": 25.0,
            "dumbbell_step": 1,
        },
        "examine": {"Combine_Flag": False, "Examine_Flag": False},
        "man_ori": {"nr": [41, 50, 51, 60, 41, 50, 51, 60]},
        "multi_planes": {
            "n_planes": 3,
            "plane_name": ["img/calib_a_cam", "img/calib_b_cam", "img/calib_c_cam"],
        },
        "orient": {
            "cc": 0,
            "interf": 0,
            "k1": 0,
            "k2": 0,
            "k3": 0,
            "p1": 0,
            "p2": 0,
            "pnfo": 0,
            "scale": 0,
            "shear": 0,
            "xh": 0,
            "yh": 0,
        },
        "pft_version": {"Existing_Target": 1},
        "ptv": {
            "allcam_flag": False,
            "chfield": 0,
            "hp_flag": True,
            "img_cal": img_cal,
            "img_name": img_name,
            "imx": imx,
            "imy": imy,
            "mmp_d": float(d),
            "mmp_n1": _N1,
            "mmp_n2": float(n2),
            "mmp_n3": float(n3),
            "pix_x": float(rig.cpar.pix_x),
            "pix_y": float(rig.cpar.pix_y),
            "tiff_flag": True,
            "splitter": False,
        },
        "sequence": {
            "base_name": base_name,
            "first": first_frame,
            "last": last_frame,
        },
        "shaking": {
            "shaking_first_frame": 10000,
            "shaking_last_frame": 10004,
            "shaking_max_num_frames": 5,
            "shaking_max_num_points": 10,
        },
        "sortgrid": {"radius": 20},
        "targ_rec": {
            "cr_sz": 2,
            "disco": 100,
            "gvthres": [25, 25],
            "nnmax": 500,
            "nnmin": 10,
            "nxmax": 100,
            "nxmin": 10,
            "nymax": 100,
            "nymin": 10,
            "sumg_min": 100,
        },
        "track": {
            "angle": 120.0,
            "dacc": 5.5,
            "dvxmax": 10.0,
            "dvxmin": -10.0,
            "dvymax": 10.0,
            "dvymin": -10.0,
            "dvzmax": 10.0,
            "dvzmin": -10.0,
            "flagNewParticles": True,
        },
        "masking": {"mask_flag": False, "mask_base_name": ""},
        "unsharp_mask": {"flag": False, "size": 3, "strength": 1.0},
    }

    yaml_path = out_dir / "parameters_Run1.yaml"
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(yaml_data, fh, sort_keys=False)

    return yaml_path


__all__ = ["write_experiment"]

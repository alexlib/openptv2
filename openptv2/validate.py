#!/usr/bin/env python
"""Runtime validation tool for the single-engine openptv2 API."""

import argparse
import sys
from pathlib import Path

import numpy as np

import openptv2

TEST_DATA = Path(__file__).parent.parent / "test_data" / "synthetic"


def _load_calibration(cam_num):
    cal = openptv2.Calibration()
    cal.from_file(
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.ori"),
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.addpar"),
    )
    return cal


def validate_runtime_info():
    info = openptv2.get_runtime_info()
    return info["engine"] == "cython3-pure-python" and isinstance(info["compiled"], bool)


def validate_transforms(tolerance):
    cpar = openptv2.ControlParams(num_cams=4)
    cpar.set_image_size((1280, 1024))
    cpar.set_pixel_size((0.012, 0.012))

    pixels = np.array(
        [[640.0, 512.0], [100.25, 200.75], [1100.5, 900.1]],
        dtype=np.float64,
    )
    metric = openptv2.convert_arr_pixel_to_metric(pixels, cpar)
    roundtrip = openptv2.convert_arr_metric_to_pixel(metric, cpar)
    return np.allclose(roundtrip, pixels, rtol=tolerance, atol=tolerance)


def validate_image_coordinates():
    cal = _load_calibration(1)
    mm = openptv2.MultimediaParams(n1=1.0, n3=1.0)
    positions = np.array(
        [[0.0, 0.0, 100.0], [10.0, -15.0, 120.0], [-25.0, 30.0, 80.0]],
        dtype=np.float64,
    )
    coords = openptv2.image_coordinates(positions, cal, mm)
    return coords.shape == (3, 2) and np.isfinite(coords).all()


def validate_epipolar_curve():
    cal1 = _load_calibration(1)
    cal2 = _load_calibration(2)
    cpar = openptv2.ControlParams(num_cams=4)
    cpar.read_control_par(str(TEST_DATA / "parameters" / "ptv.par"))
    vpar = openptv2.VolumeParams()
    vpar.read_volume_par(str(TEST_DATA / "parameters" / "criteria.par"))
    curve = openptv2.epipolar_curve(
        np.array([640.0, 512.0], dtype=np.float64), cal1, cal2, 10, cpar, vpar
    )
    return curve.shape == (10, 2) and np.isfinite(curve).all()


def validate_segmentation():
    np.random.seed(42)
    image = (np.random.rand(256, 256) * 10).astype(np.uint8)
    image[50:55, 60:65] = 200
    image[120:128, 180:188] = 180

    cpar = openptv2.ControlParams(num_cams=4)
    cpar.set_image_size((256, 256))
    tpar = openptv2.TargetParams(
        discont=10,
        gvthresh=[50, 50, 50, 50],
        pixel_count_bounds=(5, 100),
        xsize_bounds=(2, 20),
        ysize_bounds=(2, 20),
        min_sum_grey=100,
    )
    targets = openptv2.target_recognition(image, tpar, 0, cpar)
    return len(targets) >= 2


def main():
    parser = argparse.ArgumentParser(
        description="Validate the single-engine openptv2 runtime"
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=1e-10,
        help="Floating-point tolerance for transform round-trips.",
    )
    args = parser.parse_args()

    checks = {
        "runtime_info": validate_runtime_info(),
        "transforms": validate_transforms(args.tolerance),
        "image_coordinates": validate_image_coordinates(),
        "epipolar_curve": validate_epipolar_curve(),
        "segmentation": validate_segmentation(),
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

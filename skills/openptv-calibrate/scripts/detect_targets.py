#!/usr/bin/env python
"""Detect calibration-plate targets for every camera in a dataset.

Closes the gap `calib.py inspect` flags as "out of scope": a genuinely fresh
dataset has cal images but no cam_N.tif_targets yet. This runs the same
detection code path the GUI's calibration "Detection" button uses
(``openptv2.gui.ptv.py_detection_proc_c``) against ``detect_plate`` thresholds
from the dataset YAML, then writes ``<img>_targets`` next to each cal image
via the same low-level writer/naming convention ``autocalibration.py``'s
``read_targets`` expects (frame_num=0 -> plain ``_targets`` suffix, no frame
number).

Usage:
    uv run python skills/openptv-calibrate/scripts/detect_targets.py <dataset>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from openptv2.autocalibration import _find_yaml, cam_files, _cpar_from_ptv
from openptv2.gui.ptv import _populate_tpar, py_detection_proc_c
from openptv2.algorithms.tracking_frame_buf import write_targets
import yaml


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: detect_targets.py <dataset>", file=sys.stderr)
        return 1
    base = Path(sys.argv[1]).resolve()

    yaml_path = _find_yaml(base)
    if yaml_path is None:
        print(f"ERROR: no parameters_*.yaml found in {base}", file=sys.stderr)
        return 1
    y = yaml.safe_load(yaml_path.read_text())
    num_cams = int(y.get("num_cams") or y["ptv"].get("num_cams"))
    ptv_params = y["ptv"]
    detect_params = {"detect_plate": y["detect_plate"]}

    images = []
    for cam in range(num_cams):
        img_path, _, _ = cam_files(base, cam)
        img = imread(str(img_path))
        if img.ndim > 2:
            img = rgb2gray(img[:, :, :3])
        img = img_as_ubyte(img)
        images.append(img)

    detections, _ = py_detection_proc_c(num_cams, images, ptv_params, detect_params)

    for cam in range(num_cams):
        img_path, _, _ = cam_files(base, cam)
        targs = detections[cam]
        n = len(targs)
        write_targets(list(targs), n, str(img_path), 0)
        print(f"cam{cam + 1}: {n} targets -> {img_path}_targets")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

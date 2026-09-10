"""End-to-end coverage: filter_by_trajectory_roi wired into the real
triangulation step (py_determination_proc_c), not just exercised in
isolation. No trajectory_roi.json -> unrestricted default; a saved ROI
strictly reduces the 3D correspondences written for that frame.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.correspondences import correspondences
from openptv2.gui import ptv
from openptv2.gui.experiment import Experiment
from openptv2.storage import RunStore
from openptv2.trajectory_roi import ROI_FILENAME, TrajectoryROI, _roi_cache

FRAME = 10001


@pytest.fixture
def temp_cavity_dir(tmp_path):
    """A private copy of test_cavity, chdir'd into, with no leftover ROI/store."""
    original_dir = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not original_dir.exists():
        pytest.skip("test_cavity directory not found")

    dest_dir = tmp_path / "test_cavity"
    shutil.copytree(
        original_dir,
        dest_dir,
        ignore=shutil.ignore_patterns(
            "mask_*.txt", "trajectory_roi.json", "run.zarr"
        ),
    )

    old_cwd = os.getcwd()
    os.chdir(dest_dir)
    _roi_cache.clear()
    try:
        yield dest_dir
    finally:
        os.chdir(old_cwd)
        _roi_cache.clear()


def _triangulate_one_frame(dest_dir, store_path):
    """Run detection + correspondence + determination for FRAME, cam 1 (index 0
    onward), through the same functions the real pipeline uses, and return the
    RunStore holding the frame's 3D correspondences."""
    yaml_file = dest_dir / "parameters_Run1.yaml"
    exp = Experiment()
    exp.pm.from_yaml(yaml_file)
    num_cams = exp.pm.num_cams

    cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(exp.pm)
    ptv_params = exp.pm.get_parameter("ptv")
    img_base_names = [spar.get_img_base_name(i) for i in range(num_cams)]

    images = ptv.read_frame_images(exp.pm, img_base_names, num_cams, FRAME)
    images = ptv.py_pre_processing_c(num_cams, images, ptv_params)

    targ_rec_params = exp.pm.get_parameter("targ_rec")
    detections, corrected = ptv.py_detection_proc_c(
        num_cams, images, ptv_params, {"targ_rec": targ_rec_params}
    )
    sorted_pos, sorted_corresp, _ = correspondences(
        detections, corrected, cals, vpar, cpar
    )

    store = RunStore(store_path, mode="a")
    ptv.py_determination_proc_c(
        num_cams,
        sorted_pos,
        sorted_corresp,
        corrected,
        cpar,
        vpar,
        cals,
        frame=FRAME,
        store=store,
    )
    return store


def test_missing_roi_defaults_to_unrestricted_correspondences(temp_cavity_dir):
    assert not (temp_cavity_dir / ROI_FILENAME).exists()

    store = _triangulate_one_frame(temp_cavity_dir, temp_cavity_dir / "run.zarr")
    pos, _ = store.read_correspondences(FRAME)

    assert len(pos) > 0, "sanity: triangulation should find 3D points"


def test_saved_roi_strictly_reduces_correspondences(temp_cavity_dir):
    baseline_store = _triangulate_one_frame(
        temp_cavity_dir, temp_cavity_dir / "baseline.zarr"
    )
    baseline_pos, _ = baseline_store.read_correspondences(FRAME)
    assert len(baseline_pos) > 0

    # Half-plane ROI (XY only) at the x-median -- keeps roughly half the cloud.
    mid_x = float(np.median(baseline_pos[:, 0]))
    y_lo, y_hi = baseline_pos[:, 1].min() - 1, baseline_pos[:, 1].max() + 1
    x_lo = baseline_pos[:, 0].min() - 1
    roi = TrajectoryROI(
        xy_polygon=np.array(
            [[x_lo, y_lo], [mid_x, y_lo], [mid_x, y_hi], [x_lo, y_hi]], dtype=float
        )
    )
    roi.to_json(temp_cavity_dir / ROI_FILENAME)
    _roi_cache.clear()

    masked_store = _triangulate_one_frame(
        temp_cavity_dir, temp_cavity_dir / "masked.zarr"
    )
    masked_pos, _ = masked_store.read_correspondences(FRAME)

    assert 0 < len(masked_pos) < len(baseline_pos)
    assert (masked_pos[:, 0] <= mid_x + 1e-6).all()

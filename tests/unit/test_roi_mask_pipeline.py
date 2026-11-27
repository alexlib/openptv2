"""End-to-end coverage for the per-camera ROI mask through the real detection
pipeline: no mask -> unmasked default, a saved mask actually removes targets
outside it, and that holds identically whether preprocessing runs with one
worker or several (each worker process gets its own roi_mask cache).
"""

import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from openptv2.gui import ptv
from openptv2.gui.experiment import Experiment
from openptv2.roi_mask import _rasterized_cache
from openptv2.tracking_framebuf import read_targets

# test_cavity images are 1280x1024 (see parameters_Run1.yaml); this polygon
# keeps only the left half.
LEFT_HALF_MASK = "0 0\n639 0\n639 1023\n0 1023\n"


@pytest.fixture
def temp_cavity_dir(tmp_path):
    """A private copy of test_cavity, chdir'd into, with no leftover masks."""
    original_dir = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not original_dir.exists():
        pytest.skip("test_cavity directory not found")

    dest_dir = tmp_path / "test_cavity"
    shutil.copytree(
        original_dir,
        dest_dir,
        ignore=shutil.ignore_patterns("mask_*.txt", "trajectory_roi.json"),
    )

    old_cwd = os.getcwd()
    os.chdir(dest_dir)
    _rasterized_cache.clear()
    try:
        yield dest_dir
    finally:
        os.chdir(old_cwd)
        _rasterized_cache.clear()


def _make_exp(dest_dir, first, last):
    yaml_file = dest_dir / "parameters_Run1.yaml"
    exp = Experiment()
    exp.pm.from_yaml(yaml_file)
    exp.target_filenames = exp.pm.get_target_filenames()

    cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(exp.pm)
    exp_mock = Mock()
    exp_mock.pm = exp.pm
    exp_mock.num_cams = exp.pm.num_cams
    exp_mock.cpar = cpar
    exp_mock.spar = spar
    exp_mock.vpar = vpar
    exp_mock.track_par = track_par
    exp_mock.tpar = tpar
    exp_mock.cals = cals
    exp_mock.target_filenames = exp.target_filenames
    spar.set_first(first)
    spar.set_last(last)

    ptv._ensure_target_output_writable(exp.target_filenames)
    return exp, exp_mock


def _clear_targets(exp, first, last):
    for frame in range(first, last + 1):
        for i_cam in range(exp.pm.num_cams):
            target_file = Path(f"{exp.target_filenames[i_cam]}.{frame:04d}_targets")
            if target_file.exists():
                target_file.unlink()


def _read_cam0_targets(exp, frame):
    # ptv.write_targets names files "<base>.<frame>_targets" (with the dot);
    # tracking_framebuf.read_targets does not add one itself, so the base
    # passed here must include it too, or every read silently returns [].
    return list(read_targets(f"{exp.target_filenames[0]}.", frame))


def test_missing_mask_defaults_to_unmasked_detection(temp_cavity_dir):
    """No mask_{cam}.txt anywhere in the working folder -> full-frame detection."""
    assert not list(temp_cavity_dir.glob("mask_*.txt"))

    exp, exp_mock = _make_exp(temp_cavity_dir, 10001, 10001)
    _clear_targets(exp, 10001, 10001)

    ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=1)

    targs = _read_cam0_targets(exp, 10001)
    assert len(targs) > 0, "sanity: the unmasked frame should detect particles"
    # A wide spread of x -- nothing was cut out by a phantom default mask.
    xs = [t.pos()[0] for t in targs]
    assert max(xs) > 640, "targets on the right half were unexpectedly removed"


def test_saved_mask_removes_targets_outside_the_polygon(temp_cavity_dir):
    exp, exp_mock = _make_exp(temp_cavity_dir, 10001, 10001)
    _clear_targets(exp, 10001, 10001)
    ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=1)
    baseline = _read_cam0_targets(exp, 10001)
    assert len(baseline) > 0

    _clear_targets(exp, 10001, 10001)
    (temp_cavity_dir / "mask_0.txt").write_text(LEFT_HALF_MASK)
    _rasterized_cache.clear()

    ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=1)
    masked = _read_cam0_targets(exp, 10001)

    assert 0 < len(masked) < len(baseline)
    assert all(t.pos()[0] < 640 for t in masked)


def test_mask_applies_identically_across_worker_counts(temp_cavity_dir):
    """Masking must not depend on which process (or how many) did the work --
    each worker rasterizes its own copy of the same mask_0.txt independently."""
    exp, exp_mock = _make_exp(temp_cavity_dir, 10001, 10002)
    (temp_cavity_dir / "mask_0.txt").write_text(LEFT_HALF_MASK)

    _clear_targets(exp, 10001, 10002)
    ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=1)
    sequential = {frame: _read_cam0_targets(exp, frame) for frame in (10001, 10002)}

    _clear_targets(exp, 10001, 10002)
    ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=2)
    parallel = {frame: _read_cam0_targets(exp, frame) for frame in (10001, 10002)}

    for frame in (10001, 10002):
        assert len(parallel[frame]) > 0
        assert len(parallel[frame]) == len(sequential[frame])
        assert all(t.pos()[0] < 640 for t in parallel[frame])

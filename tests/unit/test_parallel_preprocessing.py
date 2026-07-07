import os
import shutil
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from openptv2.gui import ptv
from openptv2.gui.experiment import Experiment
from openptv2.tracking_framebuf import read_targets


@pytest.fixture
def temp_cavity_dir(tmp_path):
    """Fixture to copy the test_cavity dataset to a temporary directory.

    This ensures that our tests are completely non-destructive and do not
    leave any generated target files in the active git workspace.
    """
    original_dir = (
        Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    )
    if not original_dir.exists():
        pytest.skip("test_cavity directory not found")

    dest_dir = tmp_path / "test_cavity"
    shutil.copytree(original_dir, dest_dir)

    old_cwd = os.getcwd()
    os.chdir(dest_dir)
    try:
        yield dest_dir
    finally:
        os.chdir(old_cwd)


def test_parallel_preprocessing_parity(temp_cavity_dir):
    """Verify that parallel and sequential pre-processing produce identical target files."""
    # 1. Load the experiment
    yaml_file = temp_cavity_dir / "parameters_Run1.yaml"
    exp = Experiment()
    exp.pm.from_yaml(yaml_file)
    exp.target_filenames = exp.pm.get_target_filenames()

    # 2. Re-populate calibration and param structures
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

    # Limit frame range to all 5 available frames (10000 to 10004) to keep it fast but thorough
    spar.set_first(10000)
    spar.set_last(10004)

    # Ensure output targets directory exists
    ptv._ensure_target_output_writable(exp.target_filenames)

    # Delete any pre-existing targets in the temp copy
    for frame in range(10000, 10005):
        for i_cam in range(exp.pm.num_cams):
            target_file = Path(f"{exp.target_filenames[i_cam]}.{frame:04d}_targets")
            if target_file.exists():
                target_file.unlink()

    # 3. Run preprocessing sequentially (num_workers=1)
    os.environ["OPENPTV_PARALLEL_PREPROCESS"] = "True"
    os.environ["OPENPTV_NUM_WORKERS"] = "1"
    ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=1)

    # Read and store baseline sequential targets
    sequential_targets = {}
    for frame in range(10000, 10005):
        sequential_targets[frame] = []
        for i_cam in range(exp.pm.num_cams):
            targs = read_targets(exp.target_filenames[i_cam], frame)
            sequential_targets[frame].append(list(targs))

    # Delete the generated sequential targets to prevent false positives
    for frame in range(10000, 10005):
        for i_cam in range(exp.pm.num_cams):
            target_file = Path(f"{exp.target_filenames[i_cam]}.{frame:04d}_targets")
            if target_file.exists():
                target_file.unlink()

    # 4. Run preprocessing in parallel (num_workers=4)
    os.environ["OPENPTV_PARALLEL_PREPROCESS"] = "True"
    os.environ["OPENPTV_NUM_WORKERS"] = "4"
    ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=4)

    # 5. Read and compare the parallel-generated targets with baseline
    for frame in range(10000, 10005):
        for i_cam in range(exp.pm.num_cams):
            parallel_targs = list(read_targets(exp.target_filenames[i_cam], frame))
            seq_targs = sequential_targets[frame][i_cam]

            assert len(parallel_targs) == len(seq_targs), (
                f"Target count mismatch on frame {frame}, cam {i_cam}: "
                f"parallel={len(parallel_targs)}, sequential={len(seq_targs)}"
            )

            for idx, (p_t, s_t) in enumerate(zip(parallel_targs, seq_targs)):
                assert p_t.pnr() == s_t.pnr(), f"pnr mismatch at target {idx}"
                assert np.allclose(p_t.pos(), s_t.pos(), atol=1e-7), (
                    f"pos mismatch at target {idx}"
                )
                assert p_t.count_pixels() == s_t.count_pixels(), (
                    f"count_pixels mismatch at target {idx}"
                )
                assert p_t.sum_grey_value() == s_t.sum_grey_value(), (
                    f"sum_grey_value mismatch at target {idx}"
                )
                assert p_t.tnr() == s_t.tnr(), f"tnr mismatch at target {idx}"

    # Clean up environment variables
    os.environ.pop("OPENPTV_PARALLEL_PREPROCESS", None)
    os.environ.pop("OPENPTV_NUM_WORKERS", None)


def test_parallel_preprocessing_file_handling_and_cleanliness(temp_cavity_dir):
    """Verify that parallel pre-processing cleanly propagates errors on missing files."""
    yaml_file = temp_cavity_dir / "parameters_Run1.yaml"
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

    spar.set_first(10000)
    spar.set_last(10001)

    # 1. Force error by deleting one of the image files
    target_img_to_delete = Path("img/cam1.10000")
    if target_img_to_delete.exists():
        target_img_to_delete.unlink()

    # 2. Ensure running parallel preprocessing raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=2)


def test_parallel_preprocessing_io_scaling_benchmark(temp_cavity_dir):
    """Run preprocessing on different worker counts and log the processing times."""
    yaml_file = temp_cavity_dir / "parameters_Run1.yaml"
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

    spar.set_first(10000)
    spar.set_last(10004)

    times = {}
    for num_workers in [1, 2, 4]:
        # Delete generated target files beforehand to force re-run
        for frame in range(10000, 10005):
            for i_cam in range(exp.pm.num_cams):
                target_file = Path(f"{exp.target_filenames[i_cam]}.{frame:04d}_targets")
                if target_file.exists():
                    target_file.unlink()

        start_time = time.time()
        ptv.preprocess_and_detect_all_parallel(exp_mock, num_workers=num_workers)
        elapsed = time.time() - start_time
        times[num_workers] = elapsed
        print(f"Workers={num_workers}: {elapsed:.3f} seconds")

    # Assert that multi-worker preprocessing doesn't crash
    assert len(times) == 3
    # Clean up
    for frame in range(10000, 10005):
        for i_cam in range(exp.pm.num_cams):
            target_file = Path(f"{exp.target_filenames[i_cam]}.{frame:04d}_targets")
            if target_file.exists():
                target_file.unlink()

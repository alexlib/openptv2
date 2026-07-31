import os
import shutil
import time
from pathlib import Path

import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.algorithms.track import (
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.tracking_run import tr_new


def read_all_calibration(num_cams, base_path="."):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


@pytest.fixture
def temp_cavity_dir(tmp_path):
    """Fixture to copy the test_cavity dataset to a temporary directory.

    This ensures that our tests are completely non-destructive and do not
    leave any generated tracking files in the active git workspace.
    """
    original_dir = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
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


def run_tracking(num_threads, add_flag):
    """Orchestrates a standard tracking run with the given thread count and add particle flag."""
    if os.path.exists("res"):
        shutil.rmtree("res")
    shutil.copytree("res_orig", "res")

    cpar = ControlPar.from_yaml("parameters.yaml")
    calib = read_all_calibration(cpar.num_cams, base_path=".")

    run = tr_new(
        SequencePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        VolumePar.from_yaml("parameters.yaml"),
        ControlPar.from_yaml("parameters.yaml"),
        4,
        20000,
        "res/rt_is",
        "res/ptv_is",
        "res/added",
        calib,
        0.0001,
    )
    run.tpar = run.tpar._replace(add=add_flag)

    track_forward_start(run)
    for step in range(run.seq_par.first, run.seq_par.last):
        trackcorr_c_loop(run, step, num_threads=num_threads)
    trackcorr_c_finish(run, run.seq_par.last)

    return run


def test_parallel_tracking_determinism(temp_cavity_dir):
    """Verify that parallel tracking runs are 100% deterministic (add=0) across 1, 2, 4, 8 threads."""
    # 1. Run sequential as baseline
    run_ref = run_tracking(num_threads=1, add_flag=0)
    ref_npart = run_ref.npart
    ref_nlinks = run_ref.nlinks

    ref_res_dir = temp_cavity_dir / "res_ref"
    shutil.copytree(temp_cavity_dir / "res", ref_res_dir)

    # 2. Run with varying multi-threading threads and compare to baseline
    for num_threads in [2, 4, 8]:
        run_mt = run_tracking(num_threads=num_threads, add_flag=0)

        # Assert high-level metrics match perfectly
        assert run_mt.npart == ref_npart, f"npart mismatch with {num_threads} threads"
        assert run_mt.nlinks == ref_nlinks, (
            f"nlinks mismatch with {num_threads} threads"
        )

        # Assert all frame files on disk are exactly identical
        for frame in range(run_ref.seq_par.first, run_ref.seq_par.last):
            ptv_ref = ref_res_dir / f"ptv_is.{frame}"
            ptv_mt = temp_cavity_dir / "res" / f"ptv_is.{frame}"

            assert ptv_ref.exists()
            assert ptv_mt.exists()

            with open(ptv_ref, "r") as f_ref, open(ptv_mt, "r") as f_mt:
                lines_ref = f_ref.readlines()
                lines_mt = f_mt.readlines()

            assert len(lines_ref) == len(lines_mt), (
                f"Line count mismatch on frame {frame} with {num_threads} threads"
            )
            for idx, (l_ref, l_mt) in enumerate(zip(lines_ref, lines_mt)):
                assert l_ref == l_mt, (
                    f"Content mismatch on frame {frame}, line {idx} with {num_threads} threads:\n"
                    f"REF: {l_ref.strip()}\n"
                    f"MT:  {l_mt.strip()}"
                )


def test_parallel_tracking_with_add_race(temp_cavity_dir):
    """Verify that parallel tracking with active particle addition (add=1) is completely deterministic and race-free."""
    # 1. Run sequential as baseline with particle additions
    run_ref = run_tracking(num_threads=1, add_flag=1)
    ref_npart = run_ref.npart
    ref_nlinks = run_ref.nlinks

    ref_res_dir = temp_cavity_dir / "res_ref_add"
    shutil.copytree(temp_cavity_dir / "res", ref_res_dir)

    # 2. Run with varying multi-threading threads and compare to baseline
    for num_threads in [2, 4, 8]:
        run_mt = run_tracking(num_threads=num_threads, add_flag=1)

        # Assert high-level metrics match perfectly (asserting no lost or overwritten tracks)
        assert run_mt.npart == ref_npart, (
            f"npart mismatch under multi-threading ({num_threads} threads)"
        )
        assert run_mt.nlinks == ref_nlinks, (
            f"nlinks mismatch under multi-threading ({num_threads} threads)"
        )

        # Assert all frame files on disk are exactly identical
        for frame in range(run_ref.seq_par.first, run_ref.seq_par.last):
            ptv_ref = ref_res_dir / f"ptv_is.{frame}"
            ptv_mt = temp_cavity_dir / "res" / f"ptv_is.{frame}"

            assert ptv_ref.exists()
            assert ptv_mt.exists()

            with open(ptv_ref, "r") as f_ref, open(ptv_mt, "r") as f_mt:
                lines_ref = f_ref.readlines()
                lines_mt = f_mt.readlines()

            assert len(lines_ref) == len(lines_mt), (
                f"ptv_is line count mismatch on frame {frame} with {num_threads} threads (add=1)"
            )
            for idx, (l_ref, l_mt) in enumerate(zip(lines_ref, lines_mt)):
                assert l_ref == l_mt, (
                    f"ptv_is content mismatch on frame {frame}, line {idx} with {num_threads} threads (add=1):\n"
                    f"REF: {l_ref.strip()}\n"
                    f"MT:  {l_mt.strip()}"
                )

            # Check added files are also identical
            added_ref = ref_res_dir / f"added.{frame}"
            added_mt = temp_cavity_dir / "res" / f"added.{frame}"

            if added_ref.exists():
                assert added_mt.exists()
                with open(added_ref, "r") as f_ref, open(added_mt, "r") as f_mt:
                    lines_ref_add = f_ref.readlines()
                    lines_mt_add = f_mt.readlines()
                assert len(lines_ref_add) == len(lines_mt_add), (
                    f"added line count mismatch on frame {frame} with {num_threads} threads"
                )
                for idx, (l_ref, l_mt) in enumerate(zip(lines_ref_add, lines_mt_add)):
                    assert l_ref == l_mt, (
                        f"added mismatch on frame {frame}, line {idx} with {num_threads} threads:\n"
                        f"REF: {l_ref.strip()}\n"
                        f"MT:  {l_mt.strip()}"
                    )


@pytest.mark.perf
def test_parallel_tracking_speedup_scaling(temp_cavity_dir):
    """Benchmark tracking speedup across thread counts (1, 2, 4, 8).

    Reports wall-clock speedup factors (speedup requires larger datasets).
    Asserts deterministic results identical across all thread counts.
    """
    thread_counts = [1, 2, 4, 8]
    times: dict[int, float] = {}
    results: dict[int, object] = {}

    for nt in thread_counts:
        t0 = time.perf_counter()
        run = run_tracking(num_threads=nt, add_flag=0)
        elapsed = time.perf_counter() - t0
        times[nt] = elapsed
        results[nt] = run

    # Report results
    baseline = times[1]
    print("\n── Tracking Speedup Scaling (cavity, 5 frames) ──")
    print(f"{'Threads':>8} {'Time (s)':>10} {'Speedup':>8} {'npart':>8} {'nlinks':>8}")
    for nt in thread_counts:
        speedup = baseline / times[nt]
        r = results[nt]
        print(f"{nt:>8} {times[nt]:>10.3f} {speedup:>7.2f}× {r.npart:>8} {r.nlinks:>8}")

    # Determinism check across all thread counts (critical correctness)
    ref_npart = results[1].npart
    ref_nlinks = results[1].nlinks
    for nt in [2, 4, 8]:
        assert results[nt].npart == ref_npart, f"npart mismatch at {nt} threads"
        assert results[nt].nlinks == ref_nlinks, f"nlinks mismatch at {nt} threads"

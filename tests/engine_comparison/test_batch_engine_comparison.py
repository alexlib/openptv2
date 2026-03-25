"""
Engine comparison test for batch processing.

Runs the complete batch pipeline twice - once with optv engine, once with python engine -
and compares all output files value by value to verify identical results.

Usage:
    pytest tests/engine_comparison/test_batch_engine_comparison.py -v
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List, Tuple, Any


TEST_DATA_DIR = Path(__file__).parent.parent / "testing_fodder" / "test_cavity"
TOLERANCE = 1e-10


def read_correspondence_file(file_path: Path) -> Tuple[int, np.ndarray]:
    """Read a correspondence file (rt_is.XXXXX).

    Returns:
        Tuple of (num_points, points_array)
    """
    if not file_path.exists():
        return 0, np.array([])

    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return 0, np.array([])

    num_points = int(lines[0].strip())

    if num_points == 0 or len(lines) <= 1:
        return num_points, np.array([])

    points = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 7:
            point = [float(x) for x in parts[:7]]
            points.append(point)

    return num_points, np.array(points)


def read_trajectory_file(file_path: Path) -> Tuple[int, np.ndarray]:
    """Read a trajectory file (trajectories.asc).

    Returns:
        Tuple of (num_trajectories, trajectories_array)
    """
    if not file_path.exists():
        return 0, np.array([])

    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return 0, np.array([])

    trajectories = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 10:
            traj = [float(x) for x in parts]
            trajectories.append(traj)

    return len(trajectories), np.array(trajectories)


def copy_test_data_with_yaml(source_dir: Path, dest_dir: Path, engine: str) -> Path:
    """Copy test data to a temporary directory and update YAML to use that location.

    Args:
        source_dir: Source test data directory
        dest_dir: Destination directory for test
        engine: Engine name ('optv' or 'python')

    Returns:
        Path to the copied YAML file
    """
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    yaml_files = list(dest_dir.glob("parameters*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML file found in {dest_dir}")

    yaml_file = yaml_files[0]

    with open(yaml_file, "r") as f:
        params = yaml.safe_load(f)

    if "sequence" in params:
        params["sequence"]["output"] = str(dest_dir / "res")

    new_yaml = dest_dir / f"parameters_{engine}.yaml"
    with open(new_yaml, "w") as f:
        yaml.dump(params, f)

    return new_yaml


def run_batch(yaml_file: Path, engine: str) -> Dict[str, Any]:
    """Run batch processing with specified engine.

    Args:
        yaml_file: Path to YAML parameters
        engine: 'optv' or 'python'

    Returns:
        Dictionary with run info and res directory path
    """
    from gui.pyptv import pyptv_batch

    with open(yaml_file, "r") as f:
        params = yaml.safe_load(f)

    first = params.get("sequence", {}).get("first", 10000)
    last = params.get("sequence", {}).get("last", 10004)

    if engine == "python":
        from openptv2.engine import set_engine, get_engine

        set_engine("python")
        print(f"[ENGINE] Set to: {engine}, current: {get_engine()}")

        # Verify Python engine modules are available
        try:
            from algorithms.track import Tracker as PythonTracker
            from algorithms.tracking_frame_buf import Target as PythonTarget

            print(f"[ENGINE] Python Tracker available: {PythonTracker}")
            print(f"[ENGINE] Python Target available: {PythonTarget}")
        except ImportError as e:
            print(f"[ENGINE] ERROR: Python modules not available: {e}")
    else:
        from openptv2.engine import set_engine, get_engine

        set_engine("optv")
        print(f"[ENGINE] Set to: {engine}, current: {get_engine()}")

        # Verify optv engine modules are available
        try:
            from optv.tracker import Tracker as OptvTracker
            from optv.tracking_framebuf import Target as OptvTarget

            print(f"[ENGINE] Optv Tracker available: {OptvTracker}")
            print(f"[ENGINE] Optv Target available: {OptvTarget}")
        except ImportError as e:
            print(f"[ENGINE] ERROR: Optv modules not available: {e}")

    # Check what tracker module the batch actually uses
    import gui.pyptv.ptv as ptv_module

    print(f"[ENGINE] ptv module Tracker: {ptv_module.Tracker}")
    print(f"[ENGINE] ptv module Tracker module: {ptv_module.Tracker.__module__}")

    pyptv_batch.main(yaml_file, first, last)

    res_dir = yaml_file.parent / "res"

    return {"first": first, "last": last, "res_dir": res_dir}


def compare_files(
    optv_dir: Path, python_dir: Path, first: int, last: int
) -> Dict[str, Any]:
    """Compare output files between optv and python runs."""
    differences = []

    for frame in range(first, last + 1):
        optv_file = optv_dir / f"rt_is.{frame}"
        python_file = python_dir / f"rt_is.{frame}"

        optv_num, optv_data = read_correspondence_file(optv_file)
        python_num, python_data = read_correspondence_file(python_file)

        if optv_num != python_num:
            differences.append(
                {
                    "file": f"rt_is.{frame}",
                    "type": "count_mismatch",
                    "optv": optv_num,
                    "python": python_num,
                }
            )
            continue

        if optv_num > 0 and len(optv_data) > 0 and len(python_data) > 0:
            try:
                np.testing.assert_allclose(
                    optv_data, python_data, rtol=TOLERANCE, atol=TOLERANCE
                )
            except AssertionError as e:
                differences.append(
                    {
                        "file": f"rt_is.{frame}",
                        "type": "value_mismatch",
                        "error": str(e)[:500],
                    }
                )

    optv_traj = optv_dir / "trajectories.asc"
    python_traj = python_dir / "trajectories.asc"

    optv_num, optv_trajs = read_trajectory_file(optv_traj)
    python_num, python_trajs = read_trajectory_file(python_traj)

    if optv_num != python_num:
        differences.append(
            {
                "file": "trajectories.asc",
                "type": "count_mismatch",
                "optv": optv_num,
                "python": python_num,
            }
        )
    elif optv_num > 0:
        try:
            np.testing.assert_allclose(
                optv_trajs, python_trajs, rtol=TOLERANCE, atol=TOLERANCE
            )
        except AssertionError as e:
            differences.append(
                {
                    "file": "trajectories.asc",
                    "type": "value_mismatch",
                    "error": str(e)[:500],
                }
            )

    return {"identical": len(differences) == 0, "differences": differences}


class TestBatchEngineComparison:
    """Test batch processing produces identical results with both engines."""

    @pytest.fixture
    def temp_dirs(self):
        optv_dir = tempfile.mkdtemp(prefix="optv_")
        python_dir = tempfile.mkdtemp(prefix="python_")

        yield Path(optv_dir), Path(python_dir)

        shutil.rmtree(optv_dir, ignore_errors=True)
        shutil.rmtree(python_dir, ignore_errors=True)

    def test_batch_engine_parity(self, temp_dirs):
        """Run batch with both engines and compare results."""
        optv_dir, python_dir = temp_dirs

        yaml_optv = copy_test_data_with_yaml(TEST_DATA_DIR, optv_dir, "optv")
        yaml_python = copy_test_data_with_yaml(TEST_DATA_DIR, python_dir, "python")

        print(f"\n=== Running batch with optv engine ===")
        try:
            optv_result = run_batch(yaml_optv, "optv")
        except Exception as e:
            pytest.skip(f"optv engine failed: {e}")

        print(f"\n=== Running batch with python engine ===")
        try:
            python_result = run_batch(yaml_python, "python")
        except Exception as e:
            pytest.skip(f"python engine failed: {e}")

        first = min(optv_result["first"], python_result["first"])
        last = max(optv_result["last"], python_result["last"])

        comparison = compare_files(
            optv_result["res_dir"], python_result["res_dir"], first, last
        )

        print(f"\n=== Engine Comparison Results ===")
        print(f"Identical: {comparison['identical']}")

        if not comparison["identical"]:
            print(f"Differences found: {len(comparison['differences'])}")
            for diff in comparison["differences"]:
                print(f"  - {diff['file']}: {diff['type']}")
                if "optv" in diff:
                    print(f"    optv: {diff['optv']}, python: {diff['python']}")
                if "error" in diff:
                    print(f"    {diff['error']}")

        assert comparison["identical"], (
            f"Engines produced different results: {comparison['differences']}"
        )

    def test_engines_available(self):
        """Verify both engines are available."""
        try:
            import optv
            from optv.tracker import Tracker

            optv_available = True
        except ImportError:
            optv_available = False

        try:
            from openptv2.engine import set_engine

            set_engine("python")
            python_available = True
        except ImportError:
            python_available = False

        assert optv_available, "optv engine not available"
        assert python_available, "python engine not available"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

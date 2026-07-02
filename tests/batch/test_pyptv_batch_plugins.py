"""Simple test for pyptv_batch_plugins.py - runs the actual code"""

import subprocess
import sys
import pytest
from pathlib import Path


def _get_env_with_pythonpath() -> dict:
    import os
    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent.parent / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir
    return env


def test_batch_plugins_runs():
    """Test that pyptv_batch_plugins runs without errors"""

    gui_dir = Path(__file__).parent.parent
    test_exp_path = Path(__file__).parent.parent.parent / "test_data" / "test_splitter"
    yaml_file = test_exp_path / "parameters_Run1.yaml"

    # Check if test experiment exists
    if not test_exp_path.exists():
        print(f"❌ Test experiment not found: {test_exp_path}")
        return False

    modes = ["both", "sequence", "tracking"]
    for mode in modes:
        cmd = [
            sys.executable,
            "-m",
            "openptv2.batch.pyptv_batch_plugins",
            str(yaml_file),
            "1000001",
            "1000005",
            "--mode",
            mode,
        ]
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=gui_dir, env=_get_env_with_pythonpath())
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            if result.returncode == 0:
                print(f"✅ Batch processing completed successfully for mode: {mode}")
            else:
                print(
                    f"❌ Process failed with return code: {result.returncode} for mode: {mode}"
                )
                pytest.fail(f"Batch processing failed for mode: {mode}")
        except subprocess.TimeoutExpired:
            pytest.fail(f"Process timed out for mode: {mode}")
        except Exception as e:
            pytest.fail(f"Error running process for mode {mode}: {e}")
    assert True


if __name__ == "__main__":
    test_batch_plugins_runs()
    print("\n🎉 Test passed!")

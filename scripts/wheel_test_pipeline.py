#!/usr/bin/env python
"""
openptv2 Wheel Build, Install & Test Pipeline

This script performs a complete local installation test:
1. Builds a binary wheel from source
2. Creates a clean virtual environment
3. Installs the wheel in the clean venv
4. Runs import verification tests
5. Runs the full test suite against the installed package

Usage:
    python wheel_test_pipeline.py              # Full pipeline
    python wheel_test_pipeline.py --build-only # Only build wheel
    python wheel_test_pipeline.py --install-only <wheel_path>  # Only install & test
    python wheel_test_pipeline.py --skip-build # Skip build, use existing wheel
    python wheel_test_pipeline.py --verbose    # Show detailed output

Requirements:
    - Python 3.11+
    - C compiler (gcc/clang/MSVC)
    - build, pip, virtualenv packages
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def colorize(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


def print_header(text: str):
    print()
    print(colorize("=" * 70, Colors.BOLD))
    print(colorize(f"  {text}", Colors.BOLD))
    print(colorize("=" * 70, Colors.BOLD))
    print()


def print_step(step: int, text: str):
    print(colorize(f"\n[Step {step}] {text}", Colors.OKCYAN))


def print_success(text: str):
    print(colorize(f"  OK: {text}", Colors.OKGREEN))


def print_failure(text: str):
    print(colorize(f"  FAIL: {text}", Colors.FAIL))


def print_info(text: str):
    print(f"  {text}")


def print_warning(text: str):
    print(colorize(f"  WARN: {text}", Colors.WARNING))


def run_cmd(
    cmd: list[str],
    cwd: Optional[Path] = None,
    verbose: bool = False,
    timeout: int = 600,
    env: Optional[dict] = None,
) -> Tuple[bool, str, float]:
    """Run a command and return (success, output, duration)."""
    start = datetime.now()
    full_env = {**os.environ, **(env or {})}

    if verbose:
        print_info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
        )
        duration = (datetime.now() - start).total_seconds()
        output = result.stdout + result.stderr
        return result.returncode == 0, output, duration
    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start).total_seconds()
        return False, f"Command timed out after {timeout}s", duration
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        return False, f"Command failed: {e}", duration


def find_python() -> str:
    """Find the Python executable to use for building."""
    # Use the Python running this script
    return sys.executable


def build_wheel(
    project_root: Path, verbose: bool = False
) -> Tuple[bool, str, Optional[Path]]:
    """
    Build a binary wheel from the project.

    Returns:
        (success, output, wheel_path)
    """
    print_step(1, "Building binary wheel")

    python = find_python()
    dist_dir = project_root / "dist"

    # Clean previous builds
    if dist_dir.exists():
        print_info("Cleaning previous dist/ directory...")
        shutil.rmtree(dist_dir)

    # Prepare sources (C + Cython)
    print_info("Preparing C sources and Cython...")
    success, output, duration = run_cmd(
        [python, "setup.py", "prepare"],
        cwd=project_root,
        verbose=verbose,
    )
    if not success:
        print_failure("Source preparation failed")
        if verbose:
            print_info(output[-500:])
        return False, output, None
    print_success(f"Source preparation complete ({duration:.1f}s)")

    # Install build dependencies
    print_info("Installing build dependencies...")
    success, output, duration = run_cmd(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "build",
            "setuptools",
            "wheel",
            "cython",
            "numpy",
        ],
        cwd=project_root,
        verbose=verbose,
    )
    if not success:
        print_failure("Build dependency installation failed")
        return False, output, None

    # Build the wheel
    print_info("Building wheel...")
    success, output, duration = run_cmd(
        [python, "-m", "build", "--wheel"],
        cwd=project_root,
        verbose=verbose,
        timeout=900,
    )

    if not success:
        print_failure(f"Wheel build failed ({duration:.1f}s)")
        if verbose:
            print_info(output[-1000:])
        return False, output, None

    print_success(f"Wheel built successfully ({duration:.1f}s)")

    # Find the wheel file
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        print_failure("No wheel file found in dist/")
        return False, "No wheel file found", None

    wheel_path = wheels[0]
    print_info(f"Wheel: {wheel_path.name}")
    print_info(f"Size: {wheel_path.stat().st_size / 1024 / 1024:.1f} MB")

    return True, output, wheel_path


def create_clean_venv(venv_path: Path, verbose: bool = False) -> Tuple[bool, str, str]:
    """
    Create a clean virtual environment.

    Returns:
        (success, output, python_path)
    """
    print_step(2, "Creating clean virtual environment")

    if venv_path.exists():
        print_info(f"Removing existing venv at {venv_path}...")
        shutil.rmtree(venv_path)

    print_info(f"Creating venv at {venv_path}...")
    try:
        venv.create(venv_path, with_pip=True, clear=True)
    except Exception as e:
        print_failure(f"Failed to create venv: {e}")
        return False, str(e), ""

    # Determine python path
    if sys.platform == "win32":
        python_path = str(venv_path / "Scripts" / "python.exe")
        pip_path = str(venv_path / "Scripts" / "pip.exe")
    else:
        python_path = str(venv_path / "bin" / "python")
        pip_path = str(venv_path / "bin" / "pip")

    if not os.path.exists(python_path):
        print_failure(f"Python not found at {python_path}")
        return False, "Python not found in venv", ""

    # Upgrade pip
    print_info("Upgrading pip in clean venv...")
    success, output, duration = run_cmd(
        [python_path, "-m", "pip", "install", "--upgrade", "pip"],
        verbose=verbose,
    )
    if not success:
        print_failure("pip upgrade failed")
        return False, output, ""

    print_success(f"Clean venv created ({duration:.1f}s)")
    return True, "", python_path


def install_wheel(
    wheel_path: Path,
    python_path: str,
    install_gui: bool = False,
    verbose: bool = False,
) -> Tuple[bool, str, float]:
    """
    Install the wheel in the clean venv.

    Returns:
        (success, output, duration)
    """
    print_step(3, "Installing wheel in clean environment")

    print_info(f"Installing: {wheel_path.name}")

    extras = "[gui]" if install_gui else ""
    cmd = [python_path, "-m", "pip", "install", f"{wheel_path}{extras}"]

    success, output, duration = run_cmd(
        cmd,
        verbose=verbose,
        timeout=600,
    )

    if not success:
        print_failure(f"Installation failed ({duration:.1f}s)")
        if verbose:
            print_info(output[-1000:])
        return False, output, duration

    print_success(f"Installation complete ({duration:.1f}s)")
    return True, output, duration


def run_import_tests(python_path: str, verbose: bool = False) -> list[dict]:
    """
    Run import verification tests against the installed package.

    Returns:
        List of test results.
    """
    print_step(4, "Running import verification tests")

    tests = [
        {
            "name": "openptv2 package import",
            "cmd": "import openptv2; print(f'openptv2 v{openptv2.__version__}')",
        },
        {
            "name": "optv Cython bindings import",
            "cmd": "from optv.tracking_framebuf import Target, TargetArray; print('optv.tracking_framebuf OK')",
        },
        {
            "name": "optv calibration import",
            "cmd": "from optv.calibration import Calibration; print('optv.calibration OK')",
        },
        {
            "name": "optv tracker import",
            "cmd": "from optv.tracker import Tracker; print('optv.tracker OK')",
        },
        {
            "name": "openptv2 engine selector",
            "cmd": "from openptv2.engine import EngineSelector, get_engine, set_engine; print('engine selector OK')",
        },
        {
            "name": "openptv2 unified imports",
            "cmd": "from openptv2 import Target, Tracker, Calibration, Correspondence; print('unified imports OK')",
        },
        {
            "name": "openptv2 version",
            "cmd": "import openptv2; print(openptv2.get_version())",
        },
        {
            "name": "openptv2 engine info",
            "cmd": "import openptv2; info = openptv2.get_engine_info(); print(info)",
        },
        {
            "name": "algorithms package import",
            "cmd": "import algorithms; print('algorithms package OK (lazy import)')",
        },
    ]

    results = []
    for test in tests:
        print_info(f"Testing: {test['name']}...")
        success, output, duration = run_cmd(
            [python_path, "-c", test["cmd"]],
            verbose=verbose,
        )

        result = {
            "name": test["name"],
            "passed": success,
            "output": output.strip()[-200:] if output else "",
            "duration": duration,
        }
        results.append(result)

        if success:
            print_success(f"{test['name']} ({duration:.2f}s)")
            if verbose and result["output"]:
                print_info(f"  Output: {result['output']}")
        else:
            print_failure(f"{test['name']} ({duration:.2f}s)")
            if verbose and result["output"]:
                print_info(f"  Error: {result['output'][-300:]}")

    return results


def run_installed_tests(
    python_path: str,
    project_root: Path,
    verbose: bool = False,
) -> Tuple[bool, str, float]:
    """
    Run the full test suite against the installed package.

    Returns:
        (success, output, duration)
    """
    print_step(5, "Running test suite against installed package")

    # Install test dependencies
    print_info("Installing test dependencies...")
    success, output, duration = run_cmd(
        [python_path, "-m", "pip", "install", "pytest", "pytest-cov"],
        verbose=verbose,
    )
    if not success:
        print_failure("Test dependency installation failed")
        return False, output, duration

    # Run tests from project root (uses installed package)
    print_info("Running pytest...")
    cmd = [
        python_path,
        "-m",
        "pytest",
        "bindings/tests/",
        "algorithms/tests/",
        "-v",
        "--tb=short",
    ]
    if not verbose:
        cmd.append("--tb=line")

    success, output, duration = run_cmd(
        cmd,
        cwd=project_root,
        verbose=verbose,
        timeout=600,
    )

    if success:
        print_success(f"Test suite passed ({duration:.1f}s)")
    else:
        print_failure(f"Test suite failed ({duration:.1f}s)")

    if verbose:
        print_info(output[-1000:] if len(output) > 1000 else output)

    return success, output, duration


def print_pipeline_summary(
    build_ok: bool,
    install_ok: bool,
    import_results: list[dict],
    test_ok: bool,
    wheel_path: Optional[Path],
    total_duration: float,
):
    """Print a summary of the entire pipeline."""
    print_header("Pipeline Summary")

    print_info(
        f"{'Wheel Build:':<30} {colorize('PASS', Colors.OKGREEN) if build_ok else colorize('FAIL', Colors.FAIL)}"
    )
    if wheel_path:
        print_info(f"{'Wheel File:':<30} {wheel_path.name}")

    print_info(
        f"{'Clean Install:':<30} {colorize('PASS', Colors.OKGREEN) if install_ok else colorize('FAIL', Colors.FAIL)}"
    )

    print()
    print_info("Import Tests:")
    import_pass = sum(1 for r in import_results if r["passed"])
    import_fail = sum(1 for r in import_results if not r["passed"])
    for r in import_results:
        status = (
            colorize("PASS", Colors.OKGREEN)
            if r["passed"]
            else colorize("FAIL", Colors.FAIL)
        )
        print_info(f"  {r['name']:<45} {status}")

    print()
    print_info(f"{'Import Tests:':<30} {import_pass} passed, {import_fail} failed")
    print_info(
        f"{'Full Test Suite:':<30} {colorize('PASS', Colors.OKGREEN) if test_ok else colorize('FAIL', Colors.FAIL)}"
    )
    print_info(f"{'Total Duration:':<30} {total_duration:.1f}s")

    print()
    all_ok = build_ok and install_ok and import_fail == 0 and test_ok
    if all_ok:
        print(colorize("ALL CHECKS PASSED!", Colors.OKGREEN))
    else:
        print(colorize("SOME CHECKS FAILED!", Colors.FAIL))

    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="openptv2 Wheel Build, Install & Test Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wheel_test_pipeline.py              # Full pipeline
  python wheel_test_pipeline.py --build-only # Only build wheel
  python wheel_test_pipeline.py --install-only dist/openptv2-*.whl  # Install & test existing wheel
  python wheel_test_pipeline.py --skip-build # Skip build, use latest wheel in dist/
  python wheel_test_pipeline.py --verbose    # Show detailed output
        """,
    )

    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the wheel, skip install and tests",
    )
    parser.add_argument(
        "--install-only",
        metavar="WHEEL",
        help="Skip build, install and test the specified wheel file",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip build, use the latest wheel in dist/",
    )
    parser.add_argument(
        "--with-gui",
        action="store_true",
        help="Install with GUI dependencies (traits, traitsui, PySide6, etc.)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the test virtual environment after completion",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory for the test venv (default: temp directory)",
    )

    args = parser.parse_args()

    # Find project root (scripts/ is a subdirectory of the project root)
    script_dir = Path(__file__).parent.absolute()
    if script_dir.name == "scripts":
        project_root = script_dir.parent
    else:
        project_root = script_dir
    start_time = datetime.now()

    # Determine wheel path
    wheel_path: Optional[Path] = None

    if args.install_only:
        wheel_path = Path(args.install_only)
        if not wheel_path.exists():
            print_failure(f"Wheel file not found: {wheel_path}")
            return 1
        build_ok = True
        print_info(f"Using existing wheel: {wheel_path.name}")
    elif args.skip_build:
        dist_dir = project_root / "dist"
        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            print_failure("No wheel found in dist/. Run with --build-only first.")
            return 1
        wheel_path = max(wheels, key=lambda p: p.stat().st_mtime)
        build_ok = True
        print_info(f"Using latest wheel: {wheel_path.name}")
    else:
        # Full build
        build_ok, _, wheel_path = build_wheel(project_root, verbose=args.verbose)
        if not build_ok:
            print_failure("Wheel build failed. Aborting pipeline.")
            return 1

    if args.build_only:
        print_success("Build-only mode complete.")
        return 0

    # Create clean venv
    if args.output_dir:
        venv_base = Path(args.output_dir)
    else:
        venv_base = Path(tempfile.mkdtemp(prefix="openptv2_test_"))

    venv_path = venv_base / "test_venv"

    venv_ok, _, python_path = create_clean_venv(venv_path, verbose=args.verbose)
    if not venv_ok:
        print_failure("Failed to create clean venv. Aborting.")
        return 1

    # Install wheel
    install_ok, _, install_duration = install_wheel(
        wheel_path,
        python_path,
        install_gui=args.with_gui,
        verbose=args.verbose,
    )
    if not install_ok:
        print_failure("Wheel installation failed. Aborting.")
        if not args.keep_venv:
            shutil.rmtree(venv_path, ignore_errors=True)
        return 1

    # Run import tests
    import_results = run_import_tests(python_path, verbose=args.verbose)

    # Run full test suite
    test_ok, _, test_duration = run_installed_tests(
        python_path,
        project_root,
        verbose=args.verbose,
    )

    # Cleanup
    if not args.keep_venv:
        print_info("Cleaning up test venv...")
        shutil.rmtree(venv_path, ignore_errors=True)

    # Summary
    total_duration = (datetime.now() - start_time).total_seconds()
    exit_code = print_pipeline_summary(
        build_ok=build_ok,
        install_ok=install_ok,
        import_results=import_results,
        test_ok=test_ok,
        wheel_path=wheel_path,
        total_duration=total_duration,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

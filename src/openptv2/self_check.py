"""Self-check and diagnostic module for openptv2.

Verifies Python environment, key dependencies, Cython compilation status,
core package imports, test data availability, and runtime validation.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import openptv2


@dataclass
class DiagnosticCheck:
    category: str
    name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    detail: str


class SelfChecker:
    """Diagnostic runner for openptv2 system self-checks."""

    def __init__(self, verbose: bool = False, strict: bool = False):
        self.verbose = verbose
        self.strict = strict
        self.checks: list[DiagnosticCheck] = []

    def log(self, category: str, name: str, status: str, detail: str) -> DiagnosticCheck:
        check = DiagnosticCheck(category=category, name=name, status=status, detail=detail)
        self.checks.append(check)
        return check

    def check_environment(self) -> None:
        """Check Python version and OS platform."""
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 9):
            self.log("Environment", "python_version", "PASS", f"Python {py_ver} ({sys.executable})")
        else:
            self.log("Environment", "python_version", "FAIL", f"Python {py_ver} (>= 3.9 required)")

        self.log("Environment", "platform", "PASS", f"{platform.system()} {platform.release()} ({platform.machine()})")
        self.log("Environment", "package_version", "PASS", f"openptv2 v{openptv2.get_version()}")

    def check_dependencies(self) -> None:
        """Check required and optional Python dependencies."""
        required = [
            ("numpy", "NumPy"),
            ("scipy", "SciPy"),
            ("Cython", "Cython"),
            ("zarr", "Zarr"),
            ("yaml", "PyYAML"),
            ("matplotlib", "Matplotlib"),
        ]

        for mod_name, label in required:
            try:
                mod = importlib.import_module(mod_name)
                ver = getattr(mod, "__version__", "unknown")
                self.log("Dependencies", f"dep_{mod_name}", "PASS", f"{label} v{ver}")
            except ImportError:
                self.log("Dependencies", f"dep_{mod_name}", "FAIL", f"Missing required dependency: {label}")

        optional = [
            ("numba", "Numba"),
            ("PySide6", "PySide6 (GUI)"),
            ("traitsui", "TraitsUI (GUI)"),
            ("marimo", "Marimo (Visualization)"),
            ("torch", "PyTorch (Differentiable PTV)"),
        ]

        for mod_name, label in optional:
            try:
                mod = importlib.import_module(mod_name)
                ver = getattr(mod, "__version__", "unknown")
                self.log("Dependencies", f"opt_{mod_name}", "PASS", f"{label} v{ver}")
            except ImportError:
                status = "FAIL" if self.strict else "WARN"
                self.log("Dependencies", f"opt_{mod_name}", status, f"Optional dependency not installed: {label}")

    def check_cython_compilation(self) -> None:
        """Check Cython compilation status across algorithms modules."""
        info = openptv2.get_runtime_info()
        is_comp = openptv2.is_compiled()

        if is_comp:
            self.log("Cython Runtime", "cython_compiled", "PASS", f"Engine: {info['engine']} (compiled)")
        else:
            status = "FAIL" if self.strict else "WARN"
            self.log(
                "Cython Runtime",
                "cython_compiled",
                status,
                f"Engine: {info['engine']} (running in pure Python fallback mode)",
            )

        # Inspect specific C-extension modules
        algo_modules = [
            ("openptv2.algorithms.calibration", "calibration"),
            ("openptv2.algorithms.correspondences", "correspondences"),
            ("openptv2.algorithms.epi", "epi"),
            ("openptv2.algorithms.imgcoord", "imgcoord"),
            ("openptv2.algorithms.multimed", "multimed"),
            ("openptv2.algorithms.orientation", "orientation"),
            ("openptv2.algorithms.parameters", "parameters"),
            ("openptv2.algorithms.segmentation", "segmentation"),
            ("openptv2.algorithms.track_kernels", "track_kernels"),
            ("openptv2.algorithms.trafo", "trafo"),
            ("openptv2.algorithms.vec_utils", "vec_utils"),
        ]

        compiled_count = 0
        for mod_path, mod_name in algo_modules:
            try:
                mod = importlib.import_module(mod_path)
                file_path = getattr(mod, "__file__", "") or ""
                # Compiled Cython extensions end in .so or .pyd
                is_ext = file_path.endswith((".so", ".pyd")) or not file_path.endswith(".py")
                if is_ext:
                    compiled_count += 1
                    if self.verbose:
                        self.log("Cython Modules", mod_name, "PASS", f"Compiled binary: {file_path}")
                else:
                    if self.verbose:
                        self.log("Cython Modules", mod_name, "WARN", f"Pure Python file: {file_path}")
            except Exception as e:
                self.log("Cython Modules", mod_name, "FAIL", f"Import error: {e}")

        self.log(
            "Cython Modules",
            "compiled_summary",
            "PASS" if compiled_count > 0 else ("FAIL" if self.strict else "WARN"),
            f"{compiled_count}/{len(algo_modules)} algorithms modules compiled as C-extensions",
        )

    def check_core_api(self) -> None:
        """Verify instantiation and basic functionality of core openptv2 objects."""
        try:
            openptv2.Calibration()
            cpar = openptv2.ControlParams(num_cams=4)
            cpar.set_image_size((1280, 1024))
            cpar.set_pixel_size((0.012, 0.012))
            openptv2.VolumeParams()
            openptv2.TargetParams()
            openptv2.MultimediaParams()
            openptv2.SequenceParams()
            openptv2.TrackingParams()
            self.log(
                "Core API",
                "param_instantiation",
                "PASS",
                "Successfully instantiated Calibration, ControlParams, VolumeParams, etc.",
            )
        except Exception as e:
            self.log("Core API", "param_instantiation", "FAIL", f"Failed to instantiate core objects: {e}")
            return

        try:
            import numpy as np

            pixels = np.array([[100.0, 200.0], [500.0, 400.0]], dtype=np.float64)
            metric = openptv2.convert_arr_pixel_to_metric(pixels, cpar)
            roundtrip = openptv2.convert_arr_metric_to_pixel(metric, cpar)
            max_diff = float(np.max(np.abs(pixels - roundtrip)))
            if max_diff < 1e-10:
                self.log("Core API", "pixel_metric_transforms", "PASS", f"Pixel-metric roundtrip max diff: {max_diff:.3e}")
            else:
                self.log("Core API", "pixel_metric_transforms", "FAIL", f"Roundtrip diff too large: {max_diff:.3e}")
        except Exception as e:
            self.log("Core API", "pixel_metric_transforms", "FAIL", f"Transform error: {e}")

    def check_test_data(self) -> Path | None:
        """Check test dataset availability."""
        pkg_root = Path(__file__).parent.parent.parent
        candidates = [
            pkg_root / "test_data" / "synthetic",
            Path.cwd() / "test_data" / "synthetic",
        ]

        test_data_dir = None
        for cand in candidates:
            if cand.exists() and (cand / "cal" / "cam1.tif.ori").exists():
                test_data_dir = cand
                break

        if test_data_dir:
            self.log("Test Data", "synthetic_fixtures", "PASS", f"Found test data at {test_data_dir}")
        else:
            self.log("Test Data", "synthetic_fixtures", "WARN", "Synthetic test data files not found in standard paths")

        return test_data_dir

    def check_runtime_validation(self) -> None:
        """Run synthetic runtime validation suite."""
        try:
            from openptv2.validate import run_validation_suite

            results = run_validation_suite(
                tolerance=1e-10,
                benchmark=False,
                iterations=1,
                min_speed_ratio=None,
                require_legacy_baseline=False,
            )

            failed = [r for r in results if r.failed]
            if not failed:
                self.log(
                    "Validation Suite",
                    "synthetic_suite",
                    "PASS",
                    f"Passed {len(results)} runtime validation checks",
                )
            else:
                details = "; ".join(f"{r.name}: {r.detail}" for r in failed)
                self.log(
                    "Validation Suite",
                    "synthetic_suite",
                    "FAIL",
                    f"{len(failed)} validation checks failed ({details})",
                )
        except Exception as e:
            self.log("Validation Suite", "synthetic_suite", "FAIL", f"Failed to execute validation suite: {e}")

    def run_all(self) -> list[DiagnosticCheck]:
        """Run all self-check diagnostic checks."""
        self.checks.clear()
        self.check_environment()
        self.check_dependencies()
        self.check_cython_compilation()
        self.check_core_api()
        self.check_test_data()
        self.check_runtime_validation()
        return self.checks

    def print_report(self, quiet: bool = False) -> int:
        """Print formatted report and return exit code (0 = success, 1 = failure)."""
        passed = [c for c in self.checks if c.status == "PASS"]
        failed = [c for c in self.checks if c.status == "FAIL"]
        warned = [c for c in self.checks if c.status == "WARN"]

        if not quiet:
            print("=" * 70)
            print(" openptv2 System Self-Check Diagnostic Report")
            print("=" * 70)

            current_category = ""
            for check in self.checks:
                if check.category != current_category:
                    current_category = check.category
                    print(f"\n[{current_category}]")

                status_fmt = f"[{check.status}]"
                if check.status == "PASS":
                    status_str = f"\033[32m{status_fmt:<7}\033[0m" if sys.stdout.isatty() else f"{status_fmt:<7}"
                elif check.status == "FAIL":
                    status_str = f"\033[31m{status_fmt:<7}\033[0m" if sys.stdout.isatty() else f"{status_fmt:<7}"
                elif check.status == "WARN":
                    status_str = f"\033[33m{status_fmt:<7}\033[0m" if sys.stdout.isatty() else f"{status_fmt:<7}"
                else:
                    status_str = f"{status_fmt:<7}"

                print(f"  {status_str} {check.name:<25} - {check.detail}")

            print("\n" + "-" * 70)
            print(
                f"Summary: {len(passed)} passed, {len(failed)} failed, {len(warned)} warnings (Total: {len(self.checks)})"
            )
            print("=" * 70)

        return 1 if len(failed) > 0 else 0


def main(args: Sequence[str] | None = None) -> int:
    """CLI entry point for self-check."""
    parser = argparse.ArgumentParser(
        prog="openptv self-check",
        description="Run comprehensive system self-checks and runtime diagnostics for openptv2.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed diagnostic output for all modules.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress detailed logs and only output errors/summary.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (e.g. uncompiled Cython, missing optional dependencies) as failures.",
    )

    parsed_args = parser.parse_args(args)

    checker = SelfChecker(verbose=parsed_args.verbose, strict=parsed_args.strict)
    checker.run_all()
    return checker.print_report(quiet=parsed_args.quiet)


if __name__ == "__main__":
    sys.exit(main())

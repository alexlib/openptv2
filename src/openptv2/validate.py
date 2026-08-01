#!/usr/bin/env python
"""Validation and benchmark tooling for the Cython 3 single-engine runtime."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import openptv2

TEST_DATA = Path(__file__).parent.parent.parent / "test_data" / "synthetic"


@dataclass
class CheckResult:
    """One validation or benchmark result."""

    name: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


def _legacy_modules() -> dict[str, Any] | None:
    try:
        return {
            "calibration": importlib.import_module("optv.calibration"),
            "epipolar": importlib.import_module("optv.epipolar"),
            "imgcoord": importlib.import_module("optv.imgcoord"),
            "orientation": importlib.import_module("optv.orientation"),
            "parameters": importlib.import_module("optv.parameters"),
            "segmentation": importlib.import_module("optv.segmentation"),
            "transforms": importlib.import_module("optv.transforms"),
        }
    except ImportError:
        return None


def _load_openptv_calibration(cam_num: int):
    cal = openptv2.Calibration()
    cal.from_file(
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.ori"),
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.addpar"),
    )
    return cal


def _load_legacy_calibration(legacy: dict[str, Any], cam_num: int):
    cal = legacy["calibration"].Calibration()
    cal.from_file(
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.ori"),
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.addpar"),
    )
    return cal


def _build_openptv_control(image_size=(1280, 1024), pixel_size=(0.012, 0.012)):
    cpar = openptv2.ControlParams(num_cams=4)
    cpar.set_image_size(image_size)
    cpar.set_pixel_size(pixel_size)
    return cpar


def _build_legacy_control(
    legacy: dict[str, Any],
    image_size=(1280, 1024),
    pixel_size=(0.012, 0.012),
):
    cpar = legacy["parameters"].ControlParams(4)
    cpar.set_image_size(image_size)
    cpar.set_pixel_size(pixel_size)
    return cpar


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs - rhs)))


def _compare_arrays(
    name: str,
    lhs: np.ndarray,
    rhs: np.ndarray,
    tolerance: float,
) -> CheckResult:
    if lhs.shape != rhs.shape:
        return CheckResult(name, "FAIL", f"shape mismatch: {lhs.shape} != {rhs.shape}")

    if not np.allclose(lhs, rhs, rtol=tolerance, atol=tolerance):
        return CheckResult(
            name,
            "FAIL",
            (
                f"max abs diff {_max_abs_diff(lhs, rhs):.3e} "
                f"exceeds tolerance {tolerance:.3e}"
            ),
        )

    return CheckResult(name, "PASS", f"max abs diff {_max_abs_diff(lhs, rhs):.3e}")


def validate_runtime_info() -> CheckResult:
    info = openptv2.get_runtime_info()
    if info["engine"] != "cython3-pure-python":
        return CheckResult(
            "runtime_info",
            "FAIL",
            f"unexpected engine {info['engine']}",
        )
    if not isinstance(info["compiled"], bool):
        return CheckResult("runtime_info", "FAIL", "compiled flag is not boolean")
    return CheckResult(
        "runtime_info",
        "PASS",
        f"compiled={info['compiled']}, package={info['package']}",
    )


def validate_transforms(tolerance: float, legacy: dict[str, Any] | None) -> CheckResult:
    pixels = np.array(
        [[640.0, 512.0], [100.25, 200.75], [1100.5, 900.1]],
        dtype=np.float64,
    )
    cpar = _build_openptv_control()
    metric = openptv2.convert_arr_pixel_to_metric(pixels, cpar)
    roundtrip = openptv2.convert_arr_metric_to_pixel(metric, cpar)
    roundtrip_result = _compare_arrays("transforms", roundtrip, pixels, tolerance)
    if roundtrip_result.failed:
        return roundtrip_result

    if legacy is None:
        return CheckResult(
            "transforms",
            "PASS",
            f"roundtrip max abs diff {_max_abs_diff(roundtrip, pixels):.3e}",
        )

    legacy_cpar = _build_legacy_control(legacy)
    legacy_metric = legacy["transforms"].convert_arr_pixel_to_metric(
        pixels,
        legacy_cpar,
    )
    return _compare_arrays("transforms", metric, legacy_metric, tolerance)


def validate_image_coordinates(
    tolerance: float,
    legacy: dict[str, Any] | None,
) -> CheckResult:
    cal = _load_openptv_calibration(1)
    mm = openptv2.MultimediaParams(n1=1.0, n3=1.0)
    positions = np.array(
        [[0.0, 0.0, 100.0], [10.0, -15.0, 120.0], [-25.0, 30.0, 80.0]],
        dtype=np.float64,
    )
    coords = openptv2.image_coordinates(positions, cal, mm)
    if legacy is None:
        if coords.shape != (3, 2) or not np.isfinite(coords).all():
            return CheckResult(
                "image_coordinates",
                "FAIL",
                "non-finite or unexpected shape",
            )
        return CheckResult("image_coordinates", "PASS", "finite coordinates produced")

    legacy_cal = _load_legacy_calibration(legacy, 1)
    legacy_mm = legacy["parameters"].MultimediaParams(n1=1.0, n3=1.0)
    legacy_coords = legacy["imgcoord"].image_coordinates(
        positions,
        legacy_cal,
        legacy_mm,
    )
    return _compare_arrays("image_coordinates", coords, legacy_coords, tolerance)


def validate_epipolar_curve(
    tolerance: float,
    legacy: dict[str, Any] | None,
) -> CheckResult:
    cal1 = _load_openptv_calibration(1)
    cal2 = _load_openptv_calibration(2)
    cpar = openptv2.ControlParams(num_cams=4)
    cpar.read_control_par(str(TEST_DATA / "parameters" / "ptv.par"))
    vpar = openptv2.VolumeParams()
    vpar.read_volume_par(str(TEST_DATA / "parameters" / "criteria.par"))
    point = np.array([640.0, 512.0], dtype=np.float64)
    curve = openptv2.epipolar_curve(point, cal1, cal2, 10, cpar, vpar)

    if legacy is None:
        if curve.shape != (10, 2) or not np.isfinite(curve).all():
            return CheckResult(
                "epipolar_curve",
                "FAIL",
                "non-finite or unexpected shape",
            )
        return CheckResult("epipolar_curve", "PASS", "finite curve produced")

    legacy_cal1 = _load_legacy_calibration(legacy, 1)
    legacy_cal2 = _load_legacy_calibration(legacy, 2)
    legacy_cpar = legacy["parameters"].ControlParams(4)
    legacy_cpar.read_control_par(str(TEST_DATA / "parameters" / "ptv.par"))
    legacy_vpar = legacy["parameters"].VolumeParams()
    legacy_vpar.read_volume_par(str(TEST_DATA / "parameters" / "criteria.par"))
    legacy_curve = legacy["epipolar"].epipolar_curve(
        point,
        legacy_cal1,
        legacy_cal2,
        10,
        legacy_cpar,
        legacy_vpar,
    )
    return _compare_arrays("epipolar_curve", curve, legacy_curve, tolerance)


def validate_segmentation(
    tolerance: float,
    legacy: dict[str, Any] | None,
) -> CheckResult:
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

    if legacy is None:
        return CheckResult(
            "segmentation",
            "PASS" if len(targets) >= 2 else "FAIL",
            f"found {len(targets)} targets",
        )

    legacy_cpar = legacy["parameters"].ControlParams(4)
    legacy_cpar.set_image_size((256, 256))
    legacy_tpar = legacy["parameters"].TargetParams(
        discont=10,
        gvthresh=[50, 50, 50, 50],
        pixel_count_bounds=(5, 100),
        xsize_bounds=(2, 20),
        ysize_bounds=(2, 20),
        min_sum_grey=100,
    )
    legacy_targets = legacy["segmentation"].target_recognition(
        image,
        legacy_tpar,
        0,
        legacy_cpar,
    )

    if len(targets) != len(legacy_targets):
        return CheckResult(
            "segmentation",
            "FAIL",
            f"target count mismatch: {len(targets)} != {len(legacy_targets)}",
        )

    for index in range(len(targets)):
        current = np.asarray(targets[index].pos(), dtype=np.float64)
        reference = np.asarray(legacy_targets[index].pos(), dtype=np.float64)
        if not np.allclose(current, reference, rtol=tolerance, atol=tolerance):
            return CheckResult(
                "segmentation",
                "FAIL",
                f"target {index} max abs diff {_max_abs_diff(current, reference):.3e}",
            )

    return CheckResult("segmentation", "PASS", f"matched {len(targets)} targets")


def validate_point_positions(
    tolerance: float,
    legacy: dict[str, Any] | None,
) -> CheckResult:
    cals = [_load_openptv_calibration(i) for i in range(1, 5)]
    cpar = _build_openptv_control()
    rng = np.random.default_rng(42)
    positions_3d = rng.uniform(-20, 20, (10, 3))
    positions_3d[:, 2] = rng.uniform(60, 120, 10)

    targets = np.empty((10, 4, 2), dtype=np.float64)
    for cam in range(4):
        targets[:, cam, :] = openptv2.image_coordinates(
            positions_3d, cals[cam], cpar.get_multimedia_params()
        )

    positions, rcm = openptv2.multi_cam_point_positions(targets, cpar, cals)

    if legacy is None:
        if positions.shape != (10, 3) or rcm.shape != (10,):
            return CheckResult(
                "point_positions",
                "FAIL",
                "unexpected shape",
            )
        return CheckResult("point_positions", "PASS", "finite positions produced")

    legacy_cals = [_load_legacy_calibration(legacy, i) for i in range(1, 5)]
    legacy_cpar = _build_legacy_control(legacy)
    legacy_cpar.get_multimedia_params().set_layers(
        np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])
    )
    legacy_positions, legacy_rcm = legacy["orientation"].multi_cam_point_positions(
        targets,
        legacy_cpar,
        legacy_cals,
    )

    pos_check = _compare_arrays(
        "point_positions_pos", positions, legacy_positions, tolerance
    )
    if pos_check.failed:
        return CheckResult(
            "point_positions", "FAIL", f"Positions mismatch: {pos_check.detail}"
        )

    rcm_check = _compare_arrays("point_positions_rcm", rcm, legacy_rcm, tolerance)
    if rcm_check.failed:
        return CheckResult(
            "point_positions", "FAIL", f"RCM mismatch: {rcm_check.detail}"
        )

    max_diff = max(
        _max_abs_diff(positions, legacy_positions), _max_abs_diff(rcm, legacy_rcm)
    )
    return CheckResult("point_positions", "PASS", f"max abs diff {max_diff:.3e}")


def _benchmark_operation(func, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - start) / iterations


def benchmark_against_legacy(
    iterations: int,
    legacy: dict[str, Any] | None,
    min_speed_ratio: float | None,
) -> CheckResult:
    if legacy is None:
        return CheckResult(
            "speed_benchmark",
            "SKIP",
            "legacy optv baseline not importable",
        )

    cpar = _build_openptv_control()
    legacy_cpar = _build_legacy_control(legacy)
    legacy_cpar.get_multimedia_params().set_layers(
        np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])
    )
    pixels = np.random.default_rng(42).random((4096, 2), dtype=np.float64)
    pixels[:, 0] *= 1280.0
    pixels[:, 1] *= 1024.0

    cal = _load_openptv_calibration(1)
    legacy_cal = _load_legacy_calibration(legacy, 1)
    mm = openptv2.MultimediaParams(n1=1.0, n3=1.0)
    legacy_mm = legacy["parameters"].MultimediaParams(n1=1.0, n3=1.0)
    positions = np.random.default_rng(7).random((2048, 3), dtype=np.float64)
    positions[:, 0] = positions[:, 0] * 80.0 - 40.0
    positions[:, 1] = positions[:, 1] * 80.0 - 40.0
    positions[:, 2] = positions[:, 2] * 80.0 + 60.0

    cals = [_load_openptv_calibration(i) for i in range(1, 5)]
    legacy_cals = [_load_legacy_calibration(legacy, i) for i in range(1, 5)]

    openptv_time = _benchmark_operation(
        lambda: openptv2.convert_arr_pixel_to_metric(pixels, cpar),
        iterations,
    )
    legacy_time = _benchmark_operation(
        lambda: legacy["transforms"].convert_arr_pixel_to_metric(pixels, legacy_cpar),
        iterations,
    )
    openptv_imgcoord = _benchmark_operation(
        lambda: openptv2.image_coordinates(positions, cal, mm),
        iterations,
    )
    legacy_imgcoord = _benchmark_operation(
        lambda: legacy["imgcoord"].image_coordinates(positions, legacy_cal, legacy_mm),
        iterations,
    )

    bench_positions_3d = np.random.default_rng(42).uniform(-20, 20, (512, 3))
    bench_targets = np.empty((512, 4, 2), dtype=np.float64)
    for cam in range(4):
        bench_targets[:, cam, :] = openptv2.image_coordinates(
            bench_positions_3d, cals[cam], mm
        )

    openptv_pointpos = _benchmark_operation(
        lambda: openptv2.multi_cam_point_positions(bench_targets, cpar, cals),
        iterations,
    )
    legacy_pointpos = _benchmark_operation(
        lambda: legacy["orientation"].multi_cam_point_positions(
            bench_targets,
            legacy_cpar,
            legacy_cals,
        ),
        iterations,
    )

    ratio_transform = legacy_time / openptv_time if openptv_time > 0 else 0.0
    ratio_imgcoord = legacy_imgcoord / openptv_imgcoord if openptv_imgcoord > 0 else 0.0
    ratio_pointpos = legacy_pointpos / openptv_pointpos if openptv_pointpos > 0 else 0.0
    min_ratio = min(ratio_transform, ratio_imgcoord, ratio_pointpos)
    detail = (
        f"pixel_to_metric ratio={ratio_transform:.3f}, "
        f"image_coordinates ratio={ratio_imgcoord:.3f}, "
        f"point_positions ratio={ratio_pointpos:.3f}"
    )

    if min_speed_ratio is None:
        return CheckResult("speed_benchmark", "PASS", detail)

    status = "PASS" if min_ratio >= min_speed_ratio else "FAIL"
    return CheckResult(
        "speed_benchmark",
        status,
        f"{detail}, required>={min_speed_ratio:.3f}",
    )


def run_validation_suite(
    tolerance: float,
    benchmark: bool,
    iterations: int,
    min_speed_ratio: float | None,
    require_legacy_baseline: bool,
) -> list[CheckResult]:
    legacy = _legacy_modules()
    results = [validate_runtime_info()]

    if require_legacy_baseline and legacy is None:
        results.append(
            CheckResult(
                "legacy_baseline",
                "FAIL",
                "optv baseline is required but not importable",
            )
        )
        return results

    results.extend(
        [
            validate_transforms(tolerance, legacy),
            validate_image_coordinates(tolerance, legacy),
            validate_epipolar_curve(tolerance, legacy),
            validate_segmentation(tolerance, legacy),
            validate_point_positions(tolerance, legacy),
        ]
    )

    if benchmark:
        results.append(benchmark_against_legacy(iterations, legacy, min_speed_ratio))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate floating-point accuracy and benchmark the single-engine runtime"
        )
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=1e-10,
        help="Floating-point tolerance used for parity comparisons.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run speed comparisons against the legacy optv baseline when available.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Benchmark iterations per operation.",
    )
    parser.add_argument(
        "--min-speed-ratio",
        type=float,
        default=None,
        help=(
            "Require legacy_time/openptv2_time to meet this minimum ratio during "
            "benchmarking. Values >= 1.0 mean openptv2 must be at least as fast."
        ),
    )
    parser.add_argument(
        "--require-legacy-baseline",
        action="store_true",
        help="Fail if the legacy optv baseline cannot be imported.",
    )
    args = parser.parse_args()

    results = run_validation_suite(
        tolerance=args.tolerance,
        benchmark=args.benchmark,
        iterations=args.iterations,
        min_speed_ratio=args.min_speed_ratio,
        require_legacy_baseline=args.require_legacy_baseline,
    )

    failed = 0
    for result in results:
        print(f"{result.name}: {result.status} - {result.detail}")
        failed += int(result.failed)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

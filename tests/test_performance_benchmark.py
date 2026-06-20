"""
OpenPTV2 speed performance benchmark test.
Compares native C/Cython modules (optv) against optimized Cython 3 Pure Python modules (algorithms).
"""

import os
import time
import numpy as np
from pathlib import Path

# Explicit imports for both engines
import optv.calibration as optv_cal
import optv.parameters as optv_params
import optv.transforms as optv_transforms
import optv.imgcoord as optv_imgcoord
import optv.epipolar as optv_epipolar

import algorithms.compat.calibration as python_cal
import algorithms.compat.parameters as python_params
import algorithms.compat.transforms as python_transforms
import algorithms.compat.imgcoord as python_imgcoord
import algorithms.compat.epipolar as python_epipolar

TEST_DATA = Path(__file__).parent.parent / "test_data" / "synthetic"

def run_benchmarks():
    print("======================================================================")
    print("                    OPENPTV2 SPEED PERFORMANCE BENCHMARK              ")
    print("        C/Cython (optv) vs. Cython 3 Pure Python (algorithms)         ")
    print("======================================================================")

    # 1. Initialize Params and Calibrations for both engines
    print("\n[Init] Setting up test parameters and inputs...")
    
    # Files
    ori_file = str(TEST_DATA / "cal" / "cam1.tif.ori")
    add_file = str(TEST_DATA / "cal" / "cam1.tif.addpar")
    ptv_par = str(TEST_DATA / "parameters" / "ptv.par")
    crit_par = str(TEST_DATA / "parameters" / "criteria.par")
    
    # OPTV objects
    c_cal = optv_cal.Calibration()
    c_cal.from_file(ori_file, add_file)
    c_cpar = optv_params.ControlParams(num_cams=4)
    c_cpar.read_control_par(ptv_par)
    c_vpar = optv_params.VolumeParams()
    c_vpar.read_volume_par(crit_par)
    c_mm = optv_params.MultimediaParams(n1=1.0, n3=1.0)
    
    # Python objects
    p_cal = python_cal.Calibration()
    p_cal.from_file(ori_file, add_file)
    p_cpar = python_params.ControlParams(num_cams=4)
    p_cpar.read_control_par(ptv_par)
    p_vpar = python_params.VolumeParams()
    p_vpar.read_volume_par(crit_par)
    p_mm = python_params.MultimediaParams(n1=1.0, n3=1.0)

    # Input sizes
    N = 100000 # 100k points for solid stats
    np.random.seed(42)
    pixels = np.random.rand(N, 2) * 1000.0
    metric = np.random.rand(N, 2) * 100.0 - 50.0
    pos3d = np.random.rand(N, 3) * 200.0 - 100.0
    
    results = {}
    
    # Helper to measure and print
    def run_test(name, optv_func, optv_args, python_func, python_args, iterations=1):
        # Warmup
        optv_func(*optv_args)
        python_func(*python_args)
        
        # Benchmark OPTV
        t0 = time.perf_counter()
        for _ in range(iterations):
            optv_func(*optv_args)
        t_optv = (time.perf_counter() - t0) / iterations
        
        # Benchmark Python
        t0 = time.perf_counter()
        for _ in range(iterations):
            python_func(*python_args)
        t_python = (time.perf_counter() - t0) / iterations
        
        speedup = t_optv / t_python if t_python > 0 else 0.0
        results[name] = {
            "optv_time_ms": t_optv * 1000,
            "python_time_ms": t_python * 1000,
            "speedup": speedup
        }
        print(f"  {name:<40}: OPTV = {t_optv*1000:7.3f} ms | Python = {t_python*1000:7.3f} ms | Speedup = {speedup:5.2f}x")

    # Benchmarks
    print("\nRunning benchmarks...")
    
    run_test(
        "convert_arr_pixel_to_metric (100k)",
        optv_transforms.convert_arr_pixel_to_metric, (pixels, c_cpar),
        python_transforms.convert_arr_pixel_to_metric, (pixels, p_cpar),
        iterations=5
    )

    run_test(
        "convert_arr_metric_to_pixel (100k)",
        optv_transforms.convert_arr_metric_to_pixel, (metric, c_cpar),
        python_transforms.convert_arr_metric_to_pixel, (metric, p_cpar),
        iterations=5
    )

    run_test(
        "correct_arr_brown_affine (100k)",
        optv_transforms.correct_arr_brown_affine, (pixels, c_cal),
        python_transforms.correct_arr_brown_affine, (pixels, p_cal),
        iterations=5
    )

    run_test(
        "distort_arr_brown_affine (100k)",
        optv_transforms.distort_arr_brown_affine, (metric, c_cal),
        python_transforms.distort_arr_brown_affine, (metric, p_cal),
        iterations=5
    )

    run_test(
        "image_coordinates (100k)",
        optv_imgcoord.image_coordinates, (pos3d, c_cal, c_mm),
        python_imgcoord.image_coordinates, (pos3d, p_cal, p_mm),
        iterations=5
    )

    run_test(
        "flat_image_coordinates (100k)",
        optv_imgcoord.flat_image_coordinates, (pos3d, c_cal, c_mm),
        python_imgcoord.flat_image_coordinates, (pos3d, p_cal, p_mm),
        iterations=5
    )
    
    # epipolar_curve is a single-point curve generator, run 200 times for robust benchmark
    curve_pt = np.array([500.0, 500.0])
    cal2_c = optv_cal.Calibration()
    cal2_c.from_file(str(TEST_DATA / "cal" / "cam2.tif.ori"), str(TEST_DATA / "cal" / "cam2.tif.addpar"))
    cal2_p = python_cal.Calibration()
    cal2_p.from_file(str(TEST_DATA / "cal" / "cam2.tif.ori"), str(TEST_DATA / "cal" / "cam2.tif.addpar"))
    
    def run_epi_c():
        for _ in range(200):
            optv_epipolar.epipolar_curve(curve_pt, c_cal, cal2_c, 20, c_cpar, c_vpar)
            
    def run_epi_p():
        for _ in range(200):
            python_epipolar.epipolar_curve(curve_pt, p_cal, cal2_p, 20, p_cpar, p_vpar)

    run_test(
        "epipolar_curve (200 iterations)",
        run_epi_c, (),
        run_epi_p, (),
        iterations=3
    )

    # Print Table in Markdown format
    print("\n\n" + "="*80)
    print("                     BENCHMARK SPEED COMPARISON TABLE")
    print("="*80)
    print(f"| {'Benchmark Task':<40} | {'C/Cython (optv)':<15} | {'Cython 3 Pure Python':<22} | {'Ratio (C/Python)':<15} |")
    print("|" + "-"*42 + "|" + "-"*17 + "|" + "-"*24 + "|" + "-"*17 + "|")
    for name, r in results.items():
        print(f"| {name:<40} | {r['optv_time_ms']:12.2f} ms | {r['python_time_ms']:19.2f} ms | {r['speedup']:14.2f}x |")
    print("="*80)

if __name__ == "__main__":
    run_benchmarks()

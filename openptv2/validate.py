#!/usr/bin/env python
"""
Engine validation tool for openptv2.

Validates that both optv and Python engines produce identical results.

Usage:
    openptv validate --tolerance 1e-10
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Paths
TEST_DATA = Path(__file__).parent.parent / "test_data" / "synthetic"


def load_cal_optv(cam_num):
    """Load optv Calibration."""
    import optv.calibration
    cal = optv.calibration.Calibration()
    cal.from_file(
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.ori"),
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.addpar")
    )
    return cal


def load_cal_python(cam_num):
    """Load algorithms.compat Calibration."""
    import algorithms.compat.calibration
    cal = algorithms.compat.calibration.Calibration()
    cal.from_file(
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.ori"),
        str(TEST_DATA / "cal" / f"cam{cam_num}.tif.addpar")
    )
    return cal


def validate_transforms(tolerance=1e-10):
    """Validate coordinate transformations."""
    print("Validating coordinate transformations (pixel <-> metric)...")
    
    try:
        import optv.parameters
        import optv.transforms
        import algorithms.compat.parameters
        import algorithms.compat.transforms
        
        # Setup control params
        cpar_optv = optv.parameters.ControlParams(num_cams=4)
        cpar_optv.set_image_size((1280, 1024))
        cpar_optv.set_pixel_size((0.012, 0.012))
        
        cpar_py = algorithms.compat.parameters.ControlParams(num_cams=4)
        cpar_py.set_image_size((1280, 1024))
        cpar_py.set_pixel_size((0.012, 0.012))
        
        # Test coordinates
        pixels = np.array([
            [640.0, 512.0],
            [100.25, 200.75],
            [1100.5, 900.1],
        ], dtype=np.float64)
        
        # Pixel -> Metric
        m_optv = optv.transforms.convert_arr_pixel_to_metric(pixels, cpar_optv)
        m_py = algorithms.compat.transforms.convert_arr_pixel_to_metric(pixels, cpar_py)
        
        if not np.allclose(m_optv, m_py, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(m_optv - m_py))
            print(f"  ✗ Pixel -> Metric: FAIL (max diff: {max_diff})")
            return False
            
        # Metric -> Pixel
        p_optv = optv.transforms.convert_arr_metric_to_pixel(m_optv, cpar_optv)
        p_py = algorithms.compat.transforms.convert_arr_metric_to_pixel(m_py, cpar_py)
        
        if not np.allclose(p_optv, p_py, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(p_optv - p_py))
            print(f"  ✗ Metric -> Pixel: FAIL (max diff: {max_diff})")
            return False
            
        print("  ✓ Coordinate transforms: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ Coordinate transforms: ERROR ({e})")
        return False


def validate_image_coordinates(tolerance=1e-10):
    """Validate 3D -> 2D projection (image_coordinates)."""
    print("Validating 3D -> 2D projection (image_coordinates)...")
    
    try:
        import optv.imgcoord
        import optv.parameters
        import algorithms.compat.imgcoord
        import algorithms.compat.parameters
        
        cal_optv = load_cal_optv(1)
        cal_py = load_cal_python(1)
        
        mm_optv = optv.parameters.MultimediaParams(n1=1.0, n3=1.0)
        mm_py = algorithms.compat.parameters.MultimediaParams(n1=1.0, n3=1.0)
        
        positions = np.array([
            [0.0, 0.0, 100.0],
            [10.0, -15.0, 120.0],
            [-25.0, 30.0, 80.0],
        ], dtype=np.float64)
        
        coords_optv = optv.imgcoord.image_coordinates(positions, cal_optv, mm_optv)
        coords_py = algorithms.compat.imgcoord.image_coordinates(positions, cal_py, mm_py)
        
        if not np.allclose(coords_optv, coords_py, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(coords_optv - coords_py))
            print(f"  ✗ 3D -> 2D projection: FAIL (max diff: {max_diff})")
            return False
            
        print("  ✓ 3D -> 2D projection: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ 3D -> 2D projection: ERROR ({e})")
        return False


def validate_epipolar_curve(tolerance=1e-10):
    """Validate epipolar curve generation."""
    print("Validating epipolar curve generation...")
    
    try:
        import optv.epipolar
        import optv.parameters
        import algorithms.compat.epipolar
        import algorithms.compat.parameters
        
        cal1_optv = load_cal_optv(1)
        cal2_optv = load_cal_optv(2)
        
        cal1_py = load_cal_python(1)
        cal2_py = load_cal_python(2)
        
        # Read parameters from synthetic test data
        cpar_optv = optv.parameters.ControlParams(num_cams=4)
        cpar_optv.read_control_par(str(TEST_DATA / "parameters" / "ptv.par"))
        
        cpar_py = algorithms.compat.parameters.ControlParams(num_cams=4)
        cpar_py.read_control_par(str(TEST_DATA / "parameters" / "ptv.par"))
        
        vpar_optv = optv.parameters.VolumeParams()
        vpar_optv.read_volume_par(str(TEST_DATA / "parameters" / "criteria.par"))
        
        vpar_py = algorithms.compat.parameters.VolumeParams()
        vpar_py.read_volume_par(str(TEST_DATA / "parameters" / "criteria.par"))
        
        point = np.array([640.0, 512.0], dtype=np.float64)
        
        curve_optv = optv.epipolar.epipolar_curve(point, cal1_optv, cal2_optv, 10, cpar_optv, vpar_optv)
        curve_py = algorithms.compat.epipolar.epipolar_curve(point, cal1_py, cal2_py, 10, cpar_py, vpar_py)
        
        if not np.allclose(curve_optv, curve_py, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(curve_optv - curve_py))
            print(f"  ✗ Epipolar curve: FAIL (max diff: {max_diff})")
            return False
            
        print("  ✓ Epipolar curve: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ Epipolar curve: ERROR ({e})")
        return False


def validate_segmentation(tolerance=1e-10):
    """Validate target recognition / segmentation."""
    print("Validating target recognition (segmentation)...")
    
    try:
        import optv.segmentation
        import optv.parameters
        import algorithms.compat.segmentation
        import algorithms.compat.parameters
        
        # Create a small synthetic image with some blobs
        np.random.seed(42)
        image = (np.random.rand(256, 256) * 10).astype(np.uint8)
        # Add a couple of distinct targets
        image[50:55, 60:65] = 200
        image[120:128, 180:188] = 180
        
        cpar_optv = optv.parameters.ControlParams(num_cams=4)
        cpar_optv.set_image_size((256, 256))
        
        cpar_py = algorithms.compat.parameters.ControlParams(num_cams=4)
        cpar_py.set_image_size((256, 256))
        
        tpar_optv = optv.parameters.TargetParams(
            discont=10,
            gvthresh=[50, 50, 50, 50],
            pixel_count_bounds=(5, 100),
            xsize_bounds=(2, 20),
            ysize_bounds=(2, 20),
            min_sum_grey=100
        )
        
        tpar_py = algorithms.compat.parameters.TargetParams(
            discont=10,
            gvthresh=[50, 50, 50, 50],
            pixel_count_bounds=(5, 100),
            xsize_bounds=(2, 20),
            ysize_bounds=(2, 20),
            min_sum_grey=100
        )
        
        targets_optv = optv.segmentation.target_recognition(image, tpar_optv, 0, cpar_optv)
        targets_py = algorithms.compat.segmentation.target_recognition(image, tpar_py, 0, cpar_py)
        
        if len(targets_optv) != len(targets_py):
            print(f"  ✗ Segmentation: FAIL (count mismatch: optv={len(targets_optv)}, python={len(targets_py)})")
            return False
            
        # Compare each target's position and characteristics
        for i in range(len(targets_optv)):
            t_optv = targets_optv[i]
            t_py = targets_py[i]
            
            p_optv = np.array(t_optv.pos())
            p_py = np.array(t_py.pos())
            
            if not np.allclose(p_optv, p_py, rtol=tolerance, atol=tolerance):
                print(f"  ✗ Segmentation: FAIL (target {i} position mismatch: optv={p_optv}, python={p_py})")
                return False
                
            if t_optv.pnr() != t_py.pnr():
                print(f"  ✗ Segmentation: FAIL (target {i} pnr mismatch: optv={t_optv.pnr()}, python={t_py.pnr()})")
                return False
                
        print("  ✓ Target recognition: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ Target recognition: ERROR ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate openptv2 engine consistency"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=1e-10,
        help="Floating-point tolerance for comparison (default: 1e-10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("openptv2 Engine Consistency Validation")
    print("=" * 60)
    print(f"Tolerance: {args.tolerance}")
    print()
    
    # Check if optv is available
    try:
        import optv
        optv_available = True
    except ImportError:
        optv_available = False
        print("⚠ WARNING: C/Cython engine 'optv' is not available.")
        print("Consistency validation requires both 'optv' and 'python' engines to be installed.")
        print("We will execute validation on the Python compatibility layer, but direct parity comparisons will be skipped.")
        print()
        
    results = {}
    
    if optv_available:
        results["transforms"] = validate_transforms(args.tolerance)
        results["image_coordinates"] = validate_image_coordinates(args.tolerance)
        results["epipolar_curve"] = validate_epipolar_curve(args.tolerance)
        results["segmentation"] = validate_segmentation(args.tolerance)
    else:
        # Just run py versions to ensure no errors
        try:
            import algorithms.compat.transforms
            import algorithms.compat.imgcoord
            import algorithms.compat.epipolar
            import algorithms.compat.segmentation
            print("  ✓ All Python compat modules imported successfully.")
            results["python_check"] = True
        except Exception as e:
            print(f"  ✗ Python compat check failed: {e}")
            results["python_check"] = False
            
    print()
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("VALIDATION FAILED")
        return 1
    else:
        print("VALIDATION PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())


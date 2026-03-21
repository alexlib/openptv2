#!/usr/bin/env python
"""
Engine validation tool for openptv2.

Validates that both optv and Python engines produce identical results.

Usage:
    openptv2-validate parameters.yaml --tolerance 1e-10
"""

import argparse
import sys
import numpy as np


def validate_tracking(tolerance=1e-10):
    """Validate tracking algorithm."""
    print("Validating tracking algorithm...")
    
    # Create test data
    np.random.seed(42)
    prev_targets = np.random.rand(100, 2) * 100
    curr_targets = prev_targets + np.random.rand(100, 2) * 5
    
    try:
        from openptv2 import Tracker
        
        # Run with both engines
        tracker_optv = Tracker(engine="optv")
        tracker_python = Tracker(engine="python")
        
        result_optv = tracker_optv.track_frame(prev_targets, curr_targets)
        result_python = tracker_python.track_frame(prev_targets, curr_targets)
        
        # Compare
        if np.allclose(result_optv, result_python, rtol=tolerance, atol=tolerance):
            print("  ✓ Tracking: PASS")
            return True
        else:
            max_diff = np.max(np.abs(result_optv - result_python))
            print(f"  ✗ Tracking: FAIL (max diff: {max_diff})")
            return False
            
    except Exception as e:
        print(f"  ✗ Tracking: ERROR ({e})")
        return False


def validate_detection(tolerance=1e-10):
    """Validate detection algorithm."""
    print("Validating detection algorithm...")
    
    # Create test image
    np.random.seed(42)
    image = np.random.rand(512, 512) * 100
    # Add some particles
    image[100:105, 200:205] = 200
    image[300:308, 400:408] = 180
    
    try:
        from openptv2 import detect_targets
        
        result_optv = detect_targets(image, engine="optv")
        result_python = detect_targets(image, engine="python")
        
        # Compare coordinates
        if result_optv.coordinates is None and result_python.coordinates is None:
            print("  ✓ Detection: PASS (no targets)")
            return True
        
        if result_optv.coordinates is None or result_python.coordinates is None:
            print("  ✗ Detection: FAIL (one engine found no targets)")
            return False
        
        if np.allclose(result_optv.coordinates, result_python.coordinates, 
                      rtol=tolerance, atol=tolerance):
            print("  ✓ Detection: PASS")
            return True
        else:
            max_diff = np.max(np.abs(result_optv.coordinates - result_python.coordinates))
            print(f"  ✗ Detection: FAIL (max diff: {max_diff})")
            return False
            
    except Exception as e:
        print(f"  ✗ Detection: ERROR ({e})")
        return False


def validate_correspondence(tolerance=1e-10):
    """Validate correspondence algorithm."""
    print("Validating correspondence algorithm...")
    
    # Skip for now - requires calibration
    print("  ⊘ Correspondence: SKIPPED (requires calibration)")
    return None


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
    print("openptv2 Engine Validation")
    print("=" * 60)
    print(f"Tolerance: {args.tolerance}")
    print()
    
    results = {
        "tracking": validate_tracking(args.tolerance),
        "detection": validate_detection(args.tolerance),
        "correspondence": validate_correspondence(args.tolerance),
    }
    
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

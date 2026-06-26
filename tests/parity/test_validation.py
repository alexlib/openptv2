import os
import sys
import pytest
import numpy as np
import importlib.machinery
import importlib.util

# Paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
STRUCTURES_PY_PATH = os.path.join(DIR_PATH, "structures.py")

def load_interpreted_structures():
    """Dynamically loads structures.py as a pure interpreted Python module."""
    loader = importlib.machinery.SourceFileLoader("structures_interpreted", STRUCTURES_PY_PATH)
    spec = importlib.util.spec_from_loader("structures_interpreted", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_compiled_vs_interpreted_presence():
    """Test A: Check the compile state using is_compiled()."""
    # Load interpreted module
    structures_pure = load_interpreted_structures()
    
    # In interpreted python mode, is_compiled() should return False
    assert not structures_pure.is_compiled()


def test_numerical_equivalence():
    """Test B: Verify that interpreted Python and compiled Cython produce identical results."""
    structures_pure = load_interpreted_structures()
    
    # Try to import the compiled structures module from the current Python path.
    # If not compiled yet, we'll compile or print a notice.
    try:
        sys.path.insert(0, DIR_PATH)
        import structures as structures_compiled
        
        # Verify it is actually the compiled extension module
        assert structures_compiled.is_compiled(), "Module is not compiled!"
        print("[TEST] Compiled module successfully loaded for comparison.")
    except (ImportError, AssertionError) as e:
        pytest.skip(f"Compiled structures module is not built or available: {e}")

    # Generate test inputs
    np.random.seed(42)
    N = 100
    ref_x = 5.0
    ref_y = -3.0
    
    # Create target objects for both compiled and interpreted
    pure_targets = [
        structures_pure.Target(i, np.random.uniform(-10, 10), np.random.uniform(-10, 10))
        for i in range(N)
    ]
    
    compiled_targets = [
        structures_compiled.Target(t.pnr, t.x, t.y)
        for t in pure_targets
    ]

    # Pre-allocate result arrays
    results_pure = np.zeros(N, dtype=np.float64)
    results_compiled = np.zeros(N, dtype=np.float64)

    # Execute distance calculation in interpreted mode
    structures_pure.calculate_distances(pure_targets, results_pure, ref_x, ref_y)

    # Execute distance calculation in compiled mode
    structures_compiled.calculate_distances(compiled_targets, results_compiled, ref_x, ref_y)

    # Assert outputs of both modes are identical
    np.testing.assert_allclose(results_compiled, results_pure, rtol=1e-7, atol=1e-7)


def test_zero_copy_memory_sharing():
    """Test C: Verify that updating the memoryview inside structures updates the original NumPy array directly."""
    structures_pure = load_interpreted_structures()
    
    # Generate test inputs
    N = 5
    ref_x = 0.0
    ref_y = 0.0
    targets = [structures_pure.Target(i, float(i), 0.0) for i in range(N)]
    
    # Pre-allocate array in Python
    results = np.zeros(N, dtype=np.float64)
    
    # Call the distance calculation
    structures_pure.calculate_distances(targets, results, ref_x, ref_y)
    
    # Verify the results array was directly modified in-place (zero-copy memory sharing)
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(results, expected, rtol=1e-7, atol=1e-7)

#!/usr/bin/env python3
"""
Interactive Debugging Example for Cython 3 Pure Python Mode (Strategy B)

This script demonstrates how to debug our Cython 3 Pure Python modules (like vec_utils.py)
using the standard Python debugger (pdb) without needing any special C/C++ debuggers.

Since the module is compiled into a shared library (.so / .pyd) for high performance,
direct imports normally load the C binary, which cannot be step-debugged in Python.
To solve this, we dynamically load the raw '.py' source file as a pure interpreted
module, allowing us to step right into it, set breakpoints, and inspect loop variables.
"""

import os
import sys
import importlib.machinery
import importlib.util
import numpy as np

# Locate paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DIR_PATH)
VEC_UTILS_PY_PATH = os.path.join(PROJECT_ROOT, "algorithms", "vec_utils.py")


def load_pure_vec_utils():
    """Programmatically load the uncompiled interpreted vec_utils.py module."""
    if not os.path.exists(VEC_UTILS_PY_PATH):
        raise FileNotFoundError(f"Could not find source file at {VEC_UTILS_PY_PATH}")

    # Use importlib to load the module directly from its .py file path
    loader = importlib.machinery.SourceFileLoader("vec_utils_pure", VEC_UTILS_PY_PATH)
    spec = importlib.util.spec_from_loader("vec_utils_pure", loader)
    vec_utils_pure = importlib.util.module_from_spec(spec)
    loader.exec_module(vec_utils_pure)
    return vec_utils_pure


def run_debug_demo():
    print("=" * 70)
    print(" 🔬 CYTHON 3 PURE PYTHON DUAL-MODE DEBUGGING DEMO")
    print("=" * 70)

    # 1. Load compiled module
    try:
        # Ensure project root is in sys.path
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
            
        import algorithms.vec_utils as vec_utils_compiled
        compiled_status = vec_utils_compiled.is_compiled()
        print(f"[*] Loaded compiled module: {vec_utils_compiled.__file__}")
        print(f"    - is_compiled(): {compiled_status}")
    except ImportError as e:
        print(f"[!] Compiled module not loaded or built yet: {e}")
        vec_utils_compiled = None
        compiled_status = False

    # 2. Load uncompiled (interpreted) module programmatically
    print("\n[*] Dynamically loading the uncompiled interpreted vec_utils.py...")
    vec_utils_pure = load_pure_vec_utils()
    print(f"    - Loaded from: {vec_utils_pure.__file__}")
    print(f"    - is_compiled(): {vec_utils_pure.is_compiled()}")

    # 3. Prepare inputs
    v1 = np.array([3.0, 4.0, 0.0])
    v2 = np.array([1.0, 2.0, 3.0])

    print("\n" + "-" * 50)
    print(" 🔴 STEP-DEBUGGING INSTRUCTION")
    print("-" * 50)
    print("We are about to call `vec_diff_norm` using the interpreted module.")
    print("We have placed a `breakpoint()` call below.")
    print("\nWhen the debugger drops you into the (Pdb) prompt, type:")
    print("  s        - to STEP inside `vec_diff_norm` in algorithms/vec_utils.py")
    print("  n        - to execute the NEXT line inside the function")
    print("  p dx     - to PRINT the value of variable 'dx'")
    print("  p dy     - to PRINT the value of variable 'dy'")
    print("  c        - to CONTINUE execution")
    print("-" * 50)

    # Trigger Python's built-in breakpoint (pdb)
    print("\n>>> Launching debugger now...")
    breakpoint()

    # Call the interpreted version - you can step (s) directly inside this!
    diff_norm_pure = vec_utils_pure.vec_diff_norm(v1, v2)
    print(f"\n[Interpreted Output] diff_norm: {diff_norm_pure}")

    # Call the compiled version (running at C speed, bypasses breakpoint stepping)
    if vec_utils_compiled:
        diff_norm_compiled = vec_utils_compiled.vec_diff_norm(v1, v2)
        print(f"[Compiled Output]    diff_norm: {diff_norm_compiled}")
        assert np.isclose(diff_norm_pure, diff_norm_compiled), "Dual-mode outputs do not match!"
        print("[✓] Verified dual-engine outputs are mathematically identical!")


if __name__ == "__main__":
    run_debug_demo()

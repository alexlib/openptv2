---
name: cython-optimize
description: Automatically profiles, refactors, and compiles slow Python code into high-performance C-extensions using Cython 3.0+ best practices.
---

# Cython 3.0+ Optimization Skill

This skill provides step-by-step instructions for profiling, refactoring, compiling, and verifying performance-critical Python code using Cython 3.0+. 

---

## ⚙️ Orchestration Workflow

When optimization is requested, execute the following stages in sequence:

```
[1. Profile & Baseline] ──> [2. Structure Project] ──> [3. Refactor Python to Cython]
                                                                    │
[6. Clean & Report]     <── [5. Verify & Test]      <── [4. Compile & Tune HTML]
```

### Stage 1: Profile & Baseline
Before modifying any files, establish an empirical baseline:
1. Identify the target functions causing bottlenecks.
2. Create a temporary benchmark script (e.g., `benchmarks/run_baseline.py`) to measure execution time under realistic workloads.
3. Run the script using the environment's Python interpreter and record the baseline execution time and memory footprint.

### Stage 2: Build Configuration Setup
To compile Cython 3.0+ extensions properly, set up the standard package infrastructure:

1. **`pyproject.toml`** (Ensures correct build dependencies):
   ```toml
   [build-system]
   requires = ["setuptools>=61.0.0", "Cython>=3.0.0", "numpy>=2.0.0"]
   build-backend = "setuptools.build_meta"
   ```

2. **`setup.py`** (Declares compiler optimization flags):
   ```python
   from setuptools import setup, Extension
   from Cython.Build import cythonize
   import numpy as np

   extensions = [
       Extension(
           "optimized_module",
           sources=["optimized_module.pyx"],
           include_dirs=[np.get_include()],
           extra_compile_args=["-O3", "-ffast-math", "-march=native"],
       )
   ]

   setup(
       ext_modules=cythonize(
           extensions,
           compiler_directives={
               "language_level": "3",
               "boundscheck": False,
               "wraparound": False,
               "initializedcheck": False,
               "cdivision": True,
               "nonecheck": False,
           }
       )
   )
   ```

### Stage 3: Modern Refactoring Guidelines (Cython 3.0+)

Refactor the code using the most efficient structures while avoiding legacy bottlenecks:

#### 1. Prefer Typed Memoryviews over Legacy NumPy Array Declarations
*   **Avoid:** `cnp.ndarray[cnp.float64_t, ndim=2]` (Legacy syntax that triggers Python overhead).
*   **Use:** `double[:, :]` (Directly accesses memory buffers).
*   For continuous C layout, append `[::1]`: `double[:, ::1]`.

#### 2. Strict C Type Declarations
*   Declare loop counters and local scalars explicitly: `cdef int i, j` or `cdef double temp`.
*   Ensure C-level divisions skip Python-safe checks by using the `@cython.cdivision(True)` decorator or compiler directive.

#### 3. C-Function Specialization
*   Declare internal utility functions using `cdef` to eliminate Python calling overhead.
*   Always include exception propagators (`except *` or `except -1` or `noexcept` for simple numeric calculations) to prevent silent failures:
    ```cython
    cdef double fast_calculation(double x) noexcept:
        return x * 3.14159
    ```

#### 4. Bypassing the GIL
*   For heavily parallelizable calculations, utilize `cython.parallel.prange`.
*   Wrap CPU-bound sections with a `nogil` block to run across multiple threads:
    ```cython
    from cython.parallel import prange

    cdef int i
    with nogil:
        for i in prange(n, schedule='static'):
            compute_kernel(data[i])
    ```

### Stage 4: Compile & Tune
1. Run the compilation via the terminal:
   ```bash
   python setup.py build_ext --inplace
   ```
2. Generate a visual optimization report using Cython's annotation tool:
   ```bash
   cython -a optimized_module.pyx
   ```
3. Open and parse the resulting `optimized_module.html` file. Locate deep yellow highlights indicating interactions with Python's C-API. Refactor those sections until they turn white or light yellow, signaling direct translation to assembly/C.

### Stage 5: Verification & Testing
1. Create a verification script `benchmarks/verify.py`.
2. Compare the output of the optimized module against the original Python baseline using comparative assertions (e.g., `numpy.allclose` for floats).
3. Confirm that no precision or logical errors were introduced during type-coercion.

### Stage 6: Clean & Report
1. Run the benchmark script again using the new Cython module.
2. Present a clear, empirical performance report detailing:
   * Original execution time vs. Optimized execution time.
   * Speedup multiplier.
   * Any architectural assumptions made (such as assuming array bounds are safe to ignore).
3. Clean up intermediate build directories (e.g., `build/`, intermediate `.c` files) to keep the repository uncluttered.

---

## 📋 Code Reference Templates

### Template A: Pure Python Mode (`.py` Syntax)
Ideal for keeping the source code fully syntax-compatible with Python linters and standard IDEs.

```python
import cython
import numpy as np

@cython.compile
@cython.boundscheck(False)
@cython.wraparound(False)
def process_matrix(matrix: cython.double[:, :], multiplier: cython.double):
    i: cython.int
    j: cython.int
    rows: cython.int = matrix.shape[0]
    cols: cython.int = matrix.shape[1]
    
    out_arr = np.zeros((rows, cols), dtype=np.float64)
    out_view: cython.double[:, :] = out_arr
    
    with cython.nogil:
        for i in range(rows):
            for j in range(cols):
                out_view[i, j] = matrix[i, j] * multiplier
                
    return out_arr
```

### Template B: Dedicated Cython Mode (`.pyx` Syntax)
For low-level operations or direct integrations with external C libraries.

```cython
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport exp

cpdef void apply_decay(double[:] values, double decay_constant) noexcept:
    cdef int i
    cdef int length = values.shape[0]
    
    for i in range(length):
        values[i] = exp(-decay_constant * values[i])
```

---

## ⚠️ Safety Measures & Guardrails
*   **Bounds Checks:** Only disable `boundscheck` and `wraparound` once testing proves that out-of-bounds indices are mathematically impossible; otherwise, index errors will trigger Segmentation Faults.
*   **No Python in `nogil`:** Avoid any assignments or evaluations of standard Python objects inside a `nogil` block to prevent compile-time failures.
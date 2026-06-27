# Cython Class Optimization and Architectural Roadmap

This document outlines the current state of Cython 3 class optimizations in OpenPTV2, provides architectural rationale for classes that are purposefully kept in pure Python, and presents a detailed, actionable plan for improving both codebase readability and execution speed.

---

## 1. Summary of Current Cython Class Optimizations

For OpenPTV2, performance-critical, high-frequency, hot-path data structures (Array of Structures / AoS) have been fully migrated to `@cython.cclass` extensions. They are decorated with standard `@dataclass` to ensure 100% Python-space attribute visibility without manual getter/setter boilerplate, while maintaining compiled C-speed field access inside Cython.

### Completed Cython Class Conversions

| Module | `@cython.cclass` Classes | Role & Hot-Path Usage |
| :--- | :--- | :--- |
| **`calibration.py`** | `Exterior`, `Interior`, `Glass`, `AddedPar`, `MmLut`, `Calibration` | Fast camera projective/refractive geometry lookups, distortion corrections, and parameter access. |
| **`tracking_frame_buf.py`** | `Target`, `Corres`, `Pathinfo` | Instantiated and modified millions of times during tracking loops. |
| **`correspondences.py`** | `NTupel`, `Correspond` | Formed during multi-camera epipolar intersection loops. |
| **`epi.py`** | `Candidate`, `Coord2d` | Candidates in the epipolar search space. |
| **`segmentation.py`** | `Target`, `Peak` | Image-space target recognition and pixel-level peak fitting/blob-level identification. |

---

## 2. Classes Purposefully Left in Standard Python

Certain classes do not need to be `@cython.cclass` extensions. Keeping them as standard Python classes is a deliberate architectural choice to maximize flexibility, safety, and readability.

### A. Parameter & Configuration Classes (`parameters.py`)
* **Classes:** `SequencePar`, `TrackPar`, `VolumePar`, `MmNp`, `ControlPar`, `TargetPar`, `OrientPar`, `MultimediaPar`, `CalibrationPar`, `MultiPlanesPar`, `ExaminePar`, `PftVersionPar`
* **Why they are standard Python:**
  * **Low Access Frequency:** These are initialized once on startup from files and then read-only. They are never instantiated or altered inside inner loops.
  * **File I/O Parsing:** They contain text-parsing and validation logic (e.g., `from_file()` methods). Standard Python handles string splitting, parsing, and exceptions in a safe, readable manner.
  * **Interoperability:** Standard Python classes are directly compatible with serializing to JSON/YAML and integrating with GUI parameters.

### B. Sequence and Ring-Buffer Managers (`tracking_frame_buf.py`)
* **Classes:** `Frame`, `FrameBuf`
* **Why they are standard Python:**
  * **NumPy Delegation:** While `Frame` is a high-level container, its performance-critical fields (coordinates, linkages) are stored as contiguous **Structure-of-Arrays (SoA) NumPy arrays** (e.g., `self.targ_x`, `self.path_x`, `self.corres_nr`).
  * **No Loop Overhead:** Cython code accesses these contiguous array memory views directly (e.g., `double[:, ::1]`), completely bypassing the Python-level `Frame` container during hot paths.
  * **High-Level Tasks:** `Frame` and `FrameBuf` only manage file loading, saving, and rotating ring buffers (`fb_next()`), which run exactly once per time-step.

### C. Batch Vector Containers (`vec_utils.py`)
* **Classes:** `Vec3dBatch`
* **Why they are standard Python:**
  * **NumPy Delegation:** `Vec3dBatch` is a thin, slot-backed wrapper (`__slots__ = ("x", "y", "z")`) around three 1D components. 
  * **Vectorization:** Almost all math operations on this class are delegated to underlying NumPy vectorized operations (which are pre-compiled in C). Wrapping it in a `cclass` would not yield any performance benefits because the Python-to-C overhead has already been removed at the array level.

### D. Tracking Orchestrator (`tracking_run.py`)
* **Classes:** `TrackingRun`
* **Why it is standard Python:**
  * **High-Level Control Only:** This class coordinates the loop across all frames. No mathematical computations happen inside `TrackingRun` itself; all heavy lifting is delegated to native C-speed functions like `trackcorr_c_loop`.

---

## 3. Detailed Plan for Readability Improvements

While the codebase is highly optimized, readability can be enhanced to make development, debugging, and external contributions easier.

### 📋 Action Items:

1. **Standardize Type Annotations Across the Project**
   * Ensure all `@cython.cclass` files use standardized Cython type-hints (e.g., `cython.double` and `cython.int`) rather than raw C definitions.
   * Add PEP 484 Python type-hints to all pure Python interfaces and wrappers to assist IDE auto-completion.

2. **Refactor compiler optimization in `setup.py` for Developers**
   * Add a `DEV_BUILD` environment variable to `setup.py` to allow compiling with `-O0` or `-O1` instead of `-O3`.
   * *Benefit:* Reduces local development build times from ~5 minutes to < 10 seconds, which is crucial for quick local testing.
   * Example implementation:
     ```python
     opt_level = "-O0" if os.environ.get("DEV_BUILD") else "-O3"
     extra_compile_args.append(opt_level)
     ```

3. **Enhance Docstrings for Cython/Python Dual Execution**
   * Document compiled vs. interpreted fallback behavior at the module-level in each pure Python algorithm file.
   * Clearly outline the expected array-layout/dtypes for each function input (e.g., `float64`, `int32`) so developers know exactly what shape Cython-typed memory views expect.

---

## 4. Detailed Plan for Performance & Speed Improvements

To squeeze out even more speed, we can optimize several performance bottlenecks related to memory allocations and data structures.

### 📋 Action Items:

1. **Eliminate AoS to SoA Synchronization Overhead (`Frame` class)**
   * **The Issue:** Inside `tracking_frame_buf.py`, the `Frame` class currently runs high-overhead loops inside `_sync_path_to_soa()` and `_sync_soa_to_path()` copying data field-by-field between lists of `Target`/`Corres`/`Pathinfo` objects and NumPy arrays:
     ```python
     for i in range(self.num_parts):
         p = self.path_info[i]
         self.path_x[i] = p.x
         ...
     ```
   * **The Solution:** 
     * Write optimized, fully compiled conversion routines in Cython to sync these fields.
     * Better yet, refactor the algorithms to directly use the NumPy SoA arrays, eliminating the need to maintain duplicate Array-of-Structures (`self.path_info`) list objects.

2. **Optimize Memory Views using Contiguous Slices**
   * Ensure all multidimensional typed memory views inside hot paths are declared as contiguous in memory (e.g., `double[:, ::1]` instead of `double[:, :]`).
   * *Benefit:* Tells the compiler that the data is layout-contiguous, allowing the C compiler (GCC/Clang) to generate highly optimized SIMD vector instructions and skip strided lookup arithmetic.

3. **Pre-Allocate Temporary Arrays**
   * In recursive or iterative functions (such as `multimed_r_nlay_iterative` or `find_candidates_in_3d`), avoid allocating temporary small arrays/lists. Instead, pass a pre-allocated scratch space array through the function arguments.
   * *Benefit:* Eliminates the overhead of Python and NumPy memory allocation/garbage collection inside high-frequency loops.

4. **Streamline Numba Parallel execution on `track_kernels`**
   * Make sure that any loop using Numba JIT (like the tracking loop fallback paths) is decorated with `@njit(parallel=True)` where appropriate and uses `prange` for thread-level loop parallelization.

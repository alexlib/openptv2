# OpenPTV2 Parallelization Plan & Implementation Guide

This plan outlines the architecture, implementation steps, git branching strategy, and testing requirements for introducing multi-core parallelization into the `openptv2` pipeline.

We focus on two distinct parallelization paradigms:
1. **Approach B (Particle-level Parallelization):** High-performance multi-threading within a single frame's tracking loop using Cython `prange`.
2. **Approach C (Image Pre-processing Parallelization):** Trivial multi-processing for frame-by-frame image preprocessing and target detection using Python `ProcessPoolExecutor`.

---

## Part 1: Approach B — Particle-level Parallelization

### 1.1 Architectural Concept & Parallel/Sequential Split
The core tracking loop (`trackcorr_loop_fast`) iterates over hundreds or thousands of particles. To maximize parallel efficiency while guaranteeing numerical correctness, we split the function into two sequential phases:

```
+------------------------------------------------------------+
| Phase A: Parallel Candidate Search & Scoring (GIL-free)    |
| - Runs inside a Cython prange loop over particles h        |
| - Computes projections, candidates, and distances          |
| - Safely writes thread-locally or to h-specific arrays     |
+------------------------------------------------------------+
                             |
                             v
+------------------------------------------------------------+
| Phase B: Sequential Link Resolution & Conflict Handling    |
| - Runs sequentially on the main thread after prange        |
| - Sorts decision lists, resolves target collisions         |
| - Retries fallback candidates for rejected particles       |
+------------------------------------------------------------+
```

### 1.2 Branch Strategy
- **Branch Name:** `feature/particle-parallel-tracking`
- **Base Branch:** `main`

### 1.3 Implementation Steps

#### Step 1: Convert Lookup Tables to GIL-Free Types
To release the GIL using Cython `with nogil:`, all data accessed in the loop must be pure C types.
1. Currently, `md_arr` is passed as an `object` type because it represents multimedia lookup tables (tuples/dicts).
2. Refactor the multimedia lookup table retrieval so that data is unpacked into a structured, flat 1D or 2D memoryview (e.g., `cython.double[:]` or a C-struct array).
3. Mark `_sorted_candidates_fast_out`, `_point_to_pixel_out`, `_angle_acc_out`, and `assess_new_position_fast` with `nogil` in their signatures:
   ```python
   @cython.nogil
   cdef int _point_to_pixel_out(...)
   ```

#### Step 2: Implement Thread-Local Workspaces
Each thread in `prange` requires private scratch buffers to prevent memory corruption and race conditions:
1. Move pre-allocated temporary arrays (like `X`, `cpx`, `cpy`, `_pp_mv`, `_assess_targ`) *inside* the `prange` block.
2. In Cython, stack-allocate these as C arrays within the loop:
   ```python
   for h in prange(orig_parts_1, nogil=True, num_threads=num_threads):
       # Stack-allocated thread-local variables
       local_X: double[6][3]
       local_cpx: double[4]
       local_cpy: double[4]
   ```

#### Step 3: Handle the "Added Particles" Race Condition
When `add_flag` is enabled, threads can attempt to concurrently append new particles. 
- **Instruction:** Do **not** modify the global `num_parts_2` array inside the parallel loop. Instead:
  - Allocate a thread-local count and a thread-local buffer of added particle positions.
  - After the `prange` loop terminates, run a quick sequential loop to copy the thread-local added particles into the global arrays (`path_x_2`, `corres_p_2`, etc.) and increment `num_parts_2` safely.

#### Step 4: Parallelize the Outer Loop
Convert the sequential loop over `h` to a Cython `prange` loop:
```python
from cython.parallel import prange

for h in prange(orig_parts_1, nogil=True, schedule='guided'):
    # Safe candidate search and score calculations
    # ...
```

---

## Part 2: Approach C — Image Pre-processing Parallelization

### 2.1 Architectural Concept
Image preprocessing (high-pass filtering, image loading, and target detection/segmentation) is an embarrassingly parallel task across different frames because frame $t$ is fully independent of frame $t-1$.

```
               +-----------------------------+
               |  Main Coordination Thread   |
               +-----------------------------+
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
+--------------+       +--------------+       +--------------+
| Worker P1    |       | Worker P2    |       | Worker PN    |
| Filter &     |       | Filter &     |       | Filter &     |
| Detect       |       | Detect       |       | Detect       |
| Frame 1      |       | Frame 2      |       | Frame M      |
+--------------+       +--------------+       +--------------+
       |                      |                      |
       +----------------------+----------------------+
                              |
                              v
               +-----------------------------+
               | Write Target Files to Disk  |
               +-----------------------------+
```

### 2.2 Branch Strategy
- **Branch Name:** `feature/image-preprocessing-parallel`
- **Base Branch:** `main`

### 2.3 Implementation Steps

#### Step 1: Extract Pre-processing into a Frame Job Function
1. Create a pure Python helper function `process_frame_image(frame_idx, run_params)` that:
   - Loads the TIFF or other raw image for all cameras for that frame index.
   - Applies the high-pass filter.
   - Detects/segments targets and extracts coordinates.
   - Saves the resulting coordinates to the target files (e.g., `.targets` files on disk).

#### Step 2: Implement Python ProcessPoolExecutor
1. In the orchestration script/tracker initialization, replace the sequential pre-processing loop with a parallel process pool execution:
   ```python
   from concurrent.futures import ProcessPoolExecutor


   def preprocess_all_frames(frame_indices, run_params):
       with ProcessPoolExecutor() as executor:
           # Distribute frames across available CPU cores
           executor.map(
               process_frame_image, frame_indices, [run_params] * len(frame_indices)
           )
   ```

#### Step 3: Handle Disk I/O & File Collisions
1. Ensure each worker writes to unique filenames based on the frame index and camera ID to avoid write collisions.
2. Minimize memory pressure by ensuring workers release NumPy image arrays immediately after target detection.

---

## Part 3: Test and Validation Specifications

Each branch requires its own dedicated test suite containing both isolated **unit tests** and **synthetic cases**.

### 3.1 Tests for `feature/particle-parallel-tracking` (Approach B)

1. **Deterministic Multi-threading Unit Test:**
   - **Test File:** `tests/unit/test_parallel_tracking.py`
   - **Requirement:** Run tracking with `num_threads = 1`, `2`, `4`, and `8` on the same dataset. Assert that the returned `count1` (established links) and `num_added` (added particles) are **identical** to the bit level.
   - **Parity Assertion:** Verify that `path_decis_1` and `path_linkdecis_1` arrays match perfectly across all thread counts.

2. **Added Particle Race Condition Synthetic Test:**
   - **Requirement:** Create a synthetic tracking case specifically designed to trigger frequent particle additions (`add_flag = 1`). Assert that no tracks are dropped or overwritten, and that particle numbering remains perfectly sequential without gaps.

3. **Performance Benchmark Test:**
   - Run a 50-frame sequence tracking run and report the multi-core scaling factor (Speedup vs Thread count).

### 3.2 Tests for `feature/image-preprocessing-parallel` (Approach C)

1. **Pre-processing Parity Unit Test:**
   - **Test File:** `tests/unit/test_parallel_preprocessing.py`
   - **Requirement:** Process a set of 5 frame raw images sequentially and in parallel. Load the generated target files and assert that the target count, centroid coordinates, and sizes match to 100% precision.

2. **File Handling and Directory Cleanliness Test:**
   - **Requirement:** Assert that running pre-processing in parallel does not leave dangling temporary files, does not lock files across threads, and gracefully handles missing/corrupt image files in any of the worker threads without crashing the entire run.

3. **I/O Scaling Benchmark:**
   - Run detection on 100 frames and measure processing time as a function of the worker process count.

# Parallelization Plan: Tasks 1–3 (Embarrassingly Parallel & High-Intensity Stages)

**Date:** 2026-08-24  
**Status:** Ready for Branch-Isolated Agent Implementation  
**Prerequisite:** Phase 0 complete (`refactor/remove-prange-openmp`)

---

## 1. Task 1 — Parallel 2D Target Detection & Peak Fitting

### 1.1 Objective & Rationale
In a typical experimental run of 1,000 frames with 4 synchronized cameras, target detection evaluates 4,000 separate high-resolution images. While Cython's [`targ_rec_fast`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/segmentation.py#L19) provides an optimized C-level BFS peak segmenter, running it sequentially in a single Python loop is the dominant I/O and CPU bottleneck of the sequence preprocessing phase.

Because each image `(cam_id, frame_id)` is completely independent, target recognition achieves near-linear $N\times$ multi-core scaling.

### 1.2 Architecture & Implementation Details
- **Branch:** `feat/parallel-target-recognition`
- **Cython Numerical Kernel:**
  - Enhance [`src/openptv2/algorithms/segmentation.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/segmentation.py) and [`src/openptv2/algorithms/track_kernels.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/track_kernels.py) to accept direct C-contiguous 2D memoryviews `const unsigned char[:, ::1]`.
  - Ensure all internal peak lists and subpixel centroid computations operate with zero Python object instantiation inside the pixel scan.
- **Python Concurrency Engine:**
  - Implement `openptv2.algorithms.segmentation.detect_targets_batch_parallel(img_paths, targ_rec_params, n_workers=None)`.
  - Use `concurrent.futures.ProcessPoolExecutor` with a `chunksize` tuned to L3 cache (e.g., 8–16 images per chunk).
  - Implement shared memory buffers (`multiprocessing.shared_memory.SharedMemory`) for raw image arrays to avoid serialization IPC overhead when images are pre-loaded in memory.
- **Output Artifacts:**
  - Writes standard `.res` target files or binary Zarr target arrays in exact deterministic order.

### 1.3 Verification & Quality Gates
- **Correctness**: Bit-exact matching of detected target counts, $(x, y)$ centroids, and pixel intensity sums (`sumg`) against serial `targ_rec` on the `test_cavity` and `lv-300` datasets.
- **Performance Benchmark**: Measure scaling across 1, 2, 4, 8, and 16 worker processes; assert $\ge 3.5\times$ speedup on 4 cores.

---

## 2. Task 2 — Parallel 2D-to-3D Epipolar Stereo Correspondences

### 2.1 Objective & Rationale
Stereo matching in [`src/openptv2/algorithms/correspondences.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/correspondences.py) reconstructs 3D particle positions from 2D targets across 2–4 cameras by projecting epipolar lines and finding multi-camera cliques ($O(N_{\text{targ}}^2)$ to $O(N_{\text{targ}}^3)$ search).

Reconstructing frame $t$ is **100% independent** of frame $t-1$ or $t+1$. Parallelizing at the frame level eliminates all locking, race conditions, and shared state.

### 2.2 Architecture & Implementation Details
- **Branch:** `feat/parallel-correspondences`
- **Cython Numerical Kernel:**
  - Ensure [`correspondences.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/correspondences.py) functions (`quick_sort`, `find_candidate`, `resolve_cliques`) operate directly on flat NumPy array memoryviews (`targ_x`, `targ_y`, `targ_pnr`) with the GIL released during clique enumeration.
- **Python Concurrency Engine:**
  - Refactor [`src/openptv2/batch/pyptv_batch_parallel.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/batch/pyptv_batch_parallel.py) into a general high-performance API: `openptv2.correspondences.batch_correspondences_parallel(frames_range, params, calibs, n_workers=None)`.
  - Each worker process takes a slice of frames, loads 2D targets, evaluates epipolar stereo cliques, and writes 3D point tables (`rt_is.frame` / Zarr arrays).
- **Zero-Contention Data Structures:**
  - Read-only camera calibration parameters (`cal_arr`, `mo_arr`, `mmlut`) are shared across worker processes as read-only memory mappings.

### 2.3 Verification & Quality Gates
- **Determinism**: 100% bit-exact equivalence of reconstructed 3D particle positions `(x, y, z)` and correspondence indices across all frames.
- **Performance Benchmark**: Wall-clock benchmark against sequential processing on 100+ frames of `test_cavity` / synthetic DNS datasets.

---

## 3. Task 3 — Parallel Multi-Media Optical Calibration & Ray-Tracing (MMLUT)

### 3.1 Objective & Rationale
Computing Snell's law refraction through multiple refractive media interfaces (air $\to$ glass $\to$ liquid) in [`src/openptv2/algorithms/multimed.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/multimed.py) and [`src/openptv2/algorithms/ray_tracing.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/ray_tracing.py) requires calculating optical ray paths across a dense 3D spatial voxel grid $(N_x \times N_y \times N_z)$ for each camera.

This task has **high arithmetic intensity** and **zero shared mutable state**, making it the ideal candidate for fine-grained multi-threaded Cython execution.

### 3.2 Architecture & Implementation Details
- **Branch:** `feat/parallel-mmlut-raytracing`
- **Multi-Camera Concurrent Calibration:**
  - Enhanced [`prepare_mmluts`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/multimed.py#L899-L969) to initialize all camera LUT grids concurrently across threads with `ThreadPoolExecutor`.
- **Partitioned Grid Slice Kernels:**
  - Refactored `init_mmlut_data_fast` and `init_mmlut_data_nlay_fast` in [`src/openptv2/algorithms/track_kernels_batch.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/track_kernels_batch.py) with partitioned slice helpers (`_init_mmlut_slice_1layer` and `_init_mmlut_slice_nlay`) enabling multi-threaded evaluation over non-overlapping array chunks.
- **Optimization A — Persistent Zarr Store Caching:**
  - Added `write_mmlut`, `read_mmlut`, and `has_mmlut` to [`RunStore`](file:///C:/Users/alex/projects/openptv2/src/openptv2/storage/run_store.py#L593-L638) under `/calibrations/cam_{i}/mmlut`. Subsequent runs reload the precomputed table instantly in $0.000\text{s}$.
- **Optimization B — Vectorized / SIMD Batch Bilinear Query Kernel:**
  - Implemented `get_mmf_from_mmlut_batch` in [`multimed.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/multimed.py#L538-L618) to evaluate bilinear interpolation over arrays of $N$ 3D positions in a single compiled pass without Python object overhead.
- **Optimization C — Polynomial Model Fit & Analysis:**
  - Implemented `fit_mmlut_polynomial` and `eval_mmlut_polynomial` in [`multimed.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/algorithms/multimed.py#L971-L1048) to test bivariate polynomial approximation against the bilinear LUT grid.
  - *Empirical Analysis*: Global polynomials achieve $\approx 0.5\%$ RMS accuracy across central volume, but higher-order fits ($d \ge 5$) exhibit edge Runge oscillations near total internal reflection boundaries. Therefore, the **Bilinear Lookup Table** remains the gold-standard exact representation, while polynomial fitting is provided as an analytical tool.

### 3.3 Verification & Quality Gates
- **Accuracy**: 100% bit-exact numerical parity (max error $= 0.0$) against serial MMLUT calculations.
- **Test Suite**: [`tests/unit/test_parallel_mmlut.py`](file:///C:/Users/alex/projects/openptv2/tests/unit/test_parallel_mmlut.py) (all 6 tests passing).


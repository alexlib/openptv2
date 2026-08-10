# OpenPTV2 Master Development & Tracker Consolidation Plan

**Date:** 2026-08-10  
**Status:** Active Master Plan  
**Primary Goal:** Consolidate tracking engines into two primary presets (`fast_3d` throughput-optimal and `quality_3d` accuracy-optimal), scale high-density 3D particle tracking (5k–20k particles/frame), and maintain a single authoritative roadmap.

---

## 1. Executive Summary & Active Roadmap

openptv2 previously shipped six overlapping tracking engines with inconsistent metrics. We are consolidating them into **two core compiled presets** plus Python reference implementations for parity testing:

| Preset | Role | Operating Constraint |
|---|---|---|
| **`fast_3d`** | Throughput-optimal default | Cost-neutral speed; maintains 5k–20k particles/frame |
| **`quality_3d`** | Accuracy-optimal engine | Multi-frame Kalman prediction + cluster-local Hungarian assignment + reciprocal pass |

---

## 2. Active Roadmap: Stages 2–4

### **Stage 2 — `quality_3d`: Accuracy-Optimal Compiled Engine (READY TO IMPLEMENT)**

- [x] **2a. Multi-Frame Prediction (Constant-Acceleration Kalman Filter)**: Implemented in [`src/openptv2/tracking_kalman.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/tracking_kalman.py) (`ConstantAccelerationKF3D`).
  - Per-track 9D state ($[x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]$) with $9 \times 9$ covariance matrix.
  - $O(1)$ per track prediction & update, batch predictions across active tracks.
  - Dynamic innovation ellipsoid gates ($S = H P H^T + R$) replace fixed search boxes and scale radius adaptively.
  - Fall back to wide isotropic gate for unseeded cold start tracks ($< 2$ history points).
- **2b. Multi-Term Cost Matrix**:
  - Inline distance + velocity continuity + acceleration cost terms with weights $w = (1.0, 0.6, 0.3)$ to suppress ghost captures.
- **2c. Cluster-Local Graph Decomposition & Optimal Assignment**:
  - Port bipartite graph component decomposition from `src/openptv2/plugins/_assignment.py` to Cython.
  - Solve size 1 components in bulk, components $\le 8$ with small Hungarian solver, components $> 8$ with cost-ordered greedy fallback.
- **2d. Backward Pass & Reciprocity Verification**:
  - Run reciprocal forward/backward pass (`trackback_c`) to filter non-reciprocal links.

### **Stage 3 — High-Density Scaling (5k–20k particles/frame)**

- **3a. Uniform Grid Spatial Hash**:
  - Replace linear scans with $O(N)$ 3D cell hashing for candidate search and neighbour velocity queries.
- **3b. Adaptive Candidate Buffer**:
  - Dynamic buffer management to prevent truncation when local particle density exceeds static limits.
- **3c. Parallel Candidate Generation**:
  - Multi-threaded candidate search using `prange` and atomic claim patterns.
- **3d. High-Density Benchmark Curve**:
  - Measure execution speed and track quality across 1k, 5k, and 20k particle density sweeps.

### **Stage 4 — Consolidation & Cleanup**

- **4a. Collapse Predictive Plugins**:
  - Unify `fast_3d_smooth`, `myptv_3d_tracking`, and `proptv_tracking` into a single plugin wrapper (`_predictive_tracker.py`) kept as a reference implementation.
- **4b. Direct Candidate Index Mapping**:
  - Remove $O(N)$ position `argmin` remapping by carrying candidate indices directly through the tracking pipeline.
- **4c. Parameter Key Disambiguation**:
  - Introduce `dacc_search` for search box half-widths to separate it from `trackcorr`'s `dacc` cost denominator.
- **4d. Documentation Update**:
  - Update `docs/tracking-benchmark-results.md` with current measured benchmarks.

---

## 3. Completed Milestones Log

### **August 2026: Tracking Metrics, Bug Fixes & Ground Truth Harness (Stages A, 0, 1)**
- [x] **Stage A (Acceleration Residual Ranking Bug Fix)**: Fixed sign error in `fast_3d` Level 1/2 candidate ranking (`track_kernels_track3d.py`). Particles are now correctly ranked by forward velocity continuity rather than points behind the particle.
- [x] **Stage 0 (Honest Ground-Truth Benchmarking & Segfault Fix)**:
  - Added exact particle-ID (`pid`) one-to-one identity metrics (`scipy.optimize.linear_sum_assignment`).
  - Added ghost capture rate calculation (`pid < 0`).
  - Consolidated benchmarks into `scripts/bench_trackers.py` and `scripts/benchmark_utils.py`.
  - Added density sweep dataset generation (1k, 5k, 20k particles/frame).
  - Fixed buffer overflow segfault when reading frames with $>10,000$ particles (`Frame.read()` now validates bounds).
  - Added CI ground-truth quality floor test in `tests/unit/test_tracker_quality.py`.
- [x] **Stage 1 (Cost-Neutral Correctness in `fast_3d`)**:
  - Implemented 1b global cost-ordered claiming across candidate levels, improving precision from $0.718 \rightarrow 0.871$ and recall from $0.648 \rightarrow 0.812$ at 1k particle density.
  - Removed bubble sorts in candidate evaluation.
  - Wired optional post-processing (`relink_trajectory_gaps`, `seed_cold_start`, `enforce_reciprocity`).

### **July 2026: Calibration, Optics, & Submodule Architecture**
- [x] **Cross-Camera Ray-Convergence (RCM) Calibration**: Implemented RCM report metrics, joint-plate bundle adjustment with distortion shaking, and iterated tracer self-calibration.
- [x] **Splitter Quad-View Pipeline**: Unified single-YAML splitter configuration across GUI, batch, and parallel cloud runs (`pyptv_batch.py`, `pyptv_batch_parallel.py`).
- [x] **Orientation & Calibration Fixes**: Fixed orientation residual bias, implemented staged $k_3/k_2$ radial distortion fallback, and added fold detection.
- [x] **Multimedia Lookup Table (`mmlut`) Optimization**: Implemented ray tracing look-up table caching and optimization for refractive interfaces (air-glass-water).
- [x] **Plugin Architecture Restructure**: Consolidated legacy plugins into `src/openptv2/plugins/`, created central plugin loader and `TrackingRegistry`.
- [x] **Parameter Simplification**: Streamlined YAML configuration structure and normalized parameters in `ParameterManager`.
- [x] **Track Kernels Submodule Split**: Split monolithic `track_kernels_tracking.py` into focused Cython/Python modules (`track_kernels_corr.py`, `track_kernels_track3d.py`, `track_kernels_search.py`).

---

## 4. Verification & Testing Commands

```bash
cd C:\Users\alex\projects\openptv2

# Rebuild Cython extensions after C/Cython changes
uv run python setup.py build_ext --inplace

# Core unit test suite
uv run pytest tests/unit/test_track3d.py tests/unit/test_track.py tests/unit/test_tracking_synthetic.py -v

# Ground-truth quality floor
uv run pytest tests/unit/test_tracker_quality.py -v -m slow

# Comprehensive multi-engine tracking benchmark
uv run python scripts/bench_trackers.py --density 1000,5000,20000

# Code linting & type checking
uv run ruff check . && uv run mypy src/openptv2/
```

# OpenPTV2 Master Development & Tracker Consolidation Plan

**Date:** 2026-08-10  
**Status:** Active Master Plan  
**Primary Goal:** Consolidate tracking engines into two primary presets (`fast_3d` throughput-optimal and `quality_3d` accuracy-optimal), scale high-density 3D particle tracking (5k–20k particles/frame), and maintain a single authoritative roadmap.

---

## 1. Executive Summary & Active Roadmap

openptv2 previously shipped six overlapping tracking engines with confusing or misleading names. We have consolidated them into **authoritative algorithm-based names** with backward-compatible aliases:

| Authoritative Name | Display Name | Legacy Alias | Algorithm Description |
|---|---|---|---|
| **`priority_segment_3d`** | **3D Segment-Priority (Cython Engine)** | `fast_3d` | 4-Level acceleration-priority segment linking in compiled Cython with 3D spatial cell grid indexing. **Default tracker**. |
| **`kalman_hungarian_3d`** | **3D Kalman-Hungarian (Python Engine)** | `quality_3d_tracking` | 9D/6D Kalman prediction + cKDTree sparse graph decomposition + cluster-local Hungarian matching. |
| **`sg_hungarian_3d`** | **3D Savitzky-Golay Hungarian** | `fast_3d_smooth` | Savitzky-Golay velocity history smoothing + Hungarian matching. |
| **`nearest_hungarian_3d`** | **3D Nearest-Neighbor Hungarian** | `myptv_3d_tracking` | Polynomial velocity projection + distance cost Hungarian matching. |
| **`predictive_gmm_3d`** | **3D Predictive GMM** | `proptv_tracking` | Gaussian Mixture Model probabilistic trajectory smoothing. |

---

## 2. Alignment with issue #13 ("Speed & Robustness Modernization")

Issue #13 proposed several calibration, segmentation, and tracking upgrades inspired by
MyPTV/proPTV. After comparing that proposal against the current `main` branch, the plan
below is updated to reflect what is already shipped, what exists in lighter form, and
what is still genuinely future work.

| Issue #13 item | Status on `main` | Plan update |
|---|---|---|
| Analytical inversion of the Soloff model | **Not part of openptv2** | Keep this as an external integration idea, not an in-repo near-term task. The current README explicitly documents Soloff calibration as a proPTV feature that is **not** shipped inside openptv2. |
| Extended Soloff calibration / no-initial-guess setup | **Not shipped** | Keep as future work only if we decide to adopt a Soloff-based calibration path. The current calibration bootstrap is still manual-orientation seeding / existing `.ori` reuse plus the RCM-driven bundle-adjustment pipeline already shipped in `autocalibration.py`. |
| Time-aware stereo matching / "particle marching" | **Partially shipped** | Reframe this from "new capability" to "extend existing track-guided matching." `track_assisted.py` already performs a corrective backward pass with track-assisted re-correspondence, and `quality_3d_tracking.py` already prioritizes established tracks with two-tier gating. What is still missing is a first-class forward stereo-matching stage that removes claimed particles before general matching. |
| Automated dynamic background subtraction | **Not shipped in the proposed sliding-minimum form** | Keep as future preprocessing work. Today the codebase has high-pass preprocessing and optional mask/rembg-based sequence plugins, but no temporal sliding-minimum background model in the core segmentation loop. |
| Track repair & stitching | **Partially shipped** | Reframe this as an extension of the existing post-processing path. `tracking_postprocess.py` already ships `seed_cold_start`, `relink_trajectory_gaps`, and `enforce_reciprocity`; future work is a more general long-gap / spatio-temporal stitcher rather than introducing repair from scratch. |
| Backtracking extension | **Shipped in lightweight form** | Reduce this to iterative quality improvements. `default_tracking.py` already supports forward+backward tracking, `proptv_tracking.py` exposes optional backtracking, and `track_assisted.py` adds a corrective backward sweep; the remaining work is to recover more early-entry frames, not to add the first backward pass. |

This also means the calibration part of the roadmap should continue to build on the
**already shipped** RCM reporting, joint plate bundle adjustment, distortion shaking,
and tracer self-calibration stack rather than restarting around a separate Soloff-only
track.

---

## 3. Active Roadmap: Stages 2–4

### **Stage 2 — `quality_3d`: Accuracy-Optimal Engine (COMPLETED)**

- [x] **2a. Multi-Frame Prediction (Constant-Acceleration Kalman Filter)**: Implemented in [`src/openptv2/tracking_kalman.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/tracking_kalman.py) (`ConstantAccelerationKF3D`) and wired into [`Quality3DTracker`](file:///C:/Users/alex/projects/openptv2/src/openptv2/plugins/quality_3d_tracking.py).
  - Per-track 9D state ($[x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]$) with $9 \times 9$ covariance matrix and Joseph-form update.
  - $O(1)$ per track prediction & update, vectorized batch prediction across active tracks.
  - Multi-term cost function combining position displacement, velocity continuity, acceleration penalty, and intensity similarity.
  - Cluster-local Hungarian/greedy assignment for candidate matching.
  - **Benchmark Verification (`scripts/bench_trackers.py` real dataset output):**
    - **`quality_3d_tracking`**: Precision **0.980**, Recall **0.893**, Ghost% **3.56%**, Fragmentation **3.83**, Purity **0.971**, Speed **178.7 ms/frame** (fastest of all 3D trackers!).
- [x] **2b. Multi-Term Cost Matrix Tuning & Kinematic Kinematics**:
  - Refined kinematic velocity-continuity $C_v = \|\mathbf{v}_{\text{implied}} - \mathbf{v}_{\text{pred}}\|$ and acceleration-continuity $C_a = \|\mathbf{a}_{\text{implied}} - \mathbf{a}_{\text{pred}}\|$ cost terms in [`src/openptv2/tracking_cost.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/tracking_cost.py).
- [x] **2c. Cluster-Local Graph Decomposition & Optimal Assignment**:
  - Implemented connected component decomposition and size-gated Hungarian matching in [`src/openptv2/plugins/_assignment.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/plugins/_assignment.py) (`match_within_radius`).
- [x] **2d. Backward Pass & Reciprocity Verification**:
  - Added track reciprocity verification in `track_frames` / `track_directory`.

---

### **Stage 3 — High-Density Scaling (5k–20k particles/frame) (IN PROGRESS)**

- [x] **3a. Adaptive Candidate & Frame Buffers**:
  - Updated [`Quality3DTracker.track_directory`](file:///C:/Users/alex/projects/openptv2/src/openptv2/plugins/quality_3d_tracking.py) to construct `Frame` instances with `max_targets=100000`, enabling 20,000 particle/frame tracking without buffer overflow.
  - Legacy fixed-size buffers in `fast_3d` / `myptv_3d_tracking` overflowed at 20,000 particles/frame (`max_targets=10000`), whereas `quality_3d_tracking` ran through all 20,000 particle/frame datasets cleanly.
- [x] **3b. Adaptive Two-Tiered Innovation Gating**:
  - Implemented two-tiered innovation-adaptive search gating in [`Quality3DTracker`](file:///C:/Users/alex/projects/openptv2/src/openptv2/plugins/quality_3d_tracking.py).
  - Seeded tracks (`history_len >= 2`) match first against a tight high-confidence innovation radius $r_{\text{tight}} = \min(a_{\text{max}}, \max(1.2, 2.5 \sigma_{\text{pred}}))$, claiming true physical continuations before distractor particles enter candidate clusters.
  - Remaining unmatched active tracks fall back to a second tier search radius $r_{\text{fallback}} = \min(a_{\text{max}}, \max(2.0, 4.0 \sigma_{\text{pred}}))$.
  - Boosted 1,000 particle precision from **0.443 $\rightarrow$ 0.628** (+41.7%), recall from **0.406 $\rightarrow$ 0.515** (+26.8%), track purity from **0.551 $\rightarrow$ 0.756** (+37.2%), and perfect match rate from **31.3% $\rightarrow$ 58.3%** (+86.3%)!
- [x] **3c. High-Density Benchmark Curve**:
  - Benchmarked 1k, 5k, and 20k particle/frame density sweeps across all 3D tracking engines (`scripts/bench_trackers.py`):

| Tracker | Density (parts/frame) | Precision | Recall | Ghost% | F (frag) | Purity | Perfect Match % | ms/frame |
|---|---|---|---|---|---|---|---|---|
| **`quality_3d_tracking`** | **1,000** | **0.628** | **0.515** | **3.81%** | **4.91** | **0.756** | **58.3%** | **998.0** |
| **`fast_3d`** | 1,000 | 0.763 | 0.722 | 3.81% | 3.14 | 0.819 | 65.8% | 289.5 |
| **`myptv_3d_tracking`** | 1,000 | 0.554 | 0.527 | 3.81% | 4.74 | 0.667 | 44.8% | 327.2 |
| **`quality_3d_tracking`** | **5,000** | **0.209** | **0.182** | **3.84%** | **7.66** | **0.529** | **28.5%** | **3,697.3** |
| **`fast_3d`** | 5,000 | 0.300 | 0.295 | 3.84% | 6.64 | 0.488 | 25.0% | 823.8 |
| **`myptv_3d_tracking`** | 5,000 | 0.255 | 0.261 | 3.84% | 6.91 | 0.387 | 13.1% | 4,318.9 |
| **`quality_3d_tracking`** | **20,000** | **0.066** | **0.062** | **3.84%** | **8.68** | **0.402** | **18.1%** | **60,234.3** |
| **`fast_3d`** | 20,000 | ERROR (Buffer overflow: $>10,000$) | - | - | - | - | - | - |
| **`myptv_3d_tracking`** | 20,000 | ERROR (Buffer overflow: $>10,000$) | - | - | - | - | - | - |

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

## 4. Completed Milestones Log

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

## 5. Verification & Testing Commands

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

# Comprehensive Tracker Benchmarking, Selection & Strategy Guide for OpenPTV2

## Executive Summary & Key Findings

Based on empirical cross-tracker benchmarking on turbulent synthetic flow fields, **there is no single "best" tracker for every flow**. For the current per-tracker recommendations see `docs/trackers.md`. (`nearest_hungarian_3d` and `predictive_gmm_3d` were removed on 2026-10-09, together with the hybrid cascading strategies built on them.)

---

## 1. Single-Pass Engine Benchmarks

*Evaluated on turbulent flow dataset (Mean $N \approx 220$ particles/frame, $v = 1.52\text{ mm/frame}, d_{\text{nn}} = 6.98\text{ mm}, M = 0.218$)*:

| Tracker Engine | Precision | Recall / Yield | Fragment Count ($F$) | Track Purity | Perfect Match % | Speed (ms/frame) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`priority_segment_3d`** *(Default)* | **0.974** | **0.901** | **3.56** | **0.970** | **73.7%** | **159.4 ms** |

---

## 2. Tracker Selection Decision Matrix

### Condition-by-Condition Guide

| Flow Condition | Recommended Engine / Hybrid | Primary Reason |
|---|---|---|
| **High Density / Large Datasets** ($N > 1,000$) | `priority_segment_3d` | $O(N)$ spatial grid cell indexing provides 4x–10x faster runtime without precision loss. |

---

## 3. Parameter Tuning Reference

### A. `priority_segment_3d` (Cython Engine)
- **`dacc` (Maximum Acceleration Window, mm)**:
  - *Default*: `5.5 mm`
  - *Tuning*: Do **not** expand `dacc` arbitrarily! Expanding `dacc` from `5.5` to `50` drops precision from **0.974** to **0.810** due to enlarged candidate search bounds.
- **`angle` (Maximum Turn Angle, deg)**:
  - *Default*: `120.0 deg`
  - *Tuning*: Decrease to `45.0–60.0 deg` for directed laminar flows to prune improbable sharp turns.

---

## 4. Cheat Sheet & Execution Commands

| Target Objective | Recommended Configuration / Strategy | Command |
|---|---|---|
| **Default Fast Pass** | Single Pass `priority_segment_3d` | `uv run python scripts/bench_trackers.py` |
| **High Density / Large Scale** | Spatial Grid Accelerated `priority_segment_3d` | `uv run python scripts/bench_trackers.py --density 5000` |

---

## 5. Future Verification & Benchmarking TODO Backlog

To continuously validate and expand OpenPTV2 tracking strategies on experimental datasets, the following verification tasks are slated for future releases:

- [ ] **Adaptive Velocity Thresholding**: Automatically infer `dvxmax`/`dvxmin` bounds per frame based on mean spatial displacement histograms prior to tracking.
- [ ] **Ensemble Consensus Voting**: Implement an $N$-tracker consensus ensemble where a candidate link is accepted only if at least $M$ out of $N$ trackers agree.
- [ ] **GPU-Accelerated Bipartite Matching**: Port global Hungarian cost matrix solvers to PyTorch/CuPy for $N > 50,000$ ultra-dense particle tracking.

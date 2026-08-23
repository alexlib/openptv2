# Comprehensive Tracker Benchmarking, Selection & Strategy Guide for OpenPTV2

## Executive Summary & Key Findings

OpenPTV2 features five 3D tracking engines, each engineered for distinct flow regimes, particle densities, and execution constraints. Based on empirical cross-tracker benchmarking on turbulent synthetic flow fields, **there is no single "best" tracker for every flow**, but rather an **optimal tracker or hybrid pipeline** depending on dataset characteristics.

---

## 1. Single-Pass Engine Benchmarks

*Evaluated on turbulent flow dataset (Mean $N \approx 220$ particles/frame, $v = 1.52\text{ mm/frame}, d_{\text{nn}} = 6.98\text{ mm}, M = 0.218$)*:

| Tracker Engine | Precision | Recall / Yield | Fragment Count ($F$) | Track Purity | Perfect Match % | Speed (ms/frame) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`priority_segment_3d`** *(Default)* | **0.974** | **0.901** | **3.56** | **0.970** | **73.7%** | **159.4 ms** |
| **`nearest_hungarian_3d`** | **0.984** | **0.904** | **3.53** | **0.982** | **76.2%** | **150.6 ms** |
| **`predictive_gmm_3d`** | 0.959 | 0.752 | 7.53 | 0.977 | **86.0%** | 732.3 ms |

---

## 2. Hybrid Cascading Strategies Benchmark

To go beyond single-pass limits, OpenPTV2 supports multi-pass hybrid cascading workflows (`scripts/demo_hybrid_strategies.py`):

| Hybrid Strategy Pipeline | Precision | Recall / Yield | Ghost % | Fragment Count ($F$) | Track Purity | Perfect Match % (PMT) | Speed (ms/frame) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Single Pass `priority_segment_3d`)** | **0.974** | **0.901** | 3.56% | 3.56 | **0.970** | 73.7% | **157.0 ms** |
| **Strategy 1: Forward-Fast / Backward-GMM** | **0.974** | **0.901** | **0.50%** | **3.15** | 0.965 | **93.6%** | 689.9 ms |
| **Strategy 2: Two-Scale Velocity Cascading** | 0.963 | 0.888 | **1.07%** | 3.67 | 0.938 | **87.0%** | 365.4 ms |

### Strategy Highlights
- **Strategy 1 (Forward Fast / Backward GMM)**: Increases **Perfect Match Trajectory Percentage (PMT%) from 73.7% to 93.6%** and reduces **Ghost capture rate from 3.56% down to 0.50%**.
- **Strategy 2 (Two-Scale Velocity Cascading)**: Achieves **87.0% PMT** and reduces Ghost capture to **1.07%** at nearly **double the speed** of Strategy 1 (365.4 ms/frame).

---

## 3. Tracker Selection Decision Matrix

```
                          [ What is your Flow Regime & Density? ]
                                            │
           ┌────────────────────────────────┴────────────────────────────────┐
           ▼                                                                 ▼
[ High / Medium Density ]                                         [ Low Density / Sparse ]
    (N > 500 / frame)                                                (N < 200 / frame)
           │                                                                 │
   ┌───────┴────────┐                                                        │
   ▼                ▼                                                        ▼
[ Smooth Flow ]  [ Strong Turbulence ]                             [ High Noise / Missing ]
   │                │                                                        │
   ▼                ▼                                                        ▼
priority_segment_3d nearest_hungarian_3d                             predictive_gmm_3d
```

### Condition-by-Condition Guide

| Flow Condition | Recommended Engine / Hybrid | Primary Reason |
|---|---|---|
| **High Density / Large Datasets** ($N > 1,000$) | `priority_segment_3d` | $O(N)$ spatial grid cell indexing provides 4x–10x faster runtime without precision loss. |
| **High Velocity Turbulences & Recirculation** | `nearest_hungarian_3d` | Global Hungarian bipartite matching prevents greedy assignment conflicts in dense eddies. |
| **Irregular Steps / High Noise** | `predictive_gmm_3d` | GMM-based predictive estimation handles complex trajectory curvatures and noise. |
| **Fragmented Tracks needing High Purity** | **Strategy 1 Hybrid** | **93.6% PMT** with **0.50% Ghost Capture rate**. |

---

## 4. Parameter Tuning Reference

### A. `priority_segment_3d` (Cython Engine)
- **`dacc` (Maximum Acceleration Window, mm)**:
  - *Default*: `5.5 mm`
  - *Tuning*: Do **not** expand `dacc` arbitrarily! Expanding `dacc` from `5.5` to `50` drops precision from **0.974** to **0.810** due to enlarged candidate search bounds.
- **`angle` (Maximum Turn Angle, deg)**:
  - *Default*: `120.0 deg`
  - *Tuning*: Decrease to `45.0–60.0 deg` for directed laminar flows to prune improbable sharp turns.

### B. `nearest_hungarian_3d` (MyPTV Kinematic Engine)
- **`search_radius` (Max Displacement Distance, mm)**:
  - *Default*: Derived from `dvxmax` (`15.5 mm`).
  - *Tuning*: Set search radius to $1.5 \times v_{\max} \Delta t$. Setting radius too large ($>3 \times v_{\max}$) degrades precision in dense regions.

---

## 5. Cheat Sheet & Execution Commands

| Target Objective | Recommended Configuration / Strategy | Command |
|---|---|---|
| **Default Fast Pass** | Single Pass `priority_segment_3d` | `uv run python scripts/bench_trackers.py` |
| **Maximum Trajectory Perfection (93.6% PMT)** | Hybrid Strategy 1 (Forward Fast / Backward Kalman) | `uv run python scripts/demo_hybrid_strategies.py` |
| **Fast Multi-Scale Flow (87.0% PMT)** | Hybrid Strategy 2 (Two-Scale Cascading) | `uv run python scripts/demo_hybrid_strategies.py` |
| **High Density / Large Scale** | Spatial Grid Accelerated `priority_segment_3d` | `uv run python scripts/bench_trackers.py --density 5000` |

---

## 6. Future Verification & Benchmarking TODO Backlog

To continuously validate and expand OpenPTV2 tracking strategies on experimental datasets, the following verification tasks are slated for future releases:

- [ ] **Real Experimental Dataset Benchmarking**: Validate Strategy 1 and Strategy 2 on physical wind tunnel / water channel multi-camera PTV datasets with ground truth or synthetic image projections.
- [ ] **Adaptive Velocity Thresholding**: Automatically infer `dvxmax`/`dvxmin` bounds per frame based on mean spatial displacement histograms prior to tracking.
- [ ] **Ensemble Consensus Voting**: Implement an $N$-tracker consensus ensemble where a candidate link is accepted only if at least $M$ out of $N$ trackers agree.
- [ ] **GPU-Accelerated Bipartite Matching**: Port global Hungarian cost matrix solvers to PyTorch/CuPy for $N > 50,000$ ultra-dense particle tracking.

# Actionable Implementation Plan: OpenPTV³ (NextGen Differentiable 3D-PTV & Auto-Research Engine)

## Executive Summary & Vision

OpenPTV³ represents the next evolutionary step in Particle Tracking Velocimetry: **transitioning from discrete, locally-tuned computer vision algorithms to an End-to-End Differentiable 3D-PTV Pipeline with Autonomous Physics Optimization**.

By rewriting core pipeline stages using PyTorch automatic differentiation and soft operators, OpenPTV³ enables **backpropagating gradients from downstream 3D Lagrangian turbulence physics metrics ($K_a$, $E_L(\omega)$, $g \epsilon t^3$) back through 100 processing steps to micro-parameters at Stage 1 (2D intensity thresholds, optical distortion parameters, and epipolar tolerances)**.

---

## Architectural Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 OpenPTV³ Differentiable Pipeline                                │
│                                                                                                 │
│  [ Stage 1: Soft-Thresholding ] ──► [ Stage 2: Soft-Argmax Centroiding ]                       │
│                                                   │                                             │
│                                                   ▼                                             │
│  [ Stage 4: Soft-Sinkhorn Tracking ] ◄── [ Stage 3: Differentiable Epipolar Ray Intersection ]  │
│                    │                                                                            │
│                    ▼                                                                            │
│  [ Stage 5: Differentiable Savitzky-Golay / Acceleration Calculation ]                         │
│                    │                                                                            │
│                    ▼                                                                            │
│  [ Lagrangian Physics Loss L_physics = w1|ΔKa| + w2|ΔE_L(ω)| - w3(T/τ_L) ]                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼ (Backprop Gradients ∂L / ∂Θ_1...5)
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   Auto-Research Optimization Engine (Local or Cloud GPU via MoLab)              │
│     Adjusts 50+ Hyperparameters Across All Stages Simultaneously via Adam / Bayesian / CMA-ES │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5-Phase Actionable Implementation Plan

### Phase 1: JHTDB Data Ingestion & Synthetic Optical Image Simulator
- **Goal**: Establish a ground-truth Direct Numerical Simulation (DNS) benchmark environment with end-to-end forward image rendering.
- **Module Locations**:
  - [`src/openptv2/benchmarking/jhtdb_client.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/benchmarking/jhtdb_client.py)
  - [`src/openptv2/benchmarking/synthetic_optical_projector.py`](file:///C:/Users/alex/projects/openptv2/src/openptv2/benchmarking/synthetic_optical_projector.py)
- **Key Tasks**:
  1. Implement REST/HDF5 client for Johns Hopkins Turbulence Database (JHTDB) $Re_\lambda \approx 433$ Homogeneous Isotropic Turbulence trajectories.
  2. Build synthetic optical projector that renders 2D multi-camera TIF images from 3D particle arrays including:
     - Gaussian Point Spread Function (PSF) intensity profiles.
     - Laser sheet Gaussian intensity attenuation and out-of-focus disappearance.
     - Refractions across air-glass-water interfaces.
     - Synthetic Gaussian sensor noise and ghost particle inclusions.

---

### Phase 2: OpenPTV³ PyTorch Differentiable Core Modules
- **Goal**: Implement PyTorch-backed soft-differentiable operators replacing discrete legacy algorithms.
- **Module Location**: `src/openptv2/differentiable/`
- **Key Tasks**:
  1. `centroiding.py`: `SoftArgmax2D` and `DifferentiableGaussianFit` for subpixel 2D target recognition.
  2. `geometry.py`: PyTorch-differentiable pinhole camera model with $k_1, k_2, p_1, p_2$ Brown-Conrady distortions and differentiable epipolar line intersection.
  3. `matching.py`: `SoftSinkhornMatcher` implementing entropy-regularized optimal transport for soft stereo correspondence.
  4. `tracking.py`: `DifferentiableSegmentTracker` and `DifferentiableSavitzkyGolay` for smooth velocity/acceleration derivation.

---

### Phase 3: Differentiable Lagrangian Loss & End-to-End Backprop Engine
- **Goal**: Formulate differentiable turbulence physics losses and execute end-to-end gradient updates.
- **Module Location**: `src/openptv2/differentiable/physics_loss.py`
- **Key Loss Formulation**:
  $$\mathcal{L}_{\text{total}} = w_1 |\Delta K_a| + w_2 |\Delta E_L(\omega)| - w_3 \min\left(1.0, \frac{\langle T \rangle}{20 \Delta t}\right) + w_4 \text{ReprojectionError} + w_5 \text{GhostPenalty}$$
- **Key Tasks**:
  1. Implement differentiable 4th-moment kurtosis loss $K_a = \frac{\mathbb{E}[a^4]}{\mathbb{E}[a^2]^2}$.
  2. Implement differentiable Fourier power spectral density loss $E_L(\omega)$.
  3. Validate end-to-end gradient flow: verify $\frac{\partial \mathcal{L}_{\text{total}}}{\partial I_{\text{threshold}}} \neq 0$.

---

### Phase 4: Auto-Research AI Agent & Global Sensitivity CLI (`openptv2-autotune`)
- **Goal**: Build an autonomous AI agent capable of tuning both differentiable PyTorch runtimes and legacy C/Cython runtimes via Global Sensitivity Analysis (Sobol/SHAP) and Bayesian Optimization.
- **Module Locations**:
  - `src/openptv2/autoresearch/agent.py`
  - `src/openptv2/autoresearch/sensitivity.py`
  - `src/openptv2/cli/autotune.py`
- **Key Tasks**:
  1. Implement Sobol variance decomposition for causal attribution between Stage 1 parameters and Stage 5 physics.
  2. Implement `openptv2-autotune` CLI command.
  3. Generate publication-ready Pareto Optimality plots comparing execution runtime vs. Lagrangian physics accuracy (`LTPS`).

---

### Phase 5: MoLab (`molab.marimo.io`) Cloud GPU Execution & Interactive Dashboard
- **Goal**: Provide a zero-setup, zero-install, cloud GPU-accelerated Auto-Research interactive dashboard via `molab.marimo.io`.
- **Module Location**: `notebooks/marimo_autoresearch_dashboard.py`
- **Key Features**:
  1. **Direct One-Click GitHub URL Execution**: Users launch the notebook instantly in their browser with zero local setup by opening:
     `https://molab.marimo.io/github/alexlib/openptv2/blob/main/notebooks/marimo_autoresearch_dashboard.py`
  2. **Free NVIDIA Cloud GPU Attachment**: Leverage CoreWeave-backed cloud GPUs on MoLab to run PyTorch backpropagation ($\nabla_{\mathbf{\Theta}} \mathcal{L}_{\text{physics}}$) and Sinkhorn optimal transport at high throughput ($N > 10,000$).
  3. **High-Bandwidth JHTDB Cloud Stream**: Ingest JHTDB DNS datasets directly within the cloud datacenter without local network download delays.
  4. **Reactive Real-Time Physics Dashboard**: Dragging UI sliders dynamically re-executes downstream cells to display live acceleration PDFs ($K_a$), velocity power spectra ($E_L(\omega)$), and interactive 3D particle trajectory plots (`wigglystuff`).

---

## Verification & Validation Strategy

1. **Unit Test Coverage**: `tests/unit/test_differentiable_*.py` verifying PyTorch operator gradient checks via `torch.autograd.gradcheck`.
2. **Parity Tests**: Ensure that as soft-argmax temperature $\tau \to 0$ and Sinkhorn temperature $\tau \to 0$, OpenPTV³ differentiable outputs converge to legacy Cython OpenPTV2 outputs.
3. **Physical Validation Benchmark**: Demonstrate that `openptv2-autotune` reduces acceleration kurtosis bias $\Delta K_a$ by $>80\%$ compared to default manual parameter tuning on JHTDB datasets.
4. **MoLab Cloud Interoperability**: Validate one-click execution of `marimo_autoresearch_dashboard.py` on `molab.marimo.io` with GPU enabled.

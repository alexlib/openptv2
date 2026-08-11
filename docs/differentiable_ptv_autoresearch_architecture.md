# End-to-End Differentiable 3D-PTV & Auto-Research Architecture
## Bridging Micro-Parameters Across 100-Step Pipelines to Lagrangian Turbulence Physics

---

## Executive Summary & Conceptual Breakthrough

A complete 3D Particle Tracking Velocimetry (3D-PTV) pipeline consists of **five tightly coupled processing stages**:

$$\text{Raw Multi-Cam Images} \xrightarrow{\text{Stage 1}} \text{2D Targets} \xrightarrow{\text{Stage 2}} \text{Subpixel Centroids} \xrightarrow{\text{Stage 3}} \text{Stereo 3D Points} \xrightarrow{\text{Stage 4}} \text{3D Trajectories} \xrightarrow{\text{Stage 5}} \text{Lagrangian Physics}$$

Historically, each stage has been optimized in isolation (e.g., Stage 2 minimizes 2D reprojection error, Stage 4 maximizes frame-to-frame link yield). However, **stage-wise local optimization does not imply optimal Lagrangian turbulence physics quality.**

A minute $0.05\text{ px}$ subpixel centroid bias introduced at Stage 2 or a subtle $0.02\text{ mm}$ epipolar line calibration error at Stage 3 propagates through 3D triangulation and frame linkage to produce false crossing swaps. When differentiated twice ($\mathbf{a} = d^2\mathbf{x}/dt^2$), a single false swap generates a massive artificial acceleration spike $a \propto 1/\Delta t^2$ that **distorts the high-kurtosis non-Gaussian intermittency tail ($K_a$) of fluid turbulence by over 300%!**

To bridge a seemingly "irrelevant" hyperparameter 100 steps upstream (e.g., a 2D intensity peak threshold) directly to the ultimate fluid turbulence outcome ($K_a$, $E_L(\omega)$, $g \epsilon t^3$), we propose a **Differentiable 3D-PTV Pipeline Architecture** combined with an **Auto-Research Sensitivity Agent**.

---

## 1. The Multi-Stage Error Propagation Cascade (The "Butterfly Effect" in PTV)

```
[ Stage 1: Preprocessing & Thresholding ]
  └─ Micro-Parameter: Intensity Threshold I_th = 12.4 vs 12.0
        │
        ▼ (propagates to)
[ Stage 2: 2D Target Detection & Centroiding ]
  └─ Artifact: 0.05 px Subpixel Centroid Shift / Asymmetric Blob Clipping
        │
        ▼ (propagates to)
[ Stage 3: Stereo Epipolar Matching & Triangulation ]
  └─ Artifact: Epipolar Miss (d_epip > tolerance) → Missing 3D Point or 0.03 mm Triangulation Bias
        │
        ▼ (propagates to)
[ Stage 4: 3D Frame-to-Frame Linkage & Tracking ]
  └─ Artifact: Temporary Track Gap or False "Crossing Swap" Assignment
        │
        ▼ (propagates through 2nd Derivative d²/dt²)
[ Stage 5: Lagrangian Turbulence Physics Outcomes ]
  └─ Physical Distortion: Massive False Acceleration Spike → Kurtosis Bias ΔK_a = +18.5
```

### Why Local Step-Wise Optimization Fails
1. **Minimizing Reprojection Error $\neq$ Maximizing Turbulence Quality**: A calibration parameters set that minimizes mean 2D reprojection error may smooth out steep spatial gradients, clipping real turbulent velocity fluctuations.
2. **Non-Linear Amplification via Differentiation**: Position errors $\delta x$ amplify quadratically in acceleration $\delta a \approx \delta x / \Delta t^2$. High frame-rate cameras ($\Delta t \to 0$) drastically exacerbate this amplification.
3. **High-Dimensional Parameter Coupling**: Changing a threshold in Stage 1 invalidates the optimal gating radius in Stage 4. Manual tuning of 50+ coupled parameters across 5 stages is humanly intractable.

---

## 2. Theoretical Frameworks for End-to-End Auto-Research

How can an automated system discover the subtle relationship between a Stage 1 micro-feature and a Stage 5 turbulent physics metric 100 steps away? We propose **three complementary methodologies**:

### Method A: End-to-End Differentiable PTV via Automatic Differentiation (PyTorch / JAX)

By replacing hard/discrete operations with **soft, differentiable operators**, the entire 3D-PTV pipeline becomes a single continuous function $f(\mathbf{\Theta})$ that admits exact backpropagation gradients $\nabla_{\mathbf{\Theta}} \mathcal{L}_{\text{physics}}$:

```
                  ┌────────────────────────────────────────────────────────┐
                  │          Lagrangian Turbulence Loss Function           │
                  │  L_physics = w1|ΔKa| + w2|ΔE_L(ω)| - w3(T / tau_L)    │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼  ∂L / ∂x(t)
                  ┌────────────────────────────────────────────────────────┐
                  │    Differentiable 3D Savitzky-Golay / Kalman Filter   │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼  ∂L / ∂X_3D
                  ┌────────────────────────────────────────────────────────┐
                  │  Differentiable Sinkhorn Matching (Soft Bipartite)     │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼  ∂L / ∂x_2D
                  ┌────────────────────────────────────────────────────────┐
                  │  Differentiable Epipolar Ray Intersection & Triangulation│
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼  ∂L / ∂I_threshold
                  ┌────────────────────────────────────────────────────────┐
                  │  Differentiable Soft-Argmax 2D Centroiding             │
                  └────────────────────────────────────────────────────────┘
```

#### Key Differentiable Primitives:
1. **Differentiable 2D Centroiding (Soft-Argmax)**:
   $$\mathbf{x}_{\text{subpixel}} = \sum_{(u,v) \in \Omega} \frac{\exp(I(u,v)/\tau)}{\sum \exp(I/\tau)} \cdot \begin{pmatrix} u \\ v \end{pmatrix}$$
2. **Differentiable Bipartite Matching (Sinkhorn-Knopp)**:
   Replaces discrete Hungarian matching with soft-assignment matrix $P = \text{Sinkhorn}(-C/\tau)$, enabling continuous gradient flow through correspondence assignment.
3. **End-to-End Gradient Backpropagation**:
   $$\frac{\partial \mathcal{L}_{\text{physics}}}{\partial \mathbf{\Theta}_{\text{stage1}}} = \frac{\partial \mathcal{L}_{\text{physics}}}{\partial \mathbf{a}(t)} \cdot \frac{\partial \mathbf{a}(t)}{\partial \mathbf{x}_{3D}} \cdot \frac{\partial \mathbf{x}_{3D}}{\partial \mathbf{x}_{2D}} \cdot \frac{\partial \mathbf{x}_{2D}}{\partial \mathbf{\Theta}_{\text{stage1}}}$$

---

### Method B: Global Causal Sensitivity Attribution (Sobol Variance & SHAP Analysis)

When analytical gradients are unavailable for legacy C/Cython algorithms, the Auto-Research agent employs **Global Sensitivity Analysis (GSA)** to compute Sobol variance indices $S_i$:

$$S_i = \frac{\text{Var}_{\Theta_i}\left(\mathbb{E}_{\mathbf{\Theta}_{\sim i}}\left[\mathcal{L}_{\text{physics}} \mid \Theta_i\right]\right)}{\text{Var}\left(\mathcal{L}_{\text{physics}}\right)}$$

- **First-Order Index $S_i$**: Measures the direct influence of micro-parameter $\Theta_i$ (e.g., 2D threshold) on the downstream acceleration kurtosis $K_a$.
- **Total Interaction Index $S_{Ti}$**: Quantifies complex non-linear couplings between Stage 1 parameters and Stage 4 tracking parameters.

---

### Method C: Physics-Informed Metamodel / Surrogate Auto-Research Agent

Train a Graph Neural Network (GNN) surrogate model $\widehat{\mathcal{M}}(\mathbf{\Theta}_1, \dots, \mathbf{\Theta}_5) \to \text{LTPS}$ on Direct Numerical Simulation (DNS) ground-truth datasets (such as the **Johns Hopkins Turbulence Database - JHTDB**):

1. **JHTDB Ground Truth**: Extract 3D fluid particle trajectories with exact position $\mathbf{x}(t)$, velocity $\mathbf{u}(t)$, and acceleration $\mathbf{a}(t)$.
2. **Forward Degradation Simulator**: Synthesize degraded multi-camera images with realistic noise, out-of-focus blur, calibration drift, and particle overlap.
3. **Surrogate Optimization Loop**: Use Bayesian Optimization / CMA-ES on the neural surrogate to discover optimal non-intuitive hyperparameter combinations across all 5 stages simultaneously.

---

## 3. The Comprehensive Auto-Research Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        1. Johns Hopkins Turbulence Database (JHTDB)                    │
│             Homogeneous Isotropic Turbulence (Re_λ ≈ 433) / Forced MHD DNS             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        2. Synthetic Multi-Camera Optical Projection                    │
│   Project 3D DNS Particles → 2D Multi-Camera Images with Synthetic Gaussian PSF Blur,   │
│   Ray-Tracing Refraction, Intensity Dips, Out-of-Focus Disappearance, & Ghosts         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        3. Full 5-Stage OpenPTV2 Processing Pipeline                    │
│   Stage 1: Preprocessing Parameters  ──► Thresholds, Filter Kernels, Background        │
│   Stage 2: Target Detection          ──► Peak Search, Subpixel Window, Min Area          │
│   Stage 3: Epipolar Calibration      ──► Epipolar Band Tolerance, Radial k1, Distortion   │
│   Stage 4: Correspondence Matching   ──► Quad/Triple Search Radii, Ray Intersection Dist  │
│   Stage 5: Trajectory Tracking       ──► priority_segment_3d, dvxmax, dacc, Angle        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        4. Lagrangian Turbulence Physics Evaluator                      │
│   • Acceleration Kurtosis Error: ΔKa = |Ka_measured - Ka_DNS|                          │
│   • Velocity Spectrum High-Freq Noise Floor: E_L(ω) deviation                          │
│   • Trajectory Continuity: Fraction of tracks spanning T > τ_L                         │
│   • Pair Dispersion Scaling: Richardson Constant Error Δg                              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   5. Auto-Research Parameter Discovery Engine                           │
│   Compute Backprop Gradients (∂LTPS / ∂Θ) or Sobol Attribution → Update Θ1...Θ5        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Roadmap for OpenPTV2

To operationalize this Auto-Research paradigm in future releases:

1. **Phase 1: JHTDB Direct Integration**: Create a benchmarking loader `openptv2.benchmarking.jhtdb` to fetch high-Reynolds-number DNS trajectories directly from JHTDB web services.
2. **Phase 2: Full-Pipeline Synthetic Generator**: Expand `openptv2.benchmarking` to project 3D DNS trajectories through camera matrices $P_k$ into synthetic 2D TIF image sequences.
3. **Phase 3: Differentiable PyTorch Core Modules**: Implement `openptv2.differentiable` providing PyTorch-backed soft-argmax centroiding, differentiable epipolar triangulation, and soft Sinkhorn tracking.
4. **Phase 4: Global Sensitivity & Auto-Tuning CLI**: Introduce `openptv2-autotune` CLI that automatically searches and tunes all 50+ pipeline parameters for user-provided flow regimes.

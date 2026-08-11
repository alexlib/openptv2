# Lagrangian Turbulence Quality Guide & Auto-Research Roadmap for OpenPTV2

## Executive Summary & Objective

Computer vision metrics such as **Precision, Recall, Yield, and Frame-to-Frame Link Count** quantify local association accuracy, but **they do not measure real physical "quality" for Lagrangian turbulence research.** 

A tracking algorithm with $98\%$ frame-to-frame precision can still create $2\%$ false crossing swaps, generating unphysical velocity jumps $\Delta v \approx d_{\text{nn}} / \Delta t$ and massive acceleration spikes $a \propto 1/\Delta t^2$. These spurious spikes corrupt the non-Gaussian intermittency tails of the acceleration probability density function (PDF), invalidate velocity autocorrelation integrals $R_v(\tau)$, and ruin pair dispersion scaling $\langle \delta r^2(t) \rangle \sim g \epsilon t^3$.

This document defines the **Lagrangian Turbulence Physics Quality Framework** and outlines an **Auto-Research Workflow** leveraging ground-truth fluid trajectories (e.g., from the **Johns Hopkins Turbulence Database - JHTDB**) to evaluate and identify the ultimate tracker configurations balancing **computational speed** and **real turbulent study quality**.

---

## 1. Technical Computer Vision Metrics vs. Physical Turbulence Quality

| Technical Metric | What It Measures | Why It Is Incomplete for Lagrangian Turbulence |
|---|---|---|
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | High precision does not guarantee physical derivative continuity. A single false link creates an extreme acceleration outlier that corrupts turbulent intermittency statistics. |
| **Recall / Yield** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | High recall with short, fragmented tracks ($L < \tau_\eta$, Kolmogorov scale) is useless for computing Lagrangian integral timescales $\tau_L$ or diffusion coefficients $D_L$. |
| **Track Count** | Total generated tracks | Splitting one long physical trajectory into 10 short fragments increases track count but destroys pair dispersion and velocity autocorrelation calculations. |
| **Track Purity** | Fraction of points from same true ID | Ignores gap-bridging capabilities across missing frames (e.g., laser sheet speckle or out-of-focus fade). |

---

## 2. The 5 Core Physical Criteria for Lagrangian Turbulence Quality

To rigorously validate whether a tracker or hybrid cascading strategy produces physically sound trajectories, evaluation must incorporate the following five fluid mechanics criteria:

### A. Trajectory Lifetime Distribution & Integral Scale Span ($\langle T \rangle / \tau_L$)
- **Physical Meaning**: Lagrangian velocity autocorrelations $R_v(\tau) = \langle v(t)v(t+\tau)\rangle$ and structure functions $D_p(\tau) = \langle |v(t+\tau) - v(t)|^p\rangle$ require continuous trajectories spanning multiple Lagrangian integral timescales $\tau_L$.
- **Target Metric**: Mean track duration $\langle T \rangle$, and the fraction of tracks exceeding $T > 10 \Delta t$, $T > 30 \Delta t$, and $T > \tau_L$.

### B. Acceleration PDF Fidelity & Intermittency Kurtosis ($K_a$)
- **Physical Meaning**: Fluid acceleration in intense turbulence is violently intermittent, characterized by heavy-tailed non-Gaussian PDFs with high kurtosis ($K_a = \langle a^4 \rangle / \langle a^2 \rangle^2 \approx 10 \dots 50$). Spurious track switches introduce artificial acceleration outliers that artificially inflate $K_a$.
- **Target Metric**: **Kurtosis Error Bias**: $\Delta K_a = |K_{a,\text{pred}} - K_{a,\text{true}}|$.

### C. Velocity Power Spectral Density (PSD) & Energy Cascade
- **Physical Meaning**: In the inertial subrange, the Lagrangian velocity spectrum follows Kolmogorov scaling $E_L(\omega) \propto \omega^{-2}$.
- **Target Metric**: **High-Frequency Noise Floor**. Spatial jitter and false links manifest as flat white noise at high frequencies $\omega > 1/\tau_\eta$.

### D. Relative Pair Dispersion & Richardson Constant ($g$)
- **Physical Meaning**: Pairs of fluid particles with initial separation $r_0$ undergo exponential separation (Batchelor regime), followed by cubic explosive dispersion $\langle |r(t) - r(0)|^2 \rangle = g \epsilon t^3$ (Richardson-Obukhov regime).
- **Target Metric**: **Pair Identity Swap Rate & Richardson Constant Error ($\Delta g$)**.

### E. Gap-Bridging & Intensity Dip Resilience
- **Physical Meaning**: Particle intensity drops below detection thresholds for $1 \dots 3$ frames due to laser sheet non-uniformity or out-of-focus motion.
- **Target Metric**: **Gap Re-link Recall Rate**—the percentage of interrupted trajectories correctly re-identified after 1–3 missing frames without resetting particle index.

---

## 3. Auto-Research Validation Pipeline Architecture

To automate the discovery of optimal tracking strategies combining **execution speed** and **turbulent physics quality**, OpenPTV2 supports an end-to-end Auto-Research benchmark pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              1. Ground Truth Generation (e.g. JHTDB)                    │
│   Direct Numerical Simulation (DNS) Direct Fluid Particle Trajectories  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              2. Synthetic Experiment Projection & Degradation           │
│   Add Gaussian Position Jitter, Intensity Dips, Out-of-Focus Missing    │
│   Detections, and Synthetic Ghost Particles                             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              3. OpenPTV2 Tracking Engine & Strategy Execution           │
│   • Single-Pass Engine (priority_segment_3d, nearest_hungarian_3d, etc.)│
│   • Hybrid Cascading Strategy 1 (Forward-Fast / Backward-Kalman)        │
│   • Hybrid Cascading Strategy 2 (Two-Scale Velocity Cascading)          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              4. Dual-Layer Performance & Physics Evaluator              │
│   • Layer A: Technical Metrics (Precision, Recall, Ghost%, ms/frame)    │
│   • Layer B: Lagrangian Physics Metrics (T/tau_L, ΔKa, PSD Noise Floor) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              5. Pareto Optimality Decision Engine                       │
│   Select Ultimate Strategy on Speed vs. Lagrangian Physics Pareto Front │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Combined Quality Score Formulation

We define a holistic **Lagrangian Turbulence Performance Score (LTPS)**:

$$\text{LTPS} = w_1 \cdot \text{Precision} + w_2 \cdot \text{PMT\%} + w_3 \cdot \min\left(1.0, \frac{\langle T \rangle}{20 \Delta t}\right) - w_4 \cdot \frac{|\Delta K_a|}{K_{a,\text{true}}} - w_5 \cdot \text{Ghost\%}$$

Where:
- $w_1 = 0.25$ (Technical Precision)
- $w_2 = 0.25$ (Perfect Match Trajectories)
- $w_3 = 0.25$ (Normalized Mean Track Length)
- $w_4 = 0.15$ (Acceleration Kurtosis Fidelity)
- $w_5 = 0.10$ (Ghost Capture Penalty)

---

## 5. Roadmap for Future Auto-Research Campaigns

- [ ] **JHTDB Ingestion Interface**: Connect `openptv2.benchmarking` to extract 3D Lagrangian trajectories from JHTDB Homogeneous Isotropic Turbulence ($Re_\lambda \approx 433$) or Forced MHD Turbulence datasets.
- [ ] **Synthetic Camera Projection**: Project 3D DNS particles onto multi-camera 2D image planes using realistic optical calibration matrices and point spread functions (PSF).
- [ ] **Automated Parameter Sweeps**: Run Bayesian optimization over tracker parameter spaces (`dacc`, `gate_threshold`, `search_radius`) using `LTPS` as the objective function.
- [ ] **Pareto Frontier Visualization**: Generate publication-ready Pareto plots comparing Execution Time (ms/frame) vs. Lagrangian Physics Quality (`LTPS`).

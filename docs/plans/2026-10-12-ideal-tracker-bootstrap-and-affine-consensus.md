# Towards the Ideal Tracker: Bootstrapped Velocity & Local Affine Consensus (Phase 2)

**Author:** Alex & DeepMind Antigravity  
**Date:** October 12, 2026  
**Status:** In Progress (Branch: `feat/two-phase-bootstrap-affine-consensus`)  
**Context:** Extension of `TwoPhaseTracker` in `openptv2` following initial Spatial Neighborhood Consensus (VFC) implementation.

---

## 1. Executive Summary & Diagnostic Findings

In Phase 1, Spatial Neighborhood Consensus (VFC, Options A & B) was implemented, reducing false links by **75%–80%** across ghost densities up to 200%. However, detailed diagnostics of the false links that **escaped** the consensus analysis revealed two primary physical failure modes:

1. **The Frame 0 Cold-Start Blind Spot (>50% of all escaped links):**  
   At $t=0$, newborn tracks have no historical trajectory ($|\text{hist}| < 2$) and are initialized with zero velocity ($\mathbf{v} = 0$). As a result, no velocity consensus can be established between frame 0 and frame 1, allowing ghosts to hijack newborn tracks before any velocity direction is formed.

2. **First-Order Continuum Deformations (Shear & Vorticity):**  
   The initial VFC assumed that velocity within radius $R$ is locally uniform ($\mathbf{v}(\mathbf{x}) \approx \bar{\mathbf{v}}$). In shear layers, boundary layers, and vortex cores, physical continuum gradients $|\nabla \mathbf{u}| \cdot R$ force the consensus gate (`vfc_gate`) to remain wide (e.g., $1.5\text{ mm}$), allowing co-flowing ghosts inside the tolerance band to escape rejection.

This plan specifies the architecture and implementation for **Phase 2**:
- **Phase 2A:** **Cold-Start Velocity Bootstrapping** via local spatial displacement mode extraction.
- **Phase 2B:** **Local Affine Consensus ($\nabla \mathbf{u}$)** with divergence-free incompressibility constraint ($\text{tr}(\nabla \mathbf{u}) = 0$).

---

## 2. Phase 2A: Cold-Start Velocity Bootstrapping

### Mathematical Formulation

Given 3D particle positions $\mathbf{X}_0 \in \mathbb{R}^{N_0 \times 3}$ and $\mathbf{X}_1 \in \mathbb{R}^{N_1 \times 3}$:

1. For any position $\mathbf{x} \in \mathbb{R}^3$ (or globally over spatial sub-volumes of size $R_{\text{boot}}$):
   - Query candidate pairs $(i, j)$ such that $\|\mathbf{X}_{0, i} - \mathbf{x}\| \le R_{\text{boot}}$ and $\|\mathbf{X}_{1, j} - \mathbf{X}_{0, i}\| \le v_{\max}$.
   - Each candidate pair defines a displacement candidate $\mathbf{d}_{ij} = (\mathbf{X}_{1, j} - \mathbf{X}_{0, i}) / \Delta t$.
2. In real fluid flows, even in high ghost densities, true particle pairs share a coherent displacement vector $\mathbf{U}(\mathbf{x})$, while spurious/ghost pairs scatter isotropically in 3D velocity space.
3. Compute the local spatial mode:
   $$\mathbf{U}_{\text{boot}}(\mathbf{x}) = \text{mode} \left(\{ \mathbf{d}_{ij} \}_{\|\mathbf{X}_{0, i} - \mathbf{x}\| \le R_{\text{boot}}} \right)$$
   using 3D histogram binning or medoid-clustering over candidate displacements.
4. **Initialization:**
   - At $t=0$, tracks are initialized with $\mathbf{v} = \mathbf{U}_{\text{boot}}(\mathbf{x}_i)$ instead of $\mathbf{v} = 0$.
   - Active track predictions at frame 1 become $\mathbf{x}_{\text{pred}} = \mathbf{x}_0 + \mathbf{U}_{\text{boot}}(\mathbf{x}_0) \Delta t$.
   - Spatial consensus (Option A & B) can be enforced immediately at frame 0 $\to$ 1!

---

## 3. Phase 2B: Local Affine Consensus ($\nabla \mathbf{u}$ Incompressible Tensor)

### Mathematical Formulation

Within a local neighborhood $\mathcal{N}(\mathbf{x}_0) = \{ j : \|\mathbf{x}_j - \mathbf{x}_0\| \le R \}$, fluid velocity follows a Taylor expansion:

$$\mathbf{v}(\mathbf{x}) = \mathbf{v}_0 + \mathbf{A} (\mathbf{x} - \mathbf{x}_0) + \mathcal{O}(\|\mathbf{x} - \mathbf{x}_0\|^2)$$

where:
- $\mathbf{v}_0 = \bar{\mathbf{v}} \in \mathbb{R}^3$ is the local translation velocity.
- $\mathbf{A} = \nabla \mathbf{u} \in \mathbb{R}^{3 \times 3}$ is the local velocity gradient tensor.

For an incompressible fluid:
$$\nabla \cdot \mathbf{u} = \text{tr}(\mathbf{A}) = A_{11} + A_{22} + A_{33} = 0$$

### Constrained Least Squares Estimation

Given $K \ge 6$ neighboring particles with positions $\mathbf{r}_k = \mathbf{x}_k - \mathbf{x}_0$ and velocities $\mathbf{v}_k$:

We solve:
$$\min_{\mathbf{v}_0, \mathbf{A}} \sum_{k=1}^K w_k \|\mathbf{v}_k - (\mathbf{v}_0 + \mathbf{A} \mathbf{r}_k)\|^2 \quad \text{subject to} \quad \text{tr}(\mathbf{A}) = 0$$

where $w_k = \exp(-\|\mathbf{r}_k\|^2 / R^2)$ or a Tukey biweight for outlier robustness.

### Consensus Evaluation with Affine Prediction

For any candidate link with displacement $\mathbf{d}_i$:
$$\Delta \mathbf{v}_i = \left\| \mathbf{d}_i - \left( \mathbf{v}_0 + \mathbf{A} (\mathbf{x}_i - \mathbf{x}_0) \right) \right\|$$

Because linear shear, strain, and solid-body rotation are accounted for by $\mathbf{A}$, the residual $\Delta \mathbf{v}_i$ reflects only measurement noise and high-order turbulence, allowing the gate tolerance to be tightened:
$$vfc\_gate_{\text{affine}} \approx 3 \sigma_{\text{noise}} \ll |\nabla \mathbf{u}| R$$

---

## 4. Configuration & API Specification

Add the following fields to `TwoPhaseTrackerConfig`:

```python
@dataclass
class TwoPhaseTrackerConfig:
    # Existing VFC fields...
    vfc_mode: str = "off"
    vfc_radius: float = 10.0
    vfc_min_neighbors: int = 3
    vfc_gate: float | None = None
    vfc_weight: float = 0.0
    vfc_post_k: float = 3.0
    vfc_post_min_tol: float = 0.5
    vfc_post_reject_cold_start: bool = True

    # New Phase 2 Fields:
    vfc_bootstrap: bool = False
    vfc_bootstrap_radius: float = 15.0
    vfc_bootstrap_grid_size: float = 20.0
    vfc_affine: bool = False
    vfc_affine_min_neighbors: int = 6
```

---

## 5. Implementation Roadmap

1. **Phase 2A Implementation:**
   - Helper function `compute_bootstrap_velocity_field(pts0, pts1, v_max, dt, grid_size, radius)`.
   - Wire into `TwoPhaseTracker._track_unidirectional` at $t=0$ and for unlinked newborn tracks.
2. **Phase 2B Implementation:**
   - Helper function `fit_local_affine_field(positions, velocities, query_positions, radius, min_neighbors)`.
   - Update `compute_local_consensus_field` to use affine prediction when `vfc_affine=True`.
3. **Verification & Benchmarking:**
   - Unit tests covering 3D shearing flow, vortex rotation, and divergence-free constraint.
   - Benchmark script run comparing Baseline vs. VFC vs. VFC+Bootstrap+Affine across ghost density sweeps.

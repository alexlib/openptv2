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
    vfc_bootstrap_grid_size: float = 20.0
    vfc_affine: bool = False
    vfc_affine_min_neighbors: int = 6
```

> `vfc_bootstrap_radius` was dropped: the mode extraction searches all candidates
> within `v_max`, so a separate radius was never needed (and was never used).

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

### 5.1 What is implemented (2026-10-19)

| Item | State |
|---|---|
| `compute_bootstrap_velocity_field` | implemented (`src/openptv2/plugins/two_phase_tracking.py`) |
| Bootstrap of the **frame-0** tracks | implemented |
| Bootstrap of the field from the **tracks of a first pass** (two-pass) | implemented, `vfc_bootstrap_passes: 2` (default), see §5.3 |
| Bootstrap of **newborn tracks later in the sequence** | *not implemented, deliberately* — a newborn has no history and starts at rest, as before Phase 2A. Its first link is protected by the VFC consensus of its *neighbours* (which have established velocities); the newborn's own velocity only matters for its next prediction, and by then it has one. |
| `fit_local_affine_divergence_free` (Phase 2B) | implemented, used by `compute_local_consensus_field` when `vfc_affine=True` |
| Unit tests | `tests/unit/test_two_phase_vfc.py` (Phase 1 + 2B), `tests/unit/test_two_phase_bootstrap.py` (Phase 2A) |
| Benchmark numbers archived in `bench/` | *open* |

### 5.2 Known limitation of the mode extraction (open, measured 2026-10-19)

`compute_bootstrap_velocity_field` estimates the cell velocity as the **median of
the displacements inside the winning histogram bin** of an 8³ histogram over
`[-v_max, v_max]³`. The premise of the plan (§2) is that the true pairs form a
coherent displacement spike while the spurious pairs scatter isotropically.

**Measured behaviour** (median |v_est - v_true| over the real particles, mm):

| case | grid | result |
|---|---|---|
| uniform flow, sparse, no ghosts | 30 mm | **0.000** (exact) |
| uniform flow, sparse, +100 % ghosts | 30 mm | **0.000** (exact) |
| uniform flow, dense (ρ = 0.12 /mm³), ±100 % ghosts | 4 mm | **0.000** (exact) |
| mild shear a = 0.1, v_max = 3.8 | 4 mm | 0.087, recovered shear slope 0.080 (80 % of the truth) |
| strong shear a = 0.4, v_max = 7.7, dense | 4 mm | slope 0.334 (84 %), median error 4.6 mm |
| strong shear a = 0.4, v_max = 4 (true speed 6.4 > v_max) | any | 3.5 mm (the true partner is outside the search radius) |

**Why it fails.** The estimator is a *mode over a mixture*: per cell there is
one true displacement per source particle (K samples) and ~K·c spurious ones
(c = candidates per particle). The true samples pile up in one bin **only if
the flow is locally uniform over the cell**; in a shear they are spread over a
*ridge* of bins, so no bin holds a clear majority and the argmax can land on a
spurious bin. Two further effects, both measured:

* the spurious population is concentrated on the **shell** |d| ≈ v_max (the
  candidate set is selected by |p1_j - p0_i| ≤ v_max), so it is not the uniform
  clutter the plan assumed;
* the displacement cloud is **centred at 0 by construction** (the candidates
  are selected by their displacement), so any estimator of the cloud's *centre*
  (mean, coordinate-wise median) returns ≈ 0 instead of the true velocity.

**Alternatives that were measured and rejected** (4000 particles, 16 mm box,
v_max = 4, grid = 4 mm):

| estimator | uniform | shear | vortex |
|---|---|---|---|
| current: mode bin + median | **0.000** | 1.12 | 0.07 |
| coordinate-wise median of the candidate displacements | 0.97 | 2.82 | 0.69 |
| median seed + mode in a narrow window around it | **0.000** | 3.03 | 0.08 |
| robust affine fit (IRLS/Tukey) over the candidate pairs | 1.00 | 3.65 | 0.75 |
| 32³ histogram + 3×3×3 peak neighbourhood | — | 3.54 | 0.10 |

The affine fit fails for a structural reason: the inlier fraction is
1/candidates-per-particle (a few percent), so the first least-squares fit is
dominated by the outliers and the robust iteration collapses onto them.

**The direction that does work.** Estimating the field from *clean* velocities
instead of from raw candidates. The Phase 2B machinery
(`fit_local_affine_divergence_free`) recovers a linear shear **exactly**
(`tests/unit/test_two_phase_vfc.py::test_fit_local_affine_divergence_free_exact`),
because the established velocities carry no candidate clutter.

**Options:**

1. ~~**Two-pass bootstrap.**~~ **Implemented (2026-10-19).** Track once with
   v = 0 at frame 0; build the frame-0 field from the *established* tracks of
   that pass (local consensus, affine when `vfc_affine` is on); re-track with
   that field as the initial velocity.  The tracker's own assignment does the
   hard part, the bootstrap only interpolates a clean field.  Cost: two
   tracker runs (2x time).
2. **Keep the mode extraction, but only trust it where it is sharp.** The peak
   sharpness (peak count vs. the number of bins with significant counts) is a
   natural confidence measure: where the pile-up is sharp the field is exact,
   where it is smeared the frame-0 gate is simply not applied and the tracker
   falls back to its default behaviour.  Cheap, honest, no extra tracking cost.
3. **Tighten the candidate search for the bootstrap only.** The mode loses to
   the shell population as `v_max` grows; a bootstrap-specific, tighter radius
   (the dropped `vfc_bootstrap_radius` had exactly this meaning) would cut the
   spurious population.  It does not fix the shear, but it makes the estimator
   usable over a larger part of the parameter space.
4. **Do nothing now.** The feature is off by default; the cost of the
   limitation is a wrong initial velocity in sheared regions, which the gate
   (Option 2) can absorb.

### 5.3 The two-pass bootstrap (implemented)

`vfc_bootstrap_passes` selects how the initial field is obtained:

* `1` — the displacement mode of the frame 0 -> 1 candidate pairs (the old
  behaviour, one pass).
* `2` (default) — two passes.  The first pass tracks the sequence from rest;
  the field for the second pass is the local consensus of the *established*
  tracks of that pass (the links of the cold-start step are left out, they
  are the least trustworthy).  With `vfc_affine` the consensus is the
  divergence-free affine fit, which reproduces a linear shear **exactly**.

**Measured on a shear flow** (96 particles in a 24 mm box, vx = a*y, the VFC
gate active with `vfc_gate = 0.5`):

| a | ghosts | 1 pass: links / frame-0 accuracy | 2 passes: links / frame-0 accuracy |
|---|---|---|---|
| 0.08 | 0 | 250 / 0.74 | **288 / 1.00** |
| 0.12 | 0 | 136 / 0.45 | **288 / 1.00** |
| 0.12 | +50 % | 275 / 0.47 | **368 / 1.00** |
| 0.20 | 0 | 69 / 0.21 | **288 / 1.00** |

The biased field of the single pass does not only give a wrong velocity, it
makes the *gate* cut the true candidates: at a = 0.12 only 136 of the 288
links survive.  With the two-pass field all of them do, and the frame-0 links
are the true ones.

**The guard.** The two-pass field is built from the tracks of the first pass,
and the first link of a track is made without a velocity.  It is only
trustworthy when the flow moves the particles by clearly less than their
spacing; above that the first link is a random neighbour and the field is
scrambled (measured: at a displacement of 1.5x the spacing the two-pass field
is *worse* than the displacement mode, 2.35 vs 1.42 mm of error).  The guard
compares the displacement of the first-step links with the local particle
spacing and falls back to the single-pass mode above
`vfc_bootstrap_shift_ratio` (0.5 by default).

Measured sweep of the two estimators against the flow displacement (in units
of the particle spacing), median |v_est - v_true|:

| displacement / spacing | first-link accuracy | mode | two-pass |
|---|---|---|---|
| 0.07 | 0.99 | 0.072 | **0.000** |
| 0.19 | 0.99 | 0.178 | **0.000** |
| 0.37 | 0.92 | 0.344 | **0.000** |
| 0.56 | 0.76 | 0.514 | **0.154** |
| 0.94 | 0.53 | 0.853 | 0.810 |
| 1.50 | 0.33 | 1.416 | 2.353 |

The two-pass wins wherever the tracking problem itself is well posed, and the
guard switches to the mode extraction where it is not.




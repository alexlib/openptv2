# Plan: Spatial Neighborhood Consensus (Vector Field Consensus / VFC) for Two-Phase Tracker

**Date:** 2026-10-12  
**Status:** Implementation Plan  
**Target:** `openptv2/src/openptv2/plugins/two_phase_tracking.py`  
**Related Docs:** `docs/plans/2026-09-30-tracker-plan.md`, `docs/plans/2026-10-09-trackcorr-two-phase-merge.md`, `docs/two-phase-tracking.md`

---

## 1. Motivation and Objective

In dense 3D Particle Tracking Velocimetry (3D-PTV) and Lagrangian Particle Tracking (LPT), optical ambiguities during multi-camera reconstruction produce **ghost particles** (false 3D points). In dense seeding regimes, ghost-to-real particle ratios can easily exceed 50–80%.

### The Current Gap in `two_phase`
The `two_phase` tracker currently provides:
1. 3D KD-tree bounding (`v_max`)
2. 2D leaf reprojection ranking + Hungarian bipartite matching per connected component
3. Epipolar ray-miss ghost weighting (`q_weight`, `q_seed`, `q_young`)
4. Photometric continuity (`blob_gate`)
5. Two-hop temporal kink confirmation (`confirm_tol`)

However, **it evaluates particles without spatial neighbor coherence**:
- Candidate links are scored purely individually or via bipartite competition against other tracks contesting the exact same detection.
- If a ghost particle happens to appear along a track's predicted path with plausible 2D projections, the Hungarian solver assigns the track to the ghost.
- **Physical Reality:** Real fluids obey continuum kinematics: velocity fields are locally smooth with bounded spatial gradients ($\nabla \mathbf{u}$). A true particle displacement $\mathbf{d}_i$ must show consensus with the displacements of neighboring particles $\mathbf{d}_{\text{nbr}}$ within a local correlation radius $R$. Random ghost trajectories and false associations violate this spatial consensus.

### Objective
Implement and benchmark **Spatial Neighborhood Consensus (Vector Field Consensus / VFC)** inside the `two_phase` tracker to eliminate ghost associations and improve trajectory continuity in high-density flows.

We formulate and test two complementary architectures:
- **Option A (Pre-Assignment Consensus):** Pruning candidate edges or penalizing costs *before* the Hungarian solver.
- **Option B (Post-Assignment Consensus):** Spatial outlier rejection *after* Hungarian matching, filtering tentative links before history commitment.

---

## 2. Mathematical Formulation of Spatial Consensus

Let active track $i$ at frame $t$ be at position $\mathbf{x}_i(t)$ with estimated velocity $\mathbf{v}_i(t)$.
Within a local neighborhood radius $R_{\text{vfc}}$ (or among $K_{\text{vfc}}$ nearest neighbors), define the active neighborhood:
$$\mathcal{N}_i = \{ j \neq i : \|\mathbf{x}_j(t) - \mathbf{x}_i(t)\| \le R_{\text{vfc}} \text{ and } \mathbf{v}_j(t) \text{ is established} \}$$

If $|\mathcal{N}_i| \ge N_{\min}$ (e.g. at least 3 neighbors), the local consensus velocity is computed using the coordinate-wise median (robust against neighbor outliers):
$$\mathbf{v}_{\text{consensus}}(\mathbf{x}_i) = \operatorname{median}_{j \in \mathcal{N}_i} \mathbf{v}_j(t)$$

For any candidate match $c$ at frame $t+1$ with position $\mathbf{y}_c$, the candidate displacement vector is:
$$\mathbf{d}_{ic} = \frac{\mathbf{y}_c - \mathbf{x}_i(t)}{\Delta t}$$

The **consensus residual** is:
$$\Delta \mathbf{v}_{ic} = \|\mathbf{d}_{ic} - \mathbf{v}_{\text{consensus}}(\mathbf{x}_i)\|$$

---

## 3. Option A: Pre-Hungarian Consensus (Gate & Cost Shaping)

### Concept
Evaluate neighborhood consensus **before** building bipartite connected components. If a candidate edge deviates unphysically from the local consensus velocity field, either drop it (hard gate) or penalize its cost in the assignment matrix.

### Detailed Workflow
1. At frame $t$, gather all active tracks with established velocities ($len \ge 2$).
2. Build a 3D KD-tree `tree_active` over active track positions $\{\mathbf{x}_i(t)\}$.
3. For each active track $i$:
   - Query `tree_active.query_ball_point(x_i, r=vfc_radius)`.
   - If $|\mathcal{N}_i| \ge vfc\_min\_neighbors$:
     - Compute $\mathbf{v}_{\text{consensus}}(i) = \operatorname{median}(\{\mathbf{v}_j\})$.
     - Compute local dispersion $\sigma_{\text{consensus}}(i) = \operatorname{median}(\|\mathbf{v}_j - \mathbf{v}_{\text{consensus}}\|)$.
   - Otherwise: mark consensus as unconstrained (insufficient local density, fall back to default `v_max`).
4. In `_match_two_phase_frame`:
   - For each candidate edge $(i, c)$:
     - Compute displacement $\mathbf{d}_{ic} = (\mathbf{x}_c(t+1) - \mathbf{x}_i(t)) / \Delta t$.
     - Compute consensus deviation $\Delta v = \|\mathbf{d}_{ic} - \mathbf{v}_{\text{consensus}}(i)\|$.
     - **Hard Gate (`vfc_gate`):** If $\Delta v > vfc\_gate$, prune edge $(i, c)$ from `row_arr`, `col_arr`, `cost_arr`.
     - **Soft Penalty (`vfc_weight`):** If $vfc\_weight > 0$:
       $$\text{Cost}_{ic} \leftarrow \text{Cost}_{ic} \cdot \left(1 + vfc\_weight \cdot \frac{\Delta v}{\max(\sigma_{\text{consensus}}, \epsilon)}\right)$$

### Pros & Cons
- **Pros:**
  - Prevents ghost particles from even entering the Hungarian bipartite graph.
  - Significantly reduces connected component sizes and degree of connectivity, which can *speed up* the Hungarian solver in dense clusters.
  - Prevents a ghost match from stealing a track when the true real candidate is also inside the search radius.
- **Cons:**
  - Requires computing consensus residuals across all candidate edges ($O(E)$ edges).
  - Tracks at flow boundaries or sparse margins with $< N_{\min}$ neighbors must bypass the gate.

---

## 4. Option B: Post-Assignment Fast-VFC Filter

### Concept
Let the two-phase Hungarian solver match as usual. Then, treat the tentative frame-to-frame displacement vectors as a vector field and apply a fast Vector Field Consensus outlier filter. Matches violating spatial coherence are severed before updating trajectories.

### Detailed Workflow
1. Run Phase 1 & Phase 2 as normal, yielding tentative matches $\mathcal{M} = \{(i, c_i)\}$.
2. For all links in $\mathcal{M}$, compute their proposed displacement vectors:
   $$\mathbf{d}_i = \mathbf{x}_{c_i}(t+1) - \mathbf{x}_i(t)$$
3. Build a 3D KD-tree `tree_links` over the starting positions $\{\mathbf{x}_i(t)\}$.
4. For each link $i$:
   - Query neighbors within `vfc_radius`: $\mathcal{N}_i^{\text{links}} = \{ j \in \mathcal{M} : \|\mathbf{x}_j(t) - \mathbf{x}_i(t)\| \le R \}$.
   - If $|\mathcal{N}_i^{\text{links}}| \ge vfc\_min\_neighbors$:
     - Compute local median displacement: $\bar{\mathbf{d}}_i = \operatorname{median}_{j \in \mathcal{N}_i^{\text{links}}} \mathbf{d}_j$.
     - Compute local MAD scale $\sigma_i = \operatorname{median}(\|\mathbf{d}_j - \bar{\mathbf{d}}_i\|)$.
     - Residual $r_i = \|\mathbf{d}_i - \bar{\mathbf{d}}_i\|$.
     - If $r_i > \max(vfc\_post\_k \cdot \sigma_i, vfc\_post\_min\_tol)$:
       - Sever link $(i, c_i)$.
       - Particle $c_i$ is released; track $i$ increments its gap counter.
5. Severed tracks coast on constant velocity prediction for up to `max_gap` frames, allowing recovery on subsequent frames when the particle moves past the ghost cluster.

### Pros & Cons
- **Pros:**
  - Extremely fast: evaluates only $N_{\text{links}}$ vectors instead of $E_{\text{edges}}$ candidate edges.
  - Fully decoupled from Hungarian matching logic and independent of 2D/3D cost modes.
  - Can clean up erroneous links produced even by complex multi-track competition.
- **Cons:**
  - If a real particle's match was stolen by a ghost during Hungarian assignment, Option B severs the ghost link, but the real particle remains lost for that frame (relying on `max_gap` to recover it later), whereas Option A would have eliminated the ghost edge early and let Hungarian pick the real particle.

---

## 5. Proposed Configuration Interface

Extend `TwoPhaseTrackerConfig` in `openptv2/src/openptv2/plugins/two_phase_tracking.py`:

```python
@dataclass
class TwoPhaseTrackerConfig:
    ...
    # --- Spatial Neighborhood Consensus (VFC) ---
    vfc_mode: Literal["off", "pre", "post", "both"] = "off"
    vfc_radius: float = 10.0          # Neighborhood correlation radius in mm
    vfc_min_neighbors: int = 3         # Minimum neighbors required to establish consensus
    
    # Option A (Pre-Assignment) parameters:
    vfc_gate: float | None = None      # Hard velocity deviation gate (mm/frame). None = off
    vfc_weight: float = 0.0            # Soft cost penalty weight (0.0 = off)
    
    # Option B (Post-Assignment) parameters:
    vfc_post_k: float = 3.0           # Multiplier over local MAD dispersion
    vfc_post_min_tol: float = 0.5     # Absolute floor tolerance in mm/frame
```

---

## 6. Testing, Benchmarking & Comparison Plan

To objectively evaluate **Speed** and **Performance (Accuracy/Physics)** between Option A and Option B:

### Test Datasets
1. **Synthetic Burgers / Hill Vortex Flow with Ghost Particles:**
   - Synthetic ground truth flow with controlled velocity gradients.
   - Inject synthetic ghost particles at varying densities:
     - Low density: 500 real + 250 ghosts (33% ghost ratio)
     - Medium density: 2,000 real + 2,000 ghosts (50% ghost ratio)
     - High density: 5,000 real + 10,000 ghosts (67% ghost ratio)
2. **Dense Jet / Boundary Layer Flow:**
   - Tests shear-gradient preservation to ensure consensus radius does not over-smooth legitimate shear layers.

### Performance Metrics to Track
1. **Speed & Scalability:**
   - Wall-clock runtime per frame (ms/frame) as a function of particle density.
   - Size of Hungarian connected components (Option A should reduce percolation).
2. **Trajectory Accuracy:**
   - Link Precision: $\text{TP} / (\text{TP} + \text{FP})$
   - Link Recall: $\text{TP} / (\text{Total True Links})$
   - Mean Trajectory Length: longer continuous tracks indicate fewer ghost-induced track breaks.
3. **Physical Turbulence Metrics (`openptv2.benchmarking.metrics`):**
   - Acceleration Kurtosis $K_a$: false links produce unphysical acceleration spikes, driving $K_a \gg 3$.
   - Velocity Jump Fraction: percentage of velocity kinks exceeding physical thresholds.

---

## 7. Implementation Roadmap

- [ ] **Step 1: Data Structures & Helpers**
  - Implement `compute_local_consensus_field(positions, velocities, radius, min_neighbors)` in `openptv2/src/openptv2/plugins/two_phase_tracking.py`.
- [ ] **Step 2: Implement Option A (Pre-Assignment)**
  - Integrate neighborhood consensus velocity into `_match_two_phase_frame`:
    - Edge pruning (`vfc_gate`).
    - Cost matrix scaling (`vfc_weight`).
- [ ] **Step 3: Implement Option B (Post-Assignment)**
  - Implement `filter_links_by_spatial_consensus(links, pts0, pts1, radius, min_neighbors, k, min_tol)`.
  - Wire into `TwoPhaseTracker.track_frames` after `_match_two_phase_frame`.
- [ ] **Step 4: Unit Testing**
  - Create `tests/unit/test_two_phase_vfc.py` covering:
    - Pure shear / uniform flow consensus verification.
    - Ghost particle rejection with Option A.
    - Post-match outlier severance with Option B.
    - Edge cases: sparse particles ($< N_{\min}$), isolated particles, boundary layers.
- [ ] **Step 5: Benchmark Script & Evaluation**
  - Create `scripts/bench_two_phase_vfc.py` running baseline vs Option A vs Option B across ghost density sweeps.
  - Record execution speed (ms/frame) and precision/recall metrics.

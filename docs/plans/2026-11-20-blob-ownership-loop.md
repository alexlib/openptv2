# Blob-Ownership Outer Loop around `two_phase`

**Date:** 2026-11-20
**Branch / Worktree:** `feat/blob-ownership-loop` (`/Users/alex/Documents/Github/openptv2-blob-ownership`)

---

## 1. Problem Statement & Root Cause

Benchmark diagnostics on `Synthetic-CompleteTest` (L0 / L1 / L2) and real `wp2` establish that the bottleneck in 3D-PTV accuracy is **neither the linker nor ghosts**:

- **Not the linker:** On exact 3D points (L0), the chain reaches `0.0009` relative velocity error, and `two_phase` finds ~99 % of reachable links.
- **Not ghosts:** Removing every ghost from the input improves `two_phase` by only 0.5–2 %.
- **It is merged blobs:** ~27 % of blob observations on the L1 benchmark (and roughly half that rate on real `wp2`) contain two or more overlapping particles.

Because `match_frame_correspondences` must assign each 2D blob to at most one 3D particle (1-to-1 ownership), every merged blob causes three compounding failures:

1. **Shifted centre:** The kept particle $p_1$ is triangulated using the merged blob's shifted centroid.
2. **Dropped partner:** The overlapping particle $p_2$ loses that camera view and is often omitted from the frame's 3D point cloud altogether.
3. **Hidden-partner corruption (the primary loss):** Membership inference among *known* particles is already ~98 % accurate, but when $p_2$ is absent from the point cloud, the blob-ownership table believes the blob contains only $\{p_1\}$. Particle $p_1$ therefore treats the merged blob as a clean single-particle observation, biasing its 3D position, velocity, and acceleration.

**Core principle:** *A detected blob is not a particle.* Bringing hidden partners back into the point cloud simultaneously increases yield/completeness **and** makes the particles we already had more accurate by exposing the merge.

---

## 2. Mathematical Model

### 2.1 Forward Model of a Merged Blob

For frame $t$, camera $c$, and detected blob $b$ with observed centroid $\mathbf{u}_{t,c,b} = (x, y)$ and integrated brightness $S_{t,c,b}$ (`sumg`), let $\mathcal{M}(t, c, b)$ be the set of 3D particles projecting into blob $b$, each with intrinsic per-camera brightness $A_{p,c}$:

$$S_{t,c,b} \approx \sum_{p \in \mathcal{M}(t,c,b)} A_{p,c}$$

$$\mathbf{u}_{t,c,b} \approx \frac{\sum_{p \in \mathcal{M}(t,c,b)} A_{p,c}\,\pi_c(\mathbf{X}_p(t))}{\sum_{p \in \mathcal{M}(t,c,b)} A_{p,c}}$$

where $\pi_c : \mathbb{R}^3 \to \mathbb{R}^2$ is the calibrated camera projection (`img_coord_batch`).

Because each particle's per-camera brightness is persistent across time (log-flicker $\sim 0.05$–$0.06$ on both L1 and real data), a trajectory $\tau_k$'s clean per-camera brightness $\hat{A}_{k,c}$ is estimated as the **robust lower-median** (e.g. 40th percentile) of $S_{t,c,b}$ over frames where $\tau_k$ is the sole known occupant of its blob in camera $c$ (using a lower quantile resists upward bias from frames with hidden partners). If a track has no clean frames in camera $c$, $\hat{A}_{k,c}$ falls back to its brightness in other cameras shifted by the per-camera median gain offset.

### 2.2 Residual Blob Inversion (Exposing Hidden Partners)

Given current occupants $\mathcal{M}(t, c, b)$ with predicted projections $\tilde{\mathbf{u}}_{p,c} = \pi_c(\tilde{\mathbf{X}}_p(t))$ and brightnesses $\hat{A}_{p,c}$:

$$S_{\text{known}} = \sum_{p \in \mathcal{M}} \hat{A}_{p,c}, \qquad S_{\text{res}} = S_{t,c,b} - S_{\text{known}}$$

Whenever $S_{\text{res}}$ exceeds the excess-brightness threshold ($S_{\text{res}} > \max(S_{\min}, \tau_{\text{excess}}\, S_{\text{known}})$), blob $b$ contains unexplained intensity from at least one hidden partner. Subtracting the known moments gives a **virtual de-blended 2D target**:

$$\mathbf{u}_{\text{res}} = \frac{S_{t,c,b}\,\mathbf{u}_{t,c,b} - \sum_{p \in \mathcal{M}} \hat{A}_{p,c}\,\tilde{\mathbf{u}}_{p,c}}{S_{\text{res}}}$$

clamped to lie within a physical merge radius $r_{\max}$ (e.g. 3.5 px) of $\mathbf{u}_{t,c,b}$ so small-$S_{\text{res}}$ noise cannot send $\mathbf{u}_{\text{res}}$ far away.

### 2.3 Joint Multi-Particle 3D Refinement (De-blending)

At each frame $t$, particles sharing at least one blob form small connected components (size 1 for clean particles; typically size 2–4 for merged groups). For each component $\mathcal{C}$, we solve the weighted nonlinear least-squares problem:

$$\min_{\{\mathbf{X}_p\}_{p \in \mathcal{C}}} \sum_{(c,b) \in \mathcal{B}(\mathcal{C})} w_{c,b} \left\| \mathbf{u}_{c,b} - \frac{\sum_{p \in \mathcal{M}(c,b)} \hat{A}_{p,c}\,\pi_c(\mathbf{X}_p)}{\sum_{p \in \mathcal{M}(c,b)} \hat{A}_{p,c}} \right\|^2$$

solved via 3 damped Gauss-Newton (Levenberg-Marquardt) steps using finite-difference $2 \times 3$ projection Jacobians (batched through `img_coord_batch`):

- **Clean blob** ($|\mathcal{M}| = 1$, no excess brightness): $w_{c,b} = 1.0$ (standard LSQ triangulation).
- **Known multi-particle blob** ($|\mathcal{M}| \ge 2$): exact brightness-weighted centroid coupling. Each particle's clean cameras pin down its 3D position, allowing the shared blob to constrain the partner without biasing either particle.
- **Unexplained excess-brightness single-occupant blob** ($|\mathcal{M}| = 1$ with $S_{t,c,b} > (1 + \tau_{\text{excess}})\,\hat{A}_{p,c}$): downweight $w_{c,b}$ (e.g. $w_{c,b} = (\hat{A}_{p,c} / S_{t,c,b})^2 \cdot w_{\text{down}}$) whenever particle $p$ has $\ge 2$ clean cameras, preventing an unrecovered hidden partner from shifting $\mathbf{X}_p$.
- **Under-constrained guard:** If a particle has fewer than 2 well-weighted camera constraints, add a small Tikhonov tether toward its initial / kinematically predicted 3D position so the linear system stays well-conditioned.

---

## 3. Outer-Loop Algorithm

The loop wraps `TwoPhaseTracker` (unchanged as the linker) for `n_iters` iterations (default `3`):

### Initialization (Iteration 0)
- Load raw per-frame targets `rows[f][c]` and initial 3D correspondences `(pos[f], cam_ids[f])` from `RunStore`.
- Build initial `project_fn` and per-camera gain offsets.

### Inside Each Iteration `it = 0 .. n_iters - 1`

1. **Link with `two_phase`**
   - For each particle $(t, i)$, form its 2D leaf vector: use the raw blob centroid in clean cameras, and the **reprojected / de-blended** 2D position $\pi_c(\mathbf{X}_{t,i})$ in cameras where the blob is shared (or flagged excess-brightness).
   - Run `TwoPhaseTracker.track_frames(...)` to obtain links and trajectories $\mathcal{T}$.
   - On the final iteration, stop after linking (or after one final joint refinement of the linked tracks) and write the refined correspondences and linkage back to `RunStore`.

2. **Track Brightness & Blob Membership Accounting**
   - Extract trajectories $\mathcal{T}$ from the links (keep tracks of length $\ge L_{\min}$, e.g. 3).
   - Associate every tracked particle with its blobs (`cam_ids`, plus projection-based attachment within `attach_radius_px` for any camera currently marked `-1` where the particle projects cleanly onto an existing blob).
   - Estimate per-track, per-camera intrinsic brightness $\hat{A}_{k,c}$ (robust lower-median on sole-occupant frames; cross-camera gain fallback otherwise).
   - Build the blob membership table $\mathcal{M}(t, c, b)$ across all frames and cameras.

3. **Hidden-Partner Recovery**
   - **3a. Track-driven gap-fill and head/tail extension into shared blobs:**
     - For every internal gap of length $1 \dots \text{max\_gap}$ along a trajectory $\tau_k$, linearly/quadratically interpolate $\tilde{\mathbf{X}}_k(t)$ at the missing frame $t$.
     - For track heads and tails (with $\ge 2$ points), linearly extrapolate one frame backward/forward (iterated across frames so extensions chain).
     - Project $\tilde{\mathbf{X}}_k(t)$ into every camera $c$:
       - Match the nearest blob within `recover_radius_px` — **whether free or already occupied**.
       - If the blob is occupied and has a valid virtual residual $\mathbf{u}_{\text{res}}$, prefer matching against $\mathbf{u}_{\text{res}}$ (else allow matching against $\mathbf{u}_b$ within `merge_radius_px`).
       - Require $\ge 2$ camera hits, with at least 1 clean (free) camera OR a consistent brightness/residual match in occupied cameras.
       - Triangulate the new 3D point (using clean cameras if $\ge 2$ are clean, else all hit cameras using $\mathbf{u}_{\text{res}}$ for occupied ones), validate that its 3D distance to $\tilde{\mathbf{X}}_k(t)$ is within the kinematic gate, and check that it does not duplicate an existing particle already close to $\tilde{\mathbf{X}}_k(t)$.
       - Append the recovered particle to frame $t$ and update $\mathcal{M}(t, c, b)$.
   - **3b. Residual-pool correspondence discovery (unseeded hidden partners):**
     - For each frame $t$ and camera $c$, build a residual target list consisting of:
       1. Free blobs (unoccupied, or occupied only by unlinked singletons), and
       2. Virtual de-blended targets $(\mathbf{u}_{\text{res}}, S_{\text{res}})$ from occupied blobs with significant excess brightness ($S_{\text{res}} > \max(S_{\min}, \tau_{\text{excess}}\, S_{\text{known}})$).
     - Run `match_frame_correspondences` on this residual target set.
     - Keep newly triangulated 3D points (seen by $\ge 2$ cameras, not duplicating existing particles within `dedup_mm`, and using at least one virtual target or reclaimed ghost blob), mapping virtual target indices back to their parent raw blob indices in `cam_ids`.
     - Append them to frame $t$ so the next `two_phase` pass can link genuine hidden partners into new tracks and ignore false combinations.

4. **Joint Multi-Particle 3D Position Refinement**
   - Recompute $\mathcal{M}(t, c, b)$ including all newly recovered particles.
   - Run the connected-component Gauss-Newton de-blender (Section 2.3) on every frame to update $\mathbf{X}_{t,i}$ for all particles in shared blobs and downweight still-unexplained excess-brightness blobs.
   - Prune unlinked singletons that were only speculative residual candidates from previous iterations if they failed to link.

---

## 4. Code Organization

| File | Role |
|---|---|
| `src/openptv2/blob_ownership.py` | Core algorithms: `BlobLoopConfig`, brightness estimator, membership table, residual virtual-target builder, track-driven shared-blob recovery, residual-pool correspondence discovery, connected-component Gauss-Newton joint de-blender, and `run_blob_ownership_loop(...)`. |
| `src/openptv2/plugins/blob_loop_tracking.py` | Tracking plugin (`Tracking`) and `track_store(...)` wrapper using the exact same `RunStore` / `exp` contract as `two_phase_tracking.py`. |
| `src/openptv2/plugins/loader.py` | Register `"blob_loop"` and `"two_phase_blob_loop"` in `BUILTIN_TRACKING_PLUGINS`. |
| `tests/unit/test_blob_ownership.py` | Unit tests: (1) synthetic 2-particle merge de-blending accuracy, (2) residual virtual target exactness, (3) track-driven recovery into an occupied blob, (4) residual-pool discovery of an unseeded hidden partner, (5) excess-brightness downweighting, (6) end-to-end loop on a synthetic store. |

---

## 5. Verification & Benchmark Plan

1. **Unit tests:**
   ```bash
   uv run pytest tests/unit/test_blob_ownership.py tests/unit/test_two_phase_tracking.py -v
   ```
2. **Benchmark vs `two_phase` and `trackcorr`** via `scripts/synth_bench.py`:
   - Primary realistic benchmark: `L1_d1_c0_k1_n200_s0.08_a0.18_f0.05`
   - Ablation runs via `+key=value` overrides:
     - `two_phase` (baseline)
     - `trackcorr` (baseline)
     - `blob_loop+discover_residual=false` (track-driven recovery + joint de-blending only)
     - `blob_loop` (full loop: track-driven + residual discovery + joint de-blending + excess downweighting)
   - Additional stress cases (if available in `WORK/cases`): higher density (e.g. `L1_d2...`) and `L2`.
3. **Success criteria:**
   - Measurable reduction in `vel_err_rel`, `acc_err_rel`, and `pos_err_mm` over `two_phase` and `trackcorr`.
   - Higher `yield` and `long_completeness`, fewer `long_fragments`, without increasing `ghost_point_frac`.
   - Runtime stays practical (target $\le 3$–$5\times$ `two_phase`).

---

## 6. Implementation Status (Handoff Summary)

- **Worktree:** `/Users/alex/Documents/Github/openptv2-blob-ownership`
- **Branch:** `feat/blob-ownership-loop` (branched from `main` at `d82e6e13`)
- **Python env:** compiled Cython `.so` files copied into `src/openptv2/algorithms/`; run with:
  ```bash
  UV_PROJECT_ENVIRONMENT=/Users/alex/Documents/Github/openptv2/.venv \
  PYTHONPATH=/Users/alex/Documents/Github/openptv2-blob-ownership/src \
  uv run --no-sync ...
  ```

### Done
1. **Plan written:** `docs/plans/2026-11-20-blob-ownership-loop.md`.
2. **Core module implemented:** `src/openptv2/blob_ownership.py`
   - `BlobLoopConfig` (including `membership_radius_mm`, `flag_merged`, `merged_noise_scale`, and ablation toggles).
   - Batched projection & finite-difference Jacobian helpers (`project_points_per_cam`, `project_and_jacobian_per_cam`, `triangulate_pixels`).
   - Trajectory extraction (`links_to_tracks`), per-camera gain offsets, blob membership table (`build_blob_membership`), and missing-blob projection attachment (`attach_missing_blobs`).
   - Lower-median per-track per-camera brightness estimator (`estimate_particle_brightness`).
   - Residual virtual-target inversion (`compute_residual_blobs_frame`).
   - **Idea 11 + Idea 4** in `recover_from_tracks`: predictor-based membership check (`membership_radius_mm = 0.6` mm) that splits merged 3D points owned by another track into constituent points (`_split_from_partner`), plus shared-blob gap-fill and head/tail extension (`_claim_into_any_blob`).
   - Residual-pool correspondence discovery (`discover_residual_correspondences`).
   - Connected-component damped Gauss-Newton mixture-centroid solver (`joint_deblend_frame`) with excess-brightness downweighting.
   - Outer loop driver (`run_blob_ownership_loop`).
3. **Plugin wrapper & registration:**
   - `src/openptv2/plugins/blob_loop_tracking.py` (`Tracking`, `track_store`).
   - Registered `"blob_loop"` and `"two_phase_blob_loop"` in `src/openptv2/plugins/loader.py`.
4. **Idea 9 support in `TwoPhaseTracker`:** `src/openptv2/plugins/two_phase_tracking.py`
   - Added optional `frame_merged` and `merged_noise_scale` parameters to `track_frames`, `_track_unidirectional`, and `_match_two_phase_frame`.
   - Tagged merged/split points inflate measurement noise in the cost matrix and **never update track velocity** (velocity stays anchored to `last_clean_pos` / `last_clean_t`).

### Not Done Yet (Next Steps)
1. **Wire `frame_merged` inside `run_blob_ownership_loop`** (`src/openptv2/blob_ownership.py`, around lines 1236–1274):
   - Build per-frame boolean `frame_merged` mask marking every particle that shares a blob (`len(membership[t][c][b]) >= 2`), came from a split/recovery, or sits on an `excess_all[t][c]` blob.
   - Pass `frame_merged=frame_merged if cfg.flag_merged else None` and `merged_noise_scale=cfg.merged_noise_scale` to `tracker.track_frames(...)` inside `_run_linker()`.
   - Note: `_split_from_partner` and `_try_add` should also mark newly split points (and their partner) in `frame_merged` right away.
2. **Unit tests:** Create `tests/unit/test_blob_ownership.py` covering residual inversion, joint mixture-centroid de-blending, excess-brightness downweighting, predictor-based membership split, Idea 9 no-velocity-update behavior in `TwoPhaseTracker`, and end-to-end loop recovery; run `uv run pytest tests/unit/test_blob_ownership.py tests/unit/test_two_phase_tracking.py -v`.
3. **Benchmarks vs `two_phase` and `trackcorr`:**
   - Run `scripts/synth_bench.py` on `L1_d1_c0_k1_n200_s0.08_a0.18_f0.05` (and other `L1_*` density/shear/jitter cases) comparing `two_phase`, `trackcorr`, and `blob_loop` (plus ablations: `discover_residual=false`, `flag_merged=false`, `joint_deblend=false`).
   - Validate yield/ghost on real datasets (`lv_multi` / `Illmenau` / `wp2`).
4. **Git commit:** Nothing has been committed on `feat/blob-ownership-loop` yet.


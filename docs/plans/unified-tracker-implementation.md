# Unified Tracker — Implementation Plan

Branch: `feat/unified-tracker` (openptv2) + `feat/unified-tracker` (openptv-cloud).
Goal: replace the 8 trackers with one self-configuring, adaptive, auditable tracker that maximizes accuracy, with speed coming from parallelism rather than dropped accuracy.

This document is the contract for multi-session implementation by multiple coding agents. Each task is scoped to be independently implementable and testable. Check off tasks as completed; update "Status" lines.

---

## 0. Principles (read first)

1. **Accuracy is the top priority.** Speed is achieved via parallel runs, not by skipping passes.
2. **One code path.** Legacy tracker names remain as aliases mapping to `unified` with fixed config bundles; no duplicate implementations.
3. **Auto-configuration from data.** Every `track.*` option defaults to `"auto"` and resolves to a concrete value before the run; resolved values are logged and stored.
4. **Probe before full run.** Short, multi-window probes select parameters; full run uses the chosen config.
5. **Fail fast and honestly.** Conditioning gate prints feasibility and refuses to silently produce garbage.
6. **Auditable.** Every run stores a `TrackingDecision` record.
7. **Backward compatible.** Existing YAMLs keep working; outputs (Zarr `ptv_is`) unchanged.
8. **Target-space boundary (no OTF / image subtraction).** Stay in `detect → correspond → track` target space so tracking remains fast and robust for non-spherical, polydisperse, and bubble tracers. Handle high density via multi-peak 2D splitting, tight-core + 2D-target pruning, non-exclusive target sharing in `track_assisted`, and conflict-component matching.

---

## 1. Current state (done)

- [x] `src/openptv2/plugins/unified_tracking.py` — `Tracking` plugin, auto-regime, backbone = trackcorr `full_forward()`.
- [x] Registered `unified` / `unified_tracking` in `plugins/loader.py` and `tracking_registry.py`.
- [x] `unified` preset + config bundle in `tracking_presets.py`.
- [x] `sample_probe_windows()` + `aggregate_probe_metrics()` static helpers.
- [x] openptv-cloud `pipeline.py` parallel gate allows `unified`.
- [x] Branches created and committed.

---

## 1.5. Execution Roadmap (Multi-Agent / Parallel Sessions)

Use this table to claim work in parallel sessions. **Rules for parallel agents:**
1. Only pick a step whose **Depends on** column is `[x]` done.
2. Steps in the same **Wave** touch **disjoint files** and can run in parallel sessions without merge conflicts.
3. When finished: run the **Verify** command (`uv run pytest ...`), check `[x]` on the step, and append a line to **§14 Progress log**.

### Wave 0 — Unblock & Baseline (run first, can be 2 parallel sessions)
- [x] **Step 0a — Fix `_auto_regime` & registry summary**
  - **Files:** `src/openptv2/plugins/unified_tracking.py`, `src/openptv2/tracking_registry.py`, `tests/unit/test_unified_tracking.py` (new)
  - **Scope:** Move `sample_probe_windows()` / `aggregate_probe_metrics()` out of `_auto_regime()`, restore `_auto_regime()` return dict, update `UNIFIED_INFO.algorithm_summary` to `trackcorr`, and add unit tests for `_auto_regime`, `sample_probe_windows`, and `aggregate_probe_metrics`.
  - **Depends on:** none
  - **Verify:** `uv run pytest tests/unit/test_unified_tracking.py -v`

- [x] **Step 0b — Minimal Synthetic GT Scorecard (`H0`)**
  - **Files:** `tests/benchmarks/test_unified_gt_scorecard.py` (new)
  - **Scope:** 1-command synthetic GT harness (clean, slow-flow/noisy, and dense-ghost cases) reporting yield, precision, median track length, and jump-step rate. Used to calibrate thresholds in Waves 1–3.
  - **Depends on:** none
  - **Verify:** `uv run pytest tests/benchmarks/test_unified_gt_scorecard.py -v`

### Wave 1 — Foundation Contracts (3 parallel sessions, disjoint files)
- [x] **Step 1a — Lightweight `TrackingDecision` (`F0` / Milestone F)**
  - **Files:** `src/openptv2/tracking_decision.py` (new), `tests/unit/test_tracking_decision.py` (new)
  - **Scope:** `TrackingDecision` dataclass + JSON serialization (`tracking_decision.json` + `RunStore` attribute helpers).
  - **Depends on:** none
  - **Verify:** `uv run pytest tests/unit/test_tracking_decision.py -v`

- [x] **Step 1b — Auto-Parameter Estimator (`A1` + `tracking_recommender` reuse)**
  - **Files:** `src/openptv2/tracking_estimator.py` (new), `tests/unit/test_tracking_estimator.py` (new)
  - **Scope:** Pure read-only `estimate_params(exp, track_cfg)` using `tracking_feasibility` + `tracking_recommender` trackability guard (`rho <= 1`), sampling via `sample_probe_windows`. Also computes conditioning ratio (`B1`).
  - **Depends on:** Step 0a
  - **Verify:** `uv run pytest tests/unit/test_tracking_estimator.py -v`

- [x] **Step 1c — Multi-Peak 2D Detection Splitting (`B0` / `L2`)**
  - **Files:** target detection module (`detect_targets`), `tests/unit/` detection tests
  - **Scope:** Optional local-maxima / watershed splitting on touching 2D blobs (no OTF).
  - **Depends on:** none
  - **Verify:** `uv run pytest tests/unit/ -k "detect_targets or target" -v`

### Wave 2 — Core Unified Pipeline (`Estimator → Engine → Refiner`) (3 parallel sessions)
- [x] **Step 2a — Wire `Estimator` + Conditioning Gate + `TrackingDecision` (`A2`, `B1`, `F2`)**
  - **Files:** `src/openptv2/plugins/unified_tracking.py`, `tests/unit/test_unified_tracking.py`
  - **Scope:** Wire `estimate_params()` + conditioning warnings (`< 0.5`, `< 0.2`) + `TrackingDecision` writing into `Tracking.do_tracking()`. Ensure resolved flags (`use_2d`, `confirm_link`, `v_max`, `delta_t`) are actually applied to `trackcorr` / `hybrid_deltat_3d` without mutating user YAML.
  - **Depends on:** Step 0a, Step 1a, Step 1b
  - **Verify:** `uv run pytest tests/unit/test_unified_tracking.py -v`

- [x] **Step 2b — Post-Run Self-Consistency Report (`D1`–`D2`)**
  - **Files:** `src/openptv2/tracking_report.py` (new), `tests/unit/test_tracking_report.py` (new)
  - **Scope:** Compute reciprocity rate, track-length distribution, jump-step rate, gap distribution, ghost estimate, and anomaly flags from `RunStore` linkage; return dict for `TrackingDecision`.
  - **Depends on:** Step 1a
  - **Verify:** `uv run pytest tests/unit/test_tracking_report.py -v`

- [x] **Step 2c — Upgrade `track_assisted` (`K3` / `B8`)**
  - **Files:** `src/openptv2/track_assisted.py`, `tests/unit/test_track_assisted.py` (or existing track_assisted tests)
  - **Scope:** Replace 2-point linear extrapolation with 3–4 point polynomial predictor; add 2-of-4 / 3-of-4 camera-subset triangulation and optional non-exclusive (shared) 2D target fallback (down-weighted in least-squares).
  - **Depends on:** Step 0b
  - **Verify:** `uv run pytest tests/ -k "track_assisted" -v`

### Wave 3 — Probing & High-Density Clean-and-Claim (2 parallel sessions)
- [x] **Step 3a — Multi-Window Probe + Coordinate Descent (`C1`–`C4`)**
  - **Files:** `src/openptv2/tracking_probe.py` (new), `src/openptv2/plugins/unified_tracking.py`, `tests/unit/test_tracking_probe.py` (new)
  - **Scope:** Isolated sub-window `probe()` (temp/in-memory `RunStore`), coordinate-descent search capped by `probe_budget_seconds`, non-TTY-safe `probe_confirm`, calibrated against Step 0b GT scorecard.
  - **Depends on:** Step 0b, Step 2a, Step 2b
  - **Verify:** `uv run pytest tests/unit/test_tracking_probe.py -v`

- [x] **Step 3b — 2-Pass Clean-and-Claim + Conflict-Component Matching (`K2`, `K4`, `K5`, `B7`)**
  - **Files:** `src/openptv2/layered_tracking.py` (new/evolved), `tests/unit/test_layered_tracking.py` (new)
  - **Scope:** Tight-Core Pass 1 → deterministic 2D-target pruning mask → Pass 2 residual + upgraded `track_assisted` (`Step 2c`); conflict-component `linear_sum_assignment` for ambiguous links (`B7`). Validate on dense GT (`Step 0b`).
  - **Depends on:** Step 0b, Step 2c
  - **Verify:** `uv run pytest tests/unit/test_layered_tracking.py tests/benchmarks/test_unified_gt_scorecard.py -v`

### Wave 4 — Parallelism, Docs & Full Validation (3 parallel sessions)
- [x] **Step 4a — Parallel & Chunked Tracking (`G1`–`G2`)** *(openptv-cloud `G3` sharding is in the separate `openptv-cloud` repo)*
  - **Files:** `src/openptv2/plugins/unified_tracking.py`, `src/openptv2/tracking_chunked.py`, `tests/unit/test_parallel_tracking_chunked.py`
  - **Scope:** Sequential probe/estimation → parallel full run (`track_sequence_chunked_parallel`); share `TrackingDecision` across chunks.
  - **Depends on:** Step 2a, Step 3a
  - **Verify:** `uv run pytest tests/unit/test_parallel_tracking_chunked.py -v`

- [x] **Step 4b — YAML Surface & Docs (`I1`–`I2`)**
  - **Files:** `src/openptv2/tracking_presets.py`, `docs/tracking.md` (new)
  - **Scope:** Document all `track.*` keys, auto defaults, derivations, and deprecation notices for legacy aliases.
  - **Depends on:** Step 2a, Step 3a, Step 3b
  - **Verify:** `uv run pytest tests/unit/ -k "preset" -v`

- [x] **Step 4c — Full GT Scorecard (`H1`)** *(openptv-cloud `H2` compare gate is in `openptv-cloud`)*
  - **Files:** `tests/benchmarks/test_unified_gt_scorecard.py`
  - **Scope:** Compare `unified` vs. legacy (`trackcorr`, `priority_segment_3d`) on `clean`, `slow_noisy`, and `dense_ghost` synthetic GT.
  - **Depends on:** Step 3a, Step 3b, Step 4a
  - **Verify:** `uv run pytest tests/benchmarks/test_unified_gt_scorecard.py -v`

### Wave 5 — Cleanup & v2 (after Wave 4 passes)
- [x] **Step 5a — Deprecate Legacy Presets & Default GUI/Cloud to `unified` (`J1`–`J4`)** *(openptv2 preset/recommender/plugin loader done; `openptv-cloud` lives in separate repo)*
  - **Depends on:** Step 4b, Step 4c
- [x] **Step 5b — Per-Trajectory Quality Score & Downstream Weighting (`E1`–`E2`, v2)**
  - **Depends on:** Step 4c

### Wave 6 — Ghost Elimination Hardening (after Wave 5 passes)
- [x] **Step 6a — Core Purity, Kinematic Splitting & Multi-Tier Peeling (`layered_tracking.py`)**
  - Raise `peel_and_track` default `min_core_length` from `2` to `3` (clamped to `min(min_core_length, len(frames))` for 2-frame runs).
  - Add kinematic smoothness splitting/trimming inside `_extract_verified_core` (rejecting/splitting triplets failing both `_is_velocity_jump` and `||Δv|| > max_accel_err`) before populating `claimed_by_frame`.
  - Add post-Pass-2 pruning of Pass-2 claimed 2D targets from unlinked residual rows before `run_corrective_pass`.
  - Add 3-tier progressive peeling (`tiers=3` option: `N_cams` strict core → `N_cams - 1` medium core → residual `>= 2` cams + gated `track_assisted`, gracefully collapsing on `<= 2`-cam rigs).
- [x] **Step 6b — Gate `track_assisted` & `relink_trajectory_gaps` (`track_assisted.py`, `tracking_postprocess.py`)**
  - Capture 3D ray-miss `_dist` in `_claim_particle` and reject when `_dist > max_ray_miss_mm`; check `||pos - predicted|| <= accel_tol` in `_extend_tracks`.
  - Pass `gap_accel_err` into `run_corrective_pass` so `0.25 * tpar.dacc` is respected.
  - Add optional `require_bidir_vel` and `uniqueness_ratio` gates to `relink_trajectory_gaps`.
- [x] **Step 6c — Persist `rcm` at Correspondence Time (`run_store.py`, `correspondences.py`, `gui/ptv.py`, `*_sequence.py`)**
  - Persist the `rcm` (mm) already returned by `point_positions(...)` via `RunStore.write_rcm` / `read_rcm` (`quality` BlockSet `"rcm"` column) and wire `read_rcm` into `peel_and_track`'s `rcm_threshold` filter.
  - **Verify:** `uv run pytest tests/unit/test_layered_tracking.py tests/unit/test_track_assisted_poly.py tests/unit/test_unified_tracking.py tests/benchmarks/test_unified_gt_scorecard.py -v` and before/after ghost benchmark (`N60_ghost50`, `N200_ghost50`, `N200_ghost70`).

---

## 2. Milestone A — Auto-parameter estimation (highest impact)

**Owner:** agent A. **Effort:** 2–3 sessions.

### A1. `estimate_params()` in unified plugin
File: `src/openptv2/plugins/unified_tracking.py`
- Add method that, given `exp`, computes:
  - `v_max` from `tracking_feasibility.measure_motion_scale` on a sample of correspondences (frames 1/2), × safety factor 1.5 (configurable `v_max_safety`).
  - `density` from correspondence counts.
  - `noise_floor_z` from `tracking_feasibility.z_noise_floor_mm`.
  - `slow_flow` flag: displacement/noise < 1.0.
  - `ghost_rate` estimate (correspondences with no successor).
- Return dict of resolved values; merge with explicit user overrides (user wins).
- Store resolved params on `self._resolved`.

### A2. Wire into `_auto_regime`
- Replace heuristic defaults with calls to `estimate_params()` when `auto`.
- Keep explicit overrides winning.

### A3. Tests
File: `tests/unit/test_unified_tracking.py` (new)
- Unit tests for `estimate_params` on a small synthetic store.
- Test that explicit overrides win over auto.

### Acceptance
- Running unified on an existing dataset with all `auto` produces a valid run; resolved values logged.

---

## 3. Milestone B — Conditioning gate

**Owner:** agent B. **Effort:** 1 session.

### B1. Feasibility check before tracking
File: `src/openptv2/plugins/unified_tracking.py` (`do_tracking`)
- Before init, compute displacement/noise ratio.
- If ratio < 0.5: print WARNING with ratio + recommendation; auto-enable coarse-to-fine (`slow_flow=True`) and 2D re-ranking.
- If ratio < 0.2: print STRONG WARNING; recommend calibration improvements; still proceed.
- Store ratio in `TrackingDecision`.

### B2. Tests
- Mock store; assert warning printed and flags set.

---

## 4. Milestone C — Probe run + grid search

**Owner:** agent C. **Effort:** 3–4 sessions.

### C1. `probe()` method
- For each window from `sample_probe_windows`, run tracking on that window with the auto config, compute metrics:
  - yield (links / particles)
  - median track length
  - reciprocity rate
  - jump-step rate (kink > threshold)
  - gap count
- Aggregate across windows via `aggregate_probe_metrics`.

### C2. Coordinate-descent search when probe is poor
- Define thresholds: if median track length < `min_acceptable_length` (default 5) OR jump_rate > 0.1, run coordinate descent (instead of full Cartesian grid, capped by `probe_budget_seconds`):
  1. Sweep `v_max × {1.0, 1.5, 2.0}` first.
  2. Sweep `delta_t ∈ {1, 3, 5}` only if `slow_flow` is true.
  3. Toggle `use_2d` / `bidirectional` only if ambiguity/reciprocity remains poor.
- Score = median_length × yield − λ·jump_rate (λ=0.5 default).
- Pick best; log comparison table.

### C3. Interactive confirm
- If `track.probe_confirm: true` (default in interactive, false in batch), print report and prompt user (Y/n/override). In batch, auto-pick best.
- Config: `probe: true|false`, `probe_windows: 5`, `probe_window_frames: 30`, `probe_confirm: auto`.

### C4. Tests
- Probe on synthetic data; assert best config chosen.

---

## 5. Milestone D — Post-run self-consistency report

**Owner:** agent D. **Effort:** 1–2 sessions.

### D1. `report()` method
After tracking, compute and print:
- reciprocity rate
- track-length distribution (median, p10, p90, max)
- jump-step rate
- gap distribution
- estimated ghost rate
- anomaly flags with suggested fixes.

### D2. Store metrics
- Write metrics to run store attributes / `tracking_decision.json`.

---

## 6. Milestone E — Per-trajectory quality

**Owner:** agent E. **Effort:** 2 sessions.

### E1. Quality score per trajectory
- In `tracking_postprocess.py` / `quality_post.py`, compute score from: length, reciprocity, smoothness, gap count, kink.
- Store in linkage (quantized `prio` or sidecar/new field) — coordinate with `storage/linkage_blocks.py` and `storage/run_store.py` (`prio` is `int32`).
- Propagate through `flowtracks`/export as weight.

### E2. Downstream weighting
- Export and ensemble binning optionally weight by quality.

---

## 7. Milestone F — Auditable decision record

**Owner:** agent F. **Effort:** 1 session.

### F1. `TrackingDecision` dataclass
File: new `src/openptv2/tracking_decision.py`
- Fields: resolved config, regime flags, probe metrics per window, chosen config, grid-search table, conditioning ratio, git version, timestamp, data hashes.
- Serialize to JSON; write to store path as `tracking_decision.json` and as run-store attribute.

### F2. Wire into plugin
- Populate throughout `do_tracking`; write at end.

---

## 8. Milestone G — Parallel as first-class

**Owner:** agent G. **Effort:** 1–2 sessions.

### G1. Auto chunk count
- In `openptv-cloud/pipeline.py` and `src/openptv2/batch/pyptv_batch_parallel.py`, compute `workers = min(requested, total//(overlap+2), cpu_count)`; already in openptv2 formula; ensure used.

### G2. Probe on 1 worker, full run parallel
- Probe sequentially; full run uses `parallel_workers`.

### G3. Cloud sharding
- Ensure `run_shard`/`stitch_shards` use the same `TrackingDecision` and resolved config.

---

## 9. Milestone H — Validation harness

**Owner:** agent H. **Effort:** 2 sessions.

### H1. Synthetic GT comparison
- Extend `tests/unit/test_parallel_tracking_chunked.py` / `tests/batch/test_synthetic_tracker.py` style to compare unified vs. legacy on synthetic cases (including dense ghost cases); report yield/precision/jump-rate.

### H2. CI gate
- Add `openptv-cloud compare` (sharded vs. unsharded) to CI.

---

## 10. Milestone I — YAML surface & docs

**Owner:** agent I. **Effort:** 1 session.

### I1. Document every `track.*` key
- In `tracking_presets.py` comments and a new `docs/tracking.md`.
- Include: control, auto default, derivation, when to override, accuracy/speed impact.

### I2. Legacy alias deprecation notices
- Print one-line notice recommending `unified` when a legacy preset is used.

---

## 10.5. Features at risk when dropping trackers (must preserve)

Dropping trackers risks losing capabilities that trackcorr backbone does not
natively provide. The unified tracker must explicitly preserve these:

1. **3D-only mode (no calibration / no 2D targets).**
   `priority_segment_3d` and `4be` operate on 3D correspondence clouds alone.
   trackcorr requires calibration for epipolar search. Unified needs a
   `mode: 3d_only` branch using `full_forward_3d()` + `track3d_loop_fast`,
   selected when cals/2D targets are unavailable.

2. **New-particle seeding in 3D-only mode.**
   `priority_segment_3d` does not seed new particles mid-sequence. Unified's
   `new_particles` option must have a 3D cold-start seeding heuristic for the
   3D-only branch (or delegate to the epipolar path when 2D is available).

3. **Configurable cost weights.**
   Only `two_phase` exposes cost weights. The 2D leaf-ranking module must
   accept weight parameters (`app_weight`, etc.) for advanced users.

4. **Splitter hardware path.**
   Keep `splitter_tracking` as a standalone (hardware niche), not dropped.

5. **Fast 3D speed path.**
   `priority_segment_3d` is the fastest. Keep an internal fast-3D option
   (`fast_3d_mode: true`) for speed-critical 3D-only runs, even though the
   default backbone is trackcorr.

6. **Bidirectional in 3D-only mode.**
   Verified: `full_backward()` works after `full_forward_3d()`, so unified
   can offer bidirectional + reciprocity in both modes.

**Architecture note (avoiding a god-plugin):**
Keep `unified_tracking.py` as a thin orchestrator over 3 composable stages sharing `RunStore`:
1. `Estimator` (read-only): `estimate_params()` + conditioning gate + probe → `TrackingDecision`.
2. `Engine` (strictly 2 backends): Epipolar `trackcorr` (when cals + 2D targets exist) or 3D-only `track3d` (when they don't, or `fast_3d_mode: true`).
3. `Refiner` (optional, composable): Clean-and-Claim residual pass → `tracking_postprocess` → `quality_post` → `report()`.

## 11. Milestone J — Cleanup / dropping trackers

**Owner:** agent J. **Effort:** 1 session.

Decision: drop the following as user-facing trackers (they become aliases → `unified`);
their *techniques* are absorbed into the unified adaptive modules:

- `priority_segment_3d` / `fast` / `fast_3d` — speed-only 3D tradeoff; unified uses trackcorr backbone + optional fast 3D path.
- `standard_forward` — forward-only tradeoff; covered by unified with `bidirectional: false`.
- `two_directional` — forward+backward without reciprocity; covered by unified `bidirectional: true, postprocess: false`.
- `4be` — absorbed as the `lookahead` tie-breaker module.
- `full_multipass` / `trackcorr` — this IS the unified backbone.

Kept as advanced standalone (hardware/regime niches), not dropped:
- `hybrid_deltat_3d` — slow high-FPS flows (also a unified module).
- `two_phase` — 2D leaf ranking for ambiguous 3D (also a unified module).
- `splitter_tracking` — 4-view splitter hardware.

### J1. Convert dropped trackers to deprecated aliases
- In `plugins/loader.py`: map dropped names → `unified` with fixed config bundles.
- Print a one-line deprecation notice recommending `unified`.

### J2. Update GUI/recommender to default to `unified`
### J3. Update openptv-cloud notebooks to use `unified`
### J4. Remove dropped trackers from `PRESET_CHOICES` user-facing list (keep aliases resolving).

---

## 12. Cross-cutting concerns

- **Tests must pass** on `feat/unified-tracker` before merge: `pytest tests/` in openptv2; openptv-cloud test suite.
- **Cython is allowed for hot paths.** Reuse existing kernels by default, and add targeted Cython (e.g., `B5–B8` solver/predictor loops, inner-loop 2D re-ranking, or `track_assisted` / Clean-and-Claim hot paths) whenever profiling shows Python/NumPy is a bottleneck.
- **I/O contract unchanged.** Zarr `ptv_is` linkage group.
- **Each task PR should be small and focused**, referencing this plan's milestone/task IDs.

---

## 12.4. Backbone improvements (trackcorr v2 — high-density target space)

Before unifying, make the backbone itself competitive in high-density regimes without leaving target space (no OTF / image subtraction). Improvements are split into Python/pre-pass (low risk, do first) and Cython-level (higher risk, later).

Python / pre-pass level:
- B0. Multi-peak 2D detection (`detect_targets`): optional local-maxima / watershed splitting on connected components so touching particles split at detection time without an OTF model.
- B1. Early ghost filtering pre-pass: compute ghost probabilities (rcm/rcm_blob) per correspondence; keep originals as backup (prefer Tight-Core + 2D-target pruning in §12.5 over hard-dropping high-rcm points).
- B2. Adaptive search radii from `measure_motion_scale` (1.5x measured displacement) capped by the `tracking_recommender.py` trackability guard (`rho = p95_displacement / mean_interparticle_distance <= 1`).
- B3. Frame-quality awareness: skip/interpolate frames with far fewer correspondences.
- B4. Kink-based link rejection + gap relink repair (`max_gap=2..3` bridges brief all-camera overlaps).

Cython / solver level (later):
- B5. 4BE n+2 lookahead as tie-breaker in epipolar scoring (only when `>= 2` candidates fall in search box).
- B6. 2D leaf re-ranking inside epipolar candidate selection (ambiguous frames/candidates only).
- B7. Conflict-component Hungarian resolution: keep fast greedy path for unambiguous 1-to-1 links; group competing candidates (`>= 2` tracks vs `>= 2` candidates) into connected components and solve optimal bipartite assignment (`linear_sum_assignment`).
- B8. Smooth 3–4 point polynomial predictor in `trackcorr` and `track_assisted` (cuts prediction error vs. 2-point linear extrapolation, enabling tighter search boxes).

### Milestone L — Backbone improvements
**Owner:** agent L. **Effort:** 2–4 sessions.
- L1. Implement B1 ghost filtering pre-pass + tests on synthetic data.
- L2. B2 adaptive radii (with trackability cap) + B0 multi-peak 2D splitting option.
- L3. B4 link rejection + gap relinking.
- L4. B3 frame quality.
- L5. Validate B0–B4 vs. baseline on synthetic GT (expect precision up at high density).
- L6 (later). B5–B8 solver & predictor upgrades (4BE lookahead, 2D re-ranking, conflict-component Hungarian, 3–4 pt polynomial predictor).

## 12.5. 2-Pass "Clean-and-Claim" (Tight-Core + 2D-Target Pruning + `track_assisted`)

High-density strategy to collapse combinatorial ghosts without image subtraction or arbitrary `N`-layer quantile tuning:

1. **Pass 1 — Tight-Core:** Track only high-purity correspondences (tighter epipolar tolerance e.g. `0.5 * eps0`, or low-`rcm` / 4-cam quadruplets) with `full_forward` + `full_backward` + strict `enforce_reciprocity`. Keep only verified core tracks (length ≥ 4, smooth).
2. **Deterministic 2D-target prune:** Mark all 2D target IDs claimed by Pass-1 tracks. Drop any remaining 3D correspondence in the cloud that references a claimed 2D target ($O(N)$ mask, no re-triangulation). Ghosts that stole a 2D target from a real particle vanish deterministically.
3. **Pass 2 — Residual + `track_assisted` expansion:**
   - Run `trackcorr` on the sparser, ghost-depleted residual cloud.
   - Run `track_assisted` (`run_configured_corrective_pass` with `release_ghosts=True`), upgraded with:
     - **3–4 point polynomial predictor** (replacing 2-point linear extrapolation).
     - **Camera-subset triangulation (2-of-4 / 3-of-4)** when 1–2 cameras are merged/occluded.
     - **Non-exclusive (shared) 2D target fallback:** allow two predicted tracks whose 2D projections land on the same merged blob to reference that target (down-weighted in least-squares triangulation) instead of losing a track.
4. **Final postprocess:** Gap relink + reciprocity + confirm on the accumulated set.

Implementation notes:
- Module `src/openptv2/layered_tracking.py` (evolving K1 primitives into 2-pass Clean-and-Claim) + upgrades to `src/openptv2/track_assisted.py`.
- Config: `peeling: auto|true|false` (`auto` enables Tight-Core + Clean-and-Claim when `ghost_rate` or `density` is high).
- Cost: 2 passes instead of `N` quantile layers; mitigated by tight Pass-1 cloud and pruned Pass-2 cloud.

### Milestone K — 2-Pass Clean-and-Claim
**Owner:** agent K. **Effort:** 3–4 sessions.
- K1. Prototype single-layer quality-filtered subset tracking → verify index remapping works. (Done)
- K2. Deterministic 2D-target pruning mask + Pass-2 residual tracking & merge into `RunStore`.
- K3. Upgrade `track_assisted.py` with 3–4 pt polynomial predictor, 2-of-4/3-of-4 subset triangulation, and optional non-exclusive shared-target mode.
- K4. Final postprocess + tests vs. synthetic dense GT (expect higher precision and yield at high density without OTF).
- K5. Wire into unified as `peeling` option (`auto` when `ghost_rate` / `density` is high).

## 12.6 Anti-Ghost Strategy (consolidated) — critical review

Three complementary levers, ordered from pipeline-level to solver-level. Each is
already scoped in the milestones above (K, L, B7); this section collects them as
one coherent strategy and records the critical review.

### Idea 1 — Two-Pass "Clean-and-Claim" (Tight-Core + deterministic 2D-target pruning)
**Where:** `src/openptv2/layered_tracking.py` (Milestone K: K2, K4, K5).
**Mechanism:** Pass 1 tracks only high-purity correspondences (tighter epipolar
tolerance, low-rcm / 4-cam quadruplets) with `full_forward` + `full_backward` +
strict reciprocity, keeping only verified core tracks (length ≥ 4, smooth). Then
mark every 2D target ID claimed by those tracks and drop any remaining 3D
correspondence referencing a claimed target (O(N) mask, no re-triangulation).
Pass 2 runs `trackcorr` on the ghost-depleted residual cloud, then `track_assisted`
with polynomial predictor / subset triangulation / shared-target fallback.
**Config:** `peeling: auto|true|false` (auto when `ghost_rate` or `density` high).

### Idea 2 — Early Ghost-Filter Pre-Pass (rcm / rcm_blob scoring)
**Where:** `src/openptv2/ghost_filter.py` (Milestone L: L1).
**Mechanism:** Compute per-correspondence ghost probability from reconstruction-
consistency (`rcm`) and blob-consistency (`rcm_blob`) before tracking; down-weight
or quarantine high-probability correspondences. Keep originals as backup rather
than hard-dropping, so real particles are not silently lost.

### Idea 3 — Conflict-Component Hungarian Resolution
**Where:** backbone solver (Milestone L: L6 / B7).
**Mechanism:** Keep the fast greedy 1-to-1 path for unambiguous links; for
components with ≥2 competing tracks vs. ≥2 candidates, solve the optimal bipartite
assignment with `scipy.optimize.linear_sum_assignment`. Cost is paid only where
ambiguity exists.

---

### Why this works (root cause, not just mechanism)

Ghosts are not a detection bug or a tracking bug in isolation — they are a **combinatorial
correspondence** phenomenon. Given N 2D targets per camera, the number of candidate 3D
points is ~N^k; real particles produce consistent multi-camera matches, while ghosts are
accidental alignments of targets from *different* particles. Each idea attacks a different
manifestation of that root cause, which is why they compose rather than duplicate:

- **Idea 2 (rcm) — geometric inconsistency.** A 3D point is only as real as the
  agreement of its constituent 2D targets: back-project the rays and a real particle's
  rays meet, while a ghost's rays miss. rcm measures that miss *before* any temporal
  tracking, so it is a pure, tracking-independent ghost signal. It must be *soft*
  because rcm has a distance dependence (volume centre vs. edges) and real particles
  near edges overlap the ghost distribution — hard thresholds delete real data.
- **Idea 1 (Clean-and-Claim) — 2D-target ownership.** A single 2D target in one
  camera at one instant can belong to at most one physical particle. Ghosts are
  *parasitic*: they almost always reuse 2D targets that belong to real particles,
  because the combinatorics of forming a ghost from entirely unused targets are far
  less likely. Once a verified (reciprocal, smooth, long) track has claimed its 2D
  targets, any other correspondence reusing one of them is by necessity a ghost (or a
  duplicate). Pruning by ownership removes the *substrate* ghosts need, deterministically
  and without re-triangulation. This is why it is safe where hard deletion is not:
  it only removes correspondences that conflict with an already-verified reality.
- **Idea 3 (Hungarian) — assignment-level suboptimality.** Ghost links are created not
  only at detection but at the *linking* step, when greedy nearest-neighbour assignment
  makes locally optimal choices inside a dense ambiguous region and those errors cascade.
  Solving the optimal bipartite matching over the whole ambiguous component (tracks vs.
  candidates) breaks the cascade. It is restricted to ambiguous components because for
  unambiguous 1-to-1 links greedy is already optimal and cheaper.

**Why "soft, not hard" is the cross-cutting principle:** ghosts and real particles overlap
in every single-feature space (rcm, displacement, smoothness) near decision boundaries.
Threshold-based deletion therefore trades recall for precision unpredictably. The safe
operations are *claim-based* (a verified reality overrules a candidate) and *global-
assignment* (optimal over a component), which never delete a real particle merely because
it looks ghost-like.

### Critical review

**Strengths**
- *Determinism over deletion.* Clean-and-Claim removes ghosts by *claiming* 2D
  targets rather than by deleting points, so it cannot accidentally drop real
  particles the way a hard rcm threshold can. This matches the empirical finding
  in `docs/plans/2026-09-30-tracker-plan.md` that soft use works and hard
  deletion does not.
- *Complementary, not redundant.* Idea 1 is a pipeline-level prune, Idea 2 a
  scoring pre-filter, Idea 3 a solver-level assignment fix — they operate at
  different stages and compose.
- *Bounded cost.* Idea 1 is 2 passes on a shrinking cloud; Idea 3 only pays the
  Hungarian cost on ambiguous components.

**Risks / weaknesses**
- **W1 — Pass-1 purity is the linchpin.** If the tight core is too loose,
  pruning deletes real residual correspondences; if too tight, too little is
  pruned and Pass 2 re-creates ghosts. The `0.5 * eps0` tolerance and
  length ≥ 4 thresholds are uncalibrated guesses (see W3 of §15). Needs
  calibration against the synthetic GT scorecard (H0/H1) before shipping.
- **W2 — 2D-target identity assumption.** Clean-and-Claim assumes a stable 2D
  target ID across frames and that a real particle "owns" its 2D target. With
  merged/occluded blobs or camera dropouts, one real particle can legitimately
  share a target, which is exactly why `track_assisted` needs the non-exclusive
  shared-target fallback — but that fallback re-opens the door to ghost survival.
  The two mechanisms must be reconciled (shared targets allowed only in Pass 2,
  never in Pass 1).
- **W3 — rcm distance dependence.** rcm scales with distance from the volume
  centre, so Idea 2 requires the per-run straight-line fit normalization already
  identified in `2026-09-30-tracker-plan.md` (A10). Using raw rcm will bias
  pruning toward the volume edges.
- **W4 — Hungarian on large components.** `linear_sum_assignment` is cubic in
  component size; at very high density an ambiguous component can grow large.
  Need a component-size cap (fall back to greedy above N) and profiling.
- **W5 — Interaction / ordering.** Running Idea 2 before Idea 1 could prune
  correspondences that Pass 1 would have used to build the tight core; running
  Idea 1 first makes Idea 2's input already depleted. The intended order is
  Idea 2 (soft scoring) → Idea 1 (hard claim) → Idea 3 (solver), but this
  ordering is not yet enforced in code and must be documented in the
  `Estimator → Engine → Refiner` stages.
- **W6 — No byte-identity guarantee.** Unified with these levers enabled will
  not reproduce legacy tracker outputs (W4 of §15). A regression test
  comparing unified (fixed bundle) vs. legacy on a reference dataset is required
  before deprecation (J1).

**Verdict:** Proceed, but gate shipping on (a) calibration of the Pass-1
tolerance/length thresholds against synthetic GT, (b) enforcing the
Idea 2 → 1 → 3 ordering, and (c) a component-size cap for the Hungarian solver.

### 12.6.1 Reshuffled alternatives (out-of-the-box)

The staged design (rcm → Clean-and-Claim → Hungarian) is not the only
factorization. Three reshuffles worth considering, with critical notes:

**Reshuffle A — Merge Clean-and-Claim and Hungarian into one "Hungarian Claim
Resolution" step.** Instead of (1) verify core tracks, (2) greedily claim their
2D targets, (3) separately solve ambiguous links, treat the whole thing as one
assignment: Pass-1 candidate tracks are the "left" set, 2D targets are the
"right" set, and a residual correspondence is a ghost iff assigning it would
conflict with a higher-weight verified track. `linear_sum_assignment` over the
target-overlap cost matrix picks the globally optimal non-overlapping subset of
tracks in one shot. This *is* Milestone K's conflict-component Hungarian (B7)
applied to claim resolution — i.e. Ideas 1 and 3 collapse into a single stage.
- *Pro:* removes the purity-threshold risk of Pass 1 (no separate "tight core"
  tolerance to calibrate); global optimum instead of greedy claim; one less
  pipeline stage.
- *Con:* Hungarian is cubic in component size and needs a component-size cap;
  and it loses the *interpretability* of "these targets are claimed by track X",
  which is valuable for the audit record (`TrackingDecision`) and for debugging.
  Also, a single assignment pass cannot express "keep the core, then expand" —
  the two-pass intent (conservative core first, aggressive expansion second)
  has to be encoded as weights/costs, which re-introduces tuning.

**Reshuffle B — One composite "realness" score, single pass.** Fuse all three
signals into one scalar per 3D correspondence / trajectory:
  `realness = w1·(1 - rcm_norm) + w2·claim_status + w3·(1 - assign_cost_norm)`,
then sort and (softly) gate seeding (`q_seed`) and continuation (`q_young`) —
exactly the mechanism that already worked in `2026-09-30-tracker-plan.md`
(−0.008…−0.013 velocity error, −0.038 at 4× density). No multi-pass, no
pruning, no Hungarian at runtime.
- *Pro:* simplest possible implementation; single tunable surface; reuses a
  *proven* mechanism (soft gating) instead of an unproven one (deterministic
  pruning); audit record is just one number per trajectory.
- *Con:* the tracker plan explicitly found that **link-cost weighting alone has
  no effect** — a pure score only works when it gates *seeding/continuation*,
  not when it re-weights links. So this reshuffle is only viable if the score
  feeds q_seed/q_young, not the link cost. Also, conflating geometric (rcm),
  ownership (claim), and assignment (cost) failure modes into one number makes
  it hard to know *why* a particle was dropped, defeating the auditability
  principle (§0.6). And it forgoes the deterministic pruning that is the
  *largest* potential gain (ghosts cause ~60% of two_phase's error).

**Reshuffle C — Hungarian-first, then rcm refines (invert the order).** Run
Hungarian on the raw correspondence cloud first to find the globally optimal
link set; treat high-cost unmatched correspondences as ghost candidates; then
apply rcm (Idea 2) as a refinement/weight on the survivors; claim targets last.
- *Pro:* attacks the assignment problem at its largest (whole cloud), where
  greedy errors are most damaging.
- *Con:* Hungarian on the *whole* cloud is O(N^3) and N is the full 3D
  correspondence count — infeasible at high density without the component
  decomposition, which is exactly what Clean-and-Claim's pruning provides. So
  this ordering is only viable *after* pruning, i.e. it is the staged order
  with the stages renamed. Not actually a reshuffle.

**Recommendation (12.6.1).** Do not pick one. Use a **two-tier** design:
  - **Tier 1 (deterministic, cheap, auditable):** Reshuffle A's Hungarian Claim
    Resolution on a *pre-filtered* cloud (rcm soft-weighted, not hard-pruned)
    to produce a high-confidence core set + an explicit claimed-target mask.
    This keeps the interpretable claim record and the conservative-then-expansive
    intent, while removing the fragile Pass-1 tolerance threshold.
  - **Tier 2 (soft):** the composite realness score from Reshuffle B, but used
    *only* to gate q_seed/q_young in `track_assisted` (Pass 2 expansion), never
    as a link weight — honouring the tracker plan's finding.

  Net effect: three ideas → two stages, with Hungarian doing double duty (claim
  resolution + conflict links) and the realness score confined to its one proven
  use. The single-score heuristic is rejected as the *primary* mechanism but
  retained as a *secondary* gating signal.

### 12.6.2 Calibration-aware and multi-cycle epipolar ideas (out-of-the-box)

Two further reshuffles that attack the same root cause from the geometry side.

**Idea 4 — Calibration-aware ghost thresholds.** rcm has a distance dependence
(volume centre vs. edges) and, more fundamentally, the *number of cameras seeing a
point* varies across the volume. With 2 cameras two rays always have a closest
point, so rcm≈0 even for ghosts; with 4 cameras ghosts have large rcm. So ghost-
ness is spatially heterogeneous and camera-coverage dependent. Instead of one
global rcm threshold (or a crude per-run straight-line fit), use the calibration
maps (camera poses, distortion, coverage) to compute an expected rcm floor and a
threshold *per region / per point*, keyed on camera count and distance from the
volume centre.
- *Pro:* fixes the rcm distance-dependence issue (W3 of §12.6) at its source;
  better recall near boundaries where real particles are hardest to track and
  where ghosts are hardest to distinguish; naturally explains why a global
  threshold misbehaves.
- *Con:* requires a calibration-quality / coverage field that may not exist for
  all rigs; couples ghost logic to calibration correctness (a bad calibration
  gives bad thresholds); adds a per-point computation; must be validated that
  the region keying doesn't just re-learn the global fit.

**Idea 5 — Multi-cycle epipolar consistency (1-2-3-4, 2-3-4-1, …).** Instead
of one fixed camera ordering for epipolar search/triangulation, run several
cyclic orderings and require cross-consistency. A ghost that happens to satisfy
epipolar constraints in ordering 1-2-3-4 (because its targets accidentally align
along that reference) frequently fails in 2-3-4-1 (different reference camera →
different epipolar lines). The disagreement across cycles is itself a ghost
signal, analogous to rcm but measured across *reference frames* rather than rays.
- *Pro:* stronger geometric filter than single-ordering epipolar search; the
  cross-cycle variance can feed the realness score (Idea B) as an additional
  feature; generalizes the existing 2-of-4/3-of-4 subset triangulation (K3) to
  the correspondence stage.
- *Pro (compounding):* combined with Idea 4, a point seen by ≥3 cameras in
  multiple cycles is high-confidence real; a 2-camera point surviving only one
  cycle is suspect.
- *Con:* cost is N_cycles × correspondence cost — needs a cap (2 cycles likely
  enough; diminishing returns beyond); some persistent ghosts agree across cycles
  (not a silver bullet); aggregation rule (intersection = strict, union = loose)
  must be chosen and calibrated; risks over-filtering real particles in sparse
  regions where only one cycle yields a match.

**How these connect to the existing plan.** Idea 4 feeds Idea 2 (rcm pre-pass)
by replacing the global threshold with a calibration-keyed one. Idea 5 feeds
either Idea 2 (as an extra realness feature) or Idea 1 (as an extra pruning
criterion: drop correspondences that fail cross-cycle consistency). Both also
belong in the realness score's feature set if Reshuffle B is adopted. Neither
requires new Cython; both are pre-pass / correspondence-stage logic.

**Recommendation (updated two-tier).**
  - **Tier 1:** Reshuffle A (Hungarian Claim Resolution) on an rcm-soft-
    weighted, **calibration-keyed** cloud (Idea 4), with cross-cycle consistency
    (Idea 5) as an additional pruning criterion. This replaces the fragile
    Pass-1 tolerance with a per-point, geometry-aware confidence.
  - **Tier 2:** composite realness score (Idea B) for q_seed/q_young gating,
    now including rcm_norm, claim_status, assign_cost_norm, camera_count, and
    cross_cycle_agreement as features.

### 12.6.4 Detailed implementation plan — ILP uniqueness + temporal joint smoothing

**Goal:** suppress ghosts by combining (1) ILP/assignment for spatial uniqueness
with (5) temporal joint smoothing, while keeping tracking fast (target: no more
than ~2× the current trackcorr runtime on the dense benchmark).

**Principle:** both levers are *opt-in via `auto`* and only escalate when the
estimator reports high density/ghost_rate. The common case stays on the fast
trackcorr path. Parallelism (G1) absorbs the rest.

#### Available building blocks (verified present)
- `scipy.optimize.milp` (scipy 1.18) — MILP solver, no new dependency.
- `scipy.optimize.linear_sum_assignment` — bipartite Hungarian, already used.
- `networkx` 3.7 — connected components on conflict graphs.
- Existing: `ghost_filter.py`, `layered_tracking.py`, `track_assisted.py`,
  `tracking_estimator.py`, `tracking_probe.py`, `RunStore`/linkage blocks.

#### Stage A — Component decomposition (cheap, O(N))
File: `src/openptv2/ghost_components.py` (new)
- Build conflict graph per frame: node = candidate 3D point; edge = two points
  share at least one 2D target.
- `connected_components()` via networkx (or union-find for speed) → list of
  component (list of point indices).
- Classify each component: size 1 (trivial, keep), size 2 (bipartite →
  Hungarian), size ≥3 (ILP or greedy fallback).
- Store component sizes in `TrackingDecision` for audit.

#### Stage B — Spatial uniqueness via ILP (per component, bounded size)
File: `src/openptv2/ghost_ilp.py` (new)
- For each component of size ≥3 (and optionally size 2 via Hungarian):
  - Variables x_i ∈ {0,1}.
  - Objective: maximize Σ conf_i · x_i
    (conf_i = rcm-based confidence, calibration-keyed, + cross-cycle agreement
    from Idea 4/5).
  - Constraints: for each 2D target t in the component:
      Σ_{i: t∈point_i} x_i ≤ 1.
  - Solve with `scipy.optimize.milp`. Cap component size at `ilp_max_size`
    (default 20); larger components fall back to greedy + log a warning.
- For 2-camera components use `linear_sum_assignment` (cheaper, exact for
  bipartite) instead of MILP.
- Output: selected point indices per frame → feeds Pass 2 / trackcorr.
- Config: `ilp_uniqueness: auto|true|false`, `ilp_max_size`, `ilp_time_limit_s`.

#### Stage C — Temporal joint smoothing (sliding window)
File: `src/openptv2/temporal_smoothing.py` (new)
- Sliding window of W frames (default 5, config `smooth_window`).
- Within a window, link selected 3D points across frames using the existing
  trackcorr nearest-neighbour + motion model (constant velocity seeded from
  last 2 frames), producing candidate trajectories.
- Jointly optimize over the window: minimize
    Σ_traj [ Σ_frame reprojection_residual² + λ·acceleration² ]
  subject to point uniqueness (each 3D point used by at most one trajectory)
  and trajectory continuity.
- Implementation: iterative re-weighted least squares (IRLS) — cheap, no solver:
  1. Link points greedily to form trajectories.
  2. Fit a smooth (e.g. cubic) model per trajectory; compute residuals.
  3. Down-weight / drop points with residual > threshold (ghosts have high
     residual because they don't follow a smooth motion).
  4. Re-link and repeat 2–3 times (config `smooth_iterations`, default 3).
- Drop trajectories whose final residual exceeds `smooth_residual_threshold`
  (calibration-keyed, like rcm).
- Config: `temporal_smoothing: auto|true|false`, `smooth_window`,
  `smooth_iterations`, `smooth_residual_threshold`.

#### Stage D — Wiring into unified tracker
File: `src/openptv2/plugins/unified_tracking.py`
- In `do_tracking()`, after estimation/conditioning gate:
  - If `ilp_uniqueness` resolves true: run Stage A+B per frame (parallel over
    frames via the existing chunked runner) → produce cleaned 3D point sets.
  - Feed cleaned sets into trackcorr (existing backbone).
  - If `temporal_smoothing` resolves true: run Stage C on the resulting
    trajectories before postprocess.
- Apply resolved config to `trackcorr`/`hybrid_deltat_3d` without mutating
  user YAML (existing pattern).
- Write ILP component stats + smoothing residual stats into `TrackingDecision`.

#### Stage E — Estimator integration
File: `src/openptv2/tracking_estimator.py`
- Add `ghost_rate` already computed; add `ilp_recommended` flag (true when
  `ghost_rate > 0.15` or `density` high) and `smoothing_recommended` flag
  (true when flow is non-stationary / probe shows high jump_rate).
- `auto` resolves these flags; explicit user override wins.

#### Stage F — Tests
- `tests/unit/test_ghost_components.py` (new): component decomposition on
  synthetic conflict graphs; size classification.
- `tests/unit/test_ghost_ilp.py` (new): ILP on small components; uniqueness
  constraint holds; fallback on oversized components.
- `tests/unit/test_temporal_smoothing.py` (new): smoothing drops high-residual
  ghost trajectories; real smooth trajectories survive.
- `tests/benchmarks/test_unified_gt_scorecard.py`: add a "ghost-heavy" case;
  assert ILP+smoothing improves precision without hurting yield beyond 2%.

#### Speed budget & fast-path rules
- MILP only on components of size 3–`ilp_max_size` (default 20); size 1–2 use
  Hungarian or trivial keep; nothing runs on the whole cloud.
- Temporal smoothing runs only when `auto` recommends it (or user forces); IRLS
  with 3 iterations is O(window × points) and vectorizable.
- Both stages run in the existing parallel chunked runner (G1) — no new
  parallelism code.
- Hard cap: `ilp_time_limit_s` (default 1.0s per component) → fall back to
  greedy on timeout; log in `TrackingDecision`.
- Fast path (low density, low ghost_rate): both stages skipped entirely →
  identical to current trackcorr runtime.

#### Waves (parallelizable)
- Wave 0: Stage A (components) + Stage E (estimator flags) — disjoint files.
- Wave 1: Stage B (ILP) depends on A; Stage C (smoothing) independent.
- Wave 2: Stage D (wiring) depends on A/B/C/E; Stage F (tests) depends on D.
- Wave 3: benchmark (F scorecard) depends on all.

#### Risks
- R1: MILP latency on large components → cap + timeout + greedy fallback.
- R2: over-smoothing deletes real but erratic particles (turbulence) → residual
  threshold must be calibration-keyed and conservative; IRLS down-weights rather
  than hard-drops first.
- R3: interaction with Clean-and-Claim (Tier 1) → ILP replaces the greedy
  claim step; keep both behind the same `peeling`/`ilp_uniqueness` flags to
  avoid double-pruning. Recommendation: ILP subsumes Clean-and-Claim's claim
  step when enabled; Clean-and-Claim remains the fallback when MILP is off.

### 12.6.3 Deeper algorithmic ideas from other fields

Ghosts are a special case of a general problem: **combinatorial false
positives from nonlinear projections** — many fields face it. Analogies and
candidate algorithms, with critical notes:

1. **Multi-dimensional assignment / ILP** (data association, sensor fusion).
   Formulate "select a subset of 3D points maximizing total confidence, subject
   to each 2D target used at most once" as an integer program. Hungarian is the
   bipartite (2-camera) special case; ILP solvers handle the k-camera case and
   give a principled optimum instead of a greedy/heuristic one.
   - *Pro:* exact-ish, principled, directly encodes the uniqueness constraint.
   - *Con:* NP-hard in general; needs LP-relaxation or branch-and-bound; may
     be too slow at high density without the Clean-and-Claim pruning first.

2. **Factor graphs / belief propagation** (SLAM, probabilistic data association).
   Model 2D targets and 3D points as a bipartite factor graph with factors for
   geometric consistency and uniqueness; run loopy BP / message passing to
   compute marginals (probability each point is real). Handles combinatorics
   probabilistically and scales better than ILP.
   - *Pro:* soft, principled, naturally produces the "realness" probability
     that Reshuffle B wants; handles uncertainty.
   - *Con:* approximate; convergence not guaranteed; more machinery than the
     current codebase.

3. **Sparse recovery / L1 regularization** (compressed sensing, tomographic
   reconstruction). The 3D point cloud is a sparse source; camera images are
   nonlinear projections. Recover the sparse source via convex L1 minimization
   (basis pursuit) instead of combinatorial search — ghosts are the dense
   "noise" that sparsity penalizes.
   - *Pro:* convex (fast, global optimum), principled, directly attacks the
     combinatorial explosion.
   - *Con:* requires a discretized 3D grid / basis; nonlinearity of projection
     makes the sensing matrix expensive; may blur point positions.

4. **Consensus voting / Hough transform** (robust fitting, RANSAC family).
   Each 2D target votes for 3D points along its ray; real points accumulate
   votes from all cameras, ghosts get fewer. Accumulate in a 3D voting space.
   rcm is a continuous version of this; voting makes the threshold explicit
   and robust.
   - *Pro:* simple, robust, geometrically intuitive, handles outliers.
   - *Con:* voting-space resolution vs. memory trade-off; ghosts that happen
     to get many votes survive.

5. **Temporal joint smoothing / sliding-window MHT** (multi-hypothesis
   tracking). Ghosts don't persist as smooth trajectories. Instead of frame-pair
   greedy linking, optimize a whole window (N frames) jointly with a motion
   model; ghosts have high residual and are pruned by the smoother. This
   exploits the one thing ghosts lack: temporal coherence.
   - *Pro:* attacks ghosts at their weakest point (no persistent motion);
     leverages data we already have.
   - *Con:* window size vs. cost; ghosts that persist a few frames survive;
     needs a motion model (constant velocity etc.).

6. **Graph spectral methods** (community detection). Build an affinity graph
   between candidate 3D points (edge weight = shared 2D targets / geometric
   consistency); ghosts form a different spectral cluster than real points.
   Spectral clustering separates them without per-point thresholds.
   - *Pro:* principled, no thresholds, captures global structure.
   - *Con:* graph can be huge; spectral decomposition is O(N^3); cluster count
     must be chosen.

**Recommended deeper direction:** combine **(1) ILP/assignment for the
uniqueness constraint** (it is exactly what Clean-and-Claim approximates, and
the codebase already uses `linear_sum_assignment`) with **(5) temporal joint
smoothing** (ghosts lack temporal coherence, and we already track through time).
Together they attack both the spatial (combinatorial) and temporal (persistence)
dimensions of ghosts, which is stronger than any single idea above. Sparse
recovery (3) is the most "out-of-the-box" but the grid/basis cost is high;
keep it as a research option.

**Verdict:** Proceed, but gate shipping on (a) calibration of the Pass-1
tolerance/length thresholds against synthetic GT, (b) enforcing the
Idea 2 → 1 → 3 ordering, and (c) a component-size cap for the Hungarian solver.
tolerance/length thresholds against synthetic GT, (b) enforcing the
Idea 2 → 1 → 3 ordering, and (c) a component-size cap for the Hungarian solver.
  - **Tier 1 (deterministic, cheap, auditable):** Reshuffle A's Hungarian Claim
    Resolution on a *pre-filtered* cloud (rcm soft-weighted, not hard-pruned)
    to produce a high-confidence core set + an explicit claimed-target mask.
    This keeps the interpretable claim record and the conservative-then-expansive
    intent, while removing the fragile Pass-1 tolerance threshold.
  - **Tier 2 (soft):** the composite realness score from Reshuffle B, but used
    *only* to gate q_seed/q_young in `track_assisted` (Pass 2 expansion), never
    as a link weight — honouring the tracker plan's finding.

  Net effect: three ideas → two stages, with Hungarian doing double duty (claim
  resolution + conflict links) and the realness score confined to its one proven
  use. The single-score heuristic is rejected as the *primary* mechanism but
  retained as a *secondary* gating signal.

## 13. Open questions

1. Should probe run be on by default in cloud/batch mode? (Proposed: on, with auto-pick; disable via `probe: false`.)
2. Quality score storage: reuse `prio` array or new block group? (Needs coordination with `storage/linkage_blocks.py`.)
3. Should `unified` fully remove legacy plugins or keep them for 1 release? (Proposed: keep as aliases for 1 release, then remove.)

---

## 14. Progress log

- 2026-11-14: Branch created; scaffold + auto-heuristic + multi-window probing committed.
- 2026-11-14: L1 (ghost filter pre-pass) implemented + tests; wired into unified.
- 2026-11-14: K1 (layered peeling primitives) implemented + tests.
- 2026-11-15: Waves 0–4 completed and verified (50/50 tests passing).
- 2026-11-15: Wave 5 completed in openptv2: legacy preset/plugin deprecation notices + `unified` recommender default (`tracking_presets.py`, `tracking_recommender.py`, `plugins/loader.py`), and `compute_trajectory_quality_scores(..., write_prio=...)` with quantized `int32` `prio` persistence (`tracking_report.py`, `test_tracking_report.py`). Verified: 51 passed (unified suite) + 48 passed, 1 skipped (preset/recommender/plugin suite).
- 2026-11-15: Wave 6 (Ghost Elimination Hardening) completed: `min_core_length=3`, `_split_kinky_chain`, post-Pass-2 pruning, `tiers=3` progressive peeling (`layered_tracking.py`), ray-miss/3D-residual/bidir-vel/uniqueness gates (`track_assisted.py`, `tracking_postprocess.py`), and `RunStore.write_rcm`/`read_rcm` persistence (`run_store.py`, `correspondences.py`, `gui/ptv.py`). Verified: 23 passed; `N200_ghost50` `tiers=3` eliminated 100% of Pass-1 ghost leaks (`3000/3000` verified, `200/200` tracks, `1.000` yield, `0.999` precision, `0.000` jump rate).
- (append as work proceeds)

---

## 15. Critical review (self-assessment)

### Strengths
- Unifies code paths; removes the "preset ran forward-only" class of bug.
- Reuses existing, tested kernels; no new Cython.
- Multi-window probing addresses non-stationarity.
- Backward-compatible aliases.

### Weaknesses and risks

**W1. Probe cost.** A full grid search (v_max×delta_t×use_2d×bidirectional) on 5 windows is ~36 short runs. On large frames this is not cheap. Mitigation: limit grid to the most impactful axis first (v_max), then conditionally try delta_t/use_2d only if probe is poor. Add a `probe_budget_seconds` cap.

**W2. Proxy metrics.** Median track length / jump rate are proxies, not ground truth. Optimizing them can overfit. Mitigation: validate the composite score against synthetic GT early (move milestone H to before C ships); keep GT comparison in CI.

**W3. Auto thresholds are uncalibrated.** Ratios 0.5/0.2 and density 500/5000 are guesses. Mitigation: derive from the existing synthetic test cases; document the calibration dataset used.

**W4. "Zero behavior change" claim is false as stated.** Unified backbone is trackcorr; e.g. `standard_forward` originally forward-only. Fixed bundles must be verified to reproduce original outputs byte-for-byte. Mitigation: add a regression test comparing unified (fixed bundle) vs. legacy on a reference dataset before deprecating.

**W5. Per-trajectory quality (E) touches storage schema.** Risk of breaking `flowtracks`, `tracking_postprocess.py`, `quality_post.py`, and readers. Mitigation: reuse the existing `prio` array (quantized to `int32`) or a parallel sidecar file; avoid new block groups until validated. Consider deferring E to a later release.

**W6. Interactive prompts in batch/cloud.** `probe_confirm` prompting is unusable in CI and cloud. Mitigation: default `probe_confirm: false` in non-TTY; auto-pick best; allow `--probe-confirm` CLI flag.

**W7. Non-stationarity beyond probing.** A single config cannot be optimal everywhere. Mitigation (advanced, later): allow per-chunk config in `pyptv_batch_parallel.py` (each shard runs `Estimator → Engine → Refiner` on its slice). Out of scope for v1 but enabled by the 3-stage architecture.

**W8. Scope creep.** 10 milestones with 8 agents is a lot. Mitigation: ship A+B+C (auto-params + gate + probe) as v1; everything else is v2. Reorder: D (report) before E/F/G/H.

**W9. Dependency on unaudited primitives.** `tracking_feasibility`, `quality_post`, `run_postprocess_passes` may have latent bugs. Mitigation: add targeted unit tests for each primitive before relying on them.

**W10. Transparency.** Users must see what auto chose and why. Mitigation: always print resolved config + rationale + probe table; store in `TrackingDecision`.

### Recommended reordering (v1 first)

1. **H0 (minimal synthetic GT scorecard)** — 1-command baseline (clean + noisy + dense ghost cases) so every threshold/heuristic in A, C, K, L is calibrated against GT from day 1.
2. **A (auto-params)** — wired into `_auto_regime` (fix `_auto_regime` / `sample_probe_windows` layout bug first) + tests against synthetic data.
3. **B (conditioning gate)** + **F0 (lightweight `TrackingDecision` JSON/dict)** — needed before parallel/sharding.
4. **C (probe)** — coordinate-descent search, budget-capped (`probe_budget_seconds`).
5. **D (post-run report)**.
6. **L + K (high-density `trackcorr v2` + 2-Pass Clean-and-Claim)** — Tight-Core + 2D-target pruning + polynomial/shared-target `track_assisted` + conflict-component Hungarian.
7. **I (docs/YAML)**.
8. **G (parallel)** — shares `TrackingDecision` from F0 across shards.
9. **H (full CI validation)**.
10. **E (per-traj quality)** — defer to v2.
11. **J (cleanup)** — after validation.

### Open questions to resolve before implementation

1. Is there a canonical synthetic GT dataset available for calibration of thresholds? (Check `tests/test_data`.)
2. Should probe be on by default in cloud? (Recommend: on, auto-pick, budget-capped.)
3. Confirm backward-compat byte-identity is achievable for all legacy presets.

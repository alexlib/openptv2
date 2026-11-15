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

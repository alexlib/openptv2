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

---

## 1. Current state (done)

- [x] `src/openptv2/plugins/unified_tracking.py` — `Tracking` plugin, auto-regime, backbone = trackcorr `full_forward()`.
- [x] Registered `unified` / `unified_tracking` in `plugins/loader.py` and `tracking_registry.py`.
- [x] `unified` preset + config bundle in `tracking_presets.py`.
- [x] `sample_probe_windows()` + `aggregate_probe_metrics()` static helpers.
- [x] openptv-cloud `pipeline.py` parallel gate allows `unified`.
- [x] Branches created and committed.

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

### C2. Grid search when probe is poor
- Define thresholds: if median track length < `min_acceptable_length` (default 5) OR jump_rate > 0.1, try variants:
  - v_max × {1.0, 1.5, 2.0}
  - delta_t ∈ {1, 3, 5} (if slow flow)
  - use_2d ∈ {False, True}
  - bidirectional ∈ {False, True}
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
- In `post.py` or tracking, compute score from: length, reciprocity, smoothness, gap count, kink.
- Store in linkage (prio array or new field) — coordinate with `storage/linkage_blocks.py`.
- Propagate through `flowtracks`/`export` as weight.

### E2. Downstream weighting
- `export.py` and ensemble binning optionally weight by quality.

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
- In `openptv-cloud/pipeline.py` and `track_parallel.py`, compute `workers = min(requested, total//(overlap+2), cpu_count)`; already in openptv2 formula; ensure used.

### G2. Probe on 1 worker, full run parallel
- Probe sequentially; full run uses `parallel_workers`.

### G3. Cloud sharding
- Ensure `run_shard`/`stitch_shards` use the same `TrackingDecision` and resolved config.

---

## 9. Milestone H — Validation harness

**Owner:** agent H. **Effort:** 2 sessions.

### H1. Synthetic GT comparison
- Extend `tests/test_sequential_vs_parallel_tracking.py` style to compare unified vs. legacy on synthetic cases; report yield/precision/jump-rate.

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
- **No new Cython.** Reuse existing kernels.
- **I/O contract unchanged.** Zarr `ptv_is` linkage group.
- **Each task PR should be small and focused**, referencing this plan's milestone/task IDs.

---

## 12.4. Backbone improvements (trackcorr v2)

Before unifying, make the backbone itself better. Improvements are split into
Python-level (low risk, do first) and Cython-level (higher risk, later).

Python-level:
- B1. Early ghost filtering pre-pass: compute ghost probabilities (rcm/rcm_blob)
  per correspondence; drop points above threshold; keep originals as backup.
- B2. Adaptive search radii from `measure_motion_scale` (1.5x measured displacement).
- B3. Frame-quality awareness: skip/interpolate frames with far fewer correspondences.
- B4. Kink-based link rejection + gap relink repair.

Cython-level (later):
- B5. 4BE n+2 lookahead as tie-breaker in epipolar scoring.
- B6. 2D leaf re-ranking inside epipolar candidate selection (ambiguous frames only).
- B7. Multi-hypothesis deferred resolution (top-k + Hungarian on components).

### Milestone L — Backbone improvements
**Owner:** agent L. **Effort:** 2–4 sessions.
- L1. Implement B1 ghost filtering pre-pass + tests on synthetic data.
- L2. B2 adaptive radii.
- L3. B4 link rejection.
- L4. B3 frame quality.
- L5. Validate B1–B4 vs. baseline on synthetic GT (expect precision up).
- L6 (later). B5–B7 Cython work.

## 12.5. Layered / peeling tracking (new idea)

User-proposed strategy to reduce ghosts and improve robustness: instead of
building the full 3D cloud up front (which inflates ghost count), track
top-down in quality layers:

1. Sort correspondences by ghost probability (rcm/rcm_blob) ascending.
2. Layer 0: track only the top quality tier (e.g. top 30%) → high-confidence
   tracks.
3. Remove tracked particles/targets from all frames ("clean the frames").
4. Layer 1: track the next tier on the cleaned, sparser data.
5. Repeat until no more tracks.
6. Final postprocess (gap relink, reciprocity, confirm) on the accumulated set.

Benefits:
- Ghosts (low quality) are tracked last against a sparse, cleaned set and
  most are rejected by gates/reciprocity.
- Each layer is easier (sparser) → fewer ambiguities.
- Natural fit with existing track-assisted re-correspondence (claims free targets).

Implementation notes:
- New module `src/openptv2/layered_tracking.py` with `peel_and_track(store, ...)`.
- Index bookkeeping: each layer's particles are a subset of the original
  correspondence indices; remap to a unified per-frame index space before
  writing `ptv_is`, or write per-layer linkage and merge.
- Config: `peeling: true|false`, `peel_layers: 3`, `peel_quantiles: [0.3,0.6,1.0]`.
- Can be invoked by unified (`peeling: auto` enables when ghost rate is high)
  and by a standalone preset.
- Cost: N full tracking passes; mitigated by smaller clouds per layer and
  parallelism.

### Milestone K — Layered peeling tracking
**Owner:** agent K. **Effort:** 3–4 sessions.
- K1. Prototype on a single layer (quality-filtered subset) → verify tracking works on filtered cloud.
- K2. Index remapping + accumulation into one linkage store.
- K3. Multi-layer loop with cleaning.
- K4. Final postprocess + tests vs. synthetic GT (expect higher precision, similar yield).
- K5. Wire into unified as `peeling` option.

## 13. Open questions

1. Should probe run be on by default in cloud/batch mode? (Proposed: on, with auto-pick; disable via `probe: false`.)
2. Quality score storage: reuse `prio` array or new block group? (Needs coordination with `storage/linkage_blocks.py`.)
3. Should `unified` fully remove legacy plugins or keep them for 1 release? (Proposed: keep as aliases for 1 release, then remove.)

---

## 14. Progress log

- 2026-11-14: Branch created; scaffold + auto-heuristic + multi-window probing committed.
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

**W5. Per-trajectory quality (E) touches storage schema.** Risk of breaking `flowtracks`, `post.py`, `export.py`, and readers. Mitigation: reuse the existing `prio` array or a parallel sidecar file; avoid new block groups until validated. Consider deferring E to a later release.

**W6. Interactive prompts in batch/cloud.** `probe_confirm` prompting is unusable in CI and cloud. Mitigation: default `probe_confirm: false` in non-TTY; auto-pick best; allow `--probe-confirm` CLI flag.

**W7. Non-stationarity beyond probing.** A single config cannot be optimal everywhere. Mitigation (advanced, later): allow per-chunk config in `track_parallel` (each shard resolves its own regime). Out of scope for v1 but should be noted.

**W8. Scope creep.** 10 milestones with 8 agents is a lot. Mitigation: ship A+B+C (auto-params + gate + probe) as v1; everything else is v2. Reorder: D (report) before E/F/G/H.

**W9. Dependency on unaudited primitives.** `tracking_feasibility`, `quality_post`, `run_postprocess_passes` may have latent bugs. Mitigation: add targeted unit tests for each primitive before relying on them.

**W10. Transparency.** Users must see what auto chose and why. Mitigation: always print resolved config + rationale + probe table; store in `TrackingDecision`.

### Recommended reordering (v1 first)

1. A (auto-params) — with tests against synthetic data.
2. B (conditioning gate).
3. C (probe) — limited grid, budget-capped.
4. D (post-run report).
5. I (docs/YAML).
6. G (parallel).
7. H (validation/CI).
8. F (decision record) — fold into C/D.
9. E (per-traj quality) — defer to v2.
10. J (cleanup) — after validation.

### Open questions to resolve before implementation

1. Is there a canonical synthetic GT dataset available for calibration of thresholds? (Check `tests/test_data`.)
2. Should probe be on by default in cloud? (Recommend: on, auto-pick, budget-capped.)
3. Confirm backward-compat byte-identity is achievable for all legacy presets.

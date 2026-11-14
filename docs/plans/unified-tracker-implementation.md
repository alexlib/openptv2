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

## 11. Milestone J — Cleanup

**Owner:** agent J. **Effort:** 1 session.

### J1. Deprecate individual plugins (not remove)
### J2. Update GUI/recommender to default to `unified`
### J3. Update openptv-cloud notebooks to use `unified`

---

## 12. Cross-cutting concerns

- **Tests must pass** on `feat/unified-tracker` before merge: `pytest tests/` in openptv2; openptv-cloud test suite.
- **No new Cython.** Reuse existing kernels.
- **I/O contract unchanged.** Zarr `ptv_is` linkage group.
- **Each task PR should be small and focused**, referencing this plan's milestone/task IDs.

---

## 13. Open questions

1. Should probe run be on by default in cloud/batch mode? (Proposed: on, with auto-pick; disable via `probe: false`.)
2. Quality score storage: reuse `prio` array or new block group? (Needs coordination with `storage/linkage_blocks.py`.)
3. Should `unified` fully remove legacy plugins or keep them for 1 release? (Proposed: keep as aliases for 1 release, then remove.)

---

## 14. Progress log

- 2026-11-14: Branch created; scaffold + auto-heuristic + multi-window probing committed.
- (append as work proceeds)

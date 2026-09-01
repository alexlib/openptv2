# Plan: Tracker stress-test matrix with ground-truth ranking + report

Date: 2026-09-01
Status: draft — for review before implementation

## 1. Context

We just finished migrating the fixed 5-frame Burgers fixture to an on-demand,
ground-truth synthetic factory (`docs/plans/2026-09-02-refactor-burgers-synthetic-tests.md`,
Phases 0-5). The user now wants to go further: use ground-truth synthetic data
to seriously **stress-test the tracking algorithms**, find where each one
breaks, explain *why* to a fluid-dynamicist user (best-practice guidance), and
surface where a new/hybrid algorithm is needed.

Staged scope, per user direction:
* **Group A**: `fast_3d` vs `standard_forward` vs `two_phase`
* **Group B**: `nearest_hungarian_3d` (MyPTV) vs `full_multipass` vs `4be`
* **Group C** (later, not built now): remaining trackers vs `predictive_gmm_3d` (proPTV)

Priority stress axes (user picked all three categories):
* density vs motion + crossings (correspondence ambiguity / mislinks)
* turbulence/curvature + gaps (non-constant velocity + temporal dropout)
* long sequences + calibration error (drift accumulation + epipolar sensitivity)

Deliverable: pytest assertions (regression-safe) **and** a report generator
(markdown, run manually) — both requested by the user.

## 2. What already exists (reuse, don't rebuild)

Research (this session) found a **mature, working benchmarking stack** — the
right move is to extend it, not build a parallel one:

* `openptv2.benchmarking.ScenarioSpec` / `generate_scenario`
  (`src/openptv2/benchmarking/scenario.py`) — already supports `noise_mm`,
  `gap_probability`, `ghost_ratio`, `flow_type="turbulent"`,
  `entering_particles`/`leaving_particles`, and `CrossingSpec` (engineered
  mislink stressor, currently only exercised by
  `scripts/create_synthetic_turbulent.py:55-58`).
* `openptv2.benchmarking.write_experiment` (`experiment.py`) — writes a full
  runnable experiment dir (`cal/`, `res/rt_is.*` + `res/origin_*.txt` ground
  truth, `img/*_targets`, `parameters_Run1.yaml`).
* **`scripts/benchmark_utils.py`** — the actual shared harness behind
  `scripts/bench_trackers.py` and the marimo dashboard:
  `run_single_tracker`/`run_all_trackers` (isolates a copy, calls
  `bm.run_tracker`, times it), `per_tracker_overrides` (derives
  **tracker-correct** parameter names/scales via
  `openptv2.tracking_recommender._suggest_params` + `compute_dataset_stats` —
  critical because `nearest_hungarian_3d` uses `v_max`/`a_max`, not
  `dvxmax`/`dacc`), `read_gt_frames`/`build_true_tracks`/`build_ghost_frames`
  (parse `origin_*.txt`), `combined_metrics` (merges
  `compute_identity_metrics` + link-level metrics + `e_track` into one row).
* **`scripts/create_synthetic_turbulent.py::make_dataset`** — the existing
  on-demand one-axis-at-a-time dataset generator, already includes a
  **Trackability Number** `M = v·dt / d_nn` computation tying particle density
  to motion — this is exactly the "density vs motion" axis, already built.
* `tests/unit/test_synthetic_calibration.py:150-160` — existing pattern for
  perturbing a `Calibration` (`cal.set_pos(...)`, `cal.set_angles(...)`,
  `cal.int_par.cc += ...`) — reuse for the new calibration-error axis.

**Two real gotchas found and must be handled, not rediscovered the hard way:**
1. **Every scenario must write to a RunStore (zarr) always — not a
   `two_phase`-specific patch.** `docs/plans/archive/2026-09-01-zarr-only-final-cutover-plan.md`
   (same-day, already executed) confirms the tracking plugins already prefer
   store input across the board — `default_tracking.py`,
   `myptv_3d_tracking.py:257`, `myptv_2d_tracking.py:243`,
   `hybrid_deltat_3d.py:279`, `proptv_tracking.py:370`,
   `two_phase_tracking.py:306/359` all call `Frame.read(...,store=store)` —
   ASCII is only ever the legacy fallback when no store exists. `two_phase`
   just has the *worst* fallback behavior (silent algorithmic downgrade
   instead of equivalent results), which is what surfaced it first, but
   every tracker in Groups A/B should be benchmarked against the same
   store-backed path the real pipeline uses, not the ASCII path nobody
   actually runs in production. Every stress scenario therefore calls
   **both** `bm.write_experiment(...)` (still needed for `cal/*.ori` +
   `parameters_Run1.yaml`) **and** `bm.write_dataset_store(rig, frame_gt,
   DatasetSpec(dir=out_dir))` (zarr) unconditionally — zarr is the read
   source of record, ASCII stays only as the interchange/debug artifact.
   Reference: `tests/helpers/synthetic_scene.py::make_cavity_scene` (this
   session's own synthetic factory) already does this correctly — it writes
   **only** to `RunStore`, no ASCII `_targets` at all — use it as the
   pattern to match, not two_phase's fallback as an excuse to special-case.
2. **On-sensor projection coverage.** A perturbed or non-standard rig
   (especially `calibration_error_scenario`) can push particles off-sensor,
   silently shrinking the effective dataset. Reuse the existing
   `project_to_pixels` + coverage-check pattern from
   `tests/unit/test_benchmarking.py::test_rig_projects_to_sensor` (asserts
   >90% of a point spread lands within `[0,imx]×[0,imy]` per camera) as a
   generator-level assertion in every `scripts/stress_scenarios.py`
   function, right after building the rig and before writing the scenario.
   `bm.make_standard_rig()` already satisfies this for the unperturbed case;
   for `calibration_error_scenario`, check coverage *after* perturbing and
   fail loudly (not silently degrade) if it drops out of range — a genuinely
   new 4-camera rig is only needed if `make_standard_rig`'s geometry can't be
   tuned (via its `volume`/`center`/`image_size`/`cc` params) to keep
   coverage high across the perturbation range actually tested.
3. `standard_forward`/`full_multipass` need their own kinematic-bound
   overrides (reusing `fast_3d`-tuned bounds yields 0 links — precedent:
   `tests/batch/test_tracking_presets_benchmark.py:121-138`); use
   `benchmark_utils.per_tracker_overrides` for **every** tracker rather than a
   single shared dict (this is exactly what bit the Phase 5 harness earlier
   this session before the frame-offset fix).

## 3. Design

### 3.1 `scripts/stress_scenarios.py` (new) — one-knob-at-a-time generators

Modeled directly on `create_synthetic_turbulent.make_dataset`. Each function
takes a `severity` (or axis-specific param) and `out_dir`, returns
`(yaml_path, first_frame, n_frames)`:

* `density_motion_scenario(out_dir, trackability_m, num_frames=10)` — vary
  `num_particles` at fixed `velocity` to hit a target Trackability Number M
  directly (reuse the `make_dataset` M formula), plus 2 `CrossingSpec`s always
  on (engineered mislink stressor is not optional — it's the point of this
  axis).
* `turbulence_gap_scenario(out_dir, velocity_jitter, gap_probability, num_frames=12)`
  — `flow_type="turbulent"`, sweep `velocity_jitter` (curvature/acceleration
  stress) and `gap_probability` (temporal dropout) together, matching the
  cavity-flow use case.
* `long_sequence_scenario(out_dir, num_frames)` — 40-80 frames, otherwise
  moderate settings; paired with a **windowed physics-metric** helper (new,
  small — `compute_physics_metrics`/`acceleration_kurtosis` take a whole
  track dict, no native windowing) that slices each track into overlapping
  windows and reports how completeness/purity/kurtosis drift over the
  sequence, not just a single end-to-end number.
* `calibration_error_scenario(out_dir, d_pos_mm, d_angle_deg, d_cc_pct, num_frames=10)`
  — build `bm.make_standard_rig()`, perturb each `rig.cals[i]` in place
  (`set_pos`, `set_angles`, `int_par.cc *= (1+d_cc_pct)`) using the pattern
  from `test_synthetic_calibration.py`, **then** `write_experiment` — this
  tests whether a tracker's gates are robust to imperfect calibration, not
  whether BA can recover it.
* All of the above call `write_dataset_store` **unconditionally**, every
  time, for every tracker — not just when `two_phase` is in the group under
  test (see gotcha 1). Each generator also runs the on-sensor coverage check
  (gotcha 2) before writing, and raises rather than silently returning a
  thinner-than-requested scenario.

### 3.2 Test files (pytest, regression-safe thresholds)

* `tests/benchmarks/test_tracker_stress_group_a.py` — `fast_3d`,
  `standard_forward`, `two_phase` × the 4 axis generators above, small/fast
  severities (keep each test under ~10s). Assert: no crash, completeness
  degrades *gracefully* (bounded, not a cliff to 0) as severity increases,
  and — the "verify the registry" angle — that `two_phase`'s
  `supports_gap_relinking=True` claim actually holds up better than `fast_3d`
  (no `supports_gap_relinking`) on the gap axis.
* `tests/benchmarks/test_tracker_stress_group_b.py` — same axes/pattern for
  `nearest_hungarian_3d`, `full_multipass`, `4be`. Assert `full_multipass`
  (accuracy_ranking="highest", supports_gap_relinking) beats `4be`
  (accuracy_ranking="standard") on the gap + crossing axes specifically —
  falsifiable claims from the registry, not arbitrary thresholds.
* Both marked `@pytest.mark.slow @pytest.mark.ci`, reuse
  `scripts/benchmark_utils.py` for run+score (confirm during implementation
  whether `scripts/benchmark_utils.py` is already imported by any test —
  `tests/unit/test_tracker_quality.py` does this, so the import path is
  proven).

### 3.3 `scripts/tracker_stress_report.py` (new) — the report generator

Modeled on `scripts/bench_trackers.py`'s CLI/table pattern. For each group
(`--groups a,b`, `c` deferred), for each axis, sweep 4-5 severities × 2-3
seeds, run `per_tracker_overrides` + `run_all_trackers`, collect rows, and
write `docs/reports/tracker-stress-YYYY-MM-DD.md` with:

1. **Per-axis table**: tracker × severity → completeness / purity /
   fragmentation / ghost_capture_rate / ms-per-frame.
2. **Robustness slope**: Δcompleteness/Δseverity per tracker per axis (flat
   = robust, steep = fragile) — the actual "which tracker for which regime"
   number.
3. **Registry cross-check**: empirical speed/accuracy ranking vs each
   tracker's self-declared `TrackerInfo.speed_ranking`/`accuracy_ranking` —
   flags where the docs and reality disagree.
4. **Narrative sections** (best-practice guidance + gap analysis): written by
   the script from the computed table (e.g. "no Group A/B tracker holds
   completeness > 0.6 above gap_probability 0.15 — candidate for a
   gap-aware hybrid"), not hand-authored prose to keep it reproducible.

## 4. Explicitly out of scope for this plan

* Group C (`predictive_gmm_3d` + remaining trackers) — noted as follow-on.
* Camera-dropout axis — not requested by the user this round; note as a
  documented future axis in the report script's docstring.
* Any actual new/hybrid algorithm implementation — this plan produces the
  *evidence* for where one is needed, not the algorithm itself.

## 5. Verification

* `uv run ruff check scripts/stress_scenarios.py scripts/tracker_stress_report.py tests/benchmarks/test_tracker_stress_group_a.py tests/benchmarks/test_tracker_stress_group_b.py`
* `uv run pytest tests/benchmarks/test_tracker_stress_group_a.py tests/benchmarks/test_tracker_stress_group_b.py -m slow -v` → all pass, each test completes in seconds not minutes
* `uv run python scripts/tracker_stress_report.py --groups a,b` → writes
  `docs/reports/tracker-stress-<date>.md`, eyeball for sane numbers (no
  all-zero rows, no crashes)
* Spot-check the RunStore-always requirement: for at least one scenario per
  group, confirm `write_dataset_store` output actually changes tracker
  behavior vs ASCII-only (rerun one scenario both ways per tracker, diff
  completeness) — `two_phase` should show the largest delta, but confirm the
  others are neutral-or-better too, not just unaffected.
* For `calibration_error_scenario`: confirm the on-sensor coverage check
  (gotcha 2) actually fires — deliberately push a perturbation past the
  coverage threshold and confirm the generator raises instead of silently
  writing a thin scenario.

## 6. Effort

Phase A (Group A: scenarios + tests): ~1 day
Phase B (Group B: tests, reusing Phase A's scenarios): ~0.5 day
Phase C (report generator): ~0.5-1 day
Total: ~2-2.5 days, staged — Group A can ship and be reviewed before Group B.

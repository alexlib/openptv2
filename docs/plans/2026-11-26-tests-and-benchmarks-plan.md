> Status 2026-11-26: P1 items 1-5 below are implemented as tests/unit/test_yaml_roundtrip.py, test_determinism.py,
> test_plugin_lifecycle.py, test_memory_growth.py, test_properties.py (and test_module_contracts.py). Retired names in this
> plan (tracking_presets.py, hybrid_deltat_3d, unified) now resolve through aliases; the registry has two_phase, trackcorr, fast_3d.

# Tests & Benchmarks Plan — openptv2

## Current State Summary

**Test suite** (~150 unit + ~16 batch + ~56 GUI + ~2 benchmark + ~1 perf):
- `tests/unit/` — per-module unit tests for algorithms (trafo, imgcoord, correspondences, tracker, calibration, segmentation, orientation, etc.)
- `tests/batch/` — CLI batch pipeline, parallel processing, plugin sequences, synthetic tracker
- `tests/gui/` — headless GUI tests (parameter managers, dialogs, CLI, image processing)
- `tests/benchmarks/` — tracker accuracy/speed benchmarks using `openptv2.benchmarking` harness
- `tests/perf/` — MMLUT performance benchmark only
- `tests/helpers/synthetic_scene.py` — synthetic scene factory (from the 2026-09-02 plan)
- pytest markers: `slow`, `gui`, `unit`, `parity`, `perf`, `integration`, `ci`
- Default run excludes `-m slow`

**Benchmarking harness** (`src/openptv2/benchmarking/`): `ScenarioSpec`, `generate_scenario`, `make_standard_rig`, `write_experiment`, `run_tracker`, `compute_identity_metrics` (proPTV F/C/purity/pmt + link-level yield/precision). Already unit-tested in `tests/unit/test_benchmarking.py`.

**Existing baseline data**: `bench/` contains 28 JSON step-by-step tuning results (Oct–Nov 2026). `scripts/bench_trackers.py` is a manual cross-tracker CLI benchmark.

## Gaps Identified

| Area | Gap |
|------|-----|
| **Module coverage** | `tracking_advice.py`, `tracking_feasibility.py`, `tracking_probe.py`, `tracking_recommender.py`, `tracking_registry.py`, `calibration_diagnostics.py`, `detect_plate.py`, `layered_tracking.py`, `reconnect.py`, `recorrespond.py`, `smoothness_filter.py`, `temporal_smoothing.py`, `validate.py`, `self_check.py` — check for missing unit tests |
| **Round-trip / serialization** | No tests that YAML round-trip parameters → calibration → tracker config and back |
| **Determinism / reproducibility** | No test that same inputs produce identical outputs across runs |
| **Plugin lifecycle** | No tests for plugin load/unload/switch (tracker presets in `tracking_registry.py`) |
| **Memory / leaks** | No memory-growth test for long sequences |
| **Property-based testing** | No hypothesis-style invariant tests (e.g., track length monotonicity, ID uniqueness) |
| **Contract tests** | No interface contract tests between correspondences → tracker → postprocess |
| **Multi-camera geometry** | No epipolar tests across varied camera baselines/angles |
| **CI regression benchmarking** | `bench/` JSONs are manual artifacts; no test compares current perf against baselines |
| **Per-module perf** | Only MMLUT has a perf benchmark; no per-algorithm timing |
| **Scaling** | No particle-count / frame-count / camera-count scaling benchmarks |
| **Interpreted vs compiled** | No benchmark measuring Cython compile speedup |
| **liboptv parity benchmark** | Parity tests exist (`test_liboptv_comparison.py`) but no timing/throughput benchmark vs liboptv |

## Recommended Tests

### P1 — Core correctness (fast, deterministic)

1. **YAML round-trip tests** (`tests/unit/test_yaml_roundtrip.py`)
   - `ControlPar`, `TrackPar`, `Calibration` → YAML → back → assert equality
   - Exercise `tracking_presets.py` preset load/save

2. **Determinism tests** (`tests/unit/test_determinism.py`)
   - Run `tracker.track()` twice on same synthetic scene → identical trajectories
   - Run correspondences twice → identical 3D cloud

3. **Plugin lifecycle** (`tests/unit/test_plugin_lifecycle.py`)
   - Load each registered tracker preset from `tracking_registry.TRACKER_REGISTRY`
   - Switch tracker mid-sequence, assert no crash/state leak
   - Test invalid preset name raises clear error

4. **Memory regression** (`tests/unit/test_memory_growth.py`, marked slow)
   - Track 500 frames, assert RSS growth is bounded (not linear in frames)

5. **Property-based invariants** (`tests/unit/test_properties.py`, using `hypothesis` if available, else manual strategies)
   - Track IDs are unique within a frame
   - Reconstructed track length ≤ true track length + gap tolerance
   - No reconstructed point lies outside the calibrated volume bounds

6. **Contract tests** (`tests/unit/test_module_contracts.py`)
   - `correspondences` output shape matches tracker input expectations
   - Tracker output schema matches postprocess input schema

### P2 — Integration & real data

7. **End-to-end pipeline on test_cavity** (`tests/integration/test_e2e_cavity.py`, marked slow)
   - Full detection → correspondences → tracking on `test_data/test_cavity`
   - Assert non-zero track count, median track length > threshold

8. **Real data regression** (`tests/integration/test_real_data_regression.py`, marked slow)
   - Run on `test_data/proptv_500_25` or `test_data/synthetic_turbulent`
   - Compare against checked-in expected metric bounds (F, C, purity)

9. **Epipolar geometry sweep** (`tests/unit/test_epipolar_sweep.py`)
   - Vary camera baseline, focal length, principal point → assert epipolar lines pass through corresponding points

### P3 — GUI & CLI (existing, extend)

10. Extend `tests/gui/` with coverage for `tracking_advice`, `tracking_recommender` parameter dialogs

## Recommended Benchmarks

### B1 — Automated CI regression benchmarks (new)

1. **Benchmark baseline comparison** (`tests/benchmarks/test_ci_regression.py`)
   - Load `bench/baseline_2026-10-01.json` as reference
   - Run `scripts/bench_trackers.py` on default synthetic_turbulent case
   - Assert metrics (precision, yield, ms/frame) within tolerance of baseline (e.g., ±5%)

2. **Per-module timing** (`tests/benchmarks/test_module_timing.py`)
   - Time each algorithm module (`correspondences`, `tracker.track`, `calibration.dlt`, etc.) on fixed synthetic input
   - Regression gate: no module may regress >10% vs baseline

### B2 — Scaling benchmarks (manual + CI)

3. **Scaling suite** (`tests/benchmarks/test_scaling.py`, marked perf/ci)
   - Particle count: 50, 200, 1000, 5000 particles/frame
   - Frame count: 10, 50, 100 frames
   - Camera count: 2, 4, 8 cameras
   - Report ms/frame and memory per configuration

4. **Interpreted vs compiled Cython** (`tests/benchmarks/test_cython_speedup.py`)
   - Run key hot-path functions interpreted (no build) vs compiled (.so)
   - Report speedup factor

5. **liboptv parity throughput** (`tests/benchmarks/test_liboptv_benchmark.py`)
   - Compare openptv2 tracker vs liboptv on same dataset: ms/frame, track yield, identity metrics

### B3 — Benchmark data management

6. **Baseline JSON versioning** — Store `bench/baseline_<date>.json` with git hash; add a script to regenerate and compare:
   - `scripts/update_benchmark_baseline.py` — regenerate baseline from current code
   - `scripts/check_benchmark_regression.py` — fail CI if metrics drift beyond tolerance

## Implementation Order

```
Week 1: P1 tests 1-4 (round-trip, determinism, plugin lifecycle, memory)
Week 2: P1 tests 5-6 (properties, contracts) + P2 test 7 (e2e cavity)
Week 3: B1 benchmark 1 (CI regression) + baseline JSON tooling
Week 4: B2 benchmarks 2-4 (module timing, scaling, Cython speedup)
Week 5: B2 benchmark 5 (liboptv) + P2 tests 8-9
```

## Validation

- `uv run pytest -m "not slow"` — full fast suite must pass
- `uv run pytest -m slow` — slow suite must pass (may be overnight)
- `uv run pytest tests/benchmarks/ -m ci` — CI regression benchmarks pass
- `uv run ruff check` and `uv run mypy` clean on new test files
- Coverage: new test files should not reduce overall coverage below current ~34%

## Open Questions

1. Is `hypothesis` available/desired for property-based tests, or should we use manual parametrized strategies?
2. What tolerance (±%) should CI regression benchmarks use before failing a PR?
3. Should benchmarks run on every PR or only on `main` merges (they can be slow)?
4. Is there a target minimum coverage percentage to enforce?

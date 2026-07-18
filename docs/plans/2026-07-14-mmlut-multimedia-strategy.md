# Multimedia / mmlut strategy: measure, fix, and actually use the look-up table

> Order: implement AFTER `2026-07-14-orientation-review-fixes.md`.

## Context

Ray tracing through refractive interfaces (air → glass → water) is the single most
repeated geometric computation in PTV: every projection of a 3D point into a camera
(`img_coord` / `flat_image_coord`) needs the radial-shift factor from
`multimed_r_nlay_iterative` (`src/openptv2/algorithms/multimed.py:220`) — an
iterative Snell solve, up to 40 iterations of `atan`/`asin`/`sin`/`tan` per call.
The multimedia look-up table (`MmLut`, `src/openptv2/algorithms/calibration.py:165`)
exists to amortize this: `init_mmlut` (`multimed.py:602`) precomputes the factor on
an (r, z) grid (spacing `rw = 2.0` mm, hardcoded) per camera, and
`get_mmf_from_mmlut` replaces the iterative solve with a bilinear interpolation.

**Research findings (code audit, 2026-07-14):**

1. **The LUT is only used by the tracker.** The single call site of `init_mmlut` in
   the pipeline is `tracking_run.py:40-50` (TrackingRun init). The correspondence
   pipeline (`epi.py: epi_mm` → `flat_image_coord`, called per target per camera
   pair inside `match_pairs`), 3D determination (`point_positions`), `sortgrid`,
   the GUI sequence loop (`gui/ptv.py` — `py_start_proc_c` never initializes it),
   and the batch pipeline all run with `cal.mmlut.data is None`, so **every
   projection outside tracking pays the full iterative solve**. The consumers are
   already LUT-aware (`imgcoord.py:396`: `mmlut = cal.mmlut if
   cal.mmlut.is_initialized else None`), so this is purely a missing-init problem.
2. **Suspected off-by-one in the LUT lookup bounds check.** In both copies of the
   lookup (`multimed.py:465 get_mmf_from_mmlut` and
   `imgcoord.py:89 _get_mmf_from_mmlut_core`): `max_v = mmlut_nr * mmlut_nz` and the
   vertex checks use `v4_x > max_v`. `data` has exactly `nr * nz` elements (valid
   indices `0 .. nr*nz - 1`), so index `== max_v` passes the check and reads one
   element past the end. Reachable when `ir == nr - 1` and `iz == 0` (then
   `v4_2 = nr*nz` exactly). Compiled with `boundscheck(False)` in
   `imgcoord.py` this is a real out-of-bounds read. (Same quirk exists in liboptv
   C; here it must be fixed, not preserved — it's memory safety, not numerics.)
   Similarly audit `if ir > mmlut_nr` / `if iz > mmlut_nz` (should exclude the last
   cell, e.g. `ir >= nr - 1` handling) — write the edge-position test first, then fix.
3. **The lookup is implemented twice** (`multimed.get_mmf_from_mmlut` and
   `imgcoord._get_mmf_from_mmlut_core`) — any bounds fix must land in both or,
   better, one should delegate to the other.
4. **No invalidation.** `orient()` copies `cal.mmlut` back into `cal_in` on success
   (`orientation.py:1040`), so a LUT built before recalibration silently survives a
   calibration change and would give wrong factors. Today this is masked only
   because the LUT is almost never initialized.
5. **No numbers.** There is no perf test quantifying (a) `init_mmlut` cost,
   (b) per-projection speedup, (c) the break-even point vs. number of
   tracers × frames × cameras, nor an accuracy test bounding the bilinear
   interpolation error vs. the direct solve. The user's hypotheses to test:
   savings depend on targets/frame, frame count, interface complexity
   (`nlay`, `n1/n2/n3` values). Note the iterative solve short-circuits to 1.0
   when `n1 == n2 == n3 == 1.0` (all-air), so the LUT buys nothing there.
6. **Minor:** `init_mmlut` stores `cal.mmlut.rw = int(rw)` (truncating cast; rw is
   conceptually a float grid spacing); the `nlay > 1` LUT fill is a pure-Python
   double loop (the `nlay == 1` path uses compiled `init_mmlut_data_fast` from
   `track_kernels_batch.py:382`).

## Goal

Make the LUT the default fast path for the whole pipeline, safely (bounds,
invalidation, accuracy), and prove the benefit with reproducible benchmarks so we
know when it pays off.

## Phase 1 — Measurement first (no behavior change)

Create `tests/perf/test_mmlut_benchmark.py` (marker `@pytest.mark.perf`, excluded
from default runs; follow the marker usage in `tests/unit/test_parallel_tracking.py`).
Use `test_data/test_cavity` calibrations/parameters (real water/glass case) plus a
synthetic all-air control. Measure with `time.perf_counter`, median of ≥5 repeats:

1. **Init cost:** `init_mmlut(vpar, cpar, cal)` per camera; record LUT dims (nr, nz).
   Cases: nlay=1 water (n1=1, n2≈1.5, n3≈1.33), nlay=2, all-air.
2. **Per-projection cost:** `flat_image_coord` for N random in-volume points,
   with `cal.mmlut.data = None` vs. initialized LUT. Report ns/projection and ratio.
3. **Break-even model:** compute and print `n_projections_to_amortize =
   init_time / (t_direct - t_lut)`; express as tracers-per-frame × frames for a
   4-camera setup (correspondence does ~2 projections per epipolar candidate per
   camera pair; determination ~num_cams per point).
4. **Accuracy:** max and RMS |mmf_lut - mmf_direct| over a dense grid of in-volume
   points, and the resulting pixel-space projection error (must stay ≪ detection
   noise, e.g. < 0.01 px). This validates `rw = 2.0` or motivates making it a
   parameter.

Persist results as printed table in the test output (no fixture files). These
numbers decide whether Phase 3/4 items are worth it — record them in the PR
description.

## Phase 2 — Correctness fixes (small, do regardless of numbers)

1. Write a failing unit test in `tests/unit/test_multimed.py` that queries
   `get_mmf_from_mmlut` at positions mapping to `ir == nr-1, iz == 0` and to the
   exact far corner, asserting no OOB (run uncompiled AND compiled; in compiled
   mode an OOB read won't crash reliably — assert the returned value equals the
   uncompiled reference instead).
2. Fix the vertex bound checks in BOTH lookups (`multimed.py` and `imgcoord.py`):
   valid flat index range is `0 .. nr*nz - 1`; the interpolation cell must satisfy
   `ir + 1 <= nr - 1` and `iz + 1 <= nz - 1`, else return 0.0 (caller falls back to
   the iterative solve — existing semantics for out-of-LUT points).
3. Deduplicate: make `multimed.get_mmf_from_mmlut` delegate to
   `imgcoord._get_mmf_from_mmlut_core` (or move the core into `multimed.py` and
   import it in `imgcoord.py` — pick the direction that avoids a circular import;
   `imgcoord` already imports from `multimed`, so the core should live in
   `multimed.py`).
4. `cal.mmlut.rw = rw` (keep float; `MmLut.rw` field type allows it — check the
   dataclass field and the C-parity expectations in `tests/unit/test_multimed.py`).
5. **Invalidation:** add `Calibration.invalidate_mmlut()` (sets `mmlut.data = None`)
   and call it at the end of a successful `orient()` and `raw_orient()` instead of
   copying the stale LUT (in `orientation.py:1040`, replace
   `cal_in.mmlut = copy.deepcopy(cal.mmlut)` with resetting the LUT). Add a unit
   test: init LUT → run orient → `is_initialized` is False afterward.

## Phase 3 — Use the LUT in the whole pipeline (the payoff)

1. Add one helper, `prepare_mmluts(vpar, cpar, cals)` in
   `src/openptv2/algorithms/multimed.py`: for each cal with
   `not cal.mmlut.is_initialized`, call `init_mmlut`. Skip entirely when
   `cpar.mm.n1 == n2 == n3 == 1.0` (all-air: iterative solve already
   short-circuits; building a LUT adds overhead for nothing).
2. Call it at the natural pipeline entry points (mirror how `tracking_run.py:50`
   already does it):
   - `gui/ptv.py: py_start_proc_c` after calibrations are loaded (covers GUI
     sequence + correspondences),
   - the batch pipeline entry (`openptv2/batch/pyptv_batch.py` and
     `pyptv_batch_parallel.py`) where cals/params are loaded,
   - keep `tracking_run.py` as is (it already guards on `is_initialized`).
   Do NOT bury the init inside `correspondences()` itself — implicit heavy work in
   a hot function is the wrong altitude; explicit init at run start matches the
   existing tracker pattern.
3. Parity guard: run the Phase 1 accuracy comparison as a fast unit test (small
   grid) so the LUT path can never silently drift from the iterative path.
4. Re-run the Phase 1 benchmark and record before/after wall time of one
   `test_cavity` sequence run (correspondences over the 4 frames) in the PR.

## Phase 4 — Optional follow-ups (create separate dated plans if the numbers justify)

- Vectorize the `nlay > 1` LUT fill (currently a Python double loop over nr × nz).
- Make `rw` configurable via `VolumePar`/YAML if Phase 1 shows accuracy or speed
  would benefit from a different grid spacing.
- Persist LUTs to disk keyed by a hash of (calibration, mm params, volume) to skip
  re-init across runs of the same experiment.
- Batch LUT lookup (array-in/array-out) for `point_positions`-style vectorized
  callers.

## Explicitly out of scope

- Parallelizing the iterative ray trace itself (the LUT makes it a non-hot path).
- Changing the physical model (`multimed_r_nlay_iterative` numerics, n_iter=40,
  tol=0.001) — parity with liboptv stands.

## Verification

```bash
# Rebuild after any algorithms/*.py change (REQUIRED)
uv run python setup.py build_ext --inplace

# Correctness
uv run pytest tests/unit/test_multimed.py tests/unit/test_multimed_coverage.py tests/unit/test_imgcoord*.py -v

# Pipeline still works end to end (correspondences + tracking smoke)
uv run pytest tests/unit/test_track.py tests/unit/test_track3d.py tests/unit/test_correspondences.py -v --tb=short

# Benchmarks (report numbers in PR)
uv run pytest tests/perf/test_mmlut_benchmark.py -m perf -v -s

# Lint
uv run ruff check src/openptv2/algorithms/multimed.py src/openptv2/algorithms/imgcoord.py tests/
```

Success criteria: no unit regressions; LUT-vs-direct projection error < 0.01 px on
test_cavity; measured per-projection speedup and break-even point documented; a
test_cavity correspondence run is measurably faster with Phase 3 enabled.

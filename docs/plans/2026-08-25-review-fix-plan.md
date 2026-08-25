# 2026-08-25 Code Review Fix Plan

Findings from the full-repo review of 2026-08-25, split into what is fixed
now and what is planned for later implementation.

## Fixed in this pass

### 1. Packaging: wheel omits `storage` / `benchmarking` (HIGH)

`[tool.setuptools] packages` in `pyproject.toml` listed only
`openptv2`, `.algorithms`, `.gui`, `.batch`, `.plugins`. The `storage/`
and `benchmarking/` subpackages were silently excluded from wheels,
breaking the declared `openptv2-convert-legacy` entry point
(`openptv2.storage.legacy:main`) for pip-installed users.

**Fix:** add `openptv2.storage` and `openptv2.benchmarking` to the
package list. Verify with a wheel build + import smoke test before the
next release.

### 2. CI: no test/lint gating (HIGH)

Only `cibuildwheel.yml` and `deploy_docs.yml` existed; pytest, ruff and
mypy never ran on PRs, and the PyPI-publish verification job is disabled
(`if: "false"`).

**Fix:** add `.github/workflows/ci.yml` running `uv run pytest -m
"not slow"` plus ruff on ubuntu/windows across supported Python versions.
(A mypy job was attempted but the codebase has ~4k pre-existing mypy
errors; type-check gating is a follow-up — see the table below. Enabling
the release `test_package` job is also left as a follow-up.)

### 3. Scratch-buffer overflow in clique matching (HIGH)

`src/openptv2/algorithms/correspondences.py`: scratch arrays are
allocated with `num_cams * NMAX` rows but downstream stages were invoked
with `scratch_size=4*NMAX`; writes past the allocation corrupt memory in
compiled mode (`boundscheck(False)`).

**Fix:** cap the per-stage scratch size at the actual allocation.

### 4. Hardcoded `max_targets=1000` in public correspondences path (HIGH)

`src/openptv2/correspondences.py` builds an `AlgoFrame(max_targets=1000)`
and assigns targets wholesale; >1000 targets/camera overruns fixed SoA
buffers in compiled mode. `tracker.py` already sizes dynamically — this
call site was missed.

**Fix:** size the frame from actual target counts (dynamic capacity).

### 5. `MAX_CANDS` stride bug in `sort_candidates_by_freq` (HIGH)

`src/openptv2/algorithms/track.py`: hardcoded stride 4 while
`constants.MAX_CANDS = 32`; inner scan crosses candidate-group
boundaries whenever `MAX_CANDS != 4`.

**Fix:** use `constants.MAX_CANDS` as the stride.

### 6. NMAX off-by-one guards (MEDIUM)

Guards of the form `if p1 > NMAX: continue` allow `p1 == NMAX`, which
indexes one row past `tusage[num_cams, NMAX]` in compiled mode.

**Fix:** change to `>= NMAX`.

### 7. Working-tree hygiene (HIGH)

- Stray untracked venv directory `openptv2/` at repo root (not covered
  by `.gitignore`, hidden by local `status.showUntrackedFiles=false`).
- Tracked artifacts: `.coverage`, `scratch/`, `.slim/`,
  `diagnose_fast3d_vs_myptv.py`.

**Fix:** delete the stray venv; untrack artifacts via `git rm --cached`
and extend `.gitignore`; remove the local git override.

## Planned for later

These review findings are recorded but intentionally not addressed yet:

| # | Finding | Where | Severity |
|---|---------|-------|----------|
| 1 | Delete/deprecate legacy duplicate store | `src/openptv2/storage/zarr_store.py` (unpadded frame keys, crash in `inspect_zarr_store` at :474) | HIGH |
| 2 | Parallel-safety contract for sequence plugins | `batch/pyptv_batch_parallel.py` + plugin Protocol (shared side files like `res/mask_areas.csv` race) | MEDIUM |
| 3 | proptv plugin ignores its config | `plugins/proptv_tracking.py:233` no-op `np.where`, `:240` hardcoded cost weights | MEDIUM |
| 4 | Deduplicate compiled vs Python four-camera matching | `algorithms/correspondences.py:416` vs `:531` (numerically divergent zero-distance guard) | MEDIUM |
| 5 | Factor shared tracking-plugin scaffolding into base class | `myptv_3d_tracking.py` / `myptv_2d_tracking.py` / `proptv_tracking.py` | MEDIUM |
| 6 | Broad exception swallowing | `zarr_store.py:25,47`, `cython_3d_tracking.py:75-97` silent success when tracker None | MEDIUM |
| 7 | Replace hot-loop `print()` with logging | `algorithms/track.py:1178-1207` | LOW |
| 8 | Dead code cleanup | `track.py` unused `dmin`, discarded arrays, duplicate candidate-search kernels | LOW |
| 9 | Store `close()` / context-manager protocol; stop reaching into `_open_run_store` | `storage/run_store.py`, plugins | LOW |
| 10 | Remove empty stub packages or populate them | `autoresearch/`, `differentiable/` | LOW |
| 11 | Enable PyPI-release `test_package` job | `.github/workflows/cibuildwheel.yml` | MEDIUM |
| 12 | Reconcile dual dev dependency declarations | `pyproject.toml` extras vs dependency-groups | LOW |
| 13 | Clean up ~4k pre-existing mypy errors, then add a mypy job to `ci.yml` (mypy `python_version` bumped to 3.12 in this pass so numpy stubs parse) | `src/openptv2/` | MEDIUM |

## Verification

- `uv run pytest tests/unit/test_correspondences.py tests/unit/test_track.py tests/unit/test_track3d.py -v`
- `uv run pytest -m "not slow"` (full suite)
- Wheel build smoke test: packages present, `openptv2-convert-legacy --help`

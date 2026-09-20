# Tracking-kernels dedup (ponytail-audit items 1-9)

Branch: `refactor/dedup-tracking-kernels` (off `main`). Committed and pushed (0e1200a0, 1b83cfa6); no PR yet.
Goal: remove duplicated / orphaned code in `src/openptv2/algorithms/track_kernels_*.py`
without changing behavior.

## What was found

The same Cython `cdef`/`nogil` kernels were copy-pasted across sibling modules
(cross-module C calls need a `.pxd`, so people copied instead). Bodies were
identical or differed only in decorators/comments. `track_kernels_pixel.pxd` and
`track_kernels_position.pxd` already existed as the sharing mechanism, so I
extended them rather than inventing anything.

## Done

Items 1-5, dedup. Owners:
- `track_kernels_pixel.py`: `_multimed_r_nlay_1layer` (added `exceptval(check=False)` + pxd entry),
  `_point_to_pixel_out`, `_candsearch_in_pix_rest_nogil`, `_pixel_to_metric_out`,
  `_dist_to_flat_out`, `_sorted_candidates_fast_out_nogil`, `candsearch_in_pix_fast_nogil`
  (the pixel copy is the newer one with `max_cands`/`out_dists`; search's was a stale, dead copy).
- `track_kernels_position.py`: `_ray_tracing_out` (added exceptval + pxd), `_angle_acc_out`
  (moved here from geom/corr, added exceptval + pxd), `_point_position_out`,
  `assess_new_position_fast_nogil`.
- Copies deleted from geom, search, transform, corr. Importers use the existing
  `if cython.compiled: cimport ... else: from .x import ...` pattern.
  Import direction is acyclic: pixel <- position <- geom/transform/corr/batch.

Item 6: deleted the `track_kernels_tracking.py` shim. Repointed `track_kernels.py`,
`setup.py` (`ALGORITHMS_MODULES`), `.github/workflows/cibuildwheel.yml` (import smoke test),
and `tests/unit/test_track_kernels_tracking_coverage.py` (now imports from owners; shim
constants defined locally; `_mod` = the pixel module, which keeps the old inert-patch behavior).

Item 9: deleted 16 wrappers with no non-test callers, plus their tests and the two
re-exports in `track_kernels.py`:
`angle_acc_fast`, `_ray_tracing_fast`, `pixel_to_metric_fast`, `dist_to_flat_fast`,
`metric_to_pixel_fast`, `_metric_to_pixel_out`, `_flat_image_coord_fast`, `_img_coord_fast`,
`img_coord_batch_fast`, `flat_image_coord_batch_fast`, `point_position_fast`,
`ray_tracing_batch_fast`, `point_position_batch_fast`, `pixel_to_metric_batch_fast`,
`metric_to_pixel_batch_fast`, `sort_candidates_by_freq_fast`.
Also dropped two search test classes that only tested the deleted dead copies.

Net so far: roughly -2,200 lines from dedup, about -840 from item 9, plus tests.

## Verification status (updated 2026-09-21)

Clean Cython rebuild of the final tree: OK.
- Hot-path tests (`test_track`, `test_track3d`, `test_correspondences`, `test_track4be`): 49 passed = baseline.
- Full suite `uv run --no-sync pytest tests`: 2017 passed, 86 skipped, 39 deselected (12 min).
- Pure-Python fallback, kernel coverage files only
  (`test_track_kernels_*_coverage.py`, 6 files): 266 passed.
- Pure-Python fallback over the whole `tests/unit/test_*_coverage.py` glob: 1227 passed, 24 failed
  (38 min; the glob now matches 29 files, not the 16 CLAUDE.md mentions, so it is slow).
  The failures I inspected (`test_epi_coverage`, `test_correspondences_coverage`, 14 of the 24)
  are all `Coord2d.__init__() got an unexpected keyword argument 'pnr'` /
  `Candidate.__init__() ... 'pnr'`: interpreted-mode constructor mismatch in modules this
  branch does not touch. Not confirmed on `main`; the other 10 were not inspected.
- Running the suite rewrites tracked `test_data/test_cavity/img/*_targets`; `git checkout -- test_data` before committing.

## Remaining

1. Optional: confirm the 24 fallback failures also occur on `main` (build `main`, run
   `tests/unit/test_epi_coverage.py tests/unit/test_correspondences_coverage.py` interpreted).
2. Optional perf sanity: `_angle_acc_out` was `ccall inline` inside corr and is now a cross-module
   C call (lost inlining). Time a tracking run before/after; if it regressed, keep a private copy in corr.
3. Cosmetic: stray banner comments in `test_track_kernels_batch_coverage.py` (about lines 84-102) and
   `test_track_kernels_transform_coverage.py` (lines 22, 352).
4. Open a PR.

## Deliberately skipped

- **Item 7** (merge forward/backward tracking loops in `track_kernels_corr.py`): not a clean
  dedup. Normalised diff of `trackcorr_loop_fast` vs `trackback_loop_fast` shows about 670 of about 900
  lines differ; forward uses the `_trackcorr_particle_fast` worker, backward is inline. Merging
  changes core tracking logic and there is no golden-output regression data. Only attempt
  with a recorded before/after trajectory comparison on a real dataset.
- **Item 8** (delete 4BE tracker: `track4be_loop_fast`, `track4be.py`, `plugins/four_be_tracking.py`):
  it is registered in `tracking_registry.py` and benchmarked in about 10 scripts/notebooks
  (`bench_*`, `benchmark_*`, `tracker_tutorial_dashboard.py`). Needs an explicit product decision.
  Do not delete without Alex saying so.

## Gotchas

- Cimported names are not importable from Python: tests and non-cimporting modules must import
  the `cpdef` names from the owner module (pixel/position), not from geom/transform.
- Cython pure mode + `.pxd`: `noexcept nogil` in the pxd requires `@cython.exceptval(check=False)`
  on the `def`, or the signatures mismatch.
- Shell cwd drift: `cd` to the repo root at the start of every command.
- Use `git grep`, not `grep -r`, at the repo root (huge build dirs; a plain grep timed out).

# Tracking-kernels dedup (ponytail-audit items 1-9)

Branch: `refactor/dedup-tracking-kernels` (off `main`). **Nothing is committed yet.**
Goal: remove duplicated / orphaned code in `src/openptv2/algorithms/track_kernels_*.py`
without changing behavior.

## What was found

The same Cython `cdef`/`nogil` kernels were copy-pasted across sibling modules
(cross-module C calls need a `.pxd`, so people copied instead). Bodies were
identical or differed only in decorators/comments. `track_kernels_pixel.pxd` and
`track_kernels_position.pxd` already existed as the sharing mechanism, so I
extended them rather than inventing anything.

## Done (uncommitted, in the working tree)

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

## Verification status

- After items 1-6 (before item 9): clean Cython rebuild OK; hot-path tests
  (`test_track`, `test_track3d`, `test_correspondences`, `test_track4be`) = 49 passed, same as baseline;
  781 related unit tests passed, one failure (`test_trackcorr_stub_zero_no_prev`) which was
  fixed (patch target must be the pixel module).
- **Not yet verified:** the item 9 deletions (sources edited after the last build), and the
  complete `uv run pytest tests` result (a full run was started in the background and
  never reported; treat it as unknown, rerun it).
- Baseline before any change: 49 passed, 3 deselected on the hot-path set.

## Next steps (in order)

1. `git status` to confirm the tree matches the above. Remove stale `.c` and `cp313` `.pyd` for
   `track_kernels*` and `rm -rf build`, then
   `uv run --no-sync python setup.py build_ext --inplace` (about 3 min).
   Use `--no-sync`: plain `uv run` reinstalls the package every time (about 1.5 min).
2. Hot-path tests: `uv run --no-sync pytest tests/unit/test_track.py tests/unit/test_track3d.py tests/unit/test_correspondences.py tests/unit/test_track4be.py -q` (expect 49 passed).
3. Full suite: `uv run --no-sync pytest tests -q` (takes over 10 min; run in background).
4. Pure-Python fallback check per `CLAUDE.md` (move `*.cp313-win_amd64.pyd` aside, run
   `tests/unit/test_*_coverage.py -m ''`, restore, rebuild). Expect all pass. The 16 coverage
   files are the only tests that exercise the interpreted path, and I edited 6 of them.
5. `uv run ruff check .` (pre-existing E701 in `track_kernels_track3d.py` and F842 in corr are not from this work).
6. Optional quick perf sanity: `_angle_acc_out` was `ccall inline` inside corr and is now a
   cross-module C call, so it lost inlining. Compare a tracking run before/after
   (`tests/perf/`, or time `test_track.py`). If it regressed, keep a private copy in corr.
7. Commit in two steps (dedup + shim; then item 9), end messages with the
   Co-Authored-By line from the session attribution. Then a PR.
8. Tidy leftovers (cosmetic): stray banner comments in `test_track_kernels_batch_coverage.py`
   (lines about 84-102) and `test_track_kernels_transform_coverage.py` (lines 22, 352).

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

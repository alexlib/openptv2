# Zarr-only transition plan: retire `*_targets`, `rt_is.*`, `ptv_is.*`, `added.*`

**Status:** planned, not started. Intended for implementation in a follow-up
session (possibly by a different/cheaper model — this doc is written to be
self-contained for that handoff).

## Scope

Stop writing and reading the four legacy per-frame ASCII file families:
`img/cam<N>.<frame>_targets`, `res/rt_is.<frame>`, `res/ptv_is.<frame>`,
`res/added.<frame>`. Everything else stays as-is: YAML parameter files,
`.ori`/`.addpar` calibration files, CSV/VTK/NetCDF exports, and any other
ASCII output are unaffected — this is specifically about the legacy PTV
run-data quartet that `openptv2.storage.RunStore` (Phases A–D,
`docs/plans/2026-08-14-storage-formats-as-built.md`) and `flowtracks.ZarrScene`
(Phase E, `postptv` repo) were built to replace.

## Where things stand (Phases A–E, all committed)

- `openptv2`: every real write path (targets, correspondences, tracking
  linkage, `tracking_postprocess.py`) threads an explicit
  `store: RunStore | None` parameter and does an **unconditional dual-write**
  when a store is given — both the store *and* the ASCII file are written,
  always. `store=None` still means pure ASCII, unchanged, for every caller
  that has never been wired to a store.
- GUI trajectory/position display reads through `RunStore` (Phase D).
- `postptv`: `flowtracks.zarr_scene.ZarrScene` reads `RunStore`'s sealed
  output directly, duck-type compatible with the legacy `Scene`.

## The five blockers, and what to do about each

### 1. Calibration/dumbbell/autocalibration never got wired to a store (openptv2)

These are genuinely ASCII-only today, always with `store=None`:

- `src/openptv2/autocalibration.py:385,394-407,914,1053` — `read_targets`/
  `write_targets` calls, all through `target_base(base, cam)` (calibration
  target images, one-off per camera, not part of a tracked run).
- `src/openptv2/gui/dumbbell_ground_truth.py:100` — `ptv.write_targets(...)`.
- `src/openptv2/gui/ptv_calibration.py:498` — `read_targets(...)`.

**Decision needed before implementing:** calibration data has no
trajectory/linkage concept — a calibration run doesn't produce `rt_is`/
`ptv_is`/`added` at all, only `_targets`. Two honest options:
  - **(a) Wire `_targets` writes here through `RunStore.write_targets`
    too**, using a store scoped to the calibration directory (not a
    tracking "run" store, but the same class — `RunStore` doesn't require
    `linkage`/`correspondences` to ever be populated). This makes the cutover
    total: no ASCII target files anywhere.
  - **(b) Leave calibration as a permanent ASCII exception.** Calibration
    target files are typically small, one-shot, human-inspected during setup
    (see `openptv-calibrate` skill's overlay-image workflow, which expects
    `_targets` files to exist on disk next to the calibration images). Cutting
    them over saves nothing operationally and risks breaking that skill's
    assumptions.

  **Recommendation: (b).** The whole point of the unified store was runs
  (thousands of frames, cloud-parallel workers) — calibration is a handful of
  frames, inspected by a human, already fast and simple as flat files. Confirm
  this with the user before implementing; if they want (a), thread a store
  the same way Phase B did (`_open_run_store`-equivalent scoped to the
  calibration dir), and update the `openptv-calibrate` skill's file-existence
  assumptions to check the store too.

### 2. `pyptv_batch.py`'s advisory conditioning check (openptv2)

`src/openptv2/batch/pyptv_batch.py:159-160`
(`_warn_if_tracking_poorly_conditioned`) reads `res/rt_is.<frame>` via raw
`np.loadtxt`, wrapped in a broad `try/except: pass` (it's explicitly
best-effort, "never blocks or alters tracking"). Once `rt_is.*` stops being
written, this silently stops firing — no crash, just a lost diagnostic.

**Fix:** thread the run's `RunStore` into `run_batch`/`_warn_if_...` (the
store is already opened once per run inside `py_sequence_loop` etc. via
`_open_run_store`; `pyptv_batch.py` needs the same call) and read via
`store.read_correspondences(frame)` instead of `np.loadtxt`. Small, isolated
change — do this in the same pass as item 4.

### 3. ~39 test files assert on ASCII content directly (openptv2)

Current count (re-verified, not the earlier rough 65 — that grep pattern was
broader):

```
batch/test_batch_small.py                 batch/test_burgers_synthetic.py
batch/test_pyptv_batch.py                  batch/test_pyptv_batch_plugins.py
batch/test_rembg_small.py                  batch/test_sequence_singleframe_parity.py
batch/test_splitter_end_to_end.py          batch/test_synthetic_tracker.py
batch/test_track_res_vs_res_orig.py        batch/test_tracking_presets_benchmark.py
batch/test_yaml_only_folder.py             gui/test_cavity_comprehensive.py
gui/test_export_ptv_is_to_paraview.py      gui/test_plot_3d_positions.py
gui/test_plot_3d_trajectories.py           gui/test_ptv_coverage_summary.py
gui/test_ptv_file_io.py                    gui/test_ptv_utilities.py
gui/test_pyptv_gui.py                      gui/test_tracker_minimal.py
gui/test_tracking_parameters.py            unit/test_batch_python_pipeline.py
unit/test_correspondences.py               unit/test_full_tracking_diagnostic.py
unit/test_parallel_preprocessing.py        unit/test_parallel_tracking.py
unit/test_run_store.py                     unit/test_suggest_eps0.py
unit/test_synthetic_tracking.py            unit/test_tracer_selfcal.py
unit/test_track3d.py                       unit/test_track3d_coverage.py
unit/test_track_kernels_tracking_coverage.py  unit/test_tracker_run_store.py
unit/test_tracking_frame_buf.py            unit/test_tracking_frame_buf_coverage.py
unit/test_tracking_synthetic.py            unit/test_tracking_synthetic_dense.py
unit/test_yaml_only_runtime.py
```

Three sub-categories, each needs a different fix:

- **`test_run_store.py`, `test_tracker_run_store.py`** — these are Phase A–C's
  *own* round-trip/back-compat tests (`import_run`/`export_run`, the byte-diff
  gate). **Keep as-is.** They exist specifically to prove ASCII import/export
  still works on demand — that's the whole point of "on demand, not during a
  run." Do not touch these.
- **`unit/test_tracking_frame_buf.py`, `test_tracking_frame_buf_coverage.py`,
  `test_track3d*.py`, `test_tracking_synthetic*.py`, `test_batch_python_pipeline.py`,
  `test_correspondences.py`, `test_parallel_{tracking,preprocessing}.py`,
  `test_full_tracking_diagnostic.py`, `test_suggest_eps0.py`,
  `test_tracer_selfcal.py`, `test_track_kernels_tracking_coverage.py`,
  `test_yaml_only_runtime.py`** — unit-level tests that call
  `write_targets`/`write_path_frame`/`read_targets`/`read_path_frame`
  directly with `store=None` to set up or assert on ASCII fixtures. These
  call the low-level functions positionally/by name already — **most don't
  need to change at all**, since `store=None` will remain valid (pure-ASCII
  mode never goes away, it's just no longer the *default* for real runs). Spot
  check each file: if it asserts on `Path(...).exists()` for an
  `_targets`/`rt_is`/`ptv_is`/`added` file *as the only way of checking the
  operation succeeded*, decide whether to also assert via a `RunStore` (better
  coverage) or leave it (still valid, since `store=None` is preserved).
- **`batch/*.py`, `gui/test_cavity_comprehensive.py`, `gui/test_ptv_utilities.py`,
  `gui/test_pyptv_gui.py`, `gui/test_tracker_minimal.py`,
  `gui/test_tracking_parameters.py`, `gui/test_plot_3d_positions.py`,
  `gui/test_plot_3d_trajectories.py`, `gui/test_export_ptv_is_to_paraview.py`,
  `gui/test_ptv_coverage_summary.py`, `gui/test_ptv_file_io.py`** — these run
  the *real* pipeline (`pyptv_batch.main`, `py_sequence_loop`, `Tracker`) and
  then assert on `res/rt_is.*`/`res/ptv_is.*` file contents as their
  verification method. **These must be rewritten** to read through `RunStore`
  instead, once step 4 below stops those files from being written for a
  store-backed run. This is the bulk of the real work in this plan.
  `test_ptv_file_io.py` specifically (34 tests) is the one exception requiring
  care: it directly tests `gui.ptv.write_targets`/`read_targets`'s own ASCII
  implementation (kept deliberately separate from the canonical one in Phase
  B, see that file's docstring) — decide whether this function keeps writing
  ASCII forever (it's a thin, still-useful legacy-import utility) or also
  gets a `write_ascii=False` mode; recommend leaving it alone, it's 34
  well-isolated tests with no run-time cost.

### 4. GUI ASCII fallback paths (openptv2, already reduced in Phase D)

`gui/plot_3d_trajectories.py:218-231` and `gui/flowtracks_utils.py:31-36,98-99`
still fall back to `flowtracks.io.trajectories_ptvis(...)` reading
`res/ptv_is.%d` when `res/run.zarr` doesn't exist. **Keep these** — they are
exactly the safety net for runs that predate this transition (imported via
`import_run`, or produced by an older openptv2 version). Once nothing writes
`ptv_is.*` by default, this fallback simply never fires for new runs; it's not
dead code, it's backward compatibility. No change needed here beyond what
Phase D already did.

### 5. ~~`matlab_to_python_3dptv`~~ — out of scope

That repo is obsolete (per the user, 2026-08-15) and is not a target for this
transition. `flowtracks.pipeline.ptv_is_to_lagrangian`
(`postptv/flowtracks/pipeline.py:14-25`) still reads `ptv_is.%d` internally,
but nothing in this plan depends on fixing it — it's dead code from
`openptv2`'s perspective once nothing else calls it. Leave it alone unless a
live consumer of it turns up later.

## Implementation order

1. **Confirm the calibration decision (blocker 1)** with the user before
   touching any code — (a) wire it, or (b) leave it, per the recommendation
   above.
2. **`pyptv_batch.py` advisory check (blocker 2)** — small, isolated, do
   first as a warm-up; verify with a quick manual run against `test_cavity`.
3. **Flip the write-side switch (openptv2):** in
   `algorithms/tracking_frame_buf.py`, change `write_targets`/
   `write_path_frame`/`write_rt_is`'s behavior when `store is not None` from
   "write ASCII and store" to "write store only" — this is the actual cutover.
   `store=None` keeps writing pure ASCII (used by `export_run`, calibration if
   (b) was chosen, and any caller that intentionally wants files on disk).
   Rebuild the Cython extension after this change
   (`uv run python setup.py build_ext --inplace`).
4. **Migrate the ~12 batch/GUI tests (blocker 3, third sub-category)** to
   assert via `RunStore` instead of reading `res/*` files — this is where
   most of the effort goes. Do this *before* step 3 lands on `main` if
   possible (or in the same PR), since they'll start failing the moment
   ASCII stops being written.
5. **Full regression pass**: `uv run pytest` in `openptv2` (expect the same 5
   pre-existing failures as Phases A–E, 0 new), then a manual `pyptv_batch`
   run against `test_cavity` with `git diff` confirming no
   `_targets`/`rt_is.*`/`ptv_is.*`/`added.*` files appear under `res/`/`img/`
   for a fresh run — that's the actual acceptance criterion for "no ASCII
   files."

## What does NOT change

- `import_run`/`export_run` (Phase A) stay exactly as they are — they're the
  permanent, on-demand compatibility boundary for anyone who still needs
  legacy files (external tools, one-off debugging, the calibration skill if
  option (b) is chosen).
- `RunStore`'s per-frame-group layout, `seal()`, `to_flowtracks_trajectories()`
  — unchanged, already built.
- Everything outside the four file families (YAML params, `.ori`/`.addpar`,
  CSV/VTK/NetCDF exports) — untouched, was never in scope.

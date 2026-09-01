# Plan: Final ASCII → Zarr Cutover — no more `_targets` / `rt_is` / `ptv_is` / `added` files at runtime

Date: 2026-09-01
Status: draft — for review before implementation
Goal: **every runtime write of `_targets`, `rt_is.*`, `ptv_is.*`, `added.*` goes to `res/run.zarr` (`RunStore`); every runtime read (pipeline + visualization) comes from the store.** ASCII survives only as an on-demand import/export boundary (`storage/legacy.py:import_run` / `export_run`) and — if the review below chooses option B — as calibration's tiny one-shot `cal/camN.tif_targets`.

This is the follow-through of `docs/plans/archive/2026-08-15-zarr-only-transition-plan.md` and `docs/plans/archive/2026-08-26-zarr-only-v0.5.6-plan.md`, which made store the DB of record but left dual-write (store + ASCII) and dual-read (`store.has_*` → store else ASCII) in place and left calibration explicitly as an exception. This plan removes the remaining ASCII legs.

## 1. Context & what "all" means

Four legacy families, all per-frame, all replaced by `docs/zarr-hdf5-storage.md:15` layout:

| Legacy | Store path | Canonical writer | Canonical reader |
|---|---|---|---|
| `img/cam<N>.<frame>_targets` | `targets/cam_<c>/frame_<n>` | `storage/run_store.py:282` `RunStore.write_targets` | `storage/run_store.py:298/312` `read/has_targets` |
| `res/rt_is.<frame>` | `correspondences/frame_<n>` | `storage/run_store.py:321` `write_correspondences` | `storage/run_store.py:343/359` `read/has_correspondences` |
| `res/ptv_is.<frame>` | `linkage/ptv_is/frame_<n>` | `storage/run_store.py:364` `write_linkage(name=ptv_is)` | `storage/run_store.py:405/420` |
| `res/added.<frame>` | `linkage/added/frame_<n>` | same | same |

Non-goals (keep forever): YAML params, `.ori`/`.addpar` calibrations, CSV/VTK/NetCDF exports, `storage/legacy.py:92` `import_run` / `:232` `export_run` (the permanent compatibility boundary — see `docs/plans/archive/2026-08-15-zarr-only-transition-plan.md:195`).

## 2. Current state (audited 2026-09-01, two parallel inventories)

### 2.1 Already store-native (no change needed, keep as reference)

* `gui/ptv.py:1041` `_open_run_store(exp)` — single opener threaded everywhere.
* `gui/ptv.py:1082-1131` `py_sequence_loop` — `read_targets(...,store=store)`, `write_targets(...,store=store)`, `store.write_correspondences` ✅
* `gui/ptv.py:1346-1501` `py_sequence_loop_python` — same via `_write_targets_canonical(...,store=store)` ✅
* `algorithms/tracking_frame_buf.py:177` `read_targets(file_base,frame,cam_idx,store)` / `:215` `write_targets(...,store)` / `:369` `read_path_frame(...,store)` / `:579` `write_path_frame(...,store)` / `:869` `Frame.read` / `:960` `Frame.write` / `:1043` `FrameBuf.read_frame_at_end` — all dual with `store.has_*` gate ✅
* `gui/ptv.py:1601` `write_targets` / `:1657` `read_targets` — dual ✅ (kept separate from canonical for `tests/gui/test_ptv_file_io.py` mock contract, `gui/ptv.py:1609`)
* `gui/ptv.py:190-251` `_process_frame_worker` — `store.write_targets` when `store` else ASCII (good when `zarr_store_path` is passed)
* `plugins/contour_sequence.py:189/216/243`, `plugins/rembg_sequence.py:73/100/127`, `plugins/rembg_contour_sequence.py:125/154/181` — all `store=` ✅
* `plugins/default_tracking.py`, `myptv_3d_tracking.py:257`, `myptv_2d_tracking.py:243`, `hybrid_deltat_3d.py:279`, `proptv_tracking.py:370`, `two_phase_tracking.py:306/359` — `Frame.read/write(...,store=store)` ✅
* `tracker.py:53` `_estimate_max_targets` — checks `store` first, ASCII fallback (intentional backward compat)
* `storage/legacy.py:58` `_load_rt_is` / `:194` `_write_targets_ascii` — intentionally ASCII (import/export)
* `benchmarking/datawriter.py:209` `write_store()` vs `:89` ASCII fixture writer — keep both (golden-file compare)
* `batch/pyptv_batch.py:146` `_warn_if_tracking_poorly_conditioned` — **already fixed** to `store.read_correspondences` (`batch/pyptv_batch.py:163`), no longer ASCII.

### 2.2 Stragglers — the actual cutover

**Writers that still emit ASCII at runtime (the "last piece" from the user report):**

| # | File:line | Symbol | Why it still writes ASCII |
|---|---|---|---|
| W1 | `gui/ptv.py:704-733` `py_correspondences_proc_c(exp)` | `write_targets(exp.detections[i], short_base, frame)` | **No `store=`** — GUI "Correspondences" button `gui/pyptv_gui.py:808`. Only call site in pipeline that never got a store. |
| W2 | `gui/ptv.py:1328` `py_sequence_loop_python` parallel branch | `preprocess_and_detect_all_parallel(exp)` | Missing `zarr_store_path=str(store.store_path)` — `py_sequence_loop:1044` passes it, python-engine variant does not, so `_process_frame_worker:251` falls to ASCII. |
| W3 | `autocalibration.py:479` `calibrate_camera` | `write_targets(detected,len, target_base(base,cam), 0)` | Calibration one-shot `cal/camN.tif_targets` (`frame 0`). Intentional per old plan `2026-08-15:42` option (b) — see §3 decision. |
| W4 | `skills/openptv-calibrate/scripts/detect_targets.py:84` | `write_targets(list(targs),n,str(tbase),0)` | Same calibration family — CLI helper. |
| W5 | `gui/dumbbell_ground_truth.py:106` | `ptv.write_targets(targs,str(short_base),frame)` + `:108` `store.write_targets` | Dual today (mirrors). Should become store-only or be deleted (see W3). |
| W6 | `gui/ptv.py:966` `_write_rt_is_file` / `algorithms/tracking_frame_buf.py:567` `write_rt_is` | `open(path,"w")` | Pure ASCII helper, called only when `store is None` (`gui/ptv.py:781` `py_determination_proc_c` non-store path). Keep for `store=None` export path, but pipeline must never hit it with a store. |
| W7 | `gui/ptv.py:248-251` `_process_frame_worker` else-branch | `write_targets(targs, short_base, frame)` | Covered by W2 — dead when caller passes store. |

**Readers that still prefer or fall back to ASCII at runtime (visualization + pipeline):**

| # | File:line | Symbol | Current behavior |
|---|---|---|---|
| R1 | `gui/ptv.py:1845` `read_rt_is_file(filename)` + `:1807` `_read_correspondences_from_zarr_fallback` | Reads ASCII `open(filename)`; zarr only if `not Path.exists` + `_read_correspondences_from_zarr_fallback` via `ZarrFrameStore` | Visualization fallback inverted — ASCII-first. Should be **zarr-first, ASCII only via `import_run`**. |
| R2 | `gui/plot_3d_positions.py:377` `_read_positions(rt_is_path,frame)` | `if not exists` → `RunStore.has_correspondences` else `ptv.read_rt_is_file` (ASCII) (`:402`) | Dual, ASCII-first. Same inversion. |
| R3 | `gui/plot_3d_positions.py:423` `read_positions_sequence` | Loops `_read_positions` per frame | Inherits R2. Caps at `max_frames=50`. |
| R4 | `gui/visualize_rt_is_nb.py:154` `plot_3d_target` | `if not exists && frame` → `RunStore` (`:178`) else `np.loadtxt(skiprows=1)` (`:199`) | Same. |
| R5 | `gui/pyptv_gui.py:938` `visualize_3d_positions` | Opens TraitsUI, delegates to `_read_positions` | Inherits R2. |
| R6 | `gui/pyptv_gui.py:1013/1039` `detect_part_track` | Already `ptv.read_targets(...,store)` ✅ — reference good pattern for others. |
| R7 | `algorithms/tracking_frame_buf.py:402` `read_path_frame` store gate | `if store.has_correspondences(frame)` → store else ASCII `open(rt_is)` | Correct gate; after cutover the `else` is only for `store=None` / pre-zarr imports. No change, just stop writing the else's files. |
| R8 | `autocalibration.py:458` `pix=read_targets(str(target_base),0)` / `:1028` `suggest_eps0` / `gui/ptv_calibration.py:498` / `skills/.../calib.py:872` | Always ASCII calibration reads | Tied to W3 decision. |
| R9 | `gui/plot_3d_trajectories.py:218` / `gui/flowtracks_utils.py:31` / `tracker.py:53` | Fallback to `flowtracks.io.trajectories_ptvis` / `open(rt_is)` when no store | Intentional backward compat for pre-zarr runs (`2026-08-15:148`). Keep, but must not fire for new runs (no files to find). |

### 2.3 Env / flag

`ParameterManager` `pft_version.Existing_Target` (`gui/legacy_parameters.py:793`, `gui/parameter_manager.py:101`, `gui/ptv.py:1020/1306`) — sole gate for "read targets vs detect". No env var (`OPENPTV_STORAGE` already removed per `docs/zarr-hdf5-storage.md:107`). Keep.

## 3. Decision required before code (1 day, blocks W3/W4/R8)

Old plan recommended **(b) leave calibration as permanent ASCII exception** (`2026-08-15:42`) — calibration is 1 frame per cam, human-inspected, overlay workflow expects files. User's new ask is "all ASCII → zarr". Choose:

* **Option A (user literal):** calibration `_targets` also go to `res/run.zarr` (or a calibration-scoped store `cal/run.zarr`). `detect_targets.py`, `autocalibration.py`, `ptv_calibration.py`, `calib.py` all read via `store.read_targets(cam,0)` with ASCII fallback hidden behind `import_run`. Benefit: total cutover, single code path. Cost: `openptv-calibrate` skill overlay-image assumptions (`cal/camN.tif_targets` existence) must be updated to check store too; human inspection now via `ZarrFrameStore.dump_frame_text` not `cat`.
* **Option B (recommended, keeps W3/W4/R8 as documented exception):** keep `cal/cam*.tif_targets` ASCII. Rationale: calibration is not a run (no `rt_is`/`ptv_is`), store buys nothing operationally, and the skill's `has_targets` check (`skills/openptv-calibrate/SKILL.md:116`) is file-existence based. If B, W3/W4/R8 are **not** in scope and the "all" in the title means "all run-data families".

**Plan below assumes B (no calibration change) unless the review picks A.** If A is chosen, add W3/W4/R8 to Phase 1/2 with same `store=` threading and update skill docs; no other phase changes.

## 4. Design

* Single store per experiment: `res/run.zarr` resolved by `storage/run_store.py:66` `resolve_store_path` / `:82` `find_existing_store`, opened once via `gui/ptv.py:161` `_open_run_store(exp)` ("explicit store threading" from `gui/ptv.py:174` comment).
* Pipeline never writes ASCII when a store is present. `store=None` stays valid **only** for `storage/legacy.py:export_run`, tests that explicitly pass `store=None`, and calibration if B. Not for `py_sequence_loop` etc.
* Visualization becomes zarr-first: `store.has_correspondences(frame)` → `store.read_correspondences` → `(N,3)` array; ASCII only when `store is None` or `not has_correspondences` **and** caller explicitly asked for a legacy file (pre-zarr run). No `Path.exists` → zarr fallback; go `zarr` directly.
* Import boundary: old ASCII runs remain readable via `import_run` ( `storage/legacy.py:92` `import_run` does `store.write_targets/write_correspondences` from ASCII). No pipeline fallback needed beyond that.
* No `OPENPTV_STORAGE` env, no new flag.

## 5. Implementation phases

### Phase 0 — Confirm decision & baseline (0.5 day)

* Review picks A vs B for calibration.
* Baseline pass: `uv run pytest -q` (record 5 pre-existing failures baseline), manual `uv run openptv2-batch test_data/test_cavity_small --mode both` and `ls res/*.targets res/rt_is.* res/ptv_is.*` count (should be >0 today, 0 after).

### Phase 1 — Kill the remaining runtime ASCII writers (1 day)

* **W1** `gui/ptv.py:704` `py_correspondences_proc_c`: add `store = _open_run_store(exp)` at top, change loop to `write_targets(exp.detections[i], short_base, frame, store=store, cam_idx=i)` — mirrors `gui/ptv.py:1104`. Keep ASCII helper `gui/ptv.py:1601` body (`if store is not None: store.write_targets else: np.savetxt`) unchanged; it already branches.
* **W2** `gui/ptv.py:1328` `py_sequence_loop_python`: change `preprocess_and_detect_all_parallel(exp)` → `preprocess_and_detect_all_parallel(exp, zarr_store_path=str(store.store_path))` and ensure the early `store = _open_run_store(exp)` (already at `:1302`) is before the call. Then `_process_frame_worker:248` always takes `store.write_targets` branch.
* **(If A)** W3/W4/W5: thread a calibration-scoped `RunStore(cal_dir / "run.zarr")` through `autocalibration.py:479`, `detect_targets.py:84`, `dumbbell_ground_truth.py:106` (change to single `ptv.write_targets(...,store=store,cam_idx=cam)` and delete the extra `store.write_targets` mirroring).
* Rebuild Cython: `uv run python setup.py build_ext --inplace` if `tracking_frame_buf.py` changed (W1/W2 don't touch it, but do if hardening `write_targets` to assert `store is not None` in pipeline).
* Verify: `grep -rn "write_targets.*short_file" src/openptv2/gui/ptv.py` should show no bare `write_targets(..., frame)` without `store=`.

### Phase 2 — Make visualization zarr-first (0.5 day)

* **R2** `gui/plot_3d_positions.py:377` `_read_positions`: invert to `store = find_existing_store(...); if store and store.has_correspondences(frame): return store.read_correspondences(frame)[:,:3]` first; only then try ASCII `ptv.read_rt_is_file` / `np.loadtxt`. Today it's `if not exists → store` else ASCII — flip. Keep `ValueError` → `(0,3)` behavior (`:404`) for empty frames.
* **R1** `gui/ptv.py:1845` `read_rt_is_file`: add optional `store` param or make it `find_existing_store` first (consistent with R2). Keep `ZarrFrameStore` fallback as is for old path, but make it primary. Same for `:1807` `_read_correspondences_from_zarr_fallback` — keep as helper but call before `open`.
* **R4** `gui/visualize_rt_is_nb.py:154` — same inversion (store first, `np.loadtxt` only if `not has_correspondences`).
* **R3/R5** inherit R2 — no direct change.
* Keep `gui/plot_3d_trajectories.py:218` / `flowtracks_utils.py:31` backward-compat fallback — it's already `if store exists else ASCII`; just ensure it doesn't synthesize files (it doesn't).
* Small new helper (optional, not required): `storage/run_store.py:343` `read_correspondences` already returns `(pos_3d, cam_ids)` — visualization needs only `pos_3d`; keep as is.

### Phase 3 — Pipeline reads already correct, just delete dead fallbacks (0.5 day)

* `algorithms/tracking_frame_buf.py:402` `read_path_frame` and `:177` `read_targets` already `has_*` gated — no change. The point is that after Phase 1, no new run will leave ASCII files for the `else` to find, so the else becomes legacy-only. Leave it (it's the `store=None` and `import_run` path) — do not delete.
* Audit `ParameterManager` `Existing_Target` semantics: when `Existing_Target=1` and store has no targets for that frame, today `read_targets(...,store)` returns `[]` (`tracking_frame_buf.py:211`) after `has_targets==False` → file miss → `[]`. That's fine — caller handles empty. No change.
* Remove any stray `Path("res/rt_is.%d" % frame).exists()` probes that assume files (none remaining after W1/W2; `tracker.py:53` already store-first).

### Phase 4 — Tests: migrate the ~12 batch/GUI tests that assert on ASCII files (1.5 days) — bulk of work

Per `2026-08-15:133` third sub-category (the only one that must move):

* Files to migrate (read via `RunStore` instead of `open`/`np.loadtxt`):
  `batch/test_batch_small.py`, `batch/test_burgers_synthetic.py`, `batch/test_pyptv_batch.py`, `batch/test_pyptv_batch_plugins.py`, `batch/test_rembg_small.py`, `batch/test_sequence_singleframe_parity.py`, `batch/test_splitter_end_to_end.py`, `batch/test_synthetic_tracker.py`, `batch/test_track_res_vs_res_orig.py`, `batch/test_tracking_presets_benchmark.py`, `batch/test_yaml_only_folder.py`, `gui/test_cavity_comprehensive.py`, `gui/test_plot_3d_positions.py`, `gui/test_plot_3d_trajectories.py`, `gui/test_export_ptv_is_to_paraview.py`, `gui/test_pyptv_gui.py`, `gui/test_tracker_minimal.py`, `gui/test_tracking_parameters.py`, `unit/test_batch_python_pipeline.py` (pipeline-backed ones).
* Keep as-is (not in scope): `unit/test_run_store.py`, `unit/test_tracker_run_store.py` (round-trip gates), `unit/test_tracking_frame_buf.py`, `test_tracking_frame_buf_coverage.py`, `test_track3d*.py`, etc. — they use `store=None` intentionally. Same for `tests/gui/test_ptv_file_io.py:34` — leave, it's testing the ASCII utility itself (`gui/ptv.py:1609` docstring says keep separate).
* Pattern to replace:
  ```py
  # before
  assert Path("res/rt_is.10000").exists()
  data = np.loadtxt("res/rt_is.10000", skiprows=1)
  # after
  from openptv2.storage import RunStore

  store = RunStore("res/run.zarr", mode="r")
  pos, cam_ids = store.read_correspondences(10000)
  assert pos.shape[0] > 0
  ```
  Same for `targets`: `store.read_targets(cam, frame)` vs `read_targets("cam1.", frame)`.
* Do this **before** landing Phase 1 on `main` (or same PR) so CI doesn't go red the moment ASCII stops being written (`2026-08-15:184`).

### Phase 5 — Hardening & cleanup (0.5 day)

* `src/openptv2/benchmarking/datawriter.py` — keep `write_dataset` ASCII + `write_store()` store variant (golden-file compare). No change.
* `tracking_postprocess.py:70`, `tracking_chunked.py:311` — already store-branched; no change.
* `.gitignore:54` `*_targets` / `:116` `**/**/*.*_targets` — keep ignoring runtime `_targets`, but add `!test_data/**/cal/*_targets` already there and (if B) keep ignoring; no change unless A then add `!res/run.zarr/**` already ignored correctly.
* Docs: update `docs/zarr-hdf5-storage.md:3` from "are not written" (today's claim is premature — W1/W2 still write) to accurate post-cutover wording; add "Visualization reads from store" note under `docs/zarr-hdf5-storage.md:46` / `gui/plot_3d_positions.py:377` docstring.

### Phase 6 — Verification (0.5 day)

* `uv run pytest -m ci -q` (`2026-08-26:59` `ci` marker: `test_run_store.py`, `test_store_parallel_io.py`, `test_parallel_tracking.py`, `test_trackcorr_store_only.py`, etc.) — target `<5 min`, 0 new failures.
* Full `uv run pytest -q` — expect same 5 pre-existing baseline, 0 new.
* Manual: fresh `test_cavity` / `test_cavity_small` batch `both` → `find . -name "*_targets" -o -name "rt_is.*"` under `res/` / `img/` must be 0 files; `res/run.zarr` must hold frames via `RunStore.has_targets/has_correspondences`; visualization `Visualize 3D positions` must show same point cloud as before (store path).
* Legacy import check: `python scripts/convert_fixtures_to_zarr.py` then `RunStore.export_frame_text(frame)` vs original ASCII bytes — already covered by `test_convert_legacy_to_zarr.py:3.1`.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GUI calibration skill (`openptv-calibrate`) expects `_targets` files next to images (`SKILL.md:116` `has_targets`) | If A: update skill to check `store.has_targets` too; if B: no risk (keep files). |
| Tests that assert `Path(...).exists()` start failing the moment W1/W2 land | Land Phase 4 in same PR / before W1/W2, or gate CI on `store` variant immediately. |
| `gui/ptv.py:1601` `write_targets` mock contract (`tests/gui/test_ptv_file_io.py` patches `np.savetxt`) breaks if unified with canonical | Keep it separate (docstring `1609` says so) — don't unify. |
| `store=None` callers that intentionally want ASCII (export, fixtures) lose ability if we delete branches | Never delete `store is None` branches; they are the permanent `export_run` path. Only stop calling them from pipeline. |
| Parallel worker file-lock vs zarr concurrency | Already chunked per `frame_*` key (`docs/zarr-hdf5-storage.md:11`), no lock; `3.4/3.5` parallel I/O tests cover. |

## 7. Alternative considered

**Delete all fallback ASCII branches (`else: open(...)`) entirely.** Rejected: `store=None` is the permanent export/legacy-compat path and is used by `import_run`/`export_run` and unit tests that deliberately pass `store=None`. Removing it would break the compatibility boundary for no runtime gain — pipeline simply stops calling it (Phase 1), but the function keeps the branch for explicit callers.

## 8. Acceptance criteria

* Fresh `pyptv_batch` run creates **0** `_targets` / `rt_is.*` / `ptv_is.*` / `added.*` files under `res/` / `img/` (calibration excluded if B).
* `res/run.zarr` contains every frame's `targets/cam_*/frame_*` and `correspondences/frame_*` + `linkage/ptv_is/frame_*` readable via `RunStore`.
* `Visualize 3D positions` / `Visualize 3D trajectories` show identical clouds before/after (manual check + `test_plot_3d_positions.py` migrated).
* `uv run pytest` shows 0 new failures vs baseline.

## 9. Effort & ordering

~4.5 days if B (calibration stays ASCII), ~5.5 if A. Order: **Phase 0 → Phase 4 (tests) → Phase 1 (W1/W2) → Phase 2 (R1/R2/R4) → Phase 5 → Phase 6**. Phase 4 before Phase 1 is required to keep CI green.

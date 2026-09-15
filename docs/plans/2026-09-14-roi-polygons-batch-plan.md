# Plan: ROI polygons (2D mask + 3D trajectory ROI) that work in GUI and batch, incl. parallel workers

Date: 2026-09-14. Branch: `feature/roi-polygon-mask`.

## 1. Goal

Two ROI mechanisms already exist for complex scenes where the coarse
`criteria` box (`X_lay`/`Zmin_lay`/`Zmax_lay`) is not enough:

1. **2D polygon ROI mask** — per-camera `mask_{i}.txt` drawn in `MaskGUI`,
   zeroes image pixels outside the polygon before detection
   (`src/openptv2/roi_mask.py`, `src/openptv2/gui/mask_gui.py`).
2. **3D trajectory ROI** — `trajectory_roi.json` (`xy/xz/yz` polygons) drawn
   on orthogonal projections of the stereo-matching cloud, dropping 3D
   points outside the selected region before tracking
   (`src/openptv2/trajectory_roi.py`,
   `src/openptv2/gui/trajectory_roi_gui.py`).

Goal: both must behave identically in the GUI, serial batch, and parallel
batch workers. Today they mostly work by accident of `cwd()` and only on
some code paths (see §3).

## 2. Confirmed semantics (no change)

3D ROI `contains()` (`trajectory_roi.py:50-60`): a point is kept iff it is
inside **every polygon that is set**. An undrawn plane imposes no
constraint; an empty/missing ROI passes everything. This is exactly the
requested "any subset of projections" behavior — keep it.

2D mask (`roi_mask.py:67-77`): per-camera file presence = active; missing
file = full frame. Keep.

Filter placement stays at **correspondence-write time** (after
`point_positions`, before `store.write_correspondences`), so filtered
points never reach tracking. No tracking-side change needed.

## 3. Gaps found (code refs on this branch)

- **G1. Custom sequence plugins bypass the 3D ROI.** Filtered in
  `gui/ptv.py:800` (`py_determination_proc_c`), `:1175`
  (`py_sequence_loop`), `:1559` (`py_sequence_loop_python`) — so the
  `default` sequence plugin is covered. But `plugins/contour_sequence.py`,
  `plugins/rembg_sequence.py`, `plugins/rembg_contour_sequence.py`
  triangulate and `write_correspondences` directly with **no ROI call**.
  ROI silently does nothing there.
- **G2. Folder is implicit (`Path.cwd()`).** `load_mask_polygon`,
  `apply_roi_mask`, `load_trajectory_roi`, `filter_by_trajectory_roi` all
  default `working_folder=None → cwd()`. Serial batch happens to work
  because `batch/pyptv_batch.py:192` and
  `batch/pyptv_batch_parallel.py:94` (`run_sequence_chunk`) `chdir(exp_path)`.
  `_process_frame_worker` (`gui/ptv.py:193-240`) and any future caller that
  does not chdir will silently see "no mask / no ROI".
- **G3. Copy-pasted triangulate→pad→write blocks** in 3 plugins + 3 `ptv.py`
  sites drifting apart (e.g. pair_flag gating landed in `correspondences`
  but ROI did not land in the custom plugins).

## 4. Design

- **Single helper, single rule.** Add
  `triangulate_filter_write(flat, cpar, cals, vpar, num_cams, sorted_corresp, store, frame, working_folder)`
  (name TBD, lives next to `filter_by_trajectory_roi` or in
  `gui/ptv.py`): `point_positions` → pad `print_corresp` to 4 rows →
  `filter_by_trajectory_roi(pos, corresp, working_folder)` → `store`.
  All 6 call sites (3× `ptv.py`, 3× custom plugins) use it.
- **Explicit folder threading.** Resolve once per run from the yaml parent /
  `exp_path` (`validate_experiment_setup` / `yaml_file.parent`), pass down
  through `py_sequence_loop(exp, working_folder=None)` (default = yaml
  parent, fallback cwd for legacy callers) into detection
  (`apply_roi_mask(img, i, folder)`) and triangulation (helper above).
  Parallel chunk workers (`run_sequence_chunk`) receive the already-resolved
  path as an argument instead of relying on their own chdir.
- **Caches stay as-is.** `_rasterized_cache` / `_roi_cache` are keyed by
  path+mtime and are per-process — correct under multiprocessing, just cold
  on first frame. No shared state, JSON is read-only during a run.
- **No behavior change when ROI absent.** Missing file / empty ROI =
  unrestricted (current default). Missing file must stay silent-but-logged
  (see §6), never an error, to preserve existing runs.

## 5. Phases

1. **P1 — Explicit folder.** Add optional `working_folder` params to
   `py_sequence_loop`, `py_sequence_loop_python`, `py_determination_proc_c`,
   `_process_frame_worker` arg tuple, `preprocess_and_detect_all_parallel`;
   default resolution: `exp.pm.yaml_path.parent` → `Path.cwd()`. Pass to
   every `apply_roi_mask` / `filter_by_trajectory_roi` call. Batch entry
   points (`pyptv_batch.py`, `pyptv_batch_parallel.py::run_sequence_chunk`)
   resolve once and pass explicitly.
2. **P2 — Central helper + cover custom plugins.** Extract helper, replace
   all 6 triangulate→write blocks. No logic change for already-filtered
   paths; custom plugins gain filtering.
3. **P3 — Observability.** Log once per run (ROI path, which of xy/xz/yz
   set, 2D masks found per camera) + per-frame kept/dropped counts at
   debug level. Warn (once) when folder has neither ROI file — makes the
   silent default visible.
4. **P4 — GUI validation.** `trajectory_roi_gui` save path: refuse/confirm
   polygons with <3 vertices (matplotlib `contains_points` on degenerate
   input is meaningless). `MaskGUI` already requires ≥4 points — keep.
5. **P5 — Docs.** Document the workflow: sequence on sample frames → draw
   2D masks + 3D ROI from the correspondence cloud (`trajectory_roi_gui`
   reads correspondences first, so ROI is drawable *before* tracking) →
   re-run sequence+tracking. Note `mask_{i}.txt` is 0-based per camera.

## 6. Parallel-worker details

- `run_sequence_chunk(yaml_file, ...)` already chdirs to `exp_path`; add an
  explicit `working_folder` derived from `yaml_file.parent` and forward it
  — robust under spawn (Windows) and any future refactor that drops the
  chdir.
- Read-only sharing: workers only read `mask_*.txt` /
  `trajectory_roi.json`; mtime cache invalidates correctly if the user
  re-saves mid-run (last-writer-wins per process, acceptable).
- Cost: one `stat()` per file per frame per worker after warm-up; rasterize
  / JSON parse happen once.

## 7. Tests

- **Unit (existing files):** `trajectory_roi.contains` — AND over set
  planes, subset (only xy set), empty passthrough; degenerate polygon
  guard. `roi_mask` — missing file passthrough, inside/outside pixels.
- **Plugin coverage (new):** each of the 4 sequence plugins with an ROI
  present filters identically (parametrize plugin × ROI).
- **Equivalence:** serial vs 2-worker parallel chunk produce identical
  kept/dropped sets with the same ROI.
- **CWD-independence (regression for G2):** run from a different cwd with
  explicit folder → still filters; with `working_folder=None` from exp dir
  → unchanged legacy behavior.
- **GUI save/load (existing pattern):** ROI round-trip
  `to_json`/`from_json`; keep `test_cavity`-style e2e untouched.

## 8. Acceptance gates

- `uv run pytest tests/unit -q -o addopts=""` green (incl. new tests).
- Manual: draw one-plane ROI in GUI → GUI sequence, serial batch, and
  2-worker parallel batch report identical correspondence counts per frame.
- No change for runs without ROI files (byte-identical `rt_is` counts on
  `test_cavity`).

## 9. Risks / non-goals

- Custom-projection or per-frame-varying ROIs are out of scope; ROI is
  static per run by design.
- Rebuilding Cython extensions is needed if `algorithms/` is touched —
  this plan touches only `gui/ptv.py`, `plugins/*`, `batch/*`,
  `trajectory_roi.py`/`roi_mask.py` signatures (pure Python), so no rebuild.
- `mask_{i}.txt` 0-based vs camera 1-based naming is kept as-is (consistent
  internally); renaming would break existing runs.

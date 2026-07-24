# Plugins done right: one YAML, same splitter pipeline in GUI, batch, and parallel cloud runs

> Order: implement AFTER `2026-07-14-orientation-review-fixes.md` and
> `2026-07-14-mmlut-multimedia-strategy.md`. Coordinate with the mmlut plan's
> Phase 3 (`prepare_mmluts` at batch entry points) — both plans touch
> `pyptv_batch.py` / `pyptv_batch_parallel.py`.

## Context

The target workflow: calibrate and tune parameters on the desktop GUI (splitter
mode: one multiplexed image carrying 4 views), save the YAML, then run thousands
of frames on the cloud in parallel at C speed from that same YAML — including
`"use splitter mode"` — with no extra flags and no GUI. Sending one 4-view image
is cheaper and simpler than sending 4 files, so the splitter path is one of the 
cloud path cases, very often used.

Commit `94ddf23` already unified plugin resolution: one loader
(`src/openptv2/plugins/loader.py`) with built-ins → entry points → experiment-local
`plugins/` dir, and "default" is itself a plugin, so GUI (`gui/ptv.py:709-720`) and
batch (`batch/pyptv_batch.py:166-178`) share one dispatch path. That part is sound.

> **Status (2026-07-24): the functional gaps below are CLOSED.** The parallel
> runner (`batch/pyptv_batch_parallel.py`) now dispatches through the plugin layer
> (`resolve_selected_plugins` + per-worker `run_sequence_plugin`, `--sequence-plugin`
> CLI), both batch CLIs read `plugins.selected_*` from the YAML, and
> `splitter_sequence.py` reads `ptv.splitter_order` from the YAML instead of a
> hardcoded order. Covered by `tests/batch/test_pyptv_batch_parallel.py`,
> `test_headless_plugins.py`, and `test_splitter_end_to_end.py` (all green). The
> only non-functional item left is gap #3 (splitter_sequence still duplicates the
> core loop) — a dedup refactor, not a blocker.

**Research findings (code audit, 2026-07-14) — the remaining gaps:**

1. **The parallel batch runner has zero plugin support.**
   `batch/pyptv_batch_parallel.py` calls `py_sequence_loop` directly
   (`run_sequence_chunk`), so splitter mode simply cannot run in parallel — the
   exact runner the cloud workflow needs. The chunking model itself (frame ranges
   over `ProcessPoolExecutor`, each worker re-loads the YAML) is the right shape;
   it just bypasses the plugin layer.
2. **The batch CLI ignores the YAML plugin selection.** `parse_command_line_args`
   defaults `--sequence-plugin` / `--tracking-plugin` to the literal `"default"`,
   never reading `plugins.selected_sequence` / `selected_tracking` that the GUI
   saves into the YAML (`pyptv_gui.py:1251-1252`). So "send the YAML and it runs
   with splitter" does not work today — the cloud job must repeat the selection on
   the command line. (The deprecated `pyptv_batch_plugins.py:119-120` is worse:
   it falls back to `available_*[0]`, not the selected one.)
3. **`splitter_sequence` duplicates the entire core sequence loop.**
   `plugins/splitter_sequence.py` re-implements detection → correspondence →
   `rt_is` writing (~150 lines that mirror `py_sequence_loop`), differing only in
   how the per-camera images are obtained (read one file, `image_split`, optional
   mask/invert). Every core improvement (e.g. mmlut init, output format fixes)
   must now be done twice, and the copies have already drifted (per-frame
   `print`s, ad-hoc filename fallbacks, `skimage` imports inside the frame loop).
4. **The split order is hardcoded.** `splitter_sequence.py:115` passes
   `order=[0, 1, 3, 2]` with a "HI-D specific" comment; `image_split`
   (`gui/ptv.py:317`) also has it as a mutable default argument. A different
   camera multiplexer silently gets the wrong view↔calibration mapping — a
   correctness trap, not just inflexibility.
5. **Two unrelated splitter flags.** Sequence uses `ptv.splitter`; the calibration
   GUI uses `cal_ori.cal_splitter` (`calibration_gui.py:453`). Both describe the
   same physical camera setup. Acceptable to keep both (calibration images may
   differ from run images) but the YAML must make the relationship explicit and
   the GUI should default one from the other.
6. **YAML stores runtime facts.** The `plugins:` section persists
   `available_tracking` / `available_sequence` lists (loader
   `discover_available_plugins`). Availability is a property of the installed
   package + local `plugins/` dir at runtime, not of the experiment; persisting
   it produces stale lists and enabled the `available[0]` bug in (2). Only the
   `selected_*` keys belong in the YAML.
7. **Headless-safety is unverified.** The loader injects `openptv2.gui.ptv` into
   every plugin. The headless/cloud install (`docs/cloud-batch.md`, commit
   `e2fb977`) must be able to import and run the full plugin path without GUI
   packages (PySide6/chaco/enable). There is no test guarding this.
8. Test infrastructure: `test_data/test_splitter/` has only a YAML + addpar (no
   images); `tests/batch/test_ext_sequence_splitter.py` and
   `tests/batch/test_pyptv_batch_plugins.py` exist and must keep passing.

## Goal

One YAML is the single contract: the GUI writes it, and `openptv2-batch`
(serial or parallel, desktop or cloud container) reproduces exactly the GUI's
sequence + tracking behavior from it — splitter included — with the core loop
implemented once and running at compiled speed.

## Phase 1 — YAML is the single source of truth

1. **Batch honors YAML selection.** In `batch/pyptv_batch.py`: change
   `--sequence-plugin` / `--tracking-plugin` CLI defaults from `"default"` to
   `None`; in `run_batch`, when the argument is `None`, read
   `experiment.pm.get_parameter("plugins")` and use `selected_sequence` /
   `selected_tracking` (falling back to `"default"` if the section is absent).
   Explicit CLI flags still override the YAML. Mirror the same in
   `pyptv_batch_parallel.py` (Phase 3) and fix the deprecated
   `pyptv_batch_plugins.load_plugins_config` to prefer `selected_*` over
   `available_*[0]`.
2. **Splitter order becomes a parameter.** Add `splitter_order` (list of 4 ints,
   default `[0, 1, 3, 2]` for backward compatibility) to the `ptv` section:
   `gui/parameter_models.py`, `gui/parameter_defaults.py`, and the parameter GUI
   next to the existing `splitter` checkbox (`gui/parameter_gui.py`). Fix the
   mutable default argument in `image_split` (`order=None` → use stored default).
   `splitter_sequence` and the calibration splitter path read it from the YAML.
3. **Stop persisting `available_*`.** `discover_available_plugins` keeps
   computing them at runtime for the GUI dropdowns, but the ParameterManager
   should treat them as transient (drop on save, ignore on load — keep reading
   them harmlessly for old YAMLs). Only `selected_tracking` / `selected_sequence`
   remain persisted. Update `tests/gui/test_parameter_manager_yaml_plugins.py`
   accordingly.
4. **Sanity check on selection.** When the selected plugin doesn't resolve,
   `run_batch` must fail fast with the loader's `PluginError` naming the YAML key
   (cloud runs should die loudly, not fall back to `default` silently).

## Phase 2 — One sequence loop, splitter as an image provider

Kill the duplication at the right altitude: the only thing splitter changes is
**how per-camera images are acquired**, so make image acquisition the pluggable
step instead of copying the whole loop.

1. In `gui/ptv.py`, extract a frame-image provider from `py_sequence_loop`:
   `read_frame_images(exp, frame) -> list[np.ndarray]` with two implementations:
   - default: read `num_cams` files (current behavior),
   - splitter: read one file via `spar.get_img_base_name(0)`, grayscale if
     needed, optional `negative`, `image_split(order=<yaml>)` — reuse the robust
     filename/mask logic currently in `splitter_sequence.py` (move it, don't
     copy it).
   Selection: `ptv_params["splitter"]` → splitter provider. `py_sequence_loop`
   then contains the loop exactly once (detection, masking, correspondence,
   target/rt_is writing).
2. Reduce `plugins/splitter_sequence.py` to the same shape as
   `default_sequence.py`: validate `ptv.splitter` is set, then call
   `self.ptv.py_sequence_loop(self.exp)` (which now picks the splitter provider
   from the same YAML). Keep the plugin name (and `ext_sequence_splitter` alias)
   for backward compatibility — after this phase "splitter mode" works even with
   `selected_sequence: default`, and the plugin remains a thin explicit alias.
3. `splitter_tracking` stays as is (it only redirects target file bases) but
   verify `py_sequence_loop` writes targets to the same
   `exp.target_filenames` bases the tracker reads (this is the GUI↔batch
   consistency point that has broken before — cover it with the Phase 4 test).
4. Move the per-frame `print`s to `logging.debug`/`info` (thousands of frames ×
   several prints per frame is real overhead on cloud stdout collection and
   makes worker logs unreadable).
5. Ensure calibration's splitter path (`calibration_gui.py:453`) uses the same
   provider function (single image → 4 views with the same `splitter_order`),
   so GUI calibration and cloud sequence cannot disagree on view mapping.

## Phase 3 — Plugins in the parallel runner (the cloud path)

1. In `pyptv_batch_parallel.run_sequence_chunk`: after loading the YAML and
   building the ProcessingExperiment (reuse the construction from
   `pyptv_batch.run_batch` — extract a shared
   `build_processing_experiment(yaml_file, seq_first, seq_last)` helper into
   `batch/` instead of keeping two copies), run the sequence via
   `run_sequence_plugin(selected_or_cli_name, proc_exp, plugins_dir)` instead of
   calling `py_sequence_loop` directly. Workers already re-create everything from
   the YAML path, so nothing unpicklable crosses process boundaries — keep that
   property (pass plugin *names*, never plugin instances).
2. Tracking remains sequential after all chunks finish (frame-to-frame
   dependency), run through `run_tracking_plugin` with the YAML-selected name.
3. Call the mmlut plan's `prepare_mmluts` once per worker at chunk start (if that
   plan landed) — LUT init amortizes per chunk, not per frame.
4. Headless guard: add a unit test that, with GUI packages absent (simulate via
   `sys.modules` monkeypatching or a subprocess with
   `-X importtime`-free minimal env), imports `openptv2.plugins`, resolves both
   splitter plugins, and imports `openptv2.gui.ptv` — asserting no
   PySide6/chaco/enable import is triggered. This pins the "runs in a slim cloud
   container" property. If `gui/ptv.py` currently drags GUI deps in, split the
   headless core out (report what you find rather than doing a large move
   silently).

## Phase 4 — End-to-end proof + benchmark

1. **Fixture:** build a tiny synthetic splitter dataset under
   `test_data/test_splitter/`: generate N (≈5) frames of 2×2-multiplexed images
   by placing Gaussian blobs consistent with 4 synthetic calibrations (reuse
   generators from `test_data/synthetic` / existing test utilities if present),
   plus the YAML with `ptv.splitter: true`, `splitter_order`, and
   `plugins.selected_sequence: splitter_sequence`,
   `selected_tracking: splitter_tracking`.
2. **Integration test** (`tests/batch/test_splitter_end_to_end.py`):
   - run `pyptv_batch.run_batch(yaml, first, last)` with NO plugin flags →
     assert splitter plugins were used (targets + `res/rt_is.*` + `res/ptv_is.*`
     produced, correspondence counts > 0);
   - run `pyptv_batch_parallel.main(yaml, first, last, n_processes=2)` → assert
     identical `rt_is` outputs to the serial run (byte-compare or numeric-compare
     per frame). This is the "same YAML, same result, GUI-free, parallel" claim,
     executable in CI.
3. **Benchmark** (extend or mirror the mmlut plan's perf pattern,
   `@pytest.mark.perf`): frames/second for the splitter sequence, serial vs
   `n_processes ∈ {2, 4}`, and (if 4-file test data is available) splitter
   single-file vs 4-file I/O — the number that justifies the "one multiplexed
   image is faster to ship and process" decision for the cloud design.
4. Update `docs/cloud-batch.md` and `docs/tutorials/plugins.md`: the workflow is
   "tune in GUI → YAML carries `splitter: true` + selected plugins → run
   `openptv2-batch <yaml> <first> <last>` (or the parallel runner) anywhere".

## Explicitly out of scope

- The cloud orchestration itself (job submission, storage, containers) — this
  plan only guarantees the runner is YAML-driven, GUI-free, parallel, and fast.
- New plugin kinds (preprocessing hooks, per-frame callbacks) — the
  Sequence/Tracking contract stays as is.
- Rewriting `splitter_tracking.do_back_tracking` (still a placeholder — leave).

## Verification

```bash
# Rebuild if any algorithms/*.py changed (not expected in this plan, but cheap)
uv run python setup.py build_ext --inplace

# Plugin + batch test suites (existing must keep passing)
uv run pytest tests/batch/ tests/gui/test_parameter_manager_yaml_plugins.py tests/gui/test_plugins_dialog.py -v

# New end-to-end splitter test (serial + parallel identical outputs)
uv run pytest tests/batch/test_splitter_end_to_end.py -v

# Headless import guard
uv run pytest tests/batch/ -k headless -v

# Perf numbers (report in PR)
uv run pytest -m perf -k splitter -v -s

# Lint
uv run ruff check src/openptv2/plugins src/openptv2/batch src/openptv2/gui/ptv.py
```

Success criteria: a YAML saved by the GUI with splitter enabled runs unmodified
through both batch runners with no CLI plugin flags; serial and parallel outputs
are identical; the sequence loop exists in exactly one place; the split order is
a parameter; headless import is test-guarded; frames/sec scaling with worker
count is documented.

# Plugin restructure plan

Date: 2026-07-13
Status: implemented (Phases 1–4)

## Problem

Plugins (`ext_sequence_*` / `ext_tracker_*`) currently live in **five places**:

| Location | Contents | State |
|---|---|---|
| `src/openptv2/gui/plugins/` | contour, denis, rembg, rembg_contour | packaged in the wheel, but **never loaded from there**; `ext_tracker_denis.py` still imports legacy `optv.*` |
| `test_data/test_cavity/plugins/` | same five files | drifted: contour/rembg/rembg_contour import legacy `optv.*` |
| `test_data/test_splitter/plugins/` | splitter sequence + tracker | imports legacy `optv.*` and `from pyptv import ptv` |
| `test_data/test_rembg/plugins/` | rembg only | legacy `optv.*` imports |
| `test_data/test_rembg_small/plugins/` | rembg only | legacy `optv.*` imports |

Root causes:

1. **Loading is cwd-based.** Both `gui/ptv.py` (`run_sequence_plugin` / `run_tracking_plugin`)
   and `batch/pyptv_batch_plugins.py` resolve `Path.cwd() / "plugins"`, append it to
   `sys.path`, and `importlib.import_module` by bare filename. Every experiment folder
   therefore needs its own copy — hence the duplication and drift.
2. **Discovery is filename sniffing.** `parameter_manager.scan_plugins_dir()` classifies
   a plugin by whether `"sequence"` or `"track"` appears in its filename.
3. **No formal contract.** The contract (a class named `Sequence` with `do_sequence()`,
   or `Tracking` with `do_tracking()`, taking `exp=`) exists only by convention and
   `hasattr` checks; errors are swallowed with `print`.
4. **Heavy deps at import time.** `ext_sequence_rembg*.py` runs
   `session = new_session("u2net")` at module import — downloads an ONNX model as a side
   effect. `rembg` is deliberately not a declared dependency, so availability is
   discovered by ImportError at run time.
5. **Dataset-specific code in shipped source.** `pyptv_batch_plugins.py` contains a
   "Patch: Ensure output files are written to 'res' directory for test_splitter", and
   the splitter plugins encode what is really a core capability (the `splitter` /
   `cal_splitter` flags already exist in `parameter_models.py`).
6. **Cloud gap.** `Dockerfile.cloud` installs the wheel non-editably and runs
   `openptv2-batch` (no plugin support); `pyptv_batch_plugins` is not an entry point and
   would only find plugins if the mounted data folder carried its own copies.

## Decision on Cython

Do **not** cythonize plugins — state this as a deliberate design decision, not an
accident. Plugins are the extensibility surface: users must be able to drop a readable
`.py` file into an experiment. They are I/O-and-glue code around the already-compiled
`algorithms/` kernels, so there is no performance argument. Keeping them pure Python is
consistent once they are moved out of the "compiled runtime" area into a dedicated
plugins package.

## Target architecture

```
src/openptv2/plugins/            # runtime feature, not a GUI feature
    __init__.py                  # registry + discovery API
    base.py                      # SequencePlugin / TrackingPlugin protocols
    loader.py                    # single resolve/load path used by GUI and batch
    default_sequence.py          # "default" is a plugin too — see Phase 3
    default_tracking.py
    splitter_sequence.py
    splitter_tracking.py
    contour_sequence.py
    rembg_sequence.py            # lazy rembg import, session created on first use
    rembg_contour_sequence.py
```

(The original plan listed `denis_sequence.py`/`denis_tracking.py` as built-ins; these
were dropped during Phase 1 instead — they referenced C-API functions and a parameter
shape that no longer exist in this codebase, and weren't used by any dataset.)

Resolution order in `loader.py` (one function, used by GUI and batch):

1. Built-ins: `openptv2.plugins.<name>` inside the package.
2. Installed third-party plugins via `importlib.metadata` entry points, group
   `"openptv2.plugins"` — the path for future separately-released plugins.
3. Experiment-local `plugins/` directory next to the parameters file — kept as the
   user-extension escape hatch (this is the *legitimate* use of the working-folder
   location), tried last and clearly logged.

Contract in `base.py`: a `Protocol` (or lightweight ABC) with `name`, `kind`
("sequence" | "tracking"), and `run(exp)`. A thin adapter keeps the legacy
`Sequence.do_sequence()` / `Tracking.do_tracking()` classes working during migration.
Discovery for the GUI dropdown / YAML `plugins:` section comes from the registry, not
from filename substring matching.

## Phases

### Phase 1 — single source of truth (mechanical, low risk)

- Create `src/openptv2/plugins/` and move the plugin files there; fix all imports to
  `openptv2.*`; delete the `from pyptv import ptv` and `optv.*` remnants.
- Delete all `test_data/*/plugins/` copies and `src/openptv2/gui/plugins/`.
- Point `run_sequence_plugin` / `run_tracking_plugin` and `pyptv_batch_plugins` at the
  new loader (built-ins + cwd fallback), keeping current YAML plugin names working via
  an alias map (`ext_sequence_splitter` → `splitter_sequence`, …).
- Update `pyproject.toml` packages list; add extras: `rembg = ["rembg"]`.
- Rewrite `tests/batch/test_pyptv_batch_plugins.py` to run on a tmp copy of
  `test_data/test_splitter` (it is currently skipped for exactly this reason) — this
  becomes the regression gate for the whole restructure.

### Phase 2 — formal plugin API

- Add `base.py` protocols and the registry; replace `scan_plugins_dir` filename
  sniffing with registry queries (experiment-local dir still scanned, but by importing
  and inspecting, not by name matching).
- Move rembg model-session creation from import time into `__init__`/first use; report
  unavailable plugins in the GUI/YAML scan as `"rembg_sequence (requires
  openptv2[rembg])"` instead of failing at run time.
- Replace `print`-and-continue error handling in the loaders with logging + raised
  exceptions (batch must exit non-zero on plugin failure — important for cloud jobs).

### Phase 3 — unify "default" into the plugin system

Revised from the original plan: rather than folding the splitter into a non-plugin
"core" pipeline, the better fit (matching how the GUI actually loads plugins at
runtime — see below) was to make **"default" itself a plugin**, so every caller
always goes through the same `Sequence(ptv=, exp=).do_sequence()` /
`Tracking(ptv=, exp=).do_tracking()` contract instead of special-casing the string
`"default"`. The splitter stays a plugin, symmetric with contour/rembg; `ptv.splitter`
/ `cal_ori.cal_splitter` remain core config flags the user sets independently of which
sequence plugin is selected.

- Added `openptv2.plugins.default_sequence`/`default_tracking`, registered as
  `"default"` in `BUILTIN_SEQUENCE_PLUGINS`/`BUILTIN_TRACKING_PLUGINS` — built-in
  lookup is tried first, so `"default"` can never be shadowed by an experiment-local
  file. `default_tracking.py` also fixed a real divergence: `pyptv_batch_plugins.py`'s
  old default-tracking branch never checked `track_mode` for 3D segment tracking,
  unlike the GUI and `pyptv_batch.py`; there's now one implementation.
- Simplified every caller (`pyptv_gui.py`'s `sequence_action`/`track_no_disp_action`,
  all `run_batch` variants) to unconditionally call the loader — no more
  `if name != "default": ... else: ...` branches anywhere.
- Auditing the GUI's plugin-selection dialog (`Plugins` class) while doing this found
  two real crash bugs, fixed as part of this phase: `Plugins.read()` called
  `ParameterManager.get_parameter("plugins")` unguarded, which raises `ValueError`
  (not `None`) for any YAML lacking a `plugins:` section — crashed GUI startup;
  `Plugins.save()` called `Experiment.get_parameter("plugins", {})` with two
  positional args against a one-arg method — crashed every "OK" click in the plugin
  dialog. Both are fixed by going through `pm.parameters` (a plain dict) directly.
- `Plugins.read()` also used to trust a session-cached, possibly-never-populated
  snapshot (`ParameterManager.from_yaml()` doesn't scan plugins at all — only the
  legacy `.par`-conversion path does); it now does a live
  `discover_available_plugins()` rescan on every open, so a plugin file dropped into
  the experiment's `plugins/` folder while the GUI is open shows up immediately.
- Plugin failures at the two GUI call sites now surface as a `pyface` warning dialog
  with an actionable message (e.g. missing `rembg` extra) instead of an unhandled
  traceback.

### Phase 4 — cloud/batch alignment

- Merged plugin selection into `openptv2-batch`: `pyptv_batch.py::run_batch`/`main`
  gained `sequence_plugin`/`tracking_plugin` params and `--sequence-plugin`
  /`--tracking-plugin` CLI flags (default `"default"`), replacing its inline
  `py_sequence_loop`/`py_trackcorr_init` dispatch with calls through the loader.
- `pyptv_batch_plugins.py::run_batch` is now a thin backward-compatible shim
  delegating to `pyptv_batch.run_batch` — kept (not deleted) because several tests
  import it directly with its legacy keyword order; `main()`'s CLI shape is
  unchanged for the same reason. Not a registered `[project.scripts]` entry point,
  same as before.
- Because built-ins ship in the wheel, `Dockerfile.cloud` gains plugin support with no
  data-folder changes; it now also bakes `test_splitter` alongside `test_cavity` and
  documents a smoke test exercising `--sequence-plugin splitter_sequence
  --tracking-plugin splitter_tracking` against it (mirrored in `docs/cloud-batch.md`).
- `pyptv_batch_parallel.py` (a third, multiprocessing batch variant, no CLI entry
  point) was explicitly left untouched — flagged as a known future gap.

## Out of scope (later)

- Actual separately-packaged plugin repos (`openptv2-plugin-rembg`) — the entry-point
  group makes this possible but nothing forces it yet.
- Sandboxing/trust policy for experiment-local plugins (cwd `sys.path` injection is
  arbitrary code execution from a data directory; acceptable for now, worth a flag
  like `--allow-local-plugins` for cloud mode later).

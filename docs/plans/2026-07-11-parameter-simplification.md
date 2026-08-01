# Parameter Simplification Plan

**Branch**: `refactor/parameter-simplification`  
**Status**: completed 2026-07-19 (see "Outcome" below — Steps 1–3 as adapted;
Steps 4–6 resolved differently than written)  
**Author**: Alex Liberzon  

## Outcome (2026-07-19)

The plan's invariants are in place, though the implementation evolved past the
plan's letter:

- **Step 1 — done** (earlier work): `Paramset` is thin (`name` + `yaml_path`,
  no parameter copy); `_open_param_dialog` swaps in a local `ParameterManager`
  loaded from the run's YAML and restores the original pm *by object
  reference* in `finally` — the active pm object is never mutated, so the
  hot-swap corruption class is gone.
- **Step 2 — done** (earlier work): `Experiment.save_active()` is the single
  save path; dialog handlers call it.
- **Step 3 — done, as pydantic instead of dataclasses**: typed section models
  live in `gui/parameter_models.py` (`AllParams` + one model per section).
  As of 2026-07-19, `ParameterManager.from_yaml` **normalizes through the
  model**: values are coerced to the declared field types at the load
  boundary (`model_dump(exclude_unset=True)` — absent sections/keys stay
  absent; unknown keys survive via `extra="allow"` on every section model).
  Invalid files fall back to raw data with a warning, as before.
  `pm.validated()` returns the fully typed `AllParams` for strict consumers.
- **Step 4 — resolved differently**: the `Int` vs `Float` TraitError class of
  bug is fixed twice over — criteria traits are declared `Float`, and the
  load boundary now coerces. The full rewrite of trait bindings to dataclass
  fields was **not** done: dict sections accessed via `get_section` /
  `get_parameter` are the settled interface, and rewriting ~100 GUI call
  sites has no remaining bug to fix.
- **Step 5 — dropped**: `_populate_cpar`-style functions keep receiving dict
  sections; those dicts are now type-coerced at the boundary, which was the
  actual point.
- **Step 6 — dropped**: `ParameterManager` still carries legacy `.par`
  parsing and plugin scanning. No observed cost; split it out if/when it
  bites.

## Problem statement

The current parameter pipeline has four layers that each hold a copy of the
same data:

```
Disk (YAML)
    ↕ ParameterManager.from_yaml / to_yaml
ParameterManager.parameters      ← dict-of-dicts, pm object lives on Experiment
    ↕ experiment.load_parameters_for_active / save_parameters
Experiment                        ← owns pm + list of Paramset (each holds another copy of dict)
    ↕ passed as experiment= arg
parameter_gui (HasTraits widgets) ← reads experiment.pm.parameters['section'][key]
    ↕ Handler.closed writes back to pm, pm.to_yaml
pyptv_gui commands                ← pulls experiment.pm.parameters['section'] directly
    ↕
algorithms: _populate_cpar, _populate_vpar, etc.
```

### Known pain points

| # | Pain point | Where it bit us |
|---|-----------|-----------------|
| 1 | **Dual storage drift** — `Paramset.parameters` (deep copy) vs `pm.parameters` drift whenever a non-active dialog is opened | Hot-swap bug: dialog opens for Run1 while Run3 is active; finally-block fails to restore, silently corrupts active | 
| 2 | **Hot-swap anti-pattern** — opening a non-active run's dialog temporarily clobbers `pm`, then restores in `finally` | Any exception between swap and restore leaves pm in wrong state |
| 3 | **Section-keyed dict access leaks everywhere** — `pm.parameters['ptv']['mmp_n2']` appears in ~10 places in pyptv_gui | Rename a section key → mass breakage |
| 4 | **No type enforcement at definition site** — `Int` trait in parameter_gui receives float YAML value → `TraitError` at runtime | `Xmin/Xmax/Zmin/Zmax` bug: declared as `Int`, YAML stores `-40.0` |
| 5 | **Manual sync** — `load_parameters_for_active()` must be called explicitly; forgetting sends stale params to algorithms | Several places do `exp.pm.parameters[...]` without reloading |
| 6 | **pm is a god object** — ParameterManager bundles YAML I/O, legacy-dir parsing, plugin scanning, n_cam detection | Hard to test individual concerns in isolation |

## Design reference

How other scientific software handles parameter pipelines:

- **OpenFOAM**: YAML/dict file is always source of truth. GUI is just an editor. Commands always load from file.
- **sklearn**: Immutable typed dataclasses passed explicitly per function call. No shared mutable state.
- **Spyder**: `Preferences` singleton with typed keys; UI widgets declare ownership of keys; single `commit()` saves all.
- **Qt Creator / VS Code**: Typed settings store; components subscribe to `onChange` per key.
- **Blender**: Properties live on the data-block (DNA); operators receive a context object, not raw dicts.

**Conclusion**: the dominant patterns are (a) single source of truth and (b) typed parameter objects instead of raw dicts. We adopt both.

## Target architecture

```
Disk (YAML)
    ↕ ParameterManager  (thin serialiser only)
Typed dataclasses  ←  PtvParams, CriteriaParams, SequenceParams, …
    ↕ Experiment.pm.params  (one copy, always current)
parameter_gui      ← binds to dataclass fields directly
    ↕ experiment.save_active()   (single save path)
pyptv_gui commands ← receive typed dataclass from experiment.pm
    ↕
algorithms
```

Key invariants in the target state:

1. **One copy** — `Paramset` is `(name, yaml_path)` only. No parameters dict on it.
2. **Dialogs own a local pm** — opening a non-active dialog creates a local `ParameterManager`, loads the YAML, shows the dialog, saves via `local_pm.to_yaml()`. The global `pm` is never touched.
3. **Typed sections** — raw dict access is replaced by dataclass attribute access.
4. **Single save** — `experiment.save_active()` is the only entry point for writing parameters to disk.
5. **Commands receive typed objects** — `_populate_cpar(ptv: PtvParams)` not `_populate_cpar(params['ptv'])`.

## Implementation steps

### Step 1 — Remove `Paramset.parameters` / kill hot-swap (highest priority)

**Files**: `experiment.py`, `parameter_gui.py`, `pyptv_gui.py`  
**Effort**: ~1 day  
**Test gate**: all existing unit tests pass; open all three run dialogs manually; verify active run unchanged after close

Changes:
- Remove `parameters` and `num_cams` fields from `Paramset`.
- Update `_load_paramset_from_yaml` to not populate those fields.
- In `_open_param_dialog`: create a `local_pm`, load `paramset.yaml_path` into it, construct dialog with `local_pm`; remove the swap/restore logic entirely.
- Remove `experiment.load_parameters_for_active()` from `_open_param_dialog.finally`.
- Update `duplicate_paramset` and `create_new_paramset` to not reference `paramset.parameters`.

```python
# Before (fragile)
experiment._override_save_path = paramset.yaml_path
experiment.pm.from_yaml(paramset.yaml_path)
dialog = dialog_cls(experiment=experiment)
# ... finally: experiment.load_parameters_for_active()

# After (safe)
local_pm = ParameterManager()
local_pm.from_yaml(paramset.yaml_path)
dialog = dialog_cls(pm=local_pm, yaml_path=paramset.yaml_path)
```

### Step 2 — Single save path

**Files**: `experiment.py`, `parameter_gui.py` (all Handlers)  
**Effort**: 2 hours  
**Test gate**: edit a parameter in each dialog, close, reopen — value persists

Changes:
- Add `Experiment.save_active()` → `self.pm.to_yaml(self.active_params.yaml_path)`.
- Each `Handler.closed(info, is_ok)` calls `info.object.experiment.save_active()` instead of reaching for `pm.to_yaml(...)` with a path.
- Commands that write parameters call `experiment.save_active()` at the end.

### Step 3 — Typed parameter dataclasses

**Files**: new `src/openptv2/gui/params.py`, `parameter_manager.py`, `parameter_gui.py`, `pyptv_gui.py`  
**Effort**: 2–3 days  
**Test gate**: parity test — load YAML via old path, load via new path, assert equal field-by-field

Create typed dataclasses for each parameter section:

```python
@dataclass
class PtvParams:
    num_cams: int = 4
    mmp_n1: float = 1.0
    mmp_n2: float = 1.46  # glass
    mmp_n3: float = 1.33  # water
    imx: int = 1280
    imy: int = 1024
    pix_x: float = 0.010
    pix_y: float = 0.010
    ...


@dataclass
class CriteriaParams:
    Xmin: float = -40.0  # Float, not Int — this is the fix for the TraitError
    Xmax: float = 40.0
    Ymin: float = -40.0
    Ymax: float = 40.0
    Zmin1: float = -40.0
    Zmax1: float = 40.0
    ...


@dataclass
class SequenceParams:
    first: int = 1
    last: int = 100
    base_name: list[str] = field(default_factory=list)
    ...


@dataclass
class AllParams:
    ptv: PtvParams = field(default_factory=PtvParams)
    sequence: SequenceParams = field(default_factory=SequenceParams)
    criteria: CriteriaParams = field(default_factory=CriteriaParams)
    track: TrackParams = field(default_factory=TrackParams)
    ...
```

`ParameterManager.from_yaml()` returns `AllParams`.  
`ParameterManager.to_yaml(path, params: AllParams)` serialises it.

### Step 4 — Update GUI widgets to bind to typed fields

**Files**: `parameter_gui.py`  
**Effort**: 1 day  
**Test gate**: open every dialog, change every field, close, re-open — no TraitError, correct value

- `Main_Params.__init__` receives `pm: ParameterManager` and reads `pm.params.ptv`, `pm.params.criteria`, etc.
- `_reload` is replaced by direct binding to dataclass fields — the `HasTraits` fields mirror the dataclass types exactly (Float for float fields, Int for int fields).
- Eliminates the `Int` vs `Float` mismatch class of bug.

### Step 5 — Update algorithm interface

**Files**: `pyptv_gui.py`, `algorithms/ptv.py` (or wherever `_populate_cpar` lives)  
**Effort**: 1 day  
**Test gate**: run a full tracking pass end-to-end on test_cavity; compare results to baseline

- `_populate_cpar(ptv: PtvParams)` — receives typed object, accesses `ptv.mmp_n2` not `params['ptv']['mmp_n2']`.
- `_populate_vpar(criteria: CriteriaParams)` — same pattern.
- Remove all `experiment.pm.parameters['section']` raw dict access from `pyptv_gui.py`.

### Step 6 — Slim down ParameterManager

**Files**: `parameter_manager.py`  
**Effort**: 0.5 day  
**Test gate**: all unit tests pass

- ParameterManager becomes a thin YAML ↔ `AllParams` serialiser: `from_yaml(path) -> AllParams`, `to_yaml(path, params)`.
- Plugin scanning, legacy-dir parsing, n_cam detection move to standalone functions (not methods on the manager).
- No more `_class_map` / legacy `.par` parsing needed once Step 3 is done (keep as separate `legacy_to_yaml(dir) -> AllParams` utility).

## Testing strategy

Each step has a mandatory gate before moving to the next:

1. `uv run pytest tests/` — all unit tests green
2. Manual GUI test (checklist in `docs/HOW_TO_TEST_GUI.md`) — open each dialog, change a value, close, reopen
3. End-to-end tracking on `test_data/test_cavity` — correspondence counts match baseline
4. Wheel build passes on CI (`uv build` locally first)

## What we are NOT doing

- No observer/event bus to sync two copies — the fix is having one copy.
- No dirty-flag mechanism — symptom, not cure.
- No live YAML writes on every keystroke — write on dialog close / explicit save.
- No backwards-compatible shim for the old dict API — once typed dataclasses exist, raw dict access is removed.

## Branch strategy

```
main
└── refactor/parameter-simplification
    ├── step1/remove-paramset-parameters
    ├── step2/single-save-path
    ├── step3/typed-dataclasses
    ├── step4/gui-typed-bindings
    ├── step5/algorithm-interface
    └── step6/slim-parameter-manager
```

Each sub-step is a single commit with a passing test gate. PR to main only after
all six steps pass the full test suite + manual GUI checklist + wheel build.

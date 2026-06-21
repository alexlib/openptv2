# Project Status - C to Python Translation

## Summary
Translating the OpenPTV C library (`lib/src/**`) into pure Python with NumPy (`algorithms/**`) following a direct, SoA-based approach.

## 📝 Translation Progress

| Module | Status | Notes |
| :--- | :--- | :--- |
| `vec_utils` | ✅ Complete | Fully translated and tested. |
| `lsqadj` | ✅ Complete | Fully translated and tested. |
| `calibration` | ✅ Complete | Fully translated and tested. |
| `parameters` | ✅ Complete | Fully translated and tested. |
| `trafo` | ✅ Complete | Fully translated and tested. |
| `multimed` | ✅ Complete | Fully translated and tested. |
| `ray_tracing` | ✅ Complete | Fully translated and tested. |
| `imgcoord` | ✅ Complete | Fully translated and tested. |
| `image_processing` | ✅ Complete | Fully translated and tested. |
| `epi` | ✅ Complete | Fully translated and tested. |
| `orientation` | ✅ Complete | Fully translated and tested. |
| `correspondences` | ✅ Complete | Fully translated and tested. |
| `segmentation` | ⏳ In Progress | `targ_rec` done, `peak_fit` needs `check_touch`. |
| `sortgrid` | ✅ Complete | Bug fixed, parity with C/Cython verified, vectorized NN. |
| `tracking_frame_buf`| ✅ Complete | Frame buffer, file I/O, SoA sync all working. |
| `tracking_run` | ✅ Complete | `tr_new`, `volumedimension`, all parameters wired up. |
| `track` | ✅ Complete | `trackcorr_c_loop`, `trackback_c`, conflict resolution with Phase 3 improvement. Numba JIT path. Parity-tested against C on burgers (exact match) and cavity (improvement over C). |
| `track3d` | ✅ Complete | `track3d_loop`, `find_candidates_in_3d`, 3-level linking. Numba JIT path. Exact parity with C/Cython on burgers. |

## 🔬 Tracking Parity Status

### Burgers dataset (5 frames, 5 particles)
- **track3d**: Python == C/Cython (exact match, 18 links)
- **trackcorr**: Python == C/Cython (exact match, 17 links)
- trackcorr gets 1 fewer link than track3d due to P2 gap at frame 10003 + empty lookahead buffer

### Cavity dataset (4 frames, ~700 particles)
- **track3d**: Python matches expected values (npart=2082, nlinks=1451)
- **trackcorr**: Python produces MORE correct links than C (945 vs 918)
  - Steps 10001-10002: exact parity with C when Phase 3 disabled (0 mismatches)
  - Extra links come from two Python improvements over C:
    1. **Phase 3 "losers retry"** — conflict losers try fallback candidates (+27 links)
    2. **Stale buffer fix** — Python clears buf[3] when no new frame available; C uses recycled data (−2 spurious links)

### Synthetic dataset (8 frames, 15 particles with crossing/curved/late-entry trajectories)
- **track3d**: 102/103 correct links (99%), 0 wrong
- **trackcorr**: 103/103 correct links (100%), 0 wrong
- trackcorr >= track3d assertion holds

### Known C bugs (not fixed in C, fixed in Python)
1. **Stale buffer recycling**: when `step >= last - 2`, C doesn't clear `buf[buf_len-1]` after rotation, causing `assess_new_position` to search stale data
2. **Overcounted count1**: C counts links inside the conflict resolution loop, so particles that lose conflicts after being counted inflate `count1`

## 🔧 Dual-Engine GUI Integration Progress

### Phase 1: Compatibility Layer — Core Objects ✅ Complete
Created `algorithms/compat/` package with optv-compatible API wrappers:
- ✅ `calibration.py` — Calibration wrapper with getter/setter methods
- ✅ `parameters.py` — ControlParams, VolumeParams, TrackingParams, SequenceParams, TargetParams, MultimediaParams wrappers
- ✅ `tracking_framebuf.py` — Target, TargetArray, Frame wrappers with constants (CORRES_NONE, PT_UNUSED)
- ✅ Test coverage: 13/13 tests passing in `test_compat_core.py`

### Phase 2: Processing Function Wrappers ✅ Complete
- ✅ `transforms.py` — Batch transform wrappers (convert_arr_pixel_to_metric, convert_arr_metric_to_pixel, etc.)
- ✅ `imgcoord.py` — Image coordinate batch wrappers (image_coordinates, flat_image_coordinates)
- ✅ `image_processing.py` — preprocess_image wrapper
- ✅ `segmentation.py` — target_recognition wrapper
- ✅ `orientation.py` — Re-export calibration functions with compat unwrapping
- ✅ `epipolar.py` — Re-export epipolar_curve from epi.py
- ✅ Test coverage: 12/12 tests passing in `test_compat_processing.py`

### Phase 3: Correspondences & Tracker ✅ Complete
- ✅ `correspondences.py` — MatchedCoords class + correspondences wrapper (~210 lines)
- ✅ `tracker.py` — Tracker class wrapping functional tracking API (~165 lines)
- ✅ Test coverage: 9/9 tests passing in `test_compat_workflow.py`
### Phase 4: Parameter Converters ✅ Complete
- ✅ Added missing parameter classes to `algorithms/parameters.py` (~65 lines):
  - CalibrationPar, MultiPlanesPar, ExaminePar, PftVersionPar
- ✅ Ported `algorithms/parameter_converters.py` from old_algorithms (~451 lines)
  - All YAML→parameter converters: get_control_par, get_volume_par, get_track_par_tuple, etc.
  - Kept convert_optv_calibrations for backward compatibility
### Phase 5: Engine Dispatch Layer ✅ Complete
- ✅ Updated `openptv2/engine.py` with OPENPTV_ENGINE env var support
  - `_detect_engine()`: Auto-detect based on env var or availability
  - `get_engine()`, `set_engine()`: Updated with initialization checks
  - `is_optv_available()`, `is_python_available()`: Availability checks
- ✅ Created 11 dispatch modules in `openptv2/` (~15 lines each):
  - calibration.py, parameters.py, correspondences.py, image_processing.py
  - segmentation.py, tracking_framebuf.py, tracker.py, transforms.py
  - imgcoord.py, orientation.py, epipolar.py
  - Each checks engine and imports from optv.* or algorithms.compat.*
- ✅ Updated `openptv2/__init__.py` to re-export all public symbols
- ✅ Tested: Both engines work, imports succeed, engine switching via env var
### Phase 6: GUI Alignment & Decoupling ✅ Complete
- ✅ Integrated the TraitsUI/Chaco-based GUI with the `openptv2` dispatcher layer.
- ✅ Replaced all `from optv.*` imports with `from openptv2.*` in GUI files:
  - gui/pyptv/ptv.py, parameter_gui.py, detection_gui.py, calibration_gui.py, code_editor.py
  - gui/pyptv/standalone_calibration.py, standalone_dumbbell_calibration.py
  - gui/pyptv/ground_truth.py, dumbbell_ground_truth.py
  - gui/pyptv/flowtracks_utils.py, tracking_viz_panel.py
  - gui/plugins/ext_sequence_*.py (3 plugin files)
- ✅ Total: 57+ import statements updated across 15 files
- ✅ Verified: GUI imports and runs successfully with both engines.

### Phase 7: Parity Tests & Code Clean Up ✅ Complete
- ✅ Created `tests/test_engine_parity.py` (9 tests).
- ✅ Implemented robust automated GUI tests under `gui/tests/` verifying all components pass perfectly (257 total tests passed!).
- ✅ Verified engine detection and switching works correctly.

## 🎉 Dual-Engine Architecture Summary

**Completed:** Phases 1-7 (all main items complete and verified)

### What We Built
1. **Compatibility Layer** (`algorithms/compat/`, ~2,000 lines)
   - Wraps pure Python `algorithms/*` with optv-compatible API.
   - Getter/setter methods, TargetArray class, batch wrappers.
   - Includes robust duck-typing supporting both pure Python and read-only C/Cython wrapper targets.

2. **Engine Dispatch** (`openptv2/`, ~600 lines)
   - 11 dispatch modules: auto-select optv.* or algorithms.compat.*
   - Environment variable: `OPENPTV_ENGINE=python` or `optv`
   - Automatic fallback if optv unavailable.

3. **GUI Integration** (15 files updated)
   - All 57+ `from optv.*` → `from openptv2.*`
   - Zero code duplication, single import source.
   - Works with both engines transparently.

### How It Works
```bash
# Use Python engine (algorithms)
export OPENPTV_ENGINE=python
python -m gui.pyptv.pyptv_gui

# Use C/Cython engine (optv, default)
export OPENPTV_ENGINE=optv
python -m gui.pyptv.pyptv_gui

# Auto-detect (prefers optv if available)
python -m gui.pyptv.pyptv_gui
```

### Benefits
1. **Cloud-friendly:** Python-only install (no C compilation).
2. **Debuggable:** Step through Python code, add print statements.
3. **Backward compatible:** Existing optv code works unchanged.
4. **Future-proof:** Easy to add algorithms-only features (Numba, JAX).
5. **Zero overhead:** Wrappers are thin, no performance penalty.

## 🔄 Cython 3 Pure Python Consolidation

See `CYTHON_3_PURE_PYTHON_PLAN.md` for the full master plan. We are eliminating the dual-engine architecture and standardizing on **Cython 3 Pure Python Mode** as the single engine.

### Current State (2026-06-21)

**Phase 1 (Housekeeping & Deletion):** NOT STARTED
- `lib/` still present (73 C/H files)
- `bindings/` still present (81 files)
- `openptv2/engine.py` and 11 forwarder modules still present
- `openptv2/__init__.py` still imports from forwarders, not `algorithms/` directly

**Phase 2 (Cython 3 Annotations):** ~30% DONE
- All 19 algorithm modules have `import cython`
- 6 of 19 modules have `@cython.ccall`/`@cython.boundscheck`/etc decorators:
  `vec_utils`, `trafo`, `imgcoord`, `multimed`, `epi`, `ray_tracing`
- 13 modules still lack performance decorators
- 11 modules still reference Numba (`HAS_NUMBA`, `@njit`, `try: import numba`)
- `track_kernels.py` has a no-op `njit` shim but still uses the decorator everywhere

**Phase 3 (Build System):** PARTIAL
- `setup.py` has `cythonize()` for `algorithms/*.py` ✅
- `pyproject.toml` still lists `numba>=0.60.0` in main, optional, and dev deps ❌

**Phase 4 (GUI Integration):** Blocked on Phase 1
**Phase 5 (Verification):** Blocked on Phases 1-4

### Next Steps

#### Step 1: Remove Numba (all references)
Clean removal of all Numba code paths, shims, and dependencies:
1. **`algorithms/track_kernels.py`**: Remove the `njit` shim and all `@njit(cache=True)` decorators. Replace with `@cython.ccall` (or `@cython.cfunc` for internal helpers). Replace `prange = range` with proper Cython parallel constructs or plain `range`.
2. **11 algorithm modules**: Remove `try: import numba` / `HAS_NUMBA` / `if HAS_NUMBA:` conditional branches. Keep only the direct function calls (the "Numba path" code is already the correct algorithm, just strip the conditionals).
3. **`algorithms/image_processing.py`**: Remove `@njit` kernels and `if HAS_NUMBA:` branches; annotate with Cython decorators instead.
4. **`pyproject.toml`**: Remove `numba>=0.60.0` from all dependency groups.
5. **Docstrings/comments**: Remove all "Numba JIT", "Numba-compiled", "when Numba is available" language.

#### Step 2: Verify C algorithm parity
Systematic audit of each `algorithms/*.py` against its C counterpart in `lib/src/*.c`:
- Line-by-line comparison of algorithm logic, constants, edge cases
- Ensure no Python-only shortcuts diverged from the C originals
- Focus on the tracking hot path (`track.py`, `track3d.py`, `track_kernels.py`)
- Run parity tests against known datasets (burgers, cavity, synthetic)

#### Step 3: Complete Cython 3 annotations (remaining 13 modules)
Add `@cython.ccall`, `@cython.cfunc`, `@cython.boundscheck(False)`, typed locals, and memoryviews to:
`calibration`, `correspondences`, `image_processing`, `lsqadj`, `orientation`,
`parameters`, `segmentation`, `sortgrid`, `track`, `track3d`, `track_kernels`,
`tracking_frame_buf`, `tracking_run`

#### Step 4: The Great Purge (Phase 1 of the plan)
Delete `lib/`, `bindings/`, `openptv2/engine.py`, forwarder modules.
Rewire `openptv2/__init__.py` to import directly from `algorithms/`.

#### Step 5: Build & test
Verify compilation, wheel builds, GUI, and full test suite.

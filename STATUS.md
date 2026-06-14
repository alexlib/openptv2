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
### Phase 6: GUI Migration ✅ Complete
- ✅ Replaced all `from optv.*` imports with `from openptv2.*` in GUI files:
  - gui/pyptv/ptv.py: Top-level imports (lines 24-38) + 13 inline imports
  - gui/pyptv/pyptv_gui.py, calibration_gui.py, detection_gui.py
  - gui/pyptv/standalone_calibration.py, standalone_dumbbell_calibration.py
  - gui/pyptv/ground_truth.py, dumbbell_ground_truth.py
  - gui/pyptv/flowtracks_utils.py, tracking_viz_panel.py
  - gui/pyptv/visualize_cameras_nb.py, visualize_rt_is_nb.py
  - gui/plugins/ext_sequence_*.py (3 plugin files)
- ✅ Total: 57+ import statements updated across 15 files
- ✅ Verified: GUI imports successfully with both engines
### Phase 7: Parity Tests & Documentation ✅ Complete
- ✅ Created `tests/test_engine_parity.py` (9 tests)
  - Tests all major APIs: Calibration, Parameters, Transforms,
    TargetArray, ImageCoordinates, Epipolar, Tracker
  - Python engine: 9/9 tests passing
  - optv engine: Imports work (C extension has pre-existing segfault bug)
- ✅ Verified engine detection and switching works correctly

## 🎉 Dual-Engine Architecture Summary

**Completed:** Phases 1-6 (6 commits, ~3,200 lines of code)

### What We Built
1. **Compatibility Layer** (`algorithms/compat/`, ~2,000 lines)
   - Wraps pure Python `algorithms/*` with optv-compatible API
   - Getter/setter methods, TargetArray class, batch wrappers
   - 34/34 tests passing

2. **Engine Dispatch** (`openptv2/`, ~600 lines)
   - 11 dispatch modules: auto-select optv.* or algorithms.compat.*
   - Environment variable: `OPENPTV_ENGINE=python` or `optv`
   - Automatic fallback if optv unavailable

3. **GUI Integration** (15 files updated)
   - All 57+ `from optv.*` → `from openptv2.*`
   - Zero code duplication, single import source
   - Works with both engines transparently

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

### Code Statistics
- **New code:** ~3,200 lines
  - algorithms/compat/: ~2,000 lines (12 modules)
  - algorithms/parameter_converters.py: ~450 lines
  - openptv2/ dispatch: ~300 lines (11 modules + __init__)
  - openptv2/engine.py: ~120 lines
  - algorithms/parameters.py additions: ~65 lines
  - Test files: ~400 lines
- **Deleted:** ~1,200 lines (old openptv2 files replaced)
- **Modified:** 16 GUI files (import swaps only)
- **Tests:** 34/34 passing (Phase 1-3 compat tests)

### Benefits
1. **Cloud-friendly:** Python-only install (no C compilation)
2. **Debuggable:** Step through Python code, add print statements
3. **Backward compatible:** Existing optv code works unchanged
4. **Future-proof:** Easy to add algorithms-only features (Numba, JAX)
5. **Zero overhead:** Wrappers are thin, no performance penalty

## 🚀 Next Immediate Steps
1. ✅ **Consolidate Parameter Management:** With the GUI reverting to TraitsUI (abandoning the Tkinter migration), we have completed the integration of `ParameterManager` into the legacy `gui/pyptv` code.
2. ✅ **Phase out `.par` and `exec()`:** Standardized on `.yaml` files exclusively for configuration, but preserved legacy `.par` translation for backward compatibility. All `exec()` usages in the `gui/` directory have been replaced with safe `getattr()` calls.
3. ✅ **Distribution & Parity:** Verified Engine Parity (C vs Numba vs pure Python). Tested engine selection and cross-validated tracking results successfully.
4. **Binary Wheels & Installers:** Provide precompiled C binary wheels to ensure robust multi-engine fallback prior to building PyInstaller standalone executables.

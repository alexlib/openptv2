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
### Phase 5: Public API Alignment ✅ Complete
- ✅ `openptv2/` now exposes a single compatibility/runtime layer
- ✅ Legacy dispatch logic has been removed from the public modules
- ✅ `openptv2.__init__` exports runtime metadata via `get_runtime_info()`
### Phase 6: GUI Alignment & Decoupling ✅ Complete
- ✅ Integrated the TraitsUI/Chaco-based GUI with the `openptv2` single runtime.
- ✅ Replaced all `from optv.*` imports with `from openptv2.*` in GUI files:
  - gui/pyptv/ptv.py, parameter_gui.py, detection_gui.py, calibration_gui.py, code_editor.py
  - gui/pyptv/standalone_calibration.py, standalone_dumbbell_calibration.py
  - gui/pyptv/ground_truth.py, dumbbell_ground_truth.py
  - gui/pyptv/flowtracks_utils.py, tracking_viz_panel.py
  - gui/plugins/ext_sequence_*.py (3 plugin files)
- ✅ Total: 57+ import statements updated across 15 files
- ✅ Verified: GUI imports and runs successfully with the unified runtime.

### Phase 7: Parity Tests & Code Clean Up ✅ Complete
- ✅ Implemented automated GUI tests under `gui/tests/`.
- ✅ Added runtime validation coverage for the unified API.
- ✅ Removed dual-engine dispatcher tests.

## 🎉 Single-Engine Architecture Summary

**Completed:** The repository now targets a single Cython 3 pure-Python runtime.

### What We Built
1. **Compatibility Layer** (`algorithms/compat/`, ~2,000 lines)
   - Wraps pure Python `algorithms/*` with optv-compatible API.
   - Getter/setter methods, TargetArray class, batch wrappers.
   - Includes robust duck-typing supporting both pure Python and read-only C/Cython wrapper targets.

2. **Public API Layer** (`openptv2/`)
   - Re-exports the compatibility/runtime surface without engine switching.
   - Reports runtime metadata via `get_runtime_info()`.

3. **GUI Integration** (15 files updated)
   - All 57+ `from optv.*` → `from openptv2.*`
   - Zero code duplication, single import source.
   - Works with the unified runtime transparently.

### How It Works
```bash
# Run interpreted modules during development
uv run python -m gui.pyptv.pyptv_gui --workdir=./test_data/test_cavity

# Or build/install the package so the same modules run compiled
uv pip install -e .
```

### Benefits
1. **Single source of truth:** one implementation path in `algorithms/`.
2. **Debuggable:** step through Python code when running uncompiled modules.
3. **High performance:** the same modules compile under Cython 3.
4. **Simpler packaging:** no separate C library or binding tree to maintain.
5. **Stable API:** `openptv2` continues to expose the compatibility surface used by the GUI.

## 🔄 Cython 3 Pure Python Consolidation

See `CYTHON_3_PURE_PYTHON_PLAN.md` for the full master plan. We are eliminating the dual-engine architecture and standardizing on **Cython 3 Pure Python Mode** as the single engine.

### Current State (2026-06-22)

**Phase 1 (Housekeeping & Deletion):** IN PROGRESS
- `openptv2/engine.py` removed
- `openptv2/*` dispatch modules now import the single compatibility/runtime layer directly
- Remaining repository cleanup is focused on deleting legacy `lib/` and `bindings/` trees plus stale comparison utilities

**Phase 2 (Cython 3 Annotations):** ✅ COMPLETE
- The translated algorithm modules use `cython` imports and decorators broadly.
- Numba-specific code paths have been completely removed.

**Phase 3 (Build System):** ✅ COMPLETE
- `setup.py` compiles and cythonizes all 18 `algorithms/*.py` modules into C extensions.
- `pyproject.toml` dependencies and setup are completely cleaned up and ready for Cython 3 Pure Python wheels.
- Tested and verified local extension compilation via `uv build --wheel`.

**Phase 4 (GUI Integration):** IN PROGRESS
- GUI entry points now run against the single runtime and ignore legacy engine-selection flags for compatibility

**Phase 5 (Verification):** IN PROGRESS
- Dual-engine tests are being replaced by single-runtime smoke and validation coverage

### Next Steps

#### Remaining cleanup
1. Delete the legacy `lib/` and `bindings/` source trees and any scripts that still import `optv`.
2. Remove or rewrite tests and docs that still describe dual-engine behavior.
3. Keep tightening the public `openptv2` namespace around the single runtime while preserving compatibility where useful.
4. Verify compilation, wheel builds, GUI, and the active test suite.

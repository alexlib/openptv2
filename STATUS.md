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
### Phase 5: Engine Dispatch Layer (Pending)
### Phase 6: GUI Migration (Pending)
### Phase 7: Parity Tests (Pending)

## 🚀 Next Step
Continue Phase 2: Create processing function wrappers for transforms, imgcoord, image_processing, segmentation, orientation, and epipolar modules.

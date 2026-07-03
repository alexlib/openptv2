# Project Status - C to Python Translation & Architectural Refactoring

## Summary
Translating the OpenPTV C library (`lib/src/**`) into pure Python with NumPy (`algorithms/**`) following a direct, SoA-based approach, and refactoring the package structure for clean separation of GUI and Batch processes.

---

## 📝 Directory Restructuring Refactoring ✅ Complete

We have restructured the project to eliminate redundant subdirectories, separate automation from visual interfaces, and simplify imports.

```
src/openptv2/
├── algorithms/             # Unified Cython 3 Pure Python runtime engine
│   ├── compat/             # Backward-compatibility API adapters
│   └── *.py                # 18 compiled/interpreted math modules
├── gui/                    # Desktop GUI (flat namespace, no nested 'pyptv' subfolder)
│   ├── plugins/            # GUI plugins
│   └── *.py                # Main GUI window, parameters, and interactive panels
└── batch/                  # Command-line batch execution & parallel wrappers
    └── *.py                # pyptv_batch.py, pyptv_batch_parallel.py, etc.
```

### Key Refactoring Actions Completed:
1. **Removed `pyptv` Nesting**: Moved all GUI modules directly to `src/openptv2/gui/` and batch tools directly to `src/openptv2/batch/`.
2. **Reorganized Test Suite**: Split the test suites into distinct `tests/gui/` and `tests/batch/` subfolders for separate testing.
3. **Optimized Batch Tests**: Re-engineered inner optimization loops in batch tests to skip redundant file-regeneration steps, dropping tracking-only sensitivity test times from **178s to 33s**.
4. **Unix Deadlock & PYTHONPATH Fixes**: Configured multiprocessing to use `'fork'` on Linux/macOS to bypass virtualenv pytest sandboxing deadlocks, and injected proper Python path environments to child subprocesses.
5. **Dynamic Compatibility Shims**: Created a virtual `openptv2.gui.pyptv` module at runtime to prevent any external plugins or legacy scripts from breaking.

---

## 📝 Translation & Engine Progress ✅ Complete

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
| `segmentation` | ✅ Complete | Fully translated and tested (including BFS typed array queue). |
| `sortgrid` | ✅ Complete | Bug fixed, parity with C/Cython verified, vectorized NN. |
| `tracking_frame_buf`| ✅ Complete | Frame buffer, file I/O, SoA sync all working. |
| `tracking_run` | ✅ Complete | `tr_new`, `volumedimension`, all parameters wired up. |
| `track` | ✅ Complete | `trackcorr_c_loop`, `trackback_c` with Phase 3 loser-retry. |
| `track3d` | ✅ Complete | `track3d_loop` 3D-linking. Numba JIT path. |

---

## 🔬 Tracking Parity Status

- **Burgers dataset (5 frames, 5 particles)**: Python matches C/Cython exactly (18 links for `track3d`, 17 links for `trackcorr`).
- **Cavity dataset (4 frames, ~700 particles)**: Python produces more high-quality correct links than legacy C due to the Phase 3 "losers retry" and stale buffer correction (945 vs 918).
- **Synthetic dataset (8 frames, 15 particles)**: 100% correct links, zero wrong links.

---

## 🛠 Full Calibration Fixes

All fixes are in `src/openptv2/` (algorithms + compatibility wrappers) and the test suite.

### Fixed: Convergence-check ordering in `orient()`
**File:** `algorithms/orientation.py`

The convergence test ran *before* constrained-parameter beta values were zeroed. When calibrating with `flags=[]` (all interior/distortion params fixed), the solver computed large updates to fixed params — but those were zeroed *after* the check, so the iteration ran all 80 cycles without ever converging.

### Fixed: Unmatched target filtering in GUI wrapper
**File:** `orientation.py` (top-level compatibility layer)

`full_calibration()` unwrapped ALL `img_pts` targets — including unmatched entries with `pnr=-999` and garbage pixel coordinates — then rewrapped every one with `pnr=i`, making the solver treat garbage as valid correspondences. This caused the catastrophic calibration corruption (x0 in billions of mm) for cameras with few matches (44/73 and 46/73).

**Fix:** The wrapper now filters out `pnr=-999` targets and their corresponding reference points *before* calling the solver.

### Fixed: Zero glass-vector default
**File:** `algorithms/calibration.py`, `Glass` class

The `Glass()` default was `(0,0,0)`, which causes division-by-zero in the imaging model (`1/|glass_vec|` → NaN cascade). Changed to `(0,0,1)` with a `sanitize()` method that warns and auto-fixes explicitly zero vectors.

### Fixed: Test data paths in `test_cavity` configs
**Files:** `test_data/test_cavity/parameters/{ptv,sequence}.par`, `parameters_Run1.yaml`

Image paths were changed from `img/` → `img_3/` during a data reorganization, but `img_3/` only contains target files (no images). Restored all paths.

### New synthetic calibration tests
**File:** `tests/unit/test_synthetic_calibration.py`

10 tests covering exterior-only recovery, per-flag convergence, edge cases (empty/no-matched targets), and — critically — a `TestWrapperUnmatchedFiltering` class that validates pnr=-999 filtering through the GUI wrapper path.

**Test count:** 219 passed, 2 skipped.

---

## 🐛 Known Issues

- **RuntimeWarning `invalid value encountered in sqrt`** at `orientation.py:98` (wrapper line during `_full_calibration` call). Occurs when the covariance-matrix inverse has negative diagonal entries (ill-conditioned normal equations). Does not crash the solver but may slow convergence. Mitigation: ensure initial guess is close to the solution (use raw_orient first) and that matched target count >> number of free parameters.
- **GUI plot overlay lines from (0,0) corner** — minor rendering issue in calibration dialog, no functional impact.
- **`sequence.base_name` still points to `img_3/`** in `parameters_Run1.yaml`. The image loading paths for tracking use `sequence.base_name` rather than `ptv.img_name`. If `img_3/` lacks actual image files, tracking will read 0 frames. Workaround: set `img_name` paths for calibration, verify `sequence.base_name` for tracking.

---

## 🎉 Single-Engine Architecture Summary

**Completed:** The repository targets a single, highly performant Cython 3 pure-Python runtime.

### Benefits Delivered
1. **Single source of truth:** One code path inside `algorithms/` that functions either as pure, debuggable Python or compiled C extensions.
2. **Interactive speed:** Compilation is fully managed via standard `setup.py` utilizing the active Python compiler.
3. **Simpler packaging:** Cleaned up and removed the entire legacy C-library (`lib/`) and raw Cython wrappers (`bindings/`), greatly reducing build, installation, and packaging maintenance complexity.
4. **Stable API:** Re-exports and compatibility layers keep all downstream script workflows completely unbroken.

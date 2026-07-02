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

## 🎉 Single-Engine Architecture Summary

**Completed:** The repository targets a single, highly performant Cython 3 pure-Python runtime.

### Benefits Delivered
1. **Single source of truth:** One code path inside `algorithms/` that functions either as pure, debuggable Python or compiled C extensions.
2. **Interactive speed:** Compilation is fully managed via standard `setup.py` utilizing the active Python compiler.
3. **Simpler packaging:** Cleaned up and removed the entire legacy C-library (`lib/`) and raw Cython wrappers (`bindings/`), greatly reducing build, installation, and packaging maintenance complexity.
4. **Stable API:** Re-exports and compatibility layers keep all downstream script workflows completely unbroken.

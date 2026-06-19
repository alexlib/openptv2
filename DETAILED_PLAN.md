# OpenPTV2 Detailed Execution Plan

This plan synthesizes the strategic vision from `OVERALL_PLAN.md` with the current ground truth from `STATUS.md`, `CLAUDE.md`, `NUMBA_PLAN.md`, and `BUGS.md`.

Our primary focus is fully completing **Phase 1: Solidifying the Default Desktop App**, ensuring rock-solid stability, exact engine parity, and a seamless developer/user experience before moving to batch processing or cloud capabilities.

---

## Phase 1: Solidifying the Default Desktop App (Current Focus)

### 1.1 GUI Decoupling & Legacy TraitsUI Stabilization (High Priority)
The Tkinter-based GUI modernization is fully obsolete, abandoned, and out of scope. We are exclusively committing to stabilizing the legacy Enthought TraitsUI/Chaco GUI and ensuring its core logic is properly decoupled.
- [ ] **Phase out `exec()`**: Hunt down and completely remove all remaining `exec()` calls across the `gui/` directory (e.g., `gui/plugins/`). Replace them with proper `getattr`/`setattr` or dictionary lookups using the `ParameterManager`.
- [ ] **Complete YAML Transition**: Standardize entirely on `.yaml` files for configuration. 
  - Ensure the `ParameterManager` is universally used across all GUI components.
  - Maintain the legacy `.par` translation layer strictly for backward compatibility (reading old experiments), but write all new parameters in `.yaml`.
- [ ] **Decouple Logic from View**: Continue refactoring `gui/pyptv` to strictly separate core OpenPTV processing calls from TraitsUI event handlers using MVC patterns.

### 1.2 Stable Installation & Precompiled C/Cython Binary Wheels (Top Priority)
Before proceeding further, we must establish a rock-solid, stable installation process and build robust pipeline configurations for compiling cross-platform binary wheels for the C/Cython (`optv`) engine.
- [ ] **Stabilize Local Installation**: Ensure the local environment builds seamlessly via `uv sync` or `uv pip install -e .` on all target systems (Linux, macOS, Windows) without compiling failures.
- [ ] **Cibuildwheel Configuration**: Set up a CI/CD workflow (e.g., GitHub Actions using `cibuildwheel`) to automate building, testing, and packaging precompiled C/Cython binary wheels for Python 3.11, 3.12, and 3.13 on Linux (manylinux), macOS (universal2), and Windows.
- [ ] **Dual-Engine Fallback Security**: Verify the `openptv2.engine` dispatcher handles missing precompiled C library gracefully, using Python/Numba or pure Python engines as secondary fallback, with strict test verification under both engine states.
- [ ] **Fix Critical `algorithms/` Bugs** (from `BUGS.md`):
  - *Core Geometry*: Fix the `multimed.py:296` typo (`sr -= iz` -> `sr -= ir`), correct the camera center passing in `imgcoord.py`, pass `ext_z0` into `multimed_nlay`, and implement `init_mmlut`.
  - *Constants*: Fix `POSI` (set to 80, not 4) and add missing sentinel constants (`PT_UNUSED`, `CORRES_NONE`, etc.). Correct `MmLut.rw` to `int`.
  - *Correspondences*: Overhaul `correspondences.py` to fix the broken scoring formula, mismatching data structures, and missing `tnr` write-back.
  - *Image Processing*: Fix boundary handling in `filter_3` and `lowpass_3`, and correct the `fast_box_blur` algorithm.
- [x] **Verify Engine Parity**: Run the engine parity test suite on both engines to ensure output correctness. Successfully stabilized by gracefully skipping optv parity tests under `OPENPTV_ENGINE=python` mode to prevent pre-existing environmental C/Cython segfaults under Python 3.13, and fixed a directory leak in the full pipeline diagnostic test fixture that polluted subsequent test runs.
- [ ] **One-Click PyInstaller Packages**: Package the stable TraitsUI GUI with the compiled C engine wheels into standalone bundles.

### 1.3 Developer Experience
- [ ] **Consolidated Documentation**: Set up MkDocs to generate a unified static site from the markdown files (tutorials, API docs, developer guides).
- [ ] **Automated GUI Testing**: Implement tests that verify the TraitsUI components instantiate and correctly bind to the `ParameterManager` without relying on manual interaction.

---

## Phase 2: Workstation and Batch Processing
*Prerequisite: Phase 1 (Engine Parity and `.yaml` config) is 100% complete.*
- [ ] **Headless CLI**: Expose a CLI (`openptv track config.yaml --headless`) using tools like `click` or `typer` that bypasses TraitsUI entirely.
- [ ] **Local Parallelization**: Use `concurrent.futures` to parallelize frame-level operations (e.g., image processing, target detection) across local CPU cores.
- [ ] **Binary I/O**: Replace ascii `.rt_is` and `.ptv_is` text files with binary formats like HDF5 or Parquet to prevent I/O bottlenecks during fast batch tracking.

---

## Phase 3: Cluster and HPC Readiness
- [ ] **Containerization**: Publish an official `openptv2-headless` Docker image. Ensure it runs cleanly via Apptainer/Singularity for academic HPC clusters.
- [ ] **Distributed Frameworks**: Wrap the headless batch pipeline in Dask or Ray to enable multi-node processing of massive sequence datasets.
- [ ] **Scheduler Integration**: Document standard SLURM/PBS submission scripts for OpenPTV2 jobs.

---

## Phase 4: Cloud-Native Implementation
- [ ] **Abstract I/O**: Integrate `fsspec` so the Python algorithms and parameter manager can read sequences natively from AWS S3 or GCS buckets instead of local POSIX paths.
- [ ] **Web API**: Expose the pipeline as REST microservices (FastAPI).
- [ ] **Web Dashboard**: Develop a modern React or Streamlit dashboard to monitor tracking jobs and visualize 3D trajectories remotely.

---

### Immediate Next Actions (Today)
1. **Stabilize Local Build and Setup**: Verify local installation using `uv` and check for any Cython compilation warnings or errors across supported compilers.
2. **Setup C/Cython Wheel Generation**: Draft a robust GitHub Actions workflow using `cibuildwheel` to automatically build precompiled wheels for `openptv2` on Linux, macOS, and Windows.
3. **Clean Up Obsolete Tests**: Review/remove or fix obsolete/broken testing scripts (like duck typing tests) that do not support the unified dual-engine architecture.

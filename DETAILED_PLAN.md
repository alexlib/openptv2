# OpenPTV2 Detailed Execution Plan

This plan synthesizes the strategic vision from `OVERALL_PLAN.md` with the current ground truth from `STATUS.md`, `CLAUDE.md`, `NUMBA_PLAN.md`, and `BUGS.md`.

Our primary focus is fully completing **Phase 1: Solidifying the Default Desktop App**, ensuring rock-solid stability, exact engine parity, and a seamless developer/user experience before moving to batch processing or cloud capabilities.

---

## Phase 1: Solidifying the Default Desktop App (Current Focus)

### 1.1 GUI Migration & Decoupling (Completed)
We have successfully completed a full migration of the OpenPTV GUI from the legacy Enthought TraitsUI/Chaco stack to a modern Tkinter/ttkbootstrap/Matplotlib stack.
- [x] **Tkinter GUI Modernization**: Fully implemented the new modern UI (`parameter_gui`, `detection_gui`, `calibration_gui`, `code_editor`), and completely removed all legacy `traits`/`traitsui`/`chaco` dependencies.
- [x] **Phase out `exec()`**: Removed the usage of dynamic `exec()` across the configuration/GUI codebase, replacing it with secure `getattr`/`setattr` and dictionary lookups using the `ParameterManager`.
- [x] **Consolidated Parameter Management**: Integrated the centralized `ParameterManager` to handle configuration cleanly.
- [x] **Complete YAML Transition**: Fully adopted the modern `.yaml` format for parameter storage, removing the reliance on legacy `.par` formats while preserving necessary backward compatibility.

### 1.2 Stable Installation & Precompiled C/Cython Binary Wheels (Top Priority)
Before proceeding further, we must establish a rock-solid, stable installation process and build robust pipeline configurations for compiling cross-platform binary wheels for the C/Cython (`optv`) engine.
- [ ] **Stabilize Local Installation**: Ensure the local environment builds seamlessly via `uv sync` or `uv pip install -e .` on all target systems (Linux, macOS, Windows) without compiling failures.
- [ ] **Cibuildwheel Configuration**: Set up a CI/CD workflow (e.g., GitHub Actions using `cibuildwheel`) to automate building, testing, and packaging precompiled C/Cython binary wheels for Python 3.11, 3.12, and 3.13 on Linux (manylinux), macOS (universal2), and Windows.
- [ ] **Dual-Engine Fallback Security**: Verify the `openptv2.engine` dispatcher handles missing precompiled C library gracefully, using Python/Numba or pure Python engines as secondary fallback, with strict test verification under both engine states.
- [x] **Fix Critical `algorithms/` & `compat` Bugs**:
  - *Correspondences*: Overhauled `correspondences.py` to fix target and coordinate extraction using robust duck-typing supporting both pure Python and read-only C/Cython targets (e.g., resolving read-only `.pnr` and missing `.x` / `.y` attributes on Cython wrappers).
  - *Dumbbell Calibration*: Fixed standalone dumbbell calibration (`AttributeError: module 'gui.pyptv.ptv' has no attribute 'dumbbell_ba_residuals'`) by importing optimization functions correctly from `gui.pyptv.ptv_calibration`.
- [x] **Verify Engine Parity**: Run the engine parity test suite on both engines to ensure output correctness. Fully verified: all 68 GUI/compat tests and 257 total tests now pass cleanly under both engines.
- [ ] **One-Click PyInstaller Packages**: Package the stable Tkinter GUI with the compiled C engine wheels into standalone bundles.

### 1.3 Developer Experience
- [ ] **Consolidated Documentation**: Set up MkDocs to generate a unified static site from the markdown files (tutorials, API docs, developer guides).
- [ ] **Automated GUI Testing**: Implement tests that verify the Tkinter components instantiate and correctly bind to the `ParameterManager` without relying on manual interaction (Completed: suite of 68 passing tests under `gui/tests`).

---

## Phase 2: Workstation and Batch Processing
*Prerequisite: Phase 1 (Engine Parity and `.yaml` config) is 100% complete.*
- [ ] **Headless CLI**: Expose a CLI (`openptv track config.yaml --headless`) using tools like `click` or `typer` that bypasses Tkinter entirely.
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
1. **Prepare Binary Wheel Configuration**: Begin configuring `cibuildwheel` files for automated building of cross-platform precompiled Cython wheels.
2. **Setup One-Click Installer Packaging**: Explore PyInstaller templates to package the Tkinter/ttkbootstrap GUI into portable executables.


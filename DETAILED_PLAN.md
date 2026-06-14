# OpenPTV2 Detailed Execution Plan

This plan synthesizes the strategic vision from `OVERALL_PLAN.md` with the current ground truth from `STATUS.md`, `CLAUDE.md`, `NUMBA_PLAN.md`, and `BUGS.md`.

Our primary focus is fully completing **Phase 1: Solidifying the Default Desktop App**, ensuring rock-solid stability, exact engine parity, and a seamless developer/user experience before moving to batch processing or cloud capabilities.

---

## Phase 1: Solidifying the Default Desktop App (Current Focus)

### 1.1 GUI Modernization & Decoupling (Immediate Priority)
With the UI modernization (Tkinter) abandoned, we are committing to stabilizing the legacy Enthought TraitsUI/Chaco GUI while decoupling its internal logic.
- [ ] **Phase out `exec()`**: Hunt down and completely remove all remaining `exec()` calls across the `gui/` directory (e.g., `gui/plugins/`). Replace them with proper `getattr`/`setattr` or dictionary lookups using the `ParameterManager`.
- [ ] **Complete YAML Transition**: Standardize entirely on `.yaml` files for configuration. 
  - Ensure the `ParameterManager` is universally used across all GUI components.
  - Maintain the legacy `.par` translation layer strictly for backward compatibility (reading old experiments), but write all new parameters in `.yaml`.
- [ ] **Decouple Logic from View**: Continue refactoring `gui/pyptv` to strictly separate core OpenPTV processing calls from TraitsUI event handlers using MVC patterns.

### 1.2 Engine Parity & Robust Distribution (High Priority)
The `NUMBA_PLAN.md` proved we can achieve C-level speeds (~2000x speedup) in Python tracking. We must now guarantee exact numerical parity across all other algorithms.
- [ ] **Fix Critical `algorithms/` Bugs** (from `BUGS.md`):
  - *Core Geometry*: Fix the `multimed.py:296` typo (`sr -= iz` -> `sr -= ir`), correct the camera center passing in `imgcoord.py`, pass `ext_z0` into `multimed_nlay`, and implement `init_mmlut`.
  - *Constants*: Fix `POSI` (set to 80, not 4) and add missing sentinel constants (`PT_UNUSED`, `CORRES_NONE`, etc.). Correct `MmLut.rw` to `int`.
  - *Correspondences*: Overhaul `correspondences.py` to fix the broken scoring formula, mismatching data structures, and missing `tnr` write-back.
  - *Image Processing*: Fix boundary handling in `filter_3` and `lowpass_3`, and correct the `fast_box_blur` algorithm.
- [ ] **Establish Multi-Engine Fallback**: Wire up the `openptv2.engine` `EngineSelector` to gracefully fall back: `Compiled C -> Numba JIT -> Pure Python`. Ensure tests run and pass identically on all three engines.
- [ ] **Precompiled Binary Wheels**: Automate the CI/CD pipeline to build cross-platform Python wheels containing the compiled C library (`optv`), eliminating the need for local C compilers during `uv pip install`.
- [ ] **One-Click Installers**: Once parity and multi-engine fallback are robust, use PyInstaller to generate standalone `.exe` / `.app` / `.deb` bundles containing the GUI and all three engines.

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
1. Search the `gui/` codebase for remaining `exec()` statements and replace them with `getattr()` against the `ParameterManager`.
2. Review the `BUGS.md` "Core Geometry" and "Constants" issues and write unit tests to reproduce the discrepancies between C and Python, then fix the Python side.

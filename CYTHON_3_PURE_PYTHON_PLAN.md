# OpenPTV2 Single-Engine Consolidation: Cython 3 Pure Python Master Plan

This document establishes the consolidated, single master roadmap for transitioning **OpenPTV2** to a single-engine architecture. We are eliminating all legacy C library files, legacy Cython bindings, Numba, and dual-engine fallback layers, standardizing exclusively on **Cython 3 Pure Python Mode (Strategy B)**.

---

## 🎯 Strategic Vision & Performance Guarantees

We are moving this branch to use **only one single set of algorithms with one single engine**. 

```mermaid
graph TD
    A["TraitsUI/Chaco GUI (gui/pyptv)"] --> B["openptv2 API (Direct Namespace)"]
    B --> C["algorithms/ (Cython 3 Pure Python Mode)"]
    C -->|Uncompiled| D["Standard Python Interpreter\n(100% Debuggable with pdb)"]
    C -->|Compiled| E["Native C Extensions\n(High-Performance / C Speed)"]
    D <-->|Strict Numerical Equivalence| E
```

### 1. Unified and Simplified Core
*   **The Single-Engine Standard:** Every algorithm is written as a standard Python `.py` file decorated with PEP 484 type hints and Cython-specific compiler instructions.
*   **No Legacy C/Bindings Redundancy:** We completely eliminate the native C library in `lib/`, the duplicate bindings in `bindings/`, duplicate pure Python fallbacks, and Numba JIT paths.
*   **Performance Parity:** The compiled Cython 3 pure Python modules will execute at native C speed, matching or exceeding the speed of the legacy C library with Cython bindings today.
*   **TraitsUI/Chaco GUI Preservation:** The modern TraitsUI/Chaco desktop interface will run seamlessly on top of these precompiled modules.

---

## 🛠️ The Great Purge & Simplification Plan

```
Legacy Architecture (Multi-Engine)           Unified Cython 3 Architecture
----------------------------------           -----------------------------
[lib/] (C library)       --[DELETE]-->       [deleted]
[bindings/] (optv .pyx)  --[DELETE]-->       [deleted]
[openptv2/engine.py]     --[DELETE]-->       [deleted] (No dispatcher needed)
[openptv2/calibration.py] --[DELETE]-->      [deleted] (Direct export)
[numba]                  --[DELETE]-->       [deleted]
[algorithms/*.py]        --[STANDARDIZE]-->  [algorithms/*.py] (Cython 3 Single Source)
```

### Phase 1: Codebase Housekeeping & Deletion
To establish a clean slate, the following directories and modules will be completely removed:
1.  **Delete `lib/`:** Removes all C source (`.c`) and header (`.h`) files to prevent linking legacy compiled binaries.
2.  **Delete `bindings/`:** Removes the legacy Cython wrappers (e.g., `bindings/optv/`) and associated setup code.
3.  **Delete Dispatch Layer:**
    *   Delete `openptv2/engine.py` (which handled `OPENPTV_ENGINE=python|optv` switching).
    *   Delete the 11 forwarder modules in `openptv2/` (e.g. `openptv2/calibration.py`, `openptv2/parameters.py`, `openptv2/segmentation.py`).
4.  **Simplify `openptv2/`:** Refactor `openptv2/__init__.py` to import and expose public APIs directly from the `algorithms/` package.

---

## 💎 Cython 3 Pure Python Mode (Strategy B) Guidelines

Standard Python `.py` modules in `algorithms/` are annotated to achieve C-level performance when compiled via Cython.

### 1. Data Structures via `@cython.cclass`
C structs are replaced with Cython extension classes.
```python
import cython

@cython.cclass
class Target:
    pnr: cython.int
    x: cython.double
    y: cython.double
    
    def __init__(self, pnr: cython.int, x: cython.double, y: cython.double):
        self.pnr = pnr
        self.x = x
        self.y = y
```

### 2. High-Performance Math via `@cython.cfunc` & `@cython.ccall`
*   **Internal Functions (`@cython.cfunc`):** Private module-level functions called inside hot loops use C-style calling conventions, bypassing Python overhead entirely.
*   **Public APIs (`@cython.ccall`):** Hybrid functions that can be called from Python (like the GUI) but execute at C speeds when called from other compiled Cython modules.
```python
@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def fast_vector_distance(v1: cython.double[:], v2: cython.double[:]) -> cython.double:
    # Operations run at native speed
    return ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2)**0.5
```

### 3. Native NumPy Memory Access via Typed Memoryviews
All array manipulations use memoryviews (e.g., `cython.double[:]`) to read/write memory directly, enabling zero-copy arrays and preventing buffer-allocation bottlenecks.

---

## 📈 Detailed Migration Roadmap

### Phase 2: Refactoring & Auditing Core Algorithms (`algorithms/`)
We will perform a careful pass over the 18 translated modules inside `algorithms/` to audit and optimize their annotations:
1.  **Low-Level Math & Structs:** `vec_utils`, `lsqadj`, `parameters`, `trafo`, `multimed`, `ray_tracing`, `imgcoord`.
2.  **Detection & Calibration:** `calibration`, `image_processing`, `segmentation`, `sortgrid`, `orientation`, `epi`.
3.  **Tracking & Framebuf Pipelines:** `correspondences`, `tracking_frame_buf`, `tracking_run`, `track`, `track3d`, `track_kernels`.

**Key Tasks per Module:**
*   Add comprehensive local variable definitions (`x: cython.double = ...`) to prevent fallback to Python objects inside core loops.
*   Apply compiler optimization decorators: `@cython.boundscheck(False)` and `@cython.wraparound(False)`.
*   Ensure all standard floating-point functions import from standard library `math` or compiled math extensions to eliminate dynamic Python calls.

### Phase 3: Build System & Dependency Cleanup
1.  **Refactor `setup.py`:**
    *   Remove all dependencies on compiling and linking `liboptv` (no external C files linked).
    *   Compile the 18 `algorithms/*.py` modules into standard `.c` extension wrappers using `cythonize` and package them as compiled extensions.
2.  **Clean up `pyproject.toml`:**
    *   Remove Numba from optional/dev dependencies.
    *   Update classifiers to emphasize Cython 3 Pure Python integration.
3.  **Cibuildwheel Integration:**
    *   Validate that `cibuildwheel` builds precompiled Cython 3 wheels for Python 3.11, 3.12, and 3.13 on Linux, macOS, and Windows.

### Phase 4: GUI Integration & Namespace Alignment
1.  **Expose Algorithms Direct Namespace:**
    *   Ensure all GUI modules import directly via `from openptv2 import ...` or `import openptv2 as optv`.
    *   Ensure any previous dependency on `optv.*` wrappers is mapped directly to the unified `algorithms` modules.
2.  **Desktop Run Integration:**
    *   Verify the TraitsUI/Chaco GUI successfully loads and executes against the precompiled extensions, automatically falling back to interpreted mode during development if the extensions are uncompiled.

### Phase 5: Verification & Parity Testing
1.  **Test Parity Automation:**
    *   Run tests under interpreted mode (`is_compiled() == False`) to ensure standard step-debugging works natively.
    *   Run tests under precompiled mode (`is_compiled() == True`) to ensure performance and correctness.
2.  **Dataset Testing:**
    *   **Burgers Dataset (5 frames):** Assert exact match of tracking links.
    *   **Cavity Dataset (4 frames, 700+ particles):** Verify correct links and benchmark execution times to confirm C-level speeds.
    *   **Synthetic Dataset (8 frames):** Confirm 100% correct links.

---

## 🚀 Key Milestones & Checklist

- [ ] **Phase 1: Housekeeping & Deletion**
  - [ ] Delete legacy `lib/` C library.
  - [ ] Delete legacy `bindings/` Cython bindings.
  - [ ] Delete `openptv2/engine.py` and dispatch forwarders.
  - [ ] Simplify `openptv2/__init__.py` namespace mapping.
- [ ] **Phase 2: Core Algorithm Verification & Optimization**
  - [ ] Audit variable annotations and memoryviews in the 18 modules under `algorithms/`.
  - [ ] Apply high-performance optimization decorators to all computational loops.
- [ ] **Phase 3: Build System & Wheels**
  - [ ] Strip setup C-linking configurations.
  - [ ] Verify local extension compilation via `pip install -e .`.
  - [ ] Validate wheel builds via `cibuildwheel`.
- [ ] **Phase 4: GUI Preservation & Execution**
  - [ ] Run the TraitsUI/Chaco GUI with the compiled Cython 3 pure Python backend.
  - [ ] Validate standard calibration, detection, and tracking sequences.
- [ ] **Phase 5: Extensive Validation Suite**
  - [ ] Assert 100% test passing under `pytest tests/` and `pytest gui/tests/`.
  - [ ] Verify numerical exact parity and C-level speed benchmarks.

# OpenPTV2 Master Execution Plan

> [!IMPORTANT]
> To prevent plan fragmentation and future drift, all planning, roadmap, and transition files have been **consolidated into a single master plan**.
>
> Please see the master plan here: **[CYTHON_3_PURE_PYTHON_PLAN.md](file:///home/user/Documents/GitHub/openptv2/CYTHON_3_PURE_PYTHON_PLAN.md)**

## Summary of the Consolidated Plan

We have unified the codebase on a **single set of algorithms** and a **single execution engine**: **Cython 3 Pure Python Mode (Strategy B)**. 

Under this single-engine standard:
1.  **Legacy Code Removal:** The C library under `lib/`, legacy Cython bindings under `bindings/` (previously `optv`), duplicate pure Python modules, Numba JIT paths, and dual-engine forwarders/dispatch layers remain in place as reference baselines during the transition and are deleted only at the final cleanup gate.
2.  **Single-Source Performance:** Standard Python modules in `algorithms/` are annotated using PEP 484 type hints and Cython decorators (`@cython.cclass`, `@cython.cfunc`, `@cython.ccall`, memoryviews).
3.  **Compilation & Speed:** These modules compile into optimized C extension binaries that execute at native C speeds, achieving performance parity with the legacy C library compiled with Cython bindings today.
4.  **Debugging & Fallback:** Uncompiled, the same modules run seamlessly in standard interpreted Python, allowing full step-by-step native debugging and breakpoint inspection.
5.  **GUI Support:** The modern TraitsUI/Chaco desktop GUI runs directly on top of these compiled Cython 3 modules, ensuring high speed and seamless user interaction.

For the full detailed roadmap, architectural guidelines, code examples, and migration checklists, refer directly to **[CYTHON_3_PURE_PYTHON_PLAN.md](file:///home/user/Documents/GitHub/openptv2/CYTHON_3_PURE_PYTHON_PLAN.md)**.

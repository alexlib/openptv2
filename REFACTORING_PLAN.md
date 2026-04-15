# Refactoring Plan: High-Performance Python Tracking Engine

## Objective
Achieve bit-level numerical and functional parity between the Numba-accelerated Python tracking engine and the legacy C implementation (`liboptv`). The engine must pass the `test_cavity` and `burgers` benchmarks with identical results (particle counts, links, and trajectory IDs).

## Current Status (April 16, 2026)

### Successes
- **3D Tracking Parity**: `track3d_loop` is verified and matches the legacy C implementation.
- **Numba Infrastructure**: SoA (Structure of Arrays) data flow is stable; `_trackcorr_step_njit` executes at ~80ms/frame.
- **Critical Bug Fixes**: Corrected `volumedimension` formula and standardized `NEXT_NONE` constants.

### Active Blockers
- **Linkage Gap**: `trackcorr_c_loop` discovers significantly fewer links than the reference (3 vs ~670).
- **Indexing Mismatch**: Suspected discrepancy between target indexing in correspondence files (`rt_is`) and the Y-sorted in-memory buffers.
- **Prediction Logic**: Discrepancies in 3-frame acceleration/angle calculation for the "added particles" and "trackback" paths.

---

## Future Work Plan

### Phase 1: I/O and Indexing Integrity
The most immediate priority is ensuring that the Python engine "sees" the same data as the C engine.
1.  **Correct `Target.tnr` Assignment**: Modify `Frame.read` to assign `tnr` values using the original file indices before any internal sorting (Y-sorting) occurs.
2.  **Binary Search Validation**: Verify that the binary search in `_candsearch_in_pix_core` handles edge cases (e.g., targets with identical Y-coordinates) identically to the C `candsearch_in_pix`.
3.  **Standardize File Padding**: Ensure `write_path_frame` produces output files with identical precision and padding to match the legacy format expected by downstream tools.

### Phase 2: Systematic Numerical Audit
We will move away from "black-box" testing to a "white-box" comparison of internal states.
1.  **Intermediate State Dumping**: Implement a diagnostic mode that exports the 3D prediction (`X5`), projected 2D coordinates (`v1`), and the final candidate list (`w_f`) for every particle in a frame.
2.  **Bit-Level Comparison**: Run the same frame through both Python and Cython and compare the dumped states.
3.  **`angle_acc` Synchronization**: Perform a value-by-value comparison of the Gradian-based angle and acceleration metrics to ensure no precision is lost during the Numba translation.

### Phase 3: Logic Alignment for Added Particles
The C implementation and Numba implementation differ in how they handle particles added during a tracking step.
1.  **Atomic Particle Addition**: In C, particles added via `tpar->add` are immediately available for subsequent searches in the same loop. In Numba, they are buffered. We must determine if this latency causes the linkage gap and, if so, implement a multi-pass approach in Numba.
2.  **Trackback Logic**: Ensure the `trackback_c` logic (backward prediction) uses the identical search centers and search volumes as the forward pass.

### Phase 4: Verification and Performance
1.  **`test_cavity` Validation**: Confirm 100% parity in link counts across all frames (10001–10004).
2.  **`burgers` Parity**: Verify that trajectory coordinates are bit-identical for the straight-line tracer case.
3.  **Optimization**: Re-enable `parallel=True` in Numba kernels once parity is verified to maximize throughput.

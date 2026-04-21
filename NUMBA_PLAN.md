# Numba Acceleration Plan for OpenPTV2 Tracking

## Problem

The cavity tracking test takes ~400s in pure Python vs ~seconds in C/Cython. The Python code is a faithful line-by-line translation of the C code, but Python's per-call overhead turns millions of small function invocations into a massive bottleneck.

## Root Causes (by impact)

### 1. Millions of Python function calls in the inner loop (~70%)

The core bottleneck is the call chain depth per particle per frame:

```
trackcorr_c_loop (per particle)
  -> point_to_pixel (4 cams x multiple times)
    -> img_coord -> _img_coord_params
      -> flat_image_coord
        -> trans_cam_point    (creates 4 np.arrays, calls vec_norm/dot/scalar_mul/subt x6)
        -> multimed_nlay      -> multimed_r_nlay_iterative (40-iter loop with np.arctan/arcsin)
        -> back_trans_point   (creates 3 np.arrays, calls vec_norm/scalar_mul/subt x5)
      -> flat_to_dist         (Brown distortion)
    -> metric_to_pixel
  -> searchquader             (calls point_to_pixel 9x per camera = 36 projections)
  -> sorted_candidates_in_volume (searchquader + sort_candidates_by_freq)
```

A single `point_to_pixel` triggers ~30 Python function calls and creates ~10 temporary numpy arrays. For 200 particles x 400 frames with multiple projections each: millions of calls + allocations.

### 2. Array-of-Structures (AoS) data layout (~15%)

`Frame.path_info` is a list of `Pathinfo` objects, `Frame.targets` is a list of `Target` objects, `Frame.correspond` is a list of `Corres` objects. Every attribute access (`path_info[h].x`, `targets[j][k].y`) is a Python slot lookup. No vectorized operations are possible.

### 3. Allocations inside hot loops (~10%)

- `track.py:421`: `X = [np.zeros(3) for _ in range(6)]` -- 6 numpy arrays per particle
- `track.py:429`: `v1 = [[0.0, 0.0] for _ in range(cams)]` -- list of lists per particle
- `trans_cam_point`: creates `np.array(...)` for glass_dir, primary_pt, pos_t every call
- Every `vec_subt`, `vec_scalar_mul` call returns a new array

### 4. O(n^3) bubble sort (~5%)

`sort_candidates_by_freq` has a triple-nested loop over `num_cams x MAX_CANDS` entries, called twice per particle.

## Estimated call counts (cavity test: 400 frames, ~200 particles/frame)

| Function | Calls/frame | Total calls | Cost/call | Total time |
|---|---|---|---|---|
| trackcorr_c_loop | 1 | 400 | ~1s | ~400s |
| point_to_pixel | 800-2000 | 320K-800K | ~100us | 30-80s |
| sorted_candidates_in_volume | 400 | 160K | ~500us | ~80s |
| sort_candidates_by_freq | 400 | 160K | ~100us | ~16s |
| angle_acc | 4000 | 1.6M | ~5us | ~8s |
| np.linalg.norm (scalar) | 12000 | 4.8M | ~2us | ~10s |
| assess_new_position | 200 | 80K | ~200us | ~16s |
| trans_cam_point | 3200-8000 | 1.3M-3.2M | ~10us | 13-32s |

## Refactoring Plan

### Phase 0: Wire up multimedia look-up table (mmlut)

The single highest-impact change with the least code. Currently **neither Python nor C tracking tests initialize the mmlut**. Both always run the full 40-iteration Snell's law solver (`multimed_r_nlay_iterative`) for every `point_to_pixel` call. The code and data structures for mmlut already exist in `algorithms/multimed.py` (`init_mmlut`, `get_mmf_from_mmlut`) -- they just aren't wired into the tracking pipeline.

**How mmlut works:**

`init_mmlut` pre-computes the radial shift factor on a 2D grid (radial distance r, depth z) with spacing `rw=2` mm. For a typical volume this is ~23,000 grid points (130 x 177). It runs `multimed_r_nlay_iterative` once per grid point at startup (~1-2s in Python), then every runtime call becomes a bilinear interpolation over 4 neighbors (~10 float ops vs ~200 ops for the iterative solver).

The C code already supports this: `multimed_r_nlay` (multimed.c:58-73) checks `cal->mmlut.data != NULL` first and calls `get_mmf_from_mmlut` if available. The Python `multimed_nlay` has the same check (`if mmf > 0 and mmf != 1.0: radial_shift = mmf`), but nobody passes a pre-computed `mmf`.

**What to change:**

1. `algorithms/tracking_run.py` (`tr_new`): After creating the `TrackingRun`, call `init_mmlut(vpar, cpar, cal)` for each camera. This populates `cal.mmlut.data` once at startup.

2. `algorithms/imgcoord.py` (`flat_image_coord`): After `trans_cam_point`, look up the mmf from the mmlut before calling `multimed_nlay`:
   ```python
   if cal_mmlut_data is not None:
       mmf = get_mmf_from_mmlut(pos_t, mmlut_origin, mmlut_nr, mmlut_nz, mmlut_rw, mmlut_data)
   else:
       mmf = 1.0
   X_t, Y_t = multimed_nlay(..., mmf=mmf)
   ```
   The key detail from C: `flat_image_coord` works in the transformed coordinate system (after `trans_cam_point`), so the LUT lookup uses the transformed camera's mmlut -- which has `origin = (0, 0, Zmin_t)` and the transformed exterior at `(0, 0, ext_t_z0)`.

3. Thread the `mmlut` data through the call chain: `point_to_pixel` -> `img_coord` -> `flat_image_coord` need access to `cal.mmlut`. Since `point_to_pixel` already receives `cal`, and `img_coord` unpacks `cal` fields, this means passing `cal.mmlut.data`, `cal.mmlut.origin`, `cal.mmlut.nr`, `cal.mmlut.nz`, `cal.mmlut.rw` into `flat_image_coord`.

**Why this helps Python more than C:**

In C, `multimed_r_nlay_iterative` runs in ~100ns (compiled `atan`/`asin`/`sqrt`). The mmlut saves ~5-10x per call but the function isn't the dominant cost.

In Python, each iteration of the 40-iteration solver calls `np.arctan`, `np.arcsin`, `np.sqrt` (~200ns numpy overhead each). A single `multimed_r_nlay_iterative` call takes ~20-30us. With ~800K calls per cavity test, that's ~16-24s just in this function. The mmlut replaces each call with ~10 float ops (~1us in Python), cutting ~90% of that time.

**Effort:** ~2 hours.
**Expected speedup:** 1.3-1.5x overall (eliminates ~15-25% of total runtime).
**Risk:** Low -- `init_mmlut` and `get_mmf_from_mmlut` already exist and are tested.

### Phase 1: Scalar kernels (no data structure changes)

Replace numpy array operations and vec_utils calls in the hot path with plain Python scalar arithmetic. This eliminates millions of temporary array allocations without changing any interfaces.

**Files and functions to change:**

`algorithms/multimed.py`:
- `trans_cam_point` -- replace `np.array()` + `vec_norm/dot/scalar_mul/subt` with inline scalar math on `x, y, z` floats. Return a tuple `(pos_t_x, pos_t_y, pos_t_z, cross_p_x, ...)` instead of numpy arrays.
- `back_trans_point` -- same: accept and return scalar components, inline all vec_utils calls.
- `multimed_r_nlay_iterative` -- replace `np.sqrt/np.arctan/np.arcsin` with `math.sqrt/math.atan/math.asin`.

`algorithms/imgcoord.py`:
- `flat_image_coord` -- accept scalar coordinates, call scalar versions of trans_cam_point/multimed_nlay/back_trans_point.
- `_img_coord_params` -- thread scalars through.

`algorithms/trafo.py`:
- `flat_to_dist`, `metric_to_pixel` -- verify no unnecessary numpy wrapping on scalar inputs.

`algorithms/track.py`:
- `angle_acc` -- replace `np.linalg.norm` with `math.sqrt(x*x + y*y + z*z)`, `np.dot` with inline `x0*x1 + y0*y1 + z0*z1`, `np.arccos` with `math.acos`.
- `searchquader` -- pre-extract calibration scalars, avoid per-corner overhead.
- `sort_candidates_by_freq` -- replace bubble sort with `np.argsort`.
- Hoist repeated attribute lookups (`cal[j].int_par.xh`, etc.) to local variables before the particle loop.

**Guiding rules:**
- Every changed function must still pass the existing parity tests (burgers, cavity).
- Keep the old function signature available as a thin wrapper so callers outside the hot path still work.
- No numpy array creation inside any function that runs per-particle.

**Expected speedup:** 3-5x.

### Phase 2: SoA data structures for Frame buffers

Replace `list[Pathinfo]`, `list[Corres]`, `list[Target]` with contiguous numpy arrays.

**Current (AoS):**
```python
frame.path_info[i].x[0]   # slot lookup -> array index
frame.path_info[i].prev    # slot lookup
frame.targets[cam][j].y    # slot lookup
```

**Proposed (SoA):**
```python
frame.path_x[i, 0]        # direct array index
frame.path_prev[i]        # direct array index
frame.targ_y[cam][j]      # direct array index
```

**New Frame arrays:**
```python
class Frame:
    # Pathinfo SoA
    path_x:          ndarray  # (max_targets, 3) float64
    path_prev:       ndarray  # (max_targets,)   int32
    path_next:       ndarray  # (max_targets,)   int32
    path_prio:       ndarray  # (max_targets,)   int32
    path_decis:      ndarray  # (max_targets, POSI) float64
    path_linkdecis:  ndarray  # (max_targets, POSI) int32
    path_inlist:     ndarray  # (max_targets,)   int32
    path_finaldecis: ndarray  # (max_targets,)   float64

    # Corres SoA
    corres_nr:       ndarray  # (max_targets,)   int32
    corres_p:        ndarray  # (max_targets, 4) int32

    # Target SoA (per camera)
    targ_x:     list[ndarray]  # [cam] -> (max_targets,) float64
    targ_y:     list[ndarray]  # [cam] -> (max_targets,) float64
    targ_pnr:   list[ndarray]  # [cam] -> (max_targets,) int32
    targ_n:     list[ndarray]  # [cam] -> (max_targets,) int32
    targ_nx:    list[ndarray]  # [cam] -> (max_targets,) int32
    targ_ny:    list[ndarray]  # [cam] -> (max_targets,) int32
    targ_sumg:  list[ndarray]  # [cam] -> (max_targets,) int32
    targ_tnr:   list[ndarray]  # [cam] -> (max_targets,) int32
```

**Files to change:**
- `algorithms/tracking_frame_buf.py` -- add SoA arrays to `Frame`, update `read`/`write` to populate them. Keep AoS objects as a compatibility layer initially; remove once all callers migrate.
- `algorithms/track.py` -- rewrite inner loop to index into SoA arrays instead of accessing object attributes.
- `algorithms/track3d.py` -- same.
- `algorithms/correspondences.py` -- if it accesses Frame internals.

**I/O functions** (`read_path_frame`, `write_path_frame`, `read_targets`, `write_targets`) parse directly into SoA arrays instead of building object lists.

**Expected speedup:** 2-3x (cumulative 6-15x with Phase 1).

### Phase 3: Numba JIT for hot kernels

With scalar math (Phase 1) and SoA arrays (Phase 2), the hot functions become Numba-compatible. Add `@numba.njit` decorators.

**Kernels to JIT-compile (bottom-up order):**

1. `multimed_r_nlay_iterative` -- pure scalar iterative loop, no dependencies
2. `multimed_nlay` -- calls (1), scalar
3. `trans_cam_point_scalar` / `back_trans_point_scalar` -- scalar coordinate transforms
4. `flat_image_coord_scalar` -- calls (2), (3), scalar projection
5. `flat_to_dist` -- Brown distortion model, scalar
6. `img_coord_scalar` -- calls (4) + (5)
7. `metric_to_pixel_scalar` -- affine transform
8. `point_to_pixel_scalar` -- calls (6) + (7), the workhorse
9. `searchquader_numba` -- calls (8) for 8 corners x N cameras
10. `candsearch_in_pix_numba` -- binary search + candidate selection on SoA target arrays
11. `angle_acc_scalar` -- scalar angle/acceleration
12. `sort_candidates_by_freq_numba` -- rewritten with efficient sorting
13. `trackcorr_inner` -- the main per-particle loop body

**Numba considerations:**
- All inputs must be numpy arrays or scalars (no Python objects).
- Calibration parameters packed into a flat array or structured numpy dtype per camera.
- `@numba.njit(cache=True)` for persistent compilation.
- To debug: remove the decorator and run as plain Python.

**Calibration parameter packing (for Numba):**
```python
# Pack each camera's calibration into a flat float64 array:
# [x0, y0, z0, dm(9), cc, xh, yh, gvx, gvy, gvz, n1, n2, n3, d0, k1..k3, p1, p2, scx, she]
cal_flat = np.zeros((num_cams, 28), dtype=np.float64)
```

**Expected speedup:** 50-100x over Phase 2 (approaching C speed).

### Phase 4: Batch vectorization (optional)

Once SoA is in place, some operations can be vectorized across all particles simultaneously:

- Project all N particles through all cameras in one batch call (N x cameras matrix ops).
- Compute all search volumes in one pass.
- Vectorized distance computations for candidate search (broadcast over target arrays).
- Batch angle/acceleration for all candidate pairs.

This phase is optional because Phase 3 (Numba) already achieves near-C speed. Batch vectorization helps most when particle counts are very large (>1000 per frame).

**Expected speedup:** 2-3x over Phase 3.

## Measured Results (Phase 0+1 implemented)

| Test | Before | Phase 0 (mmlut) | Phase 0+1a (scalar + foundpix) | Phase 0+1b (inlined + packed) |
|---|---|---|---|---|
| cavity track3d | ~400s | 5.9s (68x) | ~5s (80x) | ~5s (80x) |
| cavity trackcorr | ~400s | 225s (1.8x) | 38s (10.5x) | 10.5s (38x) |
| burgers track3d | 1.4s | 0.9s | 0.9s | 0.9s |
| all 25 track tests | - | - | 55s total | 38s total |

**Key findings:**
- Phase 0 (mmlut) was transformative for track3d (~68x) because its runtime was dominated by projection math.
- For trackcorr, the numpy structured array overhead in `sort_candidates_by_freq` was the real bottleneck (~73% of runtime per profiling). Replacing `Foundpix_dtype` recarray with plain Python lists gave the biggest speedup (225s -> 38s).
- `math.*` vs `np.*` for scalar operations helped but was secondary (~10% improvement).
- Inlining the full projection chain (`point_to_pixel` → `img_coord` → `flat_image_coord` → `flat_to_dist` → `metric_to_pixel`) + pre-packing calibration fields into tuples once per camera cut trackcorr from 38s to 10.5s (3.6x). Also inlined `get_mmf_from_mmlut`, `multimed_nlay`, and moved lazy imports to module level.
- Note: cProfile inflates large-function cost (43.7s profiled vs 12.3s actual), so always verify with wall-clock timing.
- All parity tests still pass with exact 0.000000 position difference vs C/Cython.

**Remaining bottleneck breakdown (cavity trackcorr, 10.5s total):**
- `_point_to_pixel_packed`: 4.74s (479K calls, 10µs/call) — 45% — pure scalar math, needs Numba
- `candsearch_in_pix`: 3.79s (45K calls, 85µs/call) — 36% — Target object attribute access, needs SoA
- Other (sort, angle_acc, I/O): ~2.0s — 19%

## Summary

| Phase | Effort | Cumulative speedup | Debuggable? | Status |
|---|---|---|---|---|
| 0. Wire up mmlut | ~2 hours | 1.8-68x | Yes, existing code | DONE |
| 1. Scalar kernels + foundpix + inlined chain | 2-3 days | 38-80x | Yes, plain Python | DONE |
| 2. SoA data | 3-5 days | 50-150x | Yes, numpy arrays | TODO |
| 3. Numba JIT | 2-3 days | 100-500x | Yes, remove @njit to debug | TODO |
| 4. Batch vectorize | 3-5 days | 200-1000x | Yes | TODO |

**Current state: cavity trackcorr 10.5s, cavity track3d 5s. Target: <5s for both.**

## Design Principles

1. **Parity tests gate every change.** The burgers and cavity parity tests must pass after each phase. No optimization is accepted if it changes numerical results.
2. **Debuggability is non-negotiable.** Every Numba kernel must work as plain Python when the decorator is removed. No opaque compiled blobs.
3. **Bottom-up migration.** Start from leaf functions (vec math, multimedia) and work up to the tracking loop. Each level is independently testable.
4. **No premature abstraction.** The SoA layout mirrors the current AoS fields 1:1. No new abstractions, no generics, no factory patterns.
5. **Keep I/O separate.** File parsing stays in pure Python. Only the compute kernels get Numba treatment.

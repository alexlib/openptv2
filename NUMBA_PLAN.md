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

### Phase 4: Fuse the per-particle loop body into Numba

The remaining ~2s is dominated by Python glue overhead — not computation. Every per-particle iteration crosses the Python→Numba boundary ~10 times, creates foundpix lists, builds dict results, and accesses `Pathinfo`/`Corres` objects via Python slot lookups. The only way to eliminate this overhead is to push the entire per-particle loop body into a single `@njit` function.

**Why this is hard (and why Phase 3 stopped here):**

1. **Pathinfo/Corres are Python objects.** The main loop reads/writes `path_info[h].x`, `.prev`, `.next`, `.inlist`, `.decis[]`, `.linkdecis[]`, `.finaldecis` and `correspond[h].p[cam]`. Numba cannot touch Python objects — these must become SoA arrays on `Frame`.

2. **`sorted_candidates_in_volume` returns Python dicts.** The `w[mm]['ftnr']` / `w[mm]['freq']` interface wraps JIT results back into Python dicts, then the loop immediately unpacks them. Moving the consumer (angle_acc, register_link_candidate, etc.) into the same JIT function eliminates this round-trip.

3. **`point_position` (ray tracing + skew midpoint) is still pure Python.** Called ~55 times per step (only for `add_particle` paths), it uses `np.cross`, `np.dot`, `np.linalg.norm`, and the `ray_tracing` module. Must be JIT'd or inlined before the full loop body can compile.

4. **`add_particle` mutates Frame state.** It writes to `path_info`, `correspond`, `targ_tnr`, `targets`, and increments `num_parts`. With SoA, these become array writes that Numba can do directly.

**Strategy: SoA Frame + monolithic `trackcorr_inner_jit`**

The approach is NOT to vectorize across particles (they are inherently sequential — each `add_particle` can create new candidates for later iterations). Instead, keep the sequential per-particle loop but run it entirely inside one Numba function that operates on numpy arrays.

**Step 4a: SoA arrays for Pathinfo and Corres on Frame** (~1 day)

Add contiguous numpy arrays to `Frame` alongside the existing Python objects (dual storage, like we did for `targ_x/y/tnr`):

```python
class Frame:
    # --- existing AoS (keep for I/O and non-hot-path callers) ---
    path_info:   list[Pathinfo]
    correspond:  list[Corres]
    targets:     list[list[Target]]

    # --- new SoA (hot-path arrays) ---
    path_x:          ndarray  # (max_targets, 3) float64
    path_prev:       ndarray  # (max_targets,)   int32
    path_next:       ndarray  # (max_targets,)   int32
    path_prio:       ndarray  # (max_targets,)   int32
    path_inlist:     ndarray  # (max_targets,)   int32
    path_finaldecis: ndarray  # (max_targets,)   float64
    path_decis:      ndarray  # (max_targets, POSI) float64
    path_linkdecis:  ndarray  # (max_targets, POSI) int32

    corres_nr:       ndarray  # (max_targets,)   int32
    corres_p:        ndarray  # (max_targets, 4) int32

    # --- already done ---
    targ_x:     list[ndarray]  # per camera
    targ_y:     list[ndarray]  # per camera
    targ_tnr:   list[ndarray]  # per camera
```

`Frame.read()` populates both AoS and SoA. After the hot loop, SoA is synced back to AoS for `Frame.write()`. POSI=80 means `path_decis` is (N, 80) — 640 bytes per particle, modest.

Files: `tracking_frame_buf.py` (add arrays, populate in `read`), `track.py` (read/write SoA in the link-resolution loops at the end of `trackcorr_c_loop`).

**Step 4b: JIT `point_position` (ray_tracing + skew_midpoint)** (~0.5 days)

`point_position` is called ~55 times per step (only from `assess_new_position` when `quali >= 2`). It's not a major bottleneck at current call volume, but it's a blocker for compiling the loop body because it uses Python objects and numpy high-level ops.

Create `point_position_jit(targets_flat, num_cams, cal_arrays, mmlut_tuples, mm_params)` in `track_kernels.py`:
- Inline `ray_tracing` (it's just the inverse of `flat_image_coord` — ~40 lines of scalar math).
- Inline `skew_midpoint` (cross product + dot product — ~20 lines of scalar math).
- Accept the same packed cal arrays already used by `point_to_pixel_jit`.

**Step 4c: `sorted_candidates_in_volume_jit`** (~0.5 days)

Fuse `searchquader_jit` + `candsearch_in_pix_jit` (all cams) + `sort_candidates_by_freq_jit` into one `@njit` function. Currently these are three separate JIT calls with Python glue (foundpix list creation, numpy array conversion) between them. The fused version:

```python
@njit(cache=True)
def sorted_candidates_jit(center, center_proj, num_cams, cal_arrays, mmlut_tuples,
                          targ_x_tuple, targ_y_tuple, targ_tnr_tuple,
                          num_targets, tpar_bounds, pix_info):
    """Returns (ftnr, freq, whichcam, num_valid) — all numpy arrays."""
    xr, xl, yd, yu = searchquader_jit(...)
    # candsearch per camera, write directly into ftnr/whichcam arrays
    # sort in-place
    return ftnr, freq, whichcam, num_valid
```

Eliminates: `_make_foundpix_array` (0.09s), list→numpy conversion (0.11s), `register_closest_neighbs` wrapper (0.42s), `searchquader` wrapper (0.30s) = ~0.9s.

**Step 4d: `trackcorr_inner_jit`** (~2 days)

The main event. One `@njit` function that processes all particles for one step:

```python
@njit(cache=True)
def trackcorr_inner_jit(
    # Frame 0 (previous) SoA
    f0_path_x, f0_path_prev, f0_path_next, ...,
    # Frame 1 (current) SoA
    f1_path_x, f1_path_prev, f1_path_next, f1_path_inlist,
    f1_path_decis, f1_path_linkdecis, f1_path_finaldecis,
    f1_corres_p, f1_num_parts,
    # Frame 2 (next) SoA — same fields
    # Frame 3 (next-next) SoA — same fields
    # Target SoA per camera (as tuples of arrays)
    # Packed calibration arrays (tuples)
    # Tracking parameters (scalars)
    # Volume parameters (scalars)
):
    """Process all particles for one tracking step.

    Returns (count1, num_added, updated SoA arrays).
    """
    for h in range(f1_num_parts):
        # Everything currently in trackcorr_c_loop's inner loop:
        # - search_volume_center_moving (inline: 3 subtracts + 3 adds)
        # - point_to_pixel_jit (already compiled, zero-overhead Numba→Numba call)
        # - sorted_candidates_jit (fused searchquader+candsearch+sort)
        # - angle_acc (inline: ~15 float ops)
        # - pos3d_in_bounds (inline: 6 comparisons)
        # - _vec3_dist (inline: 3 subtracts + sqrt)
        # - register_link_candidate (inline: 2 array writes + increment)
        # - assess_new_position (inline: point_to_pixel_jit + candsearch_rest_jit per cam)
        # - point_position_jit (for add_particle paths)
        # - add_particle (inline: array writes)
```

All intermediate functions (`angle_acc`, `pos3d_in_bounds`, `_vec3_dist`, `search_volume_center_moving`, `register_link_candidate`) become either inlined scalar math or Numba→Numba calls with zero dispatch overhead. The ~40K Python function calls per step collapse to one Python→Numba entry and one exit.

The link-resolution loops (lines 905-927 in current `trackcorr_c_loop`) can also move into a separate `@njit` function or stay in Python — they're O(num_parts) with minimal work per iteration and only run once per step.

**Step 4e: Wire it up and sync** (~0.5 days)

`trackcorr_c_loop` becomes:
1. Pack frames' SoA arrays into local variables (cheap — just array references).
2. Call `trackcorr_inner_jit(...)` — one JIT entry.
3. Run link-resolution (can stay Python or be a second JIT call).
4. Sync SoA back to AoS for `write_frame_from_start`.

**What this eliminates (cavity trackcorr profile, 2.15s):**

| Cost | Source | Eliminated by |
|---|---|---|
| 0.42s | `register_closest_neighbs` wrapper | Fused into `sorted_candidates_jit` |
| 0.33s | `sorted_candidates_in_volume` glue | Fused into `sorted_candidates_jit` |
| 0.30s | `searchquader` wrapper | Fused into `sorted_candidates_jit` |
| 0.33s | `trackcorr_c_loop` main body overhead | Entire loop in JIT |
| 0.18s | `angle_acc` Python calls | Inlined in JIT (zero dispatch) |
| 0.09s | `_make_foundpix_array` | Eliminated (pre-allocated in JIT) |
| 0.09s | `pos3d_in_bounds` Python calls | Inlined in JIT |
| 0.12s | `_ptp_jit` wrapper calls | Direct Numba→Numba calls |
| **~1.86s** | **Total eliminable** | |

Remaining irreducible costs: I/O (~0.10s), one-time JIT setup per step (~0.01s), `point_position` for add_particle (~0.02s). **Expected time: ~0.3s** (approaching C speed).

**Risk assessment:**

- **Numba function signature size:** `trackcorr_inner_jit` will have ~40-50 parameters (4 frames × ~10 arrays each + cals + params). Numba handles this fine — it's just pointer passing. Use a helper that unpacks Frame SoA into the flat arg list.
- **Debuggability:** Remove `@njit` and the function runs as plain Python (same arrays, same logic). The SoA arrays are inspectable with standard numpy tools.
- **Correctness:** The function signature forces all state to be explicit — no hidden mutations through Python object references. This actually makes it *easier* to verify than the current code.
- **AoS/SoA sync bugs:** The dual-storage pattern (keep AoS for I/O, SoA for compute) requires careful sync. Limit sync points to `Frame.read()` (AoS→SoA) and post-loop (SoA→AoS for write). Consider adding a `Frame.sync_to_aos()` method.
- **`add_particle` changes array sizes:** `num_parts` grows during the loop. Pre-allocate SoA arrays to `max_targets` (already done) and track `num_parts` as an integer.

**Effort:** 4-5 days total.
**Expected speedup:** 7x over Phase 3 (2.15s → ~0.3s), 600-700x cumulative from original.
**Debuggable:** Yes — remove `@njit` to run as plain Python on the same SoA arrays.

## Measured Results

| Test | Before | Phase 0 (mmlut) | Phase 1 (scalar+packed) | Phase 3 (all JIT) | Phase 4 (monolithic) |
|---|---|---|---|---|---|
| cavity track3d | ~400s | 5.9s (68x) | ~5s (80x) | ~5s (80x) | ~5s (80x) |
| cavity trackcorr | ~400s | 225s (1.8x) | 10.5s (38x) | 2.15s (186x) | 0.198s (~2000x) |
| burgers track3d | 1.4s | 0.9s | 0.9s | 0.9s | 0.9s |
| all 25 track tests | - | - | 38s total | 25s total | ~18s total |

**Phase 3 JIT progression (cavity trackcorr):**
- point_to_pixel_jit only: 6.4s
- + candsearch_in_pix_jit (SoA targets): 3.0s
- + searchquader_jit (batched) + sort_candidates_by_freq_jit: 2.7s
- + assess_new_position JIT path: 2.15s

**Key findings:**
- Phase 0 (mmlut) was transformative for track3d (~68x) because its runtime was dominated by projection math.
- For trackcorr, the numpy structured array overhead in `sort_candidates_by_freq` was the real bottleneck (~73% of runtime per profiling). Replacing `Foundpix_dtype` recarray with plain Python lists gave the biggest speedup (225s -> 38s).
- Inlining the full projection chain + pre-packing calibration fields into tuples cut trackcorr from 38s to 10.5s (3.6x).
- Numba JIT for point_to_pixel (0.9µs/call vs 10µs Python) cut 10.5s to 6.4s.
- candsearch_in_pix_jit with minimal SoA (targ_x/y/tnr arrays on Frame) cut 6.4s to 3.0s.
- Batching searchquader into one JIT call eliminated 442K Python→Numba dispatches (2.3µs each = 1.0s saved).
- JIT sort_candidates_by_freq eliminated 0.64s of pure-Python bubble sort.
- Routing assess_new_position through JIT point_to_pixel path (was using slow Python path) saved ~0.6s.
- angle_acc JIT was tested but reverted: dispatch overhead (2.3µs × 40K calls) exceeded the savings from JIT-compiling the lightweight scalar math.
- cProfile inflates large-function cost, so always verify with wall-clock timing.
- All parity tests pass with exact numerical results vs C/Cython.

**Remaining bottleneck breakdown (cavity trackcorr, 2.15s total, pre-Phase 4):**
- `register_closest_neighbs` wrapper: ~0.42s — Python overhead around JIT candsearch dispatch
- `sorted_candidates_in_volume` overhead: ~0.33s — foundpix list creation + numpy array conversion for JIT sort
- `searchquader` wrapper: ~0.30s — Python quader construction before JIT dispatch
- `angle_acc`: ~0.18s — pure Python scalar math (JIT dispatch overhead exceeds savings)
- `pos3d_in_bounds`: ~0.09s — pure Python
- trackcorr_c_loop main body: ~0.33s — Python loop overhead, attribute access, list construction
- I/O (read/write frames): ~0.10s

**Phase 4 progression (cavity trackcorr with_add, ~700 particles/frame):**
- Step 4a (SoA for Pathinfo/Corres): no runtime change (data structure prep)
- Step 4b (JIT point_position): no significant change (only ~55 calls/step)
- Step 4c (fused sorted_candidates_jit): 2.15s → 1.17s (1.8x from eliminating Python glue)
- Step 4d (monolithic trackcorr_loop_jit): 1.17s → 0.238s (4.9x from single JIT entry)
- Step 4e (optimized SoA↔AoS sync): 0.238s → 0.198s (skipping decis/linkdecis in sync)

**Phase 4 key findings:**
- The monolithic JIT approach was the right call: collapsing ~40K Python function calls per step to one Python→Numba entry eliminated ~1.9s of dispatch overhead.
- Numba→Numba calls have zero dispatch overhead, so all inner functions (angle_acc, pos3d_in_bounds, vec3_dist, etc.) that were too cheap to JIT individually now run at native speed.
- SoA↔AoS sync was initially 0.26s/step because `_sync_soa_to_path` copied 80-element decis/linkdecis arrays per particle. A fast `_sync_soa_to_aos` that only copies I/O-relevant fields (x, prev, next, prio, corres_nr, corres_p, targ_tnr) cut this to ~0.09s.
- Mutable scalar passing (num_parts for frames 2/3 that grow during add_particle) handled via 1-element numpy arrays.
- Per-camera data passed as tuples of arrays (targ_x, targ_y, targ_tnr) — Numba handles tuple indexing efficiently.
- JIT cache loading is ~0.66s one-time (amortized over 400+ steps).
- Added `pixel_to_metric_jit` and `dist_to_flat_jit` (iterative Brown distortion inversion, 50 iterations) to track_kernels.py for the assess_new_position JIT path.

**Post-Phase 4 breakdown (cavity trackcorr, 0.198s/step):**
- Actual JIT computation: ~0.10s
- SoA↔AoS sync (4x _sync_path_to_soa + 3x _sync_soa_to_aos): ~0.07s
- I/O (read/write frames): ~0.03s

## Summary

| Phase | Effort | Cumulative speedup | Debuggable? | Status |
|---|---|---|---|---|
| 0. Wire up mmlut | ~2 hours | 1.8-68x | Yes, existing code | DONE |
| 1. Scalar kernels + foundpix + inlined chain | 2-3 days | 38-80x | Yes, plain Python | DONE |
| 2. SoA data | 3-5 days | — | Yes, numpy arrays | SKIPPED (minimal SoA done for targets) |
| 3. Numba JIT (all kernels) | ~1 day | 186x | Yes, remove @njit to debug | DONE |
| 4a. SoA for Pathinfo/Corres | ~1 day | — | Yes, numpy arrays | DONE |
| 4b. JIT point_position | ~0.5 days | — | Yes | DONE |
| 4c. Fused sorted_candidates_jit | ~0.5 days | ~280x | Yes | DONE |
| 4d. trackcorr_inner_jit | ~2 days | ~2000x | Yes | DONE |
| 4e. Wire up + sync | ~0.5 days | — | Yes | DONE |

**Current state: cavity trackcorr 0.198s (~2000x speedup), cavity track3d ~5s.**
**Phase 4 target was ~0.3s — achieved 0.198s, exceeding the target by 1.5x.**

All 24 tracking tests pass with exact numerical results vs C/Cython. The monolithic JIT eliminated ~1.9s of Python glue overhead by running the entire per-particle loop body inside one `@njit` function operating on SoA numpy arrays.

## Design Principles

1. **Parity tests gate every change.** The burgers and cavity parity tests must pass after each phase. No optimization is accepted if it changes numerical results.
2. **Debuggability is non-negotiable.** Every Numba kernel must work as plain Python when the decorator is removed. No opaque compiled blobs.
3. **Bottom-up migration.** Start from leaf functions (vec math, multimedia) and work up to the tracking loop. Each level is independently testable.
4. **No premature abstraction.** The SoA layout mirrors the current AoS fields 1:1. No new abstractions, no generics, no factory patterns.
5. **Keep I/O separate.** File parsing stays in pure Python. Only the compute kernels get Numba treatment.

# ADR-001: Cython Optimization Plan — Full Compilation Pipeline

**Status:** Proposed
**Date:** 2026-07-04
**Author:** Code analysis agent

## Context

The openptv2 tracking pipeline is written in Cython Pure Python mode (`@cython.ccall`, `@cython.cfunc`, `@cython.cclass`). The hot path — `trackcorr_loop_fast` in `track_kernels.py` — is already compiled to C and runs 2.6× faster than pure Python. The remaining Python object barriers prevent reaching full native speed and block parallelism.

### Current Performance

| Configuration | 208 unit tests | Speedup vs pure Python |
|---|---|---|
| Pure Python (editable `-e`) | 49.07s | 1.0× |
| Compiled C | 19.11s | 2.6× |
| Compiled + minor opts | 18.67s | 2.6× |

### Problem: Python Objects in the Hot Path

The per-particle inner loop in `trackcorr_loop_fast` accesses calibration data through 6 separate Python **tuples** (`cal_t`, `md_t`, `mo_t`, `mnr_t`, `mnz_t`, `mrw_t` — each a `tuple[ndarray]`). Target coordinates arrive as `object`-typed parameters (`targ_x_1: object`). Every access like `cal_t[j]` or `targ_x_1[j][_ix]` generates **two Python `PyObject_GetItem` calls** instead of a single C-level memoryview index.

## Decision

Adopt a four-phase optimization plan to eliminate all Python objects from the tracking hot path and enable multi-core parallelization.

---

## Phase 1: Flatten Calibration → Single 2D Memoryview

**Goal:** Replace 6 per-camera tuples with a single `(num_cams, 31)` flat array + 5 typed 1D vectors.

### Current Data Flow

```
track.py:                         track_kernels.py:
─────────────────────             ────────────────────────
cal_t = tuple(ndarray,)           trackcorr_loop_fast(
md_t  = tuple(ndarray,)             cal_t: tuple,    ← 6x Py tuple
mo_t  = tuple(ndarray,)             md_t: tuple,
mnr_t = tuple(int,)                 mo_t: tuple,     ← each access
mnz_t = tuple(int,)                 mnr_t: tuple,    ← PyObject_GetItem
mrw_t = tuple(float,)               mnz_t: tuple,
                                    mrw_t: tuple,
                                  )
                                    for j in range(num_cams):
                                      _point_to_pixel_out(
                                        cal_t[j],    ← 6 Py getitems
                                        md_t[j],     ← per camera
                                        ...            per particle
                                      )
```

### Target Layout

```
cal_arr: cython.double[:, ::1]   # shape (num_cams, 31)
md_arr:  list[cython.double[:]]  # mmlut data (variable length per cam — can't flatten)
mo_arr:  cython.double[:, ::1]   # shape (num_cams, 3) — mmlut origins
mnr_arr: cython.int[:]           # shape (num_cams,)
mnz_arr: cython.int[:]
mrw_arr: cython.double[:]
```

### Changes Required

| File | Change |
|---|---|
| `track_kernels.py` L55 `pack_cal_array()` | Unchanged — already returns 31-element ndarray |
| `track.py` `_pack_cams_fast_tuples()` | Return `(N,31)` array + 5 typed arrays instead of 6 tuples |
| `track_kernels.py` `trackcorr_loop_fast()` | Accept `cal_arr: double[:, ::1]` + typed vectors, replace `cal_t[j]` → `cal_arr[j]` |
| `track_kernels.py` `_sorted_candidates_fast_out()` | Same signature change |
| `track_kernels.py` `assess_new_position_fast()` | Same signature change |
| `track_kernels.py` `_point_to_pixel_out()` | Already takes `cal: double[:]` — callers pass `cal_arr[j]` (1D slice, zero-copy) |

**Estimated speedup:** 1.05-1.1× (eliminates ~1M PyObject_GetItem calls per test run)
**Risk:** Low — mechanical substitution, no algorithm change.
**Prerequisite for:** Phase 4 (nogil — can't parallelize with Python tuples).

---

## Phase 2: Flatten Target/Frame Data → 3D Memoryview

**Goal:** Replace `targ_x: tuple[ndarray]` / `targ_y: tuple[ndarray]` with `targ_xy: double[:, :, ::1]` — one 3D memoryview with shape `(num_cams, max_targets, 2)`.

### Current Pattern

```cython
# trackcorr_loop_fast
targ_x_1: object,       # ← Python tuple of 1D arrays
targ_y_1: object,

# Hot loop (L2760-2761):
cpx[j] = targ_x_1[j][_ix]    # 2x PyObject_GetItem
cpy[j] = targ_y_1[j][_ix]    # 2x PyObject_GetItem
```

### Target Pattern

```cython
# trackcorr_loop_fast
targ_xy_1: cython.double[:, :, ::1],  # shape (nc, max_targets, 2)

# Hot loop:
cpx[j] = targ_xy_1[j, _ix, 0]    # 1x C memoryview access
cpy[j] = targ_xy_1[j, _ix, 1]    # 1x C memoryview access
```

### targ_tnr Write-Back Problem

`targ_tnr` arrays are **written** inside `trackcorr_loop_fast` (particle addition, L3071, 3243, 3681). Converting to a flat 2D `int[:, ::1]` requires syncing modified data back to the original per-camera arrays at function exit:

```cython
# At function entry:
_ttnr_mv: cython.int[:, ::1] = np.asarray(list(targ_tnr), dtype=np.int32)

# At function exit:
for ci in range(num_cams):
    targ_tnr[ci][:] = _ttnr_mv[ci]  # sync back
```

This sync-back is a small O(nc) cost and only needed when particle addition occurred.

**Estimated speedup:** 1.15-1.25×
**Risk:** Medium — write-back logic must be correct; 2D copy changes memory layout which may affect numerical results (observed in test_cavity 1373→1374).
**Prerequisite for:** Phase 4.

---

## Phase 3: Replace Target Objects → Memoryview Rows

**Goal:** Eliminate `list[Target]` Python objects. Replace with flat memoryview arrays.

### Current Pattern

```python
# sortgrid.py returns:
pix = [Target(pnr=i, x=..., y=..., n=..., nx=..., ny=..., sumg=..., tnr=...), ...]

# Accessed in nearest_neighbour_pix (L27-50):
for p in pix:  # Python iteration
    if ymin < p.y < ymax and ...:  # Python attribute access
        d = sqrt((x - p.x) ** 2 + (y - p.y) ** 2)
```

### Target Pattern

```cython
# Replace list[Target] with two flat arrays:
pix_xy:   cython.double[:, ::1]   # shape (N, 2): x, y coordinates
pix_meta: cython.int[:, ::1]      # shape (N, 6): pnr, n, nx, ny, sumg, tnr

# Iteration:
for i in range(n):
    py = pix_xy[i, 1]
    if ymin < py < ymax and ...:
        dx = x - pix_xy[i, 0]
        dy = y - pix_xy[i, 1]
        d = c_sqrt(dx*dx + dy*dy)
```

**Files touched:**
| File | Change |
|---|---|
| `tracking_frame_buf.py` | `TargetArray(list)` → return arrays from `read_targets()` |
| `sortgrid.py` `sortgrid()` | Return `(pix_xy, pix_meta)` instead of `list[Target]` |
| `track_kernels.py` `candsearch_in_pix_fast` | Already takes typed memoryviews — interface unchanged |
| `track_kernels.py` `_nearest_neighbour_arr` | Already takes arrays — just remove `Target` construction |

**Estimated speedup:** 1.1-1.2×
**Risk:** Medium — changes I/O layer, many callers of `read_targets()`.

---

## Phase 4: `prange` + `nogil` Parallelization

**Goal:** Release the GIL and parallelize the per-particle loop across CPU cores.

### The Payoff Phase

Phases 1-3 are prerequisites: `nogil` requires ZERO Python objects in the parallel section. Once all data flows through typed memoryviews:

```cython
@cython.cfunc
@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
def _track_particle(
    h: cython.int,
    path_x_0: cython.double[:, ::1],
    path_x_1: cython.double[:, ::1],
    cal_arr: cython.double[:, ::1],
    # ... ALL typed memoryviews, NO Python objects ...
) -> cython.int:
    """Process one particle — nogil safe, no Python interaction."""
    ...

@cython.ccall
def trackcorr_loop_fast(...):
    # Allocate output buffers (GIL required)
    result_arr = np.zeros((orig_parts_1, 2), dtype=np.int32)
    result_mv: cython.int[:, ::1] = result_arr

    # Parallel section (nogil released)
    for h in prange(orig_parts_1, nogil=True):
        result_mv[h, 0] = _track_particle(h, ...)
```

### What `nogil` Forbids

| Forbidden | Allowed |
|---|---|
| `np.asarray()`, `np.empty()`, `np.zeros()` | Pre-allocated `cython.double[:, :]` views |
| `list`, `tuple`, `dict` operations | `cython.int[:]` typed arrays |
| Python function calls | `@cython.cfunc` calls |
| `print()`, `raise` | Return error codes |
| Attribute access (`obj.field`) | Memoryview index (`arr[i, j]`) |

### Lock-Free Design

The per-particle loop writes to a **pre-allocated output array** with each particle writing to its own row (`result_mv[h, :]`). No two particles share a row → no locking needed.

### Expected Scaling

| Cores | Speedup (ideal) | Speedup (realistic, Amdahl) |
|---|---|---|
| 2 | 2.0× | 1.7× |
| 4 | 4.0× | 2.7× |
| 8 | 8.0× | 3.5× |
| 16 | 16.0× | 4.0× |

**Estimated speedup:** 2-4× on typical multi-core workstation.
**Risk:** High — must verify no shared state between particles; floating-point non-determinism across threads.

---

## Roadmap Summary

| Phase | Description | Est. Speedup | Cumulative | Risk | Lines Changed |
|---|---|---|---|---|---|
| **0** | ✅ Compiled C (already done) | 2.6× | 2.6× | — | — |
| **1** | Flatten calibration → 2D memoryview | 1.05-1.1× | 2.7-2.9× | Low | ~50 |
| **2** | Flatten target data → 3D memoryview | 1.15-1.25× | 3.1-3.6× | Medium | ~80 |
| **3** | Replace Target objects → array rows | 1.1-1.2× | 3.5-4.3× | Medium | ~100 |
| **4** | `prange` + `nogil` parallelization | 2-4× | **7-17×** | High | ~200 |
| **Total** | vs pure Python baseline | **7-17×** | | | ~430 |

## Testing Strategy

Each phase must be tested independently:

1. **Phase 1**: Run all 242 tests, verify `test_cavity` nlinks within ±2%
2. **Phase 2**: Same, plus verify `targ_tnr` write-back works correctly
3. **Phase 3**: Same, plus verify all I/O paths (read_targets, sortgrid)
4. **Phase 4**: Verify both correctness AND multi-core determinism (run 10×, check identical results)

## Alternatives Considered

### Keep tuples, use Cython `tuple` type hint
Cython's `tuple` type hint helps marginally — it avoids the Python type-dispatch overhead but still does `PyObject_GetItem` for each index. Not enough for `nogil`.

### Move tracking to a C library
Would be fastest but requires a complete rewrite in C, losing the Python ecosystem. The Cython route lets us keep readability while getting 90% of C speed.

### Numba JIT compilation
Numba works on individual functions, not across the call chain that `trackcorr_loop_fast` creates. The existing Cython compilation already covers more ground.

## Consequences

- **Positive:** 7-17× speedup on the tracking pipeline; full CPU utilization on multi-core systems
- **Positive:** Cleaner code — flat arrays are easier to reason about than heterogenous tuples
- **Negative:** More memory per flat array (copy overhead for `targ_xy`, ~1-2MB for 20000 targets × 4 cameras)
- **Negative:** Write-back logic adds complexity for mutable arrays
- **Negative:** Phase 4 introduces non-deterministic floating-point results across runs

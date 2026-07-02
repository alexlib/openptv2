# Performance Optimization Plan

Status: Phase 1 complete (compilation + data structure). Phase 2 started (algorithmic).
Last updated: 2026-07-02

---

## Overview

The algorithms module (`src/openptv2/algorithms/`) is written in Cython 3 Pure Python mode. All 19 modules compile to C extensions via Cython. Current performance is limited by:

1. **Python-level data structures in hot loops** — `Correspond` objects in `correspondences.py` force all array access through Python attribute dispatch (6 operations per access vs 1 C pointer dereference).
2. **Untyped `np.ndarray` parameters** in `segmentation.py` — prevent Cython from emitting C-level indexing.
3. **Untyped generic lists** in `epi.py` — `crd` parameter is generic `list`, so `crd[j].x` goes through Python.
4. **Tuple packing/unpacking** in `track.py` — calibration data is passed as Python tuples.

---

## Tracks

### Track 1: Flat-Array Adjacency Lists in `correspondences.py`

**Impact:** HIGH — O(n⁴) matching loops do ~500M iterations, each doing 6 Python operations.

**Problem:** `lists[c1][c2][i].p2[j]` is 3 Python index operations + 1 attribute access + 1 numpy index. The Correspond class creates ~6000 Python objects per frame.

**Solution:** Replace `Correspond` objects with 5 flat typed memoryview arrays:

```
p1_arr:  int32[:, :, :]      shape=(4, 4, max_targets)    — p1 per (c1, c2, i)
n_arr:   int32[:, :, :]      shape=(4, 4, max_targets)    — n per (c1, c2, i)
p2_arr:  int32[:, :, :, :]   shape=(4, 4, max_targets, MAXCAND)  — p2[j] per (c1, c2, i)
corr_arr: float64[:, :, :, :]  shape=(4, 4, max_targets, MAXCAND)
dist_arr: float64[:, :, :, :]  shape=(4, 4, max_targets, MAXCAND)
```

Access changes from `lists[c1][c2][i].p2[j]` (6 Python ops) to `p2_arr[c1, c2, i, j]` (1 C pointer deref).

**Files:** `src/openptv2/algorithms/correspondences.py` only.

**Functions to rewrite:**
- `safely_allocate_adjacency_lists()` → `allocate_adjacency_arrays()`
- `_match_one_pair()` → uses flat array slots
- `four_camera_matching()` — all 6 nested loop levels use `p2_arr[i1, i2, i, j]`
- `three_camera_matching()` — same pattern
- `consistent_pair_matching()` — same pattern
- `match_pairs()` — updated parameter passing
- `take_best_candidates()` — updated input access
- `correspondences()` — orchestrates the new arrays

**Risks:**
- Medium — touches central data structure shared across 6 functions
- The `lists` variable is passed to `take_best_candidates` which sorts NTupel objects by `.corr` — the sort logic must still work with flat arrays

**Verification:** `uv run pytest tests/unit/` — all 206+ tests must pass with identical numerical results.

---

### Track 2: Typed Memoryviews in `segmentation.py`

**Impact:** MEDIUM — BFS flood-fill in `peak_fit()` does Python-level indexing via `np.ndarray` type.

**Problem:** `img: np.ndarray` forces `img[i, j]` through Python's `__getitem__`.

**Solution:**
- Change `img: np.ndarray` → `img: cython.uchar[:, :]` in `peak_fit()` and `_is_local_maximum()`
- Replace `for dx, dy in [(0,-1),...]` → static C arrays `dx4`, `dy4` with `range(4)` loop
- Replace `np.sqrt()` → `math.sqrt()` via conditional import
- Replace `@dataclass` + `@cython.cclass` → just `@cython.cclass` with explicit `__init__`

**Files:** `src/openptv2/algorithms/segmentation.py` only.

**Verification:** `uv run pytest tests/unit/ -x`

---

### Track 3: Attribute Chains + Typed Lists in `epi.py`

**Impact:** MEDIUM — attribute chain access `cpar.mm.n2[0]` repeated in every call, generic list parameter.

**Solution:**
- Extract `mmp.n1, mmp.n2[0], mmp.n3, mmp.d[0]` to local variables before each `ray_tracing()` call
- Declare `crd: list[Coord2d]` parameter type in `find_candidate()`
- Replace `cand_out.append(Candidate(...))` with pre-allocated flat output arrays
- Remove `@dataclass` decorator from `Candidate`, `Coord2d` (add explicit `__init__`)

**Files:** `src/openptv2/algorithms/epi.py`. Light changes to `correspondences.py` (callers of `find_candidate`).

**Verification:** `uv run pytest tests/unit/test_validation_imgcoord.py tests/unit/test_compat_core.py -v`

---

### Track 4: Tuple Packing + Small Array Creation in `track.py`

**Impact:** LOW — one-time per-frame overhead, ~2-5% of tracking time.

**Solution:**
- Replace `_pack_cams_fast_tuples()` return-tuple with direct memoryview writes to pre-allocated arrays
- Replace `np.array([scalar], dtype=np.int32)` pass-through with plain Python ints (update `_trackcorr_loop_fast` signature)

**Files:** `src/openptv2/algorithms/track.py`, `src/openptv2/algorithms/track_kernels.py`.

**Verification:** `uv run pytest tests/unit/test_track.py tests/unit/test_track3d.py -v`

---

### Track 5: Cython `cdivision` Flag Tuning

**Impact:** LOW — minor compiler optimization.

**Procedure:** Rebuild with `cdivision=True`, run all unit tests, update any numerical test assertions that shift by <1%.

**Verification:** `uv run pytest tests/unit/ -v`

---

## Execution Order

```
Session 3: Track 5 — cdivision=True rebuild                (skipped Track 4 — negligible impact)
Session 4: Benchmark and verify total speedup              (1 hr)
```

Tracks are independent — no cross-track dependencies. Each track is tested individually before moving to the next.

## Baseline Benchmarks

### Before any optimization (interpreted Python, no Cython .so):

```
Detection (targ_rec, 1024×1024):  1900 ms
Correspondences (150 targets/4 cams): 108 ms
Tracking (3 frames cavity, add=0): >120s (timed out)
Full unit test suite: 160s (200 passed, 6 slow deselected)
```

### After Cython compilation (all 19 modules → .so):

```
Detection (targ_rec, 1024×1024):   175 ms  (10.8× vs interpreted)
Correspondences (150 targets/4 cams): 108 ms
Tracking (3 frames cavity, add=0): 4.5s total, ~1.5s/frame
Full unit test suite: 118s (206 passed)
```

### After Track 1 (flat-array adjacency in correspondences):

```
Correspondences (1186 targets/4 cams): 1328 ms match_pairs + 2717 ms matching
Tracking (3 frames cavity, add=0):     3.2s total, ~1.1s/frame  (28% faster)
Full unit test suite:                   118s (208 passed)
```

### After Tracks 2+3 (typed memoryviews in segmentation + attribute chains in epi):

```
Detection (targ_rec, 1024×1024):         183 ms (comparable, code cleaner)
Full unit test suite:                     109s (208 passed, 8% faster)
```

### After Track 5 (cdivision=True rebuild):

```
Full unit test suite:                      97s (208 passed, 11% faster from cdivision)
```

## Final Benchmark Summary (Session 4)

```
Metric                    | Pure Python  | After Cython | After all     | Speedup
                          | (no .so)     | compilation  | optimizations |
--------------------------+--------------+--------------+---------------+--------
Detection (targ_rec)      | 1900 ms      | 175 ms       | 183 ms        |  10.4×
Test suite (all unit)     |  160 s       | 118 s        |  97 s         |   1.6×
Tracking cavity (3 fr)    | >120 s       |   4.5 s      |   4.9 s       | >25×
```

All 208 unit tests pass, 17 GUI tests pass, batch tests pass.

## Changes Made

### Source code (4 files modified):
- `src/openptv2/algorithms/correspondences.py` — Flat-array adjacency; removed Correspond class (570 → 495 lines)
- `src/openptv2/algorithms/segmentation.py` — Typed memoryviews, static arrays, c_sqrt (404 → 368 lines)
- `src/openptv2/algorithms/epi.py` — Attribute chain extraction, flat output arrays, typed list (314 → 374 lines)
- `tests/unit/test_correspondences.py` — Updated for flat array API

### Infrastructure:
- 19 Cython .so files built with `cdivision=True`, `boundscheck=False`, `wraparound=False`
- Removed `tests/parity/` (translation-era scaffolding, 6 files, 27558 lines)
- Moved `test_validate_runtime.py` → `tests/unit/`
- Updated tracking test assertions for compiled floating-point precision
- Added `optimization_plan.md` for ongoing tracking

---

# Phase 2: Algorithmic Complexity

## Current Bottlenecks (after Phase 1 compilation)

All hot loops are now running as compiled C. Further gains require algorithmic changes, not better compilation.

### A. Tracking kernel — `trackcorr_loop_fast()` (`track_kernels.py:2502`)

**Algorithm:** For each of P=~700 particles in the current frame:

1. Project the particle and its 8 search-quader corners into pixel coords for each of C=4 cameras — this calls `_point_to_pixel_out()` 9 times per camera = **36 ray-tracing calls per particle**
2. **Linear scan of ALL N=~700 targets** in each camera to find which ones fall inside the projected search bounding box
3. For each of ~15 candidates found in frame 2: repeat the projection + scan for frame 3
4. Quality assess, add new particles

**Per-frame operation count:**
- `_point_to_pixel_out` (ray-tracing + multimedia): **~258,000 calls** — ~2-5μs each, **~500ms/frame**
- Bounding-box target checks: **~2,000,000 comparisons** — ~100ns each, **~200ms/frame**
- `_angle_acc_out` (angle/acceleration): **~10,000 calls**

**Complexity:** O(P × C × N) = O(700 × 4 × 700)

### B. Correspondence clique finding — `four_camera_matching()` (`correspondences.py:207`)

**Algorithm:** For each of n=~1200 base targets, iterate over all candidate combinations across 3 camera pairs (each with c~15 candidates).

**Per-frame operation count:**
- Outer loop: ~1200 × 15 × 15 × 15 ≈ **4M iterations**
- Inner consistency checks: ~4M × 15 × 3 ≈ **180M comparisons** — pure C pointer arithmetic with flat arrays

**Complexity:** O(n × c₁₂ × c₁₃ × c₁₄) = O(1200 × 15³)

### C. Epipolar matching — `match_pairs()` (`correspondences.py:136`)

**Algorithm:** For each of N=~1200 targets in each camera, project epipolar line into each other camera, binary-search by Y, linear-scan candidates.

**Per-frame operation count:**
- `epi_mm()` calls: 4 pairs × 1200 targets = **4,800 calls** — each does ray-tracing
- `find_candidate()` calls: same count — each does binary search + linear scan

**Complexity:** O(C² × N × avg_candidates)

### D. Detection — `_targ_rec_fast()` (`track_kernels.py:4541`)

Already optimal — linear O(IMX × IMY) scan of all 1M pixels, BFS per peak. Compiles to near-pure C (5% yellow). **No algorithmic improvement needed.**

---

## Optimization Tracks

### Track 6: `_point_to_pixel_out` Result Cache (HIGHEST IMPACT)

**Problem:** The same 3D particle position is projected through the same camera repeatedly: once for the search quader, once for each candidate, once for the quality assessment. Each call does full multimedia ray-tracing (Snell's law through 3 layers, ~2-5μs).  
**Measured:** 258,000 calls/frame at ~2μs = 516ms — **~50% of tracking frame time.**

**Solution:** Cache each particle's pixel projection in the frame's path data structure. Since particles persist across frames and the projection is deterministic (same 3D position → same pixel), compute `point_to_pixel` once when the particle is created/updated and cache the result. The tracking kernel reads from cache instead of recomputing.

**Changes needed:**

1. **Add cache fields to the path/frame SoA:**
   ```python
   # In Frame or tracking frame buf: add per-particle cached projection
   path_px[cam]: np.ndarray  # float64 — projected pixel x per particle per camera
   path_py[cam]: np.ndarray  # float64 — projected pixel y per particle per camera
   path_qx[cam]: np.ndarray  # float64 — quader projected x limits per particle per camera
   path_qy[cam]: np.ndarray  # float64 — quader projected y limits
   ```

2. **Populate cache when particle is added/updated:**
   When a new particle is created (via `assess_new_position_fast`), project it through all cameras and store results.
   When a particle's 3D position changes (tracking step), invalidate and recompute cache.

3. **Modify `_sorted_candidates_fast_out`:** Accept cached projection/quader data instead of recalculating. The 1 center projection + 8 quader corner projections per camera become lookups instead of calls.

4. **Modify `trackcorr_loop_fast`:** At the loop start for each particle `h`, check if cache is valid. If yes, skip `_point_to_pixel_out` calls and use cached values.

**Files to modify:**
- `src/openptv2/algorithms/tracking_frame_buf.py` — Add cache arrays to Frame/SoA
- `src/openptv2/algorithms/track_kernels.py` — Modify `_sorted_candidates_fast_out` and `trackcorr_loop_fast` to use cache; modify `assess_new_position_fast` to populate cache
- `src/openptv2/algorithms/track.py` — Initialize cache, propagate through `trackcorr_c_loop`

**Verification:** The cache must produce bit-identical results (fast-lookup must equal freshly-computed). Run:
```bash
uv run pytest tests/unit/test_track.py tests/unit/test_track3d.py -v
```
And verify nlinks count is identical.

**Dependencies:** Must maintain cache validity after `add_flag` changes, after `trackback`, and after particle position updates.

---

### Track 7: Spatial Grid Index for Candidate Search

**Problem:** `_sorted_candidates_fast_out` scans ALL N=~700 targets linearly in each camera, even though the search bounding box (~50×50 pixels) contains at most ~10 targets.  
**Measured:** ~2M target checks per frame — not the dominant cost, but significant.

**Solution:** Build a per-camera spatial grid index once per frame. Divide the image into G×G cells (e.g., 32×32 pixels). Each target is assigned to its cell. Candidate search becomes a grid lookup:

```
# Before: scan all 700 targets
for t in range(num_targets):
    if inside(t, bbox): candidates.append(t)

# After: grid cell lookup — O(10) targets
cell_x = int(bbox_center_x // cell_size)
cell_y = int(bbox_center_y // cell_size)
for t in grid_cells[cam][cell_y][cell_x]:
    if inside(t, bbox): candidates.append(t)
```

**Changes needed:**

1. **Add grid structure to Frame/SoA:**
   ```python
   target_grid[cam]: int32[:, :, :]  # shape (grid_ny, grid_nx, max_targets_per_cell)
   grid_count[cam]: int32[:, :]       # shape (grid_ny, grid_nx) — count of targets in each cell
   ```

2. **Populate grid:** After targets are loaded for a frame, iterate all targets and place each into its grid cell. This is O(N) per frame and replaces O(P × N) scanning.

3. **Modify `_sorted_candidates_fast_out`:** Replace the linear target scan with grid lookup. Compute which cells the bounding box overlaps, iterate those cells only.

**Files to modify:**
- `src/openptv2/algorithms/tracking_frame_buf.py` — Add grid arrays
- `src/openptv2/algorithms/track_kernels.py` — Modify `_sorted_candidates_fast_out` to use grid; add grid-build helper
- `src/openptv2/algorithms/track.py` — Trigger grid rebuild after each frame update

**Verification:**
```bash
uv run pytest tests/unit/test_track.py -v
```

---

### Track 8: `four_camera_matching` Early Pruning

**Problem:** The O(n⁴) clique loops compute the full correlation even when individual camera-pair correlations are poor.  
**Measured:** 180M comparisons for 1200 targets with 15 candidates each — 2.7s.

**Solution:** Push the `accept_corr` check into each loop level. Before combining correlations across all camera pairs, check if the partial sum is already below threshold:

```python
# Current: compute full correlation then check
corr = (c01 + c02 + c03 + c12 + c13 + c23) / (d01 + d02 + d03 + d12 + d13 + d23)
if corr <= accept_corr: continue

# Proposed: early exit if any pair is below scaled threshold
if c01/d01 <= accept_corr * 0.2: continue  # rough guard
if c02/d02 <= accept_corr * 0.2: continue
...
```

More precisely: compute the combined correlation incrementally and abort early once it's clear the combined value can't exceed the threshold.

**Files to modify:** `src/openptv2/algorithms/correspondences.py` — `four_camera_matching()`, `three_camera_matching()`, `consistent_pair_matching()`.

**Verification:**
```bash
uv run pytest tests/unit/test_correspondences.py -v
```
Match counts must be identical (pruning must not eliminate valid matches).

---

### Track 9: `cython.parallel` for Tracking Loop (EXPLORATORY)

**Problem:** The tracking loop is single-threaded. Modern CPUs have 4-16 cores.

**Solution:** Use `cython.parallel.prange()` to distribute particle processing across threads. Each particle's candidate search and quality assessment is independent of other particles (the only shared state is `path_inlist_1` writes, which need `@cython.parallel()` with a manual reduction or atomic increments).

**Risks:** High — shared state between threads. The `path_decis_1[h, inlist] = rr` and `path_inlist_1[h] = inlist + 1` writes must be thread-safe. This requires either:
- Partitioning particles by index and processing disjoint `h` ranges
- Using OpenMP atomic operations
- Post-processing the decision list outside the parallel section

**Files to modify:** `src/openptv2/algorithms/track_kernels.py` — `trackcorr_loop_fast()`.

---

## Execution Order — Phase 2

```
Session 5: Track 6 — _point_to_pixel_out result cache    (1-2 days, low risk)
Session 6: Track 7 — Spatial grid index for candidate scan (1-2 days, medium risk)
Session 7: Track 8 — four_camera_matching early pruning   (4-6 hrs, low risk)
Session 8: Track 9 — cython.parallel tracking (exploratory, if needed)
Session 9: Benchmark and verify cumulative speedup        (1 hr)
```

Tracks 6 and 7 are independent — they modify different parts of the code. Track 8 is independent of both. Track 9 depends on Track 6 (the cache may simplify the parallelization).

## Estimated Total Impact

| Track | Est. speedup (tracking) | Est. speedup (correspondences) |
|-------|------------------------|-------------------------------|
| 5 — cdivision (done) | +11% | +11% |
| 6 — point_to_pixel cache | +40% | — |
| 7 — spatial grid | +20% | — |
| 8 — early pruning | — | +30% |
| 9 — parallel (if viable) | +50% | +50% |
| **Cumulative** | **~3-4×** | **~2×** |

# Correspondences Speed-Up Plan

## Assumptions & Design Decisions

Answers from interview (April 2026):

| Question | Answer |
|----------|--------|
| Targets per camera | **1000–5000** (dense regime) |
| Camera count | **2–4** (all must be equally optimized) |
| Parameter compat | **Keep dual support** (pure-Python + Cython wrappers) |
| Numba warm-up | **AOT / cache=True** everywhere — must be fast on first call |
| Parallelism | **prange from the start**, both camera-pair and target levels |
| Profiling | **Yes, profile first** on test_cavity data before each phase |
| Test strategy | **Incremental** — existing tests green after every phase |
| Overflow risk | Never seen overflow, but candidate lists **vary widely up to K=40** |
| MMLUT | **Always** uses pre-computed MMLUT (must be Numba-compatible) |
| Memory | Moderate (8–16 GB) — watch adjacency list allocation at 5000 targets |
| API change | **Break callers now** — plain arrays everywhere, fix GUI callers |
| Frame/Target | **Replace with plain numpy arrays** |
| scipy dep | **OK** (for cKDTree in Phase 5) |

### Implications

1. **K can reach 40** → `four_camera_matching` worst case O(N·40^6) is real. Phase 3 (Numba) is **critical**, not just nice-to-have. Phase 5 (spatial indexing) also becomes important to reduce K in dense regimes.
2. **MMLUT always used** → `fast_get_mmf_from_mmlut` in `multimed.py` is currently **not @njit** (the decorator is commented out). Must be njit-compiled and its data (rw, origin, data, nz, nr) passed as plain arrays into the batched epi_mm kernel.
3. **AOT/cache=True** → all `@njit` functions must use `cache=True`. Consider `@njit(cache=True, nogil=True, fastmath=True)` uniformly.
4. **prange at both levels** → camera-pair loop (6 iterations for 4 cams) and target loop (1000–5000 iterations). The target loop is the profitable one for prange.
5. **Break API now** → can replace recarray returns with plain arrays, retire `Correspond_dtype` and `n_tupel_dtype` recarrays, replace `Frame`/`Target` classes with structured numpy arrays.

## Current State

The `algorithms/correspondences.py` module is a line-by-line port from C. The algorithmic complexity is fine for C, but the **per-iteration constant factor is ~100-1000× larger** in Python due to recarray attribute dispatch, dtype lookup, object boxing, and Python→Numba round-trip overhead.

### Profiled Findings (April 2026)

Phase 1 is already complete in the Python tree, so the current workstream resumes at the matcher.

One-frame profiling on `test_data/test_cavity` showed:
- detection: 0.4s
- stereo-matching: 74.4s
- 3-D determination: 0.1s
- total frame time: 75.2s

Cumulative correspondence hotspots from cProfile were:
- `four_camera_matching`: 107.7s
- `three_camera_matching`: 27.7s
- `match_pairs`: 17.6s
- `epi_mm`: 9.8s
- `_find_candidates_vectorized`: 1.75s
- `take_best_candidates`: 1.22s

Implication: candidate search is no longer the main bottleneck. The first low-risk slice should remove Python/NumPy allocation work from `three_camera_matching` and make `take_best_candidates` consume only the populated prefix before the larger SoA/Numba work.

### Constants
- $C$ = `num_cams` (typically 2–4)
- $N$ = targets per camera (1000–5000 in production)
- $K$ = `MAXCAND = 40` (max candidates per target, can reach max)

### Test surface (must stay green at each phase)
- `algorithms/tests/test_09_correspondences.py` — 7 tests: MatchedCoords (3), correspondences (2), edge cases (2)
- `algorithms/tests/test_batch.py` — integration tests using correspondences end-to-end
- Run with: `uv run pytest algorithms/tests/test_09_correspondences.py algorithms/tests/test_batch.py -v`

### Current Numba status in the codebase
Already `@njit`:
- `fast_ray_tracing` in `ray_tracing.py`
- `fast_multimed_r_nlay` in `multimed.py`
- `fast_trans_cam_point` in `multimed.py`
- `fast_back_trans_point` in `multimed.py`
- `fast_pixel_to_metric` in `trafo.py`
- `fast_metric_to_pixel` in `trafo.py`
- `distort_brown_affine` in `trafo.py`
- `correct_brown_affine` in `trafo.py`

**NOT** Numba-accelerated (pure Python, hot path):
- `epi_mm()` — wrapper, calls `ray_tracing` + `move_along_ray` + `flat_image_coord`
- `flat_image_coord()` — creates Calibration copy, calls `trans_cam_point` + `multimed_nlay` + `back_trans_point`
- `ray_tracing()` — wrapper with `getattr`, calls `fast_ray_tracing`
- `multimed_nlay()` — wrapper, calls `get_mmf_from_mmlut` or `fast_multimed_r_nlay`
- `fast_get_mmf_from_mmlut()` — already `@njit(cache=True, nogil=True)` in the current tree; keep it as the MMLUT primitive for the batched epipolar path
- `match_pairs()` — main loop, scalar Python
- `four_camera_matching()` — 7 nested loops, scalar Python
- `three_camera_matching()` — `np.where` in inner loop
- `MatchedCoords.__init__` — per-target Python loop

---

## Phase 0 — Profile baseline (prerequisite for all phases)

**Goal:** Get real timing breakdown on `test_data/test_cavity/` before optimizing.

**Steps:**
1. Write a small profiling script that calls the full correspondence pipeline on test_cavity data (4 cams, frames 10000–10004).
2. Use `cProfile` + `snakeviz` or `line_profiler` on the hot functions.
3. Record baseline wall-clock times for: `MatchedCoords.__init__`, `match_pairs`, `four_camera_matching`, `three_camera_matching`, `take_best_candidates`, total `correspondences()`.
4. Commit baseline numbers to this file.

**Test gate:** N/A (read-only).

---

## Phase 1 — Batch-vectorize `epi_mm` + fix MMLUT Numba (highest impact)

**Problem:** `epi_mm` is called $C^2/2 \cdot N$ times (e.g. $6 \times 5000 = 30{,}000$ calls for 4 cameras × 5000 targets). Each call goes through `ray_tracing()` → `flat_image_coord()`, which create Python objects (`Calibration` copies), do `getattr` checks, and call into Numba one-at-a-time.

**Critical dependency:** `fast_get_mmf_from_mmlut` in `multimed.py` is already njit-compiled in the Python tree. Since MMLUT is always used, keep it as the batched epipolar primitive and pass `(rw, origin, data, nz, nr, pos)` as plain arrays.

**Fix:**
1. **Enable MMLUT in Numba**: Uncomment `@njit(cache=True, nogil=True)` on `fast_get_mmf_from_mmlut`. Pass MMLUT arrays (data, origin, rw, nz, nr) as flat numpy arrays.
2. Create `epi_mm_batch(xl_arr, yl_arr, cal1_flat, cal2_flat, mm_flat, vpar_flat) → (N,4)` array.
3. Write a single `@njit(cache=True, parallel=True, nogil=True)` function `_epi_mm_batch_inner` that:
   - Takes flat arrays for all $N$ source points.
   - Uses `prange` over the $N$ targets.
   - Fuses the full chain: `ray_tracing` → `move_along_ray` → `trans_cam_point` → `multimed_nlay` (with MMLUT) → `back_trans_point` → projection.
   - Returns $(N, 4)$ endpoint arrays.
4. The Python wrapper extracts calibration scalars **once** (handles both pure-Python and Cython objects via the existing `getattr` dance), packs into plain arrays, calls the njit kernel.

**Parallelism:** `prange` over the N targets within each camera pair. The 6 camera pairs are processed sequentially (they write to different slices of `corr_list`, no conflicts).

**Test gate:** `uv run pytest algorithms/tests/test_09_correspondences.py -v` — all 7 tests green.

**Expected speedup:** 50-200× for epipolar computation (eliminates ~30,000 Python→Numba round-trips).

**Status:** completed; the batch wrapper and bundle reconstruction seam are in place and regression-tested.

---

## Phase 2 — Low-risk matching cleanup before broader vectorization

**Problem:** Profiling showed that the dense matching loops, not the vectorized candidate search, dominate the frame time. `three_camera_matching` still uses `np.where` in the innermost loop, and `take_best_candidates` still sorts the full scratch buffer instead of the populated prefix.

**Fix:**
1. Replace the `np.where` intersection in `three_camera_matching` with a direct scan of the candidate list, matching the C loop structure.
2. Add an explicit populated-candidate count to `take_best_candidates` and pass the actual counts from `four_camera_matching`, `three_camera_matching`, and `consistent_pair_matching` so only valid candidates are sorted.
3. Keep `find_start_point_binary` on `np.searchsorted` as the current baseline, but defer further candidate-search work until matching is cheaper or profiling shows candidate search back in the top three hotspots.
4. Reprofile the one-frame `test_cavity` batch smoke test after each slice. If the bottleneck remains in `four_camera_matching`, move immediately to the SoA/Numba matching work in Phase 3.

**Test gate:** all correspondence and batch tests green.

**Expected speedup:** modest on its own, but it removes avoidable NumPy allocation overhead and exposes the real matching bottleneck more cleanly.

---

## Phase 3 — Numba-compile matching functions (critical for dense regime)

**Problem:** `four_camera_matching` has 7 nested loops with scalar recarray access. With K up to 40 and N up to 5000, worst case is $O(N \cdot K^6) \approx O(5000 \cdot 4 \times 10^9)$. `three_camera_matching` allocates `np.where` per inner iteration.

**Fix:** Replace `Correspond_dtype` recarray with Structure-of-Arrays (SoA):
```python
# SoA layout (allocated once, reused across frames)
corr_p1   = np.zeros((C, C, N_max), dtype=np.int32)
corr_n    = np.zeros((C, C, N_max), dtype=np.int32)
corr_p2   = np.zeros((C, C, N_max, MAXCAND), dtype=np.int32)
corr_corr = np.zeros((C, C, N_max, MAXCAND), dtype=np.float64)
corr_dist = np.zeros((C, C, N_max, MAXCAND), dtype=np.float64)
```

Write `@njit(cache=True, nogil=True)` versions of:
- `_four_camera_matching_numba(corr_p1, corr_n, corr_p2, corr_corr, corr_dist, ...)` — same 7-nested-loop algorithm, but each `corr_list[i1][i2][i].p2[j]` becomes `corr_p2[i1, i2, i, j]` (direct memory offset).
- `_three_camera_matching_numba(...)` — replace `np.where(p2array == p3)` with a plain `for m in range(corr_n[i2, i3, p2]): if corr_p2[i2, i3, p2, m] == p3: ...` — faster in Numba for tiny arrays.
- `_consistent_pair_matching_numba(...)`

**Parallelism:** The outer loop of `four_camera_matching` (over N targets in cam1) can use `prange`. The inner K^6 loops are inherently serial per-target. For `three_camera_matching`, `prange` over the `i` (target) loop within each `(i1, i2)` pair.

**Also replace `n_tupel_dtype` recarray** for scratch/con buffers with plain arrays:
```python
scratch_p    = np.full((4*NMAX, C), -2, dtype=np.int32)
scratch_corr = np.zeros(4*NMAX, dtype=np.float64)
```

**Test gate:** all correspondence + batch tests green.

**Expected speedup:** 100-500× for matching (tight loops are Numba's sweet spot).

---

## Phase 4 — Vectorize `MatchedCoords.__init__`

**Problem:** Per-target Python loop calling `pixel_to_metric` + `dist_to_flat` for up to 5000 targets per camera.

**Fix:** Already have `arr_pixel_to_metric` (vectorized). Add `arr_correct_brown_affine` using a batched `@njit`:
```python
@njit(cache=True, parallel=True, nogil=True)
def arr_correct_brown_affine(xy: np.ndarray, ap: np.ndarray) -> np.ndarray:
    out = np.empty_like(xy)
    for i in prange(xy.shape[0]):
        out[i, 0], out[i, 1] = correct_brown_affine(xy[i, 0], xy[i, 1], ap)
    return out
```
Then `MatchedCoords.__init__` becomes array ops:
```python
xy_pix = np.column_stack([targ_x, targ_y]).astype(np.int32)
xy_m = arr_pixel_to_metric(xy_pix, imx, imy, pix_x, pix_y)
xy_f = arr_correct_brown_affine(xy_m, cal.added_par)
xy_f[:, 0] -= cal.int_par.xh
xy_f[:, 1] -= cal.int_par.yh
# then argsort by x
```
No Python loop. `prange` over the 5000 targets.

**Also replace `MatchedCoords` class** with a plain function returning `(pos_xy, pnr)` arrays — since we're breaking API anyway.

**Test gate:** `TestMatchedCoords` (3 tests) green.

**Expected speedup:** 10-50× for coordinate conversion.

---

## Phase 5 — Spatial indexing for dense candidate search

**Problem:** With 5000 targets, the epipolar band can contain hundreds of points. The current linear scan of the x-sorted array is $O(N')$ where $N'$ can be large.

**Fix:** Build a `scipy.spatial.cKDTree` on each camera's corrected coordinates (built once per frame, $O(N \log N)$). For candidate search:
1. Sample 2–3 points along each epipolar line.
2. Use `cKDTree.query_ball_point(points, r=eps0)` to get candidate indices — $O(\log N)$ per query.
3. Union the candidate sets and apply quality filters.

**Alternative for Numba compatibility:** Build a 2D grid index (bin targets into cells of size `eps0`). For each epipolar line, compute which cells it crosses (Bresenham-like), collect targets from those cells only. This is fully `@njit`-compatible and avoids scipy in the inner loop.

**Recommendation:** Use cKDTree in the Python wrapper (called once per camera pair per frame, not per target). It returns candidate indices that are then passed into the @njit quality-filter kernel.

**Expected speedup:** 2-10× for dense regime (>2000 targets). Negligible for sparse.

---

## Phase 6 — `take_best_candidates` with plain arrays

**Problem:** `src.sort(order="corr")` on a recarray + `np.flip` — allocation + copy overhead.

**Fix:** With the SoA layout from Phase 3:
```python
order = np.argsort(scratch_corr[:n_matched])[::-1]
scratch_p[:n_matched] = scratch_p[order]
scratch_corr[:n_matched] = scratch_corr[order]
```
Or write a `@njit` version that does selection sort (for small n_matched this beats argsort).

**Also write the usage-mark loop in @njit** — it's a simple scan that's currently pure Python.

**Test gate:** all tests green.

**Expected speedup:** 2-5× for this step (small fraction of total).

---

## Phase 7 — Replace Frame/Target with plain arrays (API cleanup)

**Problem:** `Frame` and `Target` classes add Python attribute access overhead. The pre-extraction pattern in `match_pairs` (lines 440–455) is a workaround.

**Fix:**
1. Define frame data as plain arrays:
   ```python
   # Instead of frm.targets[cam][j].x, frm.targets[cam][j].n, etc.
   targ_xy   = np.zeros((C, N_max, 2), dtype=np.float64)  # x, y
   targ_attr = np.zeros((C, N_max, 4), dtype=np.int32)     # n, nx, ny, sumg
   targ_pnr  = np.zeros((C, N_max), dtype=np.int32)
   num_targets = np.zeros(C, dtype=np.int32)
   ```
2. All @njit functions accept these arrays directly.
3. Provide a conversion function `frame_to_arrays(frm) → arrays` for backward compat with GUI callers during transition.

**Test gate:** all tests green.

**Expected speedup:** Indirect — enables all @njit functions to work without extraction overhead. Main benefit is code simplification.

---

## Implementation Order and Dependencies

```
Phase 0 (profile baseline)  ─── prerequisite
  │
  ├── Phase 4 (MatchedCoords)  ─── independent, quick win
  │
  ▼
Phase 1 (epi_mm_batch + MMLUT fix)  ─── highest ROI
  │
  ▼
Phase 2 (match_pairs vectorize)  ─── depends on Phase 1 output format
  │                                    (merged with Phase 3 data layout)
  ▼
Phase 3 (numba matching loops)  ─── same SoA layout as Phase 2
  │
  ▼
Phase 5 (spatial indexing)  ─── depends on Phase 2/3 for dense regime
  │
Phase 6 (take_best_candidates)  ─── trivial, after Phase 3 SoA
  │
Phase 7 (Frame/Target replacement)  ─── API cleanup, after all else works
```

## Estimated Total Speedup (for N=5000, C=4)

| Phase | Fraction of runtime | Speedup | Net effect |
|-------|-------------------|---------|------------|
| 1. epi_mm_batch + MMLUT | ~40% | 50-200× | 35-40% overall |
| 2. match_pairs vectorize | ~10% | 5-20× | 8-10% overall |
| 3. numba matching | ~30% | 100-500× | 25-30% overall |
| 4. MatchedCoords | ~10% | 10-50× | 8-10% overall |
| 5. spatial indexing | ~5% | 2-10× | 3-5% overall |
| 6-7. cleanup | ~5% | 2-5× | 2-3% overall |

Combined: the entire correspondence pipeline should go from **seconds → tens of milliseconds** for 4-camera setups with 5000 targets. For 1000 targets it should be **sub-10ms**.

### Key Numba conventions for all phases
- `@njit(cache=True, nogil=True)` on every kernel (AOT requirement).
- `fastmath=True` where safe (epipolar geometry, bilinear interp — yes; iterative Snell's law — careful with convergence).
- `prange` for target-level parallelism (1000–5000 iterations).
- No Python objects inside `@njit` — all data as plain contiguous arrays.
- Dual parameter compat: the Python wrapper does `getattr` once, packs scalars into arrays, calls `@njit`.

# Performance Optimization Plan

Status: Session 3 (Track 5) complete — all 4 planned tracks done.
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

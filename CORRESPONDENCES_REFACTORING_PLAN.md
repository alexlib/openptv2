# Correspondences Speed-Up Plan

## Context

- `algorithms/correspondences.py` is a line-by-line port from `lib/src/correspondences.c`
- Same algorithm, but Python is ~100-300× slower due to recarray field access (~200ns vs ~2ns), Python loop overhead, and 30K function-call round-trips in `match_pairs`
- Target: **< 1 second per frame** for 4 cameras, 5000 targets (matching C)

### Profiled hotspots (test_cavity, 4 cams, ~1000 targets)

| Function | Time | Root cause |
|----------|------|------------|
| `four_camera_matching` | 107.7s | recarray access in 78M loop iterations |
| `three_camera_matching` | 27.7s | recarray access in nested loops |
| `match_pairs` | 17.6s | 30K scalar `epi_mm` calls (batch kernel exists but unused) |
| `epi_mm` (cumulative) | 9.8s | Python→Numba round-trips |
| `_find_candidates_vectorized` | 1.75s | 30K temporary array allocations |
| `take_best_candidates` | 1.22s | recarray sort |

### Constants
- $C$ = `num_cams` (2–4), $N$ = targets/camera (1000–5000), $K$ = `MAXCAND = 40`

### Already `@njit` (building blocks ready)
`fast_ray_tracing`, `fast_flat_image_coord_raw`, `move_along_ray`, `fast_get_mmf_from_mmlut`, `find_start_point_binary`, `quality_ratio`, `correct_brown_affine`, `fast_pixel_to_metric`, `_epi_mm_batch_inner` (parallel batch kernel)

### Test surface (must stay green at each phase)
- `uv run pytest algorithms/tests/test_09_correspondences.py -v` — 10 tests
- `uv run pytest algorithms/tests/test_batch.py -v` — 12 tests (5 pre-existing failures, unrelated)

---

## Phase 0 — Profile baseline: DONE

## Phase 1 — Batch epi_mm kernel: DONE
`_epi_mm_batch_inner` exists in `epi.py` but `match_pairs()` still calls scalar `epi_mm()`. Phase 3A fixes this.

## Phase 2 — Safe matching cleanup: DONE (commit 7e1c73f)
- Local variable caching in `four_camera_matching` and `three_camera_matching`
- Replaced `np.where` with direct scan in `three_camera_matching`
- **REVERTED (broke tests, do NOT reattempt):** `np.searchsorted` in `find_start_point_binary`, upper-bound searchsorted in `_find_candidates_vectorized`, `num_cands` prefix + `np.flip` removal in `take_best_candidates`

---

## Phase 3 — SoA + Numba pipeline (THE CRITICAL PHASE)

Replace the entire hot path with `@njit` functions operating on plain numpy arrays.

### Phase 3A — SoA adjacency + fused fill kernel

**Replace** `Correspond_dtype` recarray with 4 flat arrays:
- `corr_n[C, C, N_max]` (int32) — candidate count
- `corr_p2[C, C, N_max, MAXCAND]` (int32) — candidate target indices
- `corr_corr[C, C, N_max, MAXCAND]` (float64) — correlation scores
- `corr_dist[C, C, N_max, MAXCAND]` (float64) — epipolar distances

**Write** `_fill_adjacency_pair()` as `@njit(cache=True, parallel=True, nogil=True)`:
- Fuses `epi_mm` + `find_candidate` into a single compiled loop per camera pair
- Uses `prange` over source targets
- Inlines the epi_mm chain: `fast_ray_tracing` → `move_along_ray` → `fast_flat_image_coord_raw` (same as `_epi_mm_batch_inner` does)
- Inlines C-style binary search + linear scan for candidates (reuse `find_start_point_binary`, inline `quality_ratio`)
- Writes directly to pre-allocated SoA output arrays — zero temporary allocations

**Write** `match_pairs_soa()` Python wrapper:
- Extract calibration/multimedia/volume params to plain arrays once (use `epi_mm_batch_inputs` pattern from `epi.py`)
- Pre-extract corrected-coord and target-property arrays (already done in current `match_pairs`)
- Call `_fill_adjacency_pair()` for each of the $C(C-1)/2$ pairs
- Result: 6 Numba calls replace 30K Python function calls

**Test:** Compare output against current `match_pairs` for test_cavity data — adjacency must match exactly.

### Phase 3B — Numba matching kernels

**Write** `@njit(cache=True, nogil=True)` versions of the matching functions operating on the SoA arrays from 3A:
- `_four_camera_matching_numba(corr_n, corr_p2, corr_corr, corr_dist, ...)` — same 7-nested-loop algorithm as C, but `corr_list[i1][i2][i].p2[j]` becomes `corr_p2[i1, i2, i, j]` (direct memory offset)
- `_three_camera_matching_numba(...)` — same nested loops, direct scan for p3 in pair_23 candidates
- `_consistent_pair_matching_numba(...)` — same n==1 filter

**Scratch buffers** — replace `n_tupel_dtype` recarray:
- `scratch_p[4*NMAX, 4]` (int32), `scratch_corr[4*NMAX]` (float64)

**Outer loop stays serial** (writes to shared scratch). At 3ns/iter × 78M iterations = ~0.23s — fast enough.

**Test:** Output matches current matching functions exactly for test_cavity.

### Phase 3C — Numba take_best_candidates

**Write** `_take_best_candidates_numba()` as `@njit`:
- Sort `scratch_corr` descending (insertion sort for <100K, or `np.argsort` which Numba supports)
- Greedy scan: skip candidates with any already-used target (`tusage[cam, tnum] > 0`), mark accepted targets

**Test:** Same greedy selection results as current `take_best_candidates`.

### Phase 3D — Vectorized MatchedCoords

**Write** `_matched_coords_inner()` as `@njit(parallel=True)`:
- `prange` over targets, inline `pixel_to_metric` + `correct_brown_affine` per target
- Return `(out_x, out_y)` arrays, then `np.argsort` by x

**Test:** `TestMatchedCoords` (3 tests) green.

### Phase 3E — Wire together

**Write** `correspondences_soa()` orchestrator:
1. Extract all params to plain arrays once
2. Allocate SoA adjacency + scratch buffers
3. Call `_fill_adjacency_pair` per pair (3A)
4. Call `_four_camera_matching_numba` → `_take_best_candidates_numba` (3B+3C)
5. Call `_three_camera_matching_numba` → `_take_best_candidates_numba`
6. Call `_consistent_pair_matching_numba` → `_take_best_candidates_numba`

**Keep** existing `correspondences()` as fallback. Add backward-compat wrapper that converts plain-array output to `n_tupel_dtype` recarrays for GUI/tracking callers.

**Test:** All 10 correspondence tests green. Run both implementations on test_cavity, verify identical output.

---

## Phase 5 — Spatial indexing: DEFERRED
Binary search + linear scan at C speed is sufficient. Same algorithm C uses. Revisit only if profiling shows need after Phase 3.

## Phase 7 — Frame/Target replacement: DEFERRED
Phase 3A pre-extracts all arrays once. The extraction cost is ~1ms, irrelevant.

---

## Implementation order

```
Phase 0 (profile) ──────────────── DONE
Phase 1 (batch epi_mm kernel) ──── DONE (kernel exists, not wired)
Phase 2 (safe cleanup) ─────────── DONE (commit 7e1c73f)
  ▼
Phase 3A (SoA + fill adjacency) ← NEXT
Phase 3B (Numba matching)
Phase 3C (Numba take_best)
Phase 3D (vectorize MatchedCoords)
Phase 3E (wire together)
  ▼
Profile → verify < 1s per frame
```

## Numba conventions
- `@njit(cache=True, nogil=True)` on every kernel. `fastmath=True` where safe.
- `prange` in `_fill_adjacency_pair` (1000–5000 targets). Serial in matching loops.
- No Python objects inside `@njit`. All data as plain contiguous arrays.
- No temporary array allocations inside loops.
- `find_start_point_binary` — keep the hand-rolled binary search (NOT `np.searchsorted`).

## Risk mitigation
- Existing `correspondences()` stays as fallback.
- Comparison test: both implementations on test_cavity, verify identical output.
- Each sub-phase has an explicit test gate.

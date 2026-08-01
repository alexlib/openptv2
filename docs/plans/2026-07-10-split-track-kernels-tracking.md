# Plan: Split `track_kernels_tracking.py` into Focused Sub-Modules

**Date:** 2026-07-10  
**File:** `src/openptv2/algorithms/track_kernels_tracking.py` (3662 lines)  
**Goal:** Easier maintenance, test coverage, and Cython optimization by splitting into
4 focused files with clear responsibilities.

---

## Current State

### Function inventory

| Lines | Size | Function | Exported? |
|-------|------|----------|-----------|
| 92–153 | 62 | `_angle_acc_out` | no |
| 154–233 | 80 | `_multimed_r_nlay_1layer` | no |
| 234–501 | 268 | `_point_to_pixel_out` | no |
| 502–629 | 128 | `candsearch_in_pix_fast_nogil` | yes (via batch) |
| 630–896 | 267 | `_sorted_candidates_fast_out_nogil` | no |
| 897–1069 | 173 | `_ray_tracing_out` | no |
| 1070–1253 | 184 | `_point_position_out` | no |
| 1254–1336 | 83 | `_candsearch_in_pix_rest_nogil` | no |
| 1337–1365 | 29 | `_pixel_to_metric_out` | no |
| 1366–1426 | 61 | `_dist_to_flat_out` | no |
| 1427–1541 | 115 | `assess_new_position_fast_nogil` | yes |
| 1542–2369 | 828 | `_trackcorr_particle_fast` | no |
| 2370–2856 | 487 | `trackcorr_loop_fast` | yes |
| 2857–3331 | 475 | `trackback_loop_fast` | yes |
| 3332–3386 | 55 | `_find_closest_in_3d` | no |
| 3387–3661 | 275 | `track3d_loop_fast` | yes |

### Intra-file call graph

```
_point_to_pixel_out              <- _multimed_r_nlay_1layer
_sorted_candidates_fast_out_nogil <- _point_to_pixel_out, candsearch_in_pix_fast_nogil
_point_position_out              <- _ray_tracing_out
assess_new_position_fast_nogil   <- _candsearch_in_pix_rest_nogil, _dist_to_flat_out,
                                    _pixel_to_metric_out
_trackcorr_particle_fast         <- __sync_bool_compare_and_swap, _angle_acc_out,
                                    _point_position_out, _point_to_pixel_out,
                                    _sorted_candidates_fast_out_nogil,
                                    assess_new_position_fast_nogil
trackcorr_loop_fast              <- _trackcorr_particle_fast
trackback_loop_fast              <- _angle_acc_out, _point_position_out, _point_to_pixel_out
track3d_loop_fast                <- _find_closest_in_3d
```

### External consumers

- `track_kernels.py` (shim) imports: `trackcorr_loop_fast`, `trackback_loop_fast`, `track3d_loop_fast`
- `track.py` imports everything via the shim
- `setup.py` lists `track_kernels_tracking` + appends `cas_shim.c` for that module

---

## Proposed Split: 4 New Files

### A. `track_kernels_pixel.py` (~700 lines)

**Responsibility:** all per-camera pixel-space math — projection, candidate search, multimedia refraction

Functions:
- `_multimed_r_nlay_1layer`
- `_point_to_pixel_out`
- `candsearch_in_pix_fast_nogil`
- `_candsearch_in_pix_rest_nogil`
- `_pixel_to_metric_out`
- `_dist_to_flat_out`
- `_sorted_candidates_fast_out_nogil`

Why together: `_sorted_candidates_fast_out_nogil` calls both `_point_to_pixel_out` and
`candsearch_in_pix_fast_nogil` — keeping all three in one file avoids circular imports.
No prange/GIL: easy to unit-test with synthetic cal arrays.

### B. `track_kernels_position.py` (~430 lines)

**Responsibility:** 3D position reconstruction from multi-camera ray tracing

Functions:
- `_ray_tracing_out`
- `_point_position_out`
- `assess_new_position_fast_nogil`

Why together: `_point_position_out` calls `_ray_tracing_out`; `assess_new_position_fast_nogil`
calls pixel helpers (imported from A). All `@cython.ccall`, no prange — unit-testable.

### C. `track_kernels_corr.py` (~1900 lines)

**Responsibility:** forward + backward 2D→3D tracking loops with CAS-atomic particle linking

Contents:
- `__sync_bool_compare_and_swap` (CAS atomic — compiled and pure-Python stubs)
- `_angle_acc_out`
- `_trackcorr_particle_fast` (828-line inner particle kernel)
- `trackcorr_loop_fast` (prange outer loop)
- `trackback_loop_fast` (backward pass)

Why together: prange + CAS + `_angle_acc_out` are tightly coupled; keeping them in one
compilation unit allows inlining of the inner kernel.  
`setup.py` `cas_shim.c` linkage must move here.

### D. `track_kernels_track3d.py` (~330 lines)

**Responsibility:** stereo-3D tracking loop (position-space only, no camera projections)

Functions:
- `_find_closest_in_3d`
- `track3d_loop_fast`

Why separate: zero shared functions with corr loops, independent optimization profile,
easy to cover fully.

---

## Implementation Phases

### Phase 0 — Read & document imports (no code changes)

**Goal:** confirm exact boilerplate and imports needed by each new file before writing any.

Checklist:
- [ ] Read lines 1–91 of `track_kernels_tracking.py` (docstring + imports + cython.declare + CAS stub)
- [ ] For each proposed file, list the exact `from X import Y` lines needed:
  - pixel.py needs: `cython`, `numpy`, libc.math (`sqrt`, `asin`, `acos`, `atan`), `searchquader_fast` from `track_kernels_geom`, `_sorted_candidates_fast_out` from `track_kernels_search`
  - position.py needs: `cython`, `numpy`, libc.math, pixel helpers from pixel.py, `point_position_fast` from `track_kernels_transform`
  - corr.py needs: `cython`, `numpy`, `prange`/`threadid`, imports from pixel.py + position.py + geom + search + transform
  - track3d.py needs: `cython`, `numpy`, libc.math, `_point_position_out` from position.py (confirm by reading body)
- [ ] Confirm which `cython.declare` constants are needed by each file (currently all are in tracking.py lines 63–82)

**Success:** zero invented imports in subsequent phases.

---

### Phase 1 — Create `track_kernels_pixel.py`

1. Create `src/openptv2/algorithms/track_kernels_pixel.py`:
   - Docstring: `"""Per-camera pixel-space math: projection, candidate search, multimedia refraction."""`
   - Copy exact header boilerplate from tracking.py lines 1–91 (excluding CAS stub)
   - Functions (cut from tracking.py): `_multimed_r_nlay_1layer`, `_point_to_pixel_out`, `candsearch_in_pix_fast_nogil`, `_candsearch_in_pix_rest_nogil`, `_pixel_to_metric_out`, `_dist_to_flat_out`, `_sorted_candidates_fast_out_nogil`

2. In `track_kernels_tracking.py`, replace function bodies with:
   ```python
   from .track_kernels_pixel import (
       _multimed_r_nlay_1layer,
       _point_to_pixel_out,
       candsearch_in_pix_fast_nogil,
       _candsearch_in_pix_rest_nogil,
       _pixel_to_metric_out,
       _dist_to_flat_out,
       _sorted_candidates_fast_out_nogil,
   )
   ```

3. Add `"track_kernels_pixel"` to `ALGORITHMS_MODULES` in `setup.py` before `"track_kernels_tracking"`.

**Verification:**
```bash
uv run python -c "from openptv2.algorithms.track_kernels_pixel import candsearch_in_pix_fast_nogil; print('ok')"
uv run pytest tests/unit/test_track_kernels_tracking_coverage.py -q --tb=short
```

Anti-patterns:
- Do NOT copy the CAS stub (`__sync_bool_compare_and_swap`) to pixel.py
- Do NOT remove `cython.declare` block from tracking.py yet

---

### Phase 2 — Create `track_kernels_position.py`

1. Create `src/openptv2/algorithms/track_kernels_position.py`:
   - Docstring: `"""3D position reconstruction via multi-camera ray tracing and assess_new_position."""`
   - Header boilerplate (cython, numpy, libc.math guards)
   - Imports: `from .track_kernels_pixel import _candsearch_in_pix_rest_nogil, _dist_to_flat_out, _pixel_to_metric_out`
   - Functions: `_ray_tracing_out`, `_point_position_out`, `assess_new_position_fast_nogil`

2. In `track_kernels_tracking.py`, replace with:
   ```python
   from .track_kernels_position import (
       _ray_tracing_out,
       _point_position_out,
       assess_new_position_fast_nogil,
   )
   ```

3. Add `"track_kernels_position"` to `ALGORITHMS_MODULES` in `setup.py` after pixel, before tracking.

**Verification:**
```bash
uv run pytest tests/unit/test_track_kernels_tracking_coverage.py -q --tb=short
```

Write `tests/unit/test_track_kernels_position_coverage.py` with:
- `_ray_tracing_out` with synthetic cal arrays (2 cams)
- `_point_position_out` with zero targets
- `assess_new_position_fast_nogil` with dummy candidate arrays

Target: ≥90% coverage on the ~430-line file.

---

### Phase 3 — Create `track_kernels_track3d.py`

1. Create `src/openptv2/algorithms/track_kernels_track3d.py`:
   - Docstring: `"""Stereo-3D tracking loop — position-space only, no camera projections."""`
   - Header boilerplate (cython, numpy, libc.math guards)
   - Import: confirm `track3d_loop_fast` body for what it needs from position.py
   - Functions: `_find_closest_in_3d`, `track3d_loop_fast`

2. In `track_kernels_tracking.py`, replace with:
   ```python
   from .track_kernels_track3d import track3d_loop_fast
   ```

3. Add `"track_kernels_track3d"` to `ALGORITHMS_MODULES` in `setup.py`.

4. Update `track_kernels.py` (shim) to import `track3d_loop_fast` from `.track_kernels_track3d` directly.

**Verification:**
```bash
uv run pytest tests/unit/test_track_kernels_tracking_coverage.py -q
```

Write `tests/unit/test_track_kernels_track3d_coverage.py`:
- `_find_closest_in_3d` with random 3D point arrays
- `track3d_loop_fast` stub (all-zero inputs cover control-flow branches)

Target: ≥90% coverage on the ~330-line file.

---

### Phase 4 — Create `track_kernels_corr.py`

1. Create `src/openptv2/algorithms/track_kernels_corr.py`:
   - Docstring: `"""Forward and backward 2D→3D tracking loops with CAS-atomic particle linking."""`
   - Full header boilerplate **including CAS stub** (both compiled and pure-Python versions)
   - Full `cython.declare` block for all 9 typed constants (`PT_UNUSED`, `POSI_K`, etc.)
   - Imports from sub-modules: pixel.py + position.py + search + geom + transform
   - Functions: `_angle_acc_out`, `_trackcorr_particle_fast`, `trackcorr_loop_fast`, `trackback_loop_fast`

2. Update `setup.py`:
   - Add `"track_kernels_corr"` to `ALGORITHMS_MODULES`
   - Move `cas_shim.c` linkage from `track_kernels_tracking` to `track_kernels_corr`:
     ```python
     if mod == "track_kernels_corr":
         sources.append("src/openptv2/algorithms/cas_shim.c")
     ```

3. Update `track_kernels.py` (shim):
   ```python
   from .track_kernels_corr import trackcorr_loop_fast, trackback_loop_fast
   from .track_kernels_track3d import track3d_loop_fast
   ```

4. Gut `track_kernels_tracking.py` to a compatibility shim (~30 lines):
   ```python
   """Compatibility re-export — content split into focused sub-modules 2026-07-10."""

   from .track_kernels_pixel import (
       candsearch_in_pix_fast_nogil,
       assess_new_position_fast_nogil,
   )  # noqa: F401
   from .track_kernels_corr import trackcorr_loop_fast, trackback_loop_fast  # noqa: F401
   from .track_kernels_track3d import track3d_loop_fast  # noqa: F401
   ```
   Remove `cas_shim.c` linkage for `track_kernels_tracking` in `setup.py`.

**Verification:**
```bash
uv run python -c "
from openptv2.algorithms.track_kernels import trackcorr_loop_fast, trackback_loop_fast, track3d_loop_fast
print('shim ok')
"
uv run pytest tests/unit/ -q --tb=short -x
```

---

### Phase 5 — Coverage tests for new files

For each new file, one focused test file. Run with per-file `--cov`:

```bash
uv run pytest tests/unit/test_track_kernels_pixel_coverage.py \
  --cov=src/openptv2/algorithms/track_kernels_pixel --cov-report=term-missing -q
```

| New file | Test file | Target |
|----------|-----------|--------|
| `track_kernels_pixel.py` | `test_track_kernels_pixel_coverage.py` | ≥90% |
| `track_kernels_position.py` | `test_track_kernels_position_coverage.py` | ≥90% |
| `track_kernels_track3d.py` | `test_track_kernels_track3d_coverage.py` | ≥90% |
| `track_kernels_corr.py` | `test_track_kernels_corr_coverage.py` | ≥60% (prange sections require integration data) |

Existing `test_track_kernels_tracking_coverage.py` tests the shim re-exports — keep it.

---

### Phase 6 — Clean Cython rebuild + full suite

```bash
rm -f src/openptv2/algorithms/track_kernels_pixel*.{c,so} \
       src/openptv2/algorithms/track_kernels_position*.{c,so} \
       src/openptv2/algorithms/track_kernels_track3d*.{c,so} \
       src/openptv2/algorithms/track_kernels_corr*.{c,so} \
       src/openptv2/algorithms/track_kernels_tracking*.{c,so}
rm -rf build/
uv run python setup.py build_ext --inplace
uv run pytest -m "not slow" -q --tb=short
```

Expected: same pass count as pre-split (930 passed, 1 pre-existing failure on test_cavity).

Run badge update:
```bash
bash scripts/update_coverage_badge.sh
```

---

## Key Constraints

1. **No functional changes** — pure file reorganization, zero algorithm edits.
2. **`cas_shim.c` must follow `track_kernels_corr`** — provides real CAS atomic for prange.
3. **Circular-import rule:** pixel.py must NOT import from position.py; position.py must NOT import from corr.py.
4. **One phase at a time** — run `pytest` after each phase. Rollback = `git checkout -- src/` if a phase breaks.
5. **Do not use git stash** (concurrent sessions on same working tree).
6. **setup.py compile order:** pixel → position → track3d → corr → tracking (shim last).

---

## Final File Sizes (estimate)

| File | Lines | Cython profile |
|------|-------|----------------|
| `track_kernels_pixel.py` | ~700 | `@ccall`/`@cfunc`, no prange |
| `track_kernels_position.py` | ~430 | `@ccall`, no prange |
| `track_kernels_track3d.py` | ~330 | `@ccall`, no prange |
| `track_kernels_corr.py` | ~1900 | `prange` + CAS, hot path |
| `track_kernels_tracking.py` | ~30 | compatibility shim only |

# Cython Optimization Plan — Phase 2

## Goal
Eliminate Python tuple creation/return overhead in hot paths by adding
`_out` variants that write to pre-allocated memoryviews instead of returning tuples.

## Files & Current State

### track_kernels.py (4218 lines) — HOTTEST PATH
**Already done:**
- `_angle_acc_out` added after `angle_acc_fast` ✓
- `_ray_tracing_out` added after `_ray_tracing_fast` ✓
- `point_position_fast` converted to use `_ray_tracing_out` ✓

**Still needed:**
- Add `_pixel_to_metric_out` after `pixel_to_metric_fast`
- Add `_metric_to_pixel_out` after `metric_to_pixel_fast`
- Add `_dist_to_flat_out` after `dist_to_flat_fast`
- Convert `assess_new_position_fast` to use `_point_to_pixel_out`, `_pixel_to_metric_out`, `_dist_to_flat_out`
- Convert `trackcorr_loop_fast` callers (lines 2150-2264: `point_to_pixel_fast` → `_point_to_pixel_out`)
- Convert `trackback_loop_fast` callers (similar pattern)

### trafo.py (673 lines)
- Add `_distort_brown_affin_core_out`
- Add `_distort_brown_affin_out`
- Add `_correct_brown_affin_out`
- Add `_correct_brown_affine_exact_out`
- Add `_flat_to_dist_out`
- Add `_dist_to_flat_out`
- Add `_old_pixel_to_metric_out`
- Add `_old_metric_to_pixel_out`
- Convert `correct_brown_affin` to use `_distort_brown_affin_core_out`
- Convert `dist_to_flat` to use `_correct_brown_affine_exact_out`

### tracking_frame_buf.py (769 lines)
- Add `@cython.boundscheck(False)` / `@cython.wraparound(False)` to hot loops
- Convert `_sync_path_to_soa` and `_sync_soa_to_path` to use typed memoryviews directly

### correspondences.py (576 lines)
- Remove `@dataclass` from `@cython.cclass` (incompatible pattern)
- Use explicit `__init__` methods

## Execution Order
Phase 1: track_kernels.py additions + conversions (most impact)
Phase 2: trafo.py additions + conversions  
Phase 3: tracking_frame_buf.py optimizations
Phase 4: correspondences.py fixes
Phase 5: Compile, annotate, test

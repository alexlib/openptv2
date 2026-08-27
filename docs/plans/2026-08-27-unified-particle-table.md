# Unified Particle Table: merging 3D positions with per-camera 2D leaves

**Date**: 2026-08-27
**Branch**: `feature/tree-forest-tracking`

## Current format (disconnected)

```
rt_is / correspondences:     targets per camera:
┌──────┬──────┬──────┐      ┌──────┬────────┬────────┐
│  X   │  Y   │  Z   │      │  x   │   y    │  tnr   │
├──────┼──────┼──────┤      ├──────┼────────┼────────┤
│ 1.2  │ 3.4  │ 5.6  │      │ 100.1│ 200.2  │   0    │  cam 0
│ 2.3  │ 4.5  │ 6.7  │      │ 150.1│ 250.2  │   1    │  cam 0
└──────┴──────┴──────┘      │ 300.3│ 400.4  │   0    │  cam 1
                            │ 350.3│ 450.4  │   1    │  cam 1
cam_ids column links:       └──────┴────────┴────────┘
row 0 → cam0_pnr=0, cam1_pnr=0
row 1 → cam0_pnr=1, cam1_pnr=1
```

The link between 3D points and 2D leaves is implicit (row index in rt_is = particle index, cam_ids column maps to target index per camera). Leaves are discarded after correspondence.

## New format (unified)

Each row = one particle at one time step, with all available 2D leaves attached:

```
┌──────┬──────┬──────┬──────┬──────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ time │  id  │  X   │  Y   │  Z   │  x0    │  y0    │  x1    │  y1    │  x2    │  y2    │  ...
├──────┼──────┼──────┼──────┼──────┼────────┼────────┼────────┼────────┼────────┼────────┤
│  0   │  0   │ 1.2  │ 3.4  │ 5.6  │ 100.1  │ 200.2  │ 300.3  │ 400.4  │  NaN   │  NaN   │
│  0   │  1   │ 2.3  │ 4.5  │ 6.7  │ 150.1  │ 250.2  │ 350.3  │ 450.4  │  NaN   │  NaN   │
│  1   │  0   │ 1.3  │ 3.5  │ 5.7  │ 101.1  │ 201.2  │ 301.3  │ 401.4  │  NaN   │  NaN   │
└──────┴──────┴──────┴──────┴──────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

Columns:
- `time`: frame number (int)
- `id`: unique particle ID within the frame (int)
- `X`, `Y`, `Z`: 3D position in mm (float64)
- `x0`, `y0`: camera 0 2D position in pixels (float64, NaN = not detected)
- `x1`, `y1`: camera 1 2D position in pixels (float64, NaN = not detected)
- `x2`, `y2`: camera 2 2D position in pixels (float64, NaN = not detected)
- `x3`, `y3`: camera 3 2D position in pixels (float64, NaN = not detected)

For 3-camera setups, x2/y2/x3/y3 columns are all NaN.

## Storage in zarr

```python
# Group structure:
particle_table/
  time      : (N,) int32
  id        : (N,) int32
  xyz       : (N, 3) float64   # [X, Y, Z] in mm
  xy_cam    : (N, num_cams, 2) float64  # per-camera [x, y] in pixels, NaN if absent
```

Or as a single structured array:
```python
dtype = [
    ('time', 'i4'),
    ('id', 'i4'),
    ('xyz', 'f8', (3,)),
    ('xy_cam', 'f8', (num_cams, 2)),  # NaN where camera didn't detect
]
```

## N-dimensional distance for tracking

Given two particles A and B across consecutive frames:

```python
def ndim_distance(A, B, cam_mask, weights=None):
    """
    A, B: rows from the unified table
    cam_mask: which cameras are valid (not NaN)
    weights: optional per-dimension weights
    """
    # 3D distance
    d3 = sqrt((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

    # 2D distance per camera (only where both have detections)
    d2_total = 0
    d2_count = 0
    for cam in range(num_cams):
        if not.isnan(A.xy_cam[cam]).any() and not.isnan(B.xy_cam[cam]).any():
            dx = A.xy_cam[cam, 0] - B.xy_cam[cam, 0]
            dy = A.xy_cam[cam, 1] - B.xy_cam[cam, 1]
            d2_total += dx^2 + dy^2
            d2_count += 1

    if d2_count > 0:
        d2 = sqrt(d2_total / d2_count)  # average 2D distance
    else:
        d2 = 0

    # Combined: 3D + alpha * 2D
    return d3 + alpha * d2
```

The weight `alpha` controls the balance between 3D kinematic consistency and 2D leaf-path consistency. This is the key parameter that enables the tree-forest approach: a candidate that is close in 3D AND close in 2D gets a much lower cost than one that is close in 3D only.

## Implementation plan

### Phase 1: Data structure (this session)
1. Create `UnifiedParticleTable` class with read/write methods
2. Add zarr storage group `particle_table/`
3. Add conversion functions: current format → unified table
4. Keep backward compatibility with existing rt_is/targets format

### Phase 2: N-dimensional tracker
1. Modify `track3d_loop_fast` to accept unified table
2. Replace Euclidean 3D distance with N-dimensional distance
3. Add camera-weight parameter (alpha)
4. Test on wp1_10_images dataset

### Phase 3: Leaf-path consistency
1. Add leaf-path smoothness check to cold_start_gate
2. Distinguish "3D jump + leaf jump" (real event) from "3D jump + leaf smooth" (correspondence error)
3. Measure improvement on wp1 GT

## Files to create/modify

- `src/openptv2/storage/unified_table.py` — new UnifiedParticleTable class
- `src/openptv2/storage/zarr_store.py` — add read/write_particle_table methods
- `src/openptv2/storage/legacy.py` — add import/export for unified format
- `src/openptv2/algorithms/track_kernels_track3d.py` — add N-dimensional distance

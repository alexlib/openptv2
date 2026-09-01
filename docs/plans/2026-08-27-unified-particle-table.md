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
    ("time", "i4"),
    ("id", "i4"),
    ("xyz", "f8", (3,)),
    ("xy_cam", "f8", (num_cams, 2)),  # NaN where camera didn't detect
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

## Test results (wp1_10_images, res/run.zarr, 10 frames)

### Key finding: Two-phase approach (3D search + 2D ranking)

Don't add 2D to the KD-tree distance metric. Instead:
1. Use 3D KD-tree to find candidates within radius
2. Use 2D distances as cost in Hungarian assignment within candidates

| Noise | 3D-only rad=2 | 3D+2D rad=2 | 3D-only rad=5 | 3D+2D rad=5 |
|-------|--------------|-------------|--------------|-------------|
| 0.0mm | 0.966/0.981 | **0.969/0.983** | 0.734/0.857 | **0.848/0.957** |
| 0.5mm | 0.893/0.866 | **0.956/0.922** | 0.664/0.776 | **0.825/0.941** |
| 1.0mm | 0.532/0.330 | **0.680/0.410** | 0.430/0.504 | **0.786/0.905** |
| 2.0mm | 0.145/0.060 | **0.208/0.084** | 0.135/0.155 | **0.401/0.457** |

This is the tree-forest architecture: 3D is the "trunk" (structural search), 2D leaves are the "signature" (disambiguation).

### Why ND-in-KD-tree doesn't work

Adding 2D to the KD-tree Euclidean distance mixes units (mm vs pixels) and dilutes the 3D signal. The 2D noise floor (~0.5 pixel) overwhelms the discriminative information.

### Why two-phase works

When 3D is noisy, multiple candidates fall within the search radius. The 2D leaf positions provide independent evidence to pick the right one. The Hungarian algorithm makes globally optimal assignments across all competing candidates.

### Clean data (no artificial noise)

| method | alpha | rad | links | TP | FP | FN | prec | rec |
|--------|-------|-----|-------|----|----|----|----|-----|
| 3D | - | 2.0 | 14533 | 14036 | 497 | 270 | 0.966 | 0.981 |
| ND | 0.1 | 2.0 | 11908 | 11629 | 279 | 2677 | 0.977 | 0.813 |
| ND | 0.5 | 2.0 | 11001 | 10879 | 122 | 3427 | 0.989 | 0.760 |

**Conclusion**: On clean data, 3D is better (higher recall). ND is more conservative (higher precision, lower recall) because the 2D dimensions expand the search space.

### Noisy 3D (1mm Gaussian noise added)

| method | alpha | rad | links | TP | FP | FN | prec | rec |
|--------|-------|-----|-------|----|----|----|----|-----|
| 3D+noise (no 2D) | - | 2.0 | 8745 | 4448 | 4297 | 9858 | 0.509 | 0.311 |
| ND(noisy3D+clean2D) | 0.1 | 2.0 | 5274 | 4649 | 625 | 9657 | 0.881 | 0.325 |
| ND(noisy3D+clean2D) | 0.1 | 5.0 | 13473 | 9469 | 4004 | 4837 | 0.703 | 0.662 |

**Conclusion**: When 3D is noisy, ND tracking with clean 2D data significantly improves precision (0.881 vs 0.509). The 2D leaf information compensates for 3D reconstruction noise.

### Why ND doesn't beat 3D on clean data

1. **Scale mismatch**: 3D distances (mm) and 2D distances (pixels) are in different units. KD-tree Euclidean distance mixes them incorrectly.
2. **On clean data**: 3D is near-perfect (96.6% prec, 98.1% rec). The 2D adds noise without disambiguating.
3. **On noisy data**: ND is dramatically better. At 1mm noise: ND prec=0.894 vs 3D prec=0.518. At 2mm noise: ND prec=0.963 vs 3D prec=0.149.

| Noise | 3D-only (rad=2) | ND alpha=0.1 (rad=2) | ND alpha=0.5 (rad=5) |
|-------|-----------------|---------------------|---------------------|
| 0.0mm | 0.966/0.981 | 0.977/0.812 | 0.963/0.812 |
| 0.5mm | 0.893/0.867 | 0.966/0.759 | 0.965/0.812 |
| 1.0mm | 0.518/0.319 | 0.894/0.331 | 0.967/0.804 |
| 2.0mm | 0.149/0.060 | 0.780/0.064 | 0.963/0.492 |

## Files to create/modify

- `src/openptv2/storage/unified_table.py` — new UnifiedParticleTable class
- `src/openptv2/storage/zarr_store.py` — add read/write_particle_table methods
- `src/openptv2/storage/legacy.py` — add import/export for unified format
- `src/openptv2/algorithms/track_kernels_track3d.py` — add N-dimensional distance

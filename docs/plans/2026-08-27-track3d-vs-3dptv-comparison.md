# Comparison: original 3dptv `track.c` vs openptv2 `track3d_loop_fast`

**Date**: 2026-08-27
**Original source**: `C:\Users\alex\Downloads\3dptv\src_c\track.c` (1111 lines, function `trackcorr_c`)
**openptv2 source**: `src/openptv2/algorithms/track_kernels_track3d.py` (function `track3d_loop_fast`)

## Summary

The original 3dptv `trackcorr_c` and openptv2's `track3d_loop_fast` are **fundamentally different algorithms** that happen to solve the same problem. `trackcorr_c` is a 2D-image-space tracker with camera projections, multi-camera frequency weighting, and a complex two-hop lookahead. `track3d_loop_fast` is a pure 3D-position-space tracker with no camera knowledge. They share only the constant-velocity prediction formula (`X2 = 2*X1 - X0`) and a greedy conflict-resolution strategy.

openptv2 also has `trackcorr_loop_fast` (in `track_kernels_corr.py`), which is the 2D-based port of `trackcorr_c`. That function is a much closer match to the original.

## 1. Architecture

| Aspect | `trackcorr_c` (3dptv) | `track3d_loop_fast` (openptv2) | `trackcorr_loop_fast` (openptv2) |
|---|---|---|---|
| Search space | 2D image space per camera | Pure 3D position space | 2D image space per camera |
| Camera knowledge | Yes (img_coord, metric_to_pixel) | None | Yes (point_to_pixel_fast) |
| Multi-camera fusion | `sortwhatfound` (cross-camera frequency) | N/A | `sorted_candidates_fast` |
| Cascade levels | 1 (but has fallback for no-prev) | 3 (has-prev / neighbor-vel / cold) | 3 (has-prev / neighbor-vel / cold) |
| Backward tracking | Yes (`trackback_c`) | No (only forward) | Yes (`trackback_loop_fast`) |

## 2. Prediction Method

Both use constant-velocity extrapolation:
```c
// track.c line 113-115
X2 = 2*X1 - X0;
Y2 = 2*Y1 - Y0;
Z2 = 2*Z1 - Z0;
```

```python
# track_kernels_track3d.py line 410-412
pred_x = 2.0 * path_x_1[i, 0] - path_x_0[prev_idx, 0]
pred_y = 2.0 * path_x_1[i, 1] - path_x_0[prev_idx, 1]
pred_z = 2.0 * path_x_1[i, 2] - path_x_0[prev_idx, 2]
```

**Identical.**

## 3. Two-Hop Lookahead (track.c lines 204-337)

The original does a **two-hop acceptance**: after finding candidate X3 in frame n+1, it predicts ahead to frame n+2 using an improved predictor:
```c
// track.c line 223-225
X5 = 0.5*(5.0*X3 - 4.0*X1 + X0);  // half-damped acceleration
Y5 = 0.5*(5.0*Y3 - 4.0*Y1 + Y0);
Z5 = 0.5*(5.0*Z3 - 4.0*Z1 + Z0);
```

Then searches for candidates at X5 in frame n+2, and only accepts the n+1 candidate if there's a valid n+2 continuation.

**openptv2 `track3d_loop_fast`**: No lookahead at all. Only checks that a candidate exists in frame n+1.

**openptv2 `track4be_loop_fast`**: Has a similar two-hop check (eq. 12: `2*q - x1`), but this is a different algorithm (4BE, not 3MA).

## 4. Cost Function

### track.c (lines 306-325)
```c
dl = (dist(prev→curr) + dist(curr→candidate)) / 2;
angle_acc(X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3, &angle, &acc);
acc = (acc0 + acc1) / 2;  // average of two frames' acceleration
angle = (angle0 + angle1) / 2;
quali = w[mm].freq + w[mm].freq;  // multi-camera frequency

if ((acc < dacc && angle < dangle) || (acc < dacc/10)) {
    rr = (dl/lmax + acc/dacc + angle/dangle) / quali;
}
```

Components:
- `dl`: average displacement (smoothness)
- `acc`: average acceleration (kinematic consistency)
- `angle`: average angle change (directional consistency)
- `quali`: multi-camera frequency (robustness)
- Gates: `(acc < dacc && angle < dangle) || (acc < dacc/10)`

### track3d_loop_fast Level 1 (lines 421-442)
```python
acc = sqrt((X2 - 2 * X1 + X0) ^ 2 + (Y2 - 2 * Y1 + Y0) ^ 2 + (Z2 - 2 * 1 + Z0) ^ 2)
dist_from_curr = sqrt((X2 - X1) ^ 2 + (Y2 - Y1) ^ 2 + (Z2 - Z1) ^ 2)
cost = acc + dist_weight * dist_from_curr
```

Components:
- `acc`: acceleration residual (same as track.c's `acc`)
- `dist_weight * dist_from_curr`: distance penalty (new, not in track.c)
- No angle term
- No multi-camera frequency
- No acceleration/angle gates

### track3d_loop_fast Level 2 (lines 504-508)
```python
acc = sqrt((cand - pred) ^ 2)  # distance from neighbor-averaged prediction
cost = acc
```

### track3d_loop_fast Level 3 (lines 587-593)
```python
dist = sqrt((cand - pred) ^ 2)  # distance from current (or flow-corrected) position
cost = dist
```

## 5. Candidate Search

### track.c (lines 166-185)
Per-camera 2D search:
```c
searchquader(X2, Y2, Z2, xr, xl, yd, yu);  // compute 2D search box
for (j=0; j<n_img; j++) {
    zaehler1 = candsearch_in_pix(t4[2][j], nt4[2][j], 
        x1[j], y1[j], xl[j], xr[j], yu[j], yd[j], philf[j]);
}
sortwhatfound(p16, &zaehler1);  // merge across cameras by frequency
```

### track3d_loop_fast (lines 414-418)
3D position-space search:
```python
n_cands = _find_closest_in_3d_grid(
    path_x_2,
    np2,
    pred_x,
    pred_y,
    pred_z,
    ax,
    ay,
    az,
    max_cands,
    cand_inds,
    cand_dists,
    ...,
)
```

The 3D grid search finds candidates by Euclidean distance in position space, with no camera projection involved.

## 6. Conflict Resolution

### track.c (lines 587-646)
1. Sort each particle's candidates by cost
2. Set `finaldecis` and `next` to best candidate
3. For each particle, check if its best candidate is already claimed:
   - If unclaimed: link
   - If claimed: compare `finaldecis` — keep the better one, discard the loser
4. Loser's `next` is set to -2 (unlinked)

This is a **global greedy with last-writer-wins** — the last particle to claim a candidate wins if its `finaldecis` is better.

### track3d_loop_fast (lines 447-457)
1. Collect all (particle, candidate) edges with costs
2. Sort edges by cost
3. Claim in ascending cost order: first claim wins

This is a **cost-ordered greedy** — the globally cheapest edge wins, regardless of processing order.

**Key difference**: track.c processes particles sequentially and uses `finaldecis` comparison for conflicts; track3d_loop_fast collects all edges first and claims in global cost order.

## 7. Multi-Camera Frequency (track.c only)

track.c uses `quali = w[mm].freq + w[mm].freq` in the cost denominator (lines 320-325). `freq` is the number of cameras that detected this candidate. A candidate seen by 3 cameras gets a lower cost (more reliable) than one seen by 1 camera.

**track3d_loop_fast**: No frequency weighting. All candidates treated equally because it has no camera knowledge.

## 8. Adding New Particles (track.c only)

When `tpar.add` is set (lines 340-436, 480-580), track.c can create new particle positions from unused candidates:
1. Search for unused targets near the predicted position
2. Triangulate new 3D position via `det_lsq_3d`
3. Volume check + displacement check
4. Add to the particle list with `prio=2`

**track3d_loop_fast**: No particle addition.

## 9. Backward Tracking

### track.c `trackback_c` (lines 740-1111)
Mirror of forward tracking but in reverse time order. Uses `mega[1][h].next` (forward link) instead of `prev`. Same cost function and conflict resolution.

### track3d_loop_fast
No backward tracking. Only forward.

openptv2 has `trackback_loop_fast` in `track_kernels_corr.py` for the 2D-based tracker.

## 10. Level Cascade

### track.c
Single level with fallback: if a particle has `prev >= 0`, use constant-velocity prediction. If `prev < 0`, use the particle's current position as prediction (cold start). No Level 2 (neighbor velocity) concept.

### track3d_loop_fast
Three-level cascade:
- **Level 1**: Particles with previous links — constant-velocity prediction
- **Level 2**: No prev link, but neighbors have velocity — average neighbor velocity
- **Level 3**: No prev, no neighbors — static prediction (or flow-corrected with `cold_start_gate`)

Level 2 is a major addition in openptv2 that doesn't exist in track.c.

## 11. Key Gaps to Address

### Critical (affect tracking quality)

1. **No two-hop lookahead** in `track3d_loop_fast` — track.c rejects n+1 candidates that have no valid n+2 continuation. `track4be_loop_fast` has this but uses a different scoring method.

2. **No multi-camera frequency weighting** — track.c's `quali` term makes 3-camera detections more reliable than 1-camera detections. `track3d_loop_fast` has no camera knowledge.

3. **No acceleration/angle gates** — track.c has `(acc < dacc && angle < dangle) || (acc < dacc/10)` as a hard gate before accepting candidates. `track3d_loop_fast` uses only cost ordering.

4. **Different cost function** — track.c uses `dl/lmax + acc/dacc + angle/dangle` (normalized, multi-component). track3d_loop_fast uses `acceleration_residual + dist_weight * distance` (unnormalized, two-component).

### Structural (different design choices)

5. **No particle addition** — track.c can create new particles from unused candidates. track3d_loop_fast cannot.

6. **No backward tracking** — track.c has `trackback_c`. track3d_loop_fast only goes forward.

7. **Level 2 (neighbor velocity)** — exists in track3d_loop_fast but not in track.c. This is an improvement in openptv2.

8. **Cost-ordered greedy vs sequential greedy** — track3d_loop_fast's edge-based claiming is arguably better than track.c's sequential last-writer-wins.

## 12. What `trackcorr_loop_fast` Already Has

`trackcorr_loop_fast` (in `track_kernels_corr.py`) is the 2D-based tracker that's a much closer port of `trackcorr_c`. It already has:
- Camera projections (img_coord, metric_to_pixel)
- Multi-camera frequency via `sorted_candidates_fast`
- Two-hop lookahead (checks frame n+2)
- Acceleration/angle gates (`dacc`, `dangle`)
- Cost function with `dl/lmax + acc/dacc + angle/dangle`
- Particle addition (`add_flag`)
- Backward tracking (`trackback_loop_fast`)

This function is the faithful port. `track3d_loop_fast` is a deliberately simplified 3D-only alternative.

## 13. Recommendation

The comparison reveals two separate questions:

**A. Is `trackcorr_loop_fast` a faithful port of `trackcorr_c`?**
→ Likely yes, based on the feature list. Need to verify cost function formula, conflict resolution details, and two-hop lookahead implementation match.

**B. Can `track3d_loop_fast` be improved by borrowing ideas from `trackcorr_c`?**
→ Yes, selectively:
- Add a two-hop check (already exists in `track4be_loop_fast`)
- Add acceleration gates to Level 1
- Consider adding multi-camera frequency as a weighting term (requires camera knowledge to be passed in)

But `track3d_loop_fast`'s pure-3D approach has advantages (simpler, faster, no camera calibration dependency) and its Level 2/3 cascade is an improvement over track.c's single-level approach.

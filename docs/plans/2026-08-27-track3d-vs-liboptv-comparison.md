# Comparison: liboptv `track3d.c` vs openptv2 `track3d_loop_fast`

**Date**: 2026-08-27
**Reference**: https://github.com/alexlib/openptv/blob/f7dfafe4840d6097893e043191015fd1a3be9bb8/liboptv/src/track3d.c
**openptv2**: `src/openptv2/algorithms/track_kernels_track3d.py` (function `track3d_loop_fast`)

## Summary

Both are pure 3D-position-space trackers with a 3-level cascade (Level 1: has-prev, Level 2: neighbor velocity, Level 3: cold start). They share the same architecture but differ in the cost function formula, candidate search implementation, and conflict resolution strategy.

**CRITICAL BUG in liboptv**: The acceleration residual formula is `curr - 2*cand + predicted` when it should be `cand - 2*curr + predicted`. This produces incorrect costs for non-constant-velocity motion. openptv2 has the correct formula.

## 1. Architecture (IDENTICAL)

| Aspect | liboptv `track3d.c` | openptv2 `track3d_loop_fast` |
|---|---|---|
| Search space | Pure 3D position space | Pure 3D position space |
| Camera knowledge | None | None |
| Cascade levels | 3 (has-prev / neighbor-vel / cold) | 3 (has-prev / neighbor-vel / cold) |
| Backward tracking | No | No |
| Particle addition | No | No |

## 2. Prediction Method (IDENTICAL)

Both use constant-velocity extrapolation:
```c
// liboptv track3d.c
predicted[d] = 2 * curr_path_inf->x[d] - prev_path_inf->x[d];
```

```python
# openptv2 track_kernels_track3d.py line 410-412
pred_x = 2.0 * path_x_1[i, 0] - path_x_0[prev_idx, 0]
pred_y = 2.0 * path_x_1[i, 1] - path_x_0[prev_idx, 1]
pred_z = 2.0 * path_x_1[i, 2] - path_x_0[prev_idx, 2]
```

## 3. Cost Function — Level 1

### liboptv `track3d.c`
```c
float diff = curr_path_inf->x[d] - 2 * next->path_info[cand_indices[k]].x[d] + prev_path_inf->x[d];
acc += diff * diff;
decis[k] = sqrtf(acc);
```

Formula: `cost = sqrt(sum((curr - 2*cand + prev)^2))`

### openptv2 `track3d_loop_fast`
```python
d0 = path_x_2[k, 0] - 2.0 * path_x_1[i, 0] + path_x_0[prev_idx, 0]
d1 = path_x_2[k, 1] - 2.0 * path_x_1[i, 1] + path_x_0[prev_idx, 1]
d2 = path_x_2[k, 2] - 2.0 * path_x_1[i, 2] + path_x_0[prev_idx, 2]
acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
```

Formula: `cost = sqrt(sum((cand - 2*curr + prev)^2))`

### BUG: liboptv has `curr - 2*cand + prev`, openptv2 has `cand - 2*curr + prev`

These are different! For constant velocity (prev=0, curr=1, cand=2):
- liboptv: `1 - 2*2 + 0 = -3` → cost = 3 (WRONG, should be 0)
- openptv2: `2 - 2*1 + 0 = 0` → cost = 0 (CORRECT)

The correct acceleration residual is `cand - 2*curr + prev` (= `x[n+1] - 2*x[n] + x[n-1]`). liboptv has the terms swapped.

### openptv2 also adds `dist_weight * dist_from_curr`

```python
dc0 = path_x_2[k, 0] - path_x_1[i, 0]
dc1 = path_x_2[k, 1] - path_x_1[i, 1]
dc2 = path_x_2[k, 2] - path_x_1[i, 2]
dist_from_curr = c_sqrt(dc0 * dc0 + dc1 * dc1 + dc2 * dc2)
edge_cost[n_edges] = acc + dist_weight * dist_from_curr
```

This term does NOT exist in liboptv. It was added to break near-ties in acceleration residual toward the physically smaller jump.

## 4. Cost Function — Level 2

### liboptv `track3d.c`
```c
for (d = 0; d < 3; d++)
    predicted[d] = curr_path_inf->x[d] + vel[d];
// ...
float diff = curr_path_inf->x[d] - 2 * next->path_info[cand_indices[k]].x[d] + predicted[d];
```

Formula: `cost = sqrt(sum((curr - 2*cand + predicted)^2))` where `predicted = curr + avg_vel`

### openptv2 `track3d_loop_fast`
```python
pred_x = cx + vel_x * inv_nvel
# ...
d0 = path_x_2[k, 0] - pred_x
d1 = path_x_2[k, 1] - pred_y
d2 = path_x_2[k, 2] - pred_z
acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
edge_cost[n_edges] = acc
```

Formula: `cost = sqrt(sum((cand - predicted)^2))` where `predicted = curr + avg_vel`

### BUG: liboptv has `curr - 2*cand + predicted`, openptv2 has `cand - predicted`

For a candidate exactly on the prediction (cand = predicted):
- liboptv: `curr - 2*predicted + predicted = curr - predicted` → nonzero (WRONG)
- openptv2: `predicted - predicted = 0` → zero (CORRECT)

## 5. Cost Function — Level 3

### liboptv `track3d.c`
```c
predicted[d] = curr_path_inf->x[d];  // static prediction
// ...
float diff = curr_path_inf->x[d] - 2 * next->path_info[cand_indices[k]].x[d] + predicted[d];
```

Formula: `cost = sqrt(sum((curr - 2*cand + curr)^2)) = sqrt(sum((2*curr - 2*cand)^2)) = 2 * sqrt(sum((curr - cand)^2))`

### openptv2 `track3d_loop_fast`
```python
pred_x = path_x_1[i, 0]  # static prediction
# ...
d0 = path_x_2[k, 0] - pred_x
d1 = path_x_2[k, 1] - pred_y
d2 = path_x_2[k, 2] - pred_z
dist = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
```

Formula: `cost = sqrt(sum((cand - curr)^2))`

### Bug is harmless here

liboptv's cost is exactly 2x openptv2's cost. Since both just sort by cost, the ordering is identical. The bug doesn't affect Level 3 results.

## 6. Candidate Search

### liboptv `find_candidates_in_3d`
```c
int find_candidates_in_3d(frame *frm, vec3d pos, double dx, double dy, double dz, int *indices, int max_cands) {
    int i, count = 0;
    for (i = 0; i < frm->num_parts; i++) {
        if (fabs(frm->path_info[i].x[0] - pos[0]) < dx &&
            fabs(frm->path_info[i].x[1] - pos[1]) < dy &&
            fabs(frm->path_info[i].x[2] - pos[2]) < dz) {
            if (count < max_cands) {
                indices[count++] = i;
            }
        }
    }
    return count;
}
```

Linear scan, returns candidates in insertion order (not sorted by distance). Up to `max_cands` (=32) candidates.

### openptv2 `_find_closest_in_3d_grid`
Uses a spatial hash grid for O(1) cell lookup. Returns candidates sorted by Euclidean distance (closest first). Up to `max_cands` candidates.

**openptv2 is faster** for large particle counts due to the grid acceleration structure.

## 7. Conflict Resolution

### liboptv `track3d.c`
Per-particle, sequential:
1. For each particle, sort its candidates by cost
2. Link to best candidate if unclaimed
3. If already claimed: first-come-first-served (no comparison)

```c
if (num_cands > 0 && next->path_info[linkdecis[0]].prev < 0) {
    curr_path_inf->next = linkdecis[0];
    next->path_info[linkdecis[0]].prev = i;
    count1++;
}
```

### openptv2 `track3d_loop_fast`
Global edge-based:
1. Collect all (particle, candidate) edges with costs
2. Sort edges by cost globally
3. Claim in ascending cost order: first claim wins

```python
if n_edges > 0:
    _order = np.argsort(_edge_cost[:n_edges]).astype(np.int32)
    for oi in range(n_edges):
        e = order[oi]
        i = edge_i[e]
        k = edge_k[e]
        if path_next_1[i] < 0 and path_prev_2[k] < 0:
            path_next_1[i] = k
            path_prev_2[k] = i
            count1 += 1
```

**Key difference**: liboptv processes particles in index order (particle 0 always wins contested candidates over particle 500). openptv2 claims globally by cost (cheapest edge wins regardless of particle index).

## 8. Search Box Parameter

### liboptv
```c
double dx = tpar->dvxmax;
double dy = tpar->dvymax;
double dz = tpar->dvzmax;
```

Always uses `dvxmax/dvymax/dvzmax` as the search box. No `dacc` parameter.

### openptv2
```python
ax = dacc if dacc > 0.0 else dx
ay = dacc if dacc > 0.0 else dy
az = dacc if dacc > 0.0 else dz
```

Uses `dacc` as the search box when > 0, otherwise falls back to `dvxmax/dvymax/dvzmax`. This allows tighter Level 1/2 search without starving Level 3.

## 9. Additional Features in openptv2

### `cold_start_gate` (Level 3 only)
Optional gate that rejects Level 3 candidates farther than a threshold from the local flow prediction. Not present in liboptv.

### `dist_weight` (Level 1 only)
Tiebreak weight for distance from current position. Not present in liboptv.

## 10. Summary of Differences

| # | Aspect | liboptv `track3d.c` | openptv2 `track3d_loop_fast` | Impact |
|---|---|---|---|---|
| 1 | **L1 cost formula** | `curr - 2*cand + prev` (BUG) | `cand - 2*curr + prev` (CORRECT) | **Critical**: wrong cost ordering |
| 2 | **L2 cost formula** | `curr - 2*cand + predicted` (BUG) | `cand - predicted` (CORRECT) | **Critical**: wrong cost ordering |
| 3 | **L3 cost formula** | `2*(curr - cand)` (harmless bug) | `cand - curr` | None (same ordering) |
| 4 | **L1 dist_weight** | Not present | `acc + dist_weight * dist` | Moderate: breaks near-ties |
| 5 | **Candidate search** | Linear scan, insertion order | Grid加速, distance-sorted | Performance only |
| 6 | **Conflict resolution** | Per-particle sequential | Global cost-ordered greedy | Moderate: different claims |
| 7 | **Search box** | Always `dvxmax/dvymax/dvzmax` | `dacc` when > 0 | Tunability |

## 11. To Make openptv2 100% Compatible with liboptv

If the goal is to reproduce liboptv's behavior exactly (bugs and all):

1. **Remove `dist_weight` term from Level 1**: Change `edge_cost[n_edges] = acc + dist_weight * dist_from_curr` to `edge_cost[n_edges] = acc`

2. **Change Level 1 cost formula**: Change `d0 = path_x_2[k, 0] - 2.0 * path_x_1[i, 0] + path_x_0[prev_idx, 0]` to `d0 = path_x_1[i, 0] - 2.0 * path_x_2[k, 0] + path_x_0[prev_idx, 0]` (swap cand and curr terms)

3. **Change Level 2 cost formula**: Change from `cand - predicted` to `curr - 2*cand + predicted`

4. **Change Level 3 cost formula**: Change from `cand - curr` to `2*(curr - cand)` (harmless, same ordering)

5. **Change conflict resolution**: Replace global edge-based greedy with per-particle sequential claiming

6. **Remove `dacc` as search box**: Always use `dvxmax/dvymax/dvzmax`

7. **Remove `cold_start_gate`**: Not present in liboptv

**However**: Steps 1-3 would make openptv2 WORSE, not better, because liboptv's formulas are buggy. The correct approach is to FIX liboptv, not to introduce bugs into openptv2.

## 12. Recommended Action

1. **Keep openptv2's cost formulas** — they are correct, liboptv's are buggy
2. **Keep `dist_weight`** — it's an improvement
3. **Keep global cost-ordered greedy** — it's fairer than sequential
4. **Optionally** make the search box configurable (it already is via `dacc`)
5. **Report the liboptv bug** to the upstream project

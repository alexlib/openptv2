# Tracking Algorithms

OpenPTV implements two tracking strategies for particle trajectory reconstruction:

## Overview

| Feature | `track.c` | `track3d.c` |
|---------|-----------|-------------|
| **Dimensionality** | Multi-camera 2D→3D | Direct 3D |
| **Candidate search** | Projects 3D box to 2D per camera | Direct 3D box search |
| **Candidate sorting** | Frequency across cameras | Acceleration metric |
| **Linking metric** | Angle + acceleration (gon) | Acceleration (second derivative) |
| **Particle addition** | Yes | No |
| **Backward tracking** | Yes | No |
| **Lines of code** | ~1275 | ~203 |

---

## `track.c` — Multi-Camera Tracking

Full pipeline for calibrated multi-camera setups. Handles the complete workflow from 2D image targets to 3D particle trajectories.

### Key Functions

- **`trackcorr_c_loop()`** — Main forward tracking loop
- **`trackback_c()`** — Backward tracking to fill gaps
- **`searchquader()`** — Projects 3D search volume to 2D image regions per camera
- **`candsearch_in_pix()`** — Finds up to 4 nearest candidates in pixel space
- **`sort_candidates_by_freq()`** — Ranks candidates by camera visibility count
- **`angle_acc()`** — Computes angle (gon) and acceleration between velocity vectors
- **`add_particle()`** — Inserts new particles from unmatched targets

### Algorithm Flow

1. For each particle in current frame:
   - Predict next position using `2*curr - prev` (linear extrapolation)
   - Project 3D search cuboid to 2D per camera via `searchquader()`
   - Find candidates in pixel space via `candsearch_in_pix()`
   - Sort candidates by frequency across cameras
2. Evaluate candidates using angle + acceleration metric
3. Link to best candidate if within thresholds (`dacc`, `dangle`)
4. Optionally add new particles from unmatched targets

### Use Case

Real experiments with 2–4 calibrated cameras. Robust to occlusions because it leverages multi-camera redundancy.

---

## `track3d.c` — Direct 3D Tracking

Simplified tracking for pre-reconstructed 3D data. No camera projection needed.

### Key Functions

- **`track3d_loop()`** — Main tracking loop
- **`find_candidates_in_3d()`** — Finds particles within 3D search box

### Algorithm Flow

Three-level linking strategy:

1. **Level 1** — Particles with previous links:
   - Predict: `2*curr - prev`
   - Search in 3D box defined by `dvxmax`, `dvymax`, `dvzmax`

2. **Level 2** — No previous link, but neighbors have links:
   - Compute average velocity from linked neighbors
   - Predict: `curr + avg_velocity`

3. **Level 3** — No previous link, no neighbor links:
   - Use current position as prediction

For each level:
- Find candidates within 3D box
- Sort by acceleration (second derivative)
- Link to best unlinked candidate

### Use Case

Pre-computed 3D data (e.g., `rt_is.*` files). Faster and simpler than full multi-camera tracking.

---

## Candidate Selection Comparison

### `track.c` — Multi-Camera Frequency

```c
// Search in each camera's image space
for (cam = 0; cam < num_cams; cam++) {
    register_closest_neighbs(targets[cam], ...);
}
// Sort by how many cameras see each candidate
num_cands = sort_candidates_by_freq(points, num_cams);
```

### `track3d.c` — Direct 3D Box

```c
// Simple box search in 3D
for (i = 0; i < frm->num_parts; i++) {
    if (fabs(x - pos[0]) < dx &&
        fabs(y - pos[1]) < dy &&
        fabs(z - pos[2]) < dz) {
        indices[count++] = i;
    }
}
```

---

## Linking Quality Metrics

### `track.c` — Angle + Acceleration

Uses angle between velocity vectors in [gon](https://en.wikipedia.org/wiki/Gradian) plus acceleration magnitude:

```c
void angle_acc(vec3d start, vec3d pred, vec3d cand,
               double *angle, double *acc)
{
    vec3d v0, v1;
    vec_subt(pred, start, v0);   // predicted velocity
    vec_subt(cand, start, v1);   // actual velocity

    *acc = vec_diff_norm(v0, v1);
    *angle = (200./M_PI) * acos(vec_dot(v0, v1) / (vec_norm(v0) * vec_norm(v1)));
}
```

Decision: link if `(acc < dacc AND angle < dangle) OR (acc < dacc/10)`.

### `track3d.c` — Acceleration Only

Uses second derivative (acceleration) as the sole metric:

```c
// Acceleration = |curr - 2*next + prev|
float acc = 0.0;
for (d = 0; d < 3; d++) {
    float diff = curr[d] - 2*next[d] + prev[d];
    acc += diff * diff;
}
decis[k] = sqrtf(acc);
```

---

## Output Format

Both algorithms print per-step statistics:

```
track.c:    step: 10000, curr: 998, next: 1043, links: 453, lost: 545, add: 1
track3d.c:  track3d step: 10001, curr: 1, next: 1, links: 0
```

---

## When to Use Which

| Scenario | Use |
|----------|-----|
| Raw 2D images + calibration | `track.c` |
| Pre-reconstructed 3D positions | `track3d.c` |
| Need backward gap filling | `track.c` |
| Fast iteration on 3D data | `track3d.c` |
| Multi-camera redundancy required | `track.c` |

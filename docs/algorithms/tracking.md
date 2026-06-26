## Tutorial: Choosing Tracking Parameters from Data Statistics

Tracking parameters (velocity, acceleration, angle limits) can be chosen systematically using simple statistics from your dataset. This approach minimizes trial-and-error and ensures robust, unambiguous tracking.

### Step-by-step workflow

1. **Run a quick probe script** to compute:
     - Maximum observed displacement per frame (velocity)
     - Maximum observed acceleration
     - Typical interparticle distance (spacing between particles in a frame)

2. **Set parameters:**
     - **Velocity window (`velocity_lims`):**
         - Set just above the maximum observed displacement, but below the typical interparticle distance to avoid ambiguity.
     - **Acceleration limit (`accel_lim`):**
         - Set just above the maximum observed acceleration.
     - **Angle limit (`angle_lim`):**
         - Set to a typical value for smooth motion (e.g., 20 gon ≈ 18°), or just above the maximum observed angle change if available.

#### Example (from Burgers dataset):

| Statistic                | Value (example) |
|--------------------------|-----------------|
| Max displacement         | 0.08 mm/frame   |
| Max acceleration         | 0.09 mm/frame²  |
| Interparticle distance   | 1.53 mm         |

**Parameter selection:**
- `velocity_lims = [[-0.088, 0.088], [-0.088, 0.088], [-0.088, 0.088]]` (10% above max displacement, but < interparticle distance)
- `accel_lim = 0.099` (10% above max acceleration)
- `angle_lim = 20` (gon)

### Reference test

See the test: `test_tracking_parameters_from_data_statistics` in [algorithms/tests/parity/test_burgers_tracking_parameter_sensitivity.py](../../algorithms/tests/parity/test_burgers_tracking_parameter_sensitivity.py)

This test demonstrates the full workflow and can be used as a template for your own datasets.

---
# Tracking Algorithms

OpenPTV implements two tracking strategies for particle trajectory reconstruction:

## Case Study: Burgers Dataset Gap Relinking

For a detailed analysis of how these algorithms differ when particles re-appear after a gap, including:
- **7 trajectories vs 6 trajectories** from the same 5-frame dataset
- **Root cause analysis** of the algorithmic divergence
- **Backward tracking recovery capabilities** and limitations
- **Numerical insights** for designing better tracking methods

See: [Burgers Gap Relinking Case Study](burgers_gap_relinking_case_study.md)

---

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

## How to Choose/Enable 3D Segment Tracking (`track3d`)

You can select and run the direct 3D segment tracking algorithm (`track3d`) over the standard multi-camera epipolar tracking algorithm (`trackcorr`) in three different ways:

### 1. Through the Desktop GUI
1. Open the **Parameter Editor** in the GUI.
2. Navigate to the **Track Parameters** (Tracking) tab.
3. Locate the **Tracking mode (0=Standard, 1=3D Seg):** parameter.
4. Set the value to:
   - `0` for Standard Epipolar tracking (`trackcorr`).
   - `1` for 3D Segment tracking (`track3d`).
5. Save the parameters. The GUI and its interactive step-by-step previewers/visualizers (`tracking_preview` and `tracking_viz_panel`) will conditionally load and execute the selected algorithm automatically.

### 2. Through Parameter Files (YAML & Legacy `.par`)
* **YAML configuration**: In your active YAML parameters, set the `track_mode` key in the `track` section:
  ```yaml
  track:
    track_mode: 1
    ...
  ```
* **Legacy `.par` configuration**: In `parameters/track.par`, the 10th line (if present) represents the tracking mode. Set it to `1` to enable 3D segment tracking:
  ```
  0.100000
  0.200000
  0.100000
  0.200000
  0.100000
  0.200000
  20.000000
  0.100000
  0
  1
  ```
  *(Note: If the 10th line is absent, the system gracefully defaults to `0` / Standard mode).*

### 3. Via the Command-Line Batch Utility (`pyptv_batch.py`)
When executing batch processing from the CLI, you can force the use of 3D tracking using the `--track3d` option:
```bash
uv run python -m openptv2.gui.pyptv.pyptv_batch --workdir=./test_data/test_cavity --track3d
```
If `--track3d` is not specified, the utility will fall back to reading the `track_mode` setting from the active parameter file.

---


## Python Translation Status

Both `track.c` and `track3d.c` have been fully translated to Python in `algorithms/track.py` and `algorithms/track3d.py`. The Python implementations include Numba JIT-compiled fast paths.

### Parity with C/Cython

| Dataset | track3d | trackcorr |
|---------|---------|-----------|
| **Burgers** (5 frames, 5 particles) | Exact match | Exact match |
| **Cavity** (4 frames, ~700 particles) | Exact match | Python produces more links (see below) |
| **Synthetic** (8 frames, 15 particles) | 99% recovery, 0 wrong | 100% recovery, 0 wrong |

### Python Improvements Over C

The Python `trackcorr_c_loop` includes two improvements not present in the C code:

#### 1. Phase 3: Losers Retry

When two particles compete for the same target in conflict resolution, C drops the loser permanently. Python lets the loser try its fallback candidates (2nd, 3rd best matches) if they're still unclaimed. On the cavity dataset, this recovers ~27 additional correct links.

```python
# Phase 3: Losers retry with fallback candidates (claim unclaimed only)
for h in range(fb.buf[1].num_parts):
    curr_path_inf = fb.buf[1].path_info[h]
    if curr_path_inf.inlist > 1 and curr_path_inf.next == NEXT_NONE:
        for ti in range(1, curr_path_inf.inlist):
            cand = curr_path_inf.linkdecis[ti]
            if fb.buf[2].path_info[cand].prev == PREV_NONE:
                curr_path_inf.next = cand
                fb.buf[2].path_info[cand].prev = h
                break
```

#### 2. Stale Buffer Fix

When `step >= last - 2`, no new frame can be loaded into the last buffer slot after rotation. C leaves stale data from a previous frame in that slot, which `assess_new_position` may search and produce spurious links. Python clears the slot:

```python
fb.fb_next()
fb.write_frame_from_start(step)
if step < run_info.seq_par.last - 2:
    fb.read_frame_at_end(step + 3, read_links=False)
else:
    fb.buf[fb.buf_len - 1].num_parts = 0  # clear stale data
```

#### C count1 Overcounting Bug

The C code increments `count1` inside the conflict resolution loop. When particle B loses a conflict to particle A (B processed first, B.next set to NEXT_NONE), B has already been counted. The final `count1` is inflated. Python counts in a separate loop after all conflicts are resolved, producing the correct count. This explains why C's `printf` reports more links than actually appear in the output files.

### Synthetic Test Suite

A synthetic test case (`algorithms/tests/test_synthetic_tracking.py`) validates both algorithms against known ground truth:

- **15 particles** with diverse trajectories:
  - 5 constant-velocity straight lines
  - 3 constant-acceleration curved paths
  - 2 near-miss paths (close approach)
  - 2 parallel neighbors (3-unit separation)
  - 2 actual crossing paths
  - 1 late entry (appears at frame 3)
- **8 frames** (10001–10008)
- **5 test cases**: link correctness, recovery rate, trajectory distance validation, and trackcorr >= track3d comparison

```bash
uv run pytest algorithms/tests/test_synthetic_tracking.py -v
```

# Creating a Small Synthetic Cavity Test Case

## Goal

Produce a self-contained, ground-truth-annotated subset of the `test_cavity` experiment:
4 cameras, 256×256 pixel images, 5 frames, real particle positions and real images — everything
traceable to a known answer at every pipeline step.

---

## Resolved Decisions

| # | Question | Decision |
|---|---|---|
| A | **Source data** | `res/rt_is.10000`–`10004` and `res/ptv_is.10000`–`10004` for all 5 frames — consistent source throughout. `res_orig/` is excluded because it lacks frame 10000. |
| B | **Coordinate system** | **Crop-relative** (Option A). `trafo.py:53` confirms raw pixel coords are `0..imx` referenced to sensor centre via `(x_px - imx/2) * pix_x`. Cropped `_targets` must use `x_px_crop = x_px_full − ox`, and `.ori` principal point must be updated per camera (see Phase 3 details). |
| C | **Partial trajectories** | Include all particles — those entering or leaving the volume mid-sequence are kept with `NaN` for missing frames. Entry/exit events are explicitly flagged and will be used to test boundary handling. |

---

## Output Layout

```
test_data/test_cavity_small/
  cal/
    cam1.tif.ori          ← copied unchanged (or adjusted, see Q-B)
    cam1.tif.addpar       ← copied unchanged
    ... (all 4 cameras)
    target_on_a_side.txt  ← copied unchanged
  img/
    cam1.10000            ← 256×256 crop of real image
    cam1.10000_targets    ← filtered + offset _targets
    ... (4 cams × 5 frames)
  img_orig/               ← same crops, unmodified backup
  parameters/
    ptv.par               ← identical except imaX=256, imaY=256
    sequence.par          ← same frame range 10000–10004
    ... (all other .par files copied unchanged)
  res/
    ptv_is.10000          ← ground truth 3D positions with forward/backward links
    rt_is.10001           ← ground truth tracking result
    added.10001           ← particles entering mid-sequence
    ... (5 frames)
  ground_truth/
    particles.csv         ← full ground truth table (see schema below)
    trajectories.csv      ← per-trajectory summary
    projections.csv       ← ground truth 2D projections per camera
```

---

## Phase 1 — Find the 256×256 Window and Extract the Particle Subset

### 1.1 Load camera models (all 4 cameras)

Use `Calibration.from_file()` from `algorithms/calibration.py`:

```python
from openptv2.algorithms.calibration import Calibration
cals = [Calibration.from_file(f"cal/cam{i}.tif.ori", f"cal/cam{i}.tif.addpar")
        for i in range(1, 5)]
```

Also load `ControlParams` from `parameters/ptv.par` — needed by projection functions for
pixel pitch (0.012 mm/px), sensor size (1280×1024), refractive indices.

### 1.2 Load all 3D particle positions — all 5 frames from `res/`

Use `res/rt_is.{frame}` for all frames 10000–10004 (3D positions with labels) and
`res/ptv_is.{frame}` for the forward/backward trajectory links.

Both files exist for all 5 frames in `res/`. All frames are treated identically — no special
casing of frame 10000.

**`rt_is` format** (one row per particle):
```
<count>
<label>  <X>  <Y>  <Z>  <n_targets_cam1>  <n_targets_cam2>  <n_targets_cam3>  <n_targets_cam4>
```

**`ptv_is` format** (one row per particle):
```
<count>
<prev_frame_id>  <next_frame_id>  <X>  <Y>  <Z>
```

### 1.3 Project all particles into all 4 cameras

Use `img_coord_batch` from `algorithms/imgcoord.py`:

```python
from openptv2.algorithms.imgcoord import img_coord_batch
# returns (x_mm, y_mm) from image centre in metric units
xy_mm = img_coord_batch(positions_xyz, cal, mm_lut)
```

Convert mm → pixels using `pixel_to_metric` inverse (divide by pixel pitch, add sensor half-size):
```
x_px = x_mm / pixel_pitch + imx / 2
y_px = y_mm / pixel_pitch + imy / 2
```

Alternatively use `point_to_pixel` from `algorithms/track.py`:
```python
from openptv2.algorithms.track import point_to_pixel
x_px, y_px = point_to_pixel(point_xyz, cal, cpar)
```

### 1.4 Choose the 256×256 window in each camera

**Candidate window centre**: project the volume centroid `(0, 2.5, 2.5)` mm through each camera.
This gives `(cx_cam, cy_cam)` in full-image pixel coordinates per camera.

Window in full-image pixels: `[cy−128 : cy+128, cx−128 : cx+128]`

**Filter**: a particle is *in the subset* if its projection falls inside the 256×256 window
in **all 4 cameras** in **at least one frame** (adjust to all frames if answering Q-C as
"all frames only").

Record per-camera crop offsets:
```
crop_offset[cam] = (ox, oy)   # top-left corner in full-image pixels
```

### 1.5 Outputs of Phase 1

- `subset_particle_ids`: set of particle labels present across frames
- `crop_offset[cam]`: (ox, oy) per camera — used in Phases 3 and 4

---

## Phase 2 — Reconstruct Ground Truth Trajectories

### 2.1 Chain trajectories across frames

From `res/rt_is.{frame}` the `label` field is consistent across all 5 frames for a tracked
particle. From `res/ptv_is.{frame}` the `prev_frame_id` / `next_frame_id` fields provide the
same linkage redundantly — use both to cross-check and catch any inconsistency.

For each particle in the subset:
- Collect `(X, Y, Z)` from `rt_is.10000`–`10004` (all frames, same source, same format)
- A particle absent in a frame gets `NaN` (entry/exit)
- Displacement per step: `(dx, dy, dz) = pos(t+1) − pos(t)` where both frames are non-NaN

### 2.2 Smooth trajectories to produce ground truth

The raw per-frame 3D positions carry reconstruction noise. For each particle with ≥ 3 valid
frames, fit a degree-2 polynomial (or degree-1 if only 2 frames) through its positions
independently per axis. Evaluate the polynomial at each frame — these smoothed positions
are the **ground truth coordinates**.

This preserves the real flow statistics and motion patterns while removing single-frame
reconstruction outliers. It is the "smooth version" of the existing flow.

### 2.2 Classify each trajectory

| Status | Condition |
|---|---|
| `full` | present in all 5 frames — main test population |
| `entry` | first appears at frame > 10000 — enters from volume boundary |
| `exit` | last present at frame < 10004 — leaves volume boundary |
| `transient` | both entry and exit mid-sequence |

All four classes are kept. Entry/exit events are test cases for the tracker's handling of
trajectory starts and ends.

### 2.3 Ground truth CSV schema

**`ground_truth/particles.csv`**
```
particle_id, frame, X, Y, Z, dx, dy, dz, status
```

**`ground_truth/projections.csv`**
```
particle_id, frame, cam, x_px_full, y_px_full, x_px_crop, y_px_crop
```
where `x_px_crop = x_px_full − ox_cam`, `y_px_crop = y_px_full − oy_cam`

**`ground_truth/trajectories.csv`**
```
particle_id, first_frame, last_frame, n_frames, status
```

---

## Phase 3 — Crop Real Images and Filter Targets

### 3.1 Crop images

For each camera `c` and frame `f`:
```python
import imageio
img = imageio.imread(f"img/cam{c}.{f}")          # full 1280×1024
ox, oy = crop_offset[c]
crop = img[oy:oy+256, ox:ox+256]
imageio.imwrite(f"test_cavity_small/img/cam{c}.{f}", crop)
```

Copy unchanged to `img_orig/` as backup.

### 3.2 Filter and offset `_targets` files

**`_targets` format**:
```
<count>
<idx>  <x_px>  <y_px>  <sumg>  <nx>  <ny>  <npix>  <flag>
```

For each `cam{c}.{f}_targets`:
1. Load all rows
2. Keep rows where `ox ≤ x_px < ox+256` and `oy ≤ y_px < oy+256`
3. Write crop-relative coordinates: `x_out = x_px − ox`, `y_out = y_px − oy`
4. Renumber `idx` from 0

### 3.3 Update `.ori` principal point per camera

`trafo.py` converts pixels via `x_metric = (x_px - imx/2) * pix_x`, so the image centre
is always pixel `(imx/2, imy/2)`. After cropping to 256×256, the new centre in full-image
pixels is `(ox + 128, oy + 128)`. The optical axis still hits the same physical point on the
sensor, so its position in the cropped coordinate system shifts.

For each camera, update the principal point in `.ori` (fields `xh`, `yh`, in mm):

```
xh_new = xh_old + (imx/2 - (ox + 128)) * pix_x
       = xh_old + (640 - ox - 128) * 0.012
       = xh_old + (512 - ox) * 0.012

yh_new = yh_old - (imy/2 - (oy + 128)) * pix_y
       = yh_old - (512 - oy - 128) * 0.012
       = yh_old - (384 - oy) * 0.012
```

(y flipped because `y_metric = (imy/2 - y_px) * pix_y`.)

All other `.ori` fields (exterior orientation, rotation matrix, focal length, reference point)
and all `.addpar` fields are copied unchanged.

### 3.4 Update `parameters/ptv.par`

Change only:
```
imaX  1280 → 256
imaY  1024 → 256
```
All other parameters (pixel pitch, refractive indices, camera count) stay identical.

---

## Phase 4 — Write Result Files

### 4.1 `res/ptv_is.{frame}` (ground truth 3D positions)

Rows for the subset particles only, using the trajectory data from Phase 2.
- `prev_frame_id`: label of this particle in the previous frame (`-1` if entry frame)
- `next_frame_id`: label of this particle in the next frame (`-2` if exit frame)

Format:
```
<count>
<prev_id>  <next_id>  <X>  <Y>  <Z>
```

### 4.2 `res/rt_is.{frame}` (ground truth tracking result, frames 10001–10004)

Rows for subset particles. The `n_targets_camX` values come from the filtered `_targets` files
(how many targets were matched to this particle in each camera).

Format:
```
<count>
<label>  <X>  <Y>  <Z>  <n1>  <n2>  <n3>  <n4>
```

### 4.3 `res/added.{frame}`

Particles that appear for the first time in this frame (status `entry` or `transient`).
Same format as `ptv_is`.

---

## Implementation Sketch

```
scripts/
  create_small_cavity.py      ← orchestrator: runs all 4 phases in order
  phase1_find_window.py       ← loads cals, projects particles, finds crop offsets
  phase2_trajectories.py      ← chains links, builds ground_truth CSVs
  phase3_crop.py              ← crops images and _targets
  phase4_write_results.py     ← writes ptv_is, rt_is, added
```

Key dependencies (all already in the project):
- `openptv2.algorithms.calibration.Calibration` — camera model I/O
- `openptv2.algorithms.imgcoord.img_coord_batch` — 3D → 2D projection
- `openptv2.algorithms.track.point_to_pixel` — 3D → pixel
- `openptv2.algorithms.parameters.ControlParams` — sensor/optics metadata
- `imageio` / `numpy` — image I/O and array ops

---

## Verification Checklist

After generation, these must all pass before the dataset is used for testing:

- [ ] Number of particles in `ptv_is.10000` matches count of unique IDs in `ground_truth/particles.csv` for frame 10000
- [ ] Every `ground_truth/projections.csv` entry falls within `[0, 256)` in both crop coordinates
- [ ] `_targets` count per camera per frame ≥ number of ground truth particles visible in that camera/frame (extra targets = real background particles not in ground truth)
- [ ] `rt_is` labels are consistent across frames for `full` trajectories
- [ ] Crop offsets applied consistently: a ground truth particle's crop projection matches the detected target within detection tolerance
- [ ] `ptv.par` in output has `imaX=256, imaY=256`
- [ ] `.ori` principal points updated using `xh_new = xh_old + (512 - ox) * 0.012` per camera
- [ ] Particles labelled `entry`/`exit`/`transient` have `NaN` positions only in the correct missing frames
- [ ] `added.{frame}` entries match the `entry` particles for that frame

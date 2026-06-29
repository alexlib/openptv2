# OpenPTV2 Consolidated Project Plan

This document integrates and outlines the remaining active development plans for the `openptv2` project. It merges the synthetic cavity dataset creation plan with the Cython/Python optimization roadmap, organized into a single, cohesive six-phase project plan.

```mermaid
graph TD
    subgraph "Part 1: Ground-Truth Dataset (Phases 1-4)"
        P1["Phase 1: Find Crop Window & Subset"] --> P2["Phase 2: Reconstruct & Smooth Trajectories"]
        P2 --> P3["Phase 3: Crop Images & Filter Targets"]
        P3 --> P4["Phase 4: Write Result Files"]
    end
    
    subgraph "Part 2: Optimization & Readability (Phases 5-6)"
        P4 --> P5["Phase 5: Readability & Dev Experience"]
        P5 --> P6["Phase 6: Performance & Speed Optimizations"]
    end
    
    style P1 fill:#d4f1f4,stroke:#333,stroke-width:2px
    style P2 fill:#d4f1f4,stroke:#333,stroke-width:2px
    style P3 fill:#d4f1f4,stroke:#333,stroke-width:2px
    style P4 fill:#d4f1f4,stroke:#333,stroke-width:2px
    style P5 fill:#ffe5ec,stroke:#333,stroke-width:2px
    style P6 fill:#ffe5ec,stroke:#333,stroke-width:2px
```

---

## Part 1: Small Synthetic Cavity Test Case (Phases 1–4)

### Goal
Produce a self-contained, ground-truth-annotated subset of the `test_cavity` experiment with 4 cameras, 256×256 pixel cropped images, and 5 frames of real particle positions and images. Everything must be traceable to a known answer at every pipeline step.

### Resolved Decisions
* **Source data**: Use `res/rt_is.10000`–`10004` and `res/ptv_is.10000`–`10004` for all 5 frames. Exclude `res_orig/` as it lacks frame 10000.
* **Coordinate system**: **Crop-relative** coordinate system. Raw pixel coords are `0..imx` referenced to the sensor center. Cropped targets will use `x_px_crop = x_px_full − ox`, and the `.ori` principal point will be updated per camera.
* **Partial trajectories**: Include all particles. Particles entering or leaving the volume mid-sequence are kept with `NaN` for missing frames.

### Target Dataset Directory Layout
```
test_data/test_cavity_small/
  cal/
    cam1.tif.ori          ← modified principal point (Phase 3)
    cam1.tif.addpar       ← copied unchanged
    ... (all 4 cameras)
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
    particles.csv         ← full ground truth table
    trajectories.csv      ← per-trajectory summary
    projections.csv       ← ground truth 2D projections per camera
```

---

### Phase 1 — Find the 256×256 Window and Extract the Particle Subset

1. **Load camera models (all 4 cameras)**:
   Use `Calibration.from_file()` from `algorithms/calibration.py`:
   ```python
   from openptv2.algorithms.calibration import Calibration
   cals = [Calibration.from_file(f"cal/cam{i}.tif.ori", f"cal/cam{i}.tif.addpar")
           for i in range(1, 5)]
   ```
   Load `ControlParams` from `parameters/ptv.par` to obtain pixel pitch (0.012 mm/px), sensor size (1280×1024), and refractive indices.

2. **Load all 3D particle positions (all 5 frames from `res/`)**:
   Use `res/rt_is.{frame}` for 3D positions with labels, and `res/ptv_is.{frame}` for forward/backward trajectory links.

3. **Project particles into all 4 cameras**:
   Use `img_coord_batch` from `algorithms/imgcoord.py` to get `(x_mm, y_mm)` relative to the image center, and convert to pixel coords:
   ```python
   # x_px = x_mm / pixel_pitch + imx / 2
   # y_px = y_mm / pixel_pitch + imy / 2
   ```
   Or use `point_to_pixel` from `algorithms/track.py`:
   ```python
   from openptv2.algorithms.track import point_to_pixel
   x_px, y_px = point_to_pixel(point_xyz, cal, cpar)
   ```

4. **Select the 256×256 window in each camera**:
   Project volume centroid `(0, 2.5, 2.5)` mm to get `(cx_cam, cy_cam)` in full-image pixels. Establish a crop window of `[cy−128 : cy+128, cx−128 : cx+128]`.
   
   **Filter**: A particle is in the subset if its projection falls inside the 256×256 window in all 4 cameras in at least one frame. Record the top-left offset `crop_offset[cam] = (ox, oy)`.

---

### Phase 2 — Reconstruct Ground Truth Trajectories

1. **Chain trajectories across frames**:
   Combine the `label` field in `res/rt_is.{frame}` with `prev_frame_id` / `next_frame_id` links in `res/ptv_is.{frame}`. 
   * Collect `(X, Y, Z)` across frames 10000–10004.
   * Represent absent frames with `NaN` (entry/exit).

2. **Smooth trajectories to produce ground truth**:
   For each particle with $\ge 3$ valid frames, fit a degree-2 polynomial (degree-1 if 2 frames) through positions independently per axis. Evaluate the polynomial at each frame to produce smoothed ground truth positions.

3. **Classify trajectories**:
   * `full`: present in all 5 frames.
   * `entry`: first appears at frame > 10000.
   * `exit`: last present at frame < 10004.
   * `transient`: both enters and exits mid-sequence.

4. **Export CSV schemas**:
   * `ground_truth/particles.csv`: `particle_id, frame, X, Y, Z, dx, dy, dz, status`
   * `ground_truth/projections.csv`: `particle_id, frame, cam, x_px_full, y_px_full, x_px_crop, y_px_crop` where `x_px_crop = x_px_full - ox`, `y_px_crop = y_px_full - oy`.
   * `ground_truth/trajectories.csv`: `particle_id, first_frame, last_frame, n_frames, status`

---

### Phase 3 — Crop Real Images and Filter Targets

1. **Crop real images**:
   For each camera `c` and frame `f`, crop raw images using offsets:
   ```python
   import imageio
   img = imageio.imread(f"img/cam{c}.{f}")  # 1280×1024
   ox, oy = crop_offset[c]
   crop = img[oy:oy+256, ox:ox+256]
   imageio.imwrite(f"test_cavity_small/img/cam{c}.{f}", crop)
   ```

2. **Filter and offset `_targets` files**:
   Filter targets where `ox <= x_px < ox+256` and `oy <= y_px < oy+256`. Offset coordinates: `x_out = x_px - ox`, `y_out = y_px - oy`, and renumber target index starting from 0.

3. **Update `.ori` principal point per camera**:
   After cropping to 256×256, the sensor center relative to the crop shifts. Update the principal points `xh` and `yh` (in mm):
   $$xh_{new} = xh_{old} + (512 - ox) \times 0.012$$
   $$yh_{new} = yh_{old} - (384 - oy) \times 0.012$$

4. **Update `parameters/ptv.par`**:
   Modify dimensions to fit the crop:
   ```
   imaX  256
   imaY  256
   ```

---

### Phase 4 — Write Result Files

1. **`res/ptv_is.{frame}`**: Write subset particles with their smoothed coordinates and `prev_frame_id` / `next_frame_id` links.
2. **`res/rt_is.{frame}`**: Write target count details `n_targets_camX` matched to the particle in each camera.
3. **`res/added.{frame}`**: Write entries corresponding to first-time appearing particles (`entry` or `transient`).

#### Verification Checklist
- [ ] Unique particle count in `ptv_is.10000` matches unique ID count in `ground_truth/particles.csv` at frame 10000.
- [ ] Every coordinate in `ground_truth/projections.csv` falls inside `[0, 256)` boundaries.
- [ ] `ptv.par` has `imaX=256, imaY=256`.
- [ ] `.ori` principal points updated correctly per formula.

---

## Part 2: Optimizations & Developer Experience (Phases 5–6)

### Phase 5 — Readability & Developer Experience Improvements

1. **Standardize Type Annotations Across the Project**:
   * Ensure all `@cython.cclass` files use standardized Cython type-hints (e.g., `cython.double` and `cython.int`) rather than raw C definitions.
   * Add PEP 484 Python type-hints to all pure Python interfaces and wrappers to assist IDE auto-completion.

2. **Support Rapid Iterative Builds**:
   * Maintain the `DEV_BUILD` environment variable toggle inside `setup.py` to compile with `-O0` or `-O1` instead of `-O3`, keeping compile times under 35 seconds.

3. **Enhance Docstrings for Cython/Python Dual Execution**:
   * Document compiled vs. interpreted fallback behavior at the module level in each pure Python algorithm file.
   * Outline expected array-layout/dtypes for each function input so developers know exactly what shape Cython-typed memory views expect.

---

### Phase 6 — Performance & Speed Optimizations

1. **Optimize AoS to SoA Synchronization (`Frame` class)**:
   * Keep `@cython.cclass` types (`Target`, `Corres`, `Pathinfo`) as primary data structures to maintain object-oriented readability.
   * Avoid synchronization overhead inside `Frame` by ensuring all local loop variables and array bindings are fully Cython-typed (`p: Pathinfo`, `c: Corres`). This compiles down to direct C pointer attribute access (`p->prev`, `c->nr`), completely bypassing Python dictionary lookup overhead.

2. **Optimize Memory Views using Contiguous Slices**:
   * Explicitly declare all multidimensional typed memory views inside hot paths as contiguous in memory (e.g., `double[:, ::1]` instead of `double[:, :]`).
   * *Benefit*: Tells the compiler that the data is layout-contiguous, allowing GCC/Clang to generate optimized SIMD vector instructions.

3. **Pre-Allocate Temporary Arrays**:
   * In high-frequency recursive or iterative functions (e.g., `multimed_r_nlay_iterative` or `find_candidates_in_3d`), avoid allocating temporary small arrays/lists. Pass a pre-allocated scratch space array through the function arguments instead to eliminate GC overhead.

4. **Streamline Parallel Execution via Numba**:
   * Ensure fallback pathways using Numba JIT are decorated with `@njit(parallel=True)` where appropriate and utilize `prange` for thread-level loop parallelization.

---

### Phase 7 — Algorithmic Modernization & Deep Cython Optimization

1. **Integration of SciPy Algorithms**:
   * Replace custom or legacy mathematical routines with highly optimized, robust implementations from the `scipy` ecosystem where applicable (e.g., `scipy.spatial`, `scipy.optimize`, `scipy.ndimage`).
   * *Benefit*: Standardizes algorithms and removes the maintenance burden of custom numeric code.

2. **Rigorous Array/Cython Type Management**:
   * Carefully ensure that we do **not mix** standard NumPy arrays (`np.ndarray`) with native Cython C-structs/classes inside tight loops.
   * Passing Python objects or generic arrays into Cython C-functions triggers the GIL and causes heavy computational penalties due to type-coercion.
   * Enforce strict boundary layers where SciPy/NumPy calculations are done in bulk, and the resulting arrays are passed directly to Cython as strictly typed Memoryviews (e.g., `double[:, ::1]`).

3. **Systematic Profiling & Compilation via `cython-optimize` Workflow**:
   * **Profile & Baseline**: Establish empirical baselines for critical algorithm components.
   * **Modern Refactoring**: Refactor code to replace Python bottlenecks with typed memoryviews, strictly typed loop counters, and `cdef` internal functions. Bypass the GIL using `nogil` blocks for purely numerical loops.
   * **Compile & Tune**: Compile the code and analyze HTML annotation reports (`cython -a`). Refactor any deep yellow (Python API) interactions down to white/light yellow.
   * **Verify**: Guarantee the output exactly matches the original algorithms via automated testing before finalizing optimization.

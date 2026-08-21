# Tracer Self-Calibration ("Shaking")

Refine an existing camera calibration using real tracked tracer particles
from your flow, instead of only the calibration plate. This is the "modern
shaking" step: after a normal plate calibration, particles that were tracked
through the actual measurement volume are fed back into a joint bundle
adjustment that couples all cameras together.

## Why this exists

A calibration plate is shallow and roughly planar. Reprojection error from
plate points alone constrains a camera's pose well in the two directions
across the plate, but poorly along the camera's line of sight (depth) —
shallow parallax leaves that direction loose. Real tracer particles are
distributed through the full depth of the measurement volume, so fitting
against them tightens exactly the direction the plate can't see.

The fit couples every free camera: minimizing the *joint* reprojection error
of shared 3D particle positions (not independent per-camera resection) is
what actually reduces cross-camera disagreement — measured as the
**ray-convergence miss distance (RCM)**, the median gap between the rays
different cameras draw through the same real particle. RCM, not per-camera
reprojection RMS, is the diagnostic this step optimizes and reports, because
RMS can look fine even when cameras subtly disagree with each other (see
[`calibration-rms-vs-rcm.md`](calibration-rms-vs-rcm.md)).

## Prerequisites

1. **A working calibration already exists.** This step *refines* `.ori`/
   `.addpar` files you already have — it does not calibrate from scratch. Do
   the normal plate calibration first (detection → manual/sortgrid → raw
   orientation → fine tuning).
2. **Tracked 3D data exists on real flow images.** The routine reads tracked
   3D positions and matches them back to per-camera 2D detections, so you
   need a completed **sequence + tracking** run on actual flow data (not the
   calibration plate) before running this step.

### Store-backed runs

Tracer self-calibration reads tracked 3D points and detections from
`res/ptv_is.*`/`img/*_targets` ASCII when present, or from `res/run.zarr`
(the `RunStore`) directly when they aren't — which is the normal case since
the zarr-only storage migration, since batch/GUI sequence+tracking runs
write only to the store now. No manual export step is needed.

## Using it from the GUI

1. **Launch calibration**: from the main `pyptv2-gui`, open your experiment
   and use the menu action that starts the calibration dialog. Standalone
   alternative: `python -m openptv2.gui.calibration_gui path/to/parameters_Run1.yaml`.
2. **Load images/parameters** (first left-panel button) — required before
   any other button, including "Tracer self-cal", becomes active.
3. **Edit calibration parameters** → the **"Shaking calibration
   parameters"** group holds the settings for this step (see table below).
4. Click **Tracer self-cal** (left panel, below "Suggest eps0"). Console
   output now prints progress as it runs:

   ```
   Tracer self-cal: button clicked, starting...
   [tracer self-cal] starting: base=..., frames=10000-10004 (5), tol_px=2.0, hold_cam=1, max_particles=400, iters=3
   [tracer self-cal] loaded 5 frames of tracked data
   [tracer self-cal] initial match: 214 particles (>=2 cams), RCM before = 84.3 um
   [tracer self-cal] iter 1/3: fitting 214 particles (this may take a moment)...
   [tracer self-cal] iter 1/3: RCM = 61.7 um (accepted, improved)
   [tracer self-cal] iter 2/3: fitting 226 particles (this may take a moment)...
   [tracer self-cal] iter 2/3: RCM = 58.9 um (accepted, improved)
   [tracer self-cal] iter 3/3: fitting 229 particles (this may take a moment)...
   [tracer self-cal] iter 3/3: RCM = 58.9 um (not improved, stopping)
   [tracer self-cal] done
   ```

5. **Read the status line / console summary**: `Tracer self-cal:
   cross-camera RCM <before> -> <after> um (N particles, camK held)`.
6. **Automatic accept/reject**: the refined calibration is kept only if
   `after < before`. If it improved, the GUI backs up your current
   `.ori`/`.addpar` and applies the new calibration in memory to every
   camera except the held one — **save** the result the same way you would
   after any other orientation step to persist it to disk. If it did not
   improve, nothing is written.

## Parameters ("Shaking calibration parameters")

| Field | Meaning | Notes |
|---|---|---|
| `shaking_first_frame` / `shaking_last_frame` | Tracked frame range to pull particles from | Use a range with good particle coverage across the volume, not just a couple of frames |
| `shaking_max_num_frames` | Subsample down to this many frames (evenly spaced) if the range is larger | Keeps the fit fast on long sequences |
| `shaking_max_num_points` | Cap on total particles used in the joint fit | 200–400 is a reasonable range; more particles cost more solver time for diminishing returns |
| `shaking_tol_px` | Pixel tolerance for matching a tracked 3D point's reprojection to a real detection in each camera | The first thing to tune if you get too few/no matches (loosen) or suspiciously large post-fit RCM (tighten — you're pulling in wrong matches) |
| `shaking_hold_cam` | 1-indexed camera held fixed as the gauge reference; every other camera's exterior is free | Pick your most trusted/best-calibrated camera |

Internally, a tracked particle only enters the fit if it is matched in at
least 2 cameras (`min_cams=2`); it is **not** restricted to 4-camera
quadruplets today — see [Future work](#future-work).

## What actually happens (algorithm)

1. Load tracked 3D positions (`res/ptv_is.*`) and per-camera detections
   (`img/*_targets`) for the configured frames.
2. **Match**: for each tracked 3D point, reproject it into every camera with
   the *current* calibration and find the nearest real detection within
   `shaking_tol_px`. Keep points matched in at least 2 cameras.
3. **Fit**: one joint least-squares bundle adjustment (`scipy.optimize.
   least_squares`, sparse Jacobian) refines every free camera's exterior
   orientation (position + angles) *and* the matched particles' free 3D
   positions together, minimizing total reprojection error. Distortion
   parameters are held fixed. The held camera's exterior fixes the 7-DOF
   gauge freedom (translation/rotation/scale) that an unconstrained joint
   fit would otherwise leave undetermined.
4. **Accept or reject**: compute the median RCM on the refit. If it beats
   the previous best, keep the new calibration and continue; otherwise stop.
5. **Iterate** (`iters`, default 3 from the GUI): re-run match → fit with
   the improved calibration, since a better calibration recovers different
   (better) matches. Stops early once RCM plateaus or a pass doesn't
   improve.

## After running it

- Re-run sequence + tracking with the refined calibration, then run tracer
  self-cal again if useful — since it's inherently iterative across
  match/fit/re-track cycles, a second pass on fresh tracking output can
  keep improving RCM further.
- Cross-check the improvement with the `visualize-calibration` skill or
  `openptv2.calibration_diagnostics` (sight-line angle, cross-camera
  symmetry, cross-camera RCM) — the single before/after number the button
  prints is a summary, not the full picture.

## Troubleshooting

| Symptom | Likely cause | Try |
|---|---|---|
| "n/a: no res/ptv_is.\* frames. Run the sequence + tracking first." | Sequence+tracking never actually completed for the configured frame range (neither ASCII nor `res/run.zarr` has linkage data) | Confirm the sequence+tracking step succeeded and produced tracking output for those exact frames; widen/adjust `shaking_first_frame`/`shaking_last_frame` |
| "only N multi-cam tracer particles" (skipped, N < 10) | Too few particles matched in ≥2 cameras for the configured frames | Widen the frame range, loosen `shaking_tol_px`, or check that detection/tracking actually found particles in those frames |
| RCM doesn't improve on iteration 1 | Starting calibration is already good for this data, or `tol_px` is too tight/loose | Try a different `hold_cam`, adjust `tol_px`, or accept that this dataset doesn't need further refinement |
| Console shows nothing after clicking the button | Fixed — progress is now printed at start, per match, and per iteration (see the example output above) | Update if you're on an older build without this |

## Future work

The current fit accepts any tracked particle seen by 2 or more cameras
(`min_cams=2`). A quadruplet-gated variant — requiring all 4 cameras to see
a particle before it enters the fit — would trade point count for
per-point confidence, since a 4-camera correspondence is far less likely to
be a spurious match than a 2-camera one. This isn't implemented yet; see
`four_camera_matching()` in `src/openptv2/algorithms/correspondences.py`
for the existing quadruplet-matching machinery this would build on.

## Related

- [`calibration-bundle-adjustment.md`](calibration-bundle-adjustment.md) —
  design doc covering both plate-based joint bundle adjustment
  (`joint_plate_bundle_adjust`) and this tracer-based step
  (`tracer_self_calibrate`), plus the wider roadmap.
- [`calibration-rms-vs-rcm.md`](calibration-rms-vs-rcm.md) — why RCM catches
  cross-camera disagreement that per-camera RMS misses.
- [`calibration_best_practices.md`](calibration_best_practices.md) — the
  broader plate-calibration workflow this step follows on from.
- `src/openptv2/autocalibration.py` — `tracer_self_calibrate()`,
  `cross_camera_rcm()`.
- `src/openptv2/gui/calibration_gui.py` — the "Tracer self-cal" button and
  "Shaking calibration parameters" dialog group.

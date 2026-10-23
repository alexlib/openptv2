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

### Zarr RunStore input

Tracer self-calibration reads tracked 3D points, correspondences, and per-camera
2D detections directly from the Zarr `RunStore` (`res/run.zarr` or `run.zarr`).
Legacy ASCII `ptv_is.*` files are no longer used.

## Why real-particle filtering and camera focus matter

Naive shaking feeds *all* matched particles (including ghosts and 2- or 3-camera
partial points) into an unweighted bundle adjustment and shakes all free cameras
equally. That has two failure modes:

1. **Improving ghosts alongside real particles**: when ghost quadruplets enter
   the fit, the solver pulls the camera orientations toward making the ghost
   rays meet, legitimizing the ghosts instead of separating them from real
   particles.
2. **Ignoring which camera causes the ghosts**: 72 % of ghosts are 3-camera
   points where one camera's epipolar search missed the true target because of a
   sub-pixel calibration offset. Holding the wrong camera fixed — or weighting
   all cameras equally — fails to focus the correction on the camera causing the
   mismatch.

`tracer_self_calibrate` solves both:

- **Per-camera ghost diagnostics (`camera_ghost_diagnostics`)**: measures each
  camera's 3-camera missing share and its 4-camera reprojection residual on
  high-confidence real particles, identifying `best_cam` and `worst_cam`.
  Passing `hold_cam="auto"` automatically holds `best_cam` as the gauge
  reference and focuses residual weights (`focus_cameras=True`) on the cameras
  with higher badness scores.
- **Real-particle selection & progressive tightening (`filter_ghosts=True`)**:
  prioritizes linked all-camera (`n_cams`) particles whose ray-convergence ghost
  probability `p_ghost` is below a threshold that tightens across iterations
  (`ghost_prob_schedule=(0.25, 0.15, 0.08)`), weighting each particle by
  `1 - p_ghost` with a robust `soft_l1` loss.
- **Increasing real-vs-ghost separation**: as the cameras align on real
  particles, real particles' RCM drops by ~2× (`37.4 -> 17.8 um` on
  `L1_d1_c0_k1_n200_s0.08`) while true ghosts stay at ~`67 um`, doubling the
  ghost-to-real separation ratio (`1.76x -> 3.77x`, AUC `0.827 -> 0.932`).
- **Immediate store update (`retriangulate_store`)**: re-triangulating the
  stored correspondences in `res/run.zarr` with the shaken calibration cuts
  trajectory ghost fraction by 34–73 % and lowers velocity error by 5–14 %
  across the benchmark suite with zero tracking runtime penalty:

| Case | Mode | `lnkP` | `vErr` | `aErr` | `ghost` | `longC` | `frag` |
|---|---|---|---|---|---|---|---|
| `L1_d1_c0_k1_n200` | Unshaken | 0.9645 | 0.1601 | 8.709 | 0.0353 | 0.532 | 2.16 |
| | **Shaken** | **0.9906** | **0.1378 (−13.9 %)** | **7.306 (−16.1 %)** | **0.0095 (−73.1 %)** | 0.526 | **2.08** |
| `L1_d1_c0_k1_n200_s0.08` | Unshaken | 0.9627 | 0.1839 | 11.310 | 0.0365 | 0.520 | 2.27 |
| | **Shaken** | **0.9799** | **0.1676 (−8.9 %)** | **10.072 (−10.9 %)** | **0.0197 (−46.0 %)** | **0.523** | **2.17** |
| `L1_d2_c0.3_k4_n150` | Unshaken | 0.9488 | 0.1593 | 2.754 | 0.0481 | 0.344 | 2.32 |
| | **Shaken** | **0.9685** | **0.1507 (−5.4 %)** | **2.708 (−1.7 %)** | **0.0297 (−38.3 %)** | **0.345** | **2.24** |
| `L1_d4_c0_k1_n150` | Unshaken | 0.9353 | 0.3828 | 24.422 | 0.0645 | 0.172 | 1.56 |
| | **Shaken** | **0.9590** | **0.3470 (−9.4 %)** | **21.251 (−13.0 %)** | **0.0429 (−33.5 %)** | 0.170 | **1.47** |

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
| `shaking_hold_cam` | 1-indexed camera held fixed as the gauge reference (`"auto"` in Python selects the cleanest camera) | Pick your most trusted/best-calibrated camera, or pass `hold_cam="auto"` in Python |

By default (`filter_ghosts=True`), `tracer_self_calibrate` prioritizes linked
4-camera (all-camera) real particles with low ghost probability (`p_ghost`) and
progressively tightens the `p_ghost` threshold across iterations.

## What actually happens (algorithm)

1. Load tracked 3D positions, correspondences, linkage, and per-camera 2D
   detections from the Zarr `RunStore` (`res/run.zarr`).
2. **Diagnose cameras** (`camera_ghost_diagnostics`): measure each camera's
   3-camera missing share and 4-camera reprojection residual on high-confidence
   real particles to identify `best_cam` and `worst_cam`.
3. **Select real particles**: filter out ghost candidates using ray-convergence
   ghost probability (`p_ghost <= q_thresh`), prioritizing linked all-camera
   particles and weighting each particle by `1 - p_ghost`.
4. **Fit**: one coupled least-squares bundle adjustment (`scipy.optimize.
   least_squares`, sparse Jacobian, `soft_l1` loss) refines every free camera's
   exterior orientation (position + scaled mrad angles) *and* the real
   particles' 3D positions together, including the held reference camera's 2D
   observations in the residual and focusing camera weights on higher-badness
   cameras.
5. **Accept or reject & tighten**: evaluate the median RCM on the reference real
   particle set and record the ghost-to-real RCM separation ratio. Tighten
   `q_thresh` (`ghost_prob_schedule`) on the next iteration and stop when RCM
   plateaus.

## After running it

- Call `retriangulate_store(base, new_cals, cpar)` to update the 3D
  correspondence positions in `res/run.zarr` immediately, or re-run sequence +
  tracking (`recorrespond` + `track_store`) with the refined calibration.
- Cross-check the improvement with the `visualize-calibration` skill or
  `openptv2.calibration_diagnostics` (sight-line angle, cross-camera
  symmetry, cross-camera RCM).

## Troubleshooting

| Symptom | Likely cause | Try |
|---|---|---|
| "no Zarr RunStore found (run sequence + tracking first)" | Sequence+tracking has not been run yet (no `res/run.zarr`) | Run sequence + tracking first so `res/run.zarr` holds correspondences and linkage |
| "only N multi-cam tracer particles" (skipped, N < 10) | Too few particles matched in ≥2 cameras for the configured frames | Widen the frame range, loosen `shaking_tol_px`, or check that detection/tracking found particles in those frames |
| RCM doesn't improve on iteration 1 | Starting calibration is already good for this data, or `tol_px` is too tight/loose | Try `hold_cam="auto"`, adjust `tol_px`, or accept that this dataset doesn't need further refinement |

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
  `camera_ghost_diagnostics()`, `retriangulate_store()`, `cross_camera_rcm()`.
- `src/openptv2/gui/calibration_gui.py` — the "Tracer self-cal" button and
  "Shaking calibration parameters" dialog group.

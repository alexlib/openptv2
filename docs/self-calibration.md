# Joint self-calibration from tracer particles

`openptv2.selfcal.joint_self_calibrate` refines the camera parameters (exterior orientation, principal point, radial and
decentering distortion) and the 3D positions of many tracer particles together, by bundle adjustment on their blob
centres. The gauge is fixed by holding one camera's exterior and by soft priors, so the world frame stays put. Existing
`autocalibration.tracer_self_calibrate` fits exteriors only, on 400 particles, with distortion fixed.

Use on a finished run (clean 4-camera points, held-out frames score the result):

    uv run python scripts/selfcal_real.py <run folder> --variants all --nfev 400 --tracks --save <new cal folder>
    uv run python scripts/loco_map.py <run folder>          # is the leave-one-camera-out floor systematic?
    uv run python scripts/selfcal_synthetic.py --scale 0.3  # known answer on the rendered synthetic recording

## Order of work: regular calibration first, self-calibration as the tuning step

1. **Regular calibration** (plate / calibration target, or dumbbell calibration) gives the starting cameras and the metric
   scale and frame.
2. **Self-calibration** then tunes those cameras on tracers, because tracers see the whole volume. Tracers constrain
   consistency between cameras, **not scale**: a uniform stretch of the volume reprojects equally well. So the fit is
   anchored to a known object seen by the same cameras, one or both of:
   - `KnownBody(xyz, obs_px)`: points with known world coordinates (the calibration target). Fixes scale and frame.
     `known_body_from_calibration(experiment_folder, cals, cpar)` builds it from the calblock and the calibration images
     (matched with `sortgrid` and the current cameras; observations farther than 3 px from their projection are dropped).
   - `Dumbbell(obs_px, length_mm)`: two points a known distance apart moved through the volume (`dumbbell_scale` in the
     YAML). Fixes scale only. `dumbbell_from_store(store, num_cams, length_mm, first=, last=)` reads frames with exactly two
     targets per camera.

```python
res = joint_self_calibrate(obs_px, x0, cals, cpar, body=known_body_from_calibration(folder, cals, cpar),
                           dumbbell=dumbbell_from_store(store, 4, 30.0, first=1, last=50), vpar=vpar)
res.length_rms_mm_before, res.length_rms_mm_after, res.body_rms_mm_after   # how well the anchors are met
```

Without an anchor the result is defined only up to a scale: use it for consistency (leave-one-camera-out), not for
absolute velocities. The exterior priors (`ext_sigma_scale`, 0.5 mm by default) keep the cameras near the regular
calibration; a large scale error needs them loosened (the unit tests use 200).

Known-answer tests (`tests/unit/test_selfcal.py`): with the cameras stretched by 2 % about camera 1, tracers alone keep the
wrong scale, a dumbbell of known length brings the camera positions back to within 0.1-0.2 mm (from up to 23 mm) and
the pixel residual to 0.0004 px; a 40-point known body recovers the cameras from a 0.3 % scale error to within 0.6 mm.

Real wp2, anchored (300 frames, 6,000 tracer points, 70 calibration-target points seen by 44-56 cameras each, all cameras
free, short fit of 60 iterations, not converged): held-out leave-one-camera-out 0.306 -> 0.298 px (-2.8 %; tracers alone
-2.7 %), known-body misfit after the fit 0.021 mm; the new cameras change positions by 0.04 mm (median) of which all but
0.0007 mm is a global affine map with linear coefficient below 0.12 %. Triangulating the 10 body points seen by three or
more cameras (after dropping mismatched targets): scale factor 1.00021 with the regular calibration, 1.00005 after
self-calibration, 3D error median 0.19 mm -> 0.18 mm. So here the regular calibration already has the right scale, and the
anchored fit keeps it (the unanchored full-run fit below let a 0.5 % linear drift in). Caveats: 10 usable body points, one
dataset, and 1-2 px median reprojection of the target points under the regular calibration (the calibration images of this
dataset are old, so the body is a modest anchor; a dumbbell moved through the volume would be a stronger one).

What it can and cannot show (2026-11-21):
- Synthetic with a known perturbation (leave-one-camera-out 0.67 px): the fit recovers the true state, position error against
  truth after removing the gauge equals that of the true calibration (0.018 mm), leave-one-camera-out 0.10 px.
- Real wp2 (1,072 frames): held-out leave-one-camera-out 0.304 -> 0.250 px (-17.5 %); track jitter unchanged (-1 %); absolute
  positions move by about 0.09 mm (median), of which a global linear part of about 0.5 % (the scale is not constrained by
  tracers). Tracer data fix consistency between cameras, not absolute scale; check the scale against a known length
  (dumbbell) before using a new calibration for velocities.

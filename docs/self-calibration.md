# Joint self-calibration from tracer particles

`openptv2.selfcal.joint_self_calibrate` refines the camera parameters (exterior orientation, principal point, radial and
decentering distortion) and the 3D positions of many tracer particles together, by bundle adjustment on their blob
centres. The gauge is fixed by holding one camera's exterior and by soft priors, so the world frame stays put. Existing
`autocalibration.tracer_self_calibrate` fits exteriors only, on 400 particles, with distortion fixed.

Use on a finished run (clean 4-camera points, held-out frames score the result):

    uv run python scripts/selfcal_real.py <run folder> --variants all --nfev 400 --tracks --save <new cal folder>
    uv run python scripts/loco_map.py <run folder>          # is the leave-one-camera-out floor systematic?
    uv run python scripts/selfcal_synthetic.py --scale 0.3  # known answer on the rendered synthetic recording

What it can and cannot show (2026-11-21):
- Synthetic with a known perturbation (leave-one-camera-out 0.67 px): the fit recovers the true state, position error against
  truth after removing the gauge equals that of the true calibration (0.018 mm), leave-one-camera-out 0.10 px.
- Real wp2 (1,072 frames): held-out leave-one-camera-out 0.304 -> 0.250 px (-17.5 %); track jitter unchanged (-1 %); absolute
  positions move by about 0.09 mm (median), of which a global linear part of about 0.5 % (the scale is not constrained by
  tracers). Tracer data fix consistency between cameras, not absolute scale; check the scale against a known length
  (dumbbell) before using a new calibration for velocities.

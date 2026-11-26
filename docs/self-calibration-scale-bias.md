# Self-calibration from tracers changes the volume: a bias that is easy to miss

**Short version.** Fitting cameras to tracer particles makes the cameras agree with each other, which is what removes the
camera-switching jump and improves tracking. It does **not** tell the fit what the size or the shape of the measurement volume is.
In every run we made, the fit also changed the volume: a scale change and an uneven stretch of a fraction of a percent up to
about one percent. The reprojection error cannot see it, the leave-one-camera-out error cannot see it, and the tracking scores
cannot see it. Nothing in a normal self-calibration report mentions it. This page documents what we measured, what it does to the
numbers you publish, what we do not know, and how to guard against it.

This is a property of the *method* (fitting reprojection residuals of tracers), not of this implementation. We measured only our
own; we make no claim about other software, but the invariance below is mathematics, so any method that fits tracers alone has it.

## Why tracers cannot define the scale

For a pinhole camera, multiplying every camera position and every particle position by the same factor `s` leaves every image
coordinate unchanged. Every reprojection residual, every ray-intersection miss distance and every leave-one-camera-out error is
therefore *identical* for the old cameras and for cameras stretched by `s`. The data contain no information about `s`.

What breaks the invariance weakly is everything that has a physical length not stretched along: the glass thickness and its
distance, the refractive indices, the lens distortion model. So the fit is weakly constrained, not unconstrained, and weakly
constrained directions drift with noise, with outliers and with whatever the optimizer prefers. The same argument applies to a
shift and rotation of the whole volume, and (more weakly) to some combinations of stretches with the glass parameters.

## What we measured

`self_calibrate_from_store` reports the best affine map between the points triangulated with the old cameras and with the new
ones, as three principal stretches (1.0 = unchanged) and a mean shift. Priors on the camera positions were left at the default
unless stated.

**Truth known (rendered syn1, the volume scale against the true positions is measured directly):**

| start | stretches after the fit | what it means |
|---|---|---|
| a **perfect** calibration (zero error) | 1.00037, 1.00069, 1.00102 | the fit *introduced* a +0.07 % scale error (and a 0.065 % spread) where there was nothing to correct. Self-calibration is then declined by the CLI; it was forced here to measure this |
| real syn1 calibration error | 1.00025, 1.00046, 1.00073 | truth-referenced scale 0.99973 → 0.99926: the fit moved the scale **away** from the truth by 0.05 % |
| doubled error | (map scale) 0.99948 → 0.99907 | away from the truth again by 0.04 % |

**Real recordings (no truth, so we cannot say which side is right):**

| recording | stretches (old -> new volume) | geometric-mean scale | mean shift |
|---|---|---|---|
| wp2 frames 1-300 | 0.9947, 1.0039, 1.0107 | +0.31 % | 2.2 mm |
| wp2 frames 700-900 | 0.9945, 1.0033, 1.0099 | +0.25 % | 2.2 mm |
| wp2 all 1072 frames | 0.9958, 1.0042, 1.0110 | +0.37 % | 2.5 mm |
| wp1 frames 1-300 | 0.9949, 1.0037, 1.0110 | +0.32 % | 2.3 mm |
| wp1 all 1072 frames | 0.9950, 1.0036, 1.0101 | +0.29 % | 2.1 mm |
| lv_multi wp5 (20 frames) | **0.9843, 0.9936, 0.9939** | **-0.94 %** | **4.8 mm** |
| lv_multi wp4 (20 frames) | **0.9822, 0.9924, 0.9926** | **-1.09 %** | **6.2 mm** |
| test_cavity (opposite cameras, 5 frames) | 0.9952, 0.9989, 0.9999 | -0.20 % | 0.26 mm |

wp1 and wp2 are the same rig and agree with each other to 0.01 %: the deformation is reproducible, which is what a real
calibration error (or a systematic property of the fit) would do, not noise. lv_multi is a different rig and the change is three
times larger.

**An anchor helps, but not with everything.** On lv_multi wp5 the 95 calibration-target points (seen by 74-88 of the 95 in each camera;
`known_body_from_calibration`) as a known body reduce the volume change from -0.94 % to -0.14 % (stretches 0.9941, 1.0004,
1.0013) and the shift from 4.78 mm to 0.04 mm, with the same removal of the camera-subset jump (24.2 -> 0.8 µm). So an anchor
removes most of the deformation, and it can be built from the calibration images in the folder. (It did not change the tracking loss
that lv_multi showed with the first version of the fit; that loss was caused by the lens-term priors and is fixed, see
`docs/calibration-consistency.md`. The deformation itself is unchanged by that fix: re-measured with it on wp1, wp2 (300 frames) and both lv_multi runs, the stretches agree with the table above to 0.1 %.) Anchoring on wp2
is not measured.

"+0.3 % scale" is the geometric mean of the three. The deformation is **not uniform**: on wp1 and wp2 one direction is stretched by about 1.1 % while another is compressed by 0.5 %; on lv_multi all three shrink, one by 1.8 %. The directions are mixtures of the lab axes (the map also
contains a rotation), not one clean depth stretch. Tighter priors on the camera positions (0.3, 1.0, 3.0 times the default)
leave the wp2 change at +0.27...+0.31 %; looser priors on focal length / glass make it worse (test_cavity: -0.2 % → -0.5 %).
So the data drive it, and more freedom gives more of it.

**What the truth-known synthetic cavity adds** (`docs/self-calibration-cavity-synthetic.md`): with cameras on both sides, after the fit the
volume keeps its shape to 0.35 % of the truth (stretches 0.9984 ... 1.0035) in 13 different experiments, and a fluid-index error of
0.005 alone gives a 0.27 % depth stretch that the fit leaves in place. The real volume changes (0.5-1.8 %) are several times larger
than anything the fit produced on its own there, which favours "correction of real calibration errors" over "bias of the fit", but
that is still an inference. The choice of the reference camera changes the unanchored volume change on real data: wp2 +0.34 %
(camera 1 held), +0.40 % (the most consistent camera, the default), +0.45 % (none held).

**What we do not know:** whether the real-data deformation is a *correction* (the old calibration really has that error and the
tracers found it) or the *bias* seen in the synthetic runs, ten times larger. The synthetic runs say the method can introduce
0.04-0.07 %; they do not say it introduces 0.3-1 %. Only an independent length (dumbbell, calibration plate, a known particle
spacing) can tell.

## What it does to your results

* **Uniform scale error `s`:** lengths, positions, velocities and accelerations scale by `s`; velocity *gradients* (strain rate,
  vorticity, and so dissipation) are unchanged to first order; kinetic energy goes with `s²`; integral length scales with `s`;
  a volume flux with `s³` (0.3 % scale = 0.9 % flux).
* **Unequal stretches (what we saw):** a stretch along one direction multiplies the velocity component along it and the
  derivatives along it differently, so strain-rate and vorticity components change too; by the order of the stretch difference,
  here up to about 1.5 %. Statistics that average over components (dissipation from all nine gradients) change less than single
  components.
* **Shift:** harmless for velocities and gradients; it matters only when coordinates are compared with something else (a wall, a
  second experiment, a mask).
* **What it does not do:** it does not reduce the benefit that was measured (consistency, tracks, ghosts). The tracking scores
  improve with the bias present because they cannot see it.

## What to do

1. **Always read the scale line** of `self_calibrate_from_store` / the CLI. It is printed whenever no anchor was given, with the
   three stretches.
2. **Anchor the fit** when absolute numbers matter (measured on lv_multi: scale -0.94 % -> -0.14 %, shift 4.8 -> 0.04 mm): pass a `KnownBody` (calibration target points) and/or a `Dumbbell` (two points
   a known distance apart through the volume) to `self_calibrate_from_store(..., body=..., dumbbell=...)`; see
   `docs/self-calibration.md`; `known_body_from_calibration(run folder, cals, cpar)` builds one from the calibration images and the
   calblock when they are in the folder. With an anchor the printed warning disappears. An anchor reduces the deformation, it does
   not make it zero (lv5: still -0.6 % along one direction).
3. **Or avoid changing the gauge:** if you must keep the validated scale and frame, use `align_partial_points` (layer 3 of
   `docs/calibration-consistency.md`), which leaves every 4-camera point exactly where it was.
4. **Check after the fact** without any anchor: triangulate the same blobs with the old and the new cameras and fit the affine map;
   `RecalibrationResult.scale_axes`, `.stretch_dirs`, `.shift_mm` are exactly that. A known length measured with both calibrations
   is better still.
5. **Report it.** If you publish velocities obtained with self-calibrated cameras, state the unanchored scale change next to the
   consistency improvement.

## What we tried and dropped

* Tighter priors on the camera exteriors: no change (above).
* Soft constraints that hold the centroid and size of the point cloud: they couple every point and make the sparse Jacobian dense
  (about 10 times slower); dropped.
* Free glass thickness / fluid index / glass distance (`free_multimedia`, `free_glass`): they are weakly identifiable (fitted
  `n3` 1.416 for a true 1.45 in the lab) and do not remove the deformation; off by default.

## Where this is tested

`bench/calibration_lab/real_selfcal.py` and `selfcal_vs_align.py` print the stretches; `tests/unit/test_calibration_consistency.py`
checks that an anchored fit does not print the warning and that a fit on already consistent cameras is declined (it would only
move the scale).

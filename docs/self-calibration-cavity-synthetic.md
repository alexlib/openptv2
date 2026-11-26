# Self-calibration on a synthetic cavity: what it does with cameras on both sides, with the truth known

Real recordings cannot tell whether a self-calibrated camera set is *right*, only whether it is *consistent*. This page uses a
synthetic rig close to `test_cavity` whose truth is known, to see what self-calibration really does, with cameras at negative and
positive z and glass vectors of both signs. Run it with `bench/calibration_lab/cavity_synth_lab.py`; regression tests are in
`tests/unit/test_selfcal_cavity_synthetic.py` and `tests/unit/test_synthetic_rig.py`. The truth is the independent ray tracer of
`docs/synthetic-rig.md`.

## The rig

A water cell between two flat windows (glass 6 mm, n = 1 / 1.46 / 1.33, windows at z = -125 and +125 mm) seen by two cameras on each
side (z about -575 and +580, geometry of `test_cavity_small`), 1280 x 1024 sensors of 12 µm, `cc` 70 mm, and per-camera lens terms
and principal-point offsets of the size the real refinement found. 900 tracers per frame in a +-30 mm cube, each camera detects a
tracer with probability 0.8 (about 41 % four-camera points, as in the real data), centroid noise 0.15 px, 6 frames. The model is the
library with a deliberately wrong calibration: one thing wrong per experiment. The blobs are matched with the model, the cameras are
fitted from those matches (`self_calibrate_from_store`, matching-yield test off so the fit is always evaluated) and the blobs are
matched again.

Scores (all against the truth): recall of the particles seen by all four cameras, share of wrong quadruplets (ghosts), the
**calibration error in image space** (rms pixel difference between the model's and the truth's image of random volume points), the
3D error, the principal stretches and shift of the volume against the truth.

## 1. Self-calibration works from a calibration that still matches, and not otherwise

| calibration error before (image-space rms) | recall before | after the fit |
|---|---|---|
| 0.0 px (control) | 0.998 | forced fit adds 0.13 px (the noise level), volume stretches within 0.02 % |
| 0.1 ... 1.5 px (the 13 experiments of section 2) | 0.76 ... 0.998 | **0.996 ... 0.998**, ghosts 4.0 % (the truth cameras: 0.998 / 3.9 %), jump 1-4 µm |
| 3.3 px (fluid index +0.02), 3.7 px (drift x1), 11 px (drift x3), 72 px (lens ignored, 0.8 mm principal point) | 0.000, ghosts 1.0 | **nothing to fit**: the matcher finds no correct quadruplet and the report says "not applicable" |

The capture range is set by the matcher's epipolar tolerance (about 3 px): the fit starts from the matches the old cameras produce.
A calibration that is further off than that cannot be repaired this way; use a plate / target calibration first.

## 2. Consistency is restored; the truth is another matter

Over 13 experiments (camera drift of all / of one camera, lens terms at 85 %, glass distance +-1 mm on either side, glass 2 mm
thicker, fluid index +0.005, a 0.3 degree tilted window, one camera pair offset by 0.3 mm), every one ends with the camera-subset jump
at 1-4 µm and the matching recall of the true cameras. The image-space calibration error against the truth, however, depends on
what was wrong and on how the fit is anchored. Mean and worst case over the 13, by what is held as the reference:

| reference in the fit | mean error after | worst | worse than before |
|---|---|---|---|
| (before the fit) | 0.65 px | 1.52 | |
| camera 1 held (the old default) | 0.52 px | **1.60** | 3 |
| the camera with the smallest residual held ("auto") | 0.47 px | 0.96 | 3 |
| no camera held (priors only) | **0.41 px** | 0.93 | **1** |

On the real recordings (wp1, wp2 300 frames, lv_multi wp4 / wp5) the three strategies give the same matching yield and the same
tracking (wp2 tracks >= 50 frames 1713 / 1715 / 1714 for camera 1 / auto / none), since the data are already consistent; they differ in
the unanchored volume change (wp2: +0.34 / +0.40 / +0.45 % mean scale, 2.2 / 2.2 / 2.9 mm shift): holding no camera lets the volume
move the most. `self_calibrate_from_store` therefore defaults to `hold_cam="auto"` (the camera with the smallest residual is held; the
result says which); `"none"` and an explicit index are available. The data above do not make a strong case for any of them.

No strategy dominates. A camera that is held keeps its own errors for good and the others are moved to agree with it: when camera 1
had drifted and was held, the fit restored consistency (recall 0.934 -> 0.998, jump 176 -> 2.6 µm) and made the calibration worse in
image space (1.03 -> **1.60 px**); holding a camera that was right gave 0.43-0.63 px. Holding none is best on average but moved a
tilted window the wrong way (0.49 -> 0.93 px). **Consistency metrics are blind to this**: the jump and the recall are the same
for all strategies while the error against the truth varies by a factor of three. Only an independent reference (a calibration target,
a dumbbell) can tell the solutions apart.

## 3. What is, and what is not, recoverable

* **Glass distance** (+-1 mm, on the -z pair, the +z pair or all four): never recovered, with or without the glass distance free in the
  fit (the error stays at +-1.00 mm). The cameras absorb it. Signs and sides behave alike (no sign bug): the same error on the +z pair
  and on the -z pair end at comparable image-space errors once the reference is not the affected side.
* **Glass thickness** (+2 mm): not recovered (fitted 6.005 for a truth of 8), absorbed.
* **Fluid index** (+0.005): partly (fitted 1.331 for 1.335); the opposite sides make it slightly more identifiable than in the one-sided lab.
* **A rigid 0.3 mm offset of one camera pair along z**: consistency restored, but the volume keeps a 0.23 mm shift against the truth
  (a translation, the gauge). Velocities do not see it.
* **Volume shape** after the fit stays within 0.35 % of the truth's (principal stretches 0.9984 ... 1.0035), against 0.04-0.10 % on
  the one-sided rendered syn1 and 0.5-1.8 % on the real recordings. The real changes are therefore much larger than this synthetic
  fit ever produces on its own: they look like corrections of real calibration errors rather than a bias of the fit, but that is an
  inference, not a measurement (a fluid-index error alone gives a 0.27 % depth stretch before the fit and the fit leaves it).

## 4. Invariance tests (identical blobs, the whole rig moved)

| move | recall after | calibration error after | stretches after |
|---|---|---|---|
| none (reference) | 0.996 | 1.209 px | 0.9996 / 0.9997 / 1.0016 |
| rotated 180 degrees about y (the +z cameras become -z, glass vectors with them) | 0.996 | 1.212 | **identical** |
| translated by (6, -12, 5) mm | 0.996 | 1.218 | 1.0018 |
| reversed camera order | 0.995 | 1.141 | 1.0024 |
| general rotation (axis (1, 2, 3), 0.9 rad) and shift | 0.985 | 1.389 | 1.0021 |
| rotated 90 degrees about x | **no matches before or after** | | |

* The half-turn gives the same answer to three digits: no sign, no z-axis dependence in the fit. Translation and order differ by 1-15 %:
  solver-path sensitivity along directions the data cannot fix (the same data with the camera order reversed and no camera held gave
  0.89 and 1.33 px). That is itself a finding: the solutions are a flat family.
* **The 90-degree case fails in the matcher, not in the fit**: the observation volume is specified by x limits and two z slabs
  (`X_lay`, `Zmin_lay`, `Zmax_lay`), which bound the epipolar search along the depth. A rig whose cameras look along y has no matches at
  all with those limits. The pipeline assumes the cameras look roughly along lab z.
* The general rotation loses a little recall (0.985) because the rotated particle cube exceeds the +-40 mm volume limits.

## 5. Noise and amount of data

| | recall after | ghosts after | calibration error after | 3D error |
|---|---|---|---|---|
| noise 0.15 px (reference, 6 frames) | 0.996 | 4.2 % | 1.21 px | 155 µm |
| noise 0.30 px | 0.989 | 5.0 % | 1.24 | 146 |
| noise 0.60 px | 0.947 | 8.7 % | 0.84 | 167 (jump 16 µm left) |
| 3 frames | 0.996 | 3.9 % | 1.38 | 207 |
| 12 frames | 0.998 | 4.0 % | 0.94 | 103 |

More frames reduce the 3D error steadily (207 -> 155 -> 103 µm for 3, 6, 12 frames): the limit is statistics, not the method.

## 6. The mechanism behind the lv_multi tracking loss, reproduced with truth

Fit data concentrated at the optical axis (a Gaussian cloud, 3600 particles), matching judged on a cloud that fills the whole sensor;
the model's lens terms are 80 % of the truth's. Recall by the largest sensor radius of the particle (mm):

| | 0-2 | 2-3.5 | 3.5-5 | 5-6.5 | 6.5-8 | **8-10** |
|---|---|---|---|---|---|---|
| true cameras | 100 % | 99.5 | 100 | 99.6 | 99.2 | 100 |
| model before the fit | 100 | 99.5 | 100 | 99.6 | 98.9 | 95.7 |
| after, loose lens priors (the old fit) | 100 | 99.5 | 100 | 99.3 | 93.6 | **51.3** |
| after, physical priors (1 px at the corner) | 100 | 99.5 | 100 | 99.6 | 98.3 | **100** |
| after, physical priors (0.25 px) | 100 | 99.5 | 100 | 99.6 | 98.9 | 97.4 |

With loose priors k2 moves by 1.9e-6 (about 14 px at the sensor corner) because the central data fix it and nothing constrains the
edges; with the physical priors it moves by 7e-10. This is the lv_multi failure, now with the cause demonstrated: see
`docs/self-calibration.md` (priors on the lens terms) and `docs/calibration-consistency.md`.

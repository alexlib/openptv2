# Calibration consistency: check it, fix it, and only then patch it

A 3D-PTV run is only as good as the agreement between its cameras. When a particle is seen by all four cameras, four rays meet
in one point. When one camera drops out, the three remaining rays meet in a *slightly different* point if the calibration is
not perfectly consistent. The measured position **jumps** whenever a track changes between "4 cameras" and "3 cameras, camera
*m* missing". Averaged over many particles the jump is a systematic offset, and a velocity (a difference of two positions) does
not cancel it. It is also the cheapest direct measure of calibration quality that a finished run offers, and it needs no
ground truth.

This page describes three layers, in the order to use them.

## 1. Check (always; seconds)

```bash
uv run python -m openptv2.calibration_consistency check <run folder>            # parameters_Run1.yaml, cal/, res/run.zarr
```

```python
from openptv2.calibration_consistency import consistency_report
print(consistency_report(store, frames, cals, cpar, vpar).text())
```

It reports, per camera, how far the blob centres lie from the projection of the 4-camera position, the smooth part of that
(the residual field), **whether that field is real** (a field fitted on the odd frames explains the even frames; noise does
not; "n/a" when there is no second half of frames), and the systematic jump per missing camera in micrometres and in pixel footprints. The verdict:

| verdict | worst jump (px footprint) | meaning |
|---|---|---|
| `inconclusive` | any | the offsets are within 3 standard errors of the noise of this sample (too few points / frames): measure on more |
| `consistent` | < 0.05 | nothing to do |
| `marginal` | 0.05 – 0.30 | small systematic offsets: re-calibrate if you can, otherwise align (layer 3) |
| `inconsistent` | > 0.30 | re-calibrate (layer 2) |

Points whose four cameras disagree (wrong matches: ghosts, merged blobs; rms reprojection residual above max(1.5 px, 3 x median)) are left out of every number and their share is printed; on the cavity data they were 12 % and had inflated a 30 µm jump to 200 µm.

Reference points (400 and 300 frames, then the full 1072): a perfect calibration shows 0.02 px (field reproducibility 0–2 %, i.e. noise); the real
recording wp2 shows 0.14 px (marginal; reproducibility ~50 %); the rendered syn1 with its real calibration error shows 0.47 px
(inconsistent). With fewer than four cameras the check says so and does nothing.

## 2. Re-calibrate from the tracers of the run (the fix)

```bash
uv run python -m openptv2.calibration_consistency selfcal <run folder> [--out new_cal_folder]
```

`self_calibrate_from_store` fits exterior orientation, principal point and `k1 k2 p1 p2` of every camera on the clean
4-camera points of the odd frames and judges the result on the **even frames**. The new cameras are accepted only when
(1) the worst jump on held-out frames falls by at least half, **and** (2) matching those frames again with the new cameras does not
lose 4-camera matches (`min_yield`, ratio >= 1.0; `--min-yield`). Lens terms are fitted with priors set in pixels at the sensor corner (`edge_px`, default 1 px). One camera's exterior is held fixed as the reference (`hold_cam`, default "auto": the camera that already agrees best); a held camera keeps its own errors for good and the others are moved to agree with it, which the consistency numbers cannot show (`docs/self-calibration-cavity-synthetic.md`, section 2). It declines when the cameras were already consistent, and when
the fit does not clearly help (then the inconsistency is not a camera-parameter error).

Why the second test: the jump is measured on points that the *old* cameras matched, and the fit saw the same kind of points,
so the jump can shrink while the calibration gets worse for the particles as a whole. That happened (below). The yield ratio
is the cheapest independent signal we found.

### When self-calibration helped, and when it did not

The first version of the fit **lost** matches and tracks on lv_multi because its priors on the lens terms were meaningless: the
distortion acts on the image radius in mm (k1 ~ r^3, k2 ~ r^5), so a fixed prior number means thousands of pixels at the sensor edge,
the data fix the terms near the centre, and the fit then wrecked the periphery (97 % of the matches beyond 5 mm lost). The priors are
now set in pixels at the sensor corner (`edge_px`, 1 px). With that:

| data | cameras | report | yield with new cameras | two_phase / trackcorr / fast_3d |
|---|---|---|---|---|
| wp2 (300 frames) | one side | marginal | +11 % | tracks >= 50 frames **+11.7 %**; points in tracks >= 20 frames +10.5 % |
| wp1 (300 frames) | one side | marginal | +9 % | tracks >= 50 frames **+15.2 %**; points in tracks >= 20 frames +10.9 % |
| wp1 / wp2, all 1072 frames | one side | marginal | not measured | tracks >= 50 frames +9.3 / +9.8 % with the old priors |
| lv_multi wp5 (20 frames) | one side | marginal | +9 % | good trajectories +1.4 / +5.4 / +4.9 % (were -9.7 / -1.5 / -1.6 %) |
| lv_multi wp4 (20 frames) | one side | marginal | +8 % | -0.5 / +5.1 / +3.8 % (were -7.4 / -1.5 / -3.2 %) |
| rendered syn1 (truth) | one side | inconsistent | see the lab page | velocity error -26 ... -42 % (fast_3d) |
| test_cavity (5 frames) | opposite | marginal, field not reproducible | +2 % | noise-level; blob noise dominates |

Still true, and still the main caveat: the volume changes (`docs/self-calibration-scale-bias.md`); and the matching-yield acceptance
test remains, because a smaller jump among points the old cameras matched is not proof of a better calibration. **Match again and
compare your own tracking before adopting new cameras.** The CLI writes the new cameras to a new folder, so trying costs nothing.

## 3. Align the partial-camera points (stopgap; explicit; marked)

```bash
uv run python -m openptv2.calibration_consistency align <run folder>
```

`align_partial_points` subtracts the smooth 4-camera residual field from the blob centres of the 2- and 3-camera points and
triangulates them again. **4-camera points are not touched at all** (tested: zero changed), so the absolute frame and scale of the
calibration stay exactly as they were. It is never run implicitly. It refuses when the cameras are already consistent, when the
stored points were matched with other calibration files (a re-calibration without new matching), or when it was already applied;
on success the run store gets the attribute `align_partial_points`.

### Where it is useful in practice

Its one real advantage over layer 2 is that **it does not change the gauge**. Self-calibration moves the volume (see
`docs/self-calibration-scale-bias.md`: stretches of 0.2-1 %); alignment cannot. Use it when:

1. **The validated scale and frame must stay.** The calibration was checked against a dumbbell, a plate or a known length, or it is
   shared with other recordings or earlier campaigns, and a change of 0.3 % in scale is not acceptable. You still want the
   camera-switching jump out of tracks that change between 4 and 3 cameras.
2. **You cannot or may not change the cameras.** A certified or published calibration, or a pipeline that takes `.ori` files as
   given, or no time to re-match.
3. **The report says `marginal` and the quantity you care about is sensitive to small systematic offsets between camera subsets**
   (accelerations, small-scale statistics of positions), not just track counts.
4. **A quick, local fix on an already matched store.** It takes seconds, needs no new matching and no new `.ori` files. Run the
   trackers afterwards.

### When to avoid it

| situation | why |
|---|---|
| the report says `consistent` | nothing to correct (it refuses by default) |
| the report says `inconclusive` | the offsets are within the noise of the sample; measure on more frames first |
| the report says `inconsistent` | alignment removes the *jump*, not the camera error: the 4-camera points stay as wrong as they are. Re-calibrate (layer 2): it was better in every measurement |
| you will re-calibrate anyway | after layer 2 and a new matching the report reads `consistent`; alignment would only add risk |
| the legacy `trackcorr` is your tracker | it changed nothing (error doubled: identical output) or little (real error: velocity error 0.574 → 0.552, -4 %) in our tests: it works from the images, not from the stored 3D points |
| few partial-camera points | the gain scales with how many 3-camera points there are |
| most partial points lie far from the 4-camera region | the field is fitted on 4-camera points; far outside them it is extrapolated (on wp2 up to 1.1 mm for the 12 % of the 3-camera points farthest away; unverified) |
| fewer than 4 cameras | not applicable |
| the stored points do not come from the cameras you pass | refused: it would move partial points by the calibration difference only |
| you need to undo it | it rewrites the stored correspondences in place and records a marker; to undo, match the frames again |
| you also use `shared_pass`, `light_check`, `shared_correction` | the combinations were not measured |

### What it did in our tests

| data | tracker | effect |
|---|---|---|
| rendered syn1, real error | fast_3d | common-set velocity error 0.311 → 0.277 (-11 %), position error -12 % |
| rendered syn1, error doubled | fast_3d | 0.405 → 0.423 (**+5 %, worse**); two_phase position error -19 %; trackcorr unchanged |
| rendered syn1, perfect calibration | any | none (it refuses; forced: 2 µm correction) |
| real wp2 (two ranges), wp1 | two_phase | tracks of 50 frames or longer +3.3 %, +2.7 %, 0 % (wp1 had almost no jump in its tracks: 2-4 µm) |

So expect a small gain at best on real recordings, a possible loss when the error is large, and nothing with the legacy tracker.
Treat it as the gauge-preserving alternative to layer 2, not as a substitute.

## What the synthetic cavity showed (truth known, cameras on both sides)

Details and tables: `docs/self-calibration-cavity-synthetic.md`. In short: it works only from a calibration that still matches (image-space
error up to about 1.5 px; beyond about 3 px the matcher finds nothing and the report says "not applicable"); consistency is restored in all
13 experiments (jump 1-4 µm, recall of the true cameras) while the error against the truth stays at 0.1-1.0 px depending on the reference
camera and on what was wrong; glass distance and thickness are never recovered (the cameras absorb them), the fluid index only
partly; the volume keeps its shape to 0.35 %; a rotation of the whole rig by 180 degrees (+z cameras to -z, glass vectors with them)
changes nothing; and the matcher itself assumes the cameras look roughly along lab z (a rig turned 90 degrees finds no matches).

## What the lab found

`bench/calibration_lab` builds a camera rig whose imaging is computed by an independent 3D ray tracer
(`openptv2.synthetic_rig`: Snell's law through air | glass | water, with optional glass tilt, wedge, wrong thickness or
index) and breaks one assumption at a time, with the truth known. See `docs/plans/2026-11-26-calibration-lab.md` for the tables.
In short:

* The library's flat-layer multimedia model agrees with the independent ray tracer to 0.001 px.
* A standard plate calibration absorbs a wrong glass thickness (+1 to +4 mm), a 1 mm glass offset, a 0.3° tilt or wedge and a 0.01
  higher fluid index into the other camera parameters: 3–9 µm of position error and no visible jump inside the plate volume.
  Even four times larger (1.2° wedge, 0.04 index) the error stays at 26–36 µm.
* What does hurt is **camera drift after the plate was recorded** (hundreds of µm of jump from 0.1 mm / 0.3 mrad), and tracer
  self-calibration removes it (jump < 1 µm, leave-one-camera-out 1.8 → 0.008 px).
* Free glass thickness / fluid index in the self-calibration are not identifiable at the level where they matter (fitted n3 moved
  1.41 → 1.416 for a true 1.45) and the effect they would remove is below 40 µm; they are available (`free_multimedia`,
  `free_glass`) and off.
* The blob detector is unbiased (≤ 0.002 px) and its pixel locking is below 0.03 px.

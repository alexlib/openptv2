# Tracks break when the set of cameras changes

**You have:** a finished run where a particle is seen by 4 cameras in one frame and by 3 in the next, and the tracks break
or show small false jumps at those moments.
**You want:** to know whether the cameras are the cause, and to repair it if they are.
**Time:** 15 minutes. **Tested:** every command below runs in CI on `test_data/test_cavity` (five frames, full images).

Course pages [5](../course/05-match.md) to [8](../course/08-calibration.md) explain the words used here.

## 1. What happens, in two sentences

With four cameras, four rays meet in one point. When one camera drops out, the other three rays meet in a *slightly different*
point if the calibration is not perfect. The track then gets a false step of that size (the *jump*).

## 2. Prepare a run

```bash
cp -r test_data/test_cavity fix_run
```

```bash
openptv2-batch --workdir fix_run --first 10000 --last 10004 --mode both
```

```text
Average over sequence, particles: 1665.2, links: 357.8, lost: 1307.5
Sealed: {'n_trajectories': 153, 'n_rows': 765, 'n_dropped': 0}
Batch processing completed successfully
```

## 3. Measure the jump

```bash
python -m openptv2.calibration_consistency check fix_run --first 10000 --last 10004
```

```text
calibration consistency: MARGINAL  (2170 four-camera points in 5 frames, 1 px = 95 um; 12 % left out as wrong matches)
  camera   residual median/p90 (px)   smooth field (px)   field reproducible   jump when missing (um): systematic +- s.e. / rms
       1      0.753 / 1.879             0.300                4 %       19.9 +- 5.0 / 217.4
       2      0.733 / 1.758             0.254                0 %        5.6 +- 5.3 / 233.7
       3      0.618 / 1.639             0.234                0 %       26.7 +- 5.8 / 251.7
       4      0.651 / 1.619             0.158                0 %        1.2 +- 4.8 / 209.5
  worst systematic jump 0.28 px footprint -> small systematic offsets between camera subsets; align_partial_points() can remove them, or re-calibrate
```

Rule: below 0.05 px, stop here, the cameras are not your problem. Between 0.05 and 0.30 px (this example), you have two
repairs, below. Above 0.30 px, calibrate again with better images (the first choice). Read
[page 8](../course/08-calibration.md) for what each column means.

## 4. Repair A: fit new cameras to the particles (self-calibration)

The program fits every camera again, using the particles of your own run. It tests the result on frames it did not use.

```bash
python -m openptv2.calibration_consistency selfcal fix_run --first 10000 --last 10004
```

```text
self-calibration ACCEPTED: the camera-subset jump on held-out frames fell; the scale moved by -0.17 % and no anchor was given: check it against a known length
  SCALE NOT ANCHORED: tracers cannot define the scale. Volume scale vs the old cameras: 0.99834 (principal stretches 0.99492, 0.99979, 1.00032). A uniform scale error s scales every velocity and length by s and leaves strain rates and vorticity unchanged; unequal stretches do change them. See docs/self-calibration-scale-bias.md.
  fit: 781 points, reprojection rms 0.737 -> 0.728 px
  matching yield with the new cameras (held-out frames): 101.6 % of the old
  reference camera held fixed in the fit: 3 (its own errors cannot be corrected; camera with the smallest residual by default)
  held-out frames: worst jump 0.32 -> 0.11 px footprint (30.5 -> 10.8 um)
  volume vs old cameras: scale 0.99834, mean shift 0.343 mm
new cameras written to fix_run/cal_selfcal (4 files); the originals are untouched. Point the yaml's cal_ori at them and match the frames again before tracking (the stored points belong to the old cameras).
```

How to read it (the program says "4 files" and means 4 cameras: each has an `.ori` and an `.addpar`, 8 files):

* **ACCEPTED** means two tests passed: the jump on held-out frames fell, and the matching did not lose points (yield 101.6 %,
  must be at least 100 %). If it says *declined*, keep your old cameras.
* **SCALE NOT ANCHORED** is the part people miss. Particles cannot tell the program how big the volume is. The fit moved the scale
  by -0.17 %. If you publish absolute numbers (velocities in mm/s), check the scale against a known length, for example a
  [dumbbell](../tutorials/dumbbell_calibration.md). Background: [Self-calibration changes the volume](../self-calibration-scale-bias.md).

Use the new cameras: copy them over the old files (the new files are named `cam_1.tif.ori`, the run folder uses `cam1.tif.ori`),
and process the frames again, because the stored points belong to the old cameras.

```bash
cp -r fix_run fix_new
```

```bash
cp fix_run/cal_selfcal/cam_1.tif.ori fix_new/cal/cam1.tif.ori
cp fix_run/cal_selfcal/cam_2.tif.ori fix_new/cal/cam2.tif.ori
cp fix_run/cal_selfcal/cam_3.tif.ori fix_new/cal/cam3.tif.ori
cp fix_run/cal_selfcal/cam_4.tif.ori fix_new/cal/cam4.tif.ori
cp fix_run/cal_selfcal/cam_1.tif.addpar fix_new/cal/cam1.tif.addpar
cp fix_run/cal_selfcal/cam_2.tif.addpar fix_new/cal/cam2.tif.addpar
cp fix_run/cal_selfcal/cam_3.tif.addpar fix_new/cal/cam3.tif.addpar
cp fix_run/cal_selfcal/cam_4.tif.addpar fix_new/cal/cam4.tif.addpar
```

```bash
openptv2-batch --workdir fix_new --first 10000 --last 10004 --mode both
```

```text
Average over sequence, particles: 1669.4, links: 357.0, lost: 1312.4
Sealed: {'n_trajectories': 151, 'n_rows': 755, 'n_dropped': 0}
Batch processing completed successfully
```

```bash
python -m openptv2.calibration_consistency check fix_new --first 10000 --last 10004
```

```text
calibration consistency: MARGINAL  (2199 four-camera points in 5 frames, 1 px = 95 um; 13 % left out as wrong matches)
  camera   residual median/p90 (px)   smooth field (px)   field reproducible   jump when missing (um): systematic +- s.e. / rms
       1      0.730 / 1.802             0.278                6 %        9.5 +- 4.6 / 202.8
       2      0.697 / 1.698             0.224                3 %       18.6 +- 5.3 / 232.9
       3      0.606 / 1.587             0.216                3 %       18.1 +- 5.6 / 246.2
       4      0.630 / 1.609             0.169                0 %       10.9 +- 4.4 / 194.7
  worst systematic jump 0.20 px footprint -> small systematic offsets between camera subsets; align_partial_points() can remove them, or re-calibrate
```

## 5. What you should see, and the honest result

On this example, the worst jump fell from 0.28 to 0.20 px, and the tracks did **not** improve: 153 trajectories before, 151 after.
(The selfcal report says 0.32 to 0.11 px: it measures on the frames it did not use. The 0.28 and 0.20 are from the check on all
frames. They differ, and both are right.) Look at the table again: the jump of cameras 1 and 3 fell (20 → 9.5 and 27 → 18 µm), but
that of cameras 2 and 4 **rose** (5.6 → 18.6 and 1.2 → 10.9 µm). The fit moved the error around more than it removed it.
Five frames is too short for self-calibration to give a gain, and 781 points is a small fit. The same method gave 5 to 15 % more
long tracks on recordings of 300 frames or more ([measurements](../self-calibration-scale-bias.md)), but it hurt on one rig before
we fixed the lens priors. So the rule is:

> Never adopt new cameras from the report alone. Process again, and compare the **tracks** of your own data.

Adopt the new cameras only if (1) the report says ACCEPTED, (2) you checked the scale against a known length when it matters,
and (3) the tracks of your frames are at least as many and as long as before.

## 6. Repair B: align only the points that miss a camera

This repair keeps your calibration, scale and frame unchanged. It moves only the 3-camera and 2-camera points to where the
4-camera solution would put them. It does not touch the 4-camera points.

```bash
cp -r fix_run fix_align
```

```bash
python -m openptv2.calibration_consistency align fix_align --first 10000 --last 10004
```

```text
align_partial_points: applied to 6156 partial-camera points: recorded in the store attributes
  partial-camera points moved by 117.2 um (median), 1707.6 um (99 %); 4-camera points changed: 0
```

The stored points changed, so track again (the matching is kept):

```bash
openptv2-batch --workdir fix_align --first 10000 --last 10004 --mode tracking
```

```text
Average over sequence, particles: 1665.2, links: 358.8, lost: 1306.5
Sealed: {'n_trajectories': 152, 'n_rows': 760, 'n_dropped': 0}
Batch processing completed successfully
```

Again no gain on five short frames (152 against 153). Align is the safe choice when you must keep the validated scale. Do not use it
for an *inconsistent* verdict (above 0.30 px): there the cameras are wrong, and aligning only hides it.

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| `too few to judge` | fewer than 300 four-camera points | use more frames, or a denser run |
| self-calibration declined | the cameras were already consistent, or the fit did not help | keep your cameras |
| ACCEPTED, but fewer tracks after re-matching | the fit helped the points it saw and hurt the rest | keep the old cameras |
| Verdict above 0.30 px even after the repair | the cameras are really wrong | calibrate again: [plates, two rigs](plate-calibration-two-rigs.md) or [classic calibration](../tutorials/calibration.md) |

## Next

[Self-calibration: how it works](../self-calibration.md), [the three layers in full](../calibration-consistency.md).

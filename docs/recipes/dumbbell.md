# Dumbbell: fix or check the scale with a rigid bar

**You have:** a calibration that is a little off, and a *dumbbell*: two markers on a rigid bar, with a known distance between them,
that you moved through the measurement volume while recording.
**You want:** (a) better cameras, or (b) a check that your lengths are right.
**Time:** 15 minutes. **Tested:** the synthetic example in sections 2 to 4 runs in CI. Section 6 (your own recording) cannot,
because it needs your images.

## 1. Why a dumbbell

A plate can only sit where it fits. The dumbbell goes through the whole volume, and its length does not change. The program
moves the cameras a little until, in every frame, the rays of each marker meet **and** the two markers are the right distance
apart. The length is the only thing in the volume that sets the **scale**.
([Particles alone cannot set the scale](../self-calibration-scale-bias.md).)

## 2. A synthetic example, with the truth known

You do not need a recording to learn this. The script below takes the cameras of `test_data/test_cavity` as the **truth**,
renders a 46 mm bar at 40 random places into the four cameras (with 0.1 px of noise), then moves cameras 2, 3 and 4 away from the
truth by a few millimetres and tries to find their way back. Camera 1 is the reference and does not move.

```bash
python scripts/synthetic_dumbbell.py dumbbell_demo
```

```text
frames used 40 of 40; true bar 46.0 mm, told to the program 46.0 mm
camera   position error (mm)   angle error (deg)
  2       3.40 ->  0.13        0.203 -> 0.013
  3       3.20 ->  0.05        0.336 -> 0.004
  4       2.61 ->  0.17        0.547 -> 0.021
bar length measured with the cameras: 45.76 -> 45.99 mm
```

Read it like this: before the fit the cameras were 2.6 to 3.4 mm from the truth, after it 0.05 to 0.17 mm. The bar, measured with the cameras
you have, went from 45.76 mm to 45.99 mm. **This last number is your check.** On real data you do not know the camera errors,
but you do know the bar.

## 3. What a wrong bar length does

Run the same case, but tell the program that the bar is 47 mm. The true bar is still 46 mm.

```bash
python scripts/synthetic_dumbbell.py dumbbell_demo_wrong --claimed-length 47
```

```text
frames used 40 of 40; true bar 46.0 mm, told to the program 47.0 mm
camera   position error (mm)   angle error (deg)
  2       3.40 ->  5.19        0.203 -> 0.017
  3       3.20 -> 22.35        0.336 -> 0.002
  4       2.61 -> 21.48        0.547 -> 0.013
bar length measured with the cameras: 45.76 -> 47.00 mm
```

The cameras are now worse than before (up to 22 mm off). They are *consistent*, the bar measures 47.00 mm, and the fit looks perfect.
Only the wrong input made them wrong. **Measure the bar with a caliper, and write down the uncertainty.** A length error of 2 %
(1 mm of 46) moved the cameras by up to 22 mm here.

## 4. The program's own command, on the example

The command line tool works on any folder that has a `dumbbell` section in its YAML file. The example folder has one:

```bash
python skills/openptv-dumbbell/scripts/dumbbell.py check dumbbell_demo
```

```text
OK: dumbbell.dumbbell_scale = 46.0  (known dumbbell length (mm))
OK: dumbbell.dumbbell_penalty_weight = 1.0  (weight of length vs ray error)
OK: dumbbell.dumbbell_eps = 0.0  (max length deviation to keep a frame (0 = keep all))
OK: dumbbell.dumbbell_step = 1  (frame stride through sequence)
```

The calibration is the same function that the script used. Without `--dry-run` it writes the new cameras and keeps the old ones
(`*.dbbak`):

```bash
python skills/openptv-dumbbell/scripts/dumbbell.py run dumbbell_demo --dry-run --fixed-cams 0
```

It prints the optimizer's progress, then the frames used (`Frames used: 40 / 40`), the cost before and after (sum of squared
residuals; it must fall), and says whether the optimizer converged.

## 5. It can fail, and the bar length shows it

In our tests (40 frames, 0.1 px noise, a start about 3 mm off) the fit came back to within 1 mm of the truth in **6 of 10 random
cases with 40 frames, and 8 of 10 with 20 frames**. In the failed cases the optimizer stopped in a wrong place, up to 18 mm from
the truth, and the bar measured 46.4 to 46.8 mm instead of 46.0. More iterations did not help. So:

* Always measure the bar with the new cameras (section 2). If it is not the real length, **do not use the cameras**.
* Start from a good calibration (a few mm off, not tens of mm), and hold the cameras you trust with `--fixed-cams`.
* Try again with another subset of frames (`--step`) if the check fails.

## 6. With your own recording

You need:

```text
<dataset>/
  parameters_<name>.yaml     with a `dumbbell` section (below)
  cal/camN.tif.ori           the existing calibration, as the first guess
  cal/camN.tif.addpar
  img/                       the dumbbell image sequence, already detected (exactly 2 targets per camera and frame)
```

```yaml
dumbbell:
  dumbbell_scale: 46.0          # the real distance between the two points, in mm: measure it carefully
  dumbbell_penalty_weight: 1.0  # 1.0 balances ray convergence against the length
  dumbbell_eps: 0               # 0 = use all frames; above 0 drops frames whose length is off by more than this (mm)
  dumbbell_step: 1              # use every Nth frame (5 = every fifth)
```

Frames in which a camera does not find exactly two targets are skipped. Then: `check`, then `run --dry-run`, then `run`, then
**measure the bar again** and do [page 8](../course/08-calibration.md) on a tracked run.

Only the camera positions and angles are fitted. The lens terms stay as they were.

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| `No valid frames found` | no frame has exactly 2 targets in every camera | detect again, or lower the threshold |
| the cost does not fall | wrong `dumbbell_scale`, or the first guess is too far off | measure the bar again; calibrate with a plate first |
| the cost falls, but the bar is not the real length | the fit stopped in a wrong place | section 5 |
| very few frames used | `dumbbell_eps` too strict | set `dumbbell_eps: 0` |

## Next

[Tracks break when the set of cameras changes](camera-set-changes.md), [Dumbbell tutorial (reference)](../tutorials/dumbbell_calibration.md).

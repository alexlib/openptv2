# Dumbbell: fix or check the scale with a rigid bar

**You have:** a calibration (from a plate, see [the classic calibration](calibration-known-target.md)) and a *dumbbell*: two
markers on a rigid bar, with a known distance between them, that you moved through the measurement volume while recording.
**You want:** either (a) to improve the cameras, or (b) to check that your lengths are right.
**Time:** 20 minutes. **Not tested in CI:** it needs your dumbbell images. Commands and output are from the dumbbell tool's own tutorial.

## 1. Why a dumbbell

A plate can only sit where it fits. The dumbbell goes through the whole volume, and its length does not change. Two uses:

* **Calibrate:** the cameras are moved a little until the two points of the bar are the right distance apart in every frame,
  and the rays of every point meet.
* **Check the scale:** measure the bar length with the cameras you will use. If you measure 46.0 mm for a 46.0 mm bar, the
  scale is right. This is the length check that [self-calibration](camera-set-changes.md) needs when it changes the volume
  (particles alone cannot tell the scale, see [the scale bias](../self-calibration-scale-bias.md)).

## 2. What you need

```text
<dataset>/
  parameters_<name>.yaml     must have a `dumbbell` section
  cal/camN.tif.ori           the existing calibration, as the first guess
  cal/camN.tif.addpar
  img/                       the dumbbell image sequence
```

In the YAML file:

```yaml
dumbbell:
  dumbbell_scale: 46.0          # the real distance between the two points, in mm: measure it carefully
  dumbbell_penalty_weight: 1.0  # 1.0 balances ray convergence against the length
  dumbbell_eps: 0               # 0 = use all frames; above 0 drops frames whose length is off
  dumbbell_step: 1              # use every Nth frame (5 = every fifth)
```

`dumbbell_scale` is the number that decides everything. A wrong value gives wrong cameras, with a good-looking result.

## 3. Do it

Set a short name for the tool, then check, run without writing, and then write:

```bash
DB=skills/openptv-dumbbell/scripts/dumbbell.py
uv run python $DB check my_dumbbell_run
uv run python $DB run my_dumbbell_run --dry-run
```

Every line of `check` must say `OK`. The dry run prints:

```text
Result:
  Frames used:       42 / 100
  RMS before (px):   2.3410
  RMS after  (px):   1.1852
  Improvement:       +1.1558 px
  (dry-run: no files written)
```

(An example from the tool's tutorial, not from the bundled data.) Good: the RMS falls, and a reasonable share of the frames is used.
Then write the cameras (the old ones are kept as `*.dbbak`):

```bash
uv run python $DB run my_dumbbell_run
```

## 4. Check it worked

1. The RMS fell. A rise means the optimizer went the wrong way (see below).
2. Do [page 8](../course/08-calibration.md) on a tracked run with the new cameras. The jump should be smaller.
3. Measure the bar length again with the new cameras. It must be the real length, to the precision you need (1 % of 46 mm is 0.46 mm).

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| the RMS does not fall | wrong `dumbbell_scale`, or the first guess is too far off | measure the bar again; calibrate with a plate first |
| very few frames used | `dumbbell_eps` too strict | set `dumbbell_eps: 0`, or use `dumbbell_step: 1` |
| the RMS rises | swapped refractive indices, or the optimizer diverged | check `mmp_n2` and `mmp_n3`; run with `--maxiter 500`; hold the good cameras with `--fixed-cams 0,2` |

More options: `--step 5` (long sequences), `--fixed-cams`, `--maxiter`.

## Next

[Tracks break when the set of cameras changes](camera-set-changes.md), [Dumbbell tutorial (reference)](../tutorials/dumbbell_calibration.md).

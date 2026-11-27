# Cameras on both sides of a water cell

**You have:** a tank or cell with flat windows, and cameras on two opposite sides (two on each side in our example).
**You want:** tracks through the whole cell.
**Time:** 20 minutes. **Tested:** every command below runs in CI on `test_data/test_cavity` (five frames).

Course pages [3](../course/03-cameras.md) (cameras and the glass) and [8](../course/08-calibration.md) (is the calibration good?)
explain the words. This page shows what is special when the cameras do not all look from the same side.

## 1. What is different

* Light goes air → glass → water. Each camera has its **own** window, and its glass vector points to its own window. Cameras on
  opposite sides have glass vectors of opposite sign.
* Opposite cameras look at the particles from opposite directions. A small error in the water index or the window distance moves
  the particle in **depth** and the two sides move it in opposite ways. This makes the cameras disagree, and makes the check on
  page 8 more sensitive than for cameras on one side.
* The program models flat, parallel windows exactly. A tilted window is not modelled
  ([the ray-tracing page](../ray-tracing-vs-radial-shift.md) shows what that costs).

## 2. Look at your cameras

```bash
cp -r test_data/test_cavity water_run
```

```bash
python -c "
for i in range(1, 5):
    t = open(f'water_run/cal/cam{i}.tif.ori').read().split()
    print(f'camera {i}: z = {float(t[2]):8.1f} mm   glass vector z = {float(t[-1]):8.1f} mm')
"
```

```text
camera 1: z =   -553.6 mm   glass vector z =   -205.2 mm
camera 2: z =   -607.0 mm   glass vector z =   -204.2 mm
camera 3: z =    588.8 mm   glass vector z =    256.0 mm
camera 4: z =    536.7 mm   glass vector z =    138.3 mm
```

Rule: the glass vector has the sign of the side its camera is on (negative z with negative z, positive with positive). The sizes
differ between cameras, because each is measured from the camera's own origin. A wrong sign is the first thing to look for when
almost nothing matches.

The materials are in the YAML file, section `ptv`: `mmp_n1` (air, 1.0), `mmp_n2` (glass, 1.46 here), `mmp_n3` (water, 1.33)
and `mmp_d` (glass thickness in mm, 6.0). Measure the thickness with a caliper, and use the water index for your temperature and
salt (for example 1.333 for fresh water at 20 °C).

## 3. Run, and check the cameras against each other

```bash
openptv2-batch --workdir water_run --first 10000 --last 10004 --mode both
```

```text
Average over sequence, particles: 1665.2, links: 357.8, lost: 1307.5
Sealed: {'n_trajectories': 153, 'n_rows': 765, 'n_dropped': 0}
Batch processing completed successfully
```

```bash
python -m openptv2.calibration_consistency check water_run --first 10000 --last 10004
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

This is the same run as page 8. Cameras 1 and 3 jump most (20 and 27 µm). Next, [Tracks break when the set of cameras changes](camera-set-changes.md)
shows the two repairs.

## 4. What goes wrong in a cell, and what it looks like

We measured these on a synthetic cell with the truth known
([Synthetic cavity study](../self-calibration-cavity-synthetic.md)):

| The truth differs from your model by | What you see | Remedy |
|---|---|---|
| a small camera drift, lens terms, window distance ±1 mm, window 2 mm thicker, water index +0.005, window tilted 0.3° | tracks still work; the jump is visible on the check | self-calibration restores consistency (jump 1 to 4 µm) |
| water index +0.02, or a large drift (≥ 3.7 px), or lens terms ignored | no correct matches at all; the check says "not applicable" | calibrate again with a plate; self-calibration cannot start from this |

Two warnings from the same study:

* Self-calibration makes the cameras **agree**. It does not prove they are **right**. A water index error of 0.005 gives a depth
  stretch of 0.27 % that the fit leaves in place. Check the size of the cell against a known length.
* In the study the mean error against the truth was 0.65 px before and 0.41 to 0.52 px after the fit, but the worst case could
  be worse than before. Compare tracks, not only the jump.

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| almost no 4-camera points | a sign of a glass vector is wrong, or the water index is wrong | check section 2, then `mmp_n3` |
| good on one side, bad on the other | the two sides do not share one coordinate frame | calibrate all four cameras together, with one target seen by all |
| the check says "not applicable" | fewer than four cameras see enough points | use more frames or more particles |

## Next

[Tracks break when the set of cameras changes](camera-set-changes.md), [Synthetic cavity study](../self-calibration-cavity-synthetic.md).

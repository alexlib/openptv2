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
not), and the systematic jump per missing camera in micrometres and in pixel footprints. The verdict:

| verdict | worst jump (px footprint) | meaning |
|---|---|---|
| `consistent` | < 0.05 | nothing to do |
| `marginal` | 0.05 – 0.30 | small systematic offsets: re-calibrate if you can, otherwise align (layer 3) |
| `inconsistent` | > 0.30 | re-calibrate (layer 2) |

Reference points (400 and 300 frames): a perfect calibration shows 0.02 px (field reproducibility 0–2 %, i.e. noise); the real
recording wp2 shows 0.14 px (marginal; reproducibility ~50 %); the rendered syn1 with its real calibration error shows 0.47 px
(inconsistent). With fewer than four cameras the check says so and does nothing.

## 2. Re-calibrate from the tracers of the run (the fix)

```bash
uv run python -m openptv2.calibration_consistency selfcal <run folder> [--out new_cal_folder]
```

`self_calibrate_from_store` fits exterior orientation, principal point and `k1 k2 p1 p2` of every camera on the clean
4-camera points of the odd frames and judges the result on the **even frames**: the new cameras are accepted only when the
worst jump on held-out frames falls by at least half. It declines when the cameras were already consistent, and when the fit
does not clearly help (then the inconsistency is not a camera-parameter error). The original `.ori` files are never touched;
the new ones go to a new folder. **Match the frames again with the new cameras before tracking**: the stored points were
triangulated with the old ones.

Tracers fix the consistency between cameras, **not the absolute frame or scale**. The result reports the scale and the shift of the
volume against the old cameras; check the scale against a known length (dumbbell, plate) before you trust new absolute
velocities, or pass a `KnownBody` / `Dumbbell` (see `docs/self-calibration.md`) to anchor it. A coarse-to-fine robust loss is
built in, so a large initial inconsistency (several pixels) converges too.

## 3. Align the partial-camera points (stopgap; explicit; marked)

```bash
uv run python -m openptv2.calibration_consistency align <run folder>
```

Only when the cameras cannot be changed. `align_partial_points` subtracts the smooth 4-camera residual field from the blob
centres of the 2- and 3-camera points and triangulates them again; 4-camera points are not touched. It is **never run
implicitly**. It refuses when the cameras are already consistent, when the stored points were matched with other calibration
files (a re-calibration without new matching), or when it was already applied; on success the run store gets the attribute
`align_partial_points`. It removes the *jump*, not the camera error itself: layer 2 does both, and was better in every
measurement below.

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

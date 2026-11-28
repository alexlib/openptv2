# Camera-subset alignment (`track.align_4cam`, opt-in)

!!! abstract "In short"
    With a calibration that is a little off, a particle seen by 3 cameras is placed a little differently from the same particle seen by 4. Along a track that shows up as a sideways jump every time the set of cameras changes, and as noise in velocity and acceleration. `track.align_4cam: true` removes that jump. It does not touch the camera files, does not touch any 4-camera position, and is **off by default**.

## The problem in plain words

A real calibration is never exact: the four lines of sight to one particle miss each other by a fraction of a pixel (0.2 to 0.4 px on our real data).

* While a particle is seen by all four cameras, that miss is almost the same from frame to frame, so it cancels when you subtract two positions to get a velocity.
* When one camera drops out (a 3-camera point; on `test_cavity` three quarters of the points), the three remaining lines meet at a slightly different place. The position jumps, by about the size of the calibration error, and the jump does not cancel.

## What it does

1. It takes the clean 4-camera points (no shared blob) and measures, per camera, what is left over between the blob centre and where the 4-camera position projects. That leftover is the calibration error, and it varies smoothly in space.
2. It fits a smooth function of position to it (a cubic polynomial, per camera, robust to a few bad points).
3. For every point seen by 2 or 3 cameras it subtracts that function from the blob centres and triangulates again.

4-camera points are not changed: by the least-squares equations of triangulation, subtracting this field leaves them where they are.

It changes **consistency**, not absolute position. The 4-camera positions keep whatever error the calibration has. What goes away is the difference between the 3-camera and the 4-camera position of the same particle. For velocities, that difference is what matters. If you need the camera files themselves improved, use [self-calibration](self-calibration.md).

## How to use it

```yaml
track:
  align_4cam: true     # any tracker; default false
```

or `--align-4cam` on `openptv2-batch`, or **Tracking Parameters → Pipeline → Align 4-camera subsets** in the GUI. It runs once before the tracker and needs at least 100 clean 4-camera points (otherwise it does nothing).

### Where the result is stored

The measured positions in `correspondences/` are **never overwritten**. The aligned positions go to a separate array, `correspondences_aligned/`, and `RunStore.read_correspondences(frame)` returns them in place of the measured ones. `read_correspondences(frame, raw=True)` returns the measured ones, and `RunStore.clear_aligned_positions()` removes the aligned ones.

* Running it twice gives the same result: the fit always starts from the measured positions.
* A run with `align_4cam: false` removes aligned positions left by an earlier run, so it uses the measured ones.
* A step that rewrites a frame's correspondences (for example a ghost filter that removes rows) drops that frame's aligned copy.
* The explicit repair `python -m openptv2.calibration_consistency align` ([recipe](recipes/camera-set-changes.md)) is a different thing: it rewrites the stored positions in place and records that in the store. It is unchanged. A run with `align_4cam: false` does not undo it.
* `trackcorr` and `fast_3d` write each frame back as they always did, with or without this option. With the option on, they write the aligned values for the frames they change.

## When it helps, and when not

The gain depends on how smooth and how large the calibration error is. We measured it as the leave-one-camera-out error on frames the fit did not see (fit on every other frame):

| Data | Median error before → after |
|---|---|
| `CompleteTest` wp2, real, 100 frames | 0.350 px → 0.201 px (−42.5 %) |
| `lv_multi` wp5, real, 20 frames | 0.400 px → 0.262 px (−34.5 %); the worst 10 % improve by only about 10 % |
| `test_cavity`, real, 5 frames | 1.307 px → 1.213 px (−7 %); the worst 10 % get slightly worse |

On `test_cavity` the default run gives 152 trajectories with the option, 153 without. Try it on your data and compare the two.

### Truth test

`scripts/synthetic_align4cam.py` (run by `tests/unit/test_align4cam_truth.py`) projects points with known positions through the true cameras, then processes them with spoiled cameras. Example result (RMS in mm; `jump` is the difference between the 3-camera and 4-camera position of the same particle):

| Case | jump before → after |
|---|---|
| no calibration error | 0.020 → 0.020 (no change, no harm) |
| calibration error | 0.995 → 0.047 |
| three times larger | 2.999 → 0.157 |
| noise 0.3 px | 0.995 → 0.073 |
| 5 % gross errors among the fit points | 0.985 → 0.049 |
| only 150 fit points | 0.991 → 0.050 |
| fitted on the inner 40 % of the volume, evaluated outside | 1.002 → 0.052 |

The 3-camera error against the truth falls to the level of the 4-camera error, and not below it.

## Limits

* It only moves points seen by fewer than all cameras. With 4 cameras and 4-camera points everywhere it does nothing.
* Three-camera points move by the calibration error, which can be large along the depth direction when the three cameras are on one side. On `test_cavity`, the median move is 0.12 mm and the largest 3.5 mm.
* The field is smooth (cubic). A calibration error with sharp structure is only partly removed.
* Absolute accuracy is not improved.

## Related, measured elsewhere

Two other changes were proposed together with this one in the pull request that introduced it, and are **not** part of the default behaviour: `refine_merged(prior=False)` (no quadratic path prior), and `log_brightness_from_arrays(mask_shared=True)` (the ghost filter ignores shared blobs when comparing brightness). Both exist as options. The synthetic benchmark numbers of that pull request (velocity error −2 % to −21 %) were measured with all three on, and have not been re-run separately.

# Two rigs of four cameras, calibrated with a hand-held plate

**You have:** two groups of four cameras (rig A: cameras 1 to 4, rig B: cameras 5 to 8), and a flat plate with a printed dot
pattern that you held by hand at many positions in front of each group.
**You want:** a calibration file for every camera, and then tracks from both rigs, merged into one set.
**Time:** a few hours for the first rig, much less for the second. Most of it is waiting for the programs.

!!! note "This recipe is not run by the automatic tests"
    The course pages are tested on bundled data. This recipe needs *your* plate images, so CI cannot run it. The steps and
    commands come from a working calibration (see the [worked example](../plate-calibration-worked-example.md) for its numbers,
    including the failures), and the library parts are covered by unit tests (`test_plate_labeler_anchor.py`,
    `test_plate_bundle.py`, `test_plate_detection.py`, `test_calibration_import.py`).

## How it differs from the classic calibration

The classic calibration (page [3](../course/03-cameras.md) and
[Camera calibration](../tutorials/calibration.md)) moves the plate by known steps. Here the plate is at **unknown** positions
and angles. The program finds each plate position together with the cameras. This is the OpenCV (Zhang) way. If you know the plate
positions, the classic way is simpler.

## What you need

1. **Images.** One folder per camera, named `plate_cam1`, `plate_cam2`, and so on. The frames are numbered the same in all
   cameras of a rig. A file is `<frame>_<anything>.tiff`: the frame number is everything before the first underscore. About
   40 plate positions per rig is a good number.
2. **A plate you know.** A flat plate with dots on a regular grid and one **coded corner** (an L-shaped group of marked dots)
   that tells the program which dot is which. You need the number of dots in x and y, the **dot spacing in mm (measure it, do
   not trust the drawing)**, and which grid dot is the coded corner.
3. **A frame convention.** Decide, and write down, where x, y and z point in your laboratory. For example: x to the right,
   y up, z from the object to the cameras, right-handed. A wrong convention does not show in any residual.
4. **The sensor.** Image size and pixel size. They are set at the top of `scripts/plate_rigs/_config.py` (`IMX`, `IMY`, `PIX`).
   Edit them for your camera.

Put a `plate.yaml` in the working folder of each rig:

```yaml
plate:
  nx: 6                 # dots in x
  ny: 7                 # dots in y
  pitch_x: 120.0        # spacing in mm, measured
  origin_frame: "00000000"   # the frame that defines the world (the "reference frame")
  datum: {ix: 2, iy: 3} # grid index of the coded corner dot
```

## Step 0: set up

Choose where the rig's data is (the scripts read these environment variables):

```bash
export PLATE_RIG_RAW="/path/to/folder/holding/plate_cam1..plate_cam8"
export PLATE_RIG_DIR="$PLATE_RIG_RAW/openptv_rig_a_4cam"   # the working folder of rig A
export PLATE_RIG_CAMS=1,2,3,4                              # the camera numbers of rig A
export OPTV="/path/to/openptv2"                            # the repository
```

The program checks that `PLATE_RIG_CAMS` agrees with the camera names in the working folder's `parameters_Run1.yaml` and
**refuses to run if not**. This protects rig A's calibration from being overwritten with data from rig B.

OpenCV is not a dependency of openptv2. The two scripts that need it run with `uv run --with opencv-python-headless`.

## Step 1: find and label the plate dots in every image

```bash
cd "$PLATE_RIG_DIR"
uv run --project $OPTV python $OPTV/scripts/plate_rigs/detect_plate_frames.py
```

This detects the dots and gives each one its grid number, in every image. It takes a few minutes and saves the result in
`cal/labelled_all_frames.npz`. The labelling uses an existing `.ori` file to know which way is "up" in the image. For a brand-new
rig with no `.ori` yet, run it once with `PLATE_RIG_NO_HINT=1`, and again after step 3.

## Step 2: find the focal length (`cc`)

```bash
uv run --project $OPTV --with opencv-python-headless python $OPTV/scripts/plate_rigs/fit_plate_cc.py
```

One flat plate cannot tell you the focal length. Many plate positions can: the right value makes all cameras agree about the
*other* plate positions. The script prints a table: for each trial `cc`, how much the cameras disagree about the plate centre. Take the `cc` with the smallest disagreement (in the worked example: 8.5858 mm). Use it in the next steps.

## Step 3: write the camera files

```bash
uv run --project $OPTV --with opencv-python-headless python $OPTV/scripts/plate_rigs/refit_plate_pinhole.py 8.5858
```

This writes `cal/cam<N>.tif.ori` and `.addpar` for each camera, as a pure pinhole with **no lens distortion**. That is on
purpose: distortion fitted from a single plane is not physical (see "Traps" below).

## Step 4: check the result (the gate)

```bash
uv run --project $OPTV python $OPTV/scripts/plate_rigs/check_plate_triangulation.py
uv run --project $OPTV python $OPTV/scripts/plate_rigs/check_epipolar.py
```

Never decide on the reprojection error. It stayed at about 0.5 px even for calibrations that were wrong. Decide on these two:

| Check | What it tells you | Good (from the worked example) |
|---|---|---|
| `check_plate_triangulation.py` | the triangulated plate is flat, has the right dot spacing and the right place | flat to 0.3 mm, spacing within 1 %, position within about 1 mm |
| `check_epipolar.py` | the epipolar line of a dot passes through the same dot in the other cameras | 0.1 to 0.3 px median, under 1 px worst case |

## Step 5: improve with a joint bundle

```bash
uv run --project $OPTV --with opencv-python-headless python $OPTV/scripts/plate_rigs/bundle_plate_poses.py 8.5858 --write
```

Step 3 is exact on the reference frame and gets worse away from it. The bundle solves all cameras and all plate positions
together and spreads the error over the whole volume. Run it without `--write` first to see what it would change. Then run
step 4 again. The result should be the same or better on both checks.

After the bundle, run step 1 once more: the labeller uses the new `.ori`, and may now label frames it got wrong before.

## Step 6: set the observation volume

```bash
uv run --project $OPTV python $OPTV/scripts/plate_rigs/check_epipolar_volume.py
```

The depth limit of the measurement volume (`Zmax_lay` in the YAML) must stay in front of the cameras. If it reaches the plane of a
camera, the epipolar lines are thrown across the image and look wrong, although the calibration is fine. The script reports the
largest safe value for your rig. This is the most common reason for "the epipolar lines are wrong".

## Step 7: rig B

Repeat steps 1 to 6 with rig B's working folder and cameras:

```bash
export PLATE_RIG_DIR="$PLATE_RIG_RAW/openptv_rig_b_4cam"
export PLATE_RIG_CAMS=5,6,7,8
```

Each rig is its own world, with its own calibration. If the rigs see different faces of the plate, they cannot be tied by
matching dots. Tie them later through the laboratory frame, or through particles that both rigs see (next section).

## After the calibration

1. **Check the calibration with particles:** [course page 8](../course/08-calibration.md).
2. **Track each rig:** [course pages 4 to 7](../course/04-detect.md), once for each rig's folder.
3. **Merge the two rigs:** [Two-rig processing chain](../dual-rig-processing-chain.md). It finds the rigid map between the rigs
   from particles that both see, removes duplicates, and writes one merged set.
4. **Many runs with one calibration:** [Multi-folder runs](../multi-folder-runs.md).

## Traps

* **Distortion fitted from one plane is not physical.** One plane cannot separate lens distortion from camera tilt. The
  program leaves `.addpar` at zero and gets the calibration right from many planes instead.
* **The focal length cannot come from one plane,** but is well determined by many (step 2).
* **A wrong labelling is silent.** A grid that looks the same after a half turn can be labelled upside down in one camera and
  nothing complains. This is why the coded corner and the "up" hint exist. If an image looks wrong in the triangulation plots,
  it was probably mislabelled. See the worked example for how this ruined two frames.
* **Mixing rigs.** The program refuses to run when the camera numbers and the folder disagree (step 0). Do not turn this off.

## If it did not work

| You see | Cause | What to do |
|---|---|---|
| `camera group / working folder mismatch` | `PLATE_RIG_CAMS` and the folder are for different rigs | set `PLATE_RIG_DIR` and `PLATE_RIG_CAMS` together |
| epipolar lines miss the dots, reprojection error is small | the observation volume reaches behind a camera, or distortion was fitted | step 6, and keep `.addpar` at zero |
| the triangulated plate is not flat or the spacing is off by more than 1 % | wrong `pitch`, wrong `cc`, or mislabelled frames | measure the pitch again; run the plots of the worked example |
| few frames are labelled | the dots are too small or too dark | see the detection section of the [how-to](../plate-calibration-howto.md) |

## More detail

[Plate calibration how-to](../plate-calibration-howto.md) explains each step, every script and the maths.
[Worked example](../plate-calibration-worked-example.md) shows the numbers, and what went wrong on the way.

## Old names

The scripts moved and the environment variables changed. If you used the old ones:

| Old | New |
|---|---|
| `scripts/illmenau/` | `scripts/plate_rigs/` |
| `scripts/calibrate_illmenau_4cam.py` (and `_part2`) | `scripts/calibrate_plate_4cam.py` (and `_part2`) |
| `scripts/detect_illmenau_4cam_part1.py` | `scripts/detect_plate_4cam_part1.py` |
| `ILLMENAU_RAW`, `ILLMENAU_DIR`, `ILLMENAU_CAMS`, `ILLMENAU_NPZ`, `ILLMENAU_NO_HINT` | `PLATE_RIG_RAW`, `PLATE_RIG_DIR`, `PLATE_RIG_CAMS`, `PLATE_RIG_NPZ`, `PLATE_RIG_NO_HINT` |
| image folders `Kalibrierung_<N>` | `plate_cam<N>` |
| working folders `openptv_illmenau_4cam`, `openptv_illmenau_5678` | `openptv_rig_a_4cam`, `openptv_rig_b_4cam` |

The old names are gone, not aliased: these are command-line tools, and an unknown name fails at once with a clear error.

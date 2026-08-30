# Calibrating the Illmenau 4-camera rig from a hand-held dots plate

How the `.ori` files for cameras 1–4 of the Illmenau barrel were produced, why the
first few attempts failed the epipolar test even at 0.5 px reprojection RMS, and
what to run next time. Cameras 5–8 follow the same recipe (see the last section).

Dataset: `C:\Users\alex\Downloads\Illmenau` — `Kalibrierung_1..4` hold 48
synchronised frames each of a hand-held 6×7 dot plate at unknown depths;
`openptv_illmenau_4cam` is the openptv2 working folder.

Everything below assumes:

```bash
export ILLMENAU_RAW="C:/Users/alex/Downloads/Illmenau"          # optional, this is the default
export ILLMENAU_DIR="$ILLMENAU_RAW/openptv_illmenau_4cam"
```

OpenCV is not a project dependency, so the two scripts that need it are run with
`uv run --with opencv-python-headless`.

---

## 1. The coordinate frame

```
+X  left  -> right
+Y  bottom -> top      (gravity is -Y)
+Z  object -> camera   (cameras 1-4 sit at +Z, cameras 5-8 at -Z)
```

Right-handed. **The origin is a physical dot**, not the plate centre: the coded
white-in-dark **L-corner dot on frame `00000000`** — third column from the left,
fourth row from both top and bottom.

In the barrel frame (origin on the test-section axis at mid-height, floor at
−3580/2 = −1790) that dot sits at `(0, −3580/2 + 615, 0) = (0, −1175, 0)`, so

```
barrel_from_plate(P) = P + [0, -1175, 0]
```

Recorded in `plate.yaml:datum`. `scripts/illmenau/draw_rig_global.py` renders the
frame, the barrel, the plate and both the nominal and calibrated camera positions
(`rig_3d_global.png`).

## 2. How the grid is numbered

`src/openptv2/plate_labeler.py:label_coded_6x7` needs no clicking. Three dots on
the plate are coded (bright centre, dark ring); `detect_plate.py` finds them by
comparing a 5×5 centre mean against an annulus mean. Of the three, the **corner**
is the one whose partners lie at ≈1·pitch and ≈2·pitch at a right angle; the
1·pitch partner defines **+Y**, the 2·pitch partner defines **+X**. On frame
`00000000` they resolve to grid `(2,3)`, `(2,4)`, `(4,3)`, so the corner is `(2,3)`.

Every other dot follows from `ix = round((p−corner)·e_x / pitch)`, likewise `iy`.
Grid is `ix = 0…5` left→right, `iy = 0…6` bottom→top, pitch 120 mm both ways.

**Point id = `iy·6 + ix + 1`** (row-major, 1-based from bottom-left), which is what
`cal/calibration_block.txt` contains:

| | ix=0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **iy=6** | 37 | 38 | 39 | 40 | 41 | 42 |
| **iy=3** | 19 | 20 | **21** | 22 | 23 | 24 |
| **iy=0** | 1 | 2 | 3 | 4 | 5 | 6 |

**The origin is point id 21** — line 21 of `calibration_block.txt` is `21 0.0 0.0 0.0`.
The block runs from `1 -240 -360 0` to `42 360 360 0`.

## 3. The two traps

These cost most of the debugging, and both are invisible to per-camera
reprojection RMS, which sat at a healthy 0.5 px throughout.

### Trap 1 — distortion fitted from a single plane is unphysical

Fitting `k1,k2,k3` with `cv2.calibrateCamera` on **one** plane is degenerate:
distortion trades against pose, and the solver happily returns a radial polynomial
that fits those 42 points and diverges everywhere else. The symptom is specific
and diagnosable: **the projection of a straight 3D ray doubles back inside the
sensor**. It cannot — a straight line projects to a straight line in a pinhole
camera — so any curvature means the distortion model is broken.

Epipolar miss distances before the fix (dense sampling of the curve, frame
`00000000`), where "straight?" counts points whose projected ray does not
double back:

| A→B | median miss | max | straight inside sensor? |
|---|---|---|---|
| 1→2 | 0.53 px | 0.75 | 4/40 |
| 3→2 | 4.98 px | 9.14 | 0/40 |
| 4→2 | 13.06 px | 18.55 | 0/40 |
| 2→1 | 7.66 px | 11.76 | 40/40 |

Cameras 2 and 4 were the visibly wrong ones in the GUI. **Fix: drop the
distortion entirely and refit as a pure pinhole** — `.addpar` is written as zeros.

### Trap 2 — `cc` cannot be fitted from one plane, but is well determined by many

Sweeping `cc` from 9.20 to 9.75 mm while re-fitting the pose each time moved the
triangulated pitch by 0.02 mm. That is not insensitivity, it is exact degeneracy:
a single plane re-fit at a different focal length gives a **self-similar**
reconstruction. `cc` is unobservable there.

It *is* observable from multiple planes. Fit each camera's pose on the reference
frame, then ask each camera separately where the plate is in some other frame. If
`cc` is right the four answers coincide; if it is wrong each camera's world frame
sits at the wrong distance and the answers spread apart, the more so the further
the plate is from the reference plane. Minimising that spread over 23 frames:

| cc [mm] | 8.20 | 8.40 | **8.586** | 8.80 | 9.00 | 9.44 |
|---|---|---|---|---|---|---|
| cross-camera spread [mm] | 84 | 62 | **33.6** | 41 | 70 | 125 |

A clean minimum. **The fitted focal length is 8.586 mm, not the nominal 9.44 mm.**

## 4. The recipe

```bash
cd "$ILLMENAU_DIR"

# 1. detect + L-code label every frame of all four cameras, cache to npz
#    (~192 images, a few minutes; writes cal/labelled_all_frames.npz)
uv run --project <openptv2> python <openptv2>/scripts/illmenau/detect_plate_frames.py

# 2. fit the ONE shared cc from multi-plane cross-camera consistency
uv run --project <openptv2> --with opencv-python-headless \
    python <openptv2>/scripts/illmenau/fit_plate_cc.py

# 3. write the .ori: pure pinhole at that cc, pose from the reference frame,
#    zero distortion in .addpar
uv run --project <openptv2> --with opencv-python-headless \
    python <openptv2>/scripts/illmenau/refit_plate_pinhole.py 8.5858

# 4. verify -- these two are the acceptance gate
uv run --project <openptv2> python <openptv2>/scripts/illmenau/check_plate_triangulation.py
uv run --project <openptv2> python <openptv2>/scripts/illmenau/check_epipolar.py

# 5. optional: the whole-dataset sweep and the frame diagram
uv run --project <openptv2> python <openptv2>/scripts/illmenau/check_all_frames.py
uv run --project <openptv2> python <openptv2>/scripts/illmenau/draw_rig_global.py
```

Steps 2 and 3 are the only ones that touch the calibration; step 1 is cached, so
re-running 2–4 with a different `cc` costs seconds.

## 5. Set the observation volume before opening the GUI

`src/openptv2/algorithms/epi.py:129` (`epi_mm`) does **not** compute a line. It
computes two *endpoints*, by walking the ray to `Z = Zmin_lay` and `Z = Zmax_lay`
(interpolated in X across `X_lay`). The epipolar line never changes; the box
decides **which piece of it gets drawn**. Two consequences:

- Change `Xmin/Xmax/Zmin/Zmax` and the segment appears to move. It has not — you
  are drawing a different portion.
- **`Zmax` must stay well below the nearest camera.** With cameras at Z ≈ 3000–3060
  and the old `±4000` box, the far endpoint was computed *behind* the camera and
  projected to absurd coordinates, throwing the segment across the image.

Measured segment behaviour for the dot at the origin:

| pair | Z box | segment length | passes through the dot? |
|---|---|---|---|
| 1→2 | ±500 | 300 px | yes |
| 1→2 | **±1500** | **786 px** | **yes** |
| 1→2 | ±2500 | 49 911 px | no — endpoint past the camera |

`parameters_Run1.yaml` is set to `X_lay ±1500`, `Zmin_lay −1500`, `Zmax_lay +1500`,
with both layer values of each bound equal so the X interpolation stays constant.
If the lines look too short, widen `Zmin/Zmax` — but never past ~2500.

## 6. What the result looks like

Final calibration, `cc = 8.586 mm` shared, zero distortion:

| cam | C (X, Y, Z) mm | reproj RMS |
|---|---|---|
| 1 | (1470, 137, 3060) | 0.565 px |
| 2 | (1482, 2253, 3001) | 0.518 px |
| 3 | (−1472, 137, 2968) | 0.579 px |
| 4 | (−1500, 2254, 3007) | 0.531 px |

**Reference frame `00000000`, triangulated:**

- plane normal `(0.0000, 0.0000, 1.0000)`, offset 0.023 mm from the origin
- planarity residual RMS **0.306 mm**, max 0.731 mm
- pitch **120.82 mm in X (+0.68 %)**, **120.12 mm in Y (+0.10 %)**
- absolute position vs the block coords, no alignment applied: **1.09 mm median**,
  1.75 mm max, bias below 0.02 mm on every axis

**Epipolar check, all 12 ordered pairs:** closest approach of the epipolar curve to
the correct dot is **0.11–0.30 px median, 0.93 px worst case**, and the curve is
straight inside the sensor for **every** point of **every** pair. Camera 2, the
worst before, is now the best.

**Accuracy away from the reference plane** (frames whose labelling is
self-consistent):

| distance from reference plane | n | planarity RMS median / max |
|---|---|---|
| 0 – 1000 mm | 2 | 0.36 / 0.41 mm |
| 1000 – 2000 mm | 2 | 0.87 / 1.33 mm |
| 2000 – 3000 mm | 7 | 2.17 / 21.99 mm |
| 3000 – 5000 mm | 12 | 3.87 / 29.28 mm |

Sub-millimetre inside the ±1500 mm observation box, degrading beyond it. Set the
box to the volume you actually measure in.

## 7. Known limitation — the labeller, not the calibration

Over all 48 frames the planarity RMS median is 17.5 mm, and only 9 frames come in
under 2 mm. That is **not** calibration error. Splitting the frames by a
per-camera PnP fit of the rigid 6×7 plate — a test that uses no cross-camera
information at all:

| | n | planarity RMS median |
|---|---|---|
| labelling self-consistent (PnP < 1 px) | 23 | 2.70 mm |
| labelling broken (PnP ≥ 1 px) | 24 | 26.57 mm |

Half the hand-held frames are mislabelled. The obvious failure mode — a 180°
relabelling, which a 6×7 grid admits since `id → 43 − id` maps it onto itself —
was checked for and does **not** occur; the errors are individual dots assigned to
wrong ids when the plate is steeply tilted or partly occluded. Frames 2, 3, 14 and
37 are the clearest cases (recovered pitch 87, 144, 88 and 849 mm).

This does not affect the delivered `.ori`, because `fit_plate_cc.py` admits a frame
only when all four cameras fit it below 1.5 px. It does mean roughly half the
captured data is wasted, so improving `label_coded_6x7` for tilted views is the
highest-value next change.

## 8. Next time / cameras 5–8

Cameras 5–8 sit on the opposite wall at −Z and see the **back** face of the plate,
so two things change:

1. Viewed from behind, the plate is mirrored — `+X` of the printed grid runs
   right→left in those images. Either pass `y_sign`/an axis flip to the labeller
   or relabel `ix → 5 − ix`, and confirm the L code still resolves to the same
   physical dot as id 21. Verify before fitting: the datum dot must be the same
   piece of plastic for both camera groups, or the two halves will not share a
   world.
2. They need their own reference frame in which all four of them see the plate.
   If cameras 1–8 ever see the plate simultaneously, use that frame for both
   groups and the whole rig lands in one frame directly. Otherwise calibrate
   group 5–8 in its own frame and tie the two together with a Kabsch fit on the
   plate points they share (`calibration_import.similarity_from_correspondences`).

The recipe itself is unchanged: cache detections, fit one shared `cc` from
multi-plane consistency, write a pure-pinhole `.ori`, then gate on
`check_plate_triangulation.py` and `check_epipolar.py`. Expect the fitted `cc` to
land near 8.59 mm again if the lenses are the same.

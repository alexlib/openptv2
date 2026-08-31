# Calibrating the Illmenau 4-camera rig from a hand-held dots plate

How the `.ori` files for cameras 1–4 of the Illmenau barrel were produced, why the
first few attempts failed the epipolar test even at 0.5 px reprojection RMS, how
far the resulting model can be trusted, and what to run next time. Cameras 5–8
follow the same recipe (see the last section).

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

## 3. From a grid at an arbitrary position to an `.ori` file

This is the whole chain, with nothing hidden. Only two of these steps involve any
fitting; the rest is bookkeeping and algebra.

**3.1 — each detected dot gets a plate coordinate.** The labeller gives every dot
an `(ix, iy)`. Because the plate is rigid and its pitch is known, that fixes the
dot's position in a coordinate system attached to *the plate*, with the datum dot
at the origin and the plate lying in `z = 0`:

```python
obj = [(ix - DATUM_IX) * pitch_x, (iy - DATUM_IY) * pitch_y, 0.0]      # = ref_pts
```

`DATUM_IX, DATUM_IY = 2, 3` for this plate. Nothing about where the plate is being
held enters here — this is the plate's own geometry, identical in every frame.

**3.2 — one frame is promoted to *the* world.** Pick a reference frame (here
`00000000`) and *declare* that in that frame the plate coordinate system **is** the
world coordinate system. That single declaration is what pins the world to a piece
of plastic. Every other frame's plate position is then an unknown to be measured,
not an input.

**3.3 — each camera's pose is solved from that one frame.** With object points
from 3.1 and their pixels in camera *i*, `cv2.solvePnP` + `solvePnPRefineLM` return
`(rvec, tvec)`: the rigid transform taking world points into camera *i*'s frame,
`x_cam = R·X_world + t`, with `R = Rodrigues(rvec)`. This is the first fit. It
needs no seed and no initial guess — six unknowns from ~42 points.

**3.4 — the intrinsic matrix.** `K = [[cc/pix_x, 0, imx/2], [0, cc/pix_y, imy/2], [0,0,1]]`
with the principal point at the sensor centre and **zero distortion**. `cc` is the
one number that cannot be obtained from a single plane (§4, Trap 2) — it comes from
step 3.6, and 3.3–3.5 are simply re-run for each trial value.

**3.5 — convert OpenCV's `[R|t]` into openptv2's `.ori`.** Pure algebra, in
`calibration_import.calibration_from_opencv`:

| OpenCV | openptv2 `.ori` | why |
|---|---|---|
| `R`, `t` | `dm = Rᵀ · S`, with `S = diag(1, −1, −1)` applied **on the right** | OpenCV's camera frame is x-right, y-**down**, z-**forward**; openptv2's is x-right, y-**up**, z-**backward** (the camera views along `−dm[:,2]`). `S` flips those two axes. |
| `R`, `t` | `C = −Rᵀ·t` | the projection centre in world coordinates — the `x0 y0 z0` line of the `.ori` |
| `dm` | `ω, φ, κ` via `angles_from_dm` | the `.ori` stores angles; `dm` is written under them and is what the code actually reads |
| `fx` | `cc = fx · pix_x` | pixels → mm |
| `cx, cy` | `xh, yh` (as an offset from the sensor centre) | principal point |
| `distCoeffs` | `.addpar` `k1 k2 k3 p1 p2 scx she` | **left at zero here** — see §4, Trap 1 |

Then `Calibration.to_file` writes `cal/camN.tif.ori` and `cal/camN.tif.addpar`.
There is no bundle adjustment, no GUI orientation clicking, and no `sortgrid`: the
L code already gave every dot its identity, which is what `sortgrid` normally has
to guess (`autocalibration._refine_and_select(presorted=True)`).

**3.6 — where `cc` came from.** The one genuinely global fit. For a trial `cc`,
run 3.3–3.5 to get four poses, then take *another* frame and ask each camera
**separately** where the plate is now (`solvePnP` again, per camera, in that
camera's own world frame). If `cc` is right the four answers coincide; if it is
wrong each camera's world sits at the wrong distance and the answers spread apart.
Minimising the median spread over the usable frames is a clean 1-D problem →
`fit_plate_cc.py`, §4 Trap 2.

**In one line:** the plate's own geometry gives object points; one frame is
declared to be the world; `solvePnP` turns each camera's view of that frame into a
pose; a change of basis turns the pose into an `.ori`; and the focal length — the
only quantity a single plane cannot see — is fixed by requiring the four cameras to
agree about *other* plate positions.

## 4. The two traps

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

It *is* observable from multiple planes, by the procedure in §3.6:

| cc [mm] | 8.20 | 8.40 | **8.586** | 8.80 | 9.00 | 9.44 |
|---|---|---|---|---|---|---|
| cross-camera spread [mm] | 84 | 62 | **33.6** | 41 | 70 | 125 |

A clean minimum. **The fitted focal length is 8.586 mm, not the nominal 9.44 mm.**

## 5. The recipe

`OPTV=<path to openptv2>`. Everything runs from the working folder; the dataset
is found through `ILLMENAU_RAW` / `ILLMENAU_DIR`. Only steps 1-3 and 5 change the
calibration; the rest only measure it.

```bash
cd "$ILLMENAU_DIR"

# --- 1. DETECT + LABEL every frame of all four cameras, cached to npz.
# ~192 images, a few minutes -> cal/labelled_all_frames.npz.  Hands the labeller
# the two things one image cannot settle: the datum's grid index, and the image
# direction of world +Y from the current .ori (section 9).  On a brand-new rig
# with no .ori yet, run once with ILLMENAU_NO_HINT=1, then re-run after step 3.
uv run --project $OPTV python $OPTV/scripts/illmenau/detect_plate_frames.py

# --- 2. FIT the one shared cc, from multi-plane cross-camera consistency.
# The only quantity a single plane cannot see (section 4, Trap 2).
uv run --project $OPTV --with opencv-python-headless \
    python $OPTV/scripts/illmenau/fit_plate_cc.py

# --- 3. WRITE the .ori: pure pinhole at that cc, poses from the reference
# frame, zero distortion in .addpar (section 4, Trap 1).
uv run --project $OPTV --with opencv-python-headless \
    python $OPTV/scripts/illmenau/refit_plate_pinhole.py 8.5858

# --- 4. GATE.  These two decide whether the calibration is usable.  Never gate
# on per-camera reprojection RMS -- it stayed at 0.5 px through every wrong
# intermediate result of this work.
uv run --project $OPTV python $OPTV/scripts/illmenau/check_plate_triangulation.py
uv run --project $OPTV python $OPTV/scripts/illmenau/check_epipolar.py

# --- 5. BUNDLE: re-solve all camera poses together with one plate pose per
# frame, so the model is no longer exact on the reference plane and drifting
# away from it (section 8b).  cc stays fixed; the reference plate pose is the
# gauge.  Omit --write for a dry run that changes nothing.
uv run --project $OPTV --with opencv-python-headless \
    python $OPTV/scripts/illmenau/bundle_plate_poses.py 8.5858 --write

# --- 6. RE-GATE (step 4 again), then validate over the whole dataset and look
# at the pictures.  plot_frame_triangulation.py fits nothing: it shows what the
# .ori on disk actually reconstruct, one figure per plate position.
uv run --project $OPTV python $OPTV/scripts/illmenau/check_plate_triangulation.py
uv run --project $OPTV python $OPTV/scripts/illmenau/check_epipolar.py
uv run --project $OPTV python $OPTV/scripts/illmenau/plot_frame_triangulation.py
uv run --project $OPTV python $OPTV/scripts/illmenau/check_all_frames.py

# --- 7. SET THE OBSERVATION VOLUME before opening the GUI (section 6).
# Reports the largest safe Zmax for this rig; it is a display and search
# parameter, not a calibration one, and the usual cause of "the epipolar lines
# are wrong".
uv run --project $OPTV python $OPTV/scripts/illmenau/check_epipolar_volume.py

# --- optional: draw the rig against the global frame, as a sanity check that
# the cameras came out where they physically are.
uv run --project $OPTV python $OPTV/scripts/illmenau/draw_rig_global.py
```

Step 1 is cached, so re-running 2-5 with a different `cc` costs seconds. After
step 5 it is worth re-running step 1 once: the labeller uses the `.ori` for its
up-hint, so a better calibration can rescue frames it previously mislabelled.

## 6. Set the observation volume before opening the GUI

`src/openptv2/algorithms/epi.py:129` (`epi_mm`) does **not** compute a line. It
computes two *endpoints*, by walking the ray to `Z = Zmin_lay` and `Z = Zmax_lay`
(interpolated in X across `X_lay`) and projecting those two 3D points into the
second camera. The GUI then draws the straight chord between them. So the box can
change what you see in three different ways, and only one of them is a bug:

**Truncation** — the segment covers only the Z range you asked for. If the true
depth is outside it the segment stops short of the matching dot. Expected, and the
usual reason lines "do not reach". Widen the box.

**Chord error** — the drawn chord equals the true epipolar curve only if that curve
is straight. It is straight for a pinhole; a non-zero `.addpar` bends it, worst in
the *middle* of a long segment, because the endpoints get dragged out to the image
periphery where the distortion polynomial is largest. **This is why shrinking the
box appeared to fix the calibration** back when `.addpar` was non-zero — it was
hiding Trap 1, not fixing it. With the delivered zero-`.addpar` files this term is
identically zero, and `check_epipolar_volume.py` confirms the miss distance is
**0.08 px at every box size from ±500 to ±3000 mm**. The line really is the same
line; only its length changes.

**Horizon flip** — the hard failure. Once the sampled point passes the plane
through the second camera's projection centre, the depth that projection divides by
goes through zero and changes sign: the endpoint lands on the far side of the
sensor and the chord is thrown across the image. Nothing is wrong with the
calibration; the box asked for a point behind the camera.
`check_epipolar_volume.py` solves for that horizon in closed form:

| A→B | 1→2 | 1→3 | 1→4 | 2→1 | 2→3 | 2→4 | 3→1 | 3→2 | 3→4 | 4→1 | 4→2 | 4→3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| horizon Z [mm] | 3499 | 3695 | 4294 | 3592 | 4554 | 3600 | 3795 | 4307 | 3479 | 4669 | 3595 | 3489 |

**The binding limit is Z = 3479 mm** (pair 3→4). With a 20 % margin, `Zmax_lay` may
go up to **≈ 2780 mm** — so a `±2500` box is fine and a `±4000` box (the one that
produced the original "epipolar lines are wrong" report) is past the horizon for
six of the twelve pairs. `parameters_Run1.yaml` currently uses `±1500`, which is
conservative; you can safely open it up to `±2500` if you need the depth.

Keep both layer values of each bound equal (`Zmin_lay: [-z, -z]`) so the X
interpolation stays constant across the volume.

*Not fixed in the library:* clamping the endpoint to the horizon inside `epi_mm`
was tried and backed out. It changes results for configurations the existing parity
tests (`test_epi_mm`, `test_epi_mm_perpendicular`) deliberately encode — those
fixtures place cameras inside the volume and expect the C original's behaviour. The
limit is reported by the diagnostic instead of enforced by the library.

## 7. What the result looks like

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

## 8. Validating the model on all 48 planes

`plot_frame_triangulation.py` triangulates every frame with the delivered
`.ori`/`.addpar` — it fits nothing — and writes `triangulation/frame_XXXXXXXX.png`
plus `summary.png` and `summary.csv`. Each figure has a 3D panel (dots coloured by
distance from the best-fit plane, with the cameras and the world origin) and a
face-on panel showing the dots against the ideal rigid 6×7 grid Kabsch-fitted onto
them.

**Three numbers separate the two things that can go wrong**, which is the point of
the exercise:

| number | what it measures | what a bad value means |
|---|---|---|
| deviation from the rigid grid | does the labelled 6×7 pattern come back as a 6×7 pattern | > ~30 mm ⇒ the **labeller** mis-assigned dots |
| ray-convergence miss (RCM) | do the sight lines of a dot actually meet | uses **no** plate model, so it cannot be fooled by the grid — a bad value with a good grid is a **calibration** error |
| planarity RMS | does the plate come back flat | weak: a systematic model error can distort a plane consistently and still look flat |

29 of the 48 frames are mislabelled (grid deviation 40–1150 mm) — the plate is
steeply tilted or partly occluded and `label_coded_6x7` gives up. Those frames say
nothing about the calibration. Frame `00000002` is the clearest example, and its
face-on panel makes the scrambling obvious at a glance.

**The 19 correctly-labelled frames are the real test, and they show a limitation
the earlier write-up missed.** The plate comes back flat and at the right pitch
everywhere, but the sight lines stop meeting as the plate moves away from the
reference plane:

| distance of the plate from the world origin | frames | planarity RMS | ray-convergence miss |
|---|---|---|---|
| 0 – 1000 mm | 2 | 0.36 mm | **0.36 mm** |
| 1000 – 2000 mm | 2 | 0.87 mm | **6.1 mm** |
| 2000 – 3000 mm | 6 | 1.83 mm | **11.1 mm** |
| 3000 – 5000 mm | 9 | 3.19 mm | **18.0 mm** |

A least-squares fit over those 19 frames gives **RCM ≈ 0.58 % of the distance from
the anchor plane**, and it is remarkably consistent frame to frame (0.30–0.66 %).
Frame `00000030` is the cleanest illustration: a textbook grid (median deviation
1.7 mm), planarity 0.90 mm, pitch 121.03 mm — and an RCM of 10.6 mm at 2131 mm.
Nothing is mislabelled there; the rays genuinely do not meet.

**Where that comes from.** All four extrinsics are solved on frame `00000000`
alone (§3.3), so the model is exact *on that plane* by construction and any small
error in the relative orientation between cameras is absorbed there and grows
linearly with distance from it. `cc` was checked and is not the cause: sweeping it
against this same RCM objective gives a shallow minimum near 8.8 mm that only moves
the median from 15.5 to 12.8 mm — it does not collapse. Refitting per-camera
intrinsics over all planes with `cv2.calibrateCamera` (including `k1`) *improves*
reprojection RMS to 0.38–0.44 px and makes RCM **worse** (20–30 mm), which is a
clean demonstration that reprojection RMS is the wrong objective for this rig.

**Practical consequence.** The delivered calibration is sub-millimetre within about
1.5 m of the reference plane, and degrades to ~1–2 cm at 3–4 m. If you measure in a
volume that big, this is the accuracy floor, and no `.addpar` will fix it.

## 8b. The joint bundle — spreading the error over the volume

`scripts/illmenau/bundle_plate_poses.py` removes the single-plane anchoring.
Unknowns are the four camera poses and one 6-dof plate pose per frame; `cc`
stays at 8.5858 mm, distortion stays zero, the principal point stays at the
sensor centre, and **the reference frame's plate pose is held at identity**.
That last one is the gauge: the world stays pinned to the coded L-corner dot of
frame `00000000`, so `calibration_block.txt`, `plate.yaml:datum` and any manual
GUI check of that frame remain valid, and with the gauge fixed there is no free
similarity, so scale cannot drift even though `cc` is not fitted.

### Rejecting outliers before the fit, not during it

A bundle fed mislabelled views diverges — that is what sank the first
`bundle_shared_cc.py`. Robust loss plus residual trimming is not enough on its
own, because a bad view still drags the early iterations. Three gates run
first, each catching something the previous one cannot see:

1. **Per-camera PnP < 1 px.** Fits the rigid plate to one camera's own labelled
   points, using no cross-camera information, so its residual is a pure
   labelling test for that one view. 155/192 views pass.
2. **Plate vertical within 5°.** See below. Catches 2 views (frames 02 and 14,
   camera 4) that are off by ~89° — grossly mislabelled but internally
   self-consistent.
3. **Cross-camera agreement < 100 mm, per dot.** Each surviving view implies
   where every dot of the plate must be in the world; correct labellings agree.
   Compared **per dot, not per plate centre** — a scrambled labelling can leave
   the centroid roughly where it belongs while the pattern around it is wrong,
   which is how frames 39 and 42 passed an earlier centre-only version of this
   test. 140 views across 44 frames survive.

Then the bundle itself trims dots above 3x the median residual over six rounds
(6215 → 4964 dots). Tightening any of these gates further *improves*
reprojection RMS while making planarity **worse** — it starves the fit. Same
trap as §8: reprojection RMS is not the objective.

### The plate is held vertical

Measured over the well-labelled frames: the plate normal is within **0.83° of
horizontal** (median 0.23°), the plate's own up axis within **1.23° of world
+Y** (median 0.70°), and the yaw about +Y spans −24° to +30°. So the plate really
is vertical and rotated only about Y, to about a degree.

That is used as an outlier gate (above) and as a **soft** penalty on the two
off-yaw rotation components, `R_f[1,0]` and `R_f[1,2]`, which vanish for a pure
yaw. Soft rather than hard because the departure is ~0.2-1.2°, not zero — the
plate is hand-held, and forcing it to zero would bias the far corners of a
720 mm plate by ~15 mm.

**Honest result: the prior changes nothing measurable here** (RCM 2.67 vs
2.71 mm with it disabled). 44 frames already determine the plate poses. It is
kept because it costs nothing, because it is what justifies the 5° outlier gate,
and because it should matter on a sparser dataset — cameras 5-8 may well yield
fewer clean frames. `BUNDLE_VERT_PX=0` disables it.

### What the bundle bought

| | anchored to frame 0 | joint bundle |
|---|---|---|
| RCM vs distance | **0.58 %** | **0.159 %** |
| RCM, 3-5 m | 18.0 mm | **4.75 mm** |
| planarity, 3-5 m | 3.19 mm | **1.39 mm** |
| planarity, median over frames | 1.83 mm | **0.60 mm** |
| recovered X pitch, median | 120.90 mm (+0.75 %) | **120.29 mm (+0.24 %)** |
| frame 0 epipolar, median range | **0.11-0.30 px** | 0.58-1.45 px |
| frame 0 absolute error | **1.09 mm** | 1.52 mm |
| camera positions | — | moved 24-39 mm |

**A 3.7x improvement in ray convergence across the volume, paid for by giving up
some of frame 0's perfection.** That is the trade the anchored fit was hiding:
it was not more accurate, it was concentrating all its accuracy on one plane.
Every epipolar pair is still under 1.5 px median and straight inside the sensor,
so the GUI check on frame 0 still passes — just less spectacularly.

Both sets of files are kept: `cal/camN.tif.ori` is the bundled result and
`cal/camN.tif.ori.prebundle` the anchored one. Re-run
`refit_plate_pinhole.py 8.5858` to get back to the anchored fit.

## 9. The labeller — three bugs, and what the calibration paid for them

Half the frames used to be unusable. None of the three causes was a detector
tuning problem, and all three produced labellings that were **wrong yet
internally self-consistent** — a per-camera PnP fit of the rigid plate to them
has a sub-pixel residual, so no single-view check can see any of them. They
surface only as a cross-camera disagreement, long after the damage is done.

### 9.1 The grid was anchored to whatever the view happened to see

`label_coded_6x7` shifted the grid so the smallest detected index became 0. A
view that could not see the leftmost column or the bottom row therefore labelled
**every dot one step off**. Fixed by anchoring on the coded corner instead:

```python
label_plate(..., corner_index=(DATUM_IX, DATUM_IY))
```

Pass it whenever you know it — which is always, since the datum has to be
recorded anyway. Regression tests: `tests/unit/test_plate_labeler_anchor.py`.

### 9.2 The coded L could resolve to the wrong corner

The three coded dots form a 1:2 right angle, but taking the `+1·pitch` dot as
the corner *also* gives legs at a near-1:2 ratio and a near-right angle. Under
strong perspective that spurious optimum can win, rotating the whole grid.

The plate is held vertical, so its own +Y is world +Y — and the calibration
knows where world +Y points in each image. `plate_labeler.image_up_direction`
computes that direction and `up_hint` rejects candidates whose +Y is more than
60° away. **This is what "use the calibration in detection" means concretely.**
Requiring merely the same half-plane is not enough: a 90° error sits exactly on
that boundary.

### 9.3 A failed coded detection fell through to the uncoded labeller

The worst of the three. With `coded_thr` fixed at 30, two views found **zero**
coded dots; `label_plate` then dispatched on `n_coded == 3` being false and
silently used `label_uncoded_grid`, which has no origin and no orientation
anchor. It returned a confidently labelled grid rotated by 90°. Two fixes:

- `label_plate` now **raises** when a coded plate yields other than 3 coded
  dots, instead of falling back to a labeller with no datum;
- the plate has exactly three coded dots, so the driver *searches* for the
  threshold that finds three (`CODED_THR = (30, 25, 20, 15, 10)`) rather than
  fixing one. Both failing views were fine at 15.

### What the fixes bought

| | before | after |
|---|---|---|
| frames whose triangulated grid is correct | **20/48** | **48/48** |
| views passing the per-camera PnP gate | 155/192 | 189/192 |
| views entering the bundle | 143 | 180 |
| dots in the bundle | 6215 | 7352 |
| non-vertical plate poses | 3/192 | 1/192 (3.2°, real tilt) |
| RCM vs distance | 0.159 % | **0.126 %** |

Frame `00000037` is the clearest case: 19 scattered points with a 1148 mm grid
deviation and a nonsense 849 mm recovered pitch, now a complete 42-dot grid at
12 mm deviation. Every frame in `triangulation/` is now worth looking at.

**The general lesson:** on a coded plate, every labelling decision the image
alone cannot settle should be settled by something outside the image — the
recorded datum index, or the calibration. Where neither can settle it, fail
loudly. A labeller that guesses produces exactly the errors reprojection RMS
cannot see.

## 10. Cameras 5–8 — the far wall, calibrated as its own rig

**Done.** Working folder `openptv_illmenau_5678/`, its own `cal/`, its own
`parameters_Run1.yaml` (cameras named cam5..cam8), its own `plate.yaml`.

### Why a separate world

Cameras 5–8 see the **back** face of the plate, and the back face carries a
*different* dot pattern with its own ids — there is no dot-to-dot correspondence
with the front to exploit. So the two groups are calibrated independently, each
anchored to its own reference frame's coded L-corner dot. Merging them is a
later job, and the plate is what makes it possible: the two dot planes are
parallel and **6 mm apart** (the plate's thickness), so the per-frame plate poses
already solved for cams 1–4 describe the same physical positions. The recipe is
recorded in `openptv_illmenau_5678/plate.yaml:relation_to_front_rig` — offset a
cams 1–4 plate pose by −6 mm along its normal, then solve the 3-dof in-plane
placement (yaw about the normal + 2D shift) shared across all frames. Heavily
over-determined, and it needs no Kabsch fit.

*The earlier guess in this section was wrong on both counts it made:* the grid is
**not** mirrored (the L code names the corner and its +X/+Y partners by leg
length, which are properties of the plate, not of the side you view it from), and
there is **no** frame in which all eight cameras see the plate.

### Running it

Every driver is camera-group agnostic via `scripts/illmenau/_config.py`:

```bash
export ILLMENAU_DIR="$ILLMENAU_RAW/openptv_illmenau_5678"
export ILLMENAU_CAMS=5,6,7,8
# then §5's recipe verbatim, plus these two first:
python $OPTV/scripts/illmenau/find_datum.py             # read the datum off the data
python $OPTV/scripts/illmenau/make_calibration_block.py # block from the same obj_of
```

The first detection pass has no `.ori` to take an up-hint from, so run
`ILLMENAU_NO_HINT=1 detect_plate_frames.py --cams 5,6,7,8` once, then re-run it
after the first `.ori` exist.

### The datum was tested, not inherited

The back face's coded L corner turns out to sit at the same grid node as the
front's, **(2,3)**, but that was established rather than assumed. `find_datum.py`
needs a view showing the complete 42-dot lattice, and the far wall rarely gives
one, so the cheaper cross-check settled it: label all 193 back-face views with
the candidate index and look at the id histogram. A wrong index pushes part of
the lattice outside the `nx × ny` rectangle, where the labeller silently drops
it. With (2,3) nothing is clipped — all 42 ids seen, per-id counts 110/191/193,
lattice edges covered 1309/1326/1147/962, against the front face's 106/192/192
and 1303/1316/1149/962.

### Result

`cc = 8.6313 mm` shared — within 0.5 % of the front wall's 8.5858 mm, which is
independent corroboration, since nothing constrained the two groups to agree.

| cam | C (X, Y, Z) mm | | cam | C (X, Y, Z) mm |
|---|---|---|---|---|
| 5 | (−1444, 128, −2998) | | 1 | (1454, 122, 3052) |
| 6 | (−1443, 2290, −3038) | | 2 | (1451, 2266, 2990) |
| 7 | (1460, 105, −3023) | | 3 | (−1456, 126, 2959) |
| 8 | (1455, 2253, −2987) | | 4 | (−1470, 2268, 2994) |

A clean mirror of the near wall across Z = 0 — also not constrained.

| | cams 1–4 (front) | cams 5–8 (back) |
|---|---|---|
| Gate A planarity / absolute error | 0.337 / 1.353 mm | 0.887 / 2.507 mm |
| frames passing the grid gate | 48/48 | 47/49 |
| RCM vs distance | 0.126 % | **0.026 %** |
| median RCM | 4.54 mm | **2.47 mm** |
| median planarity | 1.364 mm | **0.796 mm** |
| recovered X pitch | 120.16 mm | 120.20 mm |

The far wall is the *better* of the two across the volume — its ray convergence
is nearly flat with distance. Its Gate B is correspondingly looser on the
reference frame (0.56–3.46 px median against 0.10–0.31 px before the bundle),
which is the §8b trade in its usual direction: accuracy spread over the volume
instead of concentrated on one plane.

### What went wrong, and what it cost

Two scripts re-detected and re-labelled the reference frame *themselves*, through
`label_plate()` with no `corner_index` — the unsafe anchoring path of §9.1. On
the near wall the reference view shows the whole lattice, so their private
labelling agreed with the cache by luck and this was invisible for the entire
cams 1–4 calibration. On the far wall it is not:

- `refit_plate_pinhole.py` put cameras 5, 6 and 8 at 2.9–4.5 px with centres
  metres from the wall they are bolted to, and Gate A reported **59 mm**
  planarity for a calibration that was fine;
- `check_epipolar.py` reported **100–400 px** misses on every pair involving
  cam7, on 25–35 matched dots instead of 41 — it was comparing dot 7 of one
  camera against dot 8 of another.

Both now read the detection cache. **One labelling, produced once by
`detect_plate_frames.py` with the datum and the up-hint, used by everything.**
A check that re-derives its own ground truth can fail in exactly the way it
exists to detect, and it will look like the calibration's fault.

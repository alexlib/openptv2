---
name: openptv-multiplane-calibrate
description: >-
  Calibrate an openptv2 multi-camera rig from a coded dots plate held at many
  unknown depths (the OpenCV multi-plane style), producing .ori files that pass
  a real epipolar test — no GUI, no manual orientation clicking, no sortgrid.
  Use when a user has folders of calibration images of a dot plate (one folder
  per camera, synchronised frame numbers), wants new .ori files, or has a
  calibration whose per-camera reprojection RMS looks fine but whose epipolar
  lines miss the corresponding points. Covers the parameters the user must
  supply, how the world origin is pinned to one physical dot, why focal length
  cannot be fitted from a single plane, why single-plane distortion is
  poisonous, and the two acceptance gates. Triggers: "multi-plane calibration",
  "calibrate from the dots plate", "epipolar lines do not cross", "OpenCV-style
  calibration for openptv2", "calibrate cameras 5-8", "my .ori looks fine but
  triangulation is wrong".
---

# Multi-plane plate calibration for openptv2

Worked reference implementation: `docs/illmenau-4cam-calibration.md` and
`scripts/illmenau/`. That is a real dataset (Illmenau barrel, 4 cameras, 6×7
coded plate, 48 hand-held frames) calibrated to 0.11–0.30 px median epipolar
error. Read it for the numbers; this file is the procedure.

## When this applies

- One image folder per camera, frames synchronised by filename prefix.
- A rigid plate with a regular dot grid, held at **many unknown depths**.
- Three coded dots forming an L, so every dot's identity is recoverable from the
  image alone. Without the code you need the uncoded path (adjacency BFS +
  RANSAC affine, `plate_labeler.label_uncoded_grid`) and an origin hint.

Because the L code gives point identity for free, **the two GUI steps are not
needed**: no manual-orientation clicking (PnP is direct) and no `sortgrid`
(identities are already known — `_refine_and_select(presorted=True)`).

## What the user MUST provide

Ask for all of these before starting. Missing any one of them makes the result
silently wrong rather than obviously wrong.

**Sensor**

| parameter | why it matters |
|---|---|
| `imx`, `imy` [px] | image size |
| `pix_x`, `pix_y` [mm/px] | sets the scale of `cc`; `cc = fx · pix_x` |
| nominal focal length [mm] | **only a bracket for the sweep**, not the answer. Sweep ±25 % around it. |

**Plate**

| parameter | why it matters |
|---|---|
| `nx`, `ny` | grid size. If both are even, or the grid is square, a 180° relabelling maps it onto itself — check for that failure explicitly. |
| `pitch_x`, `pitch_y` [mm] | absolute scale of the whole reconstruction. Measure it; do not trust the drawing. |
| L-code geometry | which coded dot is the corner, which partner is at 1·pitch (+Y) and which at 2·pitch (+X) |
| **datum grid index** `(ix, iy)` of the L corner | the world origin. See below. |

**Frame convention** — state all three axes and the handedness explicitly, and
draw it before fitting anything (`scripts/illmenau/draw_rig_global.py`). Getting
this wrong is not detectable from residuals. For Illmenau: `+X` left→right,
`+Y` bottom→top, `+Z` object→camera, right-handed.

**Lab-frame offset** — where the datum dot sits in the user's lab/rig frame, so
results can be related to the facility (`barrel_from_plate(P) = P + offset`).

**Observation volume** `X_lay`, `Zmin_lay`, `Zmax_lay` — see the epipolar
section; this is a *display and search* parameter, not a calibration one, but it
is the single most common cause of "the epipolar lines are wrong".

**Detection thresholds** — `gvthres`, `sumg_min`, `nnmin/nnmax` for the plate
dots (separate from the tracer `targ_rec` block — plate dots are much larger),
plus `coded_thr` for the bright-centre test. Tune on one frame with
`--save-debug` before running all of them.

**Rig seed** — approximate camera positions and look-at directions. Note this is
**not** needed by the fit; PnP converges from nothing. It is needed to *check*
the answer: compare the fitted positions against the nominal mounts and against
pairwise camera distances, which are frame-invariant.

## Procedure

### 1. Pin the origin to a physical dot

Do not use the plate centre. Use the **coded L-corner dot on the reference
frame** — a specific piece of plastic the user can point at. Everything (other
planes, other camera groups, the lab frame) is then related to a thing that
exists.

The labeller places the L corner at its own grid origin; shift the reference
coordinates by the datum index so that dot becomes `(0,0,0)`:

```python
ref_pts[:, 0] -= DATUM_IX * pitch_x
ref_pts[:, 1] -= DATUM_IY * pitch_y
```

Verify the datum index from the data, do not assume it. Print the grid indices
of the three coded dots and overlay them on the image. Point id convention:
`id = iy·nx + ix + 1`, written to `cal/calibration_block.txt`.

### 2. Cache detections for every frame

Detect + label once, store `(ids, pixels)` per camera per frame. Everything
downstream is then seconds instead of minutes, which matters because you will
sweep `cc`. → `detect_plate_frames.py`

### 3. Fit ONE shared focal length from multi-plane consistency

**This is the step that makes or breaks the calibration.**

`cc` is *exactly degenerate* on a single plane: re-fit the pose at a different
`cc` and you get a self-similar reconstruction with identical residuals and
identical recovered pitch. Any "fit" of `cc` from one plane is fitting noise.

The observable that does determine it: fit each camera's pose on the reference
frame, then for every other frame ask each camera **separately** where the plate
is. Right `cc` → the four answers coincide. Wrong `cc` → each camera's world
frame sits at the wrong distance and the answers spread apart, worse the further
the plate is from the reference plane. Minimise the median spread.
→ `fit_plate_cc.py`

Use **one shared `cc`** when the cameras have the same lens — it is far better
conditioned than four independent ones. Expect a clean single minimum; a flat
curve means the frames do not span enough depth.

### 4. Write the .ori as a pure pinhole

Zero distortion, principal point at the sensor centre, the fitted `cc`, pose
from the reference frame via `solvePnP` + `solvePnPRefineLM`, converted with
`calibration_import.calibration_from_opencv`. → `refit_plate_pinhole.py`

**Do not fit distortion from a single plane.** `k1,k2,k3` trade against pose
there, and the solver returns a polynomial that fits those points and diverges
elsewhere. The diagnostic signature is unmistakable: **the projection of a
straight 3D ray doubles back inside the sensor**, which is geometrically
impossible for a pinhole. On Illmenau this made cameras 2 and 4 miss by up to
13 px while their own reprojection RMS sat at 0.5 px.

Before adding any distortion term, prove it is needed:

1. Decompose the residuals into radial and tangential. Distortion is purely
   radial and grows with radius. If the radial component is no larger than the
   tangential and is flat in radius, there is nothing to fit.
2. Check whether `k1` reduces the step-3 spread. On Illmenau the optimum was
   `k1 = −0.02` for a 4 % improvement, within noise, while making reprojection
   RMS worse — so no distortion term was used.
3. Note the radius the plate actually covers. On Illmenau it never exceeded
   r ≈ 1000 px of a 1638 px half-diagonal, so the sensor corners were untested.
   If particles will be imaged there, capture plate frames that reach the
   corners rather than extrapolating a polynomial into them.

### 5. Gate on two checks — never on reprojection RMS

Per-camera reprojection RMS is **blind** to the errors that matter here. It
stayed at 0.5 px through every wrong intermediate result.

**Gate A — triangulate the plate** (`check_plate_triangulation.py`). Expect:
plane normal along the plate normal, planarity residual RMS well under 1 mm,
recovered pitch within a few tenths of a percent, and — the strong one —
**absolute** positions against the known block coordinates with **no alignment
applied**. Illmenau: normal `(0,0,1)`, planarity 0.31 mm, pitch +0.68 % X /
+0.10 % Y, absolute error 1.09 mm median.

**Gate B — epipolar geometry** (`check_epipolar.py`). For every *ordered* pair,
trace each dot's ray from A, project it densely into B, and measure the closest
approach to the same dot as B detected it. Sample densely and keep only samples
landing on the sensor — do **not** approximate the curve by the chord between
two far endpoints, which is what produced a spurious 289 px reading during the
Illmenau work. Also assert the projected ray is monotone (straight) inside the
sensor. Illmenau final: 0.11–0.30 px median, 0.93 px worst, straight for all
points of all 12 pairs.

**Gate C — the whole dataset, and the thing planarity hides**
(`plot_frame_triangulation.py`). Apply the finished model to every frame,
fitting nothing, and read three numbers per frame:

| number | measures | a bad value means |
|---|---|---|
| deviation from the rigid grid (Kabsch-fit the ideal pattern onto the triangulated dots) | did the pattern come back as the pattern | the **labeller** mis-assigned dots |
| ray-convergence miss (RCM) — closest approach of the sight lines of one dot | do the rays actually meet | uses **no** plate model, so the grid cannot fool it: a bad RCM with a good grid is a **calibration** error |
| planarity RMS | is the plate flat | little — a systematic model error distorts a plane consistently and still looks flat |

**Planarity alone will mislead you.** On Illmenau, frame `00000030` has a
textbook grid (1.7 mm median deviation), 0.90 mm planarity and the right pitch —
and a 10.6 mm RCM at 2131 mm from the anchor plane. Nothing is mislabelled; the
rays genuinely do not meet.

Because the extrinsics are solved on one reference frame, the model is exact on
that plane by construction and the error grows linearly away from it. Illmenau
measured **RCM ≈ 0.58 % of the distance from the anchor plane**, consistent
across 19 clean frames. Expect the same shape on any rig calibrated this way,
and quote it as the accuracy floor of the volume.

If that floor is too high, the fix is **not** distortion and **not** `cc` (both
were swept against RCM on Illmenau and neither collapses it; refitting
per-camera intrinsics over all planes even improves reprojection RMS while
making RCM worse). The fix is to stop anchoring to one plane.

### The joint bundle — `openptv2.plate_bundle`

`bundle_plate_poses` solves every camera pose together with one rigid plate pose
per frame. Keep `cc` and the distortion fixed (they are not what is wrong), and
**hold the reference frame's plate pose at identity**. That last point is the
gauge, and it matters twice: the world stays pinned to the physical dot that
defines it, so the calibration block, the datum record and any manual check of
that frame stay valid; and with the gauge fixed there is no free similarity, so
scale cannot drift even though `cc` is not being fitted.

Gate the views *before* the bundle, not during it. A robust loss plus residual
trimming still lets a bad view drag the early iterations — that is what sinks
naive attempts. Three gates, each seeing what the previous cannot:

1. per-camera PnP residual — uses no cross-camera information, so it is a pure
   labelling test for one view;
2. a known plate orientation, if you have one — a grossly mislabelled view
   yields a plate pose tens of degrees off while still fitting its own points;
3. per-dot cross-camera agreement (`plate_bundle.agreeing_views`) — **per dot,
   not per plate centre**: a scramble can leave the centroid roughly in place
   while the pattern around it is wrong.

Then trim on the bundle's own residuals. Resist tightening the gates to chase
reprojection RMS: on Illmenau every tightening improved RMS while making
planarity worse, because it starved the fit.

Illmenau result: RCM went from 0.58 % of distance to **0.126 %** — 18 mm down to
under 5 mm at 3-5 m — planarity at 3-5 m from 3.19 to 1.50 mm, recovered pitch
from +0.75 % to +0.13 %. The price is the reference frame, whose epipolar error
rises from 0.11-0.30 px to 0.7-1.9 px. **That trade is the point:** the anchored
fit was not more accurate, it was concentrating all of its accuracy on one plane.

### Using a known plate orientation

If the plate is held vertical (or any known orientation), measure the departure
before imposing anything — on Illmenau the normal was within 0.83° of horizontal
and the plate's up within 1.23° of world +Y, with the yaw spanning ±30°. Then
use it as a **soft** penalty on the off-yaw rotation components, never a hard
constraint: a hand-held plate really is a degree off, and forcing that to zero
biases a 720 mm plate's corners by ~15 mm.

Be honest about what it buys. On Illmenau the penalty changed nothing measurable
(RCM 2.67 vs 2.71 mm without it) because 44 frames already pinned the poses. Its
real value was elsewhere: as the *outlier gate* above, and as the thing that let
the detector fix its own labelling (see the failure modes below). Keep it for a
sparse dataset, and say plainly that it is inert on a rich one.

## The observation volume and the epipolar display

`algorithms/epi.py:epi_mm` does not compute a line. It computes two *endpoints*,
walking the ray to `Z = Zmin_lay` and `Z = Zmax_lay`, interpolated in X across
`X_lay`. Therefore:

- Changing the box makes the drawn segment appear to move. It has not — you are
  seeing a different piece of the same, unchanged line.
- **`Zmax` must stay well below the nearest camera.** Past the camera plane the
  endpoint projects to absurd coordinates and throws the segment across the
  image. Illmenau cameras sit at Z ≈ 3000; a `±4000` box was broken, `±2500`
  gave a 49 911 px segment, `±1500` was correct.
- If lines look "too short", widen `Zmin/Zmax` — but never past the cameras.
- Keep both layer values of each bound equal so the X interpolation is constant.

Three distinct effects hide behind "the box moved my epipolar line", and only
one is a bug:

1. **Truncation** — the segment covers only the Z range you asked for. Expected.
2. **Chord error** — the GUI draws the straight chord between the two endpoints,
   which equals the true curve only for a pinhole. A non-zero `.addpar` bends
   the curve, worst in the *middle* of a long segment because the endpoints get
   dragged to the image periphery. **This is why shrinking the box can appear to
   fix a calibration** — it hides bad distortion instead of fixing it. With zero
   `.addpar` the miss is box-independent to numerical precision (Illmenau: 0.08
   px at every box from ±500 to ±3000 mm).
3. **Horizon flip** — past the plane through the second camera's projection
   centre the depth changes sign, the endpoint lands on the far side of the
   sensor and the chord is thrown across the image. Solve for it in closed form
   (depth along the ray is affine in Z) rather than guessing: Illmenau's binding
   pair horizons at Z = 3479 mm, so Zmax ≤ ~2780 mm with a 20 % margin, and the
   ±4000 box that started the whole investigation was past it for six of twelve
   pairs. → `check_epipolar_volume.py`

Note this is a *configuration* limit, not a library bug: `epi_mm` reproduces the
C original, whose parity tests deliberately place cameras inside the volume.
Clamping inside `epi_mm` was tried on openptv2 and backed out for that reason.

## Failure modes, and how to tell them apart

**Labelling error vs calibration error.** Fit the rigid plate to each camera's
own labelled points with PnP. That uses no cross-camera information, so a bad
fit means bad labelling. Admit a frame to the fit only when all cameras clear
the gate.

That test is necessary but not sufficient: the labelling failures that matter
are **wrong yet internally self-consistent**, and those fit their own points to
a sub-pixel residual. Three of them cost the Illmenau run half its data, and all
three are avoidable by construction:

1. **Never let the labeller anchor the grid on what it happened to detect.** If
   it shifts indices so the smallest detected one becomes 0, any view missing
   the first column or row relabels the entire plate one step off. Pass the
   coded corner's grid index (`corner_index=` in `label_coded_6x7`) — you have
   to record the datum anyway.
2. **Settle the coded L with the calibration, not the geometry.** Three dots at
   a 1:2 right angle have a second, spurious solution — taking the `1·pitch` dot
   as the corner gives a near-1:2 ratio and a near-right angle too — and
   perspective lets it win, rotating the whole grid 90°. If the plate's
   orientation is known (held vertical, say), project that axis into the image
   with `plate_labeler.image_up_direction` and pass it as `up_hint`. Require
   real agreement, not just the same half-plane: a 90° error sits exactly on
   that boundary.
3. **Never fall back from the coded labeller to the uncoded one.** The uncoded
   path has no origin and no orientation anchor, so it returns a confidently
   labelled grid that can be rotated and offset arbitrarily. If the coded
   detection does not find exactly the expected number of coded dots, that is a
   detection failure — raise. And rather than fixing one `coded_thr`, search for
   the threshold that finds the right number; the count is known a priori.

Together these took Illmenau from 20/48 to 48/48 frames reconstructing
correctly. The general rule: **any labelling decision the image alone cannot
settle must be settled from outside the image — the recorded datum, or the
calibration — and where neither can, fail loudly.**

**180° relabelling.** A grid that maps onto itself under 180° (`id → nx·ny+1−id`
for the row-major convention) lets one camera label the plate the opposite way
round while still fitting its own points perfectly. Detect by computing the
plate pose implied by each camera separately and checking they agree; try both
labellings and keep the consistent one. Checked for on Illmenau — it did not
occur there, but it is invisible to every per-camera metric.

**Wrong handedness in a plot.** Mapping world `(X,Y,Z)` onto matplotlib
`(x,z,y)` swaps two axes and renders a left-handed frame. Reverse one axis limit
(`xlim=(R,-R)`) to fix the view without touching data or ticks.

## Second camera group (e.g. cameras 5–8 opposite)

Cameras on the far side see the **back** face of the plate, so the grid is
mirrored: `+X` of the printed pattern runs right→left in those images. Flip the
axis in the labeller or remap `ix → nx−1−ix`, and **confirm the L code resolves
to the same physical dot** as the first group's datum — otherwise the two halves
do not share a world.

If any frame is seen by all cameras of both groups, use it as the common
reference and everything lands in one frame directly. Otherwise calibrate each
group in its own frame and tie them with a Kabsch fit on shared plate points
(`calibration_import.similarity_from_correspondences`), rotating cameras, never
points: `dm_new = A·dm_old`, `C_new = A·C_old + b`.

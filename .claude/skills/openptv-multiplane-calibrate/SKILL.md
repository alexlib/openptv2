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

Then sweep every frame (`check_all_frames.py`) to characterise accuracy versus
distance from the reference plane, and set the observation volume to the range
where it is acceptable.

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

## Failure modes, and how to tell them apart

**Labelling error vs calibration error.** Fit the rigid plate to each camera's
own labelled points with PnP. That uses no cross-camera information, so a bad
fit means bad labelling. On Illmenau this split 48 frames into 23 good
(planarity 2.70 mm) and 24 mislabelled (26.57 mm) — the whole-dataset median of
17.5 mm was a labeller problem, not a calibration one. Admit a frame to the fit
only when all cameras clear the gate.

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

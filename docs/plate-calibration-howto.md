# From plate images to `.ori` / `.addpar`

A complete walk-through of calibrating a multi-camera openptv2 rig from folders
of coded-dot-plate images, naming every piece of code involved and what it is
for. Worked on the Illmenau barrel (two groups of four cameras, 6×7 coded
plate, ~48 hand-held plate positions each); the numbers quoted throughout are
from that rig.

Companion documents: `docs/illmenau-4cam-calibration.md` records what happened
on that dataset, including the failures; `.claude/skills/openptv-multiplane-calibrate/SKILL.md`
is the condensed procedure.

---

## 0. What you must supply

Three kinds of input. Missing any of them makes the result **silently wrong**
rather than obviously wrong, which is the theme of this whole document.

### Images

One folder per camera, frames synchronised by filename prefix:

```
<ILLMENAU_RAW>/Kalibrierung_<cam>/<frame>_<serial>.tiff
```

The frame number is everything before the first `_`. Every camera of a group
must use the same frame numbering.

### Metadata — `plate.yaml` in the working folder

| field | meaning | why it matters |
|---|---|---|
| `nx`, `ny` | lattice size | a grid that maps onto itself under 180° admits a relabelling no per-camera check can see |
| `pitch_x`, `pitch_y` [mm] | dot spacing | the absolute scale of the entire reconstruction. **Measure it; do not trust the drawing** |
| `thickness` [mm] | front/back plane separation | only needed if a second group views the other face |
| `datum.ix`, `datum.iy` | grid index of the coded L corner | pins the world to a physical dot — see §2 |
| `origin_frame` | which frame defines the world | the gauge |
| sensor `imx`, `imy`, `pix_x`, `pix_y` | in `_config.py` | `pix_x` sets the scale of `cc`, since `cc = fx · pix_x` |

The nominal focal length is **not** an input. It is a bracket for the sweep in
§4, nothing more.

### Physical rig constraints

These are not fitted. They are used to *check* the answer, and in one case to
make the detection work at all:

- **Frame convention**, stated with handedness and drawn before fitting
  anything. Illmenau: `+X` left→right, `+Y` bottom→top, `+Z` object→camera,
  right-handed. Getting this wrong is not visible in any residual.
- **Approximate camera positions.** PnP needs no seed; these exist so you can
  compare fitted positions against the mounts, and pairwise camera distances,
  which are frame-invariant.
- **Plate orientation, if constrained.** The Illmenau plate is held vertical —
  measured: normal within 0.83° of horizontal, plate-up within 1.23° of world
  +Y, yaw spanning ±30°. This is used three ways in §3, §6 and §7.
- **Lab-frame offset** of the datum dot, so results relate to the facility.
  Illmenau: `barrel_from_plate(P) = P + [0, −1175, 0]`.

### Selecting a camera group

Everything is driven by `scripts/illmenau/_config.py`, so a second group of
cameras is a matter of environment rather than editing code:

```bash
export ILLMENAU_RAW=".../Illmenau"                       # holds Kalibrierung_<cam>/
export ILLMENAU_DIR="$ILLMENAU_RAW/openptv_illmenau_5678" # holds cal/, parameters, plate.yaml
export ILLMENAU_CAMS=5,6,7,8                              # physical camera numbers
export OPTV=".../openptv2"
```

`_config.py` cross-checks `ILLMENAU_CAMS` against the camera names in the
folder's `parameters_Run1.yaml` and **refuses to run on a mismatch**. Each
group is a separate world with its own calibration; writing one into the
other's folder destroys it, and that has happened.

---

## 1. The chain, in one picture

```
images ──► detect ──► label ──► object points ──► pose ──► change of basis ──► .ori
             │          │            │              │              │
       detect_plate  plate_labeler  _config     cv2.solvePnP  calibration_import
                                    .obj_of                   .calibration_from_opencv
```

Only two steps in the whole chain fit anything: the pose (§5) and the focal
length (§4). Everything else is bookkeeping and algebra.

---

## 2. Pin the world to a physical dot

**Do not use the plate centre.** Use the coded L-corner dot on the reference
frame — a specific piece of plastic someone can point at. Every other plate
position, every other camera group, and the lab frame are then related to a
thing that exists.

`src/openptv2/plate_labeler.py:label_coded_6x7` finds it without any clicking.
Three dots on the plate are coded (bright centre, dark ring);
`src/openptv2/detect_plate.py` identifies them by comparing a 5×5 centre mean
against an annulus mean. Of the three, the **corner** is the one whose partners
lie at ≈1·pitch and ≈2·pitch at a right angle; the 1·pitch partner defines
`+Y`, the 2·pitch partner `+X`. Every other dot follows from
`ix = round((p − corner) · e_x / pitch)`.

Because the L code gives every dot its identity, **the two GUI steps are not
needed**: no manual-orientation clicking (PnP is direct) and no `sortgrid`
(`autocalibration._refine_and_select(presorted=True)`).

**Verify the datum index from the data:**

```bash
python $OPTV/scripts/illmenau/find_datum.py
```

It labels a view that sees the *complete* lattice with `corner_index=None` —
the one situation where anchoring on the smallest detected index is genuinely
correct — reads off which node the corner occupies, and votes across views. It
recovers Illmenau's known `(2,3)` independently. Put the answer in
`plate.yaml:datum`.

If no view shows the complete lattice (common when the far wall sees the plate
obliquely), use the cross-check instead: label everything with a candidate
index and look at the id histogram over all views. A wrong index pushes part of
the lattice outside the `nx × ny` rectangle, where the labeller silently drops
it, so the right candidate is the one that leaves no id unseen and no lattice
edge under-covered. That is how the Illmenau back face was confirmed to share
the front's `(2,3)`: all 42 ids seen, per-id counts 110/191/193, edges
1309/1326/1147/962 against the front's 106/192/192 and 1303/1316/1149/962.

Then generate the point table openptv2 reads:

```bash
python $OPTV/scripts/illmenau/make_calibration_block.py
```

It writes `cal/calibration_block.txt` from the *same* `_config.obj_of` every
other step uses, so the block cannot disagree with the object points. Point id
convention is `id = iy·nx + ix + 1`, and the datum dot sits at the origin — for
Illmenau that is **id 21**, line `21 0.0 0.0 0.0`.

---

## 3. Detect and label every frame, once

```bash
python $OPTV/scripts/illmenau/detect_plate_frames.py --cams 5,6,7,8
```

Writes `cal/labelled_all_frames.npz` as `{(camera index, frame): (ids, pixels)}`.
Everything downstream reads this cache. Detection is the slow part (~200 images);
caching means a `cc` sweep costs seconds.

**One labelling, produced here, used by everything.** Two scripts used to
re-detect and re-label the reference frame themselves; on the near wall their
private labelling agreed with the cache by luck, and on the far wall it did
not — putting cameras metres from their mounts and reporting 100–400 px
epipolar misses for a calibration that was fine. A check that re-derives its
own ground truth can fail in exactly the way it exists to detect.

Three pieces of knowledge are handed to the labeller that one image cannot
settle, and each of them fixed a bug that was invisible to per-camera residuals:

**`corner_index=(ix, iy)`** — without it the grid is anchored to the smallest
*detected* index, so any view missing the leftmost column or bottom row labels
every dot one step off. Wrong, yet perfectly self-consistent: a per-camera PnP
fit lands sub-pixel. Regression tests in `tests/unit/test_plate_labeler_anchor.py`.

**`up_hint`** — the image direction of a known world axis, from
`plate_labeler.image_up_direction(cal, cpar, pixel)`. The three coded dots have
a second, spurious solution: taking the 1·pitch dot as the corner also gives a
near-1:2 leg ratio at a near-right angle, and perspective lets it win, rotating
the whole grid 90°. The plate is held vertical, so its own +Y is world +Y, and
the calibration knows where that points in each image. **This is what "use the
calibration in detection" means concretely.** The tolerance must be a real angle
(60° default) — requiring merely the same half-plane fails, because a 90° error
sits exactly on that boundary.

**An adaptive coded threshold** — the plate has exactly three coded dots, so the
driver searches `coded_thr ∈ (30, 25, 20, 15, 10)` for the one that finds three,
rather than fixing one. With a fixed 30, two views found *zero*, and
`label_plate` used to fall through to `label_uncoded_grid`, which has no origin
and no orientation anchor and returns a confidently labelled grid rotated 90°.
It now raises instead.

The first run on a new rig has no `.ori` for the up-hint. Run it once with
`ILLMENAU_NO_HINT=1`, then re-run after §5 — a better calibration rescues frames
the labeller previously got wrong. On Illmenau that loop took the usable set from
**20/48 to 48/48**.

---

## 4. Fit the one shared focal length

```bash
uv run --with opencv-python-headless python $OPTV/scripts/illmenau/fit_plate_cc.py
```

**`cc` is exactly degenerate on a single plane.** Re-fit the pose at a different
focal length and you get a self-similar reconstruction: identical residuals,
identical recovered pitch. Sweeping 9.20→9.75 mm moved the triangulated pitch by
0.02 mm. Any "fit" of `cc` from one plane is fitting noise.

It *is* observable from many planes. Fit each camera's pose on the reference
frame, then ask each camera **separately** where the plate is in some other
frame. Right `cc` → the answers coincide; wrong `cc` → each camera's world sits
at the wrong distance and they spread apart, worse the further from the
reference plane. Minimise the median spread:

| cc [mm] | 8.20 | 8.40 | **8.586** | 8.80 | 9.00 | 9.44 |
|---|---|---|---|---|---|---|
| cross-camera spread [mm] | 84 | 62 | **33.6** | 41 | 70 | 125 |

Illmenau cams 1–4: **8.5858 mm** against a nominal 9.44. Cams 5–8, fitted
independently: **8.6313 mm** — agreeing to 0.5 %, which is corroboration since
nothing constrained the two groups to agree.

Use **one shared `cc`** when the cameras share a lens; it is far better
conditioned than four independent ones. Expect a clean single minimum. A flat
curve means the frames do not span enough depth — capture more.

---

## 5. Write the `.ori`

```bash
uv run --with opencv-python-headless python $OPTV/scripts/illmenau/refit_plate_pinhole.py 8.5858
```

Object points from §2, pixels from the §3 cache, `cv2.solvePnP` +
`solvePnPRefineLM` for the pose, then a change of basis. `.addpar` is written as
**zeros** — see §8.

The conversion is pure algebra, in `src/openptv2/calibration_import.py:calibration_from_opencv`:

| OpenCV | openptv2 `.ori` | why |
|---|---|---|
| `R`, `t` | `dm = Rᵀ · S`, `S = diag(1, −1, −1)` applied **on the right** | OpenCV's camera frame is x-right, y-**down**, z-**forward**; openptv2's is x-right, y-**up**, z-**backward** (the camera views along `−dm[:,2]`) |
| `R`, `t` | `C = −Rᵀ·t` | projection centre in world coordinates — the `x0 y0 z0` line |
| `dm` | `ω, φ, κ` via `angles_from_dm` | the file stores angles; `dm` is written under them and is what the code reads |
| `fx` | `cc = fx · pix_x` | pixels → mm |
| `cx, cy` | `xh, yh` | principal point, offset from the sensor centre |
| `distCoeffs` | `.addpar` `k1 k2 k3 p1 p2 scx she` | left at zero |

`Calibration.to_file` writes `cal/cam<N>.tif.ori` and `.addpar`. A reprojection
RMS much above ~1 px here means the reference frame is mislabelled for that
camera, not that the pose is hard to fit.

---

## 6. Spread the error over the volume — the joint bundle

```bash
uv run --with opencv-python-headless python $OPTV/scripts/illmenau/bundle_plate_poses.py 8.5858 --write
```

Solving every pose on one plane makes the model exact **there by construction**
and lets its error grow linearly away. Measured on Illmenau: ray-convergence
miss ≈ **0.58 % of the distance from the anchor plane** — sub-millimetre at the
plane, ~18 mm at 3–4 m — while per-camera reprojection RMS sat at 0.5 px
throughout.

The solver is `src/openptv2/plate_bundle.py`, deliberately cv2-free. Unknowns:
every camera pose, plus one rigid plate pose per frame. Held fixed on purpose:
`cc`, the distortion, the principal point, and **the reference frame's plate
pose at identity**. That last one is the gauge, and it matters twice — the world
stays pinned to the datum dot so the calibration block and any manual check stay
valid, and with it fixed there is no free similarity, so scale cannot drift even
though `cc` is not being fitted.

**Reject outliers before the fit, not during it.** A robust loss plus residual
trimming still lets a bad view drag the early iterations; that is what sinks
naive attempts. Three gates, each catching what the previous cannot:

1. **Per-camera PnP < 1 px** — uses no cross-camera information, so it is a pure
   labelling test for one view.
2. **Plate vertical within 5°** — a grossly mislabelled view yields a plate pose
   tens of degrees off while still fitting its own points. Catches views a
   residual cannot.
3. **Per-dot cross-camera agreement < 100 mm** — per **dot**, not per plate
   centre: a scramble can leave the centroid roughly in place while the pattern
   around it is wrong.

Then six rounds of trimming at 3× the median residual. Resist tightening these
to chase reprojection RMS: every tightening improved RMS while making planarity
**worse**, because it starved the fit.

A known plate orientation can also enter as a **soft** penalty on the off-yaw
rotation components (`BUNDLE_VERT_PX`, default 10 px per 1°). Soft, never hard —
a hand-held plate really is ~1° off vertical, and forcing that to zero biases a
720 mm plate's corners by ~15 mm. Be honest about what it buys: on Illmenau it
changed nothing measurable (RCM 2.67 vs 2.71 mm disabled), because 44 frames
already pinned the poses. Its value was as the gate above, and on a sparser
dataset.

| | anchored to one plane | joint bundle |
|---|---|---|
| RCM vs distance | 0.58 % | **0.126 %** |
| RCM at 3–5 m | 18.0 mm | **4.75 mm** |
| planarity at 3–5 m | 3.19 mm | **1.39 mm** |
| recovered X pitch | +0.75 % | **+0.24 %** |
| reference frame epipolar | **0.11–0.30 px** | 0.58–1.45 px |

3.7× better ray convergence across the volume, paid for with some of the
reference frame's perfection. **That trade is the point:** the anchored fit was
not more accurate, it was concentrating all of its accuracy on one plane.

---

## 7. Accept or reject — and never on reprojection RMS

Per-camera reprojection RMS is **blind** to everything that matters here. It
stayed at 0.5 px through unphysical distortion, a focal length wrong by 10 %,
poses anchored to one plane, and half the frames mislabelled. It is even
*anti-correlated*: refitting per-camera intrinsics over all planes improves RMS
to 0.38–0.44 px while making ray convergence worse.

Three numbers that do work:

| number | measures | a bad value means |
|---|---|---|
| **ray-convergence miss (RCM)** — closest approach of one dot's sight lines | do the rays actually meet | uses **no** plate model, so the grid cannot fool it: bad RCM + good grid = **calibration** error |
| **deviation from the rigid grid** — Kabsch-fit the ideal pattern onto the triangulated dots | did the pattern come back as the pattern | the **labeller** mis-assigned dots |
| planarity RMS | is the plate flat | little — a systematic model error distorts a plane consistently and still looks flat |

Planarity alone misleads. Illmenau frame 30: textbook grid (1.7 mm median
deviation), 0.90 mm planarity, correct pitch — and a 10.6 mm RCM at 2131 mm.
Nothing mislabelled; the rays genuinely did not meet.

```bash
python $OPTV/scripts/illmenau/check_plate_triangulation.py   # gate A
python $OPTV/scripts/illmenau/check_epipolar.py              # gate B
python $OPTV/scripts/illmenau/plot_frame_triangulation.py    # every frame, as pictures
python $OPTV/scripts/illmenau/check_all_frames.py            # accuracy vs depth
```

**Gate A** — triangulate the reference plate: plane normal along the plate
normal, planarity well under 1 mm, pitch within a few tenths of a percent, and
the strong one: **absolute** positions against the known block coordinates with
**no alignment applied**. Illmenau: normal (0,0,1), planarity 0.34 mm, absolute
error 1.35 mm median.

**Gate B** — for every *ordered* pair, trace each dot's ray from A, project it
densely into B, and measure the closest approach to the same dot as B detected
it. Sample densely and keep only samples on the sensor; do **not** approximate
the curve by the chord between two far endpoints, which produced a spurious
289 px reading. Also assert the projected ray is **monotone** inside the sensor:
a straight 3D ray that doubles back means the distortion model is unphysical,
which no miss distance reveals.

**The pictures** — `plot_frame_triangulation.py` fits nothing; it applies the
`.ori` on disk to every frame and writes a 3D panel (dots coloured by distance
from the best-fit plane, cameras, sight lines) and a face-on panel (dots against
the ideal rigid grid, with ids). A mislabelled dot is obvious there and nowhere
else.

---

## 8. Distortion: prove it before you fit it

**Do not fit distortion from a single plane.** `k1,k2,k3` trade against pose and
the solver returns a polynomial that fits those 42 points and diverges
everywhere else. The signature is unmistakable: **the projection of a straight
3D ray doubles back inside the sensor**, which a pinhole cannot do. On Illmenau
this made cameras 2 and 4 miss by up to 13 px while their reprojection RMS sat
at 0.5 px.

Before adding any term:

1. Decompose the residuals into radial and tangential. Distortion is purely
   radial and grows with radius. Illmenau: radial 0.27–0.35 px, *smaller* than
   tangential 0.44–0.46 px, and flat in radius — nothing to fit.
2. Check whether `k1` reduces the §4 spread. Illmenau: best `k1 = −0.02` for a
   4 % improvement, within noise, while worsening reprojection RMS.
3. Note the radius the plate actually covers. Illmenau never exceeded r ≈ 1000 px
   of a 1638 px half-diagonal, so the sensor corners are untested. If particles
   will be imaged there, capture plate frames that reach them rather than
   extrapolating a polynomial into them.

The delivered `.addpar` are zeros.

---

## 9. Set the observation volume before opening the GUI

```bash
python $OPTV/scripts/illmenau/check_epipolar_volume.py
```

`src/openptv2/algorithms/epi.py:epi_mm` does **not** compute a line. It computes
two *endpoints*, walking the ray to `Z = Zmin_lay` and `Z = Zmax_lay`
(interpolated in X across `X_lay`), and the GUI draws the straight chord between
them. So the box changes what you see in three ways, only one of which is a bug:

- **Truncation** — the segment covers only the Z range you asked for. If the true
  depth is outside it the segment stops short. Expected; widen the box.
- **Chord error** — the chord equals the true curve only if the curve is
  straight. A non-zero `.addpar` bends it, worst in the *middle* of a long
  segment, because the endpoints get dragged to the image periphery where the
  polynomial is largest. **This is why shrinking the box can appear to fix a
  calibration** — it hides bad distortion. With zero `.addpar` the miss is
  box-independent to 0.08 px from ±500 to ±3000 mm.
- **Horizon flip** — past the plane through the second camera's projection
  centre, the depth changes sign and the endpoint lands on the far side of the
  sensor, throwing the chord across the image. Nothing is wrong with the
  calibration; the box asked for a point behind the camera.

The script solves the horizon in closed form. Illmenau's binding pair is at
Z = 3479 mm, so `Zmax_lay` ≤ ~2780 mm with a 20 % margin. The ±4000 box that
started the original "epipolar lines are wrong" report was past the horizon for
six of twelve pairs. Keep both layer values of each bound equal so the X
interpolation stays constant.

*Not fixed in the library:* clamping the endpoint to the horizon inside `epi_mm`
was implemented and **backed out** — it changes results for configurations
`test_epi_mm` and `test_epi_mm_perpendicular` deliberately encode. The limit is
reported by the diagnostic instead of enforced by the library.

---

## 10. Sanity-check against the physical rig

```bash
python $OPTV/scripts/illmenau/draw_rig_global.py
```

Draws the frame, the test section, the plate and both nominal and fitted camera
positions. Compare fitted positions against the mounts, and pairwise camera
distances, which are frame-invariant. Illmenau cams 5–8 came out at
X ≈ ±1450–1460, Z ≈ −2990…−3040, mirroring cams 1–4 at Z ≈ +2960…+3055 — not
constrained to, so it is a real check.

---

## 11. A second camera group

Each group is calibrated **standalone**, in its own working folder, anchored to
its own reference frame's datum dot. Two separate worlds.

Whether they can ever be merged depends on the physics, not the software. On
Illmenau the two groups measure **different halves of the circular test
section**, so the plate never visited the same positions twice and there is no
shared observation to register against. Any apparent correspondence between the
two sets of plate positions is the mirror symmetry of the *placement procedure*,
not real point pairs — which is a convincing false signal, so beware of it.

If the groups do share a volume, note that plate poses alone cannot determine
the transform when the plate is always held vertical: every rotation is a yaw
about the same axis, which leaves the translation rank-deficient along it
(measured: all 40 rotation axes within 7.1° of each other; the system's
singular values 6.62 / 2.05 / **0.05**, worst direction exactly vertical). Tie
the groups through the lab frame, or through a target both groups can see.

---

## Script reference

| script | role |
|---|---|
| `_config.py` | camera group, paths, plate geometry, datum; refuses group/folder mismatches |
| `find_datum.py` | reads the coded L corner's grid index off the data |
| `make_calibration_block.py` | writes `cal/calibration_block.txt` from the same `obj_of` |
| `detect_plate_frames.py` | detect + label every frame once, cache to npz |
| `fit_plate_cc.py` | the one shared focal length, from multi-plane consistency |
| `refit_plate_pinhole.py` | poses on the reference frame → `.ori` + zeroed `.addpar` |
| `bundle_plate_poses.py` | joint bundle over all plate positions, `cc` fixed |
| `check_plate_triangulation.py` | gate A — plane, pitch, absolute positions |
| `check_epipolar.py` | gate B — epipolar curve vs the corresponding dot |
| `plot_frame_triangulation.py` | one figure per frame; fits nothing |
| `check_all_frames.py` | accuracy vs distance, RCM-gated |
| `check_epipolar_volume.py` | the safe observation volume, in closed form |
| `draw_rig_global.py` | fitted vs nominal rig geometry |
| `match_plate_positions.py` | pairs two groups' frames by plate position — see §11 for its limits |

| library module | role |
|---|---|
| `detect_plate.py` | plate ROI, dot detection, coded-dot classification |
| `plate_labeler.py` | L-coded and uncoded labelling; `image_up_direction` |
| `calibration_import.py` | OpenCV ↔ `.ori` conversion, both directions |
| `plate_bundle.py` | the joint bundle, cv2-free |
| `plate_calibration.py` | multi-plane OpenCV solver (`method='opencv'` / `'dlt'`) |
| `calibration_seed.py` | `rig.yaml` → seed `.ori` |
| `calibration_registry.py` | named calibration sets |
| `autocalibration.py` | `presorted=True` skips `sortgrid` |

Tests: `tests/unit/test_plate_labeler_anchor.py` (labelling anchor + up-hint
contract), `tests/unit/test_plate_bundle.py` (bundle on a synthetic rig with
known truth), `tests/unit/test_plate_detection.py`,
`tests/unit/test_calibration_import.py`.

---

## The one-paragraph version

The plate's own geometry gives object points; one frame is *declared* to be the
world, anchored on a coded dot whose grid index you verified; `solvePnP` turns
each camera's view of that frame into a pose; a change of basis turns the pose
into an `.ori`; the focal length — the only quantity a single plane cannot see —
comes from requiring the cameras to agree about *other* plate positions; and a
joint bundle then spreads the remaining error over the measurement volume
instead of concentrating it on one plane. Gate the result on ray convergence and
epipolar geometry, never on reprojection RMS.

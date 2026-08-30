# Calibration hub: multiple sources/algorithms -> `.ori`/`.addpar`

**Status:** implementation plan, ready to execute. Supersedes the
2026-08-29/30 draft of this file (phased-roadmap framing), which is folded in
as Step 8 and Appendix C.

---

## Context

openptv2 today calibrates exactly one way: a surveyed 3D calibration body
(`calblock.txt`) plus a 4-point manual click per camera, then
`external_calibration` -> `sortgrid` -> `full_calibration`
(`src/openptv2/autocalibration.py:calibrate_camera`).

Users arrive from DaVis/LaVision, easyWand, proPTV, MyPTV and plain OpenCV
already holding a calibration. openptv2's model is the only *physical* one in
that set -- camera at real mm coordinates, real focal length, explicit
refraction through the glass -- which is what makes x,y,z measurable in
millimetres and multimedia ray tracing possible. So openptv2 should be the
**backbone** any of those can feed into, not a competing ecosystem.

### The governing idea

A calibration is two separable things:

1. **The measurements** -- "this lab XYZ appears at this pixel". Pure data.
   *Universal across every package.*
2. **The model** -- how a camera turns 3D into pixels. Tsai, Soloff cubic,
   OpenCV Brown, DLT-11, openptv2's pinhole+Brown+refraction. *Every package
   parameterises this differently; there is no 1:1 mapping.*

**Always land on (1), then let openptv2 fit its own (2).**

### The four doors

All funnel into the existing `CalibrationPointSet` -> `_refine_and_select` ->
`.ori`/`.addpar`:

```
 A. points file      (MyPTV, proPTV, Multiview-Cal, DaVis) --+
 B. pinhole model    (OpenCV, COLMAP, Metashape, DLT-11)   --+   (XYZ,uv)     openptv2
 C. polynomial model (proPTV Soloff, DaVis 3rd-order)      --+-> pairs   --> bundle  --> .ori/.addpar
 D. images           (calblock, checkerboard, multiplane)  --+   + seed       adjust
```

| door | input | how |
|---|---|---|
| **A. points** | MyPTV `camN_cal_points`, proPTV `markers_cN.txt`, Multiview-Cal `cN_xyXYZ.txt`, DaVis plate export | **all four are the same five columns `x y X Y Z`**, differing only in separator and header. **One reader covers four packages.** Primary door, full accuracy. |
| **B. pinhole model** | OpenCV `(K,dist,rvec,tvec)`, COLMAP, Metashape, DLT-11 | algebraic convert -> seed `Calibration`. Numpy only. The first three are *the same model* up to the p-swap, the half-pixel origin, `R` vs `R.T` and mm-vs-normalised scaling: **one converter with three thin front-ends**, not three converters. |
| **C. polynomial model** | proPTV Soloff, DaVis 3rd-order, MyPTV extendedZolof | no exterior orientation exists to convert at all; *evaluate* the foreign model on a grid over the measurement volume -> regenerates door A. Only valid inside the fitted volume -- a cubic diverges fast outside it. |
| **D. images** | calblock, checkerboard, multiplane, wand | the old plan's Phases 2 and 4; the only doors needing cv2. |

**First target: the Ilmenau barrel dataset** (`../Multiview-Calibration`),
with the fewest conversions and the least error. It is the ideal first case
because it is a convection cell **in air** (`mmp_n1=n2=n3=1.0, mmp_d=1.0,
interf: 0`), so OpenCV's model and openptv2's are the *same* model
(pinhole + Brown) and the conversion is exact rather than approximate.

---

## The optics layer (policy)

**Reuse `ptv.par` / YAML `mmp_n1..n3`, `mmp_d`. Invent no new config.**
`ControlPar.from_file` (`algorithms/parameters.py:637`) and `from_yaml`
(`:693`) already read them, and `test_data/burgers/parameters/ptv.par` already
ships the `1/1/1/1` no-glass case.

- **Default `n1=n2=n3=1, d=1`.** openptv2 then reproduces the foreign model
  exactly. The glass vector defaults to a unit vector along the camera axis;
  it is numerically inert while `n1 == n2 == n3` (verified), but must be
  non-zero unless you want the explicit pinhole short-circuit at
  `imgcoord.py:196`.
- **Real optics only together with a re-fit on points.** A calibration shot
  through water has **already absorbed refraction into its distortion
  coefficients**. Importing it and then switching `n3=1.33` on
  **double-counts refraction**. This is the loudest warning in this document.

The upgrade path is the actual value proposition: import points from anywhere,
declare the real optics, re-run openptv2's bundle adjustment -> the distortion
coefficients collapse to genuine lens distortion, the camera lands at a
physically measurable mm position, and rays trace correctly through the
interface.

---

## Verified facts (measured 2026-08-30 -- do not re-derive)

Measured numerically against `img_coord` + `metric_to_pixel` on a synthetic
camera, **and** independently re-derived from the OpenCV/Brown source
definitions. Both routes agree.

### OpenCV -> openptv2, exact to 1e-12 px when `xh=yh=0`

```python
S    = np.diag([1.0, -1.0, -1.0])       # OpenCV cam frame -> photogrammetric cam frame
R_cv = Rotation.from_rotvec(rvec).as_matrix()      # world -> camera
C    = -R_cv.T @ tvec                   # camera centre -> ext.x0, y0, z0   [mm]
dm   = R_cv.T @ S                       # NOTE: S on the RIGHT

cc    = fx * pix_x                      # [mm]
pix_y = cc / fy                         # anisotropy MUST go here, not into scx
xh    = (cx - imx/2.0) * pix_x          # [mm]
yh    = (imy/2.0 - cy) * pix_y          # [mm]  <- sign flip: openptv2 y is UP

k1 = k1_cv / cc**2                      # OpenCV r is normalised, openptv2 r is mm
k2 = k2_cv / cc**4
k3 = k3_cv / cc**6
p1 =  p2_cv / cc                        # SWAPPED
p2 = -p1_cv / cc                        # SWAPPED *and* sign-flipped (the y inversion)
scx, she = 1.0, 0.0
```

`dm = Rx(omega) @ Ry(phi) @ Rz(kappa)` (from `calibration.py:44
compute_rotation_matrix`), so:

```python
phi   = np.arcsin(np.clip(dm[0, 2], -1.0, 1.0))
kappa = np.arctan2(-dm[0, 1], dm[0, 0])
omega = np.arctan2(-dm[1, 2], dm[2, 2])
```

equivalently `scipy Rotation.from_matrix(dm).as_euler('XYZ')` (capital =
intrinsic). Lowercase `'xyz'` is extrinsic and gives the WRONG answer -- the
`../Multiview-Calibration` prototype uses lowercase, **and** puts `S` on the
left; both are wrong. Do not copy it.

### Frame change is absorbable into the exterior (3e-12 px)

```python
dm_new = A @ dm_old                     # A: rotation from old lab frame to new
C_new  = A @ C_old + b
```

Never transform the points; rotate the cameras. `A`, `b` from a Kabsch fit
(`np.linalg.svd`, ~8 lines) on >=3 known correspondences. If a scale `s` is
also needed, scale positions only -- `cc` is a sensor quantity and is
unaffected.

### The one true residual

openptv2 centres distortion on the **image centre**
(`imgcoord.py:61 _flat_to_dist_core` distorts `flat + xh,yh`); OpenCV centres
it on the **principal point**. Measured, on-sensor points only:

| principal-point offset | distortion | max | RMS |
|---|---|---|---|
| 0 px | any | 0.0000 px | 0.0000 px |
| 5, 3 px | mild | 0.0374 px | 0.0186 px |
| 17, 11 px | mild | 0.1240 px | 0.0633 px |
| 5, 3 px | 10x | 0.3549 px | 0.1795 px |
| 17, 11 px | 10x | 1.2366 px | 0.6168 px |

This is why door B yields a **seed**, not a final answer: one
`full_calibration` pass absorbs it.

### Pixel pitch is a gauge freedom

OpenCV fixes only `fx = cc/pix_x`, `fy = cc/pix_y`. You may *choose* `pix_x`
and set `cc = fx*pix_x`; the projection is bit-identical and only the printed
focal length is fictional. World-space quantities are unaffected -- their
scale comes from the calibration object, not the sensor. The real pitch
matters only if you want `cc` to be physically meaningful. Ilmenau has it
anyway (0.005 mm).

### Open question -- resolve with data, do not assume

**A possible 0.5 px origin offset.** OpenCV puts integer coordinates at pixel
**centres** (centred principal point = `(w-1)/2`); COLMAP/Metashape at the
top-left **corner** (`w/2`). openptv2's `metric_to_pixel` uses `imx/2`, which
looks like the corner convention, but what actually matters is the convention
of the detector that produced the target coordinates. It appears as a
**constant 0.5 px bias in the residual field**. Step 7 measures it.

---

## Traps in the existing code (each verified)

1. **`raw_orient` zeroes all distortion** before solving
   (`algorithms/orientation.py:540-546` sets `k1=k2=k3=p1=p2=0, scx=1,
   she=0`). `external_calibration` is a thin wrapper over it. So
   `calibrate_from_source`'s path 2 (`initial_cal` + `fix4`/`pix4`)
   **destroys an imported distortion model**. Door B must always use path 1
   (`point_set.seed`).
2. **`_refine_and_select` always calls `sortgrid`**
   (`autocalibration.py:383, 390, 397`). Imported point sets are *already
   matched*; sortgrid re-matches by proximity within `eps` px and drops
   non-matches to `pnr=-999`. Needs a bypass.
3. **OpenCV is not a dependency** -- absent from `pyproject.toml` (core,
   `gui`, `rembg`, `viz`, `dev`) *and* from `uv.lock`. The 2026-08-29 draft of
   this document claimed otherwise. Doors A/B/C need **zero** cv2; do not add
   it.
4. **Setting `Exterior.dm` directly is silently discarded.** Demonstrated:
   assigning a rotation matrix to `ext.dm` then building a `Calibration`
   projected **8e7 px** off, because `dm` was recomputed from the still-zero
   angles. Going through angles + `compute_rotation_matrix()` gave 3e-12 px on
   identical geometry. **Always convert R -> angles first, never write `dm`.**
5. `.ori` stores angles *and* `dm`, and the reader trusts the file's `dm`
   (`calibration.py:428`) -- the mirror of trap 4 on the read side. Also the
   forward model is hand-inlined three times in `imgcoord.py`
   (`_flat_image_coord_core`, `_img_coord_batch_impl`,
   `_flat_image_coord_batch_impl`), so round-trip tests must exercise
   `img_coord` **and** `img_coord_batch`.

---

## Step 0 -- verify the unrun Phase 1 code (blocking)

`src/openptv2/calibration_registry.py`, the `_refine_and_select` extraction in
`autocalibration.py`, and `tests/unit/test_calibration_registry.py` were
written 2026-08-29/30 and **never run**.

```bash
uv run pytest tests/unit/test_autocalibration.py tests/unit/test_calibration_registry.py -v
uv run ruff check .
```

`test_autocalibrate_cavity` must still pass -- that is what proves the
`_refine_and_select` extraction was behaviour-preserving. Fix anything that
surfaces before touching anything else. **Do not proceed on a red suite.**

---

## Step 1 -- `src/openptv2/calibration_import.py` (new, numpy/scipy only)

No cv2. `scipy.spatial.transform.Rotation` is fine (scipy is already a core
dependency).

### 1.1 Angle/matrix helpers

```python
def angles_from_dm(dm: np.ndarray) -> tuple[float, float, float]:
    """Invert Exterior.compute_rotation_matrix: dm = Rx(omega)Ry(phi)Rz(kappa)."""
```

Use the closed form above. Raise `ValueError` when `abs(dm[0,2]) > 1 - 1e-9`
(gimbal lock: `cos(phi) -> 0`, omega/kappa degenerate) rather than returning
silently-wrong angles.

`scripts/calibrate_proptv_dlt.py:118 rotation_matrix_to_angles` is the same
function; leave that script alone (it works and is self-tested), but add a
comment there pointing at this module as the canonical copy.

```python
def exterior_from_rotation(C, dm) -> Exterior:
    """Build an Exterior from a camera centre and a camera->world matrix.

    Goes through omega/phi/kappa + compute_rotation_matrix() -- writing `dm`
    directly is silently discarded (trap 4). Asserts the round-trip.
    """
```

Must `assert np.abs(ext.dm - dm).max() < 1e-12` before returning.

### 1.2 The converter

```python
def calibration_from_opencv(
    K, dist, rvec, tvec, *,
    imx: int, imy: int, pix_x: float, pix_y: float | None = None,
    glass_vec=(0.0, 0.0, 1.0), pixel_origin: str = "corner",
) -> tuple[Calibration, float]:
    """Convert an OpenCV calibration to openptv2's .ori/.addpar model.

    Returns (Calibration, pix_y_used). `pix_y` is derived as cc/fy when not
    given, because openptv2's `scx` is isotropic and cannot carry fx != fy.
    `pixel_origin`: "corner" (no shift) or "centre" (+0.5 px on cx, cy) --
    see the open question above.
    """
```

Body follows the verified block exactly. `dist` is OpenCV order
`(k1, k2, p1, p2[, k3[, ...]])` -- accept length 4, 5, 8, 12 or 14, use the
first five, and **raise** if any of `k4..k6, s1..s4, taux, tauy` is non-zero,
naming which (openptv2 has no representation for them; the caller must use
door C and resample instead).

Glass: `Glass(vec_x, vec_y, vec_z, n1=1.0, n2=1.0, n3=1.0, d=1.0)`. Note the
refractive indices on `Glass` are stored-for-reference only -- the imaging
chain reads them from `cpar.mm`. Use the glass-vector sign that puts the
interface between camera and volume.

### 1.3 The inverse (needed for the round-trip test)

```python
def opencv_from_calibration(cal, *, imx, imy, pix_x, pix_y):
    """Inverse of calibration_from_opencv. Returns (K, dist, rvec, tvec)."""
```

### 1.4 Readers

```python
def read_xyXYZ(path) -> tuple[np.ndarray, np.ndarray]:
    """Read the universal 5-column point file -> (img_pts (n,2), ref_pts (n,3)).

    Covers proPTV `markers_cN.txt`, MyPTV `camN_cal_points`,
    Multiview-Calibration `cN_xyXYZ.txt` and DaVis plate exports -- all the
    same `x y X Y Z`, differing only in separator and header.
    """
```

Skip `#` comment lines; accept whitespace **or** comma separation
(`np.genfromtxt` with `comments="#"`, retry with `delimiter=","`). Tolerate an
optional 6th column (MyPTV PR#67's view index) and return it when present.
Raise on <5 columns, naming the file and the line.

```python
def read_opencv_flat15(path) -> dict:
    """Read the Ilmenau `calib_cN.txt`: 15 floats, one per line.

    Layout (from Multiview-Calibration's own extract_calibration.py):
        [0:3]  rvec        [3:6]  tvec
        [6] fx  [7] fy  [8] cx  [9] cy
        [10:15] dist = k1, k2, p1, p2, k3
    """
```

### 1.5 Kabsch (only if Step 7 shows a frame mismatch -- otherwise skip, YAGNI)

```python
def similarity_from_correspondences(src, dst, *, with_scale=False):
    """Kabsch/Procrustes: returns (A, b, s) with dst ~= s * A @ src + b."""
```

Apply via `dm_new = A @ dm_old`, `C_new = A @ C_old + b` (verified).

---

## Step 2 -- `presorted` bypass in `_refine_and_select`

`src/openptv2/autocalibration.py:363`. Add a keyword-only parameter:

```python
def _refine_and_select(cam, cal, cpar, fix, nfix, eps, pix, *, presorted: bool = False) -> CamResult:
```

When `presorted=True`, skip **all three** `sortgrid` calls and the coarse
pre-pass; set `sorted_pix = pix` and `n_matched = len(pix)`, then go straight
to the `CANDIDATE_FLAGS` loop. `pix` must already be index-aligned with `fix`
-- that is what `_target_from_xy(i, ...)` guarantees, since it sets `pnr=i`.

Default `False`, so `calibrate_camera` is unchanged and
`test_autocalibrate_cavity` stays byte-for-byte identical.

Thread the flag through `calibrate_from_source` (`autocalibration.py:489`) as
a keyword-only `presorted: bool = False`.

Also: strengthen `calibrate_from_source`'s docstring on path 2 to state that
`external_calibration` **wipes `k1..she`** (trap 1), so any source carrying a
converted distortion model must use path 1 (`point_set.seed`).

---

## Step 3 -- register the new sources

In `src/openptv2/calibration_registry.py`, add two `CalibrationSourceInfo`
entries alongside `calibration_object`:

- `name="opencv_model"`, `produces_seed=True` -- requires K, dist, rvec, tvec,
  image size and pixel pitch; `best_for` an existing OpenCV/COLMAP/Metashape
  calibration; `avoid_when` the foreign model used rational/thin-prism/tilt
  terms.
- `name="points_file"`, `produces_seed=False` -- requires a 5-column
  `x y X Y Z` file per camera plus a seed pose from elsewhere; `best_for`
  proPTV / MyPTV / DaVis / Multiview-Calibration exports.

Metadata only, mirroring the existing entry's style. Keep the module a plain
registry -- no dynamic dispatch until there is more than one implementation to
dispatch between.

---

## Step 4 -- tests

### `tests/unit/test_calibration_import.py` (new)

Follow the style of `tests/unit/test_synthetic_calibration.py`; reuse its
`make_calibration` / `make_control_par` helpers where they fit. Fast and
file-I/O-free except the reader tests.

| test | assertion |
|---|---|
| `test_angles_roundtrip` | random rotations: `dm -> angles -> dm` within 1e-12 |
| `test_angles_gimbal_raises` | `phi = pi/2` raises `ValueError` |
| `test_exterior_from_rotation_survives` | the built `Exterior.dm` matches the input within 1e-12 (regression guard for trap 4) |
| `test_opencv_projection_matches_img_coord` | build a `Calibration` with `xh=yh=0` and full distortion; project ~60 points both with `img_coord`+`metric_to_pixel` and with a numpy transcription of OpenCV's model using `opencv_from_calibration`'s output; **max diff < 1e-9 px** |
| `test_opencv_roundtrip` | `calibration_from_opencv(*opencv_from_calibration(cal))` reproduces `cal`'s projections within 1e-9 px |
| `test_distortion_centre_gap_bounded` | with `xh,yh` = 17,11 px and strong distortion, on-sensor max diff is **> 0.1 px and < 2 px** -- bounds the known limitation in both directions, so a change that silently widens it fails |
| `test_batch_matches_scalar` | `img_coord_batch` agrees with `img_coord` on the imported calibration (trap 5) |
| `test_frame_change_absorbed` | random `A`, `b`: `dm_new = A@dm`, `C_new = A@C+b` reproduces projections within 1e-9 px |
| `test_read_xyXYZ_variants` | whitespace, comma, `#` header, optional 6th column; a `<5`-column file raises naming the path |
| `test_read_opencv_flat15` | a 15-line fixture unpacks to the documented slots |
| `test_rejects_unsupported_distortion` | an 8-element `dist` with non-zero `k4` raises, naming `k4` |

### `tests/unit/test_calibration_registry.py` (add)

| test | assertion |
|---|---|
| `test_presorted_skips_sortgrid` | synthetic `fix`/`pix` deliberately spaced closer than `eps` so sortgrid would mis-match; `presorted=True` recovers the known pose, `presorted=False` does not |
| `test_new_sources_registered` | `opencv_model` and `points_file` present with the documented `produces_seed` values |

Fixtures: generate the tiny text files in a `tmp_path` fixture. No new repo
data.

---

## Step 5 -- `scripts/import_calibration.py` (new CLI)

Modelled on `scripts/calibration_diagnostics.py` (argparse, `main()`, no
hardcoded paths -- unlike `scripts/calibrate_proptv_dlt.py`, which has
`C:/Users/alex/...` baked into `main()`).

```bash
uv run python scripts/import_calibration.py \
    --model-dir  <dir with calib_c*.txt>   \
    --points-dir <dir with c*_xyXYZ.txt>   \
    --num-cams 4 --imx 2560 --imy 2048 --pix 0.005 \
    --out <dir> [--no-refine] [--pixel-origin corner|centre] [--json report.json]
```

Behaviour:

1. Per camera: `read_opencv_flat15` -> `calibration_from_opencv` -> a seed
   `Calibration`.
2. `--no-refine`: write `camN.ori`/`.addpar` from the seed and stop. This is
   the "least conversions" mode -- a pure algebraic import, no fitting.
3. Otherwise: `read_xyXYZ` -> `CalibrationPointSet(ref_pts, img_pts,
   seed=cal)` -> `calibrate_from_source("opencv_model", cam, cpar, point_set,
   presorted=True)` -> write from the refined `CamResult`.
4. Report (Step 7) to stdout, and to `--json` when given.

Write with `Calibration.to_file(ori_path, add_file)`. Back up any existing
files the way `calibrate_dataset` does (`.autobck` suffix).

Do **not** add a `[project.scripts]` entry yet -- there is no calibration
entry point today and adding one is a separate decision.

---

## Step 6 -- the Ilmenau run (acceptance test for the whole feature)

Data lives in `../Multiview-Calibration`, branch **`origin/nov_2025`** (the
checked-out `main` is ~1% of the repo). **No LICENSE file -- reference only,
do not vendor any code or data into openptv2.**

Inputs:

- `markers_Nov_2025/calibration/calib_c{0..3}.txt` -- the 15-float OpenCV
  parameters.
- `c{0..3}_xyXYZ.txt` -- ~8550 rows of `x,y,X,Y,Z` per camera.
- Geometry: 2560x2048 px, pixel pitch 0.005 mm, markers on a 40 mm grid.
- Optics: **air** -- `mmp_n1 = mmp_n2 = mmp_n3 = 1.0`, `mmp_d = 1.0`,
  `interf: 0` (from `openptv/parameters_Run1.yaml`).

No frame conversion is expected: those extrinsics are already in the plate
frame (X,Y on the 40 mm grid with Y negated, Z along the traverse), which is
exactly a calblock frame. Step 7's camera-separation check will show if that
assumption is wrong.

Run both modes and compare:

```bash
# pure algebraic import, zero fitting
uv run python scripts/import_calibration.py --model-dir ... --no-refine --out cal_seed
# seed + bundle adjust on all ~8550 points per camera
uv run python scripts/import_calibration.py --model-dir ... --points-dir ... --out cal_refined
```

**What this deletes from the current Multiview-Calibration workflow**: the
farthest-point sampling of 40 markers, `calibration_block.txt`, the
PIL-synthesised fake `cam*.tif` dot images and the 4-point GUI click -- i.e.
all of `notebooks/manual_openptv_orientation_from_opencv_pipeline_nb.py`.
Five stages plus a human become two, and the refinement uses **8550 points per
camera instead of 40**.

---

## Step 7 -- the report (existing code only, no new maths)

Print, for both modes:

- per-camera reprojection RMS (`autocalibration.rms_px` on the `CamResult`);
- **`autocalibration.cross_camera_rcm(results, cpar)`** -- ray-convergence
  miss distance in mm (median/p90/p95/max). This is the check per-camera RMS
  cannot make;
- **pairwise camera separations** `||C_a - C_b||` -- frame-invariant, the one
  number no choice of lab frame can change;
- the bounding box of the calibration points, with the note that only inside
  it is the calibration supported by data;
- **the mean residual vector per camera** -- a constant ~0.5 px bias is the
  pixel-origin question above; if it appears, re-run with
  `--pixel-origin centre` and record which is right.

MyPTV PR#67 measured a synthetic case sitting at **0.14 px reprojection while
coordinates were 3.8 mm wrong** (shape correct to 4 um) -- focal length,
principal point and depth trade off against each other. So RMS alone is not
acceptance. **Acceptance is: RCM and camera separations physically sensible,
*and* refined RMS <= seed RMS.**

Also cross-check with `src/openptv2/calibration_diagnostics.py` (`load_model`,
`compute_diagnostics`) and the `visualize-calibration` skill -- both are
model-agnostic and work off any `.ori` for free.

---

## Step 8 -- later doors (documented, not built now)

In this order:

- **Door A standalone** -- a `points_file` source seeded by
  `scripts/calibrate_proptv_dlt.py:dlt_resection` (numpy-only; its
  `_self_test()` passes, recovering C, cc and all three angles exactly). This
  removes OpenCV from the loop entirely whenever >=6 non-coplanar known points
  exist.
- **Multiplane** -- `MultiPlanesPar` (`algorithms/parameters.py:1092`) parses
  `multi_planes.par` and nothing consumes it. Concatenate all planes' points
  per camera and solve once over the union.
  `alexlib/proPTV_OpenPTV_MyPTV_Test_case_1024_15` is a ready-made 6-plane
  fixture in openptv2's own layout. PR#67 measured focal lengths 0.59% out
  with the board confined to 12 mm vs 0.17% with 200 mm of depth wander --
  depth spread is a **conditioning requirement**, not a convenience.
- **Door C, model resampling** -- evaluate a Soloff/DaVis polynomial on a grid
  over the fitted volume to regenerate door-A points. proPTV version trap:
  v1.0 is 19 coefficients, v1.1 is 20 (adds `Z^3`), indistinguishable except
  by counting lines, and the shipped example data is 19-line v1.0 against a
  v1.1 evaluator. **Count before interpreting.**
- **Door D** -- checkerboard on a translation stage, then freely moved
  (homography + Zhang), per the old plan's Phases 2 and 4. The only doors
  needing cv2; put it in a new optional extra, never in core.
- **Phase 6 (unchanged)** -- extend
  `.claude/skills/openptv-calibrate/SKILL.md` with a "choose your source"
  section once >=2 sources exist; add
  `print_source_table()`/`print_source_detail()` mirroring
  `tracking_registry.print_tracker_table()`.

### Standing policies

- **DaVis is a point source, not a model source.** Its calibration lives in
  undocumented `.set`/`.im7` buffer attributes; neither `lvpyio`, `ReadIM` nor
  `libim7` surfaces the mapping coefficients. Ask the user for a plate marker
  list. (DaVis is also the only other package here that models refraction
  explicitly -- Wieneke's 3-media pinhole -- so a DaVis pinhole calibration
  may already be physical.)
- **easyWand/DLT-11 silently drops distortion**: the `kc` vector lives in
  separate `_camNTforms.mat` files, not in `dltCoefs.csv`, so a bare DLT
  import yields `.addpar = 0 0 0 0 0 1 0`.
- **Do not copy `../Multiview-Calibration`'s conversion prototype.** It has
  two bugs: `S` applied on the left, and scipy's lowercase `'xyz'` (extrinsic)
  Euler convention. See the verified block above.

---

## Files

**New:** `src/openptv2/calibration_import.py`,
`tests/unit/test_calibration_import.py`, `scripts/import_calibration.py`.

**Modified:** `src/openptv2/autocalibration.py` (presorted bypass +
docstring), `src/openptv2/calibration_registry.py` (two entries),
`tests/unit/test_calibration_registry.py` (two tests).

**Referenced, unchanged:** `algorithms/{calibration,imgcoord,trafo,orientation,
sortgrid,parameters}.py`, `calibration_diagnostics.py`,
`scripts/calibrate_proptv_dlt.py`.

**Not touched:** anything under `algorithms/` -- **no Cython rebuild is needed
for any step in this plan.**

---

## Verification

```bash
uv run pytest tests/unit/test_autocalibration.py \
              tests/unit/test_calibration_registry.py \
              tests/unit/test_calibration_import.py -v
uv run ruff check .
```

- `test_autocalibrate_cavity` green and unchanged -- proves the `presorted`
  parameter did not disturb the existing path.
- `test_opencv_projection_matches_img_coord` < 1e-9 px -- proves the
  conversion.
- `test_distortion_centre_gap_bounded` inside 0.1-2 px -- proves the known
  limitation is still the known limitation.
- Ilmenau: refined RMS <= seed RMS, and `cross_camera_rcm` median plus the
  pairwise camera separations physically sensible for a 2.38 m x 7.0 m barrel.
- No Cython rebuild required; `uv run python setup.py build_ext --inplace` is
  **not** part of this plan.

Commit discipline: the doc and each step land separately. The Phase 1 code
(`calibration_registry.py`, `test_calibration_registry.py`, the
`autocalibration.py` modification) stays uncommitted until Step 0 is green.

---

## Appendix A -- format catalogue

| Package | Model | Physical exterior? | Refraction | On-disk format | OSS reference |
|---|---|---|---|---|---|
| **OpenCV** | pinhole + Brown rational (+prism, tilt) | yes (`rvec`/`tvec`) | no | ad-hoc `FileStorage` YAML/XML/JSON | opencv/opencv (Apache-2.0) |
| **proPTV** | Soloff polynomial, 19 (v1.0) / 20 (v1.1) coefs per axis | **no** | implicit | `soloff_cN{x,y}.txt`, one float per line | RobinBarta/proPTV |
| **MyPTV (Tsai)** | pinhole + `E` (3x5) image-space correction | yes (`O`, `theta`) | implicit | `camN`, positional text, tagged `Tsai model camera` | ronshnapp/MyPTV (MIT) |
| **MyPTV (extZolof)** | cubic fwd `A` + cubic ray-dir `B` + common origin `O` | partial (`O` only) | implicit | `camN`, keyed `O i / A i j / B i j` | same |
| **easyWand/DLTdv** | DLT-11 + separate Bouguet `kc` | recoverable | no | `*_dltCoefs.csv` (11xN), `*_camNTforms.mat` | tlhedrick/dltdv, dltutil |
| **DaVis** | 3rd-order polynomial **or** 3-media pinhole + Scheimpflug | yes (pinhole) | **yes** (explicit) | inside `.set`/`.im7`; **undocumented** | none |
| **Metashape** | Brown, normalised, `f/cx/cy/B1/B2/K1-4/P1-2` | yes (4x4 transform) | no | Agisoft calibration `.xml` | documented + Python API |
| **Australis** | Brown, mm, `c/xp/yp/K1-3/P1-2/B1-2` | yes | no | proprietary text report | maths in NASA ASP |
| **COLMAP** | 16+ models, OpenCV-compatible | yes (quaternion, world->cam) | no | `cameras.txt`/`images.txt`/`points3D.txt` | colmap/colmap (BSD) |
| **openptv2** | pinhole + Brown (mm) + affine + explicit glass/multimedia | yes | **yes** (explicit) | `.ori` + `.addpar` | this repo |

Two conclusions this table drives:

- **Point-level interchange beats model-level interchange for four of them.**
  proPTV, MyPTV, DaVis and Multiview-Calibration all emit the same five
  columns. One reader.
- **Model-level import is worth building only for OpenCV, COLMAP and
  Metashape**, and those three are the *same* model up to the p-swap, the
  half-pixel origin, `R` vs `R.T` and mm-vs-normalised scaling. One converter,
  three thin front-ends.

## Appendix B -- conventions checklist

Every one of these has bitten someone in this exact problem domain.

1. **Y-axis flip (image).** CV `v` increases downward; photogrammetry /
   openptv2 metric `y` increases upward (`trafo.py:120`).
2. **Y/Z flip (camera frame).** CV: +Y down, +Z forward. Photogrammetric: +Y
   up, +Z backward. Bridge: `S = diag(1,-1,-1)`.
3. **Pixel origin -- three conventions.** OpenCV/Kalibr/DLTdv8: integer at
   pixel centres, centred principal point `(w-1)/2`. COLMAP/Metashape: corner
   origin, `w/2`. openptv2 uses `imx/2`.
4. **`R` vs `R.T`.** OpenCV/COLMAP/Metashape `[R|t]` is world->camera;
   openptv2's `dm` is camera->world, used transposed. `C = -R.T t`.
5. **Distortion direction.** OpenCV / Metashape / COLMAP / openptv2: ideal ->
   distorted. Classical Brown / Australis / easyWand's stated `kc` formula:
   distorted -> undistorted. Wrong direction gives ~2x the distortion with the
   right sign -- looks plausible, is not. Test on the widest lens's corners.
6. **`p1`/`p2` swap.** Brown/Australis/Metashape/openptv2 put `P1` on
   `(r^2 + 2x^2)`; OpenCV/COLMAP put `p2` there.
7. **Distortion coefficient units.** CV family: dimensionless, on `X/Z`.
   Brown/Australis/openptv2: on **millimetres**. `k1` differs by `cc^2`, `k2`
   by `cc^4`, `k3` by `cc^6`, `p1/p2` by `cc`.
8. **Focal length units.** `fx, fy` in pixels vs `cc` in mm. Needs the pitch.
9. **Principal point: absolute vs offset.** OpenCV/COLMAP `cx,cy` absolute;
   Metashape `cx,cy` and openptv2 `xh,yh` are offsets from the image centre
   (openptv2's in mm, with `yh` sign inverted).
10. **Aspect ratio and skew.** openptv2's `scx` is **isotropic** -- it
    multiplies both outputs (`trafo.py:334-335`), so `fx != fy` must go into
    `pix_x != pix_y`.
11. **Euler order and sign.** openptv2 uses `Rx(omega)Ry(phi)Rz(kappa)`.
    Always transport the rotation **matrix**, derive angles last, and assert
    the round-trip within 1e-12.
12. **Where distortion is centred.** openptv2 adds `xh,yh` *before* distorting
    (`imgcoord.py:74-75`), so the radial term is centred on the sensor centre;
    OpenCV centres it on the principal point. They coincide only at
    `xh = yh = 0`.
13. **Polynomial models have no exterior orientation at all.** proPTV,
    DaVis-polynomial and MyPTV-extendedZolof carry no camera position to
    convert. The only bridge is door C.

## Appendix C -- grounding references

- **`ronshnapp/MyPTV#67`** (open, `stebrizzo`, +6999/-1) -- checkerboard
  calibration in two modes. **The architectural precedent**: not one line of
  MyPTV's camera models was touched; the new code only emits an ordinary
  `eta zeta x_lab y_lab z_lab` points file. Three moves worth copying: (i) an
  optional trailing **view-index column** old positional readers ignore, which
  buys per-view diagnostics free -- openptv2's `read_calblock`
  (`sortgrid.py:91`) slices `[:,1:4]`, so the same trick works; (ii) the
  `origin_hint` parameter carried over between frames as the minimum human
  input for unambiguous corner labelling; (iii) **validating on
  frame-invariant quantities** rather than reprojection RMS.
- **`OpenPTV/proPTV_OpenPTV_MyPTV_comparison`** -- proPTV's Soloff calibration
  (`proPTV_LineOfSight/soloff.py`), the door-C reference.
- **`alexlib/proPTV_OpenPTV_MyPTV_Test_case_1024_15`** -- a real 6-plane
  multiplane dataset in openptv2's classic layout
  (`parametersMultiPlane/multi_planes.par`), already parseable by
  `MultiPlanesPar.from_file`. The Step 8 multiplane fixture.
- **`alexlib/Multiview-Calibration` (Ilmenau barrel)** -- 4 cameras, 18 planes
  used, 2.38 m x 7.0 m barrel at 2560x2048, 40 mm marker pitch, ~8550
  points/camera. Live branch is `origin/nov_2025`. Pipeline: flat-Z0 init ->
  per-camera `cv2.calibrateCamera` -> `stereoCalibrate` on plane 0 ->
  4-camera DLT triangulation -> recalibrate -> `cN_xyXYZ.txt`. **That whole
  loop exists only to recover unknown plate Z**; the open TODO "enforce the
  40 mm constraint" suggests the recovered Z's do not respect the known pitch,
  which is worth chasing. **No LICENSE file -- reference only.**

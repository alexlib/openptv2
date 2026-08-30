# Illmenau dots-plate pipeline: hand-held multi-plane → OpenCV → openPTV

**Status:** approved sidecar to `2026-08-30-calibration-hub-multi-source.md`. Hub is the
source-agnostic spec (four doors, verified OpenCV→`.ori` conversion `S` on right,
`calibration_import.py` / `presorted` / `rig.yaml` seed). This doc is the
**Illmenau-first vertical slice** through it: hand-held plate at unknown `Z`,
two plate profiles, tunable detector for the small-dot failure mode, `L`-coded
orientation, `Z`-sign reconciliation with the `run3` rig, and straight-to-sequence.

Two parts remain independently executable: hub Part 1 (import existing
calibration) and hub Part 2 (seed a first `.ori`). Everything below threads
through them — no fork.

---

## 0) What this plan adds beyond the hub plan

* **New images, not `cN_xyXYZ.txt`:** `C:\Users\alex\Downloads\Illmenau\Kalibrierung_1\*.tiff`
  (`48` frames) and `…\Kalibrierung_2\…` are the input. Hand-held plate,
  `>10` unknown `Z`s. Detected once with our own code, not Robin's
  `Ilmenau_xy/c{cam}/marker/*.txt` — `cv2.findCirclesGrid` fails on Illmenau's
  few-px black dots without tuning.
* **Two plate profiles, one pipeline (your call):**

  | profile | `nx` (X left→right) | `ny` (Y bottom→top) | dots | code |
  |---|---|---|---|---|
  | `large_25x19` | 25 | 19 | 475 | uncoded, regular grid |
  | `small_6x7_coded` | 6 | 7 | 42 | 3× white-in-black `L`, rest solid black |

  `pitch_x` / `pitch_y` is a user param (default `40 mm`, `hub:40mm` case). No
  hard-coded `marker_distance = (40,40)` beyond the default. `small_6x7_coded`
  is the `C:\Users\alex\Downloads\…\00000039_00000000005C3047.tiff` /
  `Kalibrierung_1/00000040_…tiff` plate you pointed at — bright centre + dark
  ring, `L` with `corner(0,0)` + `Y` at `+1·pitch` + `X` at `+2·pitch` (black at
  `+1·pitch,0` between). That anchors origin and `+X/+Y` per plane without
  external IDs.
* **Lab frame locked per your note:** `+X` left→right, `+Y` bottom→top
  (gravity top→bottom, i.e. `-Y`), `+Z` object→camera. Cameras sit at `+Z` on
  one side, look toward `−Z`. This is a sign flip from `multiview_calibration.py:75`
  `Y = -arange(ny)*40` (`top-left (0,0)` with `Y` down). One `y_sign` branch
  covers both; documented once, not a silent flip.
* **Rough rig + comparison:** `C:\Users\alex\Dropbox\3DPTV_Illmenau\Multiview-Calibration\cal\run3\*.ori`
  is the rough seed (see below for its apparent `−Z` cameras). Keep both pose
  tracks — `rig.yaml` look-at seed vs OpenCV extrinsics `C = -Rᵀt` — and allow
  rig adjustment from correspondences (`joint_plate_bundle_adjust` / Kabsch).

**Code path previewed in `manual_openptv_orientation_from_opencv_pipeline.html`:**
`file:///C:/Users/alex/Dropbox/3DPTV_Illmenau/Multiview-Calibration/manual_openptv_orientation_from_opencv_pipeline.html`
is the Jupyter export of the hub's Multiview loop — `Parameter` (`25×19`, `40`,
`2560×2048`), `DLT(P1..P4, xy1..4)` 8×4 `A` via `xy·P₃−P₁` SVD, flat-`Z0`
`cv2.calibrateCamera` per cam → `stereoCalibrate` plane `0` → 4-cam DLT tri →
`P[0]=XYZ0` → recalibrate with `CALIB_USE_INTRINSIC_GUESS` → `cN_xyXYZ.txt`.
This plan ingests that pipeline headless and runs it behind `plate_calibration.py`
before the hub's algebraic OpenCV→`.ori` tail.

---

## 1) Pipeline

```
raw c{cam}_{plane}.tif  (Kalibrierung_1 small_6x7_coded; Kalibrierung_2 / large_25x19)
  → preprocess (hp) → tunable blob detector → centroids + per-dot type {black, coded}
  → labeler:
       coded 6×7:  L-anchored (corner +1Y +2X) → (ix,iy) grid
       uncoded 25×19: adjacency BFS + RANSAC affine
     → (X = ix·pitch_x, Y = iy·pitch_y·y_sign, Z = 0_plane)  y_sign = +1 (Illmenau small) / -1 (legacy)
  → multi-plane solve for Z_per_plane + intrinsics
       O: cv2.calibrateCamera flat-Z0 → stereoCalibrate plane0 → 4-cam DLT tri → recalibrate   (Multiview loop)
       L: DLT resection  scripts/calibrate_proptv_dlt.py:36 per cam, ≥2 non-coplanar planes, coplanarity guard
  → calibration_from_opencv  (hub verified: S=diag(1,−1,−1) on RIGHT, dm=RᵀS, C=−Rᵀt, cc=fx·pix_x, pix_y=cc/fy, xh/yh, k/p scaling)
     → angles_from_dm (calibration.py:44 Rx(omega)·Ry(phi)·Rz(kappa), phi=asin clip, kappa=−atan2, omega=−atan2, assert 1e−12)
  → CalibrationPointSet(ref_pts,img_pts,seed) → autocalibration.py:363 _refine_and_select(presorted=True) → full_calibration flags → .ori/.addpar (calibration.py:568)
  → compare vs rig.yaml look-at seed (calibration_seed.py:M6 dm_from_lookat) + optional Kabsch A,b (SVD, dm_new=A·dm_old, C_new=A·C_old+b) + joint_plate_bundle_adjust:705
  → verify: point_position_batch:153 triangulate → per-plane SVD planarity + pitch + cross_camera_rcm:633 + per-cam RMS
  → sequence  correspondences.py:902 correct_frame → tracker.py:67 Batch/pyptv_batch.py
```

---

## 2) The `run3` rig and the `Z` sign

`C:\Users\alex\Dropbox\3DPTV_Illmenau\Multiview-Calibration\cal\run3\*.ori`:

* `cam1` `C = (1921, 649, −3101)`, `cam4` `(1540, 595, −3368)` with `Z < 0`,
  `cam2` `(4922, −1251, 230)` and `cam3`-like `(1549, 1432, −3056)` with mixed
  signs, `glass = (0,0,−1)` (`src/openptv2/algorithms/calibration.py:88 Glass`
  sanitizes `(0,0,0)` → `(0,0,1)`). `calibration_block.txt` at `hub:40mm` grid
  with `Z` around `1499` / `740` etc. In this frame the cameras only *appear*
  on `−Z`; in Illmenau convention they are all on the same `+Z` side as you
  noted — the difference is a lab-frame rotation/translation, not four separate
  placements. Handle as:

  * declare Illmenau lab frame (`+Z` object→camera) in `rig.yaml`
    `volume_centre` + `position`/`target`/`up`/`roll`/`focal_mm`,
  * keep `run3` as the rough seed example, show the sign reconciliation via
    `similarity_from_correspondences` (Kabsch `UΣVᵀ`) — rotate cameras, never
    points (`dm_new=A·dm_old`, `C_new=A·C_old+b`, hub `108` verified to
    `3e−12 px`). Document the `0.5px` `pixel_origin` `imx/2` vs `(imx−1)/2`
    branch in the same place.

---

## 3) Steps (Illmenau order — `S0→S1→S4→S5` first to sequence)

**S0 Gate** — `uv run pytest tests/unit/test_autocalibration.py tests/unit/test_calibration_registry.py -v`
and `ruff check` green (hub Step 0). Proves the `_refine_and_select` extraction
is invariant; `test_autocalibrate_cavity` must remain byte-identical. S0 already
required fixing the unrun Phase-1 `TestRefineAndSelect` (synthetic 49-point
planar grid at `±30mm`, `pitch 10`, look-at pose `C=(80,30,−500)`, modest
`1mm/0.005rad` perturb, `eps=60` — see the committed test).

**S1 Detection — `src/openptv2/detect_plate.py` (new, no `cv2` in default path)**

* Preprocess `image_processing.py:preprocess_image(hp_flag)` per `ControlPar`.
* Detect via `src/openptv2/segmentation.py:_vectorized_targ_rec` under YAML
  block `detect_plate:` (`gvthres`/`discont`/`nnmin`××/`sumg_min` — separate
  from tracer `targ_rec`, plate dots are larger). Tunable on one
  `Kalibrierung_1/00000040_…tiff` frame; `cv2.findCirclesGrid` only as
  `--detector opencv` behind the optional extra.
* **Type classification (`small_6x7_coded`):** per `Target`, sample `5×5` centre
  mean `I_c` and annulus `r±2px` mean `I_r`; `coded ⇔ I_c>hi ∧ I_r<lo ∧
  (I_c−I_r)>thr`. YAML `coded_thr`; `--save-debug` thumbnails per dot.

**S2 Labeler — `src/openptv2/plate_labeler.py` (two profiles, auto-detected)**

* Auto-detect coded vs uncoded by `n_coded == 3` → coded path else uncoded
  (so no explicit `profile` flag is needed, though one is accepted for
  forcing). Pitch is always a user param; `nx/ny/y_sign` per profile above.

  *Coded `6×7` (Illmenau `Kalibrierung_1`):* find 3 coded centroids, identify
  corner as the coded dot with one coded neighbour at `≈pitch` in `+Y` and one
  at `≈2·pitch` in `+X` (black dot at `+1·pitch,0` between). Check right angle
  (`>0.9`) and distances `±0.15·pitch`. Bases
  `e_x=(coded_X2−corner)/(2·pitch)`, `e_y=(coded_Y1−corner)/pitch`; assign
  `ix=round(((p−corner)·e_x)/pitch)`, `iy=round(((p−corner)·e_y)/pitch)`.

  *Uncoded `25×19` (Multiview):* Delaunay median spacing → `1.4·pitch`
  adjacency → BFS index assignment → RANSAC affine outlier reject; require
  `>0.85·nx·ny` completeness else flag view for `origin_hint` click (hub
  App. C / MyPTV PR#67 trick). `hub:700` TODO ("enforce 40mm constraint") is
  where this path earns its keep.

* Emits `ref_pts = (ix·pitch_x, iy·pitch_y·y_sign, Z_plane)` (`y_sign` as above,
  `Z` left free). `x y X Y Z` 5-col file compatible with `cN_xyXYZ.txt`.

**S3 Multi-plane solve — `src/openptv2/plate_calibration.py`**

* `solve_opencv(views, image_size, init_intrinsics?)` reproduces the
  `manual_…html` / `multiview_calibration.py:87-118` loop (exposes `Zs` per
  plane, `CALIB_USE_INTRINSIC_GUESS` on second pass). Keep the flat-`Z0`
  assumption on the first plane explicit (straight-ray assumption noted there
  and valid only in air `mmp_n1=n2=n3=1` / `parameters.py:699`).
* `solve_dlt(views)` thin wrapper over `scripts/calibrate_proptv_dlt.py:36`
  (`dlt_resection`) per cam on `≥2` labelled planes with near-coplanar
  singular-value guard (raise, naming coplanarity). Both emit
  `K,dist,rvec,tvec` + `P` (ref pts with solved `Zs`).

**S4 Import / seed layer — `src/openptv2/calibration_seed.py` +
`src/openptv2/calibration_import.py`** (hub Steps 1/`S1`, no `cv2`)

* `calibration_seed.py`: `angles_from_dm`, `exterior_from_rotation(C,dm)` via
  angles+`compute_rotation_matrix()` (`calibration.py:44`) and `assert 1e−12`
  (trap 4 — never assign `Ext.dm` direct), `dm_from_lookat(C,target,up,roll)`
  (`hub M6`), `seed_from_lookat(position,target,focal_mm,…)` (`cc=focal_mm`,
  `xh=yh=0`, glass from `benchmarking/camera_rig.py:221`), `seed_from_dlt`,
  `read_rig`/`seed_rig`/`write_rig_ori` (`rig.yaml` next to `parameters/`,
  `calibration.py:568` `to_file`, `.autobck`, refuse clobber without
  `overwrite=True`, `focal_mm` required — hub M5 `cc=imx·pix_x` is `4-31×`
  wrong, `cc<=0` guard naming the `.ori`).

* `calibration_import.py`: `calibration_from_opencv(K,dist,rvec,tvec, imx,imy,pix_x,…)`
  exact air path above, `opencv_from_calibration` inverse for round-trip
  `1e−9px`, `read_xyXYZ` (whitespace/comma, `#`, optional 6th col, hub `1.4`),
  `read_opencv_flat15` (`rvec3+tvec3+fxfy+cxcy+k1k2p1p2k3`), `similarity_from_correspondences`
  Kabsch, accept `dist` len `4/5/8/12/14`, raise naming `k4..s1…` when
  unsupported (use door C instead).

**S5 `presorted` + registry** — `src/openptv2/autocalibration.py:363`
`_refine_and_select(presorted=False)` bypasses all three `sortgrid:383,390,397`
calls (`sorted_pix=pix`, `n_matched=len(pix)`), `calibrate_from_source:489`
threads `presorted`, docstring notes `raw_orient:540` zeroes `k1..she` so door B
must use `point_set.seed` (trap 1) and `calibration_registry.py:59` registers
`opencv_model`/`points_file` + `rig_lookat`/`dlt_resection` (hub Steps 3/`S4`).

**S6 Rig seed & comparison**

* `rig.yaml` declares Illmenau frame explicitly (all `+Z` one side). Hub
  optics policy `hub:82-92` applies: air default `1/1/1/1` only; switching to
  real `n3=1.33` only together with a re-fit, or refraction is double-counted.
* Show both pose tracks for `run3`: rough `rig.yaml` vs OpenCV `C=−Rᵀt`.
  Optional frame reconciliation via Kabsch as above; optional adjustment
  `autocalibration.py:705 joint_plate_bundle_adjust` (`reg_weight` gauges the
  shared plate) or `tracer_self_calibrate:1186` (`hold_cam`) when you ask to
  adjust the rig from correspondences.

**S7 Verification (your "single planar + pitch" gate)** — `scripts/verify_plate.py`
(headless, no GUI):

* `orientation.py:153 multi_cam_point_positions` on dots grouped by
  `(ix,iy,plane)` (4/3/2-ray split), per-plane SVD plane fit `RMS/max` +
  histogram, per-plane neighbour pitch median vs `pitch_x/y` (Multiview
  `multiview_calibration.py:152` `horizontal/vertical_error_XYZ` pattern), plus
  hub Step 7 (`autocalibration.py:633 cross_camera_rcm` `median/p90/p95/max`,
  `rms_px:110`, pairwise `‖C_a−C_b‖`, bbox, mean residual for `0.5px`
  `pixel_origin`).

**S8 CLI**

* `scripts/import_calibration.py` (hub Step 5/`S6`):
  `--model-dir/--points-dir --num-cams 4 --imx 2560 --imy 2048 --pix 0.005
  --pixel-origin corner|centre --no-refine --json report.json` and
  `seed --rig rig.yaml --dataset --dry-run --overwrite`. Writes via
  `Calibration.to_file`, `.autobck`. `--verify` runs S7.
* No `[project.scripts]` entry yet.

**S9 Acceptance (Illmenau run, both old and new plates)**

* Run both `--no-refine` seed and refined modes on the existing
  `C:\Users\alex\projects\Multiview-Calibration\cN_xyXYZ.txt` (8530/plane ≈
  `25·19·18` minus misses) and on new `Downloads/Illmenau/Kalibrierung_*/…tiff`
  detection output. Criteria: `refined RMS ≤ seed RMS`,
  `cross_camera_rcm` median sensible for the `2.38×7.0m` barrel, planarity
  `RMS<0.15mm`, pitch median `±0.5%` of user param, `>95%` of expected
  `nx·ny·planes` dots in 4-ray. `diagnostics.py` / `visualize-calibration`
  cross-check for free. No vendor of `Multiview` code/data beyond reference
  (no LICENSE there, per `README:351`).

---

## 4) Files

*New:* `src/openptv2/detect_plate.py`, `src/openptv2/plate_labeler.py`,
`src/openptv2/plate_calibration.py`, `src/openptv2/calibration_seed.py`,
`src/openptv2/calibration_import.py`, `scripts/import_calibration.py`,
`scripts/verify_plate.py`, `tests/unit/test_calibration_import.py`,
`tests/unit/test_calibration_seed.py`, `tests/unit/test_plate_detection.py`.

*Modified:* `src/openptv2/autocalibration.py` (`presorted` + derived `eps` /
hub `S3`), `src/openptv2/calibration_registry.py` (four entries),
`scripts/calibrate_proptv_dlt.py` (shims after move), zero-`cc` guards
(`gui/ptv_calibration.py:653` etc.), `.claude/skills/openptv-calibrate/scripts/calib.py:306`
→ `seed_from_lookat`.

*Untouched:* `algorithms/` except the guard — no Cython rebuild.

---

## 5) Tests (follow `tests/unit/test_synthetic_calibration.py` helpers)

`test_angles_roundtrip`, `test_exterior_from_rotation_survives` (trap 4),
`test_opencv_projection_matches_img_coord` `xh=yh=0` `<1e−9px`, roundtrip inverse,
`test_distortion_centre_gap_bounded` `0.1-2px`, batch/scalar parity (hub
`trap 5`), `similarity_from_correspondences`, `read_xyXYZ/read_opencv_flat15`
variants + unsupported `k4..` raise, coded/uncoded labeler on synthetic grids,
`rig.yaml` `focal_mm` required + clobber/count guards, and the committed
`TestRefineAndSelect`/`test_autocalibrate_cavity` gate.

---

## 6) Risks

* `findCirclesGrid` on small dots → handled by `detect_plate` `TargetPar` path,
  with per-dot `coded` thumbnails for tuning; falling back to manual `origin_hint`
  on a single failing view only.
* Hand-held `Zs` near-coplanar → `seed_from_dlt` raises on coplanar singular
  value; OpenCV `Z` loop remains the default for that case.
* `runs/run3` `Z` sign confusion → made explicit via `rig.yaml` frame + Kabsch
  reconciliation, with pairwise `‖C_a−C_b‖` (frame-invariant) reported.

---

## 7) Execution order (as you asked)

`S0` (already green after the committed test fix) → `S4` hub core
(`calibration_seed` + `calibration_import`) + `S5` (`presorted`/`registry`) →
acceptance on the `Multiview-Calibration` `cN_xyXYZ` algebraic path (`sequence`
unblocked) → `S1-S3` detector/labeler on `Downloads/Illmenau/Kalibrierung_1`
small plate (`verify_plate` overlays), then tracer sequence/tracking
(`correspondences` / `tracker`).

`docs/plans/2026-08-30-calibration-hub-multi-source.md` stays the spec;
this doc is its Illmenau companion and references it — not a replacement.

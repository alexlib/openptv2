# Illmenau 4-Camera Calibration — Hand-Held 6×7 Dots Plate (Kalibrierung_1..4 → OpenPTV2)

This document describes the complete headless calibration used for the
Barrel of Ilmenau 4-camera wall rig. It was developed live in
`notebooks/illmenau_4cam_pipeline.py` (marimo-pair) and then split into
two scripts so the heavy image work and the solver are independent:

* **Part 1** `scripts/detect_illmenau_4cam_part1.py` — `Kalibrierung_1..4` TIFFs → per-camera `XYZ↔xy` pairs
* **Part 2** `scripts/calibrate_illmenau_4cam_part2.py` — pairs → OpenPTV `cal/camN.tif.ori/.addpar`

Run headless after tuning `pitch/gv/sumg` in the notebook:

```bash
uv run --with opencv-python python -u scripts/detect_illmenau_4cam_part1.py --pitch 120 --gv 20 --sumg 5000
uv run --with opencv-python python -u scripts/calibrate_illmenau_4cam_part2.py
# verify
uv run python scripts/verify_plate.py --cals openptv_illmenau_4cam/cal --points-dir openptv_illmenau_4cam/cal
```

---

## 1. What is being calibrated

### 1.1 Plate

* Small Illmenau plate `6×7` (`6` columns `X` left→right, `7` rows `Y` bottom→top), `pitch 120 mm`, dot `Ø 60 mm`, thickness `6 mm`. World `Z=0` on the plate.
* **39** dots are **dark-on-white** (black circles on white plate), **3 dots white-in-dark** (white centre inside a dark ring) form an `L`: corner `(0,0)`, `+1·pitch` in `+Y`, `+2·pitch` in `+X` with a black dot between. The `L` disambiguates `+X/+Y` without coded markers.
* Two physical realisations exist: the small `6×7` coded above and a large `25×19 uncoded` (`25`×`19`, same pitch concept) — the labeler auto-detects by `n_coded==3` (`src/openptv2/plate_labeler.py:258`).

![Plate ROI filtered 42 after NEG+neighbor-cost — left stray outside lattice is rejected](../docs/images/illmenau_roi_42.png)
*Caption: `Inside ROI NEG: 42` — the cyan dashed rect is the ROI from a very-blurred bright rectangle. Lime are the kept 42 lattice points; the single left green outside the white plate is the positional outlier dropped by the neighbor-cost filter.*

### 1.2 Rig for cameras 1–4 (south wall, positive Z)

```
top view (XZ)                     back view (XY, from south)
 Z
 ^  -Z (north, cams 5-8)          Y
 |   ● 5-8                        ^ 2900 ── ●2 ●4
 |                                |        ●1 ●3 ── 700
 +--------> X                     +--------> X
        +Z (south, cams 1-4)              left  right
```

* `C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam\rig.yaml:11`:
  ```yaml
  volume_centre: [0,615,0]
  cameras:
    - {position: [ 2528, 700, 2528], target: [0,615,0], up: [0,1,0]} # 1 SE bottom-right
    - {position: [ 2528,2900, 2528], target: [0,615,0], up: [0,1,0]} # 2 SE top-right
    - {position: [-2528, 700, 2528], target: [0,615,0], up: [0,1,0]} # 3 SW bottom-left
    - {position: [-2528,2900, 2528], target: [0,615,0], up: [0,1,0]} # 4 SW top-left
  ```
* `position` = `C` camera centre world `[X,Y,Z]` mm. Origin `Y=0` floor/heating-plate, `Y=615` datum marker `ix=2,iy=3` on `000000`, `Z=+3575·cos45≈+2528` south wall, `X=±2528` left/right, `Y=700/2900` (2200 sep). Test section `Ø7150 r3575 h3580`.
* `target` = `T` Look-At point. `src/openptv2/calibration_seed.py:80 dm_from_lookat(C,T,up)` builds `back=(C-T)/|C-T|` (= camera `+Z`), `right=up×back`, `up'=back×right`, `dm=[right,up',back]` (camera→world). All four `T` identical so axes converge on the datum.
* `up:[0,1,0]` = world `+Y` up. It is the world direction that maps to image `+y` (up). `0,0,1` would roll 90°. `roll` adds `Rz(roll)`.

OpenPTV world: `X` right, `Y` up, `Z` toward camera (`+Z` south for `1-4`, `−Z` north for opposite rig `5-8` which is `180°` about `Y`: `(X,Z)→(−X,−Z)`).

![Top/back rig — wall r3575, Y 700/2900, Z +2528 south](../docs/images/illmenau_rig_top_back.png)

### 1.3 OpenPTV calibration model

OpenPTV is a pinhole + Brown distortion + optional glass interface (`src/openptv2/algorithms/calibration.py`, `src/openptv2/calibration_seed.py`, `src/openptv2/calibration_import.py`):

* **Interior** `Interior(xh,yh,cc)` — `cc` principal distance (`focal_mm`, `cc=fx·pix_x`), `xh,yh` principal point (`xh=(cx-imx/2)·pix_x`, `yh=(imy/2-cy)·pix_y`). `pix_x=0.005 mm`, `imx=2560 imy=2048`.
* **Exterior** `Exterior(X0,Y0,Z0,omega,phi,kappa)` — `C=(X0,Y0,Z0)`, `dm=Rx(omega)·Ry(phi)·Rz(kappa)=[right,up',back]` camera→world. Stored as angles; `dm` is recomputed via `compute_rotation_matrix()` (`calibration_seed.py:57 exterior_from_rotation` asserts `|dm−dm_recomp|<1e-12`).
* **Distortion** `AddedPar(k1,k2,k3,p1,p2,scx,she)` — Brown: `r²=(x²+y²)/cc²`, `x' = x(1+k1 r²+k2 r⁴+k3 r⁶)+2p1 xy+p2(r²+2x²)` etc. `k` scales as `k1=k1_cv/cc²` when converting from OpenCV (`calibration_import.py:147`).
* **Glass** `Glass(vec,n1,n2,n3,d)` — interface normal (unit) and refractive indices; air is `1/1/1/1` along `+Z` (`benchmarking/camera_rig.py:221`).
* **Pixel→metric** `trafo.py:pixel_to_metric(x,y,cpar)` → `(x−cx)·pix + xh`.

Conversion OpenCV→OpenPTV (verified block `calibration_import.py:124`):
```
S=diag(1,−1,−1)   R_cv=Rot(rvec)  C=−R_cvᵀ·tvec  dm=R_cvᵀ·S
cc=fx·pix_x  xh=(cx−imx/2)·pix_x  yh=(imy/2−cy)·pix_y
k1=k1_cv/cc²  k2=k2_cv/cc⁴  k3=k3_cv/cc⁶  p1=p2_cv/cc  p2=−p1_cv/cc
```

---

## 2. Pipeline overview

```
Kalibrierung_1..4/*.tiff (pre-underscore sync 48 frames, 000000 origin)
  → per-image: find_plate_roi → work8_neg → target_recognition (ROI only) → 43→42 positional filter → label_plate 6×7 → XYZ↔xy
  → per-cam flat collections (39/35/40/41 frames, ~1300 pts)
  → cv2.calibrateCamera → K/dist/rvecs/tvecs → calibration_from_opencv → cal/camN.tif.ori/.addpar
```

---

## 3. Step-by-step

### 3.1 Sync frames (`notebooks/illmenau_4cam_pipeline.py:50`, `detect_illmenau_4cam_part1.py:groups`)

```
groups[frame][ci]=Path   # frame = name.split("_")[0]
sync_frames=[k for k,v in groups.items() if len(v)==4]  # 48, e.g. 00000000..00000047
`000000` defines origin; `frame` number is pre-underscore, second half random suffix ignored.
```

### 3.2 ROI: very-blurred bright rectangle (`find_plate_roi:43`)

![Flood without ROI — 43 lime everywhere, plate dots missed, nxmax=80 gated](../docs/images/illmenau_flood_without_roi.png)

Without ROI, whole-image `target_recognition(gv20 sumg5000)` finds ceiling/floor/ladder speculars and misses plate dots (`nxmax=80` gates 60 mm dots at ~3.5 m).

```
work8 = clip((raw−p1)/(p99.5−p1)·255)  # percentile autoscale uint16→uint8
blurred = gaussian_filter(work8, σ=25)  # merges 60 mm dots, plate becomes bright blob
thresh = Otsu(blurred)                  # 256-bin histogram, max inter-class variance
bw = blurred > thresh
labeled,n = label(bw);  areas sorted;  largest = plate (area ~169k vs 15k wall)
roi = (x−w·pad, x+w·pad, y−h·pad, y+h·pad)  pad 0.07 → [736:1155,1243:1811] for 00000000 cam1
```

![ROI from blurred Otsu 84 — cyan dashed rect eliminates background](../docs/images/illmenau_blurred_roi.png)

Rationale: plate is the only large bright rectangle after heavy blur; `σ=25` erases dots but preserves the plate plateau. All later detection is `subrange_x/y=roi` only.

### 3.3 NEG for dark-on-white (`detect_plate_points:179`)

Plate is white with black dots. `segmentation.target_recognition` is a bright-peak detector (`src/openptv2/segmentation.py:50` `interior>gvthres` + 8-neighbour max + `discont` flood). Dark dots must be bright:

```
work8_neg = 255−work8
hp = preprocess_image(work8_neg,1,cpar,25)  # high-pass 25
tg = target_recognition(hp,tpar,0,cpar, subrange_x=roi, subrange_y=roi)  # gv20 sumg5000 nn10-5000 nx8-80
```

`POS` → `4–10` dots, `NEG` → `42–43` dots; `work8_neg` makes black dots bright, the 3 white-in-dark dots are still found via their dark outer ring (annulus) whose weighted centroid falls on the white centre.

### 3.4 Positional outlier → 42 lattice (`reject_outside_grid_v2:159`)

The raw ROI still yields `43` with one stray left of the plate (outside white, `outer_mean 73` vs median `159` on plate):

```
_outer_mean: 51×51 window minus 13×13 centre → outer plate brightness
reject_by_neighbor_cost: KDTree k=5 cost=sum(d[:,1:]) keep 42 smallest
  # regular 6×7 lattice: interior 4 neighbours ~60 mm, edge 3, corner 2 → stray isolated cost high
+ intensity gate: if filt outer<100, swap with next best high-outer from cost order
```

```
raw 43 filt 42 labeled 36  # 00000000 cam1
raw 43 filt 42 labeled 19  # 00000000 cam2 (oblique, fewer labeled)
```

OpenCV alt on same ROI (for parity) is also computed per frame:

```python
roi = work8[ymin:ymax, xmin:xmax]
found,corners = cv2.findCirclesGrid(roi,(6,7), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
# +xmin/+ymin back to full image; inject custom SimpleBlobDetector via blobDetector= kwarg if needed
```

### 3.5 Label to world XYZ (`plate_labeler.py:258`)

```python
img_pts,ref_pts,_idx = label_plate(cent,None, pitch_x=120,pitch_y=120,nx=6,ny=7,y_sign=1)
# ref_pts: (n,3) X=ix·120, Y=iy·120, Z=0  (y_sign +1 bottom→top matches world Y up)
# auto-dispatch: n_coded==3 → coded L (6×7 Illmenau), else uncoded BFS+RANSAC (25×19)
```

Uncoded path estimates `pitch_img=median NN`, builds `query_ball_point 1.4·pitch` graph, BFS from central seed, `cov` PCA for `ex/ey`, then `lstsq` affine `A=[ix,iy,1] → (x,y)` and prunes `resid<0.5·pitch`. Thickness `6 mm` is two `Z` planes (`0` and `6` along plate normal) when `5-8` are added.

### 3.5b What *are* the pairs we store? (no pose needed)

We **do not** know the plate pose in the world and we do **not** know the camera pose — and we do not need them for Part 1. For every `camera × plane` (`plane` = one hand-held `Kalibrierung` frame, e.g. `00000000`) we store, per detected dot, the **correspondence**

```
image xy  (n,2) pixel  — measured: centroid from 3.3–3.4
  ↕  id = (ix,iy) from labeling
world XYZ (n,3) mm     — synthetic: X=ix·120, Y=iy·120·y_sign, Z=0   (plate-local)
```

`XYZ` is **plate-local**, not world `0,615,0`. All planes share the same local `Z=0` definition; the *true* `Z` shuffling of the hand-held plate (`>10` `Z`s free) and the *true* camera `C/dm` are **unknowns solved in Part 2**. The `id` is what makes the pair useful: the same physical dot `ix=2,iy=3` (datum at `615 mm` above `Y=0` on `000000`) has the same `XYZ` in every plane file, but different `xy` per plane/camera because the plate moved — that is exactly what `cv2.calibrateCamera` needs: `N` views of a known planar pattern with *shared* intrinsics and *per-view* extrinsics `rvec/tvec` (plate→camera). No global `R|t` is assumed at this stage.

Concretely `collections[frame][cam] = {img_pts, ref_pts}` (`qnEH`/`detect_illmenau_4cam_part1.py:flat_collections`) and `cal/pairs_camN.npz` `ref_i/img_i` are the cut. `frame` itself is the plane index; the solver will later assign each `frame` its own `rvec/tvec`.

### 3.6 Flat collections (Part 1 output = end of image work)

`detect_illmenau_4cam_part1.py` loops `sync_frames`:

```python
flat_collections[ci].append((ref_pts,img_pts,frame)) if len(img_pts)>=20
# e.g. cam1 39 frames 1312 pts, cam2 35 1197, cam3 40 1309, cam4 41 1350
np.savez(cal/pairs_camN.npz, ref_0,img_0,... frames=[...])
np.savez(cal/collections.npz, camN_refs,camN_imgs)
```

This is the **cut point**: everything up to here is pure image work; Part 2 is pure geometry and does not re-read TIFFs.

### 3.7 Second half — Part 2: from pairs to OpenPTV calibration (`calibrate_illmenau_4cam_part2.py`)

Part 2 is **pure geometry**, no TIFFs are re-read. It consumes the Part 1 pairs.

Per cam, `N≥6` planes, each plane `n` varies (`6–36` labeled):

```python
objp=[ref.astype(float32) for ref,_ in pairs]  # (n,3) Z=0 plate-local per plane (from 3.5b)
imgp=[img.astype(float32) for _,img in pairs]  # (n,2) image
ret,K,dist,rvecs,tvecs = cv2.calibrateCamera(objp,imgp,(2560,2048),None,None,
                                             flags=CALIB_FIX_K3 if len<10 else 0)
# → shared intrinsics K/dist + per-plane extrinsics rvecs/tvecs (plate→camera)
# RMS ~1.4–3.0 px on this hand-held set (refine with bundle later)
rvec0,tvec0 = rvecs[idx_of_00000000]  # plate at world origin for this frame
cal,pix_y = calibration_from_opencv(K,dist,rvec0,tvec0, imx=2560,imy=2048,pix_x=0.005)
cal.to_file(cal/camN.tif.ori, cal/camN.tif.addpar)  # Interior xh,yh,cc + Exterior X0,Y0,Z0,ω,φ,κ + AddedPar k1,k2,k3,p1,p2
```

* `K/dist` are **shared** across all planes of that camera (`cc=fx·pix_x`, `xh=(cx−imx/2)·pix_x`, distortion `k1=k1_cv/cc²` … from `calibration_import.py:147`).
* `rvecs[t]/tvecs[t]` are the **solved unknowns** you asked about: the pose of the plate at `frame t` w.r.t. that camera. Picking `00000000` as `idx0` anchors the world at the datum (`ix=2,iy=3` → `0,615,0`) for the rig `target`. The `C=−Rᵀ·t` from that frame is the `C` written to the `.ori`; averaging `Cmean` over all planes is reported for sanity.
* Without `cv2`, DLT fallback `seed_from_dlt(all_ref with fake Z=pi·200, all_img, cpar)` gives `cc` and `C` but is degenerate on single-plane data and currently yields unrealistic `C` (≈8 m) because the stacked fake `Z` does not model the true hand-held `Z` distribution — install `opencv-python` (`uv add opencv-python` or `uv run --with opencv-python ...`).
* Full `solve_opencv_multiview` loop (`plate_calibration.py:21` flat-Z0 → `stereoCalibrate` on `plane 0` → 4-cam DLT tri to solve true `Zs` → `CALIB_USE_INTRINSIC_GUESS` recalibrate) replaces the single-frame anchor with a joint 4-cam `Z` when `1-4` are solved together.
* The opposite rig `5-8` is `180°` about `Y`: `(X,Z)→(−X,−Z)` with same `Y` and same `T` `[0,615,0]`, same `6 mm` two-plane handling, then joint bundle `openptv2.autocalibration` (`bundle`/`joint_plate_bundle_adjust`) to `RMS<1.0 px` `RCM<0.1 mm`.

### 3.8 How OpenCV → OpenPTV and how we verify reprojection

**Coordinate system (what `S` does).** OpenCV `R_cv|t` is *world→camera*: `x_cam = R_cv·X_world + t`. OpenPTV stores the inverse `C=−R_cvᵀ·t` (camera centre world) and `dm=R_cvᵀ·S` (`camera→world`) with `S=diag(1,−1,−1)` (`calibration_import.py:124` verified block). `S` flips `Y` (image `y` down vs world `Y` up) and `Z` (OpenCV `+Z` out of camera vs OpenPTV `−Z` toward scene). `Interior` then is `cc=fx·pix_x (pix_x=0.005)`, `xh=(cx−imx/2)·pix_x`, `yh=(imy/2−cy)·pix_y` (`pix_y=cc/fy`), so `(imx/2,imy/2)` is the sensor centre. Distortion converts as `k1=k1_cv/cc², k2=k2_cv/cc⁴, k3=k3_cv/cc⁶, p1=p2_cv/cc, p2=−p1_cv/cc` (`calibration_import.py:147`); tray `k4…tau` must be zero (door C resample otherwise). `Glass` is air `1/1/1/1` along `+Z` for now.

**What `rvec0/tvec0` is.** `ret,K,dist,rvecs,tvecs = cv2.calibrateCamera(objp,imgp,…)` returns **per-plane** `rvecs[t],tvecs[t]` (plate `t` → camera). We take `t0=idx_of_00000000` (the `000000` plate at datum `ix=2,iy=3→0,615,0`) as the *anchor*:

```python
rvec0,tvec0 = rvecs[idx0]                      # plate at world origin
cal,_ = calibration_from_opencv(K,dist,rvec0,tvec0, imx=2560,imy=2048,pix_x=0.005)  # → .ori
```

That `.ori` *is* the OpenPTV camera pose: `Exterior C/dm` from `rvec0/tvec0`, `Interior xh,yh,cc` from `K`, `AddedPar k1…p2` from `dist`. Any other `t` would give a valid `C/dm` for *that* plate position; anchoring on `000000` merely says “world = plate at `000000`”. `Cmean` over all `t` (`C=−Rᵀt` per plane) is printed for sanity; the joint `solve_opencv_multiview` loop replaces the single anchor by triangulating true `Zs` (`Pc` DLT `8×4` `A=[yP2−P1;P0−xP2]`, `X=V[-1]/w`) and re-optimizes with `CALIB_USE_INTRINSIC_GUESS`.

**Where do the “new 3D positions” for verification come from?**

* *Single-camera check:* the same `ref_pts` (`X=ix·120,Y=iy·120,Z=0` plate-local) you stored in 3.5b, transformed by the *solved* `rvecs[t]/tvecs[t]` per plane. No new triangulation.
* *Multi-camera check:* `solve_opencv_multiview`/`bundle` first triangulates the lattice in 3D using the *current* `Pc` (4-cam DLT, `plate_calibration.py:98 P=[M|p]`, `X=V[-1]`), producing `P_planes[t]` with solved `Zs` (plane `0` pinned at `Z=0`). Those `X_3D` are the “new” positions.

**Verification = reproject with OpenPTV and compare to the original `xy` pairs.** For every stored pair `(XYZ, xy_detected)`:

```python
# OpenPTV forward model (src/openptv2/algorithms/imgcoord.py / trafo.py)
metric = pixel_to_metric(x,y,cpar)          # → (x_m,y_m)
X_cam = R_cv·(XYZ − C)                      # R_cv = (dm·S)ᵀ
x_proj, y_proj = -cc·X_cam[:2]/X_cam[2]     # pinhole
x_proj, y_proj = distort(x_proj,y_proj, k1,k2,k3,p1,p2)  # Brown
x_pix, y_pix = metric_to_pixel(x_proj,y_proj,cpar)      # → pixel
resid = hypot(x_pix−x, y_pix−y)
```

`scripts/verify_plate.py` does this for *all* planes/cams and reports `RMS` (mean `resid`), `RCM` (ray-convergence miss across `1-4` for the same `id`), planarity `SVD` residual on `P_planes`, pitch via `cKDTree` median, and `|C_a−C_b|`. If the conversion were wrong (`S` on left, `+0.5 px` centre, `p1/p2` swap, `cc` sign) you see a *systematic* field (not random): e.g. `+0.5 px` bias ~0.5 px everywhere, `S` on left → `8e7 px` blow-up (`calibration_seed.py:57` assert). Round-trip `calibration_import:174 opencv_from_calibration` must reproduce `K/dist/rvec/tvec` within `1e-9 px` when `xh=yh=0`.

```bash
uv run python scripts/verify_plate.py --cals openptv_illmenau_4cam/cal --points-dir openptv_illmenau_4cam/cal
# checks per-plane planarity SVD residual, pitch via cKDTree median, cross-camera RCM, |C_a−C_b|
uv run python scripts/import_calibration.py --model-dir ... --points-dir ...  # alternative door
```

GUI/batch now sees `parameters_Run1.yaml` (`num_cams 4`, `cal/cam*.tif.ori`) + `rig.yaml`.

---

## 4. Files

* `scripts/detect_illmenau_4cam_part1.py:43` `find_plate_roi`, `158` `reject_outside_grid_v2`
* `scripts/calibrate_illmenau_4cam_part2.py:load_pairs`
* `src/openptv2/plate_labeler.py:67` `label_coded_6x7` / `156` `label_uncoded_grid`
* `src/openptv2/calibration_import.py:33` `calibration_from_opencv` (verified `S` on right)
* `src/openptv2/calibration_seed.py:80` `dm_from_lookat` / `137` `seed_from_lookat` / `256` `seed_from_dlt`
* `src/openptv2/plate_calibration.py:21` `solve_opencv_multiview` (flat-Z0 → stereo → DLT Zs → recalibrate, the `manual_openptv_orientation_from_opencv_pipeline.html` loop)
* `openptv_illmenau_4cam/rig.yaml:11` look-at seed, `cal/cam*.ori/.addpar`, `cal/collections.npz`

---

## 5. Limitations & next

* Hand-held `Z` per plane is `0` in object files; true `Z` is solved only in `solve_opencv_multiview` via 4-cam DLT triangulation and reuse via `CALIB_USE_INTRINSIC_GUESS`. With `cv2` 5.0 the current RMS `1.4–3.0` reflects the unmodeled perspective of the `6×7` affine labeler on tilted views — a homography labeler and joint `bundle` with `k1,k2,p1,p2` (and `interf` for splitter) will bring `RMS<1.0` and `RCM<0.1 mm` (`docs/calibration_best_practices.md:6`).
* `cv2.findCirclesGrid` on the ROI is logged but not yet used as primary detector; it already succeeds when the ROI is tight and can replace `target_recognition` entirely for the plate by injecting a `SimpleBlobDetector` tuned to `60 mm` via `blobDetector=` kwarg.

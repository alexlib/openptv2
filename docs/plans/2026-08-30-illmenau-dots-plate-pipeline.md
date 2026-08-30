# Illmenau dots-plate pipeline: hand-held multi-plane → OpenCV → openPTV

**Status (2026-08-31):** cameras **1–4 are calibrated and accepted**. The route
taken differs from the original plan in two substantive ways — see §0. The
delivered procedure is `docs/illmenau-4cam-calibration.md` (tutorial, numbers)
and `.claude/skills/openptv-multiplane-calibrate/SKILL.md` (reusable recipe);
the code is `scripts/illmenau/`. **Next: cameras 5–8 on the −Z wall (§8).**

Companion to `2026-08-30-calibration-hub-multi-source.md`, which stays the
source-agnostic spec (four doors, verified OpenCV→`.ori` conversion `S` on the
right, `calibration_import.py` / `presorted` / `rig.yaml` seed).

---

## 0) What actually happened vs what was planned

Kept from the plan: the L-coded labeller, the datum-anchored world, the
algebraic OpenCV→`.ori` tail (`calibration_from_opencv` → `angles_from_dm` →
`Calibration.to_file`), and `presorted` (the L code makes `sortgrid` unnecessary).

Two things the plan got wrong, both found by measurement:

**(a) The plan assumed distortion should be fitted.** It must not be, from a
single plane. `k1,k2,k3` trade against pose there and the solver returns a
polynomial that fits the 42 points and diverges elsewhere. Signature: the
projection of a straight 3D ray **doubles back inside the sensor**, impossible
for a pinhole. This made cameras 2 and 4 miss by up to 13 px while their own
reprojection RMS sat at 0.5 px. The delivered `.ori` are **pure pinhole, zero
`.addpar`**. Two independent tests say no distortion term is warranted: the
radial residual component (0.27–0.35 px) is smaller than the tangential
(0.44–0.46 px) and flat in radius; and `k1` improves multi-plane consistency by
4 % (within noise) while worsening reprojection RMS.

**(b) The plan treated `cc` as known (`focal_mm` from the lens).** `cc` is
*exactly degenerate* on a single plane — re-fit the pose at another `cc` and the
reconstruction is self-similar, with identical residuals and identical recovered
pitch. It **is** well determined by multi-plane cross-camera consistency, which
gave **8.586 mm against a nominal 9.44 mm**. `rig_1-4.yaml` still carries
`focal_mm: 35`, which is wrong by ~4×; do not seed from it.

Also dropped: the hub's `joint_plate_bundle_adjust` route. A full bundle over
all frames converged badly (37 → 13.9 px) because half the frames are
mislabelled; the 1-D consistency fit is both simpler and correct. `S3
solve_opencv`'s flat-`Z0` `stereoCalibrate` loop was not needed either.

---

## 1) Frame and datum (settled)

```
+X left → right,  +Y bottom → top,  +Z object → camera     (right-handed)
```

**World origin = the coded L-corner dot of frame `00000000`** — labeller grid
`(ix,iy) = (2,3)`, block **point id 21**, confirmed from the three coded dots at
`(2,3)/(2,4)/(4,3)`. Not the plate centre. In the barrel frame that dot is
`(0, −3580/2 + 615, 0) = (0, −1175, 0)`, so `barrel_from_plate(P) = P + [0,−1175,0]`.

Point id = `iy·6 + ix + 1`; `cal/calibration_block.txt` runs `1 −240 −360 0` …
`42 360 360 0`. Recorded in `plate.yaml:datum`.

The `run3` `−Z` camera confusion in the old §2 is resolved: it was a lab-frame
difference, and the frame above is now declared explicitly and drawn by
`scripts/illmenau/draw_rig_global.py`.

## 2) Pipeline as built

```
Kalibrierung_{1..4}/{frame}_*.tiff
  → detect_plate.detect_plate_targets  (ROI + negative + coded-dot classification)
  → plate_labeler.label_coded_6x7      (L corner → (ix,iy) → point id)   [scripts/illmenau/detect_plate_frames.py, cached to npz]
  → fit ONE shared cc from multi-plane cross-camera consistency          [fit_plate_cc.py]
  → solvePnP on the reference frame, zero distortion
     → calibration_from_opencv → angles_from_dm → .ori/.addpar           [refit_plate_pinhole.py]
  → GATE A  triangulate the plate: plane, pitch, ABSOLUTE positions      [check_plate_triangulation.py]
  → GATE B  epipolar curve vs the corresponding dot, all ordered pairs   [check_epipolar.py]
  → whole-dataset sweep, accuracy vs depth                               [check_all_frames.py]
```

**Never gate on per-camera reprojection RMS.** It stayed at 0.5 px through every
wrong intermediate result.

## 3) Result for cameras 1–4

`cc = 8.586 mm` shared, zero distortion:

| cam | C (X, Y, Z) mm | reproj RMS |
|---|---|---|
| 1 | (1470, 137, 3060) | 0.565 px |
| 2 | (1482, 2253, 3001) | 0.518 px |
| 3 | (−1472, 137, 2968) | 0.579 px |
| 4 | (−1500, 2254, 3007) | 0.531 px |

- Gate A: normal `(0,0,1)`, planarity RMS **0.306 mm**, pitch +0.68 % X /
  +0.10 % Y, absolute error **1.09 mm** median (no alignment applied)
- Gate B: **0.11–0.30 px median**, 0.93 px worst, straight inside the sensor for
  every point of all 12 ordered pairs
- Accuracy vs distance from the reference plane: 0.36 mm (<1 m), 0.87 mm (1–2 m),
  2.2 mm (2–3 m), 3.9 mm (3–5 m)

## 4) Observation volume — a display parameter that looks like a calibration bug

`algorithms/epi.py:epi_mm` computes two *endpoints*, walking the ray to
`Z = Zmin_lay` / `Zmax_lay` interpolated in X across `X_lay`. The line never
changes; the box picks which piece is drawn. **`Zmax` must stay well below the
nearest camera** — past the camera plane the endpoint projects to absurd
coordinates and throws the segment across the image.

`parameters_Run1.yaml` is set to `X_lay ±1500`, `Zmin_lay −1500`,
`Zmax_lay +1500`, layer values equal. Measured for the origin dot, pair 1→2:
±500 → 300 px segment (hits), **±1500 → 786 px (hits)**, ±2500 → 49 911 px (misses).

## 5) Known limitation — the labeller

Over all 48 frames the planarity RMS median is 17.5 mm, but that is a labelling
problem, not calibration. Splitting by a per-camera PnP fit of the rigid plate
(no cross-camera information):

| | n | planarity RMS median |
|---|---|---|
| labelling self-consistent (PnP < 1 px) | 23 | 2.70 mm |
| labelling broken (PnP ≥ 1 px) | 24 | 26.57 mm |

Roughly half the captured data is wasted. The 180° relabelling failure mode
(`id → 43 − id` maps a 6×7 grid onto itself) was checked for and does **not**
occur; the errors are individual dots misassigned on steeply tilted or partly
occluded views. **Improving `label_coded_6x7` for tilted views is the
highest-value next change** — it would roughly double the usable frames and
tighten the `cc` fit (current median cross-camera spread at the optimum is still
33.6 mm).

## 6) Files

*Delivered:* `scripts/illmenau/{detect_plate_frames, fit_plate_cc,
refit_plate_pinhole, check_plate_triangulation, check_epipolar, check_all_frames,
draw_rig_global}.py`, `docs/illmenau-4cam-calibration.md`,
`.claude/skills/openptv-multiplane-calibrate/SKILL.md`. Dataset paths come from
`ILLMENAU_RAW` / `ILLMENAU_DIR`. OpenCV is not a project dependency — the two
scripts that need it run under `uv run --with opencv-python-headless`.

*Already in place from the earlier phases:* `src/openptv2/{detect_plate,
plate_labeler, plate_calibration, calibration_seed, calibration_import}.py`,
`autocalibration.presorted`, `calibration_registry`.

*Not needed after all:* the `joint_plate_bundle_adjust` / Kabsch reconciliation
path for cameras 1–4 (but Kabsch is still needed for 5–8, see below), and
`interactive_plate_clicker.py` (the L code removed the need to click).

---

## 7) Verification checklist for any rerun

1. `uv run pytest tests/unit/test_autocalibration.py tests/unit/test_calibration_registry.py -v` and `uv run ruff check .` green.
2. Overlay the labeller's grid indices on one frame and confirm the datum dot is where you think it is.
3. `fit_plate_cc.py` shows a clean single minimum. A flat curve means the frames do not span enough depth — capture more.
4. Gate A: planarity < 1 mm, pitch within a few tenths of a percent, absolute error ~1 mm.
5. Gate B: every ordered pair under ~1 px median **and** straight inside the sensor.
6. Set the observation volume to the depth range where §3's accuracy table is acceptable.

## 8) Next: cameras 5–8 (−Z wall)

Nominal positions are already written down in `rig_1-4.yaml`'s trailing comment
(`(∓2528, 700/2900, −2528)`), but treat `focal_mm: 35` there as wrong — expect
`cc` near 8.59 mm again if the lenses match.

Two genuine differences from the 1–4 run, both to settle **before** fitting:

1. **The plate is seen from behind, so the grid is mirrored** — `+X` of the
   printed pattern runs right→left in those images. Flip the axis in the
   labeller or remap `ix → 5 − ix`, and **confirm the L code resolves to the same
   physical dot** as id 21. If it does not, the two camera groups will not share
   a world, and nothing downstream will reveal it.
2. **A common world.** If any frame is seen by cameras of both groups, use it as
   the shared reference and the whole 8-camera rig lands in one frame directly —
   this is by far the preferable option, so check the data for such a frame
   first. Otherwise calibrate 5–8 in their own frame and tie the two with a
   Kabsch fit on shared plate points
   (`calibration_import.similarity_from_correspondences`), rotating **cameras,
   never points**: `dm_new = A·dm_old`, `C_new = A·C_old + b`.

Then the procedure is unchanged: cache detections → fit one shared `cc` →
pure-pinhole `.ori` → Gate A → Gate B. Follow
`.claude/skills/openptv-multiplane-calibrate/SKILL.md`.

Afterwards, sequence with `correspondences.py:correct_frame` → `tracker.py` /
`pyptv_batch.py`; the `run3` rig and the `Multiview-Calibration` `cN_xyXYZ`
dataset remain reference-only (no LICENSE there).

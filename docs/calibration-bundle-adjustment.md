# Bundle adjustment in OpenPTV calibration

This page explains how OpenPTV calibrates today, why that leaves a cross-camera
gap ([RCM](calibration-rms-vs-rcm.md)), and how the **joint bundle adjustment**
closes it — plus the roadmap to a full self-calibration.

## 1. What OpenPTV does today: per-camera resection

The default calibration (`orient` / `full_calibration`, driven headlessly by
`autocalibration.calibrate_dataset`) is **space resection, one camera at a time**:

```
for each camera independently:
    known 3D calibration points  Xᵢ   (fixed, from the calblock file)
    detected pixels              xᵢ   (matched to Xᵢ by sortgrid)
    solve the camera's exterior (position, angles) + interior/distortion
    that minimize   Σᵢ ‖ xᵢ − project(Xᵢ) ‖²      ← this camera's RMS only
```

Each camera is fit against the **same, fixed** 3D points, but **separately**. The
distortion flag-set is chosen greedily by lowest RMS (`CANDIDATE_FLAGS`:
`cc,xh,yh` → `+k1,k2` → `+k3,p1,p2` → `+interf`).

**The gap:** there is no term in the cost that couples camera A to camera B.
Nothing asks whether their rays actually meet in 3D. So every camera can reach
sub-pixel RMS while the set triangulates poorly — high RCM, low RMS (the
[RMS-vs-RCM](calibration-rms-vs-rcm.md) divergence).

### Why "just bundle-adjust the plate" is subtle

The instinct is: optimize all cameras together against the plate. But if the plate
3D points stay **fixed**, a joint fit is **mathematically identical** to
per-camera resection — each camera's residuals depend only on its own parameters,
so nothing is coupled and RCM cannot change. A joint fit only helps when there is a
**shared unknown** linking the cameras. That shared unknown is the **3D points
themselves**.

## 2. Joint plate bundle adjustment (`joint_plate_bundle_adjust`)

The coupling is created by making the 3D plate points **free, shared parameters**:

```
free parameters:  every camera's exterior (position + angles)
                  + one shared 3D position per plate point   ← couples the cameras
minimize:         Σ_cameras Σ_points ‖ detected − project(point, camera) ‖²
                  + reg_weight · Σ_points ‖ point − nominal_calblock ‖²   ← anchor
```

Now moving a shared point changes **every** camera's residual at once, so the
optimizer trades camera poses against a common geometry — it can only lower total
reprojection by making the cameras **mutually consistent**. That is exactly what
lowers RCM.

- **The anchor term** (`reg_weight · ‖point − nominal‖`) is not optional. Without
  it the problem has a 7-DOF **gauge freedom**: you can translate/rotate/scale the
  whole cloud-plus-cameras with zero reprojection change, and the solver drifts.
  Anchoring every point to its manufactured calblock coordinate both fixes the
  gauge and encodes trust in the known plate geometry. (`reg_weight ≤ 0` is
  rejected.)
- **Distortion is held fixed** at the per-camera-fit values in this first cut —
  only exteriors and points move (see §4 for why distortion is added later, and
  one group at a time).
- It reuses the same machinery as the dumbbell bundle adjustment
  (`ptv_calibration.dumbbell_ba_residuals`), generalized from 2 dumbbell points to
  the N-point plate.

### Usage

```bash
uv run python skills/openptv-calibrate/scripts/calib.py run <dataset> \
    --output report.json --joint-ba          # preview: prints RCM before -> after
uv run python skills/openptv-calibrate/scripts/calib.py run <dataset> \
    --output report.json --joint-ba --write  # writes refined .ori/.addpar IF improved
```

`--write` only overwrites when RCM actually drops (backups: `*.jointba-bck`).

```python
from openptv2.autocalibration import joint_plate_bundle_adjust
new_results, info = joint_plate_bundle_adjust(results, cpar)
print(info["rcm_before"], "->", info["rcm_after"])   # median mm
```

### When it helps — and when it doesn't

On a rig whose cameras are **misaligned relative to each other** (exterior
inconsistency), the joint BA pulls them back together and RCM drops sharply — on a
perturbed synthetic rig it took median RCM **0.77 mm → ~0**.

But on `TT13_aorta` the joint BA moved RCM only **0.076 → 0.075 mm (~1%)**. That
near-zero gain is itself the diagnosis: the residual RCM there is **not** exterior
misalignment — it is shallow parallax + detection noise + the distortion model's
ceiling, none of which exterior+point BA can fix. **A BA that doesn't help tells
you where the error is coming from.** The remaining error needs §3/§4, not more of
§2.

## 3. Roadmap — driving RCM down further

The full plan lives in
[`docs/plans/2026-07-24-cross-camera-rcm-calibration-report.md`](plans/2026-07-24-cross-camera-rcm-calibration-report.md).
Status: RCM reporting **done**, joint-plate BA first cut **done**, the rest below
**TODO**.

### The full-blown self-calibration (target design)

Today we use **forward reprojection only**. The full version treats the calibration
body as a cloud of 3D dots and closes the loop both ways:

1. **Forward + backward residual.** Keep forward reprojection (3D→image), and add
   the backward term: back-project each detection to a ray (`ray_tracing`) and
   penalize the **ray-to-ray / ray-to-point miss** — i.e. make **RCM itself a
   residual**, not just a report.
2. **Epipolar-distance term.** For each point seen by ≥2 cameras, add the
   point-to-epipolar-line distance (`epi.py`) as a cheap, stabilizing image-space
   check — keeping the fit consistent with the correspondence geometry that
   tracking/stereo-matching later rely on.
3. **Full "shaking" of the calibration files.** Let exteriors, the glass/interface
   vector, the distortion terms, **and** the 3D cloud all float, minimizing (1)+(2)
   jointly, with the plate nominal as a soft prior. Real tracer particles (below)
   extend the cloud into the depth the plate can't reach.
4. **Add parameters one group at a time.** Freeing exterior + glass + all of
   `k1,k2,k3,p1,p2,scale,shear` + 3D points at once is **ill-conditioned** —
   parameters trade off against each other (`cc` vs depth, `k`-terms vs principal
   point) and the fit goes unstable. Instead **greedily**: free one new group,
   re-solve, and **accept it only if RCM (on a held-out split) improves** — else
   roll back. Same idea as the existing greedy `CANDIDATE_FLAGS`, but gated on
   cross-camera RCM instead of single-camera RMS.

### #2 — tracer self-calibration ("shaking"), iterated

The highest-value fix for shallow-parallax rigs, because the plate can't reach the
measurement depth but the particles can:

```
plate-calibrate → track particles → self-calibrate on triangulation disparity/RCM
at real depth → re-track → repeat   (Shake-the-Box lineage)
```

Formalize the existing `calib.py snapshot-refine` into an iterated stage, reusing
the §2 residual with the tracer cloud in place of (or alongside) the plate cloud;
RCM is the convergence/stopping criterion.

## 4. Why this matters downstream

Lower RCM ⇒ tighter epipolar bands ⇒ fewer false correspondences ⇒ better
stereo-matching ⇒ longer, cleaner tracks and more accurate 3D
positions/velocities/accelerations. The calibration error propagates into every
Lagrangian and Eulerian result, so it is worth measuring (RCM) and minimizing
(BA / self-calibration) rather than trusting RMS alone.

## Summary

| step | couples cameras? | frees 3D points? | lowers RCM? | status |
|---|---|---|---|---|
| per-camera resection (default) | no | no | no | shipped |
| joint plate BA (`--joint-ba`) | **yes** (shared points) | yes | yes, if exterior-limited | shipped (first cut) |
| + backward/epipolar + one-at-a-time distortion shaking | yes | yes | further | TODO |
| tracer self-calibration (#2) | yes | yes (real particles at depth) | most, on shallow rigs | TODO |

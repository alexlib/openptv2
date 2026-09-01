# Two-subrig calibration: exploiting (and fixing) the test_cavity rig geometry

## Context

`test_data/test_cavity` uses a 4-camera rig where cam1-cam2 and cam3-cam4
behave like two short-baseline sub-pairs, not four symmetric viewpoints.
Evidence (frames 10001-10004, current committed `parameters_Run1.yaml`):

- **2-camera correspondence ("pair") composition is skewed toward the two
  within-subrig combinations**: cam1-cam2 = 18% of all pairs, cam3-cam4 =
  23% (together 41%, vs 33% expected if all 6 combinations fired equally).
  A short baseline means near-parallel epipolar geometry, which lets many
  unrelated 2D detections accidentally satisfy the epipolar tolerance --
  more candidate crossings, whether real or not.
- **Ghost rate by camera multiplicity** (no plausible neighbor within 2mm in
  an adjacent frame): 2-cam pairs 64%, 3-cam triplets 38%, 4-cam quads 16%.
  Ghost rate per pair *identity* is fairly uniform (58-70% for all 6
  combinations) -- cam1-2/cam3-4 aren't individually worse per match, they're
  just far more numerous, so they dump the largest absolute ghost count.
- **Per-camera z-sensitivity** (px of image motion per mm of true z motion,
  via `image_coordinates`): cam1=1.12, cam2=1.58, cam3=1.68, cam4=1.86,
  vs ~10.2-10.9 px/mm in-plane (x/y) -- roughly 7x worse depth sensitivity
  than in-plane, for every camera.
- **Earlier full joint bundle adjustment** (Phase 4b, all 4 cameras' exteriors
  free except a held gauge camera, fit against 706 quad-only points from
  these 4 frames): RMS dropped 35-60% on its own fit objective, but
  independent checks (z-sensitivity, reprojection against detected targets,
  and trajectory correctness via `rank_check.py`) showed **no measurable
  improvement** -- the correction was real but too small/generic to matter.

## Root-cause reframing

The within-pair short baseline is a physical fact; no calibration removes
it. But it points at *why* a flat 4-camera bundle adjustment under-helps:
the quantity that actually governs 3D depth precision here is the **relative
transform between the two rigid sub-rigs** (cam1-cam2 as a unit, cam3-cam4
as a unit) -- not each camera's individual exterior pose. A standard bundle
adjustment treats all 24 exterior DOF as independent and lets a shallow
calibration target under-constrain each camera's along-ray direction
roughly equally, diluting the one relationship that matters (inter-rig
depth) across many loosely-constrained parameters instead of concentrating
information on it.

## Phase 1 -- cross-pair-weighted joint self-calibration (cheap, do first)

Re-run the same joint bundle adjustment as Phase 4b (hold one camera as
gauge reference, jointly refine the other 3 exteriors + all free 3D points),
but source observations from **triplets and quads only**, not pairs. Every
triplet/quad point, by construction, includes cameras from *both* sub-rigs
(each sub-rig only has 2 members, so no 3-or-4-camera correspondence can be
within-subrig-only) -- this automatically concentrates the fit on
cross-rig-informative observations without needing per-pair bookkeeping.
Also increases the observation count from 706 (quad-only) to ~4000
(triplet+quad), which should better-condition the solve.

**Success criteria** (compare against the Phase 4b baseline numbers above):
- z-sensitivity (px/mm along z, per camera) increases meaningfully from the
  ~1.1-1.9 baseline.
- Independent reprojection error (`corres_check.py`, detected targets, not
  the fit's own points) improves from the ~1.53-1.54px baseline.
- `rank_check.py` wrong-pick rate improves from the ~51% baseline.

If none of these move, that's evidence the inter-rig transform is already
about as well-determined as this data permits, and Phase 2 is unlikely to
help either -- worth stopping and reconsidering rather than proceeding.

## Phase 2 -- rigid-subrig reparametrization (if Phase 1 helps)

Reparametrize the bundle adjustment around the 2-rigid-subrig structure
explicitly:

1. Calibrate cam1<->cam2's internal relative pose and cam3<->cam4's internal
   relative pose independently (well-conditioned for x/y, weak only for z --
   short baseline is not a problem for in-plane precision).
2. Hold each sub-rig's internal relative pose fixed.
3. Solve for a single 6-DOF rigid transform between the two sub-rig frames,
   using cross-rig triplet/quad observations. This is 6 free parameters
   aimed directly at the weak direction, instead of 24 DOF where the weak
   direction is buried inside 4 individually-underdetermined poses.

This requires new code in `autocalibration.py` -- there is currently no
rigid-subrig parametrization, only per-camera free/held exteriors
(`tracer_self_calibrate`'s `hold_cam` is a single camera, not a rigid pair).
Not a parameter tweak; a real (small) feature.

## Verification protocol (both phases)

Reuse the session's existing scratchpad diagnostics against the same 4
frames (10001-10004), applying each phase's calibration output to a fresh
copy of the experiment and re-running `--mode sequence` (detection doesn't
depend on calibration; correspondence and 3D positions do):

- `scale.py` -- per-camera px/mm along z (depth sensitivity)
- `corres_check.py` -- reprojection error against detected targets
  (independent of whatever the fit optimized against)
- `ghost_check.py` / `pair_asymmetry.py` -- ghost rate and pair-composition
  skew, to see if better depth conditioning reduces spurious pair matches
- `rank_check.py` -- fraction of tracker links that picked a farther
  candidate over a closer available one (the actual trajectory-quality
  metric, not a proxy)

Do not apply either phase's calibration to the committed
`test_data/test_cavity/cal/*.ori`/`.addpar` files until the verification
protocol shows genuine improvement on the *independent* checks, not just
the fit's own optimization objective -- Phase 4b's history of a real-looking
fit objective improvement with zero independent-metric improvement is the
reason this protocol exists.

## Results (both phases executed, 2026-08-14)

| metric | baseline | Phase 1 (18-DOF, triplet+quad obs) | Phase 2 (rigid 6-DOF) |
|---|---|---|---|
| z-sensitivity cam1 (px/mm) | 1.12 | 1.05 | 1.05 |
| z-sensitivity cam2 | 1.58 | **2.08** | **2.12** |
| z-sensitivity cam3 | 1.68 | 1.79 | 1.76 |
| z-sensitivity cam4 | 1.86 | 1.99 | 2.00 |
| reprojection median (independent, vs detected targets) | 1.53px | 1.52px | 1.54px |
| ghost rate: pairs / triplets / quads | 64% / 38% / 16% | 63% / 36% / 16% | 65% / 38% / 18% |
| trajectories (4 frames) | 731 | 747 | 723 |
| `dz/dxy` | 1.90 | 1.98 | 1.97 |

**Both phases plateau at essentially the same answer.** Phase 2's 6-DOF
rigid parametrization was built on the hypothesis that Phase 1's 18 free
DOF diluted the one direction that matters (inter-rig transform) across
too many loosely-constrained parameters. That hypothesis predicts Phase 2
should do meaningfully *better* than Phase 1 on the same data. It didn't --
the two land within noise of each other on every metric, including the
z-sensitivity gain the fit was aimed at (cam2 +32% vs +34%, cam3 +7% vs
+5%, cam4 +7% vs +8%, cam1 -6% vs -6%). That means Phase 1's extra DOF
weren't actually being wasted on this data; the limiting factor is **data
quantity/diversity** (4 frames, one volume pass, no depth-spanning tracer
set), not the parametrization.

Neither phase moved the metrics that determine trajectory quality:
independent reprojection error, ghost rate at any camera-multiplicity
level, or trajectory count/kinematics -- all flat within noise. Per the
stopping rule above, **neither calibration was applied to the committed
files.** The z-sensitivity gain, while real for cam2 (+32%), is a ~1.3px/mm
change against a ~7x in-plane/z gap -- nowhere near closing it.

**Conclusion:** for this dataset (4 frames, current seeding density),
recalibration is not the lever that fixes the SNR/z-noise problem behind
short trajectories. The tracker's greedy-vs-nearest algorithmic gap
(documented elsewhere this session: ~51% of links pick a farther candidate
over a closer available one at a loose gate) remains the more promising
remaining lever. Revisiting this plan would need genuinely new information
-- more frames, or tracer particles spanning more of the volume's depth --
not a different fit of the same 4-frame data.

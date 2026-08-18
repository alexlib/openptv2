# Holistic 3D-PTV Systems: Co-Designing Calibration and Tracking

*Research program plan — started 2026-08-18, growing out of
`docs/plans/2026-08-17-lagrangian-accuracy-program.md`'s tracker-survival
work. That plan is the day-to-day working log; this document is the
longer-lived research direction it fed into.*

## 0. The claim this document makes

Every current 3D-PTV pipeline — openptv2 included — runs as a **strict
sequential chain**: calibrate once, detect, solve correspondences once,
triangulate, track. Each stage is optimized against its own local
objective (calibration minimizes reprojection RMS; correspondence-solving
maximizes match count within an epipolar tolerance; tracking maximizes
link yield), and errors only ever flow *forward*. Nothing downstream is
ever allowed to inform anything upstream.

This document's claim, grounded in concrete evidence gathered this week
(§1), is that **this one-way structure is itself a source of avoidable
error** — not a neutral engineering convenience. A long, kinematically
consistent, multi-frame trajectory is strong evidence that the calibration
and correspondence decisions *along that trajectory* were correct; a
sequential pipeline computes that trajectory and then throws the evidence
away instead of using it to refine the calibration that produced it. The
research direction here is **closing that loop**: tracking and calibration
informing each other, not just calibration informing tracking once at the
start.

This is explicitly **not** the "Shake-The-Box" direction (Schanz, Gesemann,
Schröder — iterative particle reconstruction that shakes candidate 3D
positions to minimize image residual, still within one calibration, still
before tracking runs). STB improves the detection→triangulation link within
a fixed calibration. This program is about the calibration↔tracking link
itself, and about which calibration *model* (not just which calibration
*fit*) is right for a given rig and flow.

## 1. The evidence base (what we actually found, this week, on this rig)

Everything below is measured, not hypothesized — reproducible via
`scripts/adapt_proptv_dataset.py --realistic`, `scripts/bench_proptv_kinematics.py`,
and `scripts/sweep_proptv_noise.py`.

1. **A real sign bug in the shared multi-media ray-tracing code** (fixed
   2026-08-18) silently mirror-flipped the back-projected ray direction for
   any camera looking the "other way" through the glass-normal convention.
   Forward reprojection (`img_coord`) was unaffected — calibration RMS
   looked fine — but 3D triangulation (`point_position`/`ray_tracing`, the
   inverse operation) put points 30-50× the true particle spacing away from
   ground truth. **A calibration bug can be invisible to the metric
   (reprojection RMS) used to certify the calibration, and only show up as
   tracking damage two stages downstream.**
2. **This rig has poor depth conditioning** (the ray-tracing investigation's
   supporting finding): a 0.15px detection-noise perturbation, propagated
   through triangulation, produces ~0.3mm 3D position noise — an
   amplification factor set entirely by camera baseline/distance geometry,
   not by anything in the tracker.
3. **A univariate noise-source sweep** (`scripts/sweep_proptv_noise.py`,
   see `docs/plans/2026-08-17-lagrangian-accuracy-program.md`'s Phase 3
   section for the full table) ranked damage, holding everything else
   fixed: **calibration residual (catastrophic) ≫ detection noise (severe)
   > particle-image merging (moderate) > missed-detection dropout (mild)**.
   Calibration residual — a systematic, per-camera, whole-run bias, not IID
   frame noise — was by a wide margin the dominant lever, and it was also
   the least-tuned parameter going in. **The upstream stage everyone treats
   as "solved once and fixed" is the one doing the most damage.**
4. **4BE's specific, confirmed failure mode** (§2 below): a genuine
   identity swap between two *real*, well-supported 3D points, not a ghost.
5. **A back-projection consistency filter** (`scripts/hybrid_backprojection_tracker.py`,
   §3 below), built and tested as a candidate fix, shows *why* a purely
   downstream, tracking-side filter has a structural ceiling: it reprojects
   candidates using the same (already slightly wrong) calibration that
   produced them, so it can catch a point with genuinely weak real support
   but not a point that is systematically, consistently wrong in the same
   direction as everything else derived from that calibration.

Point 5 is the crux of why this document argues for co-design rather than
"add one more downstream filter": **a downstream filter built on top of an
imperfect calibration inherits that calibration's blind spot.** The fix has
to reach back into the calibration itself.

## 2. Case study: 4BE's inherent vulnerability

`priority_segment_3d` (3MA) and `4be` are both pure 3D-only linkers — they
never touch 2D image space, so a spurious 3D point (real support or not)
looks identical to a real one once triangulated. On the "mild" realistic
dataset, 4BE's acceleration kurtosis was **K_a=761** against a truth of
19.80 — 25-30× worse than 3MA/trackcorr/myptv at the *identical* noise
setting (K_a 25-29).

Traced directly (not inferred): predicted track 260 correctly follows true
particle 42303 for frames 13→14 (position error 0.11-0.14, noise-level),
then at frame 15 its nearest true match switches to a *different* real
particle, 46936, at distance 2.85 — far outside noise. Both p1 (true
continuation) and p2 (wrongly accepted) are real, correspondence-solved 3D
points with real multi-camera support; a back-projection filter (§3)
cannot distinguish them, because there is nothing structurally wrong with
either point in isolation.

4BE's distinguishing mechanism (Ouellette et al. 2006, eq. 12) is scoring a
frame-*n*→*n+1* candidate by how well it predicts a *real particle two
frames ahead* — its whole design is "trust a candidate more if something
plausible exists past it."

**Proven, not just hypothesized** (2026-08-18, live-traced at this exact
junction via a decompiled debug build of `track_kernels_track3d.py`): the
correct candidate (k=481) was by far the closest match to the frame n+1
constant-velocity prediction (`cand_dist=0.315`, next-closest candidate was
6× farther) with n+2 support distance 0.72. The wrongly-accepted candidate
(k=464, the same point 3.069/-8.536/10.565→1.877/-9.877/6.021 traced above)
was **16× farther from the prediction** (`cand_dist=5.17`) yet had a
*slightly smaller* n+2 support distance, 0.48. Because 4BE's cost for a
supported candidate was `sup_dists[0]` alone (eq. 14, literally, with
nothing else in the comparison), the kinematically absurd candidate won on
a coincidence: some real particle happened to sit fractionally closer to
its bad n+2 extrapolation than the correct particle's real continuation
did. The extra-frame lookahead that is 4BE's whole advantage over 3MA on
clean data is exploitable exactly because it was used as a *substitute*
for prediction-consistency rather than *combined* with it — the same
AND-gated-evidence discipline trackcorr's two-hop acceptance already has
(neither hop can compensate for the other failing).

**Fixed same day** (`track_kernels_track3d.py`, `track4be_loop_fast`):
supported-candidate cost is now `sup_dists[0] + cand_dists[ci]` — n+2
support distance plus prediction-consistency distance, so a candidate
can no longer win purely by coincidence while grossly failing the other
criterion. Measured effect on the full 5-tracker benchmark (`mild`
severity, identical dataset/seed): 4BE's K_a dropped **761.45 → 103.09**
(7.4×), meanlen rose 21.77 → 25.56, moving 4BE from a 25-30× outlier among
the five survivor trackers to roughly proptv_tracking's range (K_a 73-103
vs. the other three's 25-29) — still worse than 3MA/trackcorr/myptv, an
open question for further work, but no longer a qualitatively different
failure mode from the rest of the field.

## 3. The back-projection filter's limited result, and why it matters

Built as directly proposed: run 3MA once, then for every point along every
resulting track, reproject it into all four cameras (same calibration the
tracker used) and require it land within `tol_px` of a real detected 2D
target in at least `min_cams` cameras; split the track wherever it fails.

- At "mild" severity: **zero effect** — every point already has ≥3-camera
  support. No ghosts to catch at this noise level; 3MA's modest K_a excess
  there (29 vs. truth 19.8) is not a ghost-contamination signature.
- At "severe" severity: the filter *increased* fragmentation (meanlen 2.73
  → 2.07) and *decreased* yield (0.092 → 0.075), with only a marginal K_a
  improvement (3.21 → 4.10, both still far from truth). At this severity
  the dominant damage is no longer "occasional spurious ghost," it's
  pervasive, systematic position bias from calibration residual and
  detection noise acting on nearly every point — which back-projects just
  as "consistently" as a correct point would, because the bias is baked
  into the same calibration used for the check.

**Conclusion the sweep and this filter jointly support:** a purely
tracking-side filter's ceiling is set by the calibration it inherits. It
can clean up isolated, weakly-supported ghosts; it structurally cannot
correct a systematic bias shared between the candidate-generation and the
candidate-verification step, because they're the same calibration. That is
the concrete argument for §4.

## 4. Toward co-design: closing the calibration↔tracking loop

Three concrete mechanisms, roughly in increasing ambition:

### 4a. Trajectory-informed calibration refinement (the direct answer to §3's ceiling)

`docs/calibration-bundle-adjustment.md` already establishes the key
technique this repo uses for camera-to-camera consistency: making the 3D
points **free, shared parameters** across cameras couples their residuals,
so the optimizer can only reduce total reprojection error by making the
rig mutually consistent (this is what closes the RMS-vs-RCM gap for static
calibration targets). The natural extension: **a high-confidence
trajectory segment is exactly this kind of shared constraint, extended
through time** — a particle's true path is smooth (bounded jerk, in a
turbulence-appropriate sense) whether or not any one camera's current
calibration says so. Feed accepted, temporally-consistent trajectory
segments back into the bundle adjustment as additional shared-point
constraints (in place of, or alongside, a static calibration target), and
the calibration itself absorbs some of the correction that a downstream
filter structurally cannot deliver.

This needs a trust mechanism — using a *wrong* trajectory to "refine" the
calibration would compound the error, not fix it — which is exactly why
§4c's declined-link bookkeeping matters here too: only stitch a
trajectory's calibration constraint from *forward+backward agreeing*,
undeclined segments (see 4b).

### 4b. Forward-backward tracking as a calibration-quality signal, not just a tracking-quality one

trackcorr already supports forward+backward passes with reciprocity
checking (`tracker.full_forward()` / `full_backward()` /
`enforce_reciprocity`) purely as a *tracking* quality control. The same
signal is calibration-informative: a trajectory segment that a
forward pass and an independent backward pass both agree on, with tight
reciprocity, is strong evidence the calibration was self-consistent along
that segment; a segment where forward/backward *disagree* is either a
tracking ambiguity or a *local* calibration/multimedia-model inadequacy
(e.g. a region where the multimedia LUT is a poor fit). Currently this
information is discarded once the reciprocity check passes/fails at the
tracking layer. It should be logged and fed to 4a as the trust signal for
which trajectory segments are allowed to inform calibration.

### 4c. Declined-link bookkeeping, carried through to calibration

Phase 3 of the accuracy-program plan already identifies this for tracking
alone: "whatever declines a link must record *why*, so a stitcher can
distinguish 'particle was not detected here' from 'I refused to guess
here'". The same distinction matters one level up: a link declined because
of genuine detection absence tells you nothing about calibration; a link
declined because two real candidates were both locally *plausible*
(competing within the dacc/angle gate) is informative about where the
calibration's residual is large enough to create ambiguity — a spatially
localized diagnostic 4a can use to know *where* in the volume the
calibration needs the most correction.

## 5. The calibration model question

openptv2's own calibration (`src/openptv2/algorithms/calibration.py`,
`orientation.py`) is an **extended-Tsai-family model**: per-camera exterior
orientation (position + 3 angles) plus interior orientation (principal
distance + principal point) plus Brown-Conrady-style radial/tangential
distortion (`k1,k2,k3,p1,p2` + `AddedPar`'s shear/scale terms), extended
with an explicit **multimedia (air/glass/water) refraction model**
(`Glass`, `MmNp`, `ray_tracing.py`) that a plain Tsai/DLT/pinhole model
does not have at all. Bootstrap is DLT (`calibrate_proptv_dlt.py`'s
approach, or the GUI's `orient`/`raw_orient`), refined by Gauss-Newton
bundle adjustment.

This is one point in a real design space, and the choice is not neutral to
the failure modes above:

- **Plain DLT / pinhole (no multimedia term)**: fastest to fit, no
  multimedia physics — wrong by construction for any tank/window
  experiment, silently absorbs refraction error into the pinhole
  parameters as a spatially-varying bias that *looks* like ordinary
  calibration residual but isn't correctable by refining pinhole
  parameters alone. A dataset run through this model would show the same
  §1.3 symptom (calibration residual dominates damage) for a structurally
  different, uncorrectable-in-that-model reason.
- **Tsai's original two-stage model**: closed-form radial-distortion-only
  solve, then refinement; well-conditioned for a single medium, doesn't
  extend naturally to multimedia without the kind of explicit ray-tracing
  extension openptv2 already carries.
- **Extended Tsai + multimedia (openptv2's own)**: correct physics for a
  tank/window rig, but exactly *because* it adds real refraction geometry,
  it's more sensitive to the depth-conditioning issue in §1.2 — the same
  physics that makes it correct also makes small angular/positional errors
  amplify along the weak-baseline axis. Points 1-2 above are symptoms of
  this specific model choice interacting with this specific rig geometry,
  not a universal 3D-PTV property.
- **Soloff/Willert-style general polynomial mapping** (a nonlinear
  polynomial fit of world→image directly, no explicit physical camera
  model): handles almost any real optical distortion including ones no
  physical model anticipates (housing imperfections, imperfect optics),
  at the cost of needing a denser, wider calibration volume to constrain
  the polynomial and offering no physically-interpretable failure
  diagnosis when it's wrong — you cannot ask "which physical parameter is
  off" the way you can with Tsai-family models, which matters directly for
  §4c's diagnostic use of declined-link bookkeeping.

**Research question, not yet answered**: does §4's trajectory-feedback
loop change which calibration model is the right default? A model with
fewer, more physically interpretable parameters (Tsai-family) is easier to
correct via 4a's trajectory constraints (the correction has an
interpretable target — e.g. "this camera's z0 is off by X"); a
general-polynomial model may fit noisy calibration data better in
isolation but gives the trajectory-feedback loop nothing physically
interpretable to correct. This is a concrete, testable comparison once 4a
exists.

## 6. Relationship to existing docs in this repo

- `docs/differentiable_ptv_autoresearch_architecture.md` and
  `docs/plans/differentiable_ptv_nextgen_plan.md` propose a parallel,
  complementary direction: replace the whole pipeline with differentiable
  soft operators (PyTorch) and backpropagate a physics loss end-to-end
  through all five stages simultaneously. That is a more radical
  restructuring (new differentiable implementations of every stage) aimed
  at the same root problem — sequential local optimization missing
  cross-stage coupling — via automatic differentiation instead of an
  explicit trajectory-feedback bundle adjustment. **This document's
  program is deliberately the more incremental path**: it works within
  openptv2's existing Cython algorithms (no new differentiable
  reimplementation required for 4a-4c), using the bundle-adjustment
  machinery `docs/calibration-bundle-adjustment.md` already built, and can
  ship value in phases without first building a full differentiable
  runtime. The two directions are not in conflict — a working
  trajectory-feedback loop here is exactly the kind of signal a future
  differentiable pipeline's physics loss would also want, and the K_a/
  a_rms/outlier-rate metrics this week's work already produces
  (`benchmark_utils.py`, `bench_proptv_kinematics.py`) are the same
  quantities `L_physics` in the differentiable plan is built from.
- `docs/calibration-bundle-adjustment.md` is 4a's direct foundation —
  extending its "shared free 3D point couples the cameras" mechanism from
  static calibration-plate points to dynamic trajectory points is the
  concrete next implementation step, not a new technique.
- `docs/plans/2026-08-17-lagrangian-accuracy-program.md` is this
  program's evidence source and day-to-day log; treat it as the working
  notebook, this document as the standing research direction it points to.

## 7. Phased roadmap

1. ~~**Instrument 4BE's actual n+2-support score**~~ — **done 2026-08-18**:
   proven (§2) and fixed (combined cost, K_a 761→103, 7.4×). Remaining gap
   to 3MA/trackcorr/myptv (K_a ~25-29) is now a smaller, open question —
   not yet root-caused, a candidate next audit target (does the same
   evidence-substitution pattern exist elsewhere in 4BE's conflict
   resolution, or is the residual gap a different mechanism entirely).
2. **Log forward/backward reciprocity agreement per trajectory segment**
   (4b) as a first-class output of `full_forward()`+`full_backward()`,
   not just a pass/fail gate — the prerequisite data for 4a.
3. **Extend `joint_plate_bundle_adjust`-style shared-point coupling to
   trajectory segments** (4a) — the smallest version: take the
   highest-confidence (forward/backward-agreeing, undeclined) trajectory
   segments from a completed tracking run, treat their 3D positions as
   additional shared free parameters, re-run calibration, re-run tracking
   on the refined calibration, measure whether K_a/a_rms move toward
   truth. This is the whole program's central, falsifiable experiment.
4. **Re-run the severity sweep (`sweep_proptv_noise.py`) through one
   iteration of step 3's loop** to see whether the calibration-residual
   damage ranking (§1.3) changes once tracking is allowed to correct the
   calibration that fed it.
5. **Only then**, revisit §5's calibration-model comparison: rerun the
   same rig/noise model through a plain-DLT (no multimedia) and a
   Soloff-polynomial calibration, with and without the step-3 feedback
   loop, to test whether the loop's benefit is model-dependent as
   hypothesized.

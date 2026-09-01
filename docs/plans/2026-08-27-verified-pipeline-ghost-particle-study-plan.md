# Plan: verify detection fidelity and the quad-uniqueness ordering-quirk (bug vs. feature) with synthetic ground truth

> **Update 2026-09-01:** After zarr-only cutover (`archive/2026-09-01-zarr-only-final-cutover-plan.md`, `6a1e81aa`), synthetic benchmarks and `detection_fidelity_check.py` should read ground truth from `RunStore` (`targets/`+`correspondences/`) consistently; Phase2 verdict `FEATURE` (+0.01-0.03 PR) stands.

Umbrella plan for the /goal stated 2026-08-27: a verified and validated
openptv2 that does correct detection, stereo-matching, and tracking — not
just "matches 3dptv.exe's GT," since GT is itself a heuristic tracker's
output, not metrological ground truth (see
`2026-08-27-track3d-beat-gt-plan.md`, "The goal, precisely stated"). This
plan is urgent per direct request and supersedes the "pause and decide
later" framing at the end of `2026-08-27-quad-uniqueness-pass-study-plan.md`.

Two findings from earlier today are now decisively quantified (that plan,
"Resolved, 2026-08-27" section) but not yet *understood*:

1. openptv2's detector finds **~22% more raw 2D targets per camera** than
   3dptv.exe's own detector, at identical threshold values
   (`img_3dptv/Cam{1-4}.{frame}_targets`, 3dptv's real output, now available
   in the wp1 dataset alongside the split raw images that produced them).
2. openptv2's clean quad-uniqueness dedup accepts **16% more quads** than
   3dptv's ordering-quirked dedup, even given *identical, correct* input
   detections.

Neither finding says which side is *right*. More targets/quads could mean
openptv2 recovers real particles 3dptv's detector or its buggy dedup
discards (openptv2 better) — or it could mean openptv2 is manufacturing
ghost particles from noise/split blobs/coincidental epipolar alignment
(openptv2 worse). Both findings need controlled, synthetic-ground-truth
verification, because real data (from either tracker) can never definitively
answer "is this specific extra point real" — only synthetic data with known
truth can.

## Existing infrastructure to build on (do not duplicate)

- `src/openptv2/benchmarking/metrics.py` — `_ghost_captures_in_frame`,
  `ghost_positions_from_frame_gt`: already defines what a "ghost" is
  (mismatched-identity correspondence) and how to detect one against known
  particle identity.
- `test_data/tracking_synthetic_dense/generate.py::build_fixture_with_correspondence`
  — already runs the REAL combinatorial correspondence matcher
  (`openptv2.algorithms.correspondences`) on noisy synthetic 2D targets, so
  ghosts arise from genuine epipolar ambiguity, not injected false
  detections. Documented real-world ghost rates from `test_cavity`:
  **pairs 64%, triplets 38%, quads 16%** ghost at that rig's density/baseline
  (`docs/plans/two-subrig-calibration.md`) — i.e. ghost quads are not a small
  edge case in this codebase's own prior measurements; they're the dominant
  failure mode at some densities. This context matters: wp1's quad ghost
  rate needs to be measured the same way and compared, not assumed novel.
- `docs/plans/2026-08-15-tracking-quality-overhaul.md` — Stage 0.5's whole
  point was building a ghost-inclusive benchmark because the ghost-free one
  "cannot show the dominant real-data failure mode." Read this in full before
  starting Phase 2 below — it may already contain relevant gates/methodology
  to reuse rather than re-derive.
- `scripts/generate_synthetic_images_from_targets.py` — closes the
  ground-truth → image → detection loop: renders synthetic TIFFs from a
  `_targets` file's exact sub-pixel centroids, runs detection, compares
  recovered centers to the originals. This is the tool for Phase 1 below,
  already built for exactly this purpose (used previously on `test_data/burgers`,
  per its docstring) — reuse it on `img_3dptv`, don't rewrite it.

## Phase 1 (urgent, most actionable — real data + tool already exist): detection fidelity

**Cheapest, highest-signal step first**: `img_3dptv` now has both 3dptv's own
raw split images *and* its own `_targets` output for the same 10 frames — an
unusually strong asset most detection-fidelity questions don't get (a real
detector's output paired with the real images it was computed from).

1. Run openptv2's `targ_rec` detector directly on `img_3dptv`'s real split
   images (same `targ_rec`/`detect_plate` parameters as the yaml — already
   confirmed identical to `targ_rec.par`) and compare per-target output
   against `img_3dptv`'s `_targets` files **one-to-one, not just by count**:
   nearest-pixel match each detected centroid to a 3dptv target, and classify
   into matched / openptv2-only (candidate ghost) / 3dptv-only (candidate
   miss). This is the direct analogue of `point_cloud_diff.py` but one stage
   earlier — 2D pixel space, not 3D — and answers "is the +22% real particles
   3dptv missed, or noise/split blobs openptv2 invented" using the actual
   images, no synthesis needed.
2. For openptv2-only detections: inspect their `sumg`/`nx`/`ny` (size,
   brightness) distribution vs. matched detections. A same-distribution
   population suggests real particles 3dptv's detector missed; a
   systematically dimmer/smaller/oddly-shaped population suggests noise or
   an over-eager peak-splitter cutting one real blob into two.
3. **Only if step 1-2 is ambiguous**, escalate to
   `generate_synthetic_images_from_targets.py`: render synthetic images from
   3dptv's own `_targets` centroids (known truth, by construction) and
   confirm openptv2's detector recovers exactly that count on a *clean*
   synthetic version of the same scene — isolates the detector algorithm
   from any real-image noise/artifact question entirely.
4. **Only if a real algorithmic gap is found** (not scope creep — a
   pre-condition): diff openptv2's peak-finding/connected-component code
   against `3dptv/src_c`'s (same treatment as `find_candidate_plus` and
   `four_camera_matching` got today) to find the specific divergence.

   **Steps 1-2 done, 2026-08-27** (`wp1_10_images/scripts/detection_fidelity_check.py`,
   run on `img_3dptv`'s real images against 3dptv's real `_targets`, all 10
   frames/4 cameras): 86.8% of 3dptv's own targets are matched by openptv2's
   detector (good agreement on real particles), but only 70.4% of openptv2's
   detections match a 3dptv target — 24,346 "ours-only" extras. These extras
   are **systematically dimmer** (`sumg` median 322 vs. 612 for matched,
   ~half) at similar size, and **spatially clustered near real particles**
   (60% within 5px of a matched detection, only 6.6% >10px away — a followup
   check beyond step 2's plan). This is not independent noise and not
   particles 3dptv missed — it's **openptv2's peak-finding/connected-component
   growth splitting a meaningful fraction of real blobs into a main peak plus
   a dim satellite fragment**, where 3dptv's algorithm keeps them merged.
   Step 3 (synthetic escalation) is no longer needed — the mechanism is
   already specific enough. **Proceed directly to step 4**: diff
   `_targ_rec_fast`'s region-growing/peak-splitting logic
   (`openptv2/algorithms/segmentation.py`, delegates to a compiled BFS) against
   3dptv's own `targ_rec`/`peak_fit` C source for the specific merge/split
   rule that differs (likely `discont`'s region-growing continuity check, or
   a local-maximum/watershed splitting criterion applied at blob edges).

   **Step 4 attempted and reverted, 2026-08-27** (`track_kernels_batch.py`'s
   `targ_rec_fast`): read 3dptv's `segmentation.c` in full. Its local-maximum
   seed check reads the SAME array its flood-fill zeroes as pixels are
   consumed (`img` in the C, confusingly opposite name from openptv2's
   `img`/`img0` pair); openptv2's `targ_rec_fast` reads the pristine,
   never-mutated `img` for that same check. Hypothesized this decoupling let
   secondary bumps near an already-claimed blob register as independent
   peaks. **Tried switching the seed check to read `img0` (the
   consumption-tracking copy) instead — this made it WORSE**
   (`ours_only` 24,346 → 25,657; rebuilt, re-ran
   `detection_fidelity_check.py` against `img_3dptv`, confirmed regression,
   reverted immediately, `git diff` confirms byte-identical to HEAD). The
   causal direction was backwards: an already-zeroed neighbor trivially
   satisfies `gv >= neighbor`, making local-max detection *more* permissive
   near consumed regions, not less. **The real mechanism is more subtle than
   a single-array swap and reasoning against a 512x512 real image with
   thousands of interacting blobs isn't a reliable way to find it.**

   **Revised approach for step 4, attempted 2026-08-27, ORACLE INVALIDATED**:
   built `scripts/literal_3dptv_targ_rec.py`, a hand-transliteration of
   `segmentation.c::targ_rec`, and `scripts/synthetic_two_blob_sweep.py`.
   Three synthetic sweeps (separation distance, brightness ratio, PSF
   sigma) and a single-blob+sensor-noise sweep all showed the literal port
   agreeing with openptv2's compiled `targ_rec_fast` at every single
   condition — apparently a strong null result (no divergence found).
   **This was wrong: the port was never validated against real ground
   truth before being trusted.** Running it on `img_3dptv`'s actual full
   image against 3dptv's own real `_targets` file (frame 100001, cam 1)
   gives `n=2551` — 54% MORE than 3dptv's real `n=1659`, and even more than
   openptv2's own `n=2152`. **The port has its own bug and was never a
   trustworthy oracle** — the earlier "no divergence" conclusions from the
   synthetic sweeps are RETRACTED; they show openptv2 agreeing with a
   broken reference, not with 3dptv. Process lesson: validate a
   hand-transliterated oracle against real, known-correct output BEFORE
   using it as ground truth for synthetic sweeps, not after — this is the
   same "reproduce against a trusted reference first" discipline that
   applies to any bug fix, and it was skipped here.

   **What remains valid and unaffected**: `detection_fidelity_check.py`'s
   direct comparison (openptv2's real detector vs. 3dptv's real `_targets`
   files, no transliteration involved) — the confirmed 22% raw-target
   excess, and the extras' brightness (~half) and spatial-proximity (60%
   within 5px of a real detection) signature. Those numbers don't depend on
   the port at all.

   **Methodology correction found while debugging the oracle, 2026-08-27**:
   `detection_fidelity_check.py` called `targ_rec` directly on raw,
   unfiltered TIFFs — but 3dptv's real pipeline (`jw_ptv.c:430`) runs a
   `highpass()` filter before `targ_rec`, and openptv2's production pipeline
   does too (`ptv.py::simple_highpass`, `hp_flag: true` in the yaml). All of
   this session's detection-stage numbers (the 22% raw-target excess, the
   brightness/proximity signature, every synthetic sweep) were computed
   WITHOUT this preprocessing step on either side of the *direct* comparison
   — an apples-to-oranges test, not a bug in openptv2 itself. Applying
   `simple_highpass` before `targ_rec` on the real image:
   `n=2293` (raw, no highpass) → `n=2152` (with highpass) — and `2152`
   **exactly matches `res/run.zarr`'s actual stored count**, confirming the
   production pipeline already applies this correctly and giving a trusted,
   apples-to-apples baseline for the first time. Swept the highpass filter
   size (3dptv's C default `sup=12` vs openptv2's `DEFAULT_HIGHPASS_FILTER_SIZE=25`
   and everything in between, 5–40): count stays in a tight 2152–2172 band
   regardless — **filter size is not the lever**, ruled out. The corrected,
   trustworthy gap is `2152` (openptv2, with highpass) vs `1659` (3dptv
   real) — **30% excess**, smaller than the uncorrected 22-38% range seen
   this session under various flawed comparisons, but still substantial and
   still unexplained.

   **What must happen before any further detection-stage work**: rerun
   `detection_fidelity_check.py` (pixel-space matched/ghost/miss
   classification, brightness/proximity signature) WITH `simple_highpass`
   applied first, across all 10 frames/4 cameras — the existing 24,346
   "ours-only" / 60%-within-5px numbers are from the uncorrected,
   no-highpass comparison and need to be re-measured on the corrected
   baseline before being trusted as the mechanism signature. Also worth
   checking whether 3dptv actually ran highpass for THIS dataset at all —
   `jw_ptv.c`'s highpass call is gated by a `mask` flag whose value isn't
   visible in the `.par` files read so far (`mask==2` triggers it in the
   no-mask default path per a code comment noting a changed condition,
   `"Beat April 090402 was ==0"` — meaning older 3dptv versions ran it
   unconditionally by default, current versions may not) — if 3dptv's own
   GT was produced WITHOUT highpass, the correct comparison baseline is
   openptv2 WITHOUT highpass too, and the `n=2293` raw number (not `2152`)
   is the one to reconcile against `1659`.

   **Next attempt at step 4** must not repeat the same mistake: before
   building any synthetic sweep, first find (or fix) a transliteration that
   reproduces 3dptv's real `n=1659` on the real `img_3dptv/Cam1.100001`
   image to within a small tolerance — treat that reproduction as the
   validation gate, not an afterthought. Likely bug candidates in the
   existing port worth checking first (cheaper than rewriting from scratch):
   the BFS queue's overflow behavior vs. 3dptv's fixed 2048-slot
   `waitlist[2048]` (the literal port's Python list has no cap — a runaway
   flood-fill that would silently truncate/corrupt in the real 2048-slot C
   array might instead complete correctly in the uncapped Python port,
   *systematically growing regions larger* and consuming more of the image
   before yielding, which could go either direction on final target count
   and needs checking empirically); and the per-camera `gvthres[nr]`
   indexing vs. the single scalar this port assumes (confirmed same value
   for this dataset, but worth a sanity check it's actually being applied
   at the right camera index). Once a validated oracle exists, only then
   redo the synthetic sweeps and trust their divergence-point findings.

## Phase 2 (urgent, needs new synthetic scenes): is the ordering quirk a bug or a feature?

3dptv's C offers no comment or documentation suggesting the
`++tim[c][p] > 1` short-circuit ordering is intentional — it reads as an
artifact of writing the check inline across 4 sequential `if` statements
rather than evidence of deliberate conservatism. But "reads like a bug" is
not proof; only a controlled test where the correct answer is known settles
it.

1. **Construct synthetic scenes with deliberate quad-candidate conflicts**,
   extending `build_fixture_with_correspondence`'s approach: place particles
   so that at least one true quad's own points also appear as an inferior
   candidate for a competing false quad (i.e. genuinely ambiguous cases, not
   just noise) — control the density/spacing to land in a regime comparable
   to wp1's actual ghost rate (measure it first per Phase 1's methodology,
   applied to 3D quads via `_ghost_captures_in_frame`, so the synthetic
   scene isn't tuned to an arbitrary density).
2. **Run both dedup variants** (clean = current `take_best_candidates`;
   buggy = the `2026-08-27-quad-uniqueness-pass-study-plan.md` ordering
   replica, already written in
   `wp1_10_images/scripts/replicate_3dptv_uniqueness_pass.py` — port its
   logic into a reusable function rather than copy-pasting again) against
   the synthetic scene's **known ground truth** (`row_gt`-style particle
   identity, per `build_fixture_with_correspondence`'s existing contract).
3. **Score both**: precision/recall/ghost_capture (reuse
   `benchmarking/metrics.py`) against the known truth, not against each
   other. This directly answers bug-vs-feature: if the buggy/quirked pass
   has *higher* precision at comparable recall (rejects more of the actual
   ghosts, keeps the real quads), the "bug" is functioning as an accidental
   conservatism feature worth keeping or reimplementing deliberately. If it
   has lower precision or recall than the clean version at the same density,
   it's genuinely just a bug openptv2 already fixed by accident, and no
   further reconciliation with GT parity is warranted at the correspondence
   stage.
4. **Sweep density/ambiguity level**, not just one scene — the two passes
   may trade places as ambiguity increases (a common shape for this kind of
   result: clean wins at low ambiguity, quirked wins as conflicts compound).
   Report the crossover point, if any, so the eventual decision (Phase 3) can
   be conditioned on wp1's actual measured density rather than a single
   synthetic point.

## Phase 2 RESULT, 2026-08-27: the ordering quirk is a FEATURE, not merely a bug

Harness: `scripts/quad_dedup_bug_vs_feature.py`. Synthetic scene, particles
sampled uniformly INSIDE the search volume (see the harness docstring for why
`build_scene`'s spacing-based scatter was unusable: ~40% of its particles fall
outside this fixture's `vpar` Z range, so the matcher correctly refuses them
and recall is measured against a denominator of unobservable particles).
Exact identity ground truth; a quad is correct iff all 4 targets come from the
same true particle.

**Design point that matters**: the quirk is *provably* strictly more
restrictive (it burns points of rejected candidates, so it can only reject
more, never accept more — asserted as gate 2 and confirmed). So "buggy has
higher precision, lower recall" would be a meaningless finding — that is just
a point on a tradeoff curve any matcher reaches by raising `corrmin`. The
experiment therefore compares the quirk against **clean dedup tightened (by
bisection on `corrmin`) to accept the SAME number of quads**.

Validation gates, all passing before any result was read:
1. sparse scene → both variants identical, precision 1.000, recall 1.000;
2. buggy ≤ clean accepted, always (the monotonicity the mechanism predicts);
3. ghost rate rises with density (0.000 → 0.030 → 0.090), i.e. the scene
   really does produce the ambiguity the question is about.

**Verdict — quirk beats plain `corrmin`-tightening 8/10, ties 2, loses 0**,
margin growing with density (~0 at n=60 where no conflicts exist, +0.007-0.014
at 150, +0.011-0.025 at 300, +0.009-0.029 at 500). Robust across noise_px
∈ {0.5, 1.0, 2.0} × corrmin ∈ {20, 33, 50} (9/9 QUIRK at n=300). At the matched
operating point the quirk dominates on **both** precision and recall (n=500
seed1: quirk P=0.949 R=0.860 vs tightened P=0.921 R=0.834), which a pure
conservatism knob cannot do.

**Why this is coherent**: the quirk burns targets belonging to *rejected*
high-`corr` candidates — i.e. targets sitting in contested/ambiguous
neighborhoods. That is **conflict-topology information**, which a global
scalar quality threshold (`corrmin`) structurally cannot express. Accidental
in origin, real in content.

**Crucial nuance — this is NOT a free win, and does not by itself justify
flipping the default.** Against today's *untightened* clean dedup, the quirk
finds fewer correct quads in absolute terms (n=500 seed1: clean 452 correct /
48 ghosts; quirk 430 correct / 23 ghosts). So adopting it trades ~22 real
particles for ~25 fewer ghosts. Whether that trade is good is a **downstream
tracking question this experiment does not answer** — though this session's
own tracking findings (ghost 3D points are what seed the jump/spliced
trajectories, and high-acceleration trajectories correlate strongly with
low GT-match fraction: 0.187 vs 0.451) suggest ghosts are the more expensive
error for trajectory quality. That must be measured, not assumed.

**Recommended implementation shape (not yet done, needs the decision below)**:
implement the *principle*, not a quirk-port — "a target contested by a
rejected high-quality candidate is down-weighted/blocked for later, lower-
quality candidates" — in `correspondences.py::take_best_candidates`, gated
opt-in so no existing dataset's behavior changes silently. A literal port of
the `++tim[...]` ordering artifact would be unexplainable and fragile; the
measured effect comes from the conflict-topology signal, which can be
expressed deliberately and tested directly.

## Phase 3: decide and implement (only after Phase 1 + 2 have answers)

- If Phase 2 shows the ordering-quirked pass is genuinely better at wp1's
  density regime: implement it properly in
  `openptv2/algorithms/correspondences.py::take_best_candidates` as a
  deliberate, documented, *tested* conservative-uniqueness mode (not a
  literal quirk-port — a clean reimplementation of whatever principle makes
  it work, e.g. "claim overlap should penalize both competing candidates,
  not just the loser"), gated so existing datasets' behavior doesn't
  silently change.
- If Phase 2 shows the clean pass is as-good-or-better: no correspondence
  change needed; the 16% quad gap vs. GT is then explained as "openptv2
  recovers more correct quads than 3dptv.exe's own buggy dedup discarded" —
  update `2026-08-27-track3d-beat-gt-plan.md`'s framing accordingly (this
  would be a concrete instance of "better than GT," not just parity).
- Detection-stage fidelity (Phase 1) fix, if any is found, is implemented
  and validated the same way every other change in this session's plans is:
  rerun `verify_3dptv_detections.py`-style comparison, rerun the wp1
  point-cloud diff, and check the synthetic detection suite doesn't regress.
- Every change from this plan must be checked against **both** wp1 (real
  data proxy) and the relevant synthetic suite (ground truth proxy) before
  being called done — same discipline as
  `2026-08-27-track3d-beat-gt-plan.md`'s "goal, precisely stated" section.

## Out of scope here

- Tracking-stage changes (Level 2 losers-retry, cold-start gate tuning,
  dacc-averaging dilution) — tracked in their own plans; this plan is
  detection + correspondence only, since that's what feeds tracking and
  should be verified first, per the /goal's own ordering (detection →
  stereo-matching → tracking).

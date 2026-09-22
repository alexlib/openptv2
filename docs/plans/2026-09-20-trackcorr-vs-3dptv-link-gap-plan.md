# Plan: locating the trackcorr vs 3dptv link gap (69.5% vs 59.0%)

**Date**: 2026-09-20
**Status**: CLOSED — cause located (config mismatch, T0.1). See "Resolution".
**Dataset**: `C:\Users\alex\Downloads\HiDImaging\wp1_10_images`
(4 cameras, 512x512, ~1300 particles/frame, 10 frames, `dvxmax=1.9`,
`dacc=1.9`, `dangle=270` gon, `add=1`)
**Reference**: `res_ground_truth_backup/` — legacy `3dptv.exe` output, per
`scripts/make_ground_truth_zarr.py` ("regression oracle for the old binary")

## Resolution (2026-09-20, evening session)

**Root cause: openptv2 ran with `dacc=0.8` while 3dptv ran with `dacc=1.9`.**
`scratch/_wp1_ab/parameters_Run1.yaml` (`track.dacc: 0.8`) overrides the
work folder's `parameters/track.par` (line 8: `1.9`, byte-identical to the
dataset's). `py_start_proc_c` returns the YAML value — every openptv2 number
in this plan was measured at `dacc=0.8`, i.e. the two engines were never
compared under the same configuration.

Evidence (all on clean frames 100003–100010, pristine inputs restored before
every run — see methodology note below):

| config | test links | exact vs ref (7387) | recall |
|---|---|---|---|
| baseline `dacc=0.8` (track.c parity flags) | 6329 | 6181 | 83.7% |
| baseline `dacc=0.8` (openptv2 defaults) | 6335 | 6184 | 83.7% |
| **`dacc=1.9`** (3dptv value) | 7384 | 7238 | **98.0%** |
| `dacc=1.9` + backward pass | 7418 | 7251 | 98.2% |
| `dvxmax` 1.9 → 2.5 (dacc=0.8) | 6328 | 6099 | 82.6% (hurts slightly) |

Mechanism check: 17.9% of reference links have 3-point `acc >= 0.8` (only
0.25% have `acc >= 1.9`), vs a 16.3% exact-link deficit — the missing mass
sits exactly in the `[0.8, 1.9)` acceleration band the tighter gate rejects.
This also explains the displacement ramp (faster links scatter more in
acceleration) that was mistaken for a search/candidate defect.

Residual 2% (149 → 136 links after backward): spread across all frames
(12–30/frame, no blowup), disproportionately fast (mean disp 0.97 mm vs 0.39
overall) — cascade + conflict noise, no second systematic cause. The backward
pass adds only +13 links, so the reference is essentially forward-made.

**Recommendation**: decide whether `dacc: 0.8` in the work YAML is an
intentional retune or an accident, and either align it with `track.par`
(`1.9`) or document the divergence where the comparison is reported. The
`~2% dacc/dangle rejects` and `dacc/sqrt(18)` noise-ceiling entries in the
"Suspects eliminated" table below were measured/computed under mismatched
assumptions (rejects at dacc=0.8; ceiling at dacc=1.9) — both are superseded
by this section.

Methodology notes (learned the hard way):
- A tracking run REWRITES `res/rt_is.*` (added particles appended) and
  updates target-file `tnr` columns in place. Re-running in the same folder
  without restoring inputs silently changes the inputs. `scripts/
  sweep_track_params.py` and `scripts/run_dacc19.py` restore pristine
  `rt_is.*` (from `res_ground_truth_backup`) + targets (from dataset
  `img_3dptv`) before every run and hash them.
- Never run two tracking processes in one work folder concurrently (steps
  re-read inputs per step; races produce garbage link counts and mystery
  deaths). One early `dvxmax=2.5` reading (673 links/step-1 + vanished
  process) was such a race artifact; clean rerun gives 82.6%.

## Addendum: dataset-folder verification (same session)

Re-ran everything directly in `C:\Users\alex\Downloads\HiDImaging\
wp1_10_images` (whose own `parameters_Run1.yaml` also carries `dacc: 0.8`)
with an in-memory `dacc=1.9` override, forward + backward
(`scripts/verify_same_trajectories.py`):

| level | result |
|---|---|
| links, clean frames 100003–100010 | 7251/7387 = **98.16%** exact |
| links, all frames 100002–100010 | 7873/8018 = 98.19% |
| full trajectory chains end-to-end | 4463/4793 = **93.1%**; length distrib. identical (mean 2.67 vs 2.69, max 10) |
| displacement overlay | ramp gone: %lost per bin ~0±5% at all speeds; mean step 0.395 vs 0.403 (was 0.394 vs 0.311) |

Residual-miss fingerprint at matched parameters (in-kernel probe, step
100005, 26 misses): 73% partner PRESENT but lost in conflict resolution,
19% partner gated downstream, 8% inlist empty — versus 96% inlist-EMPTY at
`dacc=0.8`. No systematic blindness remains; the rest is greedy-resolution
divergence + cascade.

Tooling fixes while verifying: `scripts/diag_inlist_probe.py` gained
`--dacc`/`--dv` overrides and now walks the buffer from `first` to `--step`
(it previously processed frame `first` while labeling/comparing `--step`,
and mislabeled output files for any `--step != first`).

Dataset left pristine: `res/` back to `{run.zarr}`, target files
byte-identical to pre-run snapshot (rt↔tnr consistent on clean frames),
`parameters_Run1.yaml` untouched (still ships `dacc: 0.8`).

## Addendum 2: single-trajectory comparison (`scripts/
compare_single_trajectories.py`, plots in `scratch/single_traj/`)

At matched parameters the residual is greedy contest resolution in dense
crossing regions, illustrated frame-by-frame:

- **case2 (crossing, len-10):** at 100006→100007 ref jumps 0.93 mm (acc
  1.02) to 1268 and continues violently (acc up to 1.88); test links 842→822
  (acc 0.27, smoother) and stops. Cross-check: ref runs 822→1237 as a len-2
  stub while test runs 1268→1237 — a genuine 2-particle tangle 1268/822 →
  1237/1238 resolved oppositely. Undecidable without ground truth.
- **case5 (ghost stitch, len-10):** ref leaves 100004:1207 a singleton; test
  threads it into a full-span chain assembled from three ref chains (len-3
  head + singleton + interiors of a len-8 and a len-6), including a 1.50 mm
  step and an acc-1.899 link at the dacc edge. Sensitive vs conservative —
  likewise undecidable from outputs alone.
- **case1 (tail miss):** test follows 8/8 frames then stops where ref
  continues through acc 1.295 — conservative tail.
- **case3/4:** 2.67 mm cold-start jump at the corrupt frame 100001 —
  artifact of the known data defect, not algorithmic.

Net: no global bias (smoother/longer/ghost-ridden in neither direction).
Every remaining difference is a local contest outcome. Pushing past 98%
requires ground truth adjudication — the synthetic replica testbed (T3.1).

## Addendum 3: crossing laboratory (`scripts/synth_crossing.py`)

Minimal scenes with known truth on the real wp1 optics (4 cams, σ=0.07 mm
noise): two particles cross between frames 3→4 of 7. Result table
(recall / crossing-decision-correct over reps):

| scenario | parity (lr0) | default (lr1) |
|---|---|---|
| S1 head-on, S2/S3 near-miss, S5 fast, S7 asymmetric | 100% 1/1 | 100% 1/1 |
| S4 head-on+noise (8 reps), S6 three-way (8) | 100% 8/8 | 100% 8/8 |
| S8 kick-at-cross (0.15→1.0, no noise) | 91.7% 1/1 | **100%** 1/1 |
| S9 brake-at-cross | 100% | 100% |
| S10 convoy pair (8) | 100% 8/8 | 100% 8/8 |
| S11 kick+noise (8) | 75% **2/8** | 75% **2/8** |
| S12 convoy+noise (8) | 96.9% 7/8 | **100%** 8/8 |
| S11 with dacc 2.5 / 3.5 | 2/8, 2/8 | 2/8, 2/8 (gate not binding) |

Mechanism, read from the kernel's own arrays (S8): the kick makes truth
expensive (rr 0.039) while the smooth theft is cheap (rr 0.027); both claim
the same row, truth's owner loses the contest (0.027 vs 0.009), parity drops
it (`next=-2`), `loser_retry` recovers via second choice. Failure modes in
S11 are swaps and stalls varying per noise seed.

Lessons: (1) for steady motion through crossings the smooth choice IS
correct and openptv2 never misses it — so real case-2's violent-vs-smooth
split likely means the particle truly maneuvered and 3dptv was right;
(2) keep `loser_retry=1` (default): strictly dominates parity on maneuvers
(S8, S12), free elsewhere; (3) maneuver+noise is structural — no parameter
rescues it; fixes would be longer temporal context, noise-aware costs, or
global (non-greedy) assignment.

## Addendum 4: 4BE on the hard crossings (`scripts/bench_4be_crossing.py`)

4BE (four-frame best estimate: candidate scored by predicting a real
particle two frames ahead) does NOT solve the maneuver crossing — it is
worse: S8 cross-ok 0/3 (vs parity 3/3, default 3/3), S11 0/8 (vs 2/8),
S12 7/8, S1/S4 100%. Reason: longer context cannot fix a violated motion
model — constant-velocity extrapolation underprices the steady lie versus
maneuvering truth at every horizon, so more context just confirms the lie
more confidently. Two-directional (fwd+bwd trackcorr) was already measured
on real data: +13 links (98.0→98.2%), same mirrored greedy logic, no help
on contests. What would help: acceleration-aware prediction,
multi-hypothesis tracking (keep both options alive 2 frames — the kick IS
resolved by f5/f6), or global assignment; none is in the codebase.

## Addendum 5: two-phase tracker on crossings (`scripts/bench_two_phase.py`)

`plugins/two_phase_tracking.py`: phase 1 = 3D KD-tree candidates within
v_max; phase 2 = Hungarian assignment on 2D pixel-distance costs. Memoryless
(no velocity, no history). Result: systematic **bounce bias** — at an
X-crossing the swap always has lower total cost than the cross, so raw
two-phase scores 0 cross-ok on EVERYTHING incl. clean S1 (83.3% recall =
one swapped step). It cannot solve hard crossings as-is.
Fix tested in-bench: constant-velocity prediction injected into phase 1
(predict points, re-project for leaf costs) + Hungarian (`two-phase+vel`):
S1/S8/S4 → 100% cross-ok, **S11 → 5/8 vs trackcorr's 2/8**. Global
assignment + prediction beats greedy + prediction on the killer. Not yet
productized (needs velocity state, cold start, gap handling in the plugin
+ wp1 validation) — candidate next step.

## Addendum 6: productized two-phase+vel (plugin) + wp1 verdict

Implemented in `plugins/two_phase_tracking.py`: velocity state per track,
zero-velocity cold start, gap bridging to `max_gap`, `project_fn` callback
built in `do_tracking` from `exp.cals`/`exp.cpar` (falls back to 3D costs),
new knobs `use_velocity` (default True), `cost_mode`, `v_max` (separate from
`dvxmax`), `max_gap`. `tests/unit/test_two_phase_tracking.py`: 5 passed.
`scripts/bench_two_phase.py` productized engine == prototype on every scene
+ S13 occlusion gap bridged 3/3.

wp1 (`scripts/verify_two_phase_wp1.py`, v_max × max_gap grid):

| config | recall | links (ref 7418) | test-only step p50/p90 |
|---|---|---|---|
| v2.0 gap1 | **96.26%** | 8443 | 0.81 / 1.95 |
| v2.0 gap2 | 95.93% | 8988 (+573 gap) | 0.82 / 1.95 |
| v2.5 gap1 | 94.98% | 8734 | 1.49 / 2.51 |
| v3.3 gap1 | 86.92% | 9246 | 2.52 / 3.96 |

vs trackcorr@dacc19: 98.16%, 7418 links. Verdict: viable ALTERNATIVE
engine (maneuver-sensitive: S11 5/8 vs 2/8), not a replacement — it lacks
trackcorr's selectivity (no acc gate / two-hop confirmation), so it
over-links plausible-but-unconfirmed tracks, and gap bridging slightly
hurts on wp1 (95.93 < 96.26: stale-velocity gap claims steal consecutive
links). Natural next step per the original vision (3D forward, then 2D):
cascade — trackcorr first, two-phase+vel only on unlinked heads/tails.

## The gap

Both engines run on identical `rt_is.*`, so a link-by-link comparison is exact.
Measured on clean frames 100003–100010:

```
3dptv 69.5%   openptv2 59.0%   reproduced 6113/7387 = 82.8%
```

The loss is monotonic in link displacement, with **no cliff**:

| link displacement | 3dptv links | openptv2 | % openptv2 misses |
|---|---|---|---|
| 0.00–0.22 mm | 3326 | 3263 | 2% |
| 0.44–0.55 | 608 | 495 | 19% |
| 0.77–0.88 | 231 | 116 | 50% |
| 1.21–1.32 | 72 | 20 | 72% |
| 1.87–1.98 | 34 | 7 | 79% |

Mean step: 3dptv 0.394 mm, openptv2 0.311 mm — openptv2's trajectories are
measurably slower, which is the originally reported symptom.

Split of the 1256 missing links: 96% the **predecessor got no link at all**,
4% it linked elsewhere. Only 0.7% of reference links are mislinked. So this is
genuine dropping, not preference reordering.

## Suspects eliminated (all by direct measurement)

| Suspect | Verdict | Evidence |
|---|---|---|
| Search-volume clamp | ✗ | no cliff; both runs have links past `dvxmax` |
| Candidate shortlist depth (top-4) | ✗ | true partner at **rank 1** median, 0.2% beyond top-4 |
| Multi-camera `freq<2` cull | ✗ | 1.2% |
| Two-hop lookahead | ✗ | 0.2% `HOP2_NOT_FOUND`; `X5` predictor byte-identical to `track.c:223` |
| `dacc`/`dangle` gate | ✗ | ~2% rejects; `dangle=270` gon = 243° is inert |
| Position noise | ✗ | sigma = 0.070 mm vs ceiling `dacc/sqrt(18)` = 0.448 mm |
| Phase-3 loser retry | ✗ | +1.6 links/frame of ~760 |
| Neighbour-velocity cold start | ✗ | 6705 -> 6704 links |
| Compiled vs pure-Python | ✗ | **byte-identical** (534 links, 523 reproduced, both) |
| Camera permutation | ✗ | cross-camera residual matrix is diagonal |
| Quadrant vs full-frame convention | ✗ | baseline 90 px vs 330–400 px for every offset variant |
| mmlut / multimedia model | ✗ | clearing the LUT gives identical residuals; projection is **0.53 px** median on clean frames |
| Frame-100001 corruption | ✗ (as gap cause) | gap persists at 69.5/59.0 on clean frames |
| Prediction cascade | ✗ | autopsy with test-run link history unchanged (69.1% GATE_PASSED) |

Faithful to `track.c`, verified line-by-line: `rr` cost formula, the
`(acc<dacc && angle<dangle) || acc<dacc/10` gate, `angle_acc` (gon, 200/pi),
the `X5 = 0.5*(5*X3-4*X1+X0)` two-hop predictor, `candsearch_in_pix` 4-nearest
cascade, `sortwhatfound` freq merge, `POSI=80`.

Two divergences from `track.c` were found, both made switchable and both
measured **inert** on this data (defaults unchanged):

```python
Tracker(..., loser_retry=0, cold_start_neighbour=0)   # track.c parity
```

## Separate confirmed data defect

**Frame 100001 is corrupt.** `rt_is.100001`'s `p1..p4` columns disagree with
`Cam*.100001_targets`: 1740 corresponded targets for 1107 particles, and 97 px
median reprojection vs 0.53 px on every other frame. The target files look
overwritten by a different detection/correspondence run. Worth fixing
independently of this investigation.

## The open contradiction

The kernel's own arrays report `inlist == 0` for 96% of missed links. An
external replay of the candidate stage — corrected for the search-centre bug
(the tracker centres on the **measured target pixel**, `track.c:136`, not on
`point_to_pixel(X2)`) and re-run with the tracker's own link history — still
says 69% of those partners are admissible at rank 1.

Both cannot be true. The kernel probe is the trustworthy one because it reads
real state rather than reimplementing. **The methodological error was replaying
the candidate stage outside the tracker instead of instrumenting inside it.**

---

# Test plan

**Governing rule:** read state the tracker actually produced, or control an
input it actually consumed. No reimplementing tracker internals externally.
Every wrong root cause in the 2026-09-20 session came from an external replay
that diverged from the real code in an unnoticed detail.

Each test states its decision rule up front.

## Phase 0 — Validate the premise (cheapest; can end the investigation)

### T0.1 — Are 3dptv's links legal under `track.par`?

Never checked. The whole comparison assumes both ran the same configuration.

- **Method**: for all reference links compute per-axis `|dx|,|dy|,|dz|`, the
  3-point `acc`, and `angle` in gon. Histogram each against `dvxmax=1.9`,
  `dacc=1.9`, `dangle=270`.
- **Decision**: a material fraction violating these bounds means the reference
  ran with **different parameters** — the gap is config, not algorithm. Stop
  and reconcile.
- **Prior**: non-trivial. Dropped links reach 3.40 mm and `1.9*sqrt(3) = 3.29`,
  so some reference links may be out of bounds per-axis.

### T0.2 — Is the reference reproducible?

- **Method**: re-run `3dptv.exe` on this folder; diff `ptv_is.*` against
  `res_ground_truth_backup`.
- **Decision**: if it does not reproduce, the reference is stale — regenerate
  before comparing. Frame 100001's corruption already makes provenance suspect.

### T0.3 — Rebuild the corrupt frame

- **Method**: re-run detection + correspondence for frame 100001; confirm
  reprojection drops to ~0.5 px.
- **Decision**: do this regardless — it is a real defect. Then re-baseline on a
  fully clean 10-frame set.

## Phase 1 — Make the loss observable from inside

### T1.1 — In-kernel gate census (highest value)

The one measurement that cannot disagree with itself.

- **Method**: add an `int[8]` counter array to `_trackcorr_particle_fast`,
  incremented at each discard point: `sorted_candidates` returned 0 ·
  displacement gate · two-hop search empty · two-hop candidate gated ·
  acc/angle gate · `inlist` full (POSI) · lost conflict · evicted. Return it
  from `trackcorr_loop_fast`. Record per particle the step size of the
  reference link it should have made.
- **Output**: death-stage histogram, split by link displacement.
- **Decision**: the dominant stage is the target. This supersedes every
  external replay, including all autopsy numbers in
  `scripts/diag_candidate_autopsy.py`.

### T1.2 — Full trace on ten specific drops

- **Method**: pick 10 dropped links spanning the speed range; dump every
  intermediate — `X0,X1,X2`, per-camera window bounds, every candidate with
  distance and `freq`, `X3,X5`, every frame-3 candidate,
  `acc/angle/dl/quali/rr`, final `inlist`/`decis`/`linkdecis`.
- **Decision**: read by hand. Histograms hide mechanism; traces show it.

## Phase 2 — Make it controllable (run T2.1 in parallel with Phase 1)

### T2.1 — Parameter sensitivity sweep (model-free localiser)

Requires no hypothesis about mechanism.

- **Method**: sweep one parameter at a time — `dvxmax` (1.9 -> 4.0), `dacc`
  (1.9 -> 6.0), `dangle` (270 -> 400), `add` (0/1) — plotting openptv2's link
  rate against 3dptv's fixed 69.5%.
- **Decision**: whichever parameter closes the gap names the binding constraint
  empirically. If none closes it, the loss is not a gate and T1.1 should show a
  structural cause.

### T2.2 — One-step isolation with seeded history

Removes cascade entirely.

- **Method**: seed `path_prev`/`path_next` from 3dptv's `ptv_is` for frames
  N-1 and N, run exactly one step, compare that step's links.
- **Decision**: matching means the gap is cascade (a small per-step deficit
  compounding). Still missing means a genuine per-step difference.

## Phase 3 — Make iteration fast

### T3.1 — Synthetic replica at real statistics

The earlier synthetic (`scripts/ab_loser_retry.py synthetic`) was too sparse
and too clean to reproduce this regime.

- **Method**: ground-truth trajectories matching wp1 — ~1300 particles/frame,
  0.36 mm mean step, 0.07 mm position noise, this calibration, 4 cameras, same
  density distribution.
- **Decision**: ~100% recall means the gap is specific to real data and
  Phase 0/1 explains it. ~59% means you have a fast, fully-observable testbed
  with a known right answer — iterate there, not on real data.

## Phase 4 — The comparison not yet done

### T4.1 — Instrumented `track.c` harness

Everything so far compares openptv2's internals against 3dptv's **final
output**. Nobody has looked at 3dptv's intermediates.

- **Method**: extract `trackcorr_c` from `C:\Users\alex\Downloads\3dptv\src_c\`
  into a standalone C harness reading the same `rt_is`/targets/`track.par`,
  with the same counters as T1.1. Compare stage-by-stage on one frame.
- **Cost**: highest — a real build project (Tcl dependencies need stripping).
- **Decision**: reserve for the case where Phases 0–3 leave it ambiguous.

## Ordering and exit criteria

| Phase | Cost | Could end it outright |
|---|---|---|
| 0 | hours | **Yes** — parameter mismatch or stale reference dissolves the premise |
| 1 | ~1 day | **Yes** — the census names the stage |
| 2 | hours | Localises empirically without any mechanism hypothesis |
| 3 | ~1 day | Converts it into a controlled, ground-truth problem |
| 4 | days | Definitive, only if needed |

**Run Phase 0 first.** Three dead ends in the originating session would have
been avoided by checking whether the two runs were configured the same way.

**Stop condition**: a stage from T1.1 accounting for the majority of missed
links, confirmed by T1.2 traces, whose removal or widening moves the link rate
toward 69.5% in T2.1.

## Tooling already built

| Script | Measures |
|---|---|
| `scripts/diag_conflict_parity.py` | `track.c` vs openptv2 link resolution, self-checking |
| `scripts/diag_speed_ceiling.py` | measured vs configured speed ceiling |
| `scripts/ab_loser_retry.py` | `synthetic` / `folder` / `gate` modes, paired statistics |
| `scripts/diag_position_noise.py` | position noise vs the `dacc/sqrt(18)` ceiling |
| `scripts/diag_link_displacement.py` | the displacement ramp table + plot |
| `scripts/diag_candidate_autopsy.py` | per-link two-stage autopsy (**external replay — see the open contradiction above; trust T1.1 over this**) |
| `scripts/diag_inlist_probe.py` | reads the kernel's real `inlist`/`linkdecis` arrays |
| `scripts/_run_one_step.py` | one step + link comparison; used for the compiled/interpreted check |

## Addendum 7: cascade (trackcorr base + additive two-phase merge)

`scripts/cascade_track.py` (`merge_links`: accept a two-phase link iff
BOTH ends are free; never steal) + `scripts/run_cascade.py` (synth/wp1
driver). Load-bearing bug found by the +0 result: trackcorr marks dropped
tails `-2`, not `-1` — the first guard (`== -1`) accepted nothing.

- synth: S13 occlusion gap fixed 94.1% → 100%; S11 unchanged (swaps have
  taken ends — additive merge fixes incompleteness, never incorrectness).
- wp1: base 7251/7387 (98.16%) → merged 7295/7387 (**98.75%**), +444 links.
  Added split: consecutive exact=44/extra=328 (11.8% added-precision);
  gap exact=0/extra=0 (no occlusions in this data). Added links are
  high-acc by construction (p50 1.46 vs ref 0.31); acc-filtering destroys
  the gain (they're maneuvers or ghosts — undecidable vs 3dptv).
- Verdict: works as designed, but 44 recovered vs 328 unconfirmed fails
  the no-ghost utility — ship as opt-in second pass, default OFF. The
  adjudication the 328 needs is a dense-data synthetic testbed with truth.

## Addendum 8: Willneff program — pipeline completion + core limits

Willneff & Grün 2002 (the algorithm track.c implements): new-method
efficiency eff_3D = links/particles 76.2/89.4/91.1% (datasets A/B/C),
gains +25/+28/+13% over object-space-only. Ingredients: image+object
space, prediction, gap bridging, bidirectional. Thesis headline: yield of
LONG trajectories for Lagrangian analysis.

wp1 chain census — ref vs ours fwd+bwd are IDENTICAL (eff 69.9/70.2%,
fullspan 334/335, len≥8 533/539): the port is faithful; the gap to
Willneff's numbers is dataset difficulty + missing pipeline stages.

Postprocess (`scripts/run_postprocess_wp1.py`, already wired into plugin
flows): cold_start +8, gap bridges +138, reciprocity severs 0 →
eff 70.2→71.5%, fullspan 335→373 (+11%), len≥8 539→608, exact unchanged
98.16% (ref lacks relinked links). Gap bridging validated with truth:
1- and 2-frame occlusions → 100% (`max_gap=2`). Note: 2-frame gaps only
bridge with sequence room — the last 2 steps kill lookahead (track.c
behavior), so tail cold-starts need the backward pass.

Global (Hungarian) Phase-2 (`scripts/proto_global_resolve.py`, no kernel
change, 20 reps): S11 20/20 (vs 7-10/20) BUT S4 13/20 and S12 7/20
cross-ok (vs 19-20/20). Drill (`scripts/drill_s4_global.py`): identical
histories, pure resolution-level loss — min-total overfits noise.
REJECTED as replacement; greedy+retry stays the best average rule.

Bottom line for trackcorr: pipeline-complete (fwd+bwd+postprocess) +
loser_retry=1 is the evidence-backed operating point (98.16% links,
71.5% eff, fullspan 373 on wp1). The S11 maneuver+noise core has no
single-step fix — needs MHT / maneuver detection (future work).

## Addendum 9: out-of-box program — profiler, appearance, ensemble

Miss profiler (`scripts/miss_profiler.py`, 145 misses vs 7873 controls):
misses are faster (step 1.02 vs 0.38), 5x more crowded (crowd≥2: 15.2 vs
3.0%), shorter history (5.3 vs 7.5), higher acc (1.14 vs 0.44); no frame
clustering except the tail frame (30 = last-frame no-lookahead boundary,
vanishes on long sequences). Cold start is NOT the problem (0% starts).
Appearance probe (sumg continuity at 26 contested divergences): truth
more similar in 65% — real but weak; tiebreak-grade, not gate-grade.

Ensemble (`scripts/collect_ensemble_wp1.py`: parity / default-lr1 / 4BE /
two-phase+vel, identical inputs): singles 98.16 / **98.33** / 88.70 /
96.26% (default-lr1 beats parity by 13 links — last missing cell).
Unanimous (4/4) links: **99.74% precision** (n=6440) — the ghost-free set.
Majority (≥3): 99.40% precision but only 96.01% recall (2-2 ties abstain).
Union (any engine right): **99.54%** — only 34/7387 links are hard for
ALL engines. Contested (engines disagree): 947 links (12.8%), majority
right 90.1% (wins: default 841, parity 828, 2p 688, 4be 129).

Shippable: per-link agreement flag (4/4 → high-confidence; 2-2 →
review) + default-lr1 as the recall engine. Kernel rebuild verified
feasible (setup.py, ~87 s) if an in-kernel change ever earns it.

## Addendum 11: brightness in cost + pulsatile freedom (kernel work)

track.c verified: tracking cost is purely kinematic
(`rr=(dl/lmax+acc/dacc+angle/dangle)/quali`) — brightness was never used.
(a) Implemented: `targ_sumg` SoA plumbing (Frame→kernel), per-particle
mean-grey signatures, `rr += w_app·|Δ|/(sum)` at all 4 cost sites,
`app_weight` flag (Tracker/TrackingRun/kernel, default 0.0), rebuilt.
S15 (4:1 brightness kick+noise): 83%→100% at w≥0.1; 18 other
scenario-variants unchanged (incl. uniform-brightness S11: proves
inertness when uninformative). wp1: -4 links at w=0.1 → default 0.0,
opt-in ≈0.1 for distinct populations. Shape/size (n/nx/ny) same pattern,
not yet plumbed.
(b) Pulsatile: mild (acc<dacc) needs nothing (S14/S14b 100% fixed).
Beyond-box systole needs wider dv (channel isolation: dv-only wins,
dacc-only inert). BUT any widening (global or per-particle, hi 1.5-2.5)
swap-bombs dense wp1 (87-88%: 832 steals, ZERO drops — smooth lies outbid
fast truth once admitted; disciplined costs + two-tier tight confirmation
don't save it). Kernel-neutrality proven: all features off reproduces
7251/98.16% + 4463/93.11% bit-exactly.
Prescription: fixed tight boxes for dense steady data (optimal);
per-particle freedom + two-tier shell opt-in for sparse/pulsatile
(S14c 61→100%); dense+systole needs a dense-systole truth scene + (a) as
the complementary lever. Scripts: synth_crossing (S14/S15/sumg),
sweep_app_weight, phase_scheduler, verify_ppar_wp1, diagnose_ppar_wp1.

## Addendum 12: ppar config-artefact correction + density-gated freedom
(shipped)

The 87-88% "ppar regression" was a config artefact: the scheduler loaded
`parameters_Run1.yaml` (dacc=0.8) while the fixed control ran dacc=1.9 --
832 fast-link drops from the tight acc gate, misattributed to scaling
(first-step gap already proved it: identical gs=1 code can't diverge).
Scheduler now takes a `yaml` param; all wp1 runs use dacc19.
Real scaling cost at matched config was only ~45 links, then engineered
to zero: (1) costs stay unscaled (discipline when competing); (2) scaled
shell keeps tight confirmation; (3) density-gated crowd veto in the
scheduler -- freedom granted only where no rival sits near the predicted
position in dense data (num_parts>=100), veto skipped in sparse data
where a lone bystander loses on cost (S14c). A kernel starvation-gate
experiment was tried and REMOVED (redundant once the veto works; it
blocked genuine shell rescues). Final, all on rebuilt kernel:
wp1 ppar-acc-hi2.5 fwd = 7238/97.98% EXACT fixed parity; S14c-vel = 100%
(61% fixed); S15-app0.1 = 100%; S11 fixed=app=ppar = 79.6%;
S1-ppar = 100%. wp1 fwd+bwd: ppar+app0.1 == fixed+app0.1 link-for-link
(7234/4439) -- backward absorbs the forward delta; app0.1 alone costs
-17/-24 vs app0 on natural brightness (7251/4463), so default app=0,
opt-in 0.1. run_adaptive_pp gained `yaml`, `mode`, `backward` params.

## Addendum 10: manual vote on the 34 union-miss links (moved intact --
was accidentally folded into Addendum 11 during editing)

Prototype `scripts/vote_links.py` (agree.<frame> sidecars: flag 4/3/2/1
+ per-engine prev) + `scripts/plot_hard_links.py` (6 mosaics in
`scratch/hard_links/`, X-Y/X-Z neighborhoods with per-engine arrows) +
`scripts/judge_hard_links.py` (same-target options table) +
`scripts/trace_divergence.py` (upstream walk) + validated
`scripts/probe_search.py` (48/48 control agreement) and per-axis legality
(31/34 legal; 3 ref dv-violations: 100005/488, 100010/860, 100010/1026).

Verdict by class (34):
- TAIL (8, all frame 100010): no-lookahead strict gate; vanishes on long
  sequences. 860/1026 are ref dv-violations — engines defensible.
- LOCAL contested, same history (14): genuine ties; options table decides
  each. Exhibits: 100006/875 (ref smoother on step AND acc AND appearance
  4-vs-139 — test steal; best case FOR appearance tiebreak), 100006/334
  (ref acc 2.13 OVER dacc — engines right; possible 3dptv overreach),
  100007/1268 (eng smoother 0.62 vs 1.02 — toss-up), 100004/1247
  REF-FAST (ref steady 2.39mm/acc 0.41, all abstain — smoothness bias
  drops fast tracks;needs velocity-alignment reward, not just acc penalty).
- LOCAL all-abstain, small-step legal (535: kernel inlist=3 incl. truth —
  S8-style maneuver contest loss, rr 0.039 vs 0.027; 421: parity blind via
  broken forward history (21.6mm prediction), default contest loss — SAME
  vote, THREE mechanisms across engines).
- UPSTREAM (10): 2 one-step swaps adjudicated (100003/1167 lean-test:
  smooth+history vs cold 2.25 jump; 100006/875 for REF, agrees with
  appearance), 1 head-formation (421), 7 birth-differences in the corrupt
  frame-100001 zone (excluded from judging — data defect).

Critical missing decisions, ranked: (1) cost function has no
velocity-alignment reward and no appearance tiebreak (875 + 1247 would
flip); (2) single-step horizon can't see maneuvers resolving (S11);
(3) upstream swaps cascade into downstream blindness backward can't
repair (needs next>=0 seeds). (1) is implementable in-kernel; (2)-(3)
need MHT.

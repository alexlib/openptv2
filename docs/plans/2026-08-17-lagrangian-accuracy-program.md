# Lagrangian accuracy and trajectory length: reduce ten trackers to the few
# that measure physics correctly (2026-08-17)

Goal, in the user's words: keep the minimum number of the best trackers.
Decide what matters for **fluid mechanics and turbulence, not numerical
exercise**; evaluate it properly, slowly and carefully; understand what in
each algorithm produces the better outcome; then reinvent a simple thing that
takes the winning parts.

The value function has **two** objectives, ranked but not exclusive:

> We value most the accuracy — correct velocity, correct acceleration in the
> Lagrangian sense, then time correlations, distance correlations and other
> things that require long trajectories. Long we can achieve later by
> smoothing and stitching; the correct ones we could not get by
> post-processing wrong trajectories.
>
> The value function is not *only* track length — track length is another
> important factor, but for a specific type of question. So we should be able
> to choose parameters that improve track length **without sacrificing
> quality**.

Read precisely, that is a **constrained multi-objective** problem, not a
priority ordering that discards length:

- **Accuracy is the constraint.** It is never traded away, because a wrong
  link cannot be repaired downstream.
- **Length is a genuine second objective**, not a nice-to-have. Lagrangian
  time correlations, the integral timescale, pair dispersion and long-lag
  structure functions are simply not computable from short fragments — those
  questions need length, and they are real questions.
- **The target is the Pareto frontier of (accuracy, length)**, and
  specifically the configurations that buy length *for free*. Where a
  parameter improves length at no accuracy cost, take it; only refuse the
  trade when length costs accuracy.

Length being recoverable later (§3.1/§3.2 of the 2026-08-16 plan made gap
bridging work) is a reason not to *panic* about length, not a reason to stop
measuring it — and stitching is itself only safe on correct fragments, so the
two objectives are coupled rather than independent.

**There is already direct evidence such free length exists.** In the §0 table
below, `dacc=3.6` + bridging beats `dacc=6` (the shipped default) on
acceleration accuracy *and* precision *and* yield simultaneously — a strict
improvement on both axes, not a trade. That single observation is the
strongest argument that the current defaults sit off the frontier, and
finding the frontier deliberately is Phase 1's second deliverable.

---

## 0. The finding that reframes the work

**Nothing this project currently measures ranks trackers the way the value
function above does.** Measured 2026-08-17 (details in §1): the tracker with
the *best link precision* is not the tracker with the *best acceleration
statistics*, and the gap is not small.

`test_data/synthetic_turbulent`, 220 particles/frame, component-wise
acceleration by second difference, against the ground truth's own:

| config | precision | yield | a_rms error | K_a (truth 3.14) | \|a\| > 5σ |
|---|---|---|---|---|---|
| ground truth | — | — | — | 3.14 | 0.000% |
| **3MA `dacc=3.6` + bridging** | 0.9709 | 0.9187 | **+2.1%** | **3.20** | **0.000%** |
| 4BE paper | **0.9851** | 0.8128 | +5.0% | 4.33 | 0.126% |
| 3MA `dacc=6` | 0.9667 | 0.8943 | +11.9% | 5.15 | 0.271% |
| 3MA `dacc=6` + bridging | 0.9596 | 0.9362 | +11.9% | 5.15 | 0.271% |
| 4BE greedy | 0.9700 | 0.8879 | +14.5% | 5.78 | 0.418% |
| 4BE paper + bridging | 0.9335 | 0.8371 | +50.3% | **343.5** | 0.221% |

Three things fall out immediately.

**1. Precision is not a proxy for kinematic accuracy.** 4BE-paper has the
best precision of any configuration and the second-worst acceleration
kurtosis of the sane ones. `dacc=3.6` has *lower* precision and reproduces
the acceleration PDF almost exactly. Ranking on precision — which is what
every benchmark in this repo does — would pick the wrong tracker.

**2. What matters is the *magnitude* of a wrong link, not its rate.** That is
the mechanism behind (1): a tight seeded-search box (`dacc=3.6`) cannot make
a large mistake, only a small one, and a small mistake barely perturbs a
second difference. A loose box makes rarer but kinematically catastrophic
mistakes. **Design rule: bound the kinematic damage of a wrong link, not just
its probability.** No existing metric expresses this.

**3. The user's hypothesis is confirmed, and sharper than stated.** Bridging
4BE's output drives K_a from 4.33 to 343 — post-processing did not merely
fail to fix bad links, it *manufactured* them (§3.4 of the 2026-08-16 plan
explains why: the bridger re-creates exactly the conflicts 4BE declined).
Correctness genuinely cannot be added afterwards.

**Confirmed independently on real (non-Gaussian) ground truth, 2026-08-17.**
`scripts/adapt_proptv_dataset.py` converts proPTV's `500_25`/`500_30` into an
openptv2 benchmark dataset (mechanics: §1a and the script's own docstring);
`scripts/bench_proptv_kinematics.py` ran every available engine at its own
auto-recommended parameters (`benchmark_utils.per_tracker_overrides`) on
`500_30` (30 frames, 500 particles, truth `K_a` **19.8**, real intermittency,
still zero injected noise):

| tracker | `a_rms` error | `K_a` | \|a\|>5σ | precision/yield |
|---|---|---|---|---|
| **`priority_segment_3d` (3MA)** | **+0.0%** | **19.80** | 0.31% | 1.00 / 1.00 |
| `kalman_hungarian_3d`, `sg_hungarian_3d`, `nearest_hungarian_3d`, `predictive_gmm_3d`, `myptv_3d_tracking`, `proptv_tracking` | +177% | 2.62 | 1.88% | ~1.00 / ~1.00 |
| `4be` | +2962% | 677 | 0.74% | 0.996 / 0.978 |
| `trackcorr` | — | — | — | failed: needs real camera-target ids, which this adapter omits (§ below) |

Same conclusion as the table above, on an independent dataset: 3MA
reproduces the truth almost exactly while the highest-precision engine (4be)
has the worst kinematics. It also exposes a **second, distinct failure
mode** the first table didn't separate: 4BE's few catastrophic wrong links
blow up the *tails* (`K_a` 677, still centered near the true `a_rms`); the
six Hungarian/nearest-neighbour-family engines instead make many small,
chronic mismatches that inflate `a_rms` broadly *and* smooth `K_a` down to
2.62 — below Gaussian, i.e. real intermittent spikes are averaged away
entirely. Both destroy the physics, by opposite mechanisms — this sharpens
Phase 3's mechanism dissection (§2's ablation list should treat "damage
concentrated in few links" vs "damage spread across many links" as separate
axes, not one).

**Two things flagged about this table — (a) now resolved, (b) still open:**

(a) **Checked, and corrected 2026-08-17 — the six names are not six
implementations.** `src/openptv2/plugins/nearest_hungarian_3d.py` and
`predictive_gmm_3d.py` are both 5-line files that `import` their classes
straight from `myptv_3d_tracking.py` / `proptv_tracking.py` — i.e.
`nearest_hungarian_3d` **is** `myptv_3d_tracking`'s code and
`predictive_gmm_3d` **is** `proptv_tracking`'s code, by design (confirmed
directly by the user's question and a source diff). `kalman_hungarian_3d.py`
is *also* a 5-line alias, but of a third, different module
(`quality_3d_tracking.py`). Only `fast_3d_smooth_tracking.py`
(`sg_hungarian_3d`) is a genuinely separate, non-aliased implementation. So
the six names in the table above represent **four distinct implementations**
(MyPTV/nearest, proPTV-GMM/predictive-GMM, Kalman, Savitzky-Golay), not six —
and all four still produced byte-identical output. That is a *stronger*
result than originally stated, not a weaker one: four structurally different
prediction models (Kalman filter, SG smoothing, plain Hungarian, GMM)
converging to the same link decisions is real evidence this operating point
(mean NN spacing ~0.07, well above the noise floor) makes the nearest true
neighbour unambiguous almost everywhere — **this density cannot discriminate
among the Hungarian/NN-family engines**, independent of which one; it takes
the higher-density case or Phase 1b noise to separate them. It also confirms
`per_tracker_overrides` was not the cause of the apparent convergence
(`sg_hungarian_3d`'s missing scaled-parameter rule, noted below, still stands
as a separate minor gap, unrelated to this). And it settles that the Phase 2
survivor decision is sound: `myptv_3d_tracking`/`proptv_tracking` are each
other's *only* alias pair among the survivors, run their own authoritative
code, and are not secretly running a deleted engine's implementation.

(b)
`trackcorr` needs real per-camera target ids and this adapter deliberately
doesn't synthesize them (see §1a) — it and any other 2D+3D-dependent engine
are structurally excluded from proPTV-adapted data until that's addressed,
which matters for Phase 2's inventory (`cython_epipolar_tracking`'s aliases).

And at higher density the situation is worse than "ranking is wrong" —
**every tracker is currently unusable for acceleration statistics.**
`synthetic_turbulent_1k`, 970 particles/frame, true `a_rms` 0.643, true
K_a 3.00:

| config | a_rms error | K_a | \|a\| > 5σ |
|---|---|---|---|
| 3MA `dacc=3.6` + bridging | +50.3% | 1956 | 0.278% |
| 3MA `dacc=6` | +67.8% | 11.1 | 2.804% |
| 4BE paper | +60.7% | 12.7 | 2.613% |
| 4BE greedy | +142.0% | 7.4 | 8.090% |

Acceleration rms inflated by 50–142% and K_a between 7 and 1956 against a
true value of 3.0. Any turbulence conclusion drawn from these trajectories at
this seeding density would be an artefact. This is the headline number for
the whole program.

---

## 1. Why the current benchmark cannot answer the question

Before building trackers, note what the evaluation can and cannot see.
Measured on both synthetic sets:

| property | measured | consequence |
|---|---|---|
| ground-truth acceleration kurtosis `K_a` | **3.14 / 3.00** (Gaussian) | Real turbulence is 10–60. There is **no intermittency in the ground truth**, so no tracker can be scored on preserving it. Criterion B of `docs/lagrangian_turbulence_quality_guide.md` is untestable here. |
| stereo position noise, per component | **0.0003 / 0.0073 mm** | Effectively noise-free. Noise-induced acceleration / true acceleration = **0.00 / 0.03**. Real PTV is 0.01–0.5 mm (`test_cavity`'s z-noise is 0.415 mm), where acceleration is *noise-dominated*. |
| Lagrangian velocity autocorrelation | 0.62, 0.36, 0.17, 0.04 at lags 1–4 | τ_L ≈ 1.7 frames. Velocity decorrelates in ~4 frames, i.e. **dt is comparable to the flow's own timescale**. Real acceleration measurement needs dt ≪ τ_η — La Porta/Voth sample at ~70 kHz precisely for this. |
| sequence length | 30 frames | ~17 τ_L. Usable for time correlations, marginal for anything longer. |
| pair statistics | not evaluated | Criterion D (Richardson dispersion) has no harness at all. |

So on this data "accuracy" degenerates into "correct links" — and since
precision is already 0.92–0.98, every tracker looks similar and the visible
differences are all fragmentation. **That is an artefact of the test bed, not
a fact about the trackers.**

The framework for the right metrics already exists
(`docs/lagrangian_turbulence_quality_guide.md`, criteria A–E, and
`benchmarking/metrics.py::compute_physics_metrics`), but the implemented
statistics are **self-consistency only** — `acceleration_kurtosis` reports
the tracker's own PDF and never compares it against a truth. That is why a
K_a of 343 was sitting there unnoticed.

---

## 2. Program

Three phases, strictly ordered. Phase 2 is worthless before Phase 1, and
Phase 3 is guesswork before Phase 2. Expect the whole thing to be slow; that
is the point.

### Phase 1 — build an evaluation that can see accuracy

Nothing else is meaningful until this exists.

**1a. Ground truth with real Lagrangian statistics.** Ranked by preference
— **re-ranked 2026-08-17** after actually measuring what's on disk:

- **proPTV `500_25` / `500_30`** (`C:/Users/alex/Github/proPTV/data/`, found
  locally, no token/download needed). Measured directly from
  `origin_*.txt` (columns `ID,X,Y,Z,U,V,W,T,P,xc0,yc0,…`, velocity given, not
  finite-differenced): **500_30** — 30 frames, 500 particles, every track the
  full unbroken 30 frames (clean, persistent IDs), accel kurtosis **K_a =
  21.8**; **500_25** — 5 frames, same 500 particles, **K_a = 13.2**. Both are
  squarely inside the real-turbulence range (10–60) that our own
  `synthetic_turbulent` set cannot produce (K_a≈3, §1). Per-camera 2D
  centroids are already in the ground-truth file
  (`xc0,yc0,xc1,yc1,xc2,yc2,xc3,yc3`), and `500_25/input/` additionally has
  `raw_images/`, `processed_images/`, and a `calibration/` folder — a real
  4-camera rendering pipeline, not just points. **This is now the primary
  Phase 1a candidate**: it's already accessible, and the format adapter
  (2026-08-15 plan §5, previously deferred as lower-priority) is the actual
  remaining blocker, not data access. Build the adapter first.
- **DNS particle tracks (Johns Hopkins Turbulence Database)** — demoted from
  first choice. **Checked 2026-08-17: no existing access** (see open question
  1's resolution below); would need a `pyJHTDB` install and a free API token,
  unverified for local/offline use. Worth revisiting later for the gold
  standard (genuine DNS physics, choosable dt/τ_η), but proPTV's data removes
  the urgency.
- **A stochastic model with the right second-order structure** (Sawford's
  two-timescale model) as a fallback if both of the above are unworkable.
  Gives correct `a_rms`, τ_η and an intermittent-ish acceleration PDF; cheap
  and local, but synthetic-of-a-model rather than real DNS or a real
  multi-camera rendering.

Requirement either way: persist **ground-truth velocity and acceleration**
per particle per frame, not just positions. openptv2's own `origin_*.txt`
stores only `(pid, x, y, z)` (the generator in `benchmarking/scenario.py`
knows `vel` and `acc` and throws them away) — proPTV's format already carries
velocity, which is one reason it's now preferred.

**1b. Injectable position noise.** A `sigma_position` knob on the synthetic
pipeline, swept over 0 → 0.5 mm. This is the single most important axis for
acceleration accuracy and it is currently pinned at ~0. Expect tracker
ranking to change with it — a tracker that leans on a smoothness/prediction
model should degrade far more gracefully than nearest-neighbour, and today
that advantage is invisible.

**1c. Truth-referenced kinematic metrics**, into
`benchmarking/metrics.py` beside `e_track`:

- per-point velocity and acceleration error against truth (bias and rms),
  computed only over correctly-linked points, so link errors and kinematic
  errors are separated rather than conflated;
- `a_rms` ratio and `K_a` ratio vs truth (the assay used in §0);
- contamination rate: fraction of predicted accelerations beyond 5σ of the
  true distribution — a direct read on "wrong links that will poison the
  statistics";
- Lagrangian velocity autocorrelation ρ(τ) and integral time vs truth;
- pair-dispersion ⟨Δr²(t)⟩ vs truth (criterion D, currently absent).

**1d. A damage-weighted link metric.** Per §0 finding 2, precision counts
wrong links; it should also weigh them by the kinematic error they inject.
Something as simple as the rms spurious acceleration per wrong link would
have ranked these six configurations correctly where precision did not.

**1e. Report both objectives together, and find the frontier.** Every
evaluation from here on reports an **accuracy axis** and a **length axis**
side by side, never one alone:

| axis | statistics |
|---|---|
| accuracy | `a_rms` error vs truth, `K_a` vs truth, >5σ contamination rate, per-point velocity/acceleration error, damage-weighted link error (1d) |
| length | mean track length, fraction > 10 and > 30 frames, `e_track` `n_perfect`, and — the one that actually matters for the physics — **span in units of the Lagrangian integral time τ_L**, since "long" is only meaningful relative to the flow's own timescale |

Then sweep the parameters that plausibly move both (`dacc` / seeded search
box, gap-bridging `max_gap` and its accel tolerance, conflict rule) and
compute the **Pareto frontier** explicitly: a configuration is kept only if
nothing else is at least as good on accuracy *and* at least as long.
Deliverable is the frontier plus a recommended default sitting on it — the
existing `dacc=3.6` observation says the shipped default is not on it.

This also gives the honest way to expose the trade to a user: not one blessed
setting, but a small number of frontier points labelled by the question they
serve ("acceleration statistics" vs "time correlations / dispersion"), since
the right point genuinely differs between those.

**Phase 1 exit criterion:** re-run the §0 table on data with realistic
intermittency and realistic noise, reporting both axes, and have the ranking
and the frontier be *stable and explicable*. If the ranking still contradicts
precision, that is the real result and Phase 2 proceeds on the new metric.

### Phase 2 — evaluate the ten trackers properly

**Survivors decided 2026-08-17 (user call, informed by but not purely
derived from the evidence above):** keep **five** —

- `priority_segment_3d` (3MA) — cost-based sequential, the default; best
  measured kinematic fidelity on every case run so far.
- `4be` — four-frame best-estimate; keep for its precision/give-up mechanism
  even though its bridged/naive form is unsafe (§3) — Phase 3 needs it to
  isolate what future-support buys.
- `trackcorr`, **forward and backward** (`full_multipass`/`two_directional`
  presets, `cython_epipolar_tracking.py`) — "our best legacy code," the
  C-translated 2D+3D epipolar engine, the only survivor that searches in
  image space during linking rather than pure 3D.
- `myptv_3d_tracking` — openptv2's port of MyPTV's method.
- `proptv_tracking` — openptv2's port of proPTV's GMM-based method.

Deleted from the inventory: `kalman_hungarian_3d`, `nearest_hungarian_3d`,
`sg_hungarian_3d`/`fast_3d_smooth_tracking`, `predictive_gmm_3d`. §0's
proPTV table already showed this whole Hungarian/NN family collapsing to
identical, worse-than-3MA kinematics at the one density tested — consistent
with cutting them, though note (from that same table) the collapse was
partly because that density was too easy to separate them, not full proof
each is individually worse than every other cut engine at every condition.
`predictive_gmm_3d` is redundant with keeping `proptv_tracking` directly
(same source method); the Hungarian-gated engines were already deprioritised
in the 2026-08-16 plan on Ouellette's conflict-breaking argument, and now
have a second, independent kinematic reason to go.

**Scoping note on trackcorr + proPTV data — plan corrected 2026-08-17.**
trackcorr's C engine performs 2D+3D epipolar search *during* linking, not
just 3D position matching — it needs calibration that is geometrically
consistent with whatever pixel coordinates it's given, not merely present.
Originally scoped as "build a Soloff→openptv2 `.ori`/`.addpar` converter,
deferred" (2026-08-15 plan §5). **User correction: don't convert Soloff at
all — calibrate directly with openptv2's own machinery instead.** proPTV's
`origin_*.txt` already carries, per frame, each particle's true 3D position
*and* its observed 2D pixel projection in every camera that saw it
(`xc0,yc0,...,xc3,yc3`) — exactly the known-3D/observed-2D control-point
pairs openptv2's own orientation/bundle-adjustment pipeline (ray tracing +
`mmlut` for refractive interfaces, `autocalibration.py`, the
`openptv-calibrate` skill) needs to solve for `.ori`/`.addpar` directly. This
is not merely nicer than a Soloff converter, it's the *only* practical
option confirmed 2026-08-17 — proPTV's `input/calibration/` only holds
Soloff coefficient text files, no calibration-target photos, so there is no
raw material to calibrate from except these ground-truth correspondences.
Concretely: pick one (or a few, for redundancy) frame's rows with 3D
position + all-camera 2D projections, feed them to openptv2's orientation
routine as control points (a synthetic calblock, in effect), solve per-camera
`.ori`/`.addpar` with the existing ray-tracing/`mmlut` model. **Decision:**
build this next as the actual unblock for trackcorr-on-proPTV, rather than
scoping trackcorr out to openptv2-native datasets only. Once real
`.ori`/`.addpar` exist, extend `adapt_proptv_dataset.py` to write per-camera
2D targets through the zarr `RunStore.write_targets` API (per
`docs/plans/2026-08-15-zarr-only-transition-plan.md` — targets are zarr-only
now; `res/rt_is.*`/`res/origin_*.txt` ASCII, `store=None`, stays valid
permanently and needs no change) and populate real (non‑`-1`) camera-index
columns in `rt_is` from the same correspondences.

Original ten-engine inventory (`plugins/loader.py`), for reference / in case
a cut needs revisiting:

| engine | aliases | free parameter(s) to sweep | status going in |
|---|---|---|---|
| `cython_3d_tracking` | `default`, `fast`, `fast_3d`, `priority_segment_3d`, `splitter_tracking`, … | `dacc` (search-box scale), bridging on/off | the default; 3MA cost |
| `cython_epipolar_tracking` | `trackcorr`, `full_multipass`, `two_directional`, `standard_forward`, … | `dacc`/`dvxmax` equivalents, bridging | the C-translated 2D+3D path |
| `four_be_tracking` | `4be` | `dacc`, `GREEDY_CONFLICTS`, `STRICT_SUPPORT` — **never bridged** (§3) | best precision, most fragmented |
| `fast_3d_smooth_tracking` | `sg_hungarian_3d`, `openptv2_3d_smooth` | SG window length, Hungarian gate radius | SG extrapolation + Hungarian |
| `kalman_hungarian_3d` | — | process/measurement noise ratio, gate radius | deprioritised (2026-08-16 §4) |
| `nearest_hungarian_3d` | — | gate radius | deprioritised |
| `predictive_gmm_3d` | `proptv`-adjacent | GMM component count / bandwidth | |
| `myptv_3d_tracking`, `myptv_2d_tracking` | — | native params, TBD | external-method ports |
| `proptv_tracking` | — | native params, TBD | external-method port |

This is a genuine two-objective ranking, not a scalar leaderboard — §0 already
shows the highest-precision engine is not the most accurate one, and the
value function (top of this doc) treats length as a real second objective, not
a tiebreaker. So "which 4-6 survive" cannot be read off a single sorted list;
different engines can be optimal in different corners of the
(accuracy, length) plane. The protocol below is designed to produce that
shape of answer rather than force a false total order.

**2a. Per-tracker frontier, not per-tracker point.** For each engine, sweep
its *own* free parameter(s) (table above) — 2026-08-16 §3.3 showed a single
shared `dacc` silently favours whichever tracker it happens to suit, so no
engine may be judged at a value chosen for another. Compute both axes (§1e)
at every grid point and keep only the engine's own Pareto-optimal points —
that reduced set is the engine's frontier.

**2b. Cross with operating conditions.** Repeat 2a at every cell of a small
condition grid: seeding density (`synthetic_turbulent` 220 p/frame,
`synthetic_turbulent_1k` 970 p/frame) × injected position noise (1b — start
with `{0, one realistic value}` once it exists; expand to a fuller sweep only
if engines re-rank between the two). An engine's frontier is reported
per-condition, not averaged across conditions — §0 already shows density
alone reorders the ranking, so collapsing conditions would hide exactly the
effect this phase exists to find.

**2c. Joint frontier and the keep/merge/delete rule.** Per condition, overlay
all ten engines' points in the same (accuracy, length) plane and compute the
*joint* Pareto frontier across all of them together. Then, per engine:

- **Keep** — it contributes at least one non-dominated point to the joint
  frontier, in a condition that corresponds to a plausible real use case (not
  a degenerate corner such as "only wins at exactly zero noise").
- **Merge** — its frontier is ~coincident with another engine's across every
  condition tested (redundant mechanism, per Phase 3); keep whichever
  implementation is simpler or faster, drop the other.
- **Delete** — dominated everywhere, in every condition, by some other
  engine's frontier point.

An engine must not be deleted for being measured at a badly chosen parameter
value — 2a's per-engine sweep exists precisely to prevent that.

**2d. Use-case tagging for survivors.** Label each surviving engine by which
region of the joint frontier it wins, in the vocabulary §1e already sets up:
the high-accuracy / short-length corner serves single- and short-lag
statistics (acceleration PDF, `K_a`, velocity gradients); the long-length
corner (subject to whatever accuracy floor open question 3 settles) serves
time correlations, τ_L, pair dispersion; a middle point is the general-purpose
default. This tagging — not a single "best tracker" — is the actual
deliverable users see.

**2e. Mechanism tie-break.** When two engines are both non-dominated and
close, prefer the one whose winning mechanism is legible and reusable (feeds
Phase 3's synthesis) over one that wins by opaque tuning — a frontier point
that can't be explained mechanistically is a worse foundation for Phase 3
than a slightly worse point that can.

**Deliverable:** a keep / merge / delete decision per engine with the joint
frontier and use-case tags attached as evidence, not a single ranked list.

**Practical sequencing:** 2b's noise axis depends on Phase 1b (not yet
built). Do not wait on it to start — run 2a/2c/2b-density-only (both existing
synthetic sets, zero injected noise) as a first pass now; it is cheap
(existing per-run cost is sub-second at these frame counts per §3.5 of the
2026-08-16 plan) and already gives a real joint frontier, just an incomplete
one. Flag it explicitly as preliminary — noise is the dominant untested axis
(§1) and engines may still re-rank once it's added — and rerun once 1b lands.

**Superseded by the actual decision** (survivors section above, decided
2026-08-17): five engines, not "4-6, to be tested" — `priority_segment_3d`,
`4be`, `trackcorr` (fwd+bwd), `myptv_3d_tracking`, `proptv_tracking`. The
2a-2e machinery above still runs, but its job now is to build each
survivor's frontier and use-case tags, not to decide who survives.

### Phase 3 — dissect the winners, then build the simple one

Only after Phase 2. For each surviving tracker, isolate *which mechanism*
earns its score, by ablation rather than by argument. The candidate
mechanisms already visible in this codebase:

- **the prediction model** — constant velocity (3MA), the candidate's own
  implied velocity two frames ahead (4BE eq. 12), Savitzky-Golay
  extrapolation (`fast_3d_smooth`), Kalman;
- **the cost** — acceleration residual vs future-support distance;
- **the search-box scale** — the §0 finding that this dominates kinematic
  damage says this may matter more than the cost function;
- **conflict resolution** — give-up (4BE), cost-ordered greedy (3MA),
  Hungarian. 2026-08-16 §3.4 measured give-up vs greedy; Hungarian is
  measured but against Ouellette's warning;
- **use of the future** — 4BE's n+2 support is the one mechanism with no
  analogue in the others, and it is what buys its precision.

The synthesis the user anticipates — "from 3–5 best trackers, a simple idea
that also will be the best" — has a concrete shape suggested by §0: a
tracker that **uses future support to accept a link (4BE's discriminating
power), keeps the search box tight enough that any mistake it does make is
kinematically small (the `dacc` finding), and declines rather than guesses
under conflict (4BE's rule)** — then recovers length by a stitching pass that
runs only on fragments it has reason to trust.

Note the coupling that §3.4 of the 2026-08-16 plan already exposed, because
it constrains this design: give-up-on-conflict and naive gap bridging fight
each other. A declined conflict looks exactly like a gap to the bridger, so
it re-creates the link the tracker refused (K_a 4.33 → 343). **Whatever
declines a link must record *why*, so the stitcher can distinguish "particle
was not detected here" from "I refused to guess here" and bridge only the
former.** That is the concrete mechanism by which length gets recovered
without spending accuracy, and it does not exist in any current tracker.

---

## 3. What NOT to do, and why

- **Do not tune for track length *alone*.** Length is a real objective and
  must be reported and optimised — but only along the accuracy axis, never
  instead of it. The failure mode to avoid is a metric that rewards guessing:
  a tracker that links optimistically scores better on mean duration while
  destroying the acceleration statistics. Length is safe to pursue when it
  comes free (an accuracy-neutral parameter change, or stitching applied to
  fragments already known to be correct) and unsafe when it is bought by
  loosening a gate. `pmt` is separately unusable as a quality rate — it is
  computed over predicted tracks and *rises* with fragmentation (2026-08-16
  §2.1) — so it is not a length metric either.
- **Do not turn on gap bridging for 4BE.** K_a 4.33 → 343.
- **Do not read the 2026-08-16 §3.3/§3.4 rankings as final.** They rank on
  precision and yield, which §0 shows is the wrong ordering. Their
  *mechanistic* findings stand; their verdicts are provisional.
- **Do not add a tracker.** The instruction is to reduce.

## 4. Open questions for the next session

1. ~~Is there existing JHTDB access/tooling in this workspace~~ **Answered
   2026-08-17: no.** `src/openptv2/benchmarking/jhtdb_client.py` (named in
   `docs/plans/differentiable_ptv_nextgen_plan.md`) does not exist; that plan
   and `docs/lagrangian_turbulence_quality_guide.md` are unimplemented
   proposal docs, not working code. "JHU data" in the 2026-08-16 plan and in
   `tracking_postprocess.py`'s docstring is informal shorthand for
   JHTDB-*like* statistics used once to sanity-check a warmup parameter
   estimate (`dvxmax~52mm`), not an actual connection or downloaded dataset.
   Phase 1a starts from scratch: the real client library is `pyJHTDB`
   (`pip install pyJHTDB`, needs a free API token from
   turbulence.pha.jhu.edu), and it has not been evaluated for local/offline
   availability, dataset size, or download cost. **Moot for now** — proPTV's
   `500_25`/`500_30` (found 2026-08-17, see §1a above) already has measured
   real intermittency (K_a 13–22) sitting on disk with no access blocker, so
   JHTDB is deferred rather than pursued next.
   - Also checked: `C:/Users/alex/projects/Alex_Ruiz_Test_cases/JHU` (raised
     by the user alongside proPTV) is a **simulated 4-camera 3D-PTV
     experiment** — synthetic images rendered in a 3D scene, per Alex Ruiz's
     published method (corrected 2026-08-17: not a real physical rig, despite
     `experiment.yaml`'s realistic-looking 1000 Hz acquisition metadata).
     149 frames (`res/post_analysis.nc` attrs: first=10001, last=10100),
     already calibrated, previously run end-to-end on openptv-cloud
     (`res/run.zarr`, `res/post_analysis.nc` — Eulerian mean/rms/TKE fields
     only, no Lagrangian output stored here). Despite the "JHU" name it is
     **not JHTDB** — no `origin`/truth files exist in *this* checked-out
     folder, and nothing references DNS or Reynolds number.

     The user also confirms (2026-08-17) it **is the same case Ron Shnapp
     (MyPTV's developer) used in this lab for a pair-dispersion-in-turbulence
     study** — i.e. an expert practitioner already validated this exact case
     for the hardest of the criteria this program cares about (criterion D,
     pair dispersion, §1 — "no harness at all" today).

     **Confirmed 2026-08-17: no ground truth is available** for this case —
     it is not recoverable, so it is **not** a Phase 1a source (1c's
     truth-referenced metrics need per-particle truth this case cannot
     supply). It still earns a role, but a different, weaker one than
     anything else in Phase 1: **cross-tracker agreement as a no-truth proxy**,
     used only in Phase 2, only after Phase 1a/1b/1c are built and every
     tracker has a tuned operating point from real ground truth. Concretely:
     - Run the Phase-2 survivors on this case and compute pair-dispersion
       ⟨Δr²(t)⟩ (criterion D) per tracker.
     - Compare trackers **against each other**, not against truth: agreement
       across independently-implemented trackers is evidence (not proof) of
       correctness, and an outlier is evidence of a tracking artefact —
       exactly the inverse framing of §0's `pmt`/precision trap, so read it
       cautiously (agreement can also mean shared bias, e.g. every engine
       using the same seeded-search assumption).
     - Sanity-check the aggregate curve against **known physics**, not
       per-particle truth: Richardson's t³ super-diffusive scaling at
       separations inside the inertial range is a real, checkable target
       independent of this dataset's own trajectories.
     - If Ron Shnapp's own published dispersion numbers for this case are
       available, treat them as an external anchor, but as literature
       comparison, not as ground truth wired into the metric harness.

     This does not replace Phase 1a — it is the only planned use of a
     genuinely dispersion-purpose-built, expert-validated case, so it is
     worth keeping as Phase 2's plausibility check even without truth. It
     also remains a `myptv_3d_tracking`-vs-upstream-MyPTV cross-software
     check on the exact case that engine's real-world counterpart was used
     for.
2. Is there a real experiment whose parameters should define the noise sweep
   and the dt/τ_η ratio, or do we take those from the literature?
3. Does the acceleration-accuracy target have a number attached — e.g. "a_rms
   within 5% and K_a within 10% of truth" — or is it "best available"? This
   matters more than it looks: an explicit accuracy floor turns the
   two-objective problem into "maximise length subject to the floor", which
   is a much easier thing to optimise and to defend than picking a point on
   a frontier by taste.
4. Which questions actually need the long trajectories, and how long is long
   — expressed in τ_L, not frames? An integral timescale needs a few τ_L; a
   Richardson-dispersion measurement needs far more. That number sets how
   hard the length objective should be pushed, and nothing in this repo
   currently states it.
5. ~~`test_cavity` is 4 frames and poorly conditioned. Is a well-conditioned
   real dataset obtainable?~~ **Answered 2026-08-17: yes, likely.**
   `C:/Users/alex/projects/Alex_Ruiz_Test_cases/JHU` — real 4-camera
   experiment, 149 frames, already calibrated, already run end-to-end once
   (`res/run.zarr`). Noise-conditioning (the actual thing that made
   `test_cavity` unusable — z-noise ≥ frame-to-frame motion) has **not** been
   checked yet; run `scripts/audit_position_noise.py`-style diagnostics
   against it before promoting it to the real-data gate. `Burger/` in the
   same repo is a second, smaller 4-camera real case worth the same check.

---

## 5. Reproducing §0 and §1

The throwaway scripts behind every number above are in this session's
scratchpad; they should be promoted into `scripts/` as part of Phase 1c
rather than rewritten:

- ground-truth physics audit (K_a, autocorrelation, τ_L) — §1 rows 1, 3, 4;
- stereo position-noise audit vs `origin_*.txt` — §1 row 2;
- per-tracker acceleration fidelity assay — the §0 tables.

Datasets: `test_data/synthetic_turbulent` (220 p/frame, 236 true tracks) and
`test_data/synthetic_turbulent_1k` (970 p/frame, 1016 true tracks), frames
10001–10030.

**Promoted 2026-08-17** (already in `scripts/`, not scratchpad): `scripts/
adapt_proptv_dataset.py` (proPTV `origin_*.txt` → openptv2 benchmark dataset,
writes `test_data/proptv_500_25`/`proptv_500_30`) and `scripts/
bench_proptv_kinematics.py` (per-tracker kinematic ranking on the adapted
data, produced the §0 proPTV table). The four scratchpad scripts referenced
above (`pareto.py`, `accel_fidelity.py`, `pos_noise.py`, `gt_physics.py`)
are still not promoted — do that as part of Phase 1c/1d rather than
rewriting them from scratch.

---

## 6. Next steps (as of 2026-08-17, end of session)

In priority order — each unblocks something later in the list:

1. ~~Calibrate proPTV's rig via openptv2's own orientation pipeline~~ **Done
   2026-08-17.** `scripts/calibrate_proptv_dlt.py`: classic DLT camera
   resection (Abdel-Aziz & Karara 1971, ~standard numpy/scipy, openptv2 has
   no from-scratch resection) bootstraps each camera's pose from
   `origin_00000.txt`'s known-3D/observed-2D correspondences, then
   openptv2's own `external_calibration` (`raw_orient`) + `full_calibration`
   refine it (interior + `k1,k2,p1,p2` distortion). No `mmlut`/refraction
   needed — proPTV's images are a pure synthetic render (n1=n2=n3=1, plain
   pinhole), confirmed by the residuals converging cleanly without it.
   Verified against a known synthetic camera before trusting it on real data
   (`calibrate_proptv_dlt.py selftest` — exact recovery of position, `cc`,
   and orientation angles). Result on `500_30`, all 4 cameras: reprojection
   RMS **0.94–1.96 px** (well inside the `openptv-calibrate` skill's own
   "≲2px is good" bar), written to `test_data/proptv_500_30/cal/camN.tif.ori`
   /`.addpar`. `parameters_Run1.yaml`'s `ptv.imx/imy/pix_x/pix_y` patched to
   match this dataset's actual convention (800×800 px, 1.0 world-unit/px —
   not the mm-scale scaffold defaults). Confirmed the new calibration doesn't
   break the existing 3MA path (500/500 links, unchanged). One real gotcha
   worth remembering if this is ever redone: the DLT projection matrix has a
   global sign ambiguity (from the SVD null vector) that plain RQ
   decomposition does not resolve — must be pinned by checking actual point
   depth/reprojection sign against the input data, not by determinant or
   diagonal-sign heuristics alone (both were tried and both were
   insufficient; see the script's comments).
2. ~~Extend `adapt_proptv_dataset.py`~~ **Partially done 2026-08-17, user
   chose "build store-backed reading" (not ASCII targets).** Correction to
   the earlier scoping note above: a first grep for `read_targets`/
   `has_targets` missed that `tracking_frame_buf.read_targets()` (the
   general entry point every `Frame.read()` call goes through) **already**
   checks `store.has_targets(cam, frame)` before falling back to ASCII, and
   that path is already threaded the whole way down
   (`Tracker.restart→TrackingRun(store=...)→TrackBuf.read_frame_at_end→
   Frame.read→read_targets`) — so no new store-reading capability needed
   building after all; only *writing* real target data into the store was
   missing.
   - **Done:** `adapt_proptv_dataset.py` now opens `res/run.zarr` and calls
     `RunStore.write_targets(cam, frame, targets)` per camera/frame from
     `origin_*.txt`'s `xc0,yc0,...,xc3,yc3` columns (NaN → not seen by that
     camera). `rt_is`'s camera-index columns now hold each particle's real
     0-based position in that camera/frame's stored target array (previously
     always `-1`) — exactly the correspondence a real detection+matching
     stage would have produced.
   - **Verified the plumbing itself works:** before this, `trackcorr`
     produced a hard 0 links every frame (500 lost/500, every step). After,
     link count is nonzero (a real, reproducible, frame-specific pattern —
     frames 3,4,16,17 of 30 get exactly 1 link each, every run) — proof the
     store-backed target/correspondence data is actually being read and
     used, not silently ignored.
   - **Investigated further 2026-08-17, real progress, not fully resolved.**
     Traced the actual kernel: `run_tracker`'s track_overrides map to
     `TrackingParams` (`dvxmax`→`set_dvxmax` etc.) which feed
     `trackcorr_loop_fast` (`src/openptv2/algorithms/track_kernels_corr.py`)
     via `lmax = norm([dvxmin-dvxmax, ...])` and `dacc` directly — `eps0`
     does **not** appear in that kernel's signature at all (confirmed by
     reading it), so widening it earlier genuinely could not have changed
     anything; it's a detection/correspondence-stage parameter, not a
     tracking one, and the "zero effect" result above is now explained, not
     mysterious. `X_lay`/`Zmin_lay`/`Zmax_lay` *do* feed the kernel (via
     `vpar`), but weren't the bottleneck either.
     
     **The real bottleneck was `dvxmax`/`dacc` scale, just far more than the
     3MA-derived guess of 0.05/0.02.** A direct sweep (holding everything
     else fixed) shows real, monotonic sensitivity trackcorr's own dvxmax=
     0.05 test didn't reveal because it was uniformly too tight:
     
     | dvxmax | dacc | tracks-with-≥2-points | mean length |
     |---|---|---|---|
     | 0.05 | 0.02 | ~4 (the original near-zero result) | ~1.0 |
     | 0.5 | 0.2 | 90 | 1.01 |
     | 2.0 | 1.0 | 1167 | 1.14 |
     | 3.0 | 1.5 | 1231 | 1.16 |
     | 5.0 | 2.5 | 1238 | 1.16 |
     | 8.0 | 4.0 | 1243 | 1.16 |
     
     So trackcorr's own effective search-box scale on this dataset is
     roughly **40-60× larger** than the value that gives 3MA a perfect
     500/500 — i.e. `dvxmax`/`dacc` are not interchangeable across engines
     even in the same physical units, consistent with 2026-08-16 §3.3's
     original finding that a shared `dacc` silently favours whichever
     tracker it suits (now independently reproduced on a different engine
     pair, different dataset). `dvxmax≈2-3`/`dacc≈1-1.5` is the right
     regime here, not `0.05`/`0.02`.
     
     **Still open, and stranger than a simple plateau: the dvxmax/dacc
     relationship is non-monotonic, ruling out a plain "too tight vs. too
     loose search box" story.** Full sweep, `dacc` held at `dvxmax/2`:

     | `dvxmax` | with-links (of 15000 track-slots) |
     |---|---|
     | 0.05 | ~4 |
     | 0.08 | 3 |
     | 0.1 | 2 |
     | 0.15 | 0 |
     | 0.2 | 5 |
     | 0.3 | 18 |
     | 0.5 | 90 |
     | 2.0 | 1167 |
     | 3.0-8.0 | ~1230-1243 (flat) |

     If this were a simple search-radius effect, yield should rise smoothly
     as `dvxmax` grows past the true velocity scale (~0.05, from this
     dataset's own measured velocity rms) — instead it *dips to zero* at
     `dvxmax=0.15` before climbing, and only becomes non-negligible once
     `dvxmax` exceeds the domain size itself (~1, the unit cube), then flats
     out at a low ~8.5% ceiling. Also checked and ruled out: `flagNewParticles`
     /`add` (0 vs 1, zero effect, same 1167 links at `dvxmax=2`).

     **Two further hypotheses read from source and tested, both real fixes,
     neither the root cause:**

     - **Cold-start prediction (2026-08-17, at the user's direction, "use
       3MA's cold start for trackcorr"):** confirmed trackcorr's cold-start
       (no previous link) branch assumes zero velocity (`X[2]=X[1]`,
       `_trackcorr_particle_fast`), unlike 3MA's Level 2 (borrow velocity
       from a spatial neighbour that already has a track,
       `track_kernels_track3d.py`) — every other survivor does something
       like 3MA's version (4BE: nearest-neighbour; `proptv_tracking`: an
       explicit NN init phase). Ported the same neighbour-velocity-averaging
       into trackcorr's cold-start branch (`orig_parts_1` threaded through
       as a new parameter, rebuilt). **Result: no meaningful change** (same
       counts within noise). Diagnosed why: the neighbour-donor pool is
       empty exactly when it matters most — on the true first frame nothing
       has a previous link yet (identical to 3MA's own situation), and since
       almost nothing links on that first hop regardless, there's nothing to
       propagate to later frames either — a chicken-and-egg gap the fix
       doesn't break. Genuine negative evidence *against* "cold-start
       choice" being the dominant cause, and points back to the structural
       cause: 3MA only needs **one** prediction-and-match to succeed
       (frame1→2); trackcorr requires **two**, cascaded and each
       independently gated (frame1→2 *and* frame2→3), before accepting
       anything — a harder bar regardless of how good the first guess is.
       Kept the fix (correct, matches every other engine's design, just not
       sufficient alone).
     - **Target sort order (2026-08-17):** found the real candidate search
       (`candsearch_in_pix_fast_nogil`, `track_kernels_search.py`) does a
       **binary-search jump** into the per-camera target array assuming
       it's sorted by y-pixel, then linear-scans with an early `break` the
       instant it sees `y > ymax` — both silently wrong on unsorted input.
       `gui/ptv.py` always calls `targs.sort_y()` before targets reach this
       code path in every other run; `adapt_proptv_dataset.py` wrote
       targets in origin-row order, violating that convention. Fixed (now
       sorts each camera's target list by y-pixel before writing, remapping
       `rt_is`'s camera-index columns to the sorted positions). **Result: no
       meaningful change either** (110 vs 90 at `dvxmax=0.5`, 1015 vs 1167
       at `dvxmax=2.0` — within noise, if anything slightly worse at the
       high end). Kept the fix regardless — a real, independently-correct
       requirement this adapter was violating, just not the dominant
       bottleneck.

     **Sixth hypothesis, 2026-08-17: FOUND AND FIXED — this was the real
     bottleneck.** `candsearch_in_pix_fast_nogil`
     (`track_kernels_pixel.py`, the copy actually cimported by
     `_trackcorr_particle_fast` — a separate, unfixed duplicate also exists
     in `track_kernels_search.py`, not on trackcorr's call path, left
     alone) hardcoded its per-camera candidate return to the **nearest 4**
     via four named variables (`p1..p4`/`d1..d4`), regardless of the outer
     `MAX_CANDS_K=32` buffer sizing every caller already assumed. It does
     correctly rank by pixel distance (verified), so this was invisible at
     low density/tight search boxes, but once the search window was wide
     enough to contain more than 4 real candidate targets, the true match
     could be silently discarded before the acc/angle quality gate ever
     saw it — exactly the failure mode a widening `dvxmax` cannot fix,
     matching the observed plateau. Rewrote it to a proper
     `max_cands`-parameterized insertion sort (generalizes the same
     4-element logic to N elements, same algorithm, same ranking), and
     updated its caller (`_sorted_candidates_fast_out_nogil`) to size its
     scratch buffers to `max_cands` instead of a literal `4`. Rebuilt;
     `tests/unit/test_track3d.py`, `test_track.py`,
     `test_cython_epipolar_direction.py` (28 tests) still pass; the 5
     direct unit tests for this function
     (`test_track_kernels_tracking_coverage.py`) updated for the new
     signature (`max_cands`, `out_dists` added) — that file only runs in
     pure-Python coverage mode, not exercised by this rebuild, but the
     calls now match the new signature for when it is.

     **Measured improvement on `proptv_500_30`:**

     | `dvxmax` | before this fix | after |
     |---|---|---|
     | 0.5 | ~90-110 | **1328** (≈12-14×) |
     | 2.0 | ~1015-1243 | **1811** (mean length also up, 1.14→1.27) |

     A real, substantial improvement. **Full re-sweep after the fix,
     replacing every earlier table in this section (measured against the
     buggy 4-candidate cap and not representative):**

     | `dvxmax` | with-links (of 15000 track-slots) | mean length |
     |---|---|---|
     | 0.1 | 1 | 1.00 |
     | 0.2 | 26 | 1.00 |
     | 0.3 | 389 | 1.05 |
     | 0.5 | 1328 | 1.17 |
     | 1.0 | 1748 | 1.28 |
     | 2.0 | 1811 | 1.27 |
     | 3.0 | 1793 | 1.27 |
     | 5.0 | 1799 | 1.27 |

     **This confirms the earlier non-monotonic pattern (dip to zero at
     `dvxmax=0.15`, erratic small counts) was substantially an artefact of
     this bug (compounded with the target-sort-order bug) — the fixed
     kernel gives a clean, monotonic, textbook search-radius curve**:
     smooth climb, plateau at `dvxmax≈1-2` around **1800 links (~12.4%
     yield)**. That plateau is real progress (up from ~8.5% pre-fix, and
     the shape is now explicable rather than mysterious) but still far
     below a usable yield — 3MA gets 100% on the same data. Consistent with
     the structural explanation already established: trackcorr's compound
     two-hop gate (frame1→2 *and* frame2→3, both required) is a genuinely
     harder bar than any single-hop survivor faces, and on real
     intermittent turbulence some real fraction of true transitions may
     legitimately fail the acc/angle smoothness check — a physics-vs-
     algorithm mismatch, not a bug, if so. `_angle_acc_out`'s live gate
     values are the next thing to check (not yet done) if this plateau is
     worth chasing further.
3. **Run Phase 2's 2a-2e protocol properly** on the five survivors: per-
   engine parameter frontier (2a) × density (2b, proPTV `500_25` vs `500_30`
   vs the two openptv2 synthetic sets) → joint frontier (2c) → use-case tags
   (2d). What's been run so far (§0's proPTV table) is a single operating
   point per engine, not a frontier — real ranking needs the sweep.
4. **Build Phase 1b (injectable position noise)** — the dominant untested
   axis per §1, needed before any of the above frontiers can be trusted as
   more than a zero-noise special case. Apply uniformly to both openptv2's
   synthetic sets and the proPTV-adapted sets (same knob, one code path).
5. **Phase 3 mechanism dissection**, once (3) produces real per-survivor
   frontiers — including the sharpened two-failure-mode framing from the
   proPTV table (concentrated-few-links damage vs. spread-many-links
   damage; §2's mechanism list should track them as separate axes).
6. Lower priority / opportunistic: resolve open questions 2-4 (real noise
   parameters, explicit accuracy floor, τ_L-denominated length target); run
   the JHU cross-tracker/pair-dispersion plausibility check (§1a) once (3)
   produces stable survivor parameters to run it with; check JHU/`Burger`
   noise-conditioning (open question 5) before promoting either to the
   real-data gate.
7. **Write a proPTV<->openptv2 calibration tutorial** (requested 2026-08-17,
   documentation deliverable, not code) — once (2)'s store-backed target
   reading exists and the trackcorr round-trip is verified end-to-end, so
   the tutorial documents a working pipeline rather than a half-finished
   one. Target: a standalone doc (e.g. `docs/proptv-openptv2-calibration.md`)
   a future session or another user can follow without re-deriving anything
   from this session. Should cover, in both directions:
   - **proPTV → openptv2** (built 2026-08-17, `scripts/
     calibrate_proptv_dlt.py`): proPTV's data layout (`origin_*.txt` columns,
     Soloff `.txt` calibration files, no calibration-target photos — why that
     forces the ground-truth-as-control-points approach rather than a format
     converter); the empirical method used to pin down openptv2's exact
     projection sign convention (`metric = -cc*(R@(X-C))_xy/(...)_z`, `dm`
     stored camera-to-world but consumed transposed by `img_coord` — found by
     direct numerical probing, not by reading the Cython source, and that's
     a deliberately reusable technique worth teaching); DLT camera resection
     as the from-scratch bootstrap (openptv2 has none) with its real gotcha
     (global sign ambiguity from the SVD null vector — resolved by depth/
     reprojection sign, not by determinant or diagonal-sign heuristics,
     both tried and both insufficient); handoff to openptv2's own
     `external_calibration`/`full_calibration` for refinement; verification
     method (self-test against a known synthetic camera *before* trusting
     real data, then reprojection RMS against the `openptv-calibrate` skill's
     own ≲2px bar).
   - **openptv2 → proPTV** (not yet attempted, direction absent from this
     session): the natural approach is the mirror of the above — sample
     `img_coord()` densely across the calibrated volume to synthesize
     `(X,Y,Z)->(pixel)` pairs, then least-squares fit proPTV's Soloff
     polynomial form to that synthetic dataset (a regression problem, no
     resection needed, since openptv2's model is already dense and exact).
     Needs: reading proPTV's actual Soloff polynomial order/basis from
     `soloff_c{n}x.txt`/`y.txt` (not yet inspected this session — do that
     first), and a validation loop (fit, then compare the Soloff fit's own
     reprojection RMS against openptv2's `img_coord` on held-out points).
   - A worked example on `500_30` (already-verified proPTV→openptv2 numbers
     from this session) plus, once built, the reverse direction's numbers on
     the same case, so a reader can see both directions validated on one
     concrete dataset rather than in the abstract.

**Docs cleanup done 2026-08-17:** removed `docs/plans/master-plan.md`
(2026-08-10) — its two-preset consolidation goal and `_predictive_tracker.py`
unification plan are directly superseded by today's five-named-engine
survivor decision. Left `docs/plans/differentiable_ptv_nextgen_plan.md`,
`docs/differentiable_ptv_autoresearch_architecture.md`, and `docs/
lagrangian_turbulence_quality_guide.md` in place — they're aspirational/
unimplemented (open question 1 already found their referenced
`jhtdb_client.py` doesn't exist), but they describe a materially different,
larger-scope effort (a differentiable PyTorch rewrite) rather than being
directly contradicted by this plan; flagged for the user to decide on
separately rather than removed unilaterally.

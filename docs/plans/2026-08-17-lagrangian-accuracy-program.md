# Lagrangian accuracy first: reduce ten trackers to the few that measure
# physics correctly (2026-08-17)

Goal, in the user's words: keep the minimum number of the best trackers.
Decide what matters for **fluid mechanics and turbulence, not numerical
exercise**; evaluate it properly, slowly and carefully; understand what in
each algorithm produces the better outcome; then reinvent a simple thing that
takes the winning parts.

The value function is explicit and it is **not** track length:

> We value most the accuracy — correct velocity, correct acceleration in the
> Lagrangian sense, then time correlations, distance correlations and other
> things that require long trajectories. Long we can achieve later by
> smoothing and stitching; the correct ones we could not get by
> post-processing wrong trajectories.

That ordering is the whole design constraint. Length is recoverable, and
§3.1/§3.2 of the 2026-08-16 plan just made it recoverable. Correctness is not.

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

**1a. Ground truth with real Lagrangian statistics.** Ranked by preference:

- **DNS particle tracks (Johns Hopkins Turbulence Database).** The gold
  standard: exact velocity *and* acceleration per particle, genuine
  intermittency, correct τ_η/τ_L separation, choosable dt. The 2026-08-15
  plan already references "the JHU data" for warmup, so some access path may
  already exist — find it before building anything.
- **A stochastic model with the right second-order structure** (Sawford's
  two-timescale model) if DNS access is a blocker. Gives correct `a_rms`,
  τ_η and an intermittent-ish acceleration PDF; cheap and local.
- proPTV `500_25` / `500_30` — still needs the format adapter deferred in the
  2026-08-16 plan §4. Useful for cross-software comparison, weaker as a
  physics reference.

Requirement either way: persist **ground-truth velocity and acceleration**
per particle per frame, not just positions. `origin_*.txt` stores only
`(pid, x, y, z)`; the generator in `benchmarking/scenario.py` knows `vel` and
`acc` and throws them away.

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

**Phase 1 exit criterion:** re-run the §0 table on data with realistic
intermittency and realistic noise, and have the ranking be *stable and
explicable*. If the ranking still contradicts precision, that is the real
result and Phase 2 proceeds on the new metric.

### Phase 2 — evaluate the ten trackers properly

Current inventory (`plugins/loader.py`), ten engines behind ~25 aliases:

| engine | aliases | status going in |
|---|---|---|
| `cython_3d_tracking` | `default`, `fast`, `fast_3d`, `priority_segment_3d`, `splitter_tracking`, … | the default; 3MA cost |
| `cython_epipolar_tracking` | `trackcorr`, `full_multipass`, `two_directional`, `standard_forward`, … | the C-translated 2D+3D path |
| `four_be_tracking` | `4be` | best precision, most fragmented |
| `fast_3d_smooth_tracking` | `sg_hungarian_3d`, `openptv2_3d_smooth` | SG extrapolation + Hungarian |
| `kalman_hungarian_3d` | — | deprioritised (2026-08-16 §4) |
| `nearest_hungarian_3d` | — | deprioritised |
| `predictive_gmm_3d` | `proptv`-adjacent | |
| `myptv_3d_tracking`, `myptv_2d_tracking` | — | external-method ports |
| `proptv_tracking` | — | external-method port |

Run every one on the Phase-1 harness across the noise sweep and at least two
seeding densities. Keep the axes honest: `dacc` must be swept per tracker
(2026-08-16 §3.3 showed a single shared value silently favours whichever
tracker it happens to suit), and each tracker gets its own best operating
point rather than one shared `BASE_OVERRIDES`.

**Deliverable:** a keep / merge / delete decision per engine, with the
evidence attached. The expectation — to be tested, not assumed — is that this
collapses to about three: one cost-based sequential tracker, one
best-estimate/4-frame tracker, and one global-assignment tracker as a
control.

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
power) while keeping the search box tight enough that any mistake it does
make is kinematically small (the `dacc` finding), and declines rather than
guesses under conflict (4BE's rule), leaving length entirely to a later
stitching pass.** That is a hypothesis for Phase 3 to test, not a conclusion.

---

## 3. What NOT to do, and why

- **Do not tune for track length, `pmt`, or mean trajectory duration.** They
  are the metrics that look best when a tracker guesses, and guessing is
  precisely what the value function forbids. `pmt` is additionally broken as
  a quality rate (2026-08-16 §2.1).
- **Do not turn on gap bridging for 4BE.** K_a 4.33 → 343.
- **Do not read the 2026-08-16 §3.3/§3.4 rankings as final.** They rank on
  precision and yield, which §0 shows is the wrong ordering. Their
  *mechanistic* findings stand; their verdicts are provisional.
- **Do not add a tracker.** The instruction is to reduce.

## 4. Open questions for the next session

1. Is there existing JHTDB access/tooling in this workspace (the "JHU data"
   the 2026-08-15 plan mentions), or does Phase 1a start from scratch?
2. Is there a real experiment whose parameters should define the noise sweep
   and the dt/τ_η ratio, or do we take those from the literature?
3. Does the acceleration-accuracy target have a number attached — e.g. "a_rms
   within 5% and K_a within 10% of truth" — or is it "best available"?
4. `test_cavity` is 4 frames and poorly conditioned. Is a well-conditioned
   real dataset obtainable? It is now the binding constraint on every
   tracker conclusion in this repo.

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

# Tracker Tutorials: What Each Engine Does, Which Parameters Matter, and How They Compare

This is a from-scratch, verified reference for the five surviving tracker
engines (see `docs/plans/2026-08-17-lagrangian-accuracy-program.md` for how
they were chosen from ten). Every number below was measured, not estimated
— reproduce them yourself with the commands in §1. Nothing is left as "tune
this yourself and see": every parameter each tracker actually reads is
listed, with its concrete resolved value on the reference dataset, and what
happens if you get it wrong.

## 0. The five trackers, in one line each

| name (as passed to `run_tracker`/`--tracker`) | what it is |
|---|---|
| `priority_segment_3d` (a.k.a. "3MA") | Pure 3D, single forward pass, 4-level acceleration-priority cascade |
| `trackcorr` | Multi-camera 2D+3D epipolar search, compound two-hop acceptance |
| `4be` | Pure 3D, scores a candidate by whether a real particle exists two frames ahead |
| `myptv_3d_tracking` | Pure 3D, polynomial velocity prediction + Hungarian assignment (Python) |
| `proptv_tracking` | Pure 3D, Gaussian-Mixture-smoothed track history + Hungarian assignment |

All five are registered in `src/openptv2/tracking_registry.py`
(`TRACKER_REGISTRY`) — that file is the machine-readable version of most of
what's below; this document adds the verified numbers, the comparison, and
the things the registry's prose doesn't capture (e.g. what actually runs by
default vs. what a preset's name suggests).

## 1. How to reproduce every number in this document

Two datasets, same 30-frame, 500-particle proPTV-derived case, same seed:

```bash
# "clean": ground-truth 3D correspondences injected directly, no detection
# or correspondence-stage error at all.
uv run python scripts/adapt_proptv_dataset.py 500_30

# "realistic": the actual detection -> correspondence -> triangulation
# pipeline, at a calibrated "mild" noise severity (see docs/
# holistic-3d-ptv-systems-research-program.md and
# docs/plans/2026-08-17-lagrangian-accuracy-program.md's noise-source sweep
# for what "mild" means and why the other severities are harsher).
uv run python scripts/adapt_proptv_dataset.py 500_30 --realistic --severity mild --seed 0
```

Then, for either dataset:

```bash
uv run python scripts/bench_proptv_kinematics.py     # one dataset, full table
uv run python scripts/bench_with_without_noise.py    # both datasets, side by side (this doc's source)
```

`bench_with_without_noise.py` prints the **actual resolved parameters**
used for each tracker (via `benchmark_utils.per_tracker_overrides`, which
auto-scales the shared velocity/acceleration bounds from the dataset's own
displacement statistics — see §2) immediately before running it, so you can
always see exactly what ran, not just what the defaults nominally are.

## 2. The parameter surface every tracker actually shares

This is the single most important thing to understand before touching any
tracker's own parameter names: **`priority_segment_3d`, `trackcorr`,
`4be`, `myptv_3d_tracking`, and `proptv_tracking` are all driven by the
same eight `track.par` / YAML `track:` fields** — `dvxmin`, `dvxmax`,
`dvymin`, `dvymax`, `dvzmin`, `dvzmax`, `dacc`, `angle`. Every tracker's own
parameter names (`v_max`, `a_max`, `maxvel`, ...) are derived FROM these
same eight fields by `src/openptv2/tracking_presets.py`'s
`unified_velocity_bound()` / `unified_angle_deg()` — not independent
settings. Verified directly (`scripts/benchmark_utils.py`'s
`run_tracker`/`runner.py`, lines ~206-213): every override you pass maps
onto exactly these `TrackPar` setters, regardless of which tracker you
asked for.

| field | meaning | unit | who reads it |
|---|---|---|---|
| `dvxmin`/`dvxmax`, `dvymin`/`dvymax`, `dvzmin`/`dvzmax` | per-axis velocity search box: how far (mm) a particle may move between consecutive frames on each axis | mm/frame | every tracker (directly for `priority_segment_3d`/`trackcorr`/`4be`; converted to an isotropic radius `v_max` = largest of the three for `myptv_3d_tracking`/`proptv_tracking`'s `maxvel`) |
| `dacc` | for `priority_segment_3d`/`trackcorr`: the SEEDED-step search radius (mm) around a velocity-extrapolated prediction, for a particle that already has an established velocity — despite the name, a position tolerance, not an acceleration bound. For `myptv_3d_tracking`: becomes `a_max`, the seeded-track search radius. For `4be`: read for API parity only — **4BE's own cost function never uses it** (§ 4be below). | mm | `priority_segment_3d`, `trackcorr`, `myptv_3d_tracking`; ignored by `4be`'s actual linking |
| `angle` | maximum angular deviation allowed between successive velocity vectors, in **gon** (400 gon = 360°, not degrees) | gon | `priority_segment_3d`, `trackcorr` directly; converted to degrees (`unified_angle_deg`) for `myptv_3d_tracking`'s cone-of-continuity filter and `proptv_tracking`'s own angle check |

**Verified reference values** (this dataset, 500 particles/40mm³ cube,
`scripts/bench_with_without_noise.py`'s printed overrides): on both the
clean and realistic-mild datasets, the auto-tuner resolved `dvxmax` ≈
**1.13mm** (clean) / **1.13mm** (realistic — noise barely shifts the
displacement statistics the tuner reads) and `dacc` ≈ **0.82mm / 0.85mm** —
a `dacc`/`dvxmax` ratio of **~0.73**, on the looser end of the registry's
documented "~0.4x when densely seeded, ~0.8x when sparse" guidance,
consistent with this dataset's moderate (not dense) 500-particle/40mm³
seeding.

**A second, independent, real-world reference point** — `test_data/test_cavity`
(a real experimental dataset, not synthetic), its own committed
`parameters_Run1.yaml`:

```yaml
track:
  dvxmax: 0.6      # mm/frame
  dvymax: 0.6
  dvzmax: 0.6
  dacc: 0.24       # mm  (ratio dacc/dvxmax = 0.4 -- exactly the "densely
                   #  seeded" end of the guidance above)
  angle: 120.0     # gon
criteria:
  eps0: 0.05       # mm -- correspondence-stage epipolar tolerance,
                   #  see §6
```

**How to choose these yourself, concretely, no guessing:**
1. Run a short probe sequence (a handful of frames) through detection +
   correspondence only, with generous bounds.
2. Compute the actual frame-to-frame displacement distribution from the
   probe's `rt_is.#`/correspondences (`openptv2.tracking_recommender.compute_dataset_stats`
   does exactly this — it's what `per_tracker_overrides` calls).
3. Set `dvxmax` = the 99th-percentile displacement + a small margin (not
   the maximum — one outlier shouldn't set your search box for every
   particle).
4. Set `dacc` = 0.4–0.8× `dvxmax`: closer to 0.4 at high seeding density
   (tighter search reduces candidate ambiguity — directly the §4be finding
   below, generalized), closer to 0.8 at low density (looser search
   recovers more genuine fast-moving particles without much added
   ambiguity cost).
5. Leave `angle` at 120 gon (108°) unless you have a specific reason to
   tighten it for laminar flow or loosen it for strongly rotational flow.

## 3. Master comparison: clean vs. realistic

Ground truth on both datasets: `a_rms=0.01101`, `K_a=19.80` (see
`docs/lagrangian_turbulence_quality_guide.md` for what these mean and why
K_a — acceleration kurtosis — is the metric that actually detects
contamination that link-count metrics (precision/yield) miss entirely; a
false trajectory doesn't just add noise, it injects the wrong kinematics
into the recovered statistics).

**Clean** (`adapt_proptv_dataset.py 500_30`, no noise, ground-truth
correspondences injected directly):

| tracker | a_rms error | K_a | precision | yield | meanlen |
|---|---|---|---|---|---|
| `priority_segment_3d` | 0.0% | 19.80 | 1.0000 | 1.0000 | 30.00 |
| `trackcorr` | 0.0% | 19.80 | 1.0000 | 1.0000 | 30.00 |
| `4be` | 0.0% | 19.80 | 1.0000 | 1.0000 | 30.00 |
| `myptv_3d_tracking` | 0.0% | 19.80 | 1.0000 | 1.0000 | 30.00 |
| `proptv_tracking` | 0.0% | 19.80 | 1.0000 | 1.0000 | 30.00 |

Every tracker recovers ground truth exactly here — expected, not
impressive: with zero ambiguity in the correspondence data, any correct
algorithm converges to the same unique answer. This dataset validates
correctness, not robustness; use §"realistic" for robustness comparison.

**Realistic, "mild" severity** (`--realistic --severity mild --seed 0`:
0.08px detection noise, 1% missed-detection rate with streak-correlated
dropout, 1px image merging, real correspondence solving with the actual
`eps0` epipolar tolerance, small per-camera calibration residual — see
`docs/holistic-3d-ptv-systems-research-program.md` §1 for the full chain
and why calibration residual specifically is the dominant damage term):

| tracker | a_rms error | K_a | precision | yield | meanlen |
|---|---|---|---|---|---|
| `trackcorr` | +414.9% | **25.36** | 0.9999 | 0.9888 | 24.28 |
| `myptv_3d_tracking` | +441.8% | **29.04** | 0.9999 | 0.9928 | 26.75 |
| `priority_segment_3d` | +442.4% | **29.15** | 0.9966 | 0.9923 | 28.82 |
| `4be` | +457.3% | **39.46** | 0.9999 | 0.9929 | 26.90 |
| `proptv_tracking` | +471.7% | **73.31** | 0.9999 | 0.9892 | 24.48 |

Ranked by K_a (closer to truth's 19.80 = better recovered physics), sorted
ascending. **Read precision/yield and K_a together, never K_a alone**: every
tracker here has precision/yield above 0.986 — nearly indistinguishable by
link-count metrics — while K_a spans a 3.7× range (25.4 to 73.3). A handful
of wrong links, invisible to precision/yield, dominates the acceleration
statistics because kurtosis weights by the 4th power. This is the concrete
demonstration of why link-count metrics alone cannot certify a tracker for
turbulence-physics work.

## 4. Per-tracker detail

### 4.1 `priority_segment_3d` ("3MA")

**Mechanism**: pure 3D, single forward pass, 4-level cascade —
Level 1 claims high-confidence particles meeting the `dacc` threshold
globally in ascending-cost order; Level 2 falls back to local
neighbor-velocity averaging; Level 3 handles unseeded/static displacement.
Never touches 2D image space or camera calibration at all — it only ever
sees the 3D correspondence positions already handed to it.

**Parameters**: exactly the shared surface in §2, nothing tracker-specific.

**Example** (exact code that produced this doc's numbers):
```python
from pathlib import Path
import benchmark_utils as bu

overrides = bu.per_tracker_overrides(
    ["priority_segment_3d"],
    src=Path("test_data/proptv_500_30"),
    first=10001,
    n_frames=30,
)
tracks, elapsed = bu.run_single_tracker(
    "priority_segment_3d",
    track_overrides=overrides["priority_segment_3d"],
    src=Path("test_data/proptv_500_30"),
    first=10001,
)
```

**Strength**: fastest of the five (9.5s clean / 13.1s realistic on this
30-frame/500-particle case — comparable to `4be`, ~30% faster than
`trackcorr`, and an order of magnitude faster than `proptv_tracking`).
Simple, predictable, no image-space dependency (so immune to any
calibration-model choice entirely).

**Weakness, proven not assumed**: never checks 2D image-space consistency
of a correspondence, so a spurious 3D point (a real epipolar-tolerance
ghost, see `docs/holistic-3d-ptv-systems-research-program.md` §1) looks
exactly like a real one. On a harsher (non-"mild") noise setting earlier
this project's own investigation measured K_a=1908 (nearly 100× truth) for
this exact reason — the largest single-tracker contamination measured this
session, on data where its own precision/yield still looked fine (0.86/0.69).

### 4.2 `trackcorr`

**Mechanism**: the only one of the five that uses 2D image space at all.
Projects a 3D search volume into each camera, finds 2D candidate targets,
and requires a **compound two-hop acceptance**: BOTH the frame *n*→*n+1*
hop AND the frame *n+1*→*n+2* hop must independently pass their own
angle+acceleration gate (see `docs/plans/2026-08-17-lagrangian-accuracy-program.md`
for the historical investigation of this engine). Neither hop can
compensate for the other failing — the discipline `4be`'s cost function
was found to have dropped (§4.3).

**A registry caveat, verified not assumed**: `trackcorr`'s
`tracking_registry.py` entry is aliased to `FULL_MULTIPASS_INFO`, whose
prose describes a **three-pass** pipeline (forward, backward, reciprocity
post-processing). Every benchmark run in this document and throughout this
project's history prints `"Running TrackCorr Tracking (Forward only)..."`
— the `trackcorr` preset, as actually invoked by `run_tracker`/`bench_*`
scripts, runs **forward-only**. If you need forward+backward, request the
`full_multipass` preset explicitly (or set `track.postprocess`/direction
config per `tracking_presets.infer_direction`) — do not assume `trackcorr`
gives you the three-pass pipeline the registry's display name implies.

**Parameters**: exactly the shared surface in §2. No tracker-specific
knobs beyond that.

**Example**:
```python
overrides = bu.per_tracker_overrides(["trackcorr"], src=..., first=10001, n_frames=30)
tracks, elapsed = bu.run_single_tracker(
    "trackcorr", track_overrides=overrides["trackcorr"], src=..., first=10001
)
```

**Strength**: best K_a of all five under realistic noise (25.36, closest to
truth's 19.80) — direct evidence that its image-space cross-check catches
some of the ghost/ambiguous-correspondence contamination the pure-3D
trackers (§4.1, §4.3-4.5) cannot see at all.

**Weakness**: ~40% slower than the pure-3D trackers (10.8-13.5s vs.
7.7-13.1s here) for that cross-check; historically (this project's earlier
investigation) needed two real bugs fixed before it worked at all on real
turbulence data — a `Target.tnr` default-zero bug that collapsed every
match to particle 0, and a hardcoded 4-candidate cap in the pixel-space
search. Both are fixed; see `docs/plans/2026-08-17-lagrangian-accuracy-program.md`.

### 4.3 `4be`

**Mechanism**: pure 3D. Scores a frame *n*→*n+1* candidate by whether a
REAL particle exists near where that candidate's own constant-velocity
extrapolation lands two frames ahead (Ouellette, Xu & Bodenschatz 2006, eq.
12/14) — "trust a candidate more if something plausible exists past it."

**Proven bug, fixed this session** (`src/openptv2/algorithms/track_kernels_track3d.py`,
`track4be_loop_fast`): the cost for a supported candidate used to be the
n+2 support distance **alone**. Live-traced at a confirmed failure
junction: a candidate 16× farther from the correct frame *n+1* prediction
won purely because a real particle happened to sit fractionally closer to
its own (kinematically wrong) n+2 extrapolation. Fixed by summing both
distances — `cost = sup_dists[0] + cand_dists[ci]` — so a coincidental
future match can no longer override overwhelming n+1 evidence. Measured
effect: K_a **761.45 → 103.09** (7.4×) from the code fix alone, then
**103.09 → 39.46** once `4be` was also given the same dataset-scaled search
window every other tracker gets (§4.3's "registry gap" below) — full
before/after in `docs/holistic-3d-ptv-systems-research-program.md` §2.

**A second, independent bug this session found and fixed**: `4be` had **no
entry in `tracking_registry.py`** until this document's benchmark run
surfaced it — `per_tracker_overrides` silently fell back to a flat,
un-scaled `dvxmax=6.0mm` (6× too loose for this dataset) instead of the
~1.13mm every other tracker was auto-tuned to. A looser search box means
more competing candidates per step, which is exactly the density that
makes the eq.12/14 coincidence above more likely — the parameter gap and
the algorithm bug compounded each other. Both are now fixed; `4be` is
registered with the same shared parameter surface as everyone else (§2).

**Parameters actually used by `4be`'s own linking**: `dvxmin`/`dvxmax`/
`dvymin`/`dvymax`/`dvzmin`/`dvzmax` (the candidate search box) — that's it.
`dacc` is accepted (for API parity with the other trackers so the same
override dict works everywhere) but **`4be`'s own cost function never
reads it** — verify directly in `track4be_loop_fast`'s signature: `dacc`
does not appear as a parameter at all, only `dx, dy, dz` (the three
per-axis half-widths). Two further knobs exist but are **not exposed via
`track.par` at all** — module-level constants in `src/openptv2/algorithms/track4be.py`:

```python
STRICT_SUPPORT = 0  # 0 = unsupported candidates fall back to a 3MA-style
#     acceleration-residual score instead of being
#     rejected outright (recovers yield on genuine
#     1-frame detection gaps). 1 = reproduce Ouellette's
#     paper literally: reject any candidate with no
#     real n+2 support.
GREEDY_CONFLICTS = 0  # 0 = the paper's give-up-on-conflict rule: a frame
#     n+1 particle claimed by more than one frame-n
#     particle links to none of them.
# 1 = cost-ordered greedy claiming across the whole
#     frame instead (a particle that loses its first
#     choice may take a later one). Ouellette reports
#     this degrades every heuristic but nearest-
#     neighbor, hence 0 is the default.
```
To change either, edit `track4be.py` directly and rebuild
(`uv run python setup.py build_ext --inplace`) — there is currently no
YAML/CLI path to them. This is a genuine gap (documented, not hidden): if
you need `STRICT_SUPPORT=1` for a literal Ouellette-paper reproduction, you
must edit source.

**Example**:
```python
overrides = bu.per_tracker_overrides(["4be"], src=..., first=10001, n_frames=30)
tracks, elapsed = bu.run_single_tracker(
    "4be", track_overrides=overrides["4be"], src=..., first=10001
)
```

**Strength**: fastest alongside `priority_segment_3d` (7.7-8.9s); after the
fixes above, competitive K_a (39.46, better than `proptv_tracking`'s 73.31).

**Weakness**: still worst of the four pure-3D-or-image-checked trackers
under realistic noise even after both fixes — an open question this
session left unresolved (`docs/holistic-3d-ptv-systems-research-program.md`
§7 roadmap item 1's remaining gap). Candidate hypothesis, not yet checked:
`GREEDY_CONFLICTS=0`'s give-up-on-conflict rule may be discarding more
correct links under noise than the other trackers' conflict resolution
does — untested.

### 4.4 `myptv_3d_tracking`

**Mechanism**: pure 3D, pure Python (not Cython — slower but fully
readable/customizable). Predicts the next position via polynomial velocity
extrapolation from track history, then solves frame-to-frame assignment
with a radius-limited Hungarian algorithm (`openptv2.plugins._assignment.match_within_radius`),
using a configurable multi-term cost matrix (distance/velocity/acceleration/
intensity — `openptv2.tracking_cost`).

**Parameters**:

| name | source | verified value on this dataset |
|---|---|---|
| `v_max` | `unified_velocity_bound(track_cfg)` — largest of `dvxmax`/`dvymax`/`dvzmax` | 1.1292mm (realistic-mild) |
| `a_max` | same `track.dacc` field as `priority_segment_3d`/`trackcorr` | 0.8516mm |
| `angle` | `_suggest_params` sets this tracker's `angle` override to a fixed **200 gon** deliberately, NOT derived from `track.angle`'s own 120 gon (see `src/openptv2/tracking_recommender.py`'s `_suggest_params`, the `nearest_hungarian_3d` branch) — this tracker's angle filter is a hard binary reject, so an overly tight bound throws away good matches instead of de-weighting them; a grid sweep found effectively-unrestricted (200 gon = half the full 400-gon circle) best. `myptv_3d_tracking`'s own code then converts that 200 gon to degrees via `unified_angle_deg` (×0.9) before comparing | **180°** (200 gon × 0.9) — i.e. myptv's cone-of-continuity filter is set intentionally wide open by the auto-tuner, not tight; if you want it tight for laminar flow, override `angle` explicitly in your own `track_overrides` dict rather than trusting the auto-tuned value for this specific tracker |
| `max_gap` | hardcoded at the plugin's own call site, NOT the constructor default and NOT threaded through `per_tracker_overrides` | **1 frame** — verified in `src/openptv2/plugins/myptv_3d_tracking.py`'s `do_tracking()`: `MyPTV3DTracker(..., max_gap=1, ...)` is hardcoded, even though `MyPTV3DTracker.__init__`'s own signature default is `max_gap: int = 2`. The constructor default is NOT what runs. To change it you must edit the `do_tracking()` call site itself. |
| `cost_weights` | plugin constructor default | distance-only unless you construct `CostWeights` yourself and pass it in |

**Example**:
```python
overrides = bu.per_tracker_overrides(
    ["myptv_3d_tracking"], src=..., first=10001, n_frames=30
)
tracks, elapsed = bu.run_single_tracker(
    "myptv_3d_tracking",
    track_overrides=overrides["myptv_3d_tracking"],
    src=...,
    first=10001,
)
```

**Strength**: second-best K_a under realistic noise (29.04, essentially
tied with `priority_segment_3d`'s 29.15) despite being pure Python and pure
3D (no image-space check) — its polynomial-prediction + Hungarian
assignment appears to resist the coincidental-candidate failure mode as
well as any tracker here except `trackcorr`.

**Weakness**: slowest-but-one of the five (8.2-10.3s here, comparable to
the Cython trackers on this small case, but pure Python does not scale the
same way to larger datasets — `max_gap`/`cost_weights` require editing the
call site, not just `track.par`, to change).

### 4.5 `proptv_tracking`

**Mechanism**: pure 3D. Fits Gaussian basis functions to each track's
time-position history (a GMM-style smoothing of the trajectory), analytically
differentiates the fit to get a smoothed velocity/acceleration estimate,
predicts the next position from that smoothed extrapolation, and links via
radius-limited Hungarian assignment on distance+velocity+acceleration
continuity cost. This IS the "smoothness-favoring" mechanism this project's
research plan hypothesized might systematically suppress recovered
turbulence intermittency (`docs/holistic-3d-ptv-systems-research-program.md`)
— **worth reading precisely**: on this dataset's realistic-mild severity,
its K_a (73.31) is the *highest* (most contaminated), not suppressed below
truth, so the smoothness-bias hypothesis is NOT confirmed by this specific
measurement — it needs testing at a noise level with more genuine ambiguity
for the smoothing to have a real choice to bias, and is still an open
question, not settled by this benchmark.

**Parameters**:

| name | source | verified value on this dataset |
|---|---|---|
| `maxvel` | `unified_velocity_bound(track_cfg)`, override with `proptv.maxvel` in YAML to set independently | 1.1292mm |
| `angle` | `track.angle` (gon) × `GON_TO_DEG` (0.9), override with `proptv.angle` in YAML | **108°** — this dataset's `track.angle=120` gon × 0.9. The registry's stated "30° default" is a fallback used ONLY when `track.angle` is absent from the config entirely; it is NOT what runs here, since `track.angle` is always present in these datasets' YAML. Always compute `track.angle × 0.9`, don't quote the fallback number. |
| `t_init` | `proptv.t_init` in YAML, plugin default | 4 frames |
| `backtracking` | `proptv.backtracking` in YAML, plugin default | `false` |
| `gaptracking` | `proptv.gaptracking` in YAML, plugin default | `false` |

**Example**:
```python
overrides = bu.per_tracker_overrides(
    ["proptv_tracking"], src=..., first=10001, n_frames=30
)
tracks, elapsed = bu.run_single_tracker(
    "proptv_tracking",
    track_overrides=overrides["proptv_tracking"],
    src=...,
    first=10001,
)
```

**Strength**: `supports_backward=True` and `supports_gap_relinking=True`
(registry-declared) — the only pure-3D tracker in this set with both,
useful for sequences with genuine multi-frame occlusion gaps.

**Weakness, measured not assumed**: by far the slowest of the five —
**54.5-64.7s** on this 30-frame/500-particle case vs. 7.7-13.5s for
everyone else (an order of magnitude), *and* worst K_a under realistic
noise (73.31, 2-3× worse than the next-worst). The GMM fitting overhead
scales with track count (registry's own `avoid_when` note); on this
dataset it is also the least accurate under noise, making its case for
production use here specifically weak — worth re-testing on a genuinely
occlusion-heavy sequence where its gap-relinking/backward support might
earn back the cost.

## 5. Decision guide

- **Need speed, data is clean or near-clean, no occlusion gaps**:
  `priority_segment_3d`. Fastest, simplest, immune to any image-space/
  calibration-model choice by construction (it never looks at either).
- **Need the best achievable accuracy under real noise and can afford
  ~40% more time**: `trackcorr` — the only engine here that cross-checks
  image-space consistency, and it shows in K_a (§3). Remember it runs
  forward-only by default (§4.2) unless you explicitly request the
  three-pass preset.
- **Want a fast pure-3D tracker with `4be`'s future-support disambiguation
  and don't need the last word in accuracy**: `4be`, now that both bugs
  found this session are fixed — still not competitive with `trackcorr`/
  `myptv_3d_tracking`/`priority_segment_3d` under noise (§4.3), an open
  research question rather than a recommendation.
- **Need Python-level customizability of the cost function (add an
  intensity term, experiment with new weighting)**: `myptv_3d_tracking` —
  competitive accuracy, pure Python so it's the one to fork/extend.
- **Have genuine multi-frame occlusion and can afford an order of magnitude
  more runtime**: `proptv_tracking` is the only one with both backward
  tracking and gap-relinking declared — but re-verify its accuracy on your
  actual occlusion pattern; this benchmark's noise model doesn't have heavy
  occlusion gaps, so this recommendation is about its *declared*
  capabilities, not a measured advantage here.

## 6. What's still open, not swept under the rug

- **`4be`'s residual gap** (§4.3): still worst of the four
  non-`proptv_tracking` trackers after two real fixes. Untested hypothesis:
  its `GREEDY_CONFLICTS=0` give-up-on-conflict rule.
- **`proptv_tracking`'s smoothness-bias hypothesis**: not confirmed by this
  specific "mild" measurement (§4.5) — needs a noise level with more
  genuine multi-candidate ambiguity to test properly, and/or a direct
  identity-swap audit (the same technique that found `4be`'s bug) rather
  than only the aggregate K_a.
- **`myptv_3d_tracking`'s `max_gap`/`cost_weights`** are not threaded
  through the same `track_overrides` dict as everything else — a real,
  documented (not hidden) gap in the "everything through one shared
  surface" story of §2.
- **`eps0`** (the correspondence-stage epipolar tolerance, `criteria.eps0`
  in YAML) is not a *tracking* parameter at all — it's set before any
  tracker runs, during correspondence solving — but it directly determines
  how many ghost/ambiguous correspondences reach the trackers compared
  above. See `docs/holistic-3d-ptv-systems-research-program.md` and
  `adapt_proptv_dataset.py`'s `_derive_eps0_mm` for how it should scale
  with detection noise and particle density; test_cavity's own real value
  (0.05mm, §2) is a second reference point beyond this document's
  synthetic case.

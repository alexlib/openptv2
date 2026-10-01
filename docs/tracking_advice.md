# tracking_advice: measure your data, get the tracking parameters

`scripts/tracking_advice.py` looks at a run, measures the few numbers that decide how the
`two_phase` tracker should be set, and prints **each recommended value together with the
measurement that led to it**, plus YAML lines to paste. It takes 1–2 seconds, reads about
100 frames and writes nothing.

For what every parameter does and how the pieces depend on each other, see
[Tracking parameters](tracking_parameters_guide.md).

## Use it

```bash
# the path is YOUR run folder (it contains parameters_*.yaml, cal/ and res/run.zarr)
uv run python scripts/tracking_advice.py ~/data/experiment/wp2/test --fps 5000
```

| Option | Meaning |
|---|---|
| `--fps` | camera frame rate; used only for the smoothing-window advice |
| `--first`, `--last` | frames to read (default 1–200) |

The run must have been through the **sequence** phase (it needs the stored correspondences
and targets). If you give a wrong path, the tool says what is missing and lists run folders
it finds below the path you gave.

## What it measures

| Measurement | Meaning | Decides |
|---|---|---|
| Density | median number of other 3D points within 5 mm of a point | whether the automatic confirmation tolerance is safe; the ghost rule strength |
| Median step | typical distance a particle moves per frame (mm) | whether data are frame-skipped (brightness gate) |
| Median kink | typical change of the step from one frame to the next (mm); with a fast camera this is the position noise | the size of the automatic confirmation tolerance |
| Kink / step | the two above divided: high = noise dominates, low = real motion dominates | whether the automatic tolerance is safe |
| 3-camera share | share of points seen by only 3 cameras | ghost rule strength |
| Flagged share | share of points whose ghost probability is above 0.2 | ghost rule strength |

The kink and the step come from a short first tracking pass (no confirmation, no ghost
rules) over the first 60 frames; the ghost numbers need the camera models from the
experiment.

## The rules it applies

| Parameter | Rule |
|---|---|
| `q_seed`, `q_young` (ghost rules) | 3-camera share above 45% **or** flagged share above 14% → **accuracy mode** `0.15 / 6`. 3-camera share below 20% **and** flagged below 4% → light rule `0.3 / 3`. Otherwise the default `0.2 / 3`. |
| `confirm_tol` | density ≤ 7 neighbours **and** kink/step ≥ 0.5 → **leave unset**: the tracker uses 8 × the median kink (the output states the value). Otherwise → fixed **0.3** mm/frame, and the reason is printed (dense data, or motion dominates = frame skipping). |
| `blob_gate` | median step ≥ 0.2 mm/frame (large steps: frame skip or slow camera) → **0.5**. |
| `trajectories.reconnect_gap`, `smooth_filter_k` | reconnect: always 6. Smoothness filter: 6 when kink/step ≥ 0.5 (noise dominated), otherwise off. |
| `trajectories.smoothing_window` | about 4 ms of frames: `0.004 × fps`, rounded to an odd number, at least 5. |

The thresholds come from the benchmark (the plan in `docs/plans/2026-09-30-tracker-plan.md`).
They can be changed in the yaml (`confirm_auto_max_neighbours`, `confirm_auto_min_ratio`,
`confirm_auto_radius`).

## Demo: the advisor on real and synthetic data

Real data are two CompleteTest recordings and two lv_multi recordings. The synthetic
cases come from `scripts/synth_bench.py` with realistic position noise and blob brightness;
their true best settings are known from the benchmark, shown in the last column.
(`step` and `kink` in mm; `nb` = neighbours within 5 mm; 3cam = share of 3-camera points;
flag = share with ghost probability above 0.2.)

| Data | pts/frame | nb | step | kink | kink/step | 3cam | flag | `q_seed`/`q_young` | `confirm_tol` | `blob_gate` | time | What the benchmark found best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **real** CompleteTest wp1 | 1184 | 5.0 | 0.123 | 0.063 | 0.51 | 39% | 12.1% | 0.2 / 3 | automatic (≈ 0.50) | – | 1.5 s | (no truth) |
| **real** CompleteTest wp2 | 1170 | 5.0 | 0.124 | 0.064 | 0.52 | 39% | 11.9% | 0.2 / 3 | automatic (≈ 0.51) | – | 1.3 s | (no truth) |
| **real** lv_multi wp4 | 1016 | 4.8 | 0.083 | 0.060 | 0.73 | 50% | 22.3% | 0.15 / 6 | automatic (≈ 0.48) | – | 0.3 s | (no truth) |
| **real** lv_multi wp5 | 1306 | 6.3 | 0.083 | 0.066 | 0.80 | 52% | 23.4% | 0.15 / 6 | automatic (≈ 0.53) | – | 0.3 s | (no truth) |
| synthetic, noise 0.04 px | 1154 | 6.0 | 0.100 | 0.042 | 0.42 | 35% | 11.5% | 0.2 / 3 | 0.3 (kink/step < 0.5) | – | 1.0 s | fixed ≈ automatic ✓ |
| synthetic, noise 0.08 px (real level) | 1154 | 6.1 | 0.116 | 0.076 | 0.65 | 35% | 11.3% | 0.2 / 3 | automatic (≈ 0.61) | – | 1.0 s | automatic: −0.018 ✓ |
| synthetic, noise 0.12 px | 1151 | 6.1 | 0.133 | 0.107 | 0.81 | 36% | 11.0% | 0.2 / 3 | automatic (≈ 0.86) | – | 1.0 s | automatic: −0.030 ✓ |
| synthetic, noise 0.16 px | 1148 | 6.0 | 0.150 | 0.137 | 0.92 | 36% | 10.6% | 0.2 / 3 | automatic (≈ 1.10) | – | 1.0 s | automatic: −0.070 ✓ |
| synthetic, density ×2 | 1872 | 9.7 | 0.118 | 0.085 | 0.72 | 47% | 18.3% | 0.15 / 6 | 0.3 (dense) | – | 1.3 s | fixed 0.3 (automatic: +0.008 worse) ✓ |
| synthetic, density ×4 | 2578 | 10.7 | 0.118 | 0.088 | 0.74 | 63% | 38.5% | 0.15 / 6 | 0.3 (dense) | – | 1.4 s | fixed 0.3 (automatic: +0.063 worse); accuracy mode better ✓ |
| synthetic, clustered + frame skip 4 | 1463 | 8.3 | 0.358 | 0.123 | 0.34 | 46% | 18.6% | 0.15 / 6 | 0.3 (dense, motion) | 0.5 | 1.3 s | fixed 0.3 (automatic: 0.1689 → 0.4293); gate −0.015 ✓ |
| synthetic, frame skip ×4 | 1124 | 6.6 | 0.328 | 0.099 | 0.30 | 37% | 11.7% | 0.2 / 3 | 0.3 (motion) | 0.5 | 1.1 s | fixed 0.3 (automatic: +0.047 worse); gate −0.007 ✓ |
| synthetic, frame skip ×8 | 1118 | 7.0 | 0.533 | 0.134 | 0.25 | 37% | 11.8% | 0.2 / 3 | 0.3 (motion) | 0.5 | 1.2 s | fixed 0.3 (automatic: +0.023 worse); gate −0.007 ✓ |

How to read it:
- The **real** recordings look like the synthetic "noise 0.08 px" case (5–6 neighbours,
  step 0.08–0.12 mm, kink 0.06): the advisor leaves the confirmation tolerance automatic
  (≈ 0.5 mm/frame), which was the clearly better setting at that noise level.
- **Dense** and **frame-skipped** data get the fixed 0.3 *and the reason*. On those the
  automatic tolerance made the result worse in the benchmark (up to 0.1689 → 0.4293), so the
  advice is the safe one.
- The **lv_multi** data have many 3-camera points (50%) and a high flagged share (22–23%),
  so the advisor recommends the accuracy mode for the ghost rules.
- The advisor agrees with the benchmark's best choice in every synthetic case. The real
  recordings have no truth, so for them it can only report what is measured and what that
  implied in the synthetic cases.

## A full example of the output

```
MEASURED ON YOUR DATA
---------------------
  3D points per frame (median)                                   1170
  other points within 5 mm of a point (median) = density         5.0
  share of points seen by only 3 cameras                         39%
  share of points with ghost probability > 0.2                   11.9%
  share of points with ghost probability > 0.5                   4.5%
  median step per frame (mm)                                     0.124
  median kink = change of the step (mm), measures the noise      0.064
  kink / step (high = noise dominated)                           0.52

RECOMMENDATION
--------------
  q_seed = 0.2
      why: default: 39% 3-camera points, 12% flagged
  q_young = 3
      why: default
  confirm_tol = leave unset (automatic)
      why: sparse (5.0 neighbours within 5 mm, limit 7) and noise dominated (median kink
           0.064 = 0.52 x median step 0.124, limit 0.5): the tolerance is set from the
           data, about 0.51 mm/frame. A pinned `confirm_tol:` in the yaml would switch this off.
  trajectories.smoothing_window = 21
      why: about 4 ms of frames at 5000 fps

YAML (track: section)
---------------------
track:
  q_seed: 0.2
  q_young: 3
  # confirm_tol: leave unset (automatic)
trajectories:
  smoothing_window: 21

NOTE: your yaml already sets confirm_tol: 0.3
```

The last line matters: the CompleteTest yaml pins `confirm_tol: 0.3`, which switches the
automatic tolerance off. Remove that line to use it.

## Limits

- It recommends starting values from a benchmark with one flow field and one camera
  geometry. When a measurement is close to a limit (density 6–8, kink/step 0.4–0.6),
  confirm on your data with the checks in the parameter guide.
- It does not know your flow. It cannot detect, for example, strong real accelerations
  that raise the kink; the kink/step ratio is the safeguard.
- `blob_gate` and the ghost model need blob brightness in the stored targets; without it the
  ghost marks fall back to ray convergence only.

## Guided tuning with an AI assistant

The repository carries a Claude Code skill, `.claude/skills/openptv-tuning-advisor`,
that walks through the whole chain (image processing, correspondences, tracker choice,
tracking parameters, checks, smoothing) in pipeline order and uses this tool in the
tracking step. Ask, for example: "which tracker and parameters for this run folder?".

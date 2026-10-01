# Tracking parameters: what to measure, how to set them, what changes

This guide is for the `two_phase` tracker. Every value here is a line in the `track:`
section of your `parameters_*.yaml` (or `trajectories:` in `experiment.yaml` for the
last two). **Changing a value in the yaml never needs a new software release.** A release
is only needed when a *new option* is added to the code.

## 1. The quick way: let the program measure your data

(Full description and a demo table on real and synthetic data: [tracking_advice](tracking_advice.md).)

```bash
# replace the path by YOUR run folder; --fps is the camera frame rate
uv run python scripts/tracking_advice.py ~/data/experiment/wp2/test --fps 5000
```

The path is a run folder (it has `parameters_*.yaml`, `cal/` and `res/run.zarr` after
the sequence phase). The program reads about 100 frames, measures the numbers in section 3,
and prints for every parameter **the value it recommends and the measurement behind it**,
then YAML lines you can paste. It writes nothing.

## 2. How the pieces fit together (what depends on what)

```
position noise (jitter) ──► size of the "kink" ──► confirmation tolerance (confirm_tol)
point density           ──► chance of a wrong neighbour ──► how tight things must be
camera matching quality ──► ray miss distance, brightness agreement ──► ghost marks (q_seed, q_young)
step per frame          ──► (large steps = frame skip / low frame rate) ──► blob_gate, fixed tolerance
frame rate + noise      ──► how long a window you need ──► smoothing window
```

Three words used below:
- **Step**: how far a particle moves between two frames (mm).
- **Kink**: how much that step changes from one frame to the next, `|x(t+1) − 2 x(t) + x(t−1)|`.
  With a fast camera, real accelerations are tiny, so the kink is almost entirely
  **position noise**. Its median is a direct measurement of your noise.
- **Ghost**: a 3D point made from blobs of different particles. It is smooth and can form
  a long wrong trajectory; the tracker cannot tell by motion alone.

## 3. The numbers to measure (the advisor prints all of them)

| Measurement | How it is measured | Typical value | What it tells you |
|---|---|---|---|
| Density | median number of other 3D points within 5 mm of a point | 5–6 sparse, 10–11 dense | many neighbours = wrong links are likelier |
| Median step | median distance between linked points in consecutive frames | 0.1 mm at 5000 fps | large (≥ 0.2) = frame-skipped or slow camera |
| Median kink | median of the kink over all linked triples | 0.04–0.14 (grows with noise) | the noise level |
| Kink / step | the two above divided | 0.4–0.9 | high = noise dominates; low = real motion dominates |
| 3-camera share | share of points seen by only 3 cameras | 35–40% (lv_multi 50%) | more = more doubtful points |
| Flagged share | share of points with ghost probability > 0.2 | about 10% (lv_multi 18%) | how many points the ghost rules will treat as doubtful |

## 4. The parameters

| Parameter | What it does (one sentence) | How to choose | Higher value does | Lower value does |
|---|---|---|---|---|
| `q_seed` (default 0.2) | a point with ghost probability above this cannot start a trajectory | 0.2 normally; 0.15 if 3-camera share > 45% or flagged > 14%; 0.3 if < 20% / < 4% | keeps more points, more ghosts | removes more ghosts, also ~1% real points |
| `q_young` (default 3) | a trajectory with fewer points than this cannot continue onto a doubtful point | 3 normally, 6 in accuracy mode | removes more ghosts | keeps more points |
| `q_model` (default `rcm_blob`) | how the ghost probability is computed: `rcm` = ray miss distance + camera count; `rcm_blob` also uses blob brightness agreement | `rcm` if blobs have no usable brightness | – | – |
| `confirm_tol` (mm/frame) | a link survives only if the next step continues within this kink | **leave unset** for sparse, noisy data (automatic); `0.3` for dense or frame-skipped data | fewer cut links, longer tracks, more outliers kept | cuts true links when the noise is larger than the tolerance |
| `confirm_auto` (default 8) | with `confirm_tol` unset: tolerance = this × the median kink | 8 | looser | tighter |
| `confirm_auto_max_neighbours` (7), `confirm_auto_min_ratio` (0.5), `confirm_auto_radius` (5 mm) | the guard: the automatic tolerance is used only if the density is at most 7 neighbours and kink/step is at least 0.5 | change only if the advisor's measurements are near the limit and you have tested | – | – |
| `confirm_ends` (default true) | a track may not end by stepping onto a stranger | keep true: it is the most valuable part of the confirmation | – | – |
| `blob_gate` (default off) | a candidate whose blob brightness changed by more than this (log units) is rejected | 0.5 for frame-skipped or low-frame-rate data (step ≥ 0.2 mm) | looser | stricter, cuts ~0.3% of real links at 0.5 |
| `v_max` (mm/frame) | search radius | about 3× the typical step, or the largest real step | more candidates | loses fast particles |
| `max_gap` | frames a track may be missing | 2 (one missing frame) | bridges longer dropouts, more mistakes | – |
| `trajectories.smoothing_window` | frames in the smoothing window | about 4 ms of frames (21 at 5000 fps) | smoother velocities, blurs real changes | noisier velocities |
| `trajectories.reconnect_gap` (6; 0 = off) | join broken pieces of a trajectory across up to `gap − 1` missing frames, using a straight line from each side; tolerance `trajectories.reconnect_tol` (4 noise sigmas) | 6 for all data; better in 8 of 9 benchmark cases | joins more pieces, risk of a wrong join | fewer joins |
| `trajectories.smooth_filter_k` (6; off = 0) | drop points farther than `k` median residuals from the curve their neighbours define | 6 when noise dominates (kink/step ≥ 0.5); **off** for frame-skipped data | milder: keeps more points | stricter: 4 gives −0.02 velocity error and drops 4% of the points |
| `trajectories.trim_doubtful` (off) | cut up to 3 doubtful end points of a trajectory | 0.3 when accuracy matters more than the number of points | cuts fewer points | cuts more (0.2: ~2% of points) |

## 5. What to expect (measured on synthetic data with known truth)

Velocity error relative to the true flow, lower is better; "perfect linker" is the best
any tracker can do on the same 3D points.

| Change | Effect on the velocity error |
|---|---|
| Ghost rules on (`q_seed` 0.2, `q_young` 3) | −0.020 (sparse) … −0.075 (4× density) |
| Brightness agreement (`q_model: rcm_blob`, instead of `rcm`) | a further −0.005 … −0.039 |
| Trim doubtful ends (0.3) + gap-aware smoother | a further −0.01 |
| Automatic confirmation tolerance, real noise level | −0.018 (and more with more noise: −0.070 at 2× the real noise) |
| Reconnect broken pieces (gap 6, 4σ) | −0.002 … −0.019 (+0.004 at 4× density) |
| Smoothness filter (k = 6) after reconnect | a further −0.001 … −0.015, about 1% fewer points (skip ×8: +0.0075, so off there) |
| `blob_gate: 0.5` on frame-skipped data | −0.007 … −0.015 |
| Accuracy mode (`q_seed` 0.15, `q_young` 6) | a further −0.004 … −0.04, about 0.5–1% fewer points |

## 6. How to check a setting on your own real data (no ground truth needed)

```bash
uv run python scripts/real_retrack.py NAME 300 q_seed=0.15 q_young=6     # a scratch copy
uv run python scripts/realism_metrics.py /private/tmp/claude-501/real_runs/NAME/res/run.zarr --last 300
```

Compare two settings with these numbers:

| Number | Better when | Why |
|---|---|---|
| `trajectories` (tracks) | lower | fewer fragments and ghost pieces |
| `mean_len` | higher | longer tracks |
| `n_ge50` (tracks of ≥ 50 frames) | about equal | long tracks must not be lost |
| `never4_share` | lower | long trajectories never seen by all 4 cameras are the best hint of ghosts |
| `jitter_z_p90` | lower | scatter of the points around a smooth curve |

**Do not use `jump_share` to judge the confirmation tolerance.** It counts large depth
changes between frames; a looser tolerance keeps more of these noise outliers although the
smoothed result is better (it reproduces identically on synthetic data, where the truth
shows the looser setting wins). Judge a tolerance by the smoothed velocities, or on a
synthetic copy of your data with known truth (`scripts/synth_bench.py`, section 7).

## 7. Testing a setting on synthetic data with known truth (optional)

```bash
uv run python scripts/synth_bench.py build --level L1 --sigma-px 0.08 --amp-jitter 0.18 --amp-flicker 0.05
uv run python scripts/synth_bench.py sweep --cases CASE --trackers "two_phase+q_seed=0.15+q_young=6" -j 3 --no-eval
uv run python scripts/synth_bench.py eval  --case CASE --trackers two_phase "two_phase+q_seed=0.15+q_young=6"
```

`--sigma-px` sets the position noise (0.08 matches real CompleteTest data); the other two
options give realistic blob brightness. Never tune on real data; tune on the synthetic copy
and check on the real data with section 6.

## 8. Limits to keep in mind

- The numbers in sections 4–5 come from one flow field and one camera geometry. Where
  your measurements are near a limit (density 6–8, kink/step 0.4–0.6) test before trusting
  the automatic tolerance.
- The automatic tolerance assumes a fast camera (kink = noise). With frame skipping the
  kink contains real motion and the guard switches it off.
- Very short smoothing windows (about 9 frames) make the looser tolerance slightly worse
  (+0.008 at the real noise level).
- See [Point quality](tracking_quality.md) for how the ghost probability is computed.

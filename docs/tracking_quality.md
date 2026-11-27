# Point quality in tracking: ghost marks, thresholds, trimming, smoothing

> For a short, plain explanation of **all** the tracking parameters (what to measure, how to
> set them, what changes) see [Tracking parameters](tracking_parameters_guide.md); this page
> goes deeper on the ghost marks.

This page explains the quality marks the `two_phase` tracker uses by default, how to
change the two thresholds, **when** to change them, and **how to test** a choice on
your own data. Evidence and measurements are in
`docs/plans/2026-09-30-tracker-plan.md` (sections 1 and 4, A10) and `bench/`.

## 1. What problem this solves

About 5–10% of the 3D points that camera matching produces are *ghosts*: made from
blobs of different particles. A ghost often links into a long, smooth, wrong
trajectory. On the synthetic benchmark ghosts cause about 60% of the remaining
velocity error of `two_phase` and nearly all of its wrong links.

A ghost's camera rays usually meet worse than a real particle's rays. The distance by
which they miss each other is the **ray miss distance (rcm)**. It is the best ghost
signal we found (AUC 0.88 per trajectory). No single cut-off works, so the tracker uses
it softly, as a *ghost probability* between 0 and 1 for every point.

## 2. How the mark is computed

1. rcm of every stored 3D point (`openptv2.point_quality.frame_rcm`), from the targets
   and calibrations in the run store.
2. rcm grows away from the volume centre on real data (about 0.035 mm near the centre,
   0.058 mm at 50–60 mm), so it is divided by a straight line in the distance from the
   centre, fitted per run on the 4-camera points (`fit_scale`).
3. The relative rcm and the number of cameras give the probability
   (`ghost_probability`; a 3-camera point is far more likely to be a ghost than a
   4-camera point with the same relative rcm).
4. The probability is stored in the run store (`quality/frame_NNNNNN`,
   `RunStore.read_point_quality`) so later steps can use it without recomputing.

**Model `q_model`** (`track` section; default **`rcm_blob`**, alternative `rcm`):

- `rcm`: steps 1-3 above.
- `rcm_blob`: also uses the **brightness agreement of the blobs** across cameras. A real
  particle shows a similar brightness in all cameras that see it (spread of the log
  brightness about 0.15 on real data); a ghost combines blobs of different particles (about
  0.4). The brightness of each camera is first divided by that camera's typical value (its
  run median), because real cameras differ in gain by 0.13-0.17 in log, as much as the
  spread itself. It is a logistic model on [relative rcm, 3-camera flag, log spread], fitted
  on synthetic data whose per-camera brightness scatter matches real data
  (`synth_bench build --amp-jitter 0.18`). If the blobs carry no brightness (constant
  values) the plugin falls back to `rcm` with a message.
- Measured on four independent realistic synthetic cases (against `rcm`): velocity error
  −0.005 (sparse), −0.015 (×2 density), −0.039 (×4 density), −0.001 (clustered with frame
  skip ×4); ghost fraction 1 to 2 points lower; true points kept at most 0.007 lower. On
  real wp2: 4.6% fewer trajectories, mean length 20.2 → 20.8, suspect trajectories 10.4% →
  9.7%, jump share unchanged. The share of points confidently flagged (probability above
  0.5) is about 5% in both the synthetic test and wp2.

Nothing is deleted at this stage.

## 3. The two thresholds

Set them in the `track` section (YAML / GUI parameters) of the `two_phase` tracker.

| parameter | default | what it does | switch off |
|---|---|---|---|
| `q_seed` | 0.2 | A point whose ghost probability is above this **may not start a trajectory** (an existing trajectory can still pick it up). It is also the limit for `q_young`. | `q_seed: null` |
| `q_young` | 3 | A trajectory with **fewer than this many points** may not continue onto a point above `q_seed`. Established trajectories are not affected. | `q_young: 0` |
| `q_model` | `rcm_blob` | ghost model: `rcm_blob` (rcm + camera count + blob brightness agreement) or `rcm` | `q_model: rcm` if your blobs have no usable brightness or the cameras are saturated |
| `blob_gate` | none | **brightness continuity gate**: a candidate whose blob brightness (mean over the cameras that see both points) changed by more than this in log units is not a candidate. A real particle keeps its brightness in each camera from frame to frame (median change 0.06, p90 0.21 on CompleteTest wp2; a random neighbour 0.33). | off. Use **0.5** for frame-skipped or low-frame-rate data (velocity error −0.007 at skip ×4 and ×8, −0.015 clustered; at most 0.005 fewer points kept; it cuts 0.3% of real links on wp2); 0.3 gains more and loses more points; 0.8 does little. Neutral on normal data (−0.001 to −0.003), so not a default |
| `q_weight` | 0 | Multiplies the link cost by `1 + q_weight * probability`. Measured: **no effect** (links onto doubtful points are rarely contested). Leave at 0. | `0` |

To switch the whole feature off: `q_seed: null` and `q_young: 0`. It also switches off
by itself, with a message, when the experiment has no calibrations (the mark needs the
camera models).

The values 0.2 / 3 were chosen from 16 combinations on five synthetic cases (normal,
real-jitter, clustered with frame skip ×4, 4× density, frame skip ×8). They are the
strongest setting where **no case lost more than 0.007 of the true points kept**. Mean
change against switching it off: velocity error −0.017, true points kept −0.004, ghost
fraction −0.021. Checked on four real recordings (CompleteTest wp1 and wp2, lv_multi wp4
and wp5): 5–21% fewer trajectories, 3–10% longer mean length, fewer long trajectories
that no frame sees with all four cameras, jump share unchanged or lower.

## 4. When to change them

The choice is a trade-off: a **lower `q_seed`** and a **higher `q_young`** remove more
ghosts and lower the error, and also remove a few real points.

| mean change vs off (5 synthetic cases) | `q_young` 3 | 6 | 10 |
|---|---|---|---|
| `q_seed` 0.10 | err −0.026, pts −0.012 | −0.029, −0.019 | −0.032, −0.026 |
| `q_seed` 0.15 | −0.022, −0.007 | −0.025, −0.011 | −0.025, −0.015 |
| `q_seed` 0.20 (default) | **−0.017, −0.004** | −0.020, −0.007 | −0.021, −0.009 |
| `q_seed` 0.30 | −0.011, −0.002 | −0.013, −0.003 | −0.013, −0.004 |

| your data | change |
|---|---|
| Many 3-camera points (above ~45%), dense seeding, many short suspect trajectories; accuracy matters more than the number of points | **Accuracy mode:** `q_seed: 0.15`, `q_young: 6`. Clustered flow gains most from a larger `q_young` (up to 10). |
| Few ghosts: good calibration, mostly 4-camera points (3-camera share below ~20%), low density | Raise `q_seed` to 0.3, or switch it off. The gain is small and points are not worth losing. |
| Every real point counts (e.g. you compute statistics that need all samples) | `q_seed: 0.3`, `q_young: 3`. |
| Very short recordings (below ~30 frames) | The scale line needs at least 200 four-camera points; with fewer it falls back to one global value. Check the flagged share (section 5). |
| Poor or drifting calibration | rcm is large everywhere; the per-run scale compensates, but if the flagged share (section 5) is far above 20% the table is mismatched: raise `q_seed` or switch off. |
| Very fast flow / large frame skip | No change needed: tested up to skip ×8 (error −0.008). |

Always combine with the two-hop link confirmation (**on by default**). Its tolerance is
set from the data when the data are sparse and noise-dominated (8 × the median kink,
about 0.5–0.6 mm/frame at the real jitter level) and is the fixed 0.3 otherwise; see
`docs/two-phase-tracking.md`. On the benchmark the confirmation has two parts: the
dead-end rule (a track may not end by stepping onto a stranger) is worth 0.02–0.1 of
velocity error everywhere, while the kink test must follow the noise: a fixed 0.3 cuts
true links as soon as the jitter exceeds the real level (2× jitter: 0.3072 with 0.3,
0.2372 with the automatic value). On lv_multi wp5 the
jump steps (velocity change above 0.3 mm in depth) were 8.4% **without** it and 0.0%
**with** `confirm_tol: 0.3`, `confirm_ends: true`; the quality marks add to that, they
do not replace it.

## 5. How to test a threshold on your data

**Sanity check, no truth needed.** Count the flagged share from the stored marks:

```python
import numpy as np
from openptv2.storage import RunStore
s = RunStore.open("res/run.zarr", mode="r")
g = np.concatenate([s.read_point_quality(f) for f in range(first, last + 1)])
print((g > 0.2).mean(), (g > 0.3).mean(), (g > 0.5).mean())
```

Expected on well-behaved data: about 10% above 0.2, 4–6% above 0.3, 0.6–2% above 0.5
(synthetic 10.3 / 5.7 / 2.0%; CompleteTest wp1/wp2 9.9 / 4.4 / 0.6%; lv_multi 18 / 10 /
2.2% because half of its points have only three cameras). A flagged share far above
this means the data differ from what the table was built on.

**Answer-free scores on real data** (`scripts/realism_metrics.py`), before and after a
change of the thresholds. Keep a setting only if:

- `jump_share` does not rise (it should fall);
- `never4_share` (long trajectories never seen by all four cameras, the best real-data
  hint of ghost trajectories) falls;
- the number of trajectories falls and `mean_len` rises (fewer fragments);
- `jitter_z_p90` does not rise.

```bash
uv run python scripts/real_retrack.py base 300 q_seed=null q_young=0   # scratch copy
uv run python scripts/real_retrack.py new  300 q_seed=0.15 q_young=6
uv run python scripts/realism_metrics.py /private/tmp/claude-501/real_runs/new/res/run.zarr --last 300
```

**With known truth** (synthetic cases from `scripts/synth_bench.py`): sweep a grid and
read velocity error against true points kept, tuning on the real-jitter case and
checking on the others; never tune on real data.

```bash
uv run python scripts/synth_bench.py build --level L1 --sigma-px 0.08
uv run python scripts/synth_bench.py sweep --cases CASE --trackers "two_phase+q_seed=0.15+q_young=6" -j 4 --no-eval
uv run python scripts/synth_bench.py eval --case CASE --trackers two_phase "two_phase+q_seed=0.15+q_young=6"
uv run python scripts/tune_summary.py q_seed q_young
```

## 6. After tracking: trimming doubtful ends and the smoother

- **Trim.** `openptv2.quality_post.trim_doubtful_ends(trajid, frame, ghost, thr=0.3,
  max_trim=3, max_frac=0.25)` removes up to three end points of a trajectory whose ghost
  probability is above `thr` (never more than a quarter of the trajectory per end). On
  the synthetic cases it lowers the velocity error by another 0.014–0.04 for 0.010 fewer
  points kept at `thr` 0.3 (0.2: more error gain, 0.018 fewer points). It is **optional**
  and costs points; lower `thr` for more accuracy, raise it to keep more. Short
  trajectories (lv_multi: about 8 frames) lose 9–16% of their points at 0.3/0.2: use 0.5
  there or skip it.
- **Smoother.** `openptv2.quality_post.weighted_savgol` is a Savitzky–Golay fit at the
  true frame times (the flowtracks filter assumes consecutive frames), with all points of a
  short trajectory fitted when it is shorter than the window (dropping one point to get an odd
  window made the error 0.008 worse), and optional point weights. With unit weights it
  lowers the velocity error by 0.002–0.008 on every case. Quality weights `(1−g)^p` and
  spike removal gave nothing (<0.001), so use **unit weights**.
- openptv-cloud uses both in its `post` step; see its configuration reference
  (`trajectories.smoothing_method`, `trajectories.trim_doubtful`).

## 7. Limits

- The probability table was measured on synthetic data (about 8% ghosts); real data may
  have a different ghost rate. The real-data checks above are the guard.
- The synthetic ghosts are almost stationary (static false targets); real data have few
  stationary trajectories. Do not build a ghost rule on speed.
- It cannot remove ghosts that have good rcm; about 40% of the possible ghost gain is
  reached (all ghosts removed: velocity error 0.1465 vs 0.1845 with the marks, 0.1977
  without, on the real-jitter case).

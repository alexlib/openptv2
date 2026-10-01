# Tracker plan, in plain words

Branch: `feat/tracker-improvements`. Date: 2026-09-30, updated 2026-10-01. **Start with the STATUS AND HANDOFF section below.**
This file replaces all earlier tracking plans, including `new_tracking_plan.md` and
`two-phase-accuracy-speed-plan.md`; their ideas are merged here.


---

# STATUS AND HANDOFF (updated 2026-10-02 evening, read this first)

## Where we are
Branch `feat/tracker-improvements`. Goal unchanged: correct Lagrangian trajectories,
accuracy first, speed second, no big slowdown (section 2 has the rules).

### Key facts learned (details in sections 1 and 4)
1. **Ghost points cause about 60% of two_phase's remaining error** and nearly all wrong
   links. With ghosts removed from the input, link precision goes 0.9325 → 0.9989 and
   velocity error 0.1977 → 0.1465 (perfect linker 0.1412) on the real-jitter case.
2. **Search-gate and score tuning do not help two_phase** (A1 tried and reverted; a
   smaller round ball gains at most 0.007).
3. **The ray miss distance (rcm) of a 3D point is the best ghost signal** (AUC 0.88 per
   trajectory). Its scale on real data depends on the distance from the volume centre,
   so it is divided by a per-run straight-line fit before use.
4. **Soft use works, hard deletion does not:** link-cost weighting alone has no effect;
   forbidding doubtful points from starting trajectories (`q_seed`) and young
   trajectories from continuing onto them (`q_young`) gives −0.008…−0.013 velocity
   error on four cases and −0.038 at 4× density, with at most 0.007 fewer points kept.
5. **Synthetic data is too gentle:** real data jitters 1.4–2× more. Tune on the
   "real jitter" case `L1_d1_c0_k1_n200_s0.08` (matches real wp2), check on the rest.
6. **`max_gap` counts steps:** `max_gap=1` = no gap bridging, `0` = no links at all.

### Done and committed
| Commit | What |
|---|---|
| 7263d414 | Bug 3: batched camera projection (21–29% faster, identical results) |
| 8bb9917a, 6dd27c02 | Bug 2: separate seen-mask, `RunStore.read_seen` |
| 81470b35 | Wrong seen-mask now raises; Step 0: `scripts/realism_metrics.py`, `synth_bench build --sigma-px`, baselines in `bench/realism/` |
| 4127a459 | A1 tried and reverted; ghost bound measured (`scripts/ghost_bound.py`); plan order changed |
| 0c10c51a, 724684d2, f6f70cea | A10 findings: rcm feature, distributions, distance dependence |
| ca5988c7 | A10 in the tracker: `openptv2/point_quality.py`, options `q_weight`, `q_seed`, `q_young` (default off) |
| f14cd3c3 + next commit | `quality_post.py` (trim ends, drop spikes, weighted Savitzky–Golay), `scripts/quality_postprocess.py`, `real_trim_check.py`, `tune_summary.py`, weighted smoothing option in `synth_bench.evaluate_pred` |

### Results of the tuning and the after-tracking work (2026-10-01 evening, committed)
Numbers: `bench/step7_tuning_and_post_2026-10-01.json`. Cases: normal, real jitter,
clustered + skip ×4, 4× density, skip ×8.

**1. Tuning `q_seed` / `q_young`** (mean change over the five cases against plain
two_phase; velocity error, then true points kept):

| q_seed \ q_young | 3 | 6 | 10 | 15 |
|---|---|---|---|---|
| 0.10 | −0.0258 / −0.0123 | −0.0291 / −0.0191 | −0.0319 / −0.0260 | −0.0303 / −0.0319 |
| 0.15 | −0.0215 / −0.0072 | −0.0246 / −0.0108 | −0.0251 / −0.0145 | −0.0251 / −0.0176 |
| 0.20 | −0.0167 / −0.0044 | −0.0196 / −0.0068 | −0.0209 / −0.0086 | −0.0213 / −0.0106 |
| 0.30 | −0.0107 / −0.0020 | −0.0131 / −0.0031 | −0.0134 / −0.0039 | −0.0124 / −0.0046 |

- It is a clean trade-off: smaller `q_seed` and larger `q_young` give lower error and fewer
  points. Error gained per point lost is best at `q_seed` 0.2 and 0.15 with `q_young` 3.
- **Recommended default: `q_seed=0.2`, `q_young=3`.** It is the only setting among the
  strong ones where every case loses at most 0.007 points (the plan's rule).
- **Accuracy mode: `q_seed=0.15`, `q_young=6`** (mean −0.0246) loses 0.014 points at 4×
  density. `q_young` larger helps the clustered case most (0.1632 → 0.1401 at 0.1 / 10).
- Real wp2 (first 300 frames) and lv_multi wp5 agree: with `q_seed=0.2`, `q_young=3`
  trajectories 16375 → 15310 (wp2) and 3004 → 2399 (wp5), mean length 19.5 → 20.2 and
  8.0 → 8.9, long trajectories never seen by 4 cameras 11.6% → 10.4% and 9.6% → 7.8%.
  The jump share barely moves (wp2 1.46 → 1.45%, wp5 8.4 → 8.1%), so wp5's wrong links
  are mostly not ghosts (see next steps).

**2. After tracking** (`src/openptv2/quality_post.py`, `scripts/quality_postprocess.py`,
real-jitter case, base = `q_seed=0.2`, `q_young=3`; velocity error / points kept):

| Variant | Result | Verdict |
|---|---|---|
| flowtracks smoother (current benchmark) | 0.1845 / 0.702 | reference |
| `weighted_savgol` with unit weights | 0.1790 / 0.701 | **better smoother, see below** |
| quality weights `(1−g)^1…3` | 0.1781–0.1785 | no value (−0.001 vs unit weights) |
| drop spikes (g > 0.3 and +0.3 over neighbours) | 0.1786 | no value |
| trim end points with g > 0.5 | 0.1757 / 0.698 | small gain |
| **trim end points with g > 0.3** | **0.1647 / 0.692** | **good: −0.014** |
| trim end points with g > 0.2 | 0.1595 / 0.683 | best error, −0.018 points |

- **Trimming doubtful ends is what works.** Over the five cases, `q_seed=0.2`,
  `q_young=3`, trim 0.3 and the new smoother, against plain two_phase with the flowtracks
  smoother (error / points kept): normal 0.1780/0.707 → 0.1420/0.693; real jitter
  0.1977/0.706 → 0.1647/0.692; clustered 0.1632/0.398 → 0.1285/0.378; 4× density
  0.4592/0.258 → 0.3928/0.240; skip ×8 0.1218/0.669 → 0.1048/0.649. Trim 0.2 gains
  another 0.005–0.01 and loses about 0.01 more points.
- **Weights and spike dropping do not pay.** After `q_seed`/`q_young`, doubtful points
  inside trajectories are few, and weighting `(1−g)^p` leaves them with half their weight.
- **A better smoother, independent of the quality marks.** The flowtracks
  `savitzky_golay` assumes consecutive frames (it ignores gaps), uses a fixed window and
  discards trajectories shorter than the window. `weighted_savgol` fits at the real frame
  times and shrinks the window for short trajectories. With unit weights it lowers the
  error by 0.002–0.008 on every case (e.g. 0.1652 → 0.1610 normal, 0.1529 → 0.1449
  clustered). Use it for the final output (upstream to postptv).
- **Real data:** trimming at 0.3 removes 3.1% (wp2) and 8.8% (wp5) of the points and
  leaves the answer-free scores unchanged on wp2; on wp5 the jump share goes 8.1 → 7.8%
  (trim 0.3) and 7.5% (trim 0.2, 15% of the points). `scripts/real_trim_check.py`.
- **Cost of trimming is points kept** (−0.010 at 0.3, −0.018 at 0.2). Under the
  "accuracy ranks above density" rule this is accepted, but it is a decision: make it an
  option with the threshold visible, default 0.3.

### Repeat of the real-data check and what was put into the product (2026-10-01 night)
**Real data, `q_seed=0.2`, `q_young=3` against off** (re-tracked on scratch copies; the
CompleteTest runs use their own settings, the lv_multi runs were re-done with
`confirm_tol=0.3`, `confirm_ends=true`):

| Dataset | Trajectories | Mean length | Long trajectories never seen by 4 cameras | Jump share |
|---|---|---|---|---|
| CompleteTest wp1 (300 fr) | 17036 → 15874 | 19.0 → 19.7 | 11.3% → 9.9% | 1.47% → 1.45% |
| CompleteTest wp2 (300 fr) | 16375 → 15310 | 19.5 → 20.2 | 11.6% → 10.4% | 1.46% → 1.45% |
| lv_multi wp4 (20 fr, confirm on) | 2106 → 1721 | 7.9 → 8.6 | 11.8% → 9.1% | 0.0% → 0.0% |
| lv_multi wp5 (20 fr, confirm on) | 2763 → 2368 | 7.3 → 7.7 | 9.6% → 8.1% | 0.0% → 0.0% |

All four agree with the synthetic result, so the values are kept. **Finding: the lv_multi
jump steps (wp4 6.7%, wp5 8.4%) were not ghosts but the missing `confirm_tol`:** the
lv_multi `parameters_*.yaml` have no `confirm_tol`; with `confirm_tol=0.3`,
`confirm_ends=true` (as the CompleteTest yaml) both go to 0.0%. Every synthetic benchmark
case already uses it, so the synthetic gains are on top of confirmation. **Open decision:**
make `confirm_tol` a default of the two_phase plugin (it is the one proven win; the scale
should follow the data, the CompleteTest value is 0.3 for a 0.08 mm step).

**Flagged share** (stored marks, share of points with ghost probability above 0.2 / 0.3 /
0.5): synthetic 10.3 / 5.7 / 2.0%; CompleteTest 9.9 / 4.4 / 0.6%; lv_multi 18 / 10 / 2.2%
(half of its points have three cameras). Used in the documentation as a sanity check.

**Implemented (committed):**
- openptv2 `33a65c06`: `two_phase` plugin has `q_seed=0.2`, `q_young=3` as defaults (switch
  off with `q_seed: null`, `q_young: 0`; it also switches off by itself, with a message,
  without calibrations); the ghost probability is stored in the run store
  (`quality/frame_NNNNNN`, `RunStore.write_point_quality` / `read_point_quality`);
  `trim_doubtful_ends` never cuts more than a quarter of a trajectory per end;
  `weighted_savgol` fits tracks shorter than the window with all their points (an "odd window" variant tried on 2026-10-02 was worse and was reverted);
  **documentation `docs/tracking_quality.md`** (what the marks are, the thresholds, when
  to change them, how to test them, trim and smoother) linked from `mkdocs.yml`,
  `docs/trackers.md` and `docs/two-phase-tracking.md`.
- openptv-cloud, branch `feat/gap-aware-smoothing-trim` (commit `614f86d`, not pushed,
  `main` untouched): `trajectories.smoothing_method` (`gap_aware` default, or
  `flowtracks`), optional `trajectories.trim_doubtful` / `trim_max_points`, validation in
  `review.py`, docs in `docs/experiment-yaml.md`, 9 new tests. `gap_aware` needs an
  openptv2 that contains `quality_post` (not in the released 0.5.11); with the old one it
  falls back to flowtracks and prints a warning. **To make it effective: release openptv2
  and raise the `openptv2>=` pin in openptv-cloud's `pyproject.toml`.**

**Baseline label change:** plain `two_phase` now includes the quality rules. The old
behaviour is `two_phase+q_seed=null+q_young=0`. Old `eval.json` files under the label
`two_phase` are the old behaviour.

### Decisions taken and release (2026-10-01 night)
The user decided **yes** to all three: (1) release openptv2 with `quality_post`;
(2) two-hop confirmation on by default; (3) merge and push the openptv-cloud branch.
- **Confirmation default** (`02033eed`): the `two_phase` plugin uses `confirm_tol=0.3`,
  `confirm_ends=true` when the `track` section has no `confirm_tol` and `v_max <= 3`
  mm/frame (the regime it was tested in); an explicit number or `null` wins; a large
  `v_max` leaves it off with a printed note. lv_multi wp4/wp5 with no overrides: jump steps
  0.0% (were 6.7% and 8.4%). `docs/two-phase-tracking.md` has the parameter row.
- **Release status (final, 2026-10-01):** tag **`v0.5.12`** (commit `02033eed`, pushed by the
  user) is **on PyPI with all nine wheels** (macOS arm64, Linux x86_64, Windows; Python
  3.11–3.13), so `pip install openptv2==0.5.12` works. **The sdist (source distribution) was
  rejected: 143 MB against PyPI's 100 MB limit** (v0.5.11: 5 MB). Cause: the earlier switch to
  versions from git tags (`88a834d0`, setuptools-scm) puts *every tracked file* into the sdist
  (test data, docs, generated C). Fixed in `f262aaf7` (`MANIFEST.in` prunes them; local sdist
  0.9 MB); it takes effect with the next release (on `feat/tracker-improvements`, not yet on
  `main`). 0.5.12 has no sdist on PyPI.
- **openptv-cloud merged and pushed** (`main` = `f3146bf`): `openptv2>=0.5.12`, lock updated,
  `trajectories.smoothing_method: gap_aware` is active (the released `quality_post` is
  used), `trim_doubtful` available. The new two_phase defaults change the lv-multi fixture's
  physics on purpose, so `tests/e2e_expected.json` was regenerated and reviewed: wp5
  2736 → 1885 tracks and 24390 → 20356 rows (confirmation removes the wrong links), wp4
  unchanged. `test_end_to_end_reproducibility` failed twice (about 2 of 14 runs) while heavy
  jobs were running and passed 9 of 9 afterwards; treat it as load-sensitive.

### Better ghost mark: blob brightness agreement (2026-10-01 night, committed `94cfe9e9`)
**Default model is now `rcm_blob`** (`q_model`, alternative `rcm`); details in
`docs/tracking_quality.md`, numbers in `bench/step10_A10_blob_brightness_model_2026-10-01.json`.
- **Idea.** A real particle shows a similar brightness in all cameras; a ghost joins blobs of
  different particles. The spread of the log blob brightness across the seeing cameras is
  about 0.15 for real points and 0.4 for ghosts. Cameras differ in gain by 0.13–0.17 on
  real data, so each camera's run median is subtracted first.
- **The old synthetic cases cannot judge it:** their real particles have exactly the same
  brightness in every camera (spread 0, AUC 0.93 is an artefact). New option
  `synth_bench build --amp-jitter 0.18` gives each particle an independent per-camera
  brightness factor; its spread distribution then matches real data (median 0.160 against
  0.173). **Use the `_a0.18` cases (`L1_d1_c0_k1_n200_s0.08_a0.18`, `L1_d2_c0_k1_n150_…`,
  `L1_d4_c0_k1_n150_…`, `L1_d2_c0.3_k4_n150_…`) for anything involving brightness.**
- **Result** (point-level AUC 0.875 → 0.917; velocity error / true points kept / ghost fraction):

  | Case | Quality off | `rcm` | **`rcm_blob` (default 0.2/3)** | accuracy mode 0.15/6 |
  |---|---|---|---|---|
  | Real jitter, sparse | 0.2136 / 0.680 / 7.0% | 0.1982 / 0.675 / 4.8% | **0.1934 / 0.675 / 3.7%** | 0.1889 / 0.670 / 3.2% |
  | Density ×2 | 0.3658 / 0.510 / 6.9% | 0.3371 / 0.502 / 5.2% | **0.3225 / 0.497 / 4.2%** | 0.3015 / 0.486 / 3.6% |
  | Density ×4 | 0.5055 / 0.231 / 10.8% | 0.4709 / 0.225 / 8.5% | **0.4307 / 0.218 / 6.5%** | 0.3940 / 0.208 / 5.6% |
  | Clustered + skip ×4 | 0.1765 / 0.376 / 6.8% | 0.1683 / 0.372 / 5.7% | **0.1647 / 0.372 / 5.5%** | 0.1564 / 0.366 / 5.1% |

  Against `rcm`: −0.005, −0.015, −0.039, −0.004; true points kept down by at most 0.007.
  The logistic weights were fitted on even frames of the first case; the other three cases are
  independent. Real wp2 (300 frames): 4.6% fewer trajectories, mean length 20.2 → 20.8,
  suspect trajectories 10.4% → 9.7%, jump 1.45 → 1.43%. Real lv_multi wp4/wp5: 4–6% fewer
  trajectories, mean length +2.5%/+4%, jump share 0, but the never-seen-by-4-cameras share is
  0.3–0.5 points *higher* (the proxy is itself rcm-based, so it does not see the new signal).
- **Caveats:** the model assumes that brightness differences between cameras are small
  once the run median is removed (true for the two real sets); constant brightness falls
  back to `rcm`; the weights come from synthetic data, so real-data checks stay mandatory.

### Session of 2026-10-02: A4, max_gap, post steps, brightness continuity
- **A4 velocity filter: reverted** (`bench/step11_…`): an α–β filter (vel_beta 0.25–0.6) is
  worse by +0.001…+0.004 and gains 0.001 completeness. The velocity guess does not limit
  the tracker. **`max_gap` 3, 4, 6: no gain either** (+0.003…+0.008).
- **Ghosts are still two thirds of the gap.** With ghosts removed, the default tracker reaches
  0.1551 on the realistic case (perfect linker 0.1375, with ghosts 0.1934, quality off 0.2136).
  The marks have captured about 38% of the ghost cost.
- **A bug in my own smoother, found by re-measuring:** making a short track's window odd (drop
  one point) cost +0.008 velocity error (unit-weight smoother 0.1790 → 0.1869, worse than
  flowtracks' 0.1845). **The released 0.5.12 contains it**, so openptv-cloud's new
  `gap_aware` default is slightly worse than flowtracks on short-track-heavy output until
  0.5.13. Fixed on `main` (`5b27bd0c`, with the sdist fix). **Needs: tag `v0.5.13`
  (the user has to push the tag), then raise the cloud pin to `>=0.5.13`.** With the fix, on
  the four realistic cases: weighted smoother −0.004…−0.009 against flowtracks, trimming ends
  at 0.3 a further −0.004…−0.007 (−0.003…−0.008 points kept); dropping whole doubtful
  trajectories adds only ~0.001 (removed); weights and spike removal nothing.
- **Brightness continuity** (`bench/step13_…`). On real wp2 a linked pair changes its blob
  brightness per camera by a median of only 0.058 (random neighbour 0.33, AUC 0.84), so the
  cross-camera brightness differences are *persistent per particle* (my first generator
  redrew them every frame; fixed: `--amp-jitter 0.18 --amp-flicker 0.05`, cases `…_f0.05`).
  As a link-cost term it has **no effect** (costs rarely decide; removed). As a candidate gate
  (`blob_gate`) it gives −0.007 at frame skip ×4 and ×8 and −0.015 on the clustered skip ×4
  case for gate 0.5, at most 0.005 fewer points, neutral on real data (cuts 0.3% of real
  links on wp2): **kept as a documented opt-in, default off.**

- **Re-check of the quality rules on the persistent-brightness cases** (velocity error /
  points kept / ghost fraction; off → `rcm` → `rcm_blob` default): sparse 0.2008/0.687/7.2% →
  0.1915/0.682/4.8% → **0.1862/0.680/3.8%**; ×2 0.3622/0.517/7.3% → 0.3369/0.509/5.4% →
  **0.3208/0.503/4.4%**; ×4 0.5110/0.239/12.5% → 0.4741/0.233/9.9% → **0.4214/0.225/7.2%**;
  clustered skip ×4 0.1694/0.380/6.8% → **0.1638**/0.375/5.7% → 0.1689/0.375/5.4%. The
  conclusions hold, with one exception: on the clustered frame-skip case `rcm_blob` is 0.005
  *worse* than `rcm` (brightness may change more over skipped frames; `q_model: rcm` is the
  setting for such data).

- **A2, A12, A13 closed by measurement** (`bench/step14_A2_A12_A13_contest_rate_2026-10-02.json`).
  Only **2.7%** (sparse) and **5.6%** (4× density) of tracks face a contested decision, and the
  average track has about **one** candidate in its search ball (0.97 and 0.78). So no cost term
  (A2), no-link cost (A12) or look-ahead (A13) can change more than a few percent of the links;
  the existing `q_weight` (cost scaled by the ghost probability, which contains the camera count
  and brightness) gives results identical to the baseline. Tracks fail by having **no
  candidate** (22% at 4× density), not by competition.

- **S0 speed profile: done** (`bench/step15_…`). About half of a 200-frame run was zarr I/O:
  correspondences and targets were read 4–5 times per frame. One read per frame now feeds
  the leaves, the seen mask, the ray miss distance and the brightness: **9.75 s → 8.1 s
  (−17%), identical results** (velocity error 0.1862, yield 0.680). The remaining time is the
  matcher's Python loops (self time 1.8 s of 4.2 s per 200 frames) and `confirm_link_tuples`
  (1.3 s); both could be vectorized for another ~15%.
- **A11 closed after two probes.** (1) Truth recovery bound: 12.8% of all truth points are
  detected in ≥3 cameras but missing from the 3D cloud. (2) Re-matching with the blobs of
  all suspects (probability above 0.2) freed reproduces exactly the same 607 points (281 real,
  326 ghost) because nothing forbids the same combination; and **46% of the suspects are
  real**, so destroying points on this mark would hurt. A real A11 needs a matcher that can
  forbid a combination and a purer mark. A further cue for 3-camera points (is there a blob at
  the projection in the missing camera?) is weak (AUC 0.69). The ghost cues available
  saturate near AUC 0.92.

- **End-to-end on the full real CompleteTest data** (1072 frames, wp1 and wp2, released openptv2
  0.5.13 through the openptv-cloud tracking, post and analyze phases; `bench/step16_…`).
  Old behaviour (no ghost marks, flowtracks smoother) → new defaults → new defaults + trimming
  (wp1 / wp2):

  | | Old | New defaults | + trim 0.3 |
  |---|---|---|---|
  | Tracks | 20778 / 19753 | **17518 / 16656** | 16999 / 16196 |
  | Mean length (frames) | 57.4 / 58.4 | **65.6 / 66.7** | 66.7 / 67.7 |
  | Tracks of ≥50 frames | 6279 / 6099 | 6218 / 5994 | 6157 / 5941 |
  | Median velocity change between frames (mm/s) | 8.13 / 8.27 | **7.94 / 8.09** | 7.86 / 8.01 |
  | Acceleration kurtosis | 58.7 / 58.6 | 66.2 / 47.1 | 71.2 / 42.1 |

  So: 16% fewer (short, junk) tracks, 14% longer mean length, the long tracks kept, 2–3%
  smoother velocities; the acceleration kurtosis is mixed (up on wp1, down on wp2). The
  Eulerian averages barely move (velocity and kinetic energy about 1.4 points closer to the
  reference, shear stress and dissipation unchanged). *(An earlier version of this text said
  the offsets come from the analysis, not from tracking. That was wrong for the table I used:
  see the next entry.)*
- **Why the Eulerian numbers differ so much from "legacy"** (`bench/step17_…`). There are two
  different references in `comparison_vs_legacy.json`; I had used the wrong one for the
  statement above.
  1. **"Legacy" table = legacy 3dptv tracks through the same analysis stage.** Same code, grid
     and thresholds, so the difference is entirely the **tracks**. Legacy tracks are much
     noisier and shorter: 30358 tracks of mean length 34.5 against 20814 of 57.3 (wp1), median
     speed 93 against 64, p99 speed 751 against 345, maximum phase-averaged velocity 2× higher.
     Squared-gradient quantities inflate with velocity noise, so ours are lower: velocity
     −4…−16%, shear stress −20…−33%, **dissipation −35…−67%**. Even our *unsmoothed* tracks
     give dissipation 0.63× the legacy value (smoothed 0.55×), so legacy velocities are noisier
     than raw finite differences of our positions.
  2. **The real MATLAB reference** (wp1–wp4, about 3× more particles per bin: 480–579k against
     168–192k) is **much closer to ours**: at phase 5 velocity +0%, kinetic energy −3%, shear
     stress +6%, dissipation +18%. It drifts later in the cycle (shear stress +28%,
     dissipation +69% at phase 8) where the flow is slow.
  3. **What explains the remaining gap, tested here:**
     - *Smoothing window:* 21 → 41 → 61 changes the worst-phase dissipation only from +69% to
       +65% (and +43% → +39%): **not temporal noise in the tracks**.
     - *Number of particles:* analysing wp1 alone (half the data) raises dissipation by 13–18%
       in phases 5–6 (388 → 438, 383 → 453), so the 2× larger reference sample accounts for a
       good part of the early-phase excess. (Later phases change sign because the set of
       voxels that pass the threshold changes.)
     - *Voxel threshold:* `min_count` 5 → 20 lowers the global dissipation by about 30%
       (388 → 274 at phase 5): the global mean of a squared-gradient quantity depends strongly
       on which sparse, edge voxels are included. The reference used the production thresholds
       (50 / 100) on 3× more data. A like-for-like comparison needs wp3 and wp4, which are not
       on this Mac (they were on the external drive).
     - *Phase alignment* of two workpackages (shift stage) may add smearing in the slow phases;
       not tested.
  4. **Jet-region columns are not comparable:** our "jet" mask selects almost stationary voxels
     (mean velocity 0.047 against 0.75), a mask-definition difference in the analysis stage
     (`mask: frozen: null` computes it adaptively; the legacy run used its own mask), not a
     tracking result.
  5. **A cloud-pipeline observation:** `openptv-cloud run` with `phases: [post, analyze]` on
     existing results printed "phases to execute: post, analyze" but did not rerun the post
     step (the run log was unchanged); calling `openptv_cloud.post.convert` directly worked.
     To be checked in openptv-cloud.

- **Why true links are still missing, and whether a track-level rule can recover them**
  (`bench/step18_…`, realistic sparse case). Of 198637 consecutive true pairs whose points
  both exist in the input, the default tracker links 95.5% raw and 97.8% after repair. **With
  the two added rules switched off it links 100.0%**: the matcher itself loses nothing. The
  2.2% that stay broken are caused by the quality rules (1.1%) and by confirmation (0.9%);
  71% of them are pairs where both points are isolated singletons (real particles judged
  doubtful). I tested deciding per *track* instead (relaxed tracker rules plus dropping whole
  trajectories whose mean mark is high): it is worse everywhere (sparse 0.1752–0.1846 against
  0.1724 for defaults + trim; dense 0.4519–0.4711 against 0.4177), because a ghost track that
  formed has already taken part in the assignments. Dropping whole doubtful tracks on top of the
  defaults gains only 0.002 (sparse) and 0.0006 (dense): not kept. **Conclusion: the point-level
  rules in the tracker are the right place; what remains broken is the price of removing
  ghosts.**

- **Synthetic data with more jitter, and the confirmation tolerance** (`bench/step19_…`; cases
  with persistent per-camera brightness at centroid noise 0.04 / 0.08 (real level) / 0.12 / 0.16 px).
  - **All errors grow with jitter, the tracker's more than the perfect linker's:** with the old
    default the gap to the perfect linker is 0.033 / 0.035 / 0.054 / 0.115 at best window.
    The gains of the ghost rules shrink with jitter (default against quality off: −0.008,
    −0.015, −0.012, −0.004).
  - **The cause is the fixed confirmation tolerance 0.3.** Separating the two parts of the
    confirmation: the **dead-end rule** is the useful part (it alone gives 0.1567 / 0.1694 /
    0.1949 / 0.2372, against 0.1708 / 0.1862 / 0.2249 / 0.3072 for the old default); the kink
    test at 0.3 cuts true links once the kink noise exceeds it, and no confirmation at all is
    worse than dead-ends-only (0.1761 / 0.1973 / 0.2275 / 0.2495).
  - **The median kink of the raw tracks measures the noise** (0.043 / 0.076 / 0.109 / 0.140 mm
    for the four levels; real wp2 0.061, lv_multi 0.056–0.062), so a tolerance of 8 × median
    kink follows the jitter. Unguarded it is **harmful** where the kink is not noise
    (clustered skip 4: 0.1689 → 0.4293; density ×4: 0.4214 → 0.4846; skip ×4/×8 worse), so
    it is guarded by two criteria that separate every benchmark case: at most 7 neighbours
    within 5 mm (sparse; density ×2/×4 have 9–11) and median kink ≥ 0.5 × median step (noise
    dominated; frame skipping gives 0.31–0.44). The real recordings pass (5–6 neighbours,
    ratio 0.52–0.78).
  - **Result of the guarded rule** (new default, `confirm_auto=8`; best window / production
    window 21): sigma 0.04 unchanged; real level 0.1862 → 0.1683 / 0.2142 → 0.2053; 0.12
    0.2249 → 0.1949 / 0.2515 → 0.2333; 0.16 0.3072 → 0.2372 / 0.3307 → 0.2771; density ×2/×4,
    clustered, skip ×4/×8 identical (guard falls back). More true points are kept
    (+0.004…+0.017). On a very short window (9 frames) the looser tolerance is slightly worse
    (+0.008 at the real level), which is not a production setting at 5000 fps.
  - **Real wp2** (300 frames, project yaml pins `confirm_tol: 0.3`, so use `confirm_auto: 8`
    to enable it): at window 21, 3.3% more points in longer tracks (7531 against 8858 tracks),
    median roughness unchanged, p95 −7%. **The raw "jump share" of the tracks is not a quality
    measure:** it reproduces on the synthetic data (1.4% with 0.3; 4.8% dead-ends-only; 5.2%
    none; real 1.4% / 5.3% / 5.8%) and counts depth-noise outliers that looser tracks keep.
    I had used it as an accuracy proxy earlier (e.g. lv_multi 8.4% → 0%); the truth-based
    evidence above is the better guide.
  - **Real wp2 across the ghost-rule variants** (300 frames; tracks, mean length, ≥50 frames,
    suspect-trajectory share): quality off 16375 / 19.5 / 1562 / 11.6%; default 14674 / 20.7 /
    1533 / 9.6%; accuracy mode (0.15/6) 14345 / 20.7 / 1515 / 8.9%; with `blob_gate` 0.5 14335 /
    20.6 / 1489 / 8.8%: monotone, with ≤3% fewer long tracks.
  - **Needs a release (0.5.14) and a cloud e2e reference update** (lv_multi has no explicit
    `confirm_tol`, so the automatic tolerance becomes active there).

## What can be next (ranked by expected gain)
The remaining error on the real-jitter case, step by step (velocity error; perfect linker
0.1412): plain two_phase 0.1977 → with quality rules 0.1845 → with better smoother 0.1790 →
with trimming at 0.3 0.1647. With all ghosts removed from the input two_phase reaches 0.1465.
So about 0.02 of the gap to the perfect linker is **fragmentation and wrong links that are
not ghosts** (link precision 0.9989, recall 0.974, 2.2 pieces per long track against 1.8,
long-track completeness 0.526 against 0.569), and about 0.02 is **ghosts that still pass**.

| Candidate | What it attacks | Expected gain | Cost / risk |
|---|---|---|---|
| **A7b** confirmation tolerance from the data | wrong links; best tolerance is loose for sparse and tight for dense data | −0.005 sparse, −0.05 at 4× density | medium: needs a rule (density, kink distribution) that picks 7σ / 3σ correctly |
| **A12 + A13** "no link" option, wait one frame | fragmentation and links forced onto doubtful points | probably −0.01 | medium; A13 only if the decision log shows lost second chances |
| **A4 + A5** velocity from several frames, gap bridging | fragmentation (longer tracks), better guesses | −0.005…−0.01; long-track completeness up | medium; must pass the frame-skip and real-jitter checks |
| **A11** feed marks back into camera matching | ghosts that still pass; also recovers real particles | up to ~0.02 (ghost bound) | high: touches correspondences |
| **Improve the mark itself** (a better ghost probability: add the 2D blob size/brightness agreement, the jitter of the trajectory, a table fitted per run) | ghosts that still pass | up to ~0.02 | low-medium; measurable with `ghost_bound`-style cheats |
| **S0 speed profile** | the quality rules add ~14% (rcm per frame, mark storage, young filter) | speed only | low; do before the next accuracy item that costs time |
| **Upstream the smoother** to postptv/flowtracks | all users of the final output | −0.002…−0.008 | needs a flowtracks release |

Recommended next: **A7b** (largest measured, concrete), then **the better mark** (cheap to
test offline with the saved features and the ghost cheats), then A4/A5 together.

## Next steps, in this order
0. **Cut `v0.5.13` now** (the user pushes the tag; `main` has the smoother fix `5b27bd0c` and the sdist fix), then raise the openptv-cloud pin to `>=0.5.13` and fix its docstrings that still say "largest odd window" (`experiment.py`, `post.py`, `review.py`).
1. ~~Finish the release chain~~ **Done** (wheels on PyPI, cloud merged). Open: put the
   sdist fix (`f262aaf7`) onto `main` and cut `v0.5.13` when there is something to release;
   consider a cloud release tag (0.9.18) for the changed defaults.
2. ~~**A7b:** data-driven `confirm_tol`.~~ **Done: measured and reverted, see section 4 A7.**
   (Original text:) data-driven `confirm_tol`. Per-case optima are in
   `bench/step8_A7_confirmation_variants_REVERTED_2026-10-01.json` (normal and real jitter
   sigma k=7, clustered k≈4, 4× density k=3, skip ×8 the absolute 0.3). Look for a statistic
   computed from the run that predicts them (nearest-neighbour spacing against the step,
   or the share of links the rule would cut), implement it as the default for the
   plugin, and verify on all five cases and the four real recordings.
3. ~~**Better ghost mark.**~~ **Done for brightness** (`rcm_blob`, see above). Further ideas
   for the mark: blob size agreement (synthetic sizes are degenerate, needs a generator
   change first), the 2D reprojection residual per camera, fitting the probability table per
   run on real data (EM with the rcm model as the first guess).
4. **A4/A5**, then **A12/A13**, then **A2**; order in section 7.
5. Later: A11, speed (S0/S1), chunks (P3).

## Commands (all from the repo root, always `uv run`)
```bash
# unit tests / lint for the touched code
uv run pytest tests/unit/test_two_phase_tracking.py tests/unit/test_point_quality.py \
  tests/unit/test_quality_post.py tests/unit/test_tracking_postprocess.py -q
uv run ruff check src scripts

# synthetic benchmark (work dir ~/Downloads/HiDImaging/tracker-bench-2026-09-29)
uv run python scripts/synth_bench.py build --level L1 --sigma-px 0.08      # real-jitter case
uv run python scripts/synth_bench.py track --case L1_d1_c0_k1_n200_s0.08 --tracker "two_phase+q_seed=0.2+q_young=3"
uv run python scripts/synth_bench.py eval  --case L1_d1_c0_k1_n200_s0.08 --trackers oracle two_phase "two_phase+q_seed=0.2+q_young=3"
uv run python scripts/synth_bench.py sweep --cases A B --trackers T1 T2 -j 4 --no-eval   # then eval
# cases: L1_d1_c0_k1_n200 (normal), L1_d1_c0_k1_n200_s0.08 (real jitter), L1_d2_c0.3_k4_n150,
#        L1_d4_c0_k1_n150, L1_d1_c0_k8_n150

# answer-free scores on real data, ghost bound, ghost features
uv run python scripts/realism_metrics.py ~/Downloads/CompleteTest-e2e-local/wp2/test/res/run.zarr --last 300
uv run python scripts/real_retrack.py LABEL 300 q_seed=0.2 q_young=3     # scratch copy, never the source
uv run python scripts/ghost_bound.py L1_d1_c0_k1_n200_s0.08
uv run python scripts/a10_features.py L1_d1_c0_k1_n200_s0.08
# after tracking: trim / weights on a tracker run, then score it
uv run python scripts/quality_postprocess.py CASE "two_phase+q_seed=0.2+q_young=3" unit trim0.3_unit trim0.2_unit
uv run python scripts/synth_bench.py eval --case CASE --trackers "two_phase+q_seed=0.2+q_young=3~trim0.3_unit"
REAL_RUN=~/Documents/Github/openptv-cloud/examples/lv-multi/wp5 uv run python scripts/real_retrack.py wp5_q 20 q_seed=0.2 q_young=3
uv run python scripts/real_trim_check.py /private/tmp/claude-501/real_runs/wp5_q 20 0.5 0.3 0.2
uv run python scripts/tune_summary.py q_seed q_young
```

## Pitfalls
- **Another agent may be committing to this branch at the same time** (it did, commit
  6dd27c02). Run `git log --oneline -5` and `git status` before starting. Use
  `git add` with explicit paths, not `git add -A`, unless the tree is only yours.
- **Shell:** zsh. Pass lists to `synth_bench` as bash/zsh arrays (`"${V[@]}"`), not as
  one string. BSD `sed -i ''`. No `timeout` command on this Mac.
- **Timing** is only meaningful with `-j 1`; sweeps with `-j 4` are for accuracy only.
  `eval` is slow (about a minute per tracker); run one process per case in parallel.
- **Never overwrite real data.** The real runs are in `~/Downloads/CompleteTest-e2e-local`
  and `~/Documents/Github/openptv-cloud/examples/lv-multi` (do not run
  `generate_synthetic_lv_multi.py` there). `wp1_10_images` is on the Windows machine only.
- One unit test, `test_run_store.py::test_traj_index_matches_legacy_reader_after_singleton_filter`,
  fails on a missing `flowtracks` module; it is not related to this work.
- The synthetic ghosts are almost stationary (static false targets); real data has far
  fewer. Never tune a ghost rule on speed.
- **openptv-cloud is a separate repo** (`~/Documents/Github/openptv-cloud`, python 3.12 venv
  with the released openptv2). Work on a branch, never on `main`. Its tests import
  `quality_post` from the openptv2 checkout (see `tests/test_trajectory_quality.py`).
  `tests/test_e2e_reproducibility.py` failed once in a full run and passed on rerun (flaky).
- zsh has no `read -a`; pass override lists to `scripts/real_retrack.py` as separate
  arguments (`key=value key=value`), never as one string.
- Scratch measurement scripts that are NOT in `scripts/` (they live in a session scratch
  folder): `rcm_geom.py`, `rcm_norm.py`, `plot_rcm.py`, `ghost_feats.py`, `a10_lite.py`.
  The results they produced are written into section 4 and `bench/`; re-create them only
  if needed.

---

## Goal
We want correct Lagrangian trajectories:
- each trajectory follows one real particle;
- no fake points;
- trajectories are long.

The tracker should also be reasonably fast. The order of importance is:
1. correct;
2. fast;
3. runs in parallel.

**Accuracy first, but no big speed penalty.** Slow, careful trackers already
exist: the proPTV-style tracker is 12× slower than two_phase. two_phase's place
is "accurate **and** fast", so it must stay fast. Section 2 sets the speed
budget.

---

## 1. What we have today

### The input
Each frame gives about 1500 3D points:
- about 92% are real particles;
- about 8% are fake points ("ghosts"). The camera-matching step makes a fake
  point when it combines blobs that belong to different particles.

### The core difficulty
Each measured position is shaky:
- about 0.09 mm along the camera viewing direction (depth);
- about 0.02 mm sideways.

A typical particle moves only about 0.085 mm per frame. So the position error is
as big as the real motion in one frame.

Most of this error is a slow offset, not frame-to-frame jitter (see "How
realistic is the synthetic data?" below). The real frame-to-frame jitter is
smaller: about 0.03 mm in depth and 0.006 mm sideways.

### What the two_phase tracker does, step by step
File: `src/openptv2/plugins/two_phase_tracking.py`. For each frame:

1. **Guess.** For every live trajectory, guess the next position from its last
   two positions ("keep going the same way"). A new trajectory has no velocity
   yet, so its guess is "stays where it is".
2. **Search.** Look for candidates inside a ball around the guess. The ball size,
   `v_max`, is set for the fastest particle, about ±2 mm.
3. **Score.** Give each candidate a score: its distance in pixels from the guess,
   measured in the cameras. With `leaf_weight=0`, plain 3D distance is used
   instead.
4. **Group.** Split the frame into groups of trajectories that compete for the
   same points.
5. **Decide who gets whom.** No two trajectories may take the same point:
   - a group with only one trajectory and one point is accepted directly;
   - a small group is solved exactly (the "Hungarian" solver);
   - a group with more than 128 members falls back to a quick approximation:
     take the cheapest pair first, then the next cheapest, and so on.
6. **Update.** Update each trajectory's position and velocity.
   - A trajectory unseen for more than `max_gap` frames is ended.
   - A point that no trajectory took starts a new trajectory.
   - A trajectory that loses a contested point just stops. It never gets a
     second choice.
7. **Confirm (optional).** Check that the next step also continues smoothly, and
   cut the link if it doesn't (`confirm_tol`).
8. **Backward run (optional).** With `bidirectional=True`, a second run goes
   backward through the frames. It may only fill in trajectory ends that are
   still free, closest first.

### What we already measured
- The 2D score, the velocity guess and the backward run each change the result
  by less than 0.01.
- The smoothness check in step 7 is the only part that clearly helps.

### What the fake points look like (measured 2026-10-01)
Measured on two_phase's output, normal case `L1_d1_c0_k1_n200`:
- **Most fake points form long fake trajectories, not one-frame blips.**
  - 90% of fake points sit in trajectories made only of fake points.
  - 57% are in fake runs longer than 30 frames.
  - Only 5% are in runs of 1–2 frames.
- **Long fake trajectories are smoother than real ones.** The frame-to-frame
  change in depth is about 0.013 mm for fake trajectories and 0.037 mm for real
  ones. So "drop what is not smooth" cannot remove them.
- **They are real ghosts, not slightly misplaced real particles.** They sit
  1–4 mm from the nearest real particle.
- **They are mostly seen by only 3 cameras.**
  - 73% of fake points are 3-camera points.
  - 17.5% of all 3-camera points are fake, against 3% of 4-camera points.
- **Over a whole trajectory the difference is sharper.** 65% of long fake
  trajectories are never seen by all 4 cameras in any frame, against 6% of long
  real trajectories.
- **At 4× density the picture changes.** Fake runs are shorter there (mostly
  3–5 frames).

### How realistic is the synthetic data? (measured 2026-10-01)
Measured on the normal case, on two_phase trajectories that follow one true
particle all along (`L1_d1_c0_k1_n200` and the image-based `L2_d1_c0_k1_n200`
agree within 5%):

| Quantity (mm) | Sideways (x, y) | Depth (z) |
|---|---|---|
| True change of velocity per frame (the flow itself) | 0.001 | 0.0004 |
| Measured change of velocity per frame (flow + noise) | 0.015 | 0.080 |
| Position error vs truth | 0.019 | 0.090 |
| Frame-to-frame jitter (from the line above) | 0.006 | 0.033 |
| Slow offset (the rest of the position error) | 0.018 | 0.083 |

In plain words:
- **The synthetic flow has almost no acceleration.** Its true change of velocity
  per frame is about 80× smaller than the noise. Real turbulence accelerates
  much more. So on this data, "keep going the same way" is almost always right,
  and every smoothness rule looks better than it will on real data.
- **Most of the position error is a slow offset.** The offset comes from the
  mismatch between the calibration used to make the images and the one used to
  process them. It follows a particle slowly, so it barely affects velocity.
  Only about a third of the error is frame-to-frame jitter. Real data may have
  more jitter: worse blob centres, real optics, vibration.
- **Consequence.** A rule tuned on the synthetic data alone may cut real,
  accelerating particles on real data. Examples: a tight smoothness tolerance
  (A7, A9), a strong velocity filter (A4), a small ball (A1). Every such rule
  must pass the realism checks in section 2.

### Real data vs synthetic data (measured 2026-10-01)
Real data has no known answer, so I compared what can be measured without it,
on the stored two_phase trajectories.

**1. Frame-to-frame numbers** (trajectories of at least 8 frames):
- the typical step per frame;
- the typical change of velocity per frame (median |2nd difference|);
- how often that change is a jump bigger than 0.3 mm in depth.

| Data | Step per frame (mm) | Typical change of velocity x, y, z (mm) | Jumps > 0.3 mm in depth |
|---|---|---|---|
| Synthetic normal case (L1) | 0.085 | 0.007, 0.006, 0.036 | 0.6% |
| Real CompleteTest wp1 (1072 frames) | 0.079 | 0.009, 0.008, 0.045 | 1.2% |
| Real CompleteTest wp2 (1072 frames) | 0.078 | 0.009, 0.008, 0.044 | 1.2% |
| Real lv_multi wp4 (20 frames) | 0.078 | 0.007, 0.007, 0.031 | 0.0% |
| Real lv_multi wp5 (20 frames) | 0.077 | 0.011, 0.010, 0.055 | **9.6%** |

**2. Position jitter.** How far points scatter around a smooth curve fitted over
11 frames. This removes real acceleration, so it measures the jitter itself.

| Data | Jitter sideways (mm) | Jitter depth (mm) | Worst 10% of windows, depth (mm) |
|---|---|---|---|
| Synthetic normal case (L1) | 0.005 | 0.027 | 0.062 |
| Real CompleteTest wp2 (first 300 frames) | 0.008 (**1.7×**) | 0.039 (**1.4×**) | 0.079 |
| Real lv_multi wp5 | 0.010 (**2×**) | 0.049 (**1.8×**) | 0.178 (**3×**) |

**3. Fake-point signal.**
- 3-camera points: 39% in real wp1/wp2, against 33% in synthetic.
- Long trajectories (11 frames or more) never seen by 4 cameras: 14% in real
  wp1/wp2, against about 9% in synthetic.

  In synthetic data, 65% of fake long trajectories are like that, against 6% of
  real ones. So the real data probably holds more fake trajectories.

In plain words:
- **Real data jitters more than the synthetic data.**
  - 1.4–2× more jitter in each direction.
  - In the worst recording (wp5), the worst tenth of trajectory pieces jitter
    3× more.

  The synthetic data is too gentle in two ways: the flow has almost no
  acceleration, and the points jitter less.
- **wp5 also has many wrong links.** About 1 in 10 steps jumps by more than
  0.3 mm, while particles move only 0.077 mm per frame. A real particle can't
  change its velocity by 4× its own speed in one frame. wp5 has more points per
  frame (about 1300 vs 1000 in wp4), with the same settings.
- **The jitter shows up in the results.** In `CompleteTest-e2e-local/outputs`,
  shear stress (VSS) and dissipation (eps) come out above the legacy MATLAB
  curve. Both come from velocity gradients, and jitter inflates gradients.
- **What the tracker can and can't do about jitter.**
  - The jitter is already in the 3D points; linking can't remove it.
  - The tracker's job is to deliver long, correct trajectories, so that the
    smoothing afterwards (Savitzky-Golay) can average the jitter out. Wrong
    links and broken trajectories are what defeat the smoothing.
  - Reducing the jitter itself belongs to blob centring, calibration and camera
    matching. Check below whether 3-camera points jitter more than 4-camera
    points.

### Score on the normal test case (velocity error, lower is better)
| Tracker | Velocity error | Fake points kept |
|---|---|---|
| Perfect linker (a cheat that knows the answer) | 0.130 mm/frame | 0.5% |
| two_phase (our best) | 0.178 mm/frame | 7% |

### How much error is due to ghosts? (measured 2026-10-01)
On the real-jitter case (`L1_d1_c0_k1_n200_s0.08`) I removed all ghost points
from the tracker's input, using the truth labels (`scripts/ghost_bound.py`),
and tracked again:

| | With ghosts | Without ghosts |
|---|---|---|
| Correct links among links made (two_phase) | 0.9325 | **0.9989** |
| Velocity error (two_phase) | 0.1977 | **0.1465** |
| Velocity error (perfect linker) | 0.1412 | 0.1236 |
| Gap between two_phase and perfect linker | 0.0565 | **0.0229** |

In plain words: **ghosts cause about 60% of two_phase's remaining error and
nearly all of its wrong links.** Without ghosts, two_phase is almost as good as
the perfect linker. So removing ghosts is the main lever. Search-ball tuning
and score tuning are not (see A1 below).

### Three problems I found in the code
1. **The search ball is far too big for slow particles.** It holds several wrong
   candidates, including fake points.
2. **The 2D score treats a missing camera as pixel (0,0).** `do_tracking` and
   `project_fn` both replace "no data" with 0 (`np.nan_to_num`). The code then
   tries to skip missing cameras, but it never sees one, because none are
   "missing" any more. So a candidate seen by 3 of 4 cameras gets a huge
   penalty. This may be why the 2D score never helped in tests.
3. **Projecting the guesses to the cameras runs one point at a time**
   (`_build_project_fn`): a Python loop over points × cameras. It is slow for
   no reason, and one batched call per camera gives the same result.
   **Done** (commit 7263d414, 21–29% faster, identical results).
4. **`max_gap` counts steps, not missing frames.** A track is searched while
   `frames since last point <= max_gap`, and a consecutive step is 1. So
   `max_gap=1` means "no gap bridging", `max_gap=2` bridges one missing frame,
   and `max_gap=0` finds no links at all. Keep this in mind for A5 ("bridge up to
   10 frames" needs `max_gap=11`), and for any benchmark label with `max_gap=0`.

---

## 2. Rules for every change

- **Every new implementation comes with a test and a before/after benchmark.**
  - **Test.** At least one unit test in `tests/unit/` that fails before the change
    and passes after it. For a bug fix, the test reproduces the bug first.
  - **Benchmark before.** Run the Step 0 scoreboard on the current code and save
    the JSON in `bench/`.
  - **Benchmark after.** Run the same cases on the same machine and save the
    JSON too.
  - **Timing.** Time one run at a time (`-j 1`). Runs executed side by side can
    only be compared with each other.
  - **Record.** Put a before/after table in the commit message or PR. No table,
    no merge.
- **One change at a time.** Each change gets its own name in the benchmark (for
  example `two_phase+smallball`), so each gets exactly one measurement.
- **Keep it only if it helps:**
  - it gains at least 0.01 on at least 2 of the synthetic cases;
  - no case gets worse by more than 0.005;
  - it keeps at least as many true points as before;
  - it keeps at least as many true trajectories as before. This matters for
    filters that drop short trajectories: a real particle that enters late
    starts out short.
  - Otherwise it is reverted the same day. This keeps the list of options short.
- **Always report "points kept" next to velocity error.** A tracker can lower its
  error just by throwing away hard particles. At 4× density, two_phase already
  "beats" the perfect linker this way: error 0.460 vs 0.535, because it keeps
  fewer points.
- **Fake points must not increase.**
- **Speed budget.**
  - Speed changes must be at least 20% faster with the same accuracy.
  - An accuracy change may make the run at most 10% slower.
  - After all accuracy steps together, two_phase must stay within 1.5× of
    today's run time on the scoreboard cases.
  - A change that needs more time goes in as an **option, off by default**. It
    may not change the default tracker.
- **Realism checks (the synthetic data is too smooth, section 1).** A change is
  kept only if it also passes:
  - **Acceleration stress.** The frame-skip cases `L1_d1_c0_k4_n150` and
    `L1_d1_c0_k8_n150`. Skipping k frames makes the change of velocity per frame
    k² larger, which brings it up to the noise level at k = 8. The change must
    not get worse there.
  - **Real jitter.** A new stress case with the jitter measured on real data:
    about 1.7× sideways and 1.5× in depth, plus a heavy tail, i.e. a few points
    with 3× jitter (see Step 0, 0.3). The change must not get worse there. This
    case, not the smooth normal case, is the main tuning case for tolerances.
  - **Real data** (CompleteTest wp1 and wp2, lv_multi wp4 and wp5,
    wp1_10_images). There is no known answer here, so check:
    - **jump share:** steps whose velocity changes by more than 0.3 mm in
      depth. It must go **down**: 1.2% today on CompleteTest wp1/wp2, 9.6% on
      lv_multi wp5. This is the main real-data accuracy score;
    - the share of long trajectories never seen by 4 cameras (14% today on
      CompleteTest wp1/wp2) should go down: it is our best real-data hint of
      fake trajectories;
    - the typical change of velocity per frame must not grow;
    - trajectory lengths must not get worse;
    - acceleration kurtosis (`compute_physics_metrics`) must not get worse;
    - the share of links cut by the smoothness check must not jump. A jump
      means the rule is cutting real, accelerating particles;
    - on wp1_10_images, agreement with the original 3dptv results must not get
      worse.
- **Tuning.** Tune on the synthetic "real jitter" case (0.3), where the answer
  is known and the jitter is realistic, then check on all the others. Never
  tune on real data, where there is no known answer.

---

## 3. Step 0: a scoreboard and a way to look inside (1 day)

### 0.1 Scoreboard
- **What.** Run each tracker once on fixed cases and save one JSON file per run in
  `bench/`. Every later change is compared against these files.
- **Where.** Extend `scripts/synth_bench.py`. It already has the cases and the
  metrics. (`scripts/bench_two_phase.py` is a different, small crossing test.)
- **Cases.**
  - normal: `L1_d1_c0_k1_n200`
  - clustered + skip ×4: `L1_d2_c0.3_k4_n150`
  - 4× density: `L1_d4_c0_k1_n150`
  - real jitter: the normal case with real-level jitter (new, see 0.3)
  - real data, long: CompleteTest wp2 (1072 frames), with wp1 as a second check
  - real data, short: lv_multi wp4 and wp5 (20 frames each), and
    `wp1_10_images` (10 frames). See 0.3.
- **Numbers.**
  - correct links;
  - wrong links;
  - points kept;
  - fake points;
  - long trajectories;
  - velocity error;
  - run time (total, and per frame: average and slowest 5%);
  - fake trajectories by length (1–2, 3–5, 6–10, 11–30, over 30 frames);
  - for long trajectories, real and fake, the share of frames seen by all
    4 cameras.

  The last two numbers track the fake-trajectory problem (section 1), which A10
  and A11 attack.
- **Fairness fix.** The perfect linker's "long trajectory" score counts gaps of up
  to 3 frames. If we let the tracker bridge gaps of 10 frames, score the perfect
  linker with 10 as well.

### 0.2 Decision log
- **What.** Add `debug=False` to `TwoPhaseTrackerConfig`. When it is on, each
  group records:
  - its candidates and their scores;
  - the winner;
  - how close the runner-up was;
  - whether the quick approximation was used.
- **Why.** Then every accuracy question below can be answered by looking, not by
  arguing. The same log feeds `tracking-brain-viz/brain_viz.py`. Today it can
  only show the final links, not why they were chosen.
- **Safety.** When it is off, nothing changes.

### 0.3 Realism check and stress cases
- **First result (section 1).**
  - Real data jitters 1.4–2× more than the synthetic data, with a heavier tail.
  - It also has more 3-camera points and more long trajectories never seen by
    4 cameras.
  - lv_multi wp5 also has 9.6% jump steps, most likely wrong links.
  - So the synthetic normal case is too gentle for tuning tolerances. Build the
    "real jitter" case below and tune on it.
  - Repeat the measurement on `wp1_10_images` once it is copied to the Mac.
- **Also check: do 3-camera points jitter more than 4-camera points?** If yes,
  the smoothing afterwards should trust 4-camera points more, and it is one
  more reason to prefer 4-camera candidates (A2).
- **Measure the same numbers on real data.** Section 1 has them for the
  synthetic cases. On confident trajectories in real data, measure:
  - the frame-to-frame jitter per axis;
  - the change of velocity per frame;
  - the position error of the "keep going the same way" guess.
- **Compare with the synthetic data.**
  - If real jitter or acceleration is several times larger, the synthetic cases
    can't be trusted for tuning tolerances.
  - In that case, build a synthetic case that matches the real numbers and use it
    as the main tuning case.
- **Build a "real jitter" stress case** in `scripts/synth_bench.py`: the normal
  case with more blob-centre noise.
  - Tune it until its jitter matches CompleteTest wp2 (section 1, table 2):
    about 0.008 mm sideways and 0.039 mm in depth, and a worst 10% near
    0.08 mm.
  - Add a heavy-tail version: about 10% of points with 3× the noise, like
    lv_multi wp5.
  - This adds frame-to-frame jitter, not slow offset.
  - Check the jitter with the same 11-frame fit used for section 1.
- **Real data 1: CompleteTest wp1 and wp2** (confirmed real, 2026-10-01).
  - **Location.** `~/Downloads/CompleteTest-e2e-local/wp1/test` and `.../wp2/test`,
    on this Mac.
  - **Size.** 1072 frames each, about 1180 3D points per frame. 4 camera views
    split from one image, 5000 fps.
  - **Ready to use.** 3D points, blobs and two_phase links are stored in
    `res/run.zarr`.
  - **Results.** `CompleteTest-e2e-local/outputs` has the flow-rate, shear-stress
    and dissipation results, with the comparison against the legacy MATLAB
    results (`outputs/run_all/figures/06_reference_overlay.png`). A better
    tracker should bring shear stress and dissipation closer to the legacy
    curve.
  - **Best use.** The main real dataset:
    - long enough for trajectory lengths, the 4-camera signal, gap bridging and
      timing;
    - the right place to profile speed (S0).
  - **Don't overwrite it.** Copy the folder before re-running.
- **Real data 2: lv_multi** (confirmed real, 2026-10-01; part of openptv-cloud).
  - **Location.** `~/Documents/Github/openptv-cloud/examples/lv-multi`, on this
    Mac.
  - **Two runs, wp4 and wp5.**
    - 20 frames each, about 1000 and 1300 3D points per frame.
    - 4 camera views split from one image.
    - Shared calibration in `cal/`.
  - **Ready to use.** 3D points, blobs and two_phase links are already stored
    in `wp4/res/run.zarr` and `wp5/res/run.zarr`.
  - **Don't overwrite it.** Copy the folder before re-running. Also, openptv-cloud
    has `scripts/generate_synthetic_lv_multi.py`, which writes synthetic images
    into `examples/lv-multi`; never run it on the real copy.
  - **Best use.** wp5 is the real-data accuracy test (9.6% jump steps today).
    wp4 is the "easy real" control.
- **Real data 3: `wp1_10_images`** (confirmed real, 2026-10-01).
  - **Location.** `C:\Users\alex\Downloads\ptv_data\wp1_10_images` on the Windows
    machine. Copy it to the Mac before starting; it is not here yet.
  - **Frames.** 10 frames, 100001–100010, 4 cameras.
  - **Reference.** It includes the original 3dptv program's results
    (`res_ground_truth_backup`). The existing scripts already compare against
    them link by link: `verify_two_phase_wp1.py`, `sweep_twophase_wp1.py`,
    `run_postprocess_wp1.py`.
  - **What 10 frames are enough for:**
    - frame-to-frame jitter;
    - change of velocity per frame;
    - the error of the "keep going the same way" guess;
    - the share of links cut by the smoothness check;
    - agreement with 3dptv.

    These are exactly the realism numbers above.
  - **What 10 frames are not enough for:** long-trajectory numbers, fake runs
    longer than 10 frames, gap bridging up to 10 frames, and timing. Use
    CompleteTest wp2 and the synthetic cases for those.
- **Synthetic, not real:** `~/Downloads/HiDImaging/Synthetic-CompleteTest` and the
  benchmark cases built from it (`tracker-bench-2026-09-29`).
- **Script.** Start from the scratch measurements used for section 1 and move
  them into `scripts/`:
  - `smoothness.py`: synthetic cases;
  - `real_smoothness.py` and `real_long.py`: real data;
  - `jitter_fit.py`: jitter around an 11-frame fit.

---

## 4. Accuracy

### What we borrow from other trackers
Checked in their code, not only in their descriptions. Numbers are from the
benchmark, normal case unless stated.

| Borrowed idea | From | Why it works there | Goes into |
|---|---|---|---|
| **Look ahead before deciding:** score a candidate by how well it continues 2 frames ahead | trackcorr (4-frame cost), 4BE (n+2 check) | In two_phase, the smoothness check (a crude after-the-fact look-ahead) is the only measured win | A13, now a firm step |
| **Velocity from a smooth fit, not from 2 points** | proPTV-style (`_smooth_history`) | Best continuity (0.572 = perfect linker) and 99.5% of true links found | A4 |
| **Camera count in the score:** candidates seen by more cameras win | trackcorr (`rr = (...) / quali`) | 17.5% of 3-camera points are fake, against 3% of 4-camera points | A2, now a firm item |
| **Trajectories with history go first; newborns guess from neighbours** | track3d (`priority_segment_3d`, 3 levels) | A newborn has no velocity; its neighbours do | A12 + new A14 |
| **Smooth motion as a soft penalty** | proPTV-style cost (velocity + acceleration terms); myPTV angle limit | myPTV has the most correct links (0.953) | A2, on smoothed velocity only |
| **Build points from blobs near the guess** | trackcorr (added points) | Rebuilds particles the camera matching missed | A11's engine |

**Not borrowed:**
- trackcorr's greedy conflict settling. The Hungarian solver is better; it is
  one reason two_phase wins at 4× density (0.460 vs 0.769).
- myPTV's hard speed, acceleration and angle limits. They lose true links: it
  finds only 88% of them.
- proPTV-style's single fixed radius for every trajectory. It is part of why
  that tracker fails with skipped frames (0.772).
- proPTV-style's refit of the whole history at every frame. It is the reason
  that tracker is 12× slower.

### A1. A small search ball shaped like the shake (TRIED 2026-10-01, REVERTED)
**Result: it does not help two_phase.** Implemented as designed: seeded tracks
searched in the ellipsoid `|d_i / sigma_i| <= k * sqrt(1+(1+g)^2+g^2)` with
`sigma = (0.008, 0.008, 0.045)` mm (the measured jitter), new tracks keep the
round ball. Full numbers: `bench/step4_A1_noise_gate_REVERTED_2026-10-01.json`.

| Case | Baseline vErr / points kept / true links found | A1, k = 6 | A1, k = 3 |
|---|---|---|---|
| Normal | 0.1780 / 0.707 / 0.987 | 0.1784 / 0.706 / 0.980 | 0.1881 / 0.694 / 0.925 |
| Real jitter | 0.1977 / 0.706 / 0.975 | 0.2018 / 0.704 / 0.963 | 0.2448 / 0.682 / 0.820 |
| Skip ×8 | 0.1218 / 0.669 / 0.930 | 0.1285 / 0.628 / 0.855 | 0.1557 / 0.512 / 0.657 |
| 4× density | 0.4592 / 0.258 / 0.937 | 0.4341 / 0.256 / 0.921 | 0.3682 / 0.241 / 0.842 |
| Clustered + skip ×4 | 0.1632 / 0.398 / 0.891 | 0.1504 / 0.386 / 0.863 | 0.1475 / 0.328 / 0.697 |

- A tighter gate **loses true links** and breaks trajectories apart. It "wins"
  at 4× density and on the clustered case only by keeping fewer points, which
  section 2 forbids.
- Link precision hardly moves (0.933 → 0.934–0.936). So extra candidates inside
  the ball are not what causes wrong links; ghosts are (section 1).
- Control: a smaller *round* ball (`v_max` 0.5) gains at most 0.007, and breaks
  skip ×8 (0.1218 → 0.1656).
- The earlier trackcorr result (0.51 → 0.18 with a tight `dacc`) came from
  trackcorr's greedy conflict settling, which two_phase does not have.
- **Kept from this work:** the `scripts/synth_bench.py build --sigma-px` option,
  the measured jitter numbers (section 1), and the knowledge that real
  accelerations and heavy-tailed noise make a Gaussian gate too tight.
- **Still open from the A1 text below:** the automatic-parameter fixes in
  `tracking_recommender._suggest_params` and `tracking_warmup._tune_from_displacements`.
  They matter for trackcorr and the other trackers, not for two_phase. Park
  them.

The original text of A1 follows, for reference.

#### A1 original design
- **Problem.** One round ball sized for the fastest particle. For a slow particle
  it holds many wrong points.
- **Change.**
  - Add `noise_sigma = (sideways, sideways, depth)` to `TwoPhaseTrackerConfig`.
    The default is None, meaning the old behaviour.
  - The ball becomes long in depth and thin sideways, "k times the shake".
  - It grows the longer a particle has been missing.
  - Use the same scaled distance in the 3D score and in `confirm_link_tuples`
    (`tracking_postprocess.py`).
  - To do this cheaply, divide the coordinates by the shake before the
    nearest-point search. Then the stretched ball becomes a plain round ball.
- **Watch out.** The guess itself is shaky, because it comes from two shaky
  positions. The ball must be about 2.5× the point shake, not 1×, or true links
  are lost. `synth_bench._kinematic_bounds` already uses this factor.
- **Automatic setting.**
  - Estimate the shake from warm-up tracks: how much confident trajectories
    wiggle, per axis (`tracking_warmup.py`).
  - Use this frame-to-frame wiggle, **not** the position error against truth.
    Most of the position error is a slow offset that doesn't affect the next
    step (section 1). Using it would make the ball about 2.5× too big in depth.
  - Also fix the automatic parameter suggestions. Today both
    `tracking_recommender._suggest_params` and
    `tracking_warmup._tune_from_displacements` size this box from how far
    particles move, not from the shake.
- **Evidence.** On trackcorr, only shrinking this box cut the hard-case error from
  0.51 to 0.18, and it also ran faster.
- **Stop if** we lose correct links while wrong links stay the same.

### A2. A fairer score
- **First, fix bug 2** (missing camera = pixel 0,0), then measure again. This alone
  may make the 2D score useful for the first time.
- **Then add the camera count (from trackcorr, firm item).** A candidate seen by
  more cameras gets a lower score, as trackcorr divides by `quali`.
  - two_phase already loads the camera ids in `do_tracking`.
  - It only helps when trajectories compete for a point. Long fake trajectories
    rarely compete, so it complements A10/A11 but doesn't replace them.
- **Only if the score still loses,** try the richer score. It adds together:
  - the 3D distance scaled by the shake;
  - the pixel distance in each camera, divided by that camera's pixel noise;
  - a **soft** penalty for changes of velocity and direction (from the
    proPTV-style cost and myPTV's angle limit).

  The direction penalty stops two particles that cross from "bouncing" off each
  other. Use it only with the **smoothed** velocity from A4: between two raw
  steps the direction is mostly noise. Never use a hard angle limit; it costs
  true links (myPTV finds only 88%).
- **Tuning.** Tune the weights on one case, then check them on the others.
  `leaf_weight=0` must still mean "3D distance only".

### A3. Crowded groups and second chances
- **Problem 1.** Groups with more than 128 members fall back to the quick
  approximation. So the most crowded frames get the weakest solver.
- **Problem 2.** A trajectory that loses a contested point just dies. It never
  tries its second-best candidate.
- **First, count** how often the quick approximation runs (decision log, 0.2).
  The small ball from A1 makes groups much smaller, so this may go away by
  itself.
- **Only if still needed:**
  - Cut giant groups into smaller pieces in space and solve each exactly.
  - Give the loser its second choice when the winner only barely won.

### A4. Smoother velocity
- **Problem.**
  - Velocity from two shaky points is about as shaky as the motion itself.
  - One bad link spoils every later guess of that trajectory, and it never
    recovers.
- **Change.**
  - Estimate velocity from several frames (a simple running filter, "α-β" or
    Kalman), not from the last two points. The filter also knows how uncertain
    its guess is, and that uncertainty sets the ball size.
  - **Borrowed from the proPTV-style tracker,** which has the best continuity.
    Keep the cost fixed per step: use the running filter, or fit only the last
    ~10 points. That tracker refits the whole history at every frame, and that
    is why it is 12× slower.
  - **Don't over-smooth.** The synthetic flow has almost no acceleration
    (section 1), so a very strong filter looks perfect there. It must pass the
    frame-skip stress cases (k = 4, 8) and the real-data check (section 2).
  - If a link is later cut by the smoothness check, undo the velocity update it
    caused.
  - Add a hook, `pred_fn`, so a different guesser can be plugged in later. One
    example is "neighbours move together". The default is None, meaning the
    built-in filter.
- **Expected.** Much better guesses. So the ball can be smaller, and fewer
  particles fall outside it.

### A5. Honest gap handling
- **Problem.** After a gap, the tracker uses a stale velocity with the same ball.
  A wrong link across a gap looks like a confident long jump.
- **Change.**
  - The ball grows with the gap. This comes free with A4.
  - Add a gap penalty to the score, so a fresh new trajectory can win against a
    stale one.
  - Links across gaps already stay out of the stored `ptv_is` files. They once
    made an 82 mm phantom jump. Extend the existing test
    `test_linkage_drops_gap_links_no_phantom` to keep it that way.
- **Reconnect afterwards.** `relink_trajectory_gaps` (`tracking_postprocess.py`)
  already reconnects broken pieces with a straight-line guess. Give it the
  smooth-curve guess from A4 or A9 instead, so it can bridge up to about
  10 frames.

### A6. Fake points: the simple rule only
- **Keep.** Add an option `min_cams`: a candidate must be seen in at least N
  cameras.
- **Drop the "blob ownership" rules for now.** They were: "two points share a
  camera blob → keep one", and "all blobs already owned by other trajectories →
  never start a trajectory". Neither can trigger on today's data:
  - The stored 3D points never share a blob. The camera-matching step refuses a
    blob that is already used (`correspondences.py`, `take_best_candidates`).
  - trackcorr's "added" points also use only unused blobs (`track.py`, the
    `t.tnr == TR_UNUSED` search).
  - Revisit these rules if a tracker ever builds 3D points that reuse blobs.
- **The real fake-point fix is A10 + A11.** A9 only removes the short fake runs.

### A7. Better smoothness check (the one proven win) - TRIED 2026-10-01, REVERTED
**Result: none of the variants beats the current rule on average**
(`bench/step8_A7_confirmation_variants_REVERTED_2026-10-01.json`). Tested on top of the
quality defaults, on five cases (base = forward check, absolute `confirm_tol=0.3`):
a kink measured per axis in units of `sqrt(6)*sigma` with `sigma=(0.008, 0.008, 0.045)` and
tolerance k = 3, 4, 5, 7, and a two-sided check (`either`: keep when the onward OR the
preceding step is consistent; `both`).

| Variant | Mean change of velocity error | Mean change of points kept |
|---|---|---|
| sigma k=3 / 4 / 5 / 7 | +0.0033 / +0.0023 / +0.0010 / +0.0083 | −0.033 / −0.015 / −0.005 / +0.004 |
| either 0.3 | +0.0035 | +0.001 |
| both 0.3 | +0.0008 | −0.008 |
| sigma 5 + either | +0.0029 | −0.004 |

- **Best tolerance depends on density.** Per case the best variant was: normal and
  real-jitter sigma k=7 (0.1604, 0.1766 against 0.1652, 0.1845); clustered + skip ×4
  `both` or k=4 (0.145–0.147 against 0.153); **4× density sigma k=3 (0.3695 against
  0.4207, −0.051)**; skip ×8 the base. A loose tolerance suits sparse data, a tight one
  dense data; one fixed value cannot win everywhere.
- **A7b measured (2026-10-01), reverted** (`bench/step9_A7b_confirm_tolerance_vs_density_REVERTED_2026-10-01.json`).
  Tolerance k = 1.5…10 in sigma units against density, seven cases. The big gains at tight k
  are **obtained by dropping true points** (4× density k=1.5: error −0.14, but 16% fewer
  true points). With the points-kept rule (no more than 0.007 fewer points):

  | Case | Points per frame | Neighbours within 5 mm | Best k | Gain vs absolute 0.3 |
  |---|---|---|---|---|
  | real jitter | 1157 | 6 | 10 | −0.0129 (and +0.008 points) |
  | skip ×8 | 1169 | 6 | none | +0.0024 |
  | clustered + skip ×4 | 1498 | 7.7 | 5 | −0.0055 |
  | density ×2 | 1914 | 10 | none | +0.001 |
  | density ×4 | 2663 | 11 | 4 | −0.0171 |

  No fixed k passes the plan's rule (k=10: +0.079 at ×4 density). A rule "k from the local
  neighbour count" fits these five points but would be tuned on the same cases it is judged
  on. **Decision: keep the absolute `confirm_tol=0.3` default.** Advice for users, in
  `docs/two-phase-tracking.md`: sparse data (about 6 neighbours within 5 mm) can try a
  looser `confirm_tol`, dense data (10 or more) a tighter one, and should check it with
  `scripts/realism_metrics.py`. A proper data-driven rule needs more cases of different
  density and, ideally, a mixture fit of the measured kink distribution.
- The code for `confirm_sigma` / `confirm_mode` was removed again (rule: a change that
  does not gain is reverted); the idea text below is kept for reference.

#### A7 original design
- **Problem.** `confirm_tol` is yes/no. It only looks forward, uses a fixed number
  in mm, and cuts on one kink, even when the kink is only shake.
- **Change.**
  - Also check backward: cut a link whose previous step kinks.
  - For long trajectories, cut only if 2 of 3 steps in a row fail.
  - Set the tolerance from the shake (A1), not as a fixed mm value.
  - Use the decision log: a kinked link that clearly won may stay; a smooth link
    that barely won is suspect.
- **Where.** `tracking_postprocess.py`, so every tracker benefits, not only
  two_phase.

### A8. Forward + backward, done right or removed
- **Problem.**
  - The backward run starts from scratch.
  - It can only fill free ends, so it can never fix a bad forward link.
  - It measured a gain of less than 0.01.
- **Change.**
  - Start the backward run from the ends of the forward trajectories.
  - Settle conflicts by score, not "forward always wins".
  - Join a forward trajectory's end to a backward trajectory's start when they
    fit. This is where the real gain would be.
- **Rule.** If it still gains less than 0.02, delete the backward option.

### A9. Smoothness filter for fake points (new, not in the earlier plans)
- **Idea.** A real particle follows a smooth path; a fake point jumps around.
- **Change.**
  - Fit a smooth curve through about 10 frames of each trajectory.
  - Drop points that fit on no smooth curve.
  - Drop trajectories that are too short to fit.
- **Gaps.** Fit the piece on each side of a gap separately. Never cut a
  reconnected gap (from A5) just because the jump across it looks like a kink.
  This is why A9 comes after A4, A5 and A7 in the work order: by then the filter
  knows where the reconnected gaps are and how clearly each link won.
- **Check.** It must keep as many true trajectories as before (section 2).
  Short trajectories from real particles that entered late must survive.
- **Expected.**
  - It removes fake points inside real trajectories and short fake runs: about
    10–15% of fake points in the normal case, more at 4× density.
  - It does **not** remove the long fake trajectories, which are smoother than
    real ones (section 1). A10 and A11 handle those.
  - It works for every tracker.

### A10. Camera test for whole trajectories (finds the long fake trajectories) - NOW THE FIRST STEP
**First measurements (2026-10-01, real-jitter case, two_phase output):**
- **Where the ghost damage is.** Dropping ghosts from the *output* gives the same
  gain as dropping them from the input (vErr 0.1977 → 0.1471 vs 0.1465). So A11
  is not needed for accuracy if ghosts can be recognized afterwards.
  - 17196 of 18736 ghost points sit in ghost-only trajectories (4739 of them).
    Dropping just those: vErr 0.1977 → 0.1636, about 60% of the gain.
  - The other 1540 ghost points sit inside real trajectories; removing them
    brings vErr to 0.1471.
- **The 4-camera rule alone is too weak.** "Long trajectory never seen by 4
  cameras" removes only 28% of ghost points but 3.2% of real ones, and vErr gets
  slightly *worse* (0.2022, points kept 0.706 → 0.682). Looser rules lose more
  real points (see the bench files). Do not use it as a hard filter.
- **Ghost trajectories are almost stationary.** Median speed 0.021 mm/frame
  against 0.126 for real ones, and much lower jitter. They come from static
  false targets in the images. The same happens in L2 (the real image
  pipeline), so it is not only an L1 artefact.
- **But do not tune on this.** Real data has almost no stationary trajectories:
  0.1% below 0.01 mm/frame against 0.9% in L1. Its slow trajectories (7% below
  0.03 mm/frame) are probably real slow flow (pulsatile flow, walls), and
  dropping them would bias the mean flow. The synthetic ghosts are probably
  easier to recognize than the real ones.
- **Ray miss distance (rcm) is the best feature found** (second measurement,
  `bench/step5_A10_rcm_postfilter_2026-10-01.json`). Ghost points miss by about
  1.8× more than real points (median 0.068 vs 0.038 mm). Per trajectory, the
  mean rcm separates ghost-only from real trajectories with AUC 0.88 (4-camera
  share 0.78 after flipping, speed 0.95 but not trusted). It needs no truth and
  no speed.
- **As a post-filter it still costs real points.** Dropping trajectories of at
  least 6 frames with mean rcm above 0.06 mm removes 35% of ghost points and 2.2%
  of real points: vErr 0.1977 → 0.1888, points kept 0.706 → 0.689. Stricter
  settings reach vErr 0.1683 (length ≥ 3, rcm > 0.05) but keep only 0.648.
  The cheat limit is 0.1636 with no points lost.
- **The absolute scale does not transfer to real data.** On real wp2 (300
  frames) the median point rcm is 0.044 (4-camera) and 0.052 (3-camera); on
  synthetic both are 0.037. A threshold of 0.06 would flag 26% of real
  trajectories against about 4% of synthetic ones. The real values probably
  include calibration error that varies over the volume, so a fixed threshold
  would delete real particles in badly calibrated regions and bias the flow.
- **Consequences for A10:**
  1. Use rcm **relative** to its local typical value (neighbouring points of the
     same frame, or the same region over time), not as an absolute number.
  2. Map real rcm against position first (is it spatially structured?) before any
     rule goes near the tracker.
  3. Prefer a **soft** use over a hard filter: add rcm to the link score (A2) and
     penalize starting a new trajectory on a high-rcm point, so doubtful points
     lose contests instead of being deleted.
  4. Amend the "points kept" rule for filters: points may be removed if the
     removed points are mostly ghosts (e.g. at least 10 ghost points removed per
     real point), because accuracy ranks above density.
- **The distributions (figure: `docs/plans/figs/rcm_distributions.png`).** No
  single cut-off is right; the ghost share changes smoothly with rcm:

  | rcm band (mm) | Ghost share, 3-camera points | Ghost share, 4-camera points |
  |---|---|---|
  | below 0.03 | 4% | 0% |
  | 0.03–0.05 | 9% | 1% |
  | 0.05–0.07 | 29% | 6% |
  | 0.07–0.10 | 52% | 27% |
  | above 0.10 | 76% | 41% |

  Real points form a narrow peak near 0.035 mm; ghosts form a broad hump from
  0.05 to 0.12 mm. At the same rcm, a 3-camera point is far more likely to be a
  ghost than a 4-camera point. Per trajectory, the mean rcm of real trajectories
  is a narrow peak near 0.04 mm, ghost-only trajectories are broad with a long
  tail, and mixed trajectories sit between them.
- **Real data is broader, partly because of position.** Real wp2 point rcm has
  median 0.044 (4-camera) and 0.052 (3-camera), with a heavy tail (3-camera
  p90 0.090 vs 0.063 synthetic). The map over the volume is structured: about
  0.035–0.04 mm in the middle and 0.065 mm or more at the edges.
- **After dividing by the local typical rcm** (median of 4-camera points in a
  10×10×10 grid of cells) the real distribution matches the synthetic one:

  | | median | p90 | p99 |
  |---|---|---|---|
  | Real wp2, 4-camera | 1.00 | 1.48 | 2.14 |
  | Synthetic real points, 4-camera | 0.99 | 1.54 | 2.20 |
  | Real wp2, 3-camera | 1.02 | 1.81 | 2.72 |
  | Synthetic real points, 3-camera | 0.97 | 1.64 | 2.54 |

  So the relative rcm is a sensible scale on real data, and the synthetic ghost
  share by relative rcm is a plausible guide. Synthetic ghost share by relative
  rcm: below 0.9 → 9% / 1%; 1.2–1.6 → 20% / 4%; 1.6–2.2 → 37% / 9%; above 2.2 →
  62% / 28% (3-camera / 4-camera).
- **Design from this (soft, no cut-off).**
  1. For every point: relative rcm = rcm / local typical rcm (map built from the
     run itself, 4-camera points, with cells of at least 60 points and the global
     median elsewhere).
  2. Turn (relative rcm, number of cameras) into a ghost probability, using the
     bands above as a first table; later fit a smooth curve.
  3. Use it in three places, all soft: add a cost to links onto doubtful points
     (A2); penalize starting a new trajectory on a doubtful point; and weight
     the points of a trajectory when smoothing afterwards, instead of deleting
     them.
  4. Judge it on the real-jitter case against the plan's rules, with the
     amended "points kept" rule.
- **Is rcm more distinctive relative to a distance? (tested 2026-10-01)** Share of
  rcm variance explained by a binned curve of each variable, 4-camera points:

  | Variable | Real wp2 | Synthetic |
  |---|---|---|
  | Distance from the volume centre | **0.115** | 0.004 |
  | Nearest image edge, any camera | **0.100** | 0.006 |
  | Edge distance, single camera 1 / 2 / 3 / 4 | 0.104 / 0.104 / 0.091 / 0.087 | about 0.004 |
  | Depth z | 0.012 | 0.001 |

  - **On real data, rcm grows away from the centre:** median 0.035 mm within
    10 mm of the centre to 0.058 mm at 50–60 mm. Equivalently, within 50 px of an
    image edge the median is 0.056 mm against 0.039 mm in the image interior.
    The two are the same effect (outer points are near image edges). Together
    they explain about 11% of the variance; edge distance adds only 0.018 once
    the centre distance is used.
  - **Depth does not matter.** So normalizing by z is useless.
  - **Synthetic data has none of this structure** (all R² below 0.01), so it
    cannot show whether such a normalization improves ghost separation; the
    synthetic ghost-vs-real AUC stays at 0.82 (3-camera) and 0.87 (4-camera)
    for every normalization.
  - **Consequence.** The right scale for real data is a smooth curve in the
    distance from the centre (or nearest image edge), fitted per run from the
    4-camera points. The 10×10×10 cell medians already capture it, but a smooth
    curve with 2–3 parameters is more stable in sparse cells. To *validate* the
    gain, build a synthetic case with a radial calibration error (rcm growing
    with distance, like real wp2) and check that ghost separation improves
    after normalization.
- **Built and measured (2026-10-01): quality marks in the tracker**
  (`src/openptv2/point_quality.py`, options `q_weight`, `q_seed`, `q_young` in
  two_phase, default off; `bench/step6_A10_quality_mark_2026-10-01.json`).
  The mark is the ghost probability from the relative rcm (scale = straight line
  in the distance from the volume centre, fitted per run on 4-camera points).

  | What the mark does | Effect on velocity error (real-jitter case) |
  |---|---|
  | Raises the cost of linking onto a doubtful point (`q_weight` 3 or 10) | none (0.1977 → 0.1976): such links are rarely contested |
  | A doubtful point may not start a trajectory (`q_seed` 0.2) | 0.1977 → 0.1929 |
  | + a trajectory with fewer than 3 points may not continue onto a doubtful point (`q_young` 3) | **0.1977 → 0.1845** |

  Best setting, `q_seed=0.2`, `q_young=3`, against the default:

  | Case | Velocity error | Points kept | Correct links | Fake points |
  |---|---|---|---|---|
  | Normal | 0.1780 → 0.1652 | 0.707 → 0.703 | 0.934 → 0.957 | 6.9% → 4.3% |
  | Real jitter | 0.1977 → 0.1845 | 0.706 → 0.702 | 0.933 → 0.951 | 6.9% → 4.8% |
  | Clustered + skip ×4 | 0.1632 → 0.1529 | 0.398 → 0.394 | 0.929 → 0.944 | 6.5% → 5.3% |
  | 4× density | 0.4592 → 0.4207 | 0.258 → 0.251 | 0.867 → 0.911 | 11.8% → 8.6% |
  | Skip ×8 | 0.1218 → 0.1134 | 0.669 → 0.667 | 0.933 → 0.946 | 6.6% → 5.5% |

  On real wp2 (first 300 frames, re-tracked with the project's own settings):
  trajectories 16375 → 15310 (fewer fragments and ghost trajectories), mean
  length 19.5 → 20.2, long trajectories never seen by 4 cameras 11.6% → 10.4%,
  jump share unchanged (1.46% → 1.45%), jitter unchanged, track time +6%.

  It meets the plan's rules: gain of at least 0.01 on four cases, nothing worse,
  fake points down everywhere, points kept down by at most 0.007 (and the removed
  points are mostly ghosts), time within the budget.
  About 40% of the possible ghost gain on the real-jitter case is reached (the
  cheat with all ghosts removed reaches 0.1465).
- **Open check.** The real tail may contain more ghosts than the synthetic one
  (real 3-camera tail is heavier). There is no truth on real data, so compare
  the share of points above relative rcm 2.2 (real vs synthetic) after each
  change, and watch the jump share.
- **Next for A10 (older text).** Find features that separate ghosts without relying on speed:
  the mix of 3- and 4-camera frames over the whole trajectory (ghost-only
  trajectories are mostly 3-camera: median share 1.0 vs 0.31), and the ray miss
  distance. Check that any rule flags a similar share of trajectories on the
  real wp2 data as it does on the synthetic data.
- **Idea.** A real particle is usually seen by all 4 cameras at least now and
  then. A long fake trajectory is usually never seen by all 4.
  - 65% of long fake trajectories are never seen by 4 cameras.
  - Only 6% of long real trajectories are like that.
- **Change.** For each long trajectory, count the frames in which it is seen by
  all 4 cameras. The camera ids are already loaded in `do_tracking`, so this is
  cheap.
- **Not a hard rule by itself.** "Drop if never seen by 4 cameras" would remove
  about 145 fake trajectories but also about 230 real ones in the normal case.
- **Combine it with a second signal:** how closely the camera rays meet,
  averaged over the whole trajectory.
  - A real particle's rays meet within the shake; a fake one's may miss more.
  - This value is not stored today, so measure first whether it separates real
    from fake.
- **Output.** A list of suspected fake trajectories. A11 uses it.

### A11. Feed tracking back into camera matching (removes fake points at birth) - THE BIGGEST LEVER
- **Idea.** A fake 3D point is made from blobs that belong to real particles, and
  those real particles then go missing. Tracking can tell which points are
  suspect; today that knowledge never goes back to the camera-matching step.
- **Change.**
  1. Take A10's suspected fake trajectories.
  2. Release their blobs.
  3. Re-run camera matching for those frames with the suspect combinations
     excluded.
  4. Track again locally.
- **Different from trackcorr.** trackcorr only builds new points from blobs
  nobody used. A11 also frees the blobs a fake point stole, so the real particle
  can come back. The original proPTV does something similar; our proPTV-style
  plugin skipped that part.
- **Check.**
  - Fake points go down.
  - True points kept go **up**: the freed blobs should give back real particles.
- **Where.** A new step between tracking and output. It calls
  `algorithms/correspondences.py` for the affected frames only.
- **Engine borrowed from trackcorr.** Rebuild points by projecting the guess
  into each camera, searching the freed blobs near it, and triangulating. Reuse
  trackcorr's code for this (`track.py` search, `assess_new_position`).
- **Speed.** It runs only on suspect trajectories and their frames, so the cost
  should be small. Check it against the speed budget (section 2).

### A12. A "no link" option in the solver (fewer broken trajectories)
- **Problem.** Starting and ending a trajectory costs nothing. So the solver
  never weighs "link to this doubtful candidate" against "end here and start
  fresh". Only the ball edge decides.
- **Change.** Give the solver an explicit "no link" choice with a cost: end
  this trajectory and start a new one.
  - A link is made only if it is cheaper than that choice.
  - Tune the cost on synthetic data so the number of trajectory pieces matches
    the perfect linker's.
- **History first (from track3d).** In a contest, a trajectory with history
  should beat a newborn. Give newborns a slightly higher score (cost) than
  established trajectories.
- **Expected.** Fewer broken trajectories and fewer doubtful links. Little effect
  on long fake trajectories: they are born once and live 30+ frames.
- **When.** Right after A1: the cost is in the same units as A1's scaled
  distance.

### A13. Look ahead before deciding (from trackcorr and 4BE)
- **Problem.** The smoothness check runs after the whole movie. A cut link frees
  nothing, and the trajectory that lost the contest is already gone.
- **Idea (borrowed).** trackcorr and 4BE score a candidate by how well it
  continues 2 frames ahead. In two_phase, the smoothness check (a crude
  after-the-fact look-ahead) is the only measured win, so moving the look-ahead
  into the decision should give more of the same gain.
- **Change.**
  - Keep the 2 best candidates of each trajectory for one extra frame.
  - Decide when the next step is known: take the pair that continues smoothly,
    including the loser's second choice.
- **Speed.** Do it only for **contested** groups. Uncontested links are decided
  as today, so the extra cost stays small.
- **Where.** `_track_unidirectional`: a small buffer of undecided links and a
  one-frame delay.
- **When.** A firm step, right after A7.

### A14. Newborns guess from their neighbours (from track3d)
- **Problem.** A newborn trajectory has no velocity, so its first guess is
  "stays where it is".
- **Change (borrowed from track3d's level 2).** For a newborn only, use the
  average velocity of nearby trajectories that already have history. Fall back
  to "stays where it is" if there are no such neighbours.
- **Why it is safe.** It only affects the first step of a new trajectory.
  - Unlike the full "neighbours move together" idea, it doesn't replace any
    trajectory's own history.
  - It doesn't need the big real-data test first. It still must pass the
    realism checks.
- **Where.** The cold-start block in `_track_unidirectional`, using the
  `pred_fn` hook from A4.

### Considered and parked or dropped
- **Parked (only if the numbers ask for it):**
  - "A new trajectory needs 2 links before it counts." Runs of 1–2 fake points
    are only 5% of fake points, and a single point already produces no link.
  - "Solve several frames together as one big optimization." It fixes swaps and
    gaps better, but it cannot fix long fake trajectories, which are valid
    smooth paths. Last resort if A3, A5 and A8 stall.
- **Dropped:**
  - "Re-check contested points with 3 of the 4 cameras." 73% of fake points
    use only 3 cameras; drop one and 2 remain, and any 2 rays meet somewhere.
    Also, long fake trajectories are rarely contested.
  - "Ball size depends on how crowded the region is." The ball should match
    the uncertainty of the guess (A1 + A4), whatever the crowding. Shrinking it
    in crowds loses true links; the solver handles crowds.

### Later: "neighbours move together"
- **Idea.** Guess a particle's next position from its neighbours' motion, through
  the `pred_fn` hook from A4.
- **Test first on real data.** Our synthetic flow was smoothed when it was made,
  so neighbours agree there by construction. The test is described in section 9.
- **Watch out.** Near a shear layer, neighbours move differently. So fall back to
  the particle's own history when the neighbours disagree.

---

## 5. Speed (after accuracy)

- **S0. Profile first** on real CompleteTest wp2 (1072 frames) and on the
  synthetic 4× density case. Likely slow spots, in
  order:
  1. camera projection one point at a time (bug 3). Batch it: big gain, same
     result;
  2. the Python loop that builds candidate scores one pair at a time. Do it with
     numpy arrays;
  3. thousands of tiny groups each sent to scipy. Solve 1-against-1 groups
     directly, without scipy;
  4. the smaller ball from A1, which makes everything above cheaper.
- **S1.** Measure how often a whole frame needs no solver at all (every group is
  1-against-1). If this is rare, frames are too crowded, which the decision log
  will show.

## 6. Running in parallel (after speed)

- **P1. Forward and backward at the same time.** The two runs are independent.
  Run them in 2 processes: about 2× faster, same result.
- **P2. Groups at the same time.** The groups in one frame are independent. Solve
  them in threads: scipy lets threads run in parallel. Keep the quick
  approximation single-threaded.
- **P3. Split the movie into chunks.** `tracking_chunked.py` already has
  `partition_tracking_chunks` and `stitch_chunked_linkages`.
  - Chunks overlap by at least `max_gap` + the smoothness-check length.
  - Accuracy is lost only at the chunk borders, and the stitching repairs it.
  - Test the stitching on synthetic data before the long real wp1 run
    (5005 frames).
- **P4. Many runs at once.** `scripts/_tracker_run_worker.py` already runs
  parameter sweeps in parallel. Just add two_phase settings to it.
- **P5. No GPU or Cython** until S0 and S1 are done. The gains are in batching
  and in the ball size, not in faster arithmetic.

---

## 7. Order of work, with the gain we expect

| # | What | Expected gain | Days |
|---|---|---|---|
Every step: a test and a before/after benchmark (section 2).

| # | What | Expected gain | Days |
|---|---|---|---|
| 1 | Step 0: scoreboard (with "points kept", fake trajectories by length, 4-camera share) + decision log + realism check and stress cases (0.3) | Honest numbers, and knowing how far to trust the synthetic data | 1.5 |
| 2 | Bug 3: batched projection | Faster, same result | 0.5 |
| 3 | Bug 2: missing camera ≠ pixel 0,0 (see below), then measure | Unknown; possibly the first time the 2D score helps | 0.5 |
| 4 | A1 small, shake-shaped ball | **Tried and reverted** (no gain, loses true links) | done |
| 5 | A10 camera test for whole trajectories, plus a check of what else separates ghost trajectories on real data (ray miss distance, jitter of 3- vs 4-camera points) | A list of suspected ghost trajectories | 1–2 |
| 6 | A11 feed tracking back into camera matching (trackcorr's add-point engine). Upper bound from `ghost_bound.py`: vErr 0.1977 → 0.1465 | Fewer fake points **and** more real ones; the main lever | 3–4 |
| 7 | A7 better smoothness check | More accurate | 1 |
| 8 | A2 camera count in the score (from trackcorr); place it in the cost that also works with `leaf_weight=0` | Fewer fake points in contests | 0.5 |
| 9 | A12 "no link" option + history first (from track3d) | Fewer broken trajectories, fewer doubtful links | 1 |
| 10 | A13 look ahead before deciding, contested groups only (from trackcorr/4BE) | More of the one proven gain; fewer trajectories dying in contests | 1–2 |
| 11 | A4 + A5 smoother velocity (from proPTV-style, fixed cost), gap handling, reconnect | Better guesses, longer trajectories | 2 |
| 12 | A14 newborns guess from neighbours (from track3d) | Better first links for new trajectories | 0.5 |
| 13 | A9 smoothness filter (gap-aware) | Removes short fake runs and fakes inside real trajectories | 2 |
| 14 | P3 parallel: split the movie into chunks | Several × faster on long runs, no accuracy loss | 1–2 |
| 15 | S0 + S1 remaining speed items | Faster | 1 |
| 16 | A8 decision on forward + backward. **Then** P1 (run both at once), only if forward + backward survives. A6 `min_cams`, the rest of A2 (richer score), A3, and the parked ideas (section 4) only if the numbers ask for them | Measured case by case | — |
| 17 | "Neighbours move together" for all trajectories (only if the real-data test in section 9 says yes) | Better guesses in dense regions | 2–3 |

Why this order:
- **Ghosts first (new, 2026-10-01).** With ghosts removed, two_phase reaches
  precision 0.9989 and cuts its gap to the perfect linker from 0.0565 to
  0.0229. Everything that only changes the search ball or the score cannot
  reach that (A1 showed it).
- The realism check (0.3) comes first. It decides whether tolerances can be
  tuned on the synthetic cases at all.
- A2's camera count comes right after A1: it is small and uses data two_phase
  already loads.
- A12 comes right after A1, because its "no link" cost is in the same units as
  A1's scaled distance.
- A13 comes right after A7: it moves A7's proven gain into the decision itself.
- A14 comes after A4, because it uses A4's `pred_fn` hook.
- A9 comes after A4, A5 and A7, so the filter knows which jumps are reconnected
  gaps and does not cut them.
- A10 and A11 come after A9. They carry the main fake-point target, and A9 has
  already removed the easy cases, so the suspects left are the long fake
  trajectories.
- P1 comes after the A8 decision, so we don't make fast something we might
  delete.

**Step 3 in detail.** Fixing bug 2 is not enough on its own:
- **Add a test** (`tests/unit/test_two_phase_tracking.py`): one candidate seen by
  3 of 4 cameras must not be penalized for the missing camera. No current test
  has a missing camera, which is how the bug survived.
- **Carry a separate "camera seen: yes/no" array** next to the pixel positions.
  Don't mark "not seen" with NaN inside the numbers: any later `nan_to_num` would
  silently break it again.

---

## 8. How we check each step
- Unit tests:
  `uv run pytest tests/unit/test_two_phase_tracking.py tests/unit/test_tracking_postprocess.py -q`
- A new unit test for the change: it fails before the change and passes after.
- The Step 0 benchmark before and after the change, on:
  - the three synthetic cases;
  - the stress cases (frame skip ×4 and ×8, more jitter);
  - the real data.

  Save both JSON files in `bench/` and put the before/after table in the commit
  message or PR.
- Timing runs one at a time (`-j 1`), checked against the speed budget.
- Look at a few trajectories in 3D with `scripts/viz_compare_3d.py`.
- Apply the rules in section 2 before keeping any change.

## 9. The real-data test for "neighbours move together"
- **Data.** Real CompleteTest wp2 (1072 frames), plus lv_multi and
  `wp1_10_images`; see Step 0, 0.3. That is enough: the test compares one-step
  guesses, which needs only a few frames per trajectory.
- **Method.** Real data has no known answer, so compare the two guesses directly.
  On confident trajectories, predict the next position two ways:
  - from the particle's own history;
  - from its neighbours' motion.

  Then measure which guess lands closer to the point actually measured next.
- **Decision.** Build this idea (step 17) only if the neighbour guess has less
  than half the error of the own-history guess. Otherwise skip it.

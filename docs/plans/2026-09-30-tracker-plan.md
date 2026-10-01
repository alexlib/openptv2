# Tracker plan, in plain words

Branch: `feat/tracker-improvements`. Date: 2026-09-30, updated 2026-10-01 (A1 tried and reverted; ghost bound measured).
This file replaces all earlier tracking plans, including `new_tracking_plan.md` and
`two-phase-accuracy-speed-plan.md`; their ideas are merged here.

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

### A7. Better smoothness check (the one proven win)
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
- **Next for A10.** Find features that separate ghosts without relying on speed:
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

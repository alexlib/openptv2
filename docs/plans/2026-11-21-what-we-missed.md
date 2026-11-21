# What we missed while improving the trackers (review of all feature branches)

Date: 2026-11-21. Follows `2026-11-21-holistic-next-phase.md` (status, results, "what we learned end to end").
Main is at `94a5bbe` (PR #44 holistic + simplified trackers, PR #45 anchored self-calibration).

## 1. What the branches did

Nearly every branch worked on **association** (which point links to which):

| Branch | Idea | Measured result (truth benchmark unless stated) | State |
|---|---|---|---|
| `feat/ilp-temporal-smoothing-impl` | ILP uniqueness, IRLS smoothing, stitching, MHT | `unified` worse than `two_phase` (0.266 vs 0.160); stitching hurts; MHT over-merges | in main, retired from pickers |
| `feat/two-phase-bootstrap-affine-consensus` | VFC spatial consensus, cold-start velocity bootstrap, divergence-free affine consensus | velocity error 0.160 -> 0.158 at 4-10x the time; a tight gate costs yield and long tracks (`bench/vfc_comparison_2026-10-21.json` on that branch) | not merged (its calibration, lsq triangulation and corrective-pass parts are in main) |
| `feat/merged-blob-split`, `feat/unified+merged-blob` | blob ownership loop, predictor split, intensity gate | `blob_loop` 0.42 vs 0.16; split + gate within +-3 %, sign changes with density | not merged; superseded |
| holistic (`two_phase` + `holistic_rounds`) | merged-blob inference loop | 0.13 -> 0.106; real tracks +42 % longer, jumps /3.5 | in main |

The linker already finds 99 % of the reachable links, and phase-averaged statistics moved by less than 2 % with any of
this. **The remaining losses lie before linking (detection) and after it (post-processing and statistics).** Those two
ends have hardly been measured. That is what this plan is about.

## 2. Missed items, in order of expected value per effort

### M1. Score the detectors against the rendered truth (first)

- **Why.** The 27 % merge rate and the 14.6 % of particles with no blob are properties of the detector and its settings,
  not of the scene. `feat/detection-prototypes-dask-trackpy-skimage-proptv` (2026-09) prototyped dask-image, trackpy,
  skimage and proPTV detectors, but compared them only with `targ_rec`, never with the truth.
- **What.** On `Synthetic-CompleteTest/syn1` (per-camera truth pixels in `truth/truth.npz`), for each detector and a
  small grid of its settings (`targ_rec`: `gvthres`, `disco`, `nnmin`; trackpy: `diameter`, `minmass`, `separation`;
  skimage LoG/peak-local-max): count per blob how many true particles it holds (the counting in the merged-blob figure
  script), then report merged share, missing share, false blobs, and centre error of clean blobs.
- **Then.** Run the best one or two through the full chain (matching, `two_phase`, holistic) on the benchmark.
- **Gate.** Merged share down by at least a third (27 % -> 18 % or lower) with missing share and false blobs not higher;
  otherwise detection stays as it is and the pixel-level fit (plan step 3 of the next phase) is the route.
- **Cost.** About a day for the scorecard. The detector code exists on the branch.

### M2. Sensitivity of the statistics to post-processing (in parallel with M1)

- **Why.** Jitter fell 30 % but the statistics moved by less than 2 %. That suggests the openptv-cloud post step dominates
  the result: Savitzky-Golay window 21 (order 2, 5000 fps), the 2.5 m/s speed cap, reconnect gap 6, minimum length 5.
  A strong smoothing window removes small-scale signal as well as noise, so it biases eps and VSS low. That could also
  explain part of the remaining gap to legacy (eps +24 %, VSS +12 % with the same cap).
- **What.** On the existing wp1/wp2 runs (scratch copies, `scripts/eulerian_compare.py`), rerun `convert` and the Eulerian
  stages with window 11 / 21 / 31, with and without the cap, and with reconnect gap 0 / 6. Report the phase-region
  metrics as now.
- **Gate.** If eps or VSS move by more than about 10 % across reasonable settings, the post step is a first-order choice:
  document it and decide the setting from M3's truth. If they stay within a few percent, the statistics really are
  robust, and the tracker work is complete for this output.
- **Cost.** Hours. Everything runs already.

### M3. A truth case for the final statistics

- **Why.** We score tracks against truth but never eps and VSS, the outputs users report. Our synthetic flows are smooth,
  with no turbulent gradients, so they cannot tell whether legacy or we are closer to the right eps.
- **What.** `feature/differentiable-ptv-autoresearch` (2026-08) has a JHTDB client (real DNS trajectories,
  Re_lambda ~ 433, with an Ornstein-Uhlenbeck fallback) and a renderer with Gaussian blobs, laser-sheet fall-off, noise
  and ghosts. Take only those two pieces, not the torch pipeline. Render one JHTDB case through our camera rig at the
  seeding density of wp2, run the whole chain (detection to Eulerian statistics) and compare eps, VSS and MKE with the
  values computed from the DNS trajectories directly.
- **Gate.** This is a measurement, not a feature. It gives the bias of each stage (detector, tracker, filter window) in
  the outputs users care about, and decides M2's setting.
- **Cost.** A few days. JHTDB access may need a token; the fallback is a synthetic turbulent flow with known statistics.

### M4. Refraction parameters in self-calibration

- **Why.** The leave-one-camera-out residual depends on depth: adding depth terms raises the explained variance from
  22-41 % to 45-54 % (`scripts/loco_map.py`). That is the signature of a multimedia error (glass thickness `mmp_d`,
  refractive indices). `joint_self_calibrate` frees the exterior, principal point and distortion, not those.
- **What.** Free `mmp_d` (and optionally `mmp_n2`) with priors, anchored by the known body (PR #45). Score by held-out
  leave-one-camera-out and by the body's triangulation error.
- **Gate.** Held-out leave-one-camera-out lower than the -17 % reached without refraction, and a smaller depth part in the
  residual field; the body error not higher.
- **Cost.** About a day.

### M5. Phase-dependent search bounds for the periodic jet

- **Why.** wp1/wp2 are a phase-averaged periodic jet, and our search bounds and the 2.5 m/s cap are global. Fast
  particles at peak phase are the likeliest to be lost or capped: a selection bias, and possibly the reason for the
  10-100x jet-region differences against legacy.
- **What.** `feature/dynamic-tracking-params` (2026-08, not merged) sets per-frame bounds from local drops in the link
  rate. First measure without any new code: yield and the share of capped points per phase bin on wp2. Only if the
  share at peak phase is clearly higher, port the per-frame bounds (or a per-phase cap) to `two_phase`.
- **Gate.** Yield at peak phase within a few points of the mean, or the dynamic bounds bring it there.
- **Cost.** Hours to measure; a day to port.

### M6. Truth-free physics checks on real data

- **Why.** On real data we use jump share, jitter and leave-one-camera-out only.
- **What.** Add two checks to `scripts/realism_metrics.py`: acceleration flatness (crossing swaps inflate it) and the
  small-lag Lagrangian structure function, whose intercept gives the position-noise variance. That variance is what eps
  needs for a noise correction, and it can be checked on M3's truth case.
- **Cost.** Hours.

## 3. Housekeeping found in the review

- **One scorecard for every branch.** Three agents worked on merged blobs in parallel and reported different metrics
  (velocity error 0.16 here, "vel 19.65" there). A claim of a gain needs the same cases and metrics, written to
  `bench/*.json` (the `synth_bench.py eval` output).
- **`hybrid_deltat_3d` was retired without a test in our regime** (5000 fps, steps of about 0.09 mm against 0.04 mm depth
  jitter, which is what it was built for). One run on the benchmark and on wp2 before its code is deleted.
- **Branches.**
  - Merge or close: the trackcorr candidate viewer (`feature/trackcorr-interactive-debug`, complete with tests and a GUI
    menu entry), the two-phase link lab (`feat/tracking-brain-viz`), and the detection crop notebook (with M1).
  - Close, keeping the measurements in this plan: VFC and affine consensus, `feat/merged-blob-split`,
    `feat/unified+merged-blob`, local `feat/two-phase-spatial-consensus-vfc`.
  - Deleting a branch needs the user's approval.
- **Retired tracker code** (`unified`, `4be`, `hybrid_deltat_3d`, splitter alias) is still in the tree; remove after the
  `hybrid_deltat_3d` check.

## 4. Order

1. M1 and M2 together (both cheap; either could outweigh all of the tracker work so far).
2. M3 (decides M2's setting and gives the bias of each stage).
3. M4, then M5 and M6.
4. Housekeeping alongside.

Rules as before: scratch copies of real data only; every number goes into this file with its case and command; a
step that gains only by dropping true points is reverted.

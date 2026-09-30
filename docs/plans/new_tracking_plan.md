│ A new tracker for openptv2: coherent tracklet fitting                            │
│                                                                                  │
│ Branch: feat/tracker-improvements. Benchmark: scripts/synth_bench.py on          │
│ ~/Downloads/HiDImaging/Synthetic-CompleteTest (work dir ~/Downloads/HiDImaging/tracker-bench-2026-09-29).                                │
│                                                                                  │
│ Context — what this branch measured                                              │
│                                                        │
│ Goal (user): after tracker + postptv repair + Savitzky-Golay, be closest to the  ue flow;                                                                       │
│ no ghosts; keep long trajty > speed.                   │
│                                                                                  nking (velocity error vs true flow, lower is better; oracle = perfect linker   │
│ on the same input)                                     │
│                                                                                  ──────────┬───────┬──────────┬──────────┬───────────┬──────┬────────────┬───── │
│ ─┐                                                     │
│ │           │       │          │ trackcor │ trackcorr │      │            │                                                                                     │
│ │   case    │ oracl │ two   │ 4be  │ nearest_hu │ gmm  │                                                                                │
│ │           │   e   │    co │      │     ng     │      │                                                                                │
│ │           │       │       │      │            │      │                                                                                │
│ ├───────────┼───────┼───────┼──────┼────────────┼───── │
│ ─┤                                                     │
│ │ native 15 │ 0.130 │ 0.1   │ 0.21 │ 0.223      │ 0.18 │
│ │                                                      │
│ │ 00/frame  │       │       │ 2    │            │ 8    │                                                                                │
│ ├───────────┼───────┼───────┼──────┼────────────┼───── │                                                                               │
│ │ clustered │       │          │          │ 0.183     │ 0.57 │            │ 0.67                                                                                │
│ │  ×2, skip │ 0.149 │ 0.1   │ 6    │ 0.290      │ 6    │                                                                                │
│ │  ×4       │       │       │      │            │      │                                                                               │
│ ├───────────┼───────┼──────────┼──────────┼───────────┼──────┼────────────┼─────                                                                                │
│ │ density   │ 0.535 │ 0.4   │ 0.96 │ 0.807      │ 0.91 │                                                                               │
│ │ ×4        │       │          │          │ (0.434)   │ 0    │            │ 7                                                                                   │
│ └───────────┴───────┴───────┴──────┴────────────┴───── │
│ ─┘                                                                                                                                                              │
│ Wall time (single process trackcorr port 25–32 s, C    │ boptv trackcorr 11 s, gmm 177 s.                                               │
│                                                        │
│ What the measurements sayssion)                        │                                                                               │
│ 1. The port of trackcorr is faithful. Link-by-link vs the original C (optv,       3dptv track.c):                                                               │
│    99.66–100 % identical nce rule and cost             │  rr=(dl/lmax+acc/dacc+angle/dangle)/quali identical.                           │
│    trackcorr's poor score is the algorithm in this regime, not a translation     │
│    bug. The C is 2–5× faster though.                                             │
│ 2. two_phase wins only because of two-hop confirmation (confirm_tol). Ablation:  │
│    leaf (2D) cost, velocity                                                  prediction, bidirectional merge each change <0.01; without confirmation       │
│    two_phase = priority_segment_3d.                                               Confirmation now exists tracker-agnostically as                               │
│    tracking_postprocess.c                              │ Gates sized by the fastest particle destroy the slow ones. Correct links move │
│    0.21 mm/frame (hard case);                                                     the gate was ±2.1 mm and dacc 1.04. dacc 0.25 alone: 0.51 → 0.18.             │
│    tracking_recommender._                              │
│    has the same flaw (dactiles).                       │ Ghosts are the dominant error and no tracker removes them. Input has ~8 %     │
│    ghost 3D points; every tracker                                                 outputs ~7 % ghost points, the oracle 0.3 % (it never links them). Of         │
│    trackcorr's 21 k wrong                              │ 13.8 k end in a ghost, 13.6 k start from one.                                 │
│ 5. The per-frame step is below the noise. Native: step 0.085 mm, depth σ 0.094    mm (in-plane 0.019 mm),                                                       │
│    NN spacing 2.06 mm. Frnobservable; smoothing        │  windows of 41–81 frames were optimal.                                         │
│ 6. Motion is extremely prent ways (measured on truth): │
│    - own history, constanon: error 0.000 mm (native),  │
│      0.001 mm (hard);                                  │  - 8 nearest neighbours' median step: 0.014 mm (native), 0.044 mm (hard) —     │
│      7–20× below the nois                              │   true successor is the nearest point to that prediction 99.6–100 % of the    │
│      time.                                                                          So candidate ambiguity comes only from noise/ghosts/dropouts, never from    │
│      kinematics.                                       │ Continuity is capped by dropouts, not linking. Per-frame recall 0.71; even    │
│    the oracle keeps only 57 %                                                     (native) / 19 % (×4) of long tracks whole with gap ≤ 3. Confirmation makes    │
│    continuity worse (3.6                               │  Caveat on 6: the truth flow is a Gaussian-smoothed 1.5 mm voxel field, so     │
│    sub-voxel coherence is                              │ construction. Real turbulence will be less coherent. Stage 0 below measures   │
│    it on real data first.                                                                                                                                       │
│ Essential steps of the exence)                         │                                                                               │
│ - trackcorr (algorithms/track.py, track_kernels_corr.py): linear predict from    prev→cur; project prediction                                                   │
│   to each camera; 2D box  = velocity bounds);          │
│   triangulate candidates incl. new 3D                                            points from unused targets (assess_new_position); cost over 3–4 frames (acc,   │
│   angle, path length, #ca                              │ greedy conflict resolution by cost; backward pass. Gate = kinematic bounds.    │
│   2D-target ownership (tn                              │
│   distinguishes used/freest machinery exists.          │priority_segment_3d / track3d: same, 3D only, 3 levels (seeded, unseeded,      │
│   cold-start NN).                                                                4be: 3D only; candidate scored by how well its own extrapolation predicts a    │
│   real particle at n+2.                                │two_phase (plugins/two_phase_tracking.py): constant-velocity predict; cKDTree  │
│   gate v_max; per connected                                                      component Hungarian on 2D reprojection cost (or 3D); two-hop confirm; survives │
│   max_gap frames; optiona                              │
│   bidirectional merge. Pure numpy/scipy, fastest.                                hybrid_deltat_3d: right idea (link over a stride where displacement ≫ noise),  │
│   wrong execution (Hungar                              │
│   on strided clouds with stride-scaled radii, then chains intermediates within a │
│   fixed gate → fragments in                                                      │
│   dense regions; weakest tracker measured).                                                                                                                │
│ The new thinking                                                                                                                                                │
│ Every current tracker ask1 is nearest to where this    │ int should go, within a gate                                                   │
│ sized by the fastest partt question is ill-posed: the  │
│ answer is inside the noise.                                                      │
│ Replace it with three ideas, each aimed at a measured error source:                                                                                        │
│ A. Noise-shaped, not velocity-shaped, gates (fixes 3, 5). Use a per-point        isotropic covariance                                                           │
│ (depth σ ≈ 5× in-plane, cy, or a fixed diag from a     │rm-up) and Mahalanobis                                                         │
│ distances everywhere a Euclidean distance is used (gate, cost, confirmation,     pair). With a good predictor the                                               │
│ gate becomes k·σ around t of the velocity bounds — the │ biguity volume shrinks ~25×                                                    │
│ and the 2D search box in ocity-sized to noise-sized    │ peed follows).                                                                 │
│                                                        │ A flow-coherent predictor (fixes 3, 6, and the clustered slow particles). Two │
│ passes per frame:                                                                ss 1 links the confident population (4-camera points, isolated, confirmed)     │
│ into tracklets; their vel                              │
│ define a local field (kNN median over ±W frames — effectively PIV from the       nfident tracks). Pass 2 predicts                                               │
│ every remaining particle from the field, not from its own 2-point noisy history, │
│ with own-history as fallback                                                     │
│ where the neighbourhood disagrees (spread of neighbour velocities > threshold =  │
│ shear layer). Measured                                                      verage: 0.014–0.044 mm prediction error vs 0.094 mm noise. This is also the    │
│ correct version of                                                               brid_deltat_3d: the "large Δt" is replaced by "many coherent neighbours × many │
│ frames".                                               │                                                                               │
│ C. Tracklets by window fitting + 2D-target ownership (fixes 4, 7).               A trajectory hypothesis is a constant-acceleration curve over W frames (W ≈    │
│   8–12 so displacement ≫                               │ accepted if the points lie within k·σ (Mahalanobis). Velocity/acceleration     │
│   come from the fit, not                               │ differences. Gap bridging then extends naturally to ~10 frames (predicted      │
│   uncertainty 0.03 mm/fra                              │ ≪ spacing), lifting the continuity ceiling above today's oracle.               │
│ - Ghost killing: every 2D one trajectory. A 3D point   │
│   whose targets are all already                                                  │
│   owned by confident tracklets is a ghost and is never allowed to seed or exa track. Two 3D points sharing a                                               │
│   target in one frame: keep the one that continues a fitted tracklet. This is    tracking-aware correspondence —                                                │
│   the information corresp Target: output ghosts 7 % →  │< 1–2 %.                                                                       │
│                                                                                  me for the plugin: coherent_tracklet (a.k.a. CTF).                             │
│                                                        │plementation plan (each stage measurable with the existing harness against the │
│ oracle)                                                                                                                                                         │
│ Stage 0 — validate the tw (½ day, no tracker code)     │
│                                                                                  Script scripts/measure_coherence.py: on the real CompleteTest run.zarr         │
│   trajectories (and on L1                              │ control), compute the own-history and 8-neighbour prediction errors of item 6, │
│   and the per-point depth                              │
│   noise from confirmed stratio prediction-error /      │ noise. Proceed with B only if the                                              │
│   neighbour prior beats t(it needs ≲ 0.5×).            │Fix tracking_recommender._suggest_params: derive dacc from second differences  │
│   of confirmed links                                                             (+3σ noise), not from displacement; keep the velocity window from              │
│   displacement. Re-run re                              │                                                                                │
│ Stage 1 — anisotropic gat ground, numpy) (1 day)       │                                                                               │
│ - plugins/two_phase_tracking.py: config noise_sigma=(σxy, σxy, σz) (or per-point covariance from                                                                │
│   store.read_corresponden); Mahalanobis in             │ _match_two_phase_frame (gate, 3D cost)                                         │
│   and in confirm_link_tuppy). Estimate σ automatically │ from a warm-up pass                                                            │
│   (residuals of confidentng_warmup.py machinery.       │ Harness: two_phase+noise_sigma=... overrides; compare to oracle on the 3       │
│   reference cases.                                     │                                                                               │
│ Stage 2 — field predictor (pass 1 / pass 2) in a new plugin (2–3 days)                                                                                          │
│ - New plugins/coherent_trhase's pieces                 │(_match_two_phase_frame, chains, links →                                       │
│   linkage via _links_to_linkage). Pass 1 = two_phase with tight gate on 4-campoints + confirmation.                                                         │
│   Field = cKDTree kNN median of pass-1 tracklet velocities per frame (±W). Pass 2 = two_phase matcher with                                                     │
│   pred = pos + field(pos) and fallback to own history when neighbour spread >   threshold.                                                                     │
│ - Register in tracking_registry.py, tracking_presets.TRACKER_CHOICES,            plugins/loader.py pattern.                                                     │
│ - Measure: clustered case (slow particles in clusters are the failing            │
│   population) and ×4.                                                            │
│                                                                                  │
│ Stage 3 — window fitting, long gap bridging, ghost ownership (3–4 days)                                                                                    │
│ - Tracklet fit (least squares, constant acceleration, Mahalanobis residual) used for: velocity for the field,                                                   │
│   acceptance test replaciap bridging up to max_gap≈10. │2D-target ownership table per frame (cam_ids from correspondences already      │
│   loaded in two_phase's                                                          do_tracking); ghost rule as in C above. Measure ghost_point_frac vs oracle.    │
│ - Rejoin step: postptv rer-cuts because its            │
│   self-calibration sees the depth noise;                                        feed it the fitted tracklets (smoother) or per-axis σ — measure                │
│   long_fragments.                                                                                                                                               │
│ Stage 4 — bring the wins to trackcorr (Cython) (2–3 days)                        │
│                                                                                  │
│ - track_kernels_corr.py: predictor hook (field prediction array per particle,    │
│   like the existing gate_scale                                              per-particle array), noise-sized search box, Mahalanobis in                    │
│   _angle_acc_out-based cost. Keep the C-parity                                   path bit-identical when the hook is absent (existing parity check:             │
│   scripts/synth_bench.py                               │
│ - Speed: profile with scripts/prof_trackcorr.py; target ≤ C time (11 s native).  │
│   The noise-sized box alone                                                 should remove the 85 s blow-up at skip ×4.                                     │
│                                                                                  les touched                                                                    │
│                                                        │
│ - src/openptv2/plugins/tw                              │ src/openptv2/plugins/coherent_tracklet.py                                      │
│ - src/openptv2/tracking_p confirm),                    │
│   src/openptv2/tracking_r                              │ src/openptv2/algorithms/track_kernels_corr.py, track.py (stage 4; Cython       │
│   rebuild required)                                    │src/openptv2/tracking_registry.py, tracking_presets.py                         │
│ - scripts/measure_coherence.py, scripts/synth_bench.py (new tracker labels)      tests: tests/unit/test_two_phase_tracking.py, test_tracking_postprocess.py,    │
│   new test_coherent_track                              │                                                                               │
│ Verification                                                                                                                                                   │
│ - Unit: uv run pytest tests/unit/test_two_phase_tracking.py                     tests/unit/test_tracking_postprocess.py tests/unit/test_track.py -q            │
│ - Parity guard for trackcorr changes: optv_c vs trackcorr@fwd link sets on       L1_d0.3_c0_k1_n12 stay ≥ 99.5 % identical with hooks off.                      │
│ - Benchmark after each stage (bash script, -j 4):                                │
│   scripts/synth_bench.py sweep --cases L1_d1_c0_k1_n200 L1_d2_c0.3_k4_n150       │
│   L1_d4_c0_k1_n150 --trackers oracle two_phase coherent_tracklet trackcorr ...   │
│   then eval/report; 3D spot checks with scripts/viz_compare_3d.py.          Success targets vs today: native vel error 0.178 → ≤ 0.145 (oracle 0.130);     │
│   ghost points 7 % → < 2 %;                                                      long-track completeness ≥ oracle's (0.57 native) with gaps ≤ 10; ×4 density ≤  │
│   0.40; trackcorr time ≤                               │
│ - Final check on the realh): confirmation-consistency  │and acceleration kurtosis                                                      │
│   (openptv2.benchmarking.metrics.compute_physics_metrics) must improve, not just the synthetic scores.                                                          │
│                                                        │sks                                                                            │
│                                                                                  Coherence on real data may be weaker than on the smoothed synthetic field      │
│   (stage 0 decides B's we                              │
│ - Field prior can drag particles across shear layers → the disagreement fallback is mandatory, and the clustered                                                │
│   case (injector blobs in test for it.                 │
│ - Ghost ownership rule caose targets were stolen by    │ghosts; measure yield a

│ SESSION HANDOFF 2026-09-30 — detailed build order (start here next session) │
│ Branch survey: `coherent_tracklet`, `measure_coherence`, tracking-side      │
│ `noise_sigma` exist on NO branch (checked all local + origin branches; the  │
│ only `noise_sigma` hits are pre-existing calibration-GUI files). Build fresh│
│ on feat/tracker-improvements. Working branch for viz experiments is         │
│ feat/tracking-brain-viz (separate project tracking-brain-viz/, do not mix). │
│                                                                             │
│ Repo state: EXISTS scripts/synth_bench.py, tracking_postprocess.py          │
│ (confirm_link_tuples), tracking_warmup.py, tracking_recommender.py,         │
│ plugins/two_phase_tracking.py (TwoPhaseTrackerConfig has NO noise_sigma).   │
│ MISSING scripts/measure_coherence.py, plugins/coherent_tracklet.py.         │
│                                                                             │
│ STAGE 0 (start here, 1/2 day, no tracker code):                             │
│  0.1 Write scripts/measure_coherence.py: inputs = CompleteTest run.zarr    │
│      trajectories + one L1 control case. Outputs per particle: (a) own-     │
│      history err |x(t+1)-(2x(t)-x(t-1))|, (b) 8-neighbour median-step err,  │
│      (c) depth noise sigma from confirmed straight tracks. Print the       │
│      ratios from plan item 6.                                              │
│  0.2 Fix tracking_recommender._suggest_params: dacc from 2nd differences   │
│      of confirmed links +3σ noise (not from displacement); velocity window │
│      stays displacement-based. Re-run hard case, expect 0.51 → ~0.18.       │
│  DONE = neighbour prior beats own-history by ≲0.5× (else skip Stage 2's B). │
│                                                                             │
│ STAGE 1 (1 day):                                                            │
│  1.1 TwoPhaseTrackerConfig.noise_sigma=(sxy, sxy, sz), default None (=      │
│      legacy Euclidean). Auto-estimate path via tracking_warmup.py residuals.│
│  1.2 Mahalanobis in _match_two_phase_frame (gate radius + 3D cost branch)  │
│      and in tracking_postprocess.confirm_link_tuples.                       │
│  1.3 Harness: two_phase+noise_sigma vs oracle on 3 reference cases.        │
│  DONE = existing tests green (test_two_phase_tracking,                     │
│      test_tracking_postprocess) + measurable gate-shrink win.               │
│                                                                             │
│ STAGE 2 (2-3 days): new src/openptv2/plugins/coherent_tracklet.py reusing  │
│  _match_two_phase_frame, _chains_from_links, _links_to_linkage from        │
│  plugins/two_phase_tracking.py. Pass 1 = two_phase, tight gate, 4-cam      │
│  points only + confirmation. Field = per-frame cKDTree kNN median of       │
│  pass-1 tracklet velocities (±W frames). Pass 2 = same matcher with        │
│  pred = pos + field(pos); fallback to own history when neighbour-velocity  │
│  spread > threshold (shear guard). Register in tracking_registry.py,       │
│  tracking_presets.TRACKER_CHOICES, plugins/loader.py pattern. New          │
│  tests/unit/test_coherent_tracklet.py (synthetic crossing + gap cases).    │
│  DONE = clustered + x4 cases improve vs Stage 1.                           │
│                                                                             │
│ STAGE 3 (3-4 days): least-squares const-accel fit over W=8-12 as           │
│  acceptance test; velocity/accel from fit; max_gap≈10 bridging; per-frame  │
│  2D-target ownership table from correspondences cam_ids (ghost rule: all   │
│  targets owned by confident tracklets → never seed/extend; shared target → │
│  keep the tracklet-continuing point); feed fitted tracklets to postptv.    │
│  DONE = ghost points 7% → <2%, long-track completeness ≥ oracle 0.57.      │
│                                                                             │
│ STAGE 4 (2-3 days): predictor hook + noise-sized box + Mahalanobis cost    │
│  in algorithms/track_kernels_corr.py; C-parity bit-identical with hook off │
│  (synth_bench L1_d0.3 ≥99.5%); profile scripts/prof_trackcorr.py ≤11 s.    │
│                                                                             │
│ After EVERY stage: unit tests → synth_bench.py sweep (L1_d1_c0_k1_n200,    │
│  L1_d2_c0.3_k4_n150, L1_d4_c0_k1_n150; trackers oracle two_phase           │
│  coherent_tracklet trackcorr) → compute_physics_metrics must improve.       │

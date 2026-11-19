# Agent Memory: ILP/Temporal-Smoothing Tracker Work

Branch: `feat/ilp-temporal-smoothing-impl` (active, pushed to origin)
Base: `feat/unified-tracker` (615be372) + cherry-picked `feat/tracker-improvements` (02912aca)

## What was built

### Core modules (src/openptv2/)
- `ghost_components.py` — union-find conflict graph decomposition
- `ghost_ilp.py` — ILP uniqueness (MILP per component, Hungarian for bipartite), conservative auto-gate (ghost_rate>0.4)
- `temporal_smoothing.py` — IRLS smoothing, residual + velocity-stats ghost filter, trajectory-level ILP, point-level smoothing, extract_trajectories, trajectory_level_ghost_filter
- `trajectory_stitch.py` — merge fragments + split kinks (smoothness-checked)
- `mht_linker.py` — sliding-window MHT (auto-disabled, over-merges)
- `epipolar_rerank.py` — multi-D soft scoring (rcm-normalized)
- `calibration_aware.py` — rcm normalization, calibration-aware thresholds, multi-cycle consistency
- `frame_quality.py` — skip/interpolate low-correspondence frames

### Bug fixes
- `default_tracking.py`: trackcorr now runs bidirectional + postprocess (was forward-only, no postprocess)
- `tracking_presets.py`: added trackcorr to _DIRECTION_BACKWARD_PRESETS
- `scenario.py`: velocity model fixed (random unit vector instead of cos(th) degeneracy)

### Unified tracker changes
- Engine selection: dense (ghost_rate>0.2 or density≥500) → 3D engine (priority_segment_3d); sparse → epipolar (trackcorr)
- Per-frame ILP removed (trajectory-level only, preserves continuity)
- Trajectory-level ghost filter + stitch + (disabled) MHT as post-passes
- max_gap tuning, corrective_passes=2 by default
- vpar guard in peeling (prevents crash on mock exps)

### Hybrid trackcorr
- trackcorr now uses 3D priority cascade + epipolar soft scoring (w_epi, epi_scores in Cython kernel)
- Cython-level B6 implemented (kernel accepts epi_scores, Python wrapper computes from store rcm)
- .so rebuilt successfully

### Quality criterion
- `tracking_report.py`: length_score saturates at target (GT) length; length_match_score added
- `compute_trajectory_quality_scores` accepts target_length param

### Benchmarks
- test_unified_gt_scorecard: 3 regimes × 3 trackers (clean/slow_noisy/dense_ghost)
- test_ilp_ghost_suppression: 12 tests (precision/recall at various densities)
- test_temporal_smoothing_ghosts: 5 tests
- test_realistic_density: 2 tests (moderate + low density)
- test_ablation: 2 tests (peeling/smoothing toggles)
- test_tracker_comprehensive: 8 tests
- All 30 benchmarks pass

### Key findings
1. **priority_segment_3d is genuinely better** for dense scenes: 4-level priority cascade (cold-start + gap bridging) recovers more links than trackcorr's epipolar search.
2. **trackcorr was running forward-only with no postprocess** (bug fixed) — but even after fix, priority_segment_3d wins on dense scenes.
3. **Fragmentation (med_len 1.0) is inherent** to dense scenes. Stitching helps (mean 4.7→6.8) but doesn't fix root cause.
4. **MHT over-merges** — auto-disabled pending tuning.
5. **Velocity-stats ghost filter** (vel_mean ~1.0, vel_std ~0.43 for real) derived from good/bad track analysis. GT vel_std=0.432 exactly matches good tracks.
6. **Test case bug**: dense_ghost (seed=12) had all-zero velocities (cos(th) degeneracy). Fixed velocity model.

### Real-data results (lv_multi wp4, before corruption)
- trackcorr: 8648 tracks, med 1.0, mean 2.76
- priority_segment_3d: 2028 tracks, med 16.0, mean 11.76
- unified: 5720 tracks, med 2.0, mean 4.21

### Best combination for optimal performance
Engine: priority_segment_3d for dense/ghosty, trackcorr for clean/sparse.
Post-process: gap bridging (max_gap=2) + cold start + reciprocity + corrective pass (2 passes).
Ghost suppression: ILP (conservative, trajectory-level) + temporal smoothing (auto-gated) + stitching (smoothness-checked).
Quality: length_match_score + per-trajectory quality scores with prio persistence.
MHT: available but auto-disabled (over-merges).

### Remaining work
- B3 frame-quality: ✅ done
- B6 Cython-level: ✅ kernel ready, wrapper computes epi_scores
- Idea 4/5: ✅ modules created, partially wired
- H2 CI gate: separate repo (openptv-cloud)
- J3 notebooks: separate repo
- Open questions §13: need user decisions
- Factor graphs/sparse recovery: research only

### Tests
- 50 unit tests pass
- 30 benchmark tests pass
- 9 pre-existing failures (flowtracks import, cython compiled-vs-interpreted) are environment-related
# openptv2 — Software Map: Tracking Algorithms, Implementations, Branches

## 1. Tracking plugins (src/openptv2/plugins/)

| Plugin | Engine | Mechanism | Strengths | Weaknesses |
|---|---|---|---|---|
| `unified` (default) | Auto-selects | Wraps trackcorr or priority_segment_3d based on density/ghost_rate; adds ILP, temporal smoothing, stitching, MHT, quality scoring | Adaptive, auditable, ghost-suppressing | Slower (0.8–1.8s); post-passes can hurt yield if over-tuned |
| `cython_epipolar_tracking` (trackcorr) | Epipolar (Cython) | 2D epipolar search → triangulate → 3D nearest-neighbor link | Fast (0.35s), high precision (99.5%), bidirectional+postprocess now default | Fragments in dense scenes (epipolar ambiguity rejects valid links) |
| `cython_3d_tracking` (priority_segment_3d) | 3D priority cascade (Cython) | 4-level priority segment linking in 3D space; cold-start gating; gap bridging | Highest yield (1.0) and purity (1.0); full-length tracks; robust to 2D ambiguity | No epipolar verification; can link spatially-close ghosts |
| `four_be_tracking` | 4-frame best estimate | 4BE cost with n+2 lookahead tie-breaker | Good for ambiguous candidates | Slower, niche |
| `hybrid_deltat_3d` | Coarse-to-fine Hungarian | Links every N-th frame (coarse) then refines intermediate frames | Slow flows at high FPS where Δt=1 is noise-dominated | Complex, slower |
| `two_phase_tracking` | 2-phase (coarse+fine) | Coarse 3D prediction + linear assignment + 2D leaf re-ranking | Configurable cost weights; ambiguous 3D disambiguation | Slower, complex |
| `splitter_tracking` | 4-view splitter hardware | Hardware-specific splitter rig | Niche hardware | Not general |

## 2. Anti-ghost modules (src/openptv2/)

| Module | Purpose | Status |
|---|---|---|
| `ghost_filter.py` | Early rcm-based ghost probability pre-filter | ✅ |
| `ghost_components.py` | Conflict graph decomposition (union-find) | ✅ |
| `ghost_ilp.py` | ILP uniqueness (MILP per component, Hungarian for bipartite) | ✅ |
| `layered_tracking.py` | Clean-and-Claim (tight core → 2D-target pruning → residual) | ✅ |
| `track_assisted.py` | Polynomial predictor, subset triangulation, shared targets | ✅ (upgraded) |
| `temporal_smoothing.py` | IRLS trajectory smoothing, residual + velocity-stats ghost filter, trajectory-level ILP | ✅ |
| `trajectory_stitch.py` | Merge fragments + split kinks (multi-frame repair) | ✅ |
| `mht_linker.py` | Sliding-window MHT (Hungarian over windows) | ✅ (auto-disabled pending tuning) |
| `epipolar_rerank.py` | Multi-D soft scoring (epipolar consistency) | ✅ |
| `calibration_aware.py` | rcm normalization, calibration-aware thresholds, multi-cycle consistency | ✅ |
| `frame_quality.py` | Skip/interpolate low-correspondence frames | ✅ |

## 3. Quality & reporting (src/openptv2/)

| Module | Purpose |
|---|---|
| `tracking_report.py` | Self-consistency metrics, per-trajectory quality scores (length, reciprocity, smoothness, gap), `length_match_score`, `prio` persistence |
| `tracking_decision.py` | Auditable `TrackingDecision` JSON record |
| `tracking_probe.py` | Multi-window probe + coordinate descent |
| `tracking_estimator.py` | Auto-parameter estimation (v_max, density, noise, ghost_rate, conditioning ratio) |

## 4. Branches

| Branch | Focus | Status |
|---|---|---|
| `feat/unified-tracker` | Unified tracker (Waves 0–5), backbone improvements, clean-and-claim | Merged into ilp branch |
| `feat/ilp-temporal-smoothing-impl` | ILP + temporal smoothing + stitching + MHT + benchmarks + quality criterion + skills | **Current (active)** |
| `feat/tracker-improvements` | Ghost elimination hardening, gap-bridge compaction, kink pruning | Cherry-picked into ilp branch |
| `main` | Stable release | Base |

## 5. Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `dvxmin/dvxmax/...` | ±10 mm | Velocity search box. Too small → fragments. Too large → ghosts. |
| `dacc` | 5.5–10 | Acceleration tolerance for gap bridging. |
| `max_gap` | 2 frames | Gap bridging length. Raise for occlusions. |
| `eps0` | 0.1–1.0 mm | Epipolar tolerance for correspondence. |
| `ilp_uniqueness` | auto | Spatial ghost suppression (ILP). Conservative by default. |
| `temporal_smoothing` | auto | Trajectory-level ghost filter (residual + velocity-stats). |
| `peeling` | auto | Clean-and-Claim (tight core → prune → residual). |
| `stitch` | auto | Trajectory stitching (merge/split). |
| `mht` | off (auto-disabled) | Sliding-window MHT (over-merges; needs tuning). |
| `probe` | auto | Multi-window parameter probe. |
| `corrective_passes` | 2 | track_assisted extension passes. |
| `postprocess` | True | Gap relinking + cold start + reciprocity. |

## 6. Optimal combination strategy (accuracy + precision + length + minimal ghosts)

### Recommended pipeline for a new dataset

```
1. Calibrate (openptv-multiplane-calibrate skill)
   └─ Verify: rcm, reprojection RMS, camera coverage

2. Detect targets (targ_rec)
   └─ One blob per particle, stable counts

3. Correspondences (criteria, eps0, X_lay/Z_lay)
   └─ Maximize 4-camera share, minimize rcm

4. Choose engine (unified auto-selects):
   ├─ Sparse/clean (density < 100, ghost_rate < 0.2): trackcorr (epipolar)
   │   └─ Fast (0.35s), high precision, good for clean flows
   └─ Dense/ghosty (density ≥ 100 or ghost_rate > 0.2): priority_segment_3d (3D cascade)
       └─ Higher yield, full-length tracks, robust to 2D ambiguity

5. Post-processing (auto-enabled):
   ├─ Gap bridging (max_gap=2, dacc)
   ├─ Cold-start seeding (first-frame recovery)
   ├─ Reciprocity enforcement (bidirectional)
   └─ Corrective pass (track_assisted, 2 passes, polynomial predictor)

6. Ghost suppression (auto-enabled for dense):
   ├─ ILP uniqueness (trajectory-level, conservative)
   ├─ Temporal smoothing (residual + velocity-stats filter)
   └─ Stitching (merge fragments + split kinks, smoothness-checked)

7. Quality scoring (compute_trajectory_quality_scores)
   ├─ length_score (saturates at GT length)
   ├─ reciprocity fraction
   ├─ smoothness (kink-free)
   └─ gap penalty
   → write_prio for downstream weighting

8. Audit (TrackingDecision JSON)
   └─ Resolved config, probe metrics, conditioning ratio, git version
```

### Best tracker by scenario (from benchmarks)

| Scenario | Best tracker | Yield | Precision | MedLen | Purity | Time |
|---|---|---|---|---|---|---|
| Clean, sparse | trackcorr | 1.000 | 1.000 | 10.0 | 1.000 | 0.35s |
| Slow/noisy | priority_segment_3d | 0.517 | 0.517 | 10.0 | 1.000 | 0.42s |
| Dense/ghosty | priority_segment_3d | 1.000 | 0.997 | 10.0* | 1.000 | 0.43s |
| Speed-critical | trackcorr | 0.611 | 0.995 | 1.0 | 0.997 | 0.37s |

*priority_segment_3d produces full-length tracks in clean/slow scenes; fragments in dense (med_len 1.0) but still recovers all particles as full tracks in some cases.

### The optimal combination (what unified should do)

1. **Engine**: priority_segment_3d for dense/ghosty (better yield + length), trackcorr for clean/sparse (faster).
2. **Gap bridging**: max_gap=2, dacc tuned to flow. This is the single biggest lever for track length.
3. **Cold-start seeding**: recovers first-frame links using established velocity field.
4. **Reciprocity**: enforces forward-backward agreement, killing ghost links.
5. **Corrective pass**: track_assisted with polynomial predictor extends tracks.
6. **ILP (conservative)**: only when ghost_rate > 0.2. Removes target-conflicting ghosts.
7. **Temporal smoothing**: residual-based ghost drop. Only when slow_flow or high jump_rate.
8. **Stitching (smoothness-checked)**: merges fragments across gaps, splits at kinks. Primary repair for fragmentation.
9. **Quality scoring**: weight downstream analysis by track quality.
10. **MHT (disabled by default)**: available for experimentation; over-merges at current tuning.

### What NOT to do
- Don't run ILP per-frame (breaks continuity). Trajectory-level only.
- Don't use wide dv (±12) — creates ghost links. Stick to ±3-6× measured displacement.
- Don't enable MHT without tuning (over-merges).
- Don't skip gap bridging (max_gap=2 is the biggest length lever).

### Remaining work for optimal performance
1. **Multi-frame MHT tuning**: better cost function (epipolar + velocity + residual) to merge without over-merging.
2. **Cython-level B6**: push epipolar scores into track3d kernel (faster, same algorithm).
3. **B3 frame-quality**: skip/interpolate low-correspondence frames.
4. **Calibration-aware thresholds**: normalize rcm by distance from centre.
5. **Multi-cycle epipolar**: run search with different camera orderings.
6. **openptv-cloud integration**: H2 CI gate, J3 notebooks.
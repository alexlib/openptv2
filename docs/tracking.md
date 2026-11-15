# Unified Tracker (`track.*` Configuration & Architecture)

The `unified` tracker (`selected_tracking: unified`) replaces the legacy single-mode presets with one self-configuring, auditable pipeline (`Estimator -> Engine -> Refiner`) over `RunStore`.

## Target-Space Boundary
OpenPTV2 intentionally operates in **target space** (`detect -> correspond -> track`) without Optical Transfer Function (OTF) image subtraction. For real-world non-spherical, polydisperse, or bubble tracers, high-density performance comes from:
- Optional multi-peak 2D blob splitting (`multipeak=True` in `segmentation.targ_rec`)
- 2-Pass Clean-and-Claim (`peeling: auto|true`) with deterministic 2D-target pruning
- Upgraded `track_assisted` (3–4 point polynomial predictor, 2-of-4 / 3-of-4 camera subsets, opt-in `allow_shared_targets`)
- Conflict-component Hungarian assignment (`resolve_conflict_components`)

---

## `track.*` YAML Reference

| Key | Type | Default | Derivation / Behavior | When to Override | Accuracy / Speed Impact |
|---|---|---|---|---|---|
| `v_max` | `auto \| float` | `"auto"` | `measure_motion_scale` × `v_max_safety` (`1.5`), capped by mean inter-particle spacing when `rho > 1` | Known max displacement (mm/frame) | Too small drops fast tracks; too large increases ambiguity |
| `v_max_safety` | `float` | `1.5` | Multiplier on measured motion scale | Turbulent/accelerating flows (`2.0`) | Higher = safer recall, slightly more candidates |
| `bidirectional` | `auto \| bool` | `"auto"` | `True` unless `density > 5000` | `False` for speed-critical runs | `True` + reciprocity cuts false links; ~2× tracking time |
| `use_2d` | `auto \| bool` | `"auto"` | `True` unless `density < 500` (and conditioning >= 0.5) | `False` for 3D-only / fast mode | `True` uses epipolar `trackcorr`; `False` uses fast 3D-only |
| `delta_t` | `auto \| int` | `"auto"` | `5` if `slow_flow` (`conditioning_ratio < 1.0`), else `1` | High-FPS slow flows (`3..10`) | `>1` beats z-noise floor via `hybrid_deltat_3d` |
| `slow_flow` | `auto \| bool` | `"auto"` | `True` when `displacement / noise_floor_z < 1.0` | Force coarse-to-fine tracking | Prevents noise-dominated velocity jitter |
| `confirm_link` | `auto \| float \| null` | `"auto"` | `None` (disabled; reciprocity guards precision) | Low-noise linear flows (`0.3..0.8` mm) | Cuts kinks, but too tight severs noisy/curved tracks |
| `postprocess` | `bool` | `true` | Runs `cold_start`, `gap_relinking`, `enforce_reciprocity` | `false` for raw single-pass debug | Improves yield & precision |
| `probe` | `auto \| bool` | `false` | Runs isolated sub-window probe when `true` or `"auto"` (`>=60` frames) | Long/non-stationary runs (`true`) | Finds better `v_max`/`delta_t` before full run; adds short probe time |
| `probe_windows` | `int` | `5` | Evenly spaced windows across `[first, last]` | Very long runs (`7..10`) | Captures regime drift |
| `probe_window_frames` | `int` | `30` | Frames per probe window | Short sequences (`15..20`) | Linear in window length |
| `probe_confirm` | `auto \| bool` | `"auto"` | Prompts only when `isatty()`; batch/CI auto-picks best | `false` in automated pipelines | Interactive safety check |
| `probe_budget_seconds` | `float` | `30.0` | Wall-clock cap for coordinate-descent probe | Tight CI (`5.0`) | Bounds probe overhead |
| `peeling` | `auto \| bool` | `false` | `"auto"` enables 2-Pass Clean-and-Claim when `ghost_rate > 0.25` | Dense ghosted scenes (`true`) | Collapses ghosts via 2D-target pruning + `track_assisted` |
| `ghost_filter` | `auto \| bool` | `"auto"` | Drops `< ghost_min_cameras` correspondences before tracking | Keep 2-cam points (`false`) | Cuts ghosts on 4-cam rigs |
| `ghost_min_cameras` | `int` | `3` | Minimum camera support for `ghost_filter` | 2-cam rigs (`2`) | Higher = purer core |
| `allow_shared_targets` | `bool` | `false` | Opt-in non-exclusive 2D target fallback in `track_assisted` | Heavy 2D blob overlap (`true`) | Recovers partial overlaps; can lower precision in high-ghost data |
| `parallel_workers` | `auto \| int` | `0` | `0` sequential; `>1` or `"auto"` (`>=24` frames) uses `track_sequence_chunked_parallel` | Large sequences (`4..8`) | Parallel speedup with identical `TrackingDecision` |
| `density_estimate` | `float` | `0` | Optional particles/frame hint (`0` = estimate from store) | Known density | Guides `use_2d` / `bidirectional` heuristics |

---

## Auditability (`tracking_decision.json`)
Every `unified` run with a `RunStore` saves a `TrackingDecision` to `<store>/tracking_decision.json` and `store.root["meta"].attrs["tracking_decision"]`, recording:
- `resolved_config`, `regime_flags`, `conditioning_ratio`
- `probe_metrics` and `grid_search_table` (when probed)
- `chosen_config` and post-run `report_metrics` (reciprocity, length percentiles, jump-step rate, ghost estimate, anomaly flags)

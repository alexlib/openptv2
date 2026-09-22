# Two-Phase Tracking: usage and parameters

Two-Phase (`selected_tracking: two_phase`,
`src/openptv2/plugins/two_phase_tracking.py`) links particles in two steps:
**Phase 1** finds candidates with a 3D KD-tree around each track's predicted
position; **Phase 2** ranks them by per-camera 2D image distance and solves a
Hungarian assignment per connected group. 3D proposes, the images dispose.

## 1. How to use it

**GUI:** Plugins page → tracking plugin → `two_phase`. Parameters live in
the `track` section (same names as below).

**Batch/YAML:** minimal setup —
```yaml
plugins:
  selected_tracking: two_phase
track:
  v_max: 5.0        # mm/frame; or dvxmax, same meaning here
  leaf_weight: 1.0
  max_gap: 2
```

**Plain Python** (no experiment needed):
```python
import numpy as np
from openptv2.plugins.two_phase_tracking import (
    TwoPhaseTracker, TwoPhaseTrackerConfig)

cfg = TwoPhaseTrackerConfig(v_max=5.0, leaf_weight=1.0, max_gap=2)
links = TwoPhaseTracker(cfg).track_frames(
    frame_particles,   # list of (N_i, 3) arrays, mm
    frame_leaves,      # list of (N_i, 2*C) arrays, px (optional)
    project_fn,        # (N,3) -> (N,2*C) re-projection (optional)
)
# links: list of (t0, row0, t1, row1); use return_chains=True to also get
# per-track histories: links, chains = ...track_frames(..., return_chains=True)
```

Without leaves/`project_fn` it falls back to pure 3D distance costs
(`cost_mode="3d"`); pass `leaf_weight=0` to force that explicitly.

## 2. Parameters

All live in the `track` YAML section (batch/GUI) or on
`TwoPhaseTrackerConfig` (Python). Units in brackets.

| parameter | default | what it does | how to set it |
|---|---|---|---|
| `v_max` (or `dvxmax`) [mm/frame] | 15.5 | 3D search radius around each prediction | ~3× your typical per-frame step. Too small: true links never become candidates. Too big: everything connects into giant groups (slow, sloppy) |
| `leaf_weight` [–] | 1.0 | weight of 2D image distance in the cost | 1.0 normally; 0 = pure 3D (no leaves needed). Lower it when calibration is poor — bad projection poisons the ranking |
| `use_velocity` [bool] | true | match predictions (`pos + vel·dt`), not positions | Keep on. Off = every crossing resolves as a bounce |
| `cost_mode` [str] | `projected` | `projected`: rank by re-projected 2D distance (needs `project_fn`/cals); `3d`: rank by 3D distance | `projected` with good calibration, `3d` otherwise |
| `max_gap` [frames] | 2 | a track survives this many unmatched frames | 2 covers single-frame dropouts (the biggest measured failure source). Higher = longer bridges, more impostors |
| `dt` [–] | 1.0 | time step for velocity | 1.0 for consecutive frames |
| `max_group_size` [nodes] | 128 | groups bigger than this skip the cubic Hungarian, greedy inside | Raise only if you can afford it; at production density frames percolate and this cap is what keeps a run from stalling |
| `allow_shared` [bool] | false | **prototype:** in groups with more tracks than detections (occlusion), losers share the winner's detection instead of dying | Enable where occlusions matter; validated on synthetic crossings (0 switches). Shared points move position but never velocity |
| `max_shared` [frames] | 2 | max consecutive shared frames per track | 2 covers brief overlaps; higher risks twin tracks that never separate |
| `share_tol` [cost] | 1.0 | a shared claim needs an edge cost below this (mutual-prediction gate) | Without it, a stranded track hijacks strangers' detections (observed live). ~5–10× your position noise; `null` disables the gate (not advised) |

## 3. How it behaves (caveats)

- **No motion model to be wrong** — but also none to help: if particles move
  farther per frame than the typical spacing, every tracker fails, this one
  first. Check step-vs-spacing before blaming parameters.
- **Gap survival is prediction-based:** a track coasts on `pos + vel`
  through gaps up to `max_gap`; a maneuver inside the gap is lost. That is
  by design — see `max_gap` above.
- **Occlusions:** with `allow_shared`, one detection may serve two tracks
  for up to `max_shared` frames (marked in chains when
  `return_chains=True`). Owners of the other engines: this is the reference
  implementation of the shared-observation rule — same idea ports to
  `track3d_loop_fast` (recorded, not claimed) and to linkage postprocess
  (mark, then bridge).
- **Speed:** ~linear in particles (KD-tree search, small per-group
  Hungarians); the `max_group_size` cap bounds the worst case.

## 4. Tuning recipe

1. Measure your typical step `s` (median linked displacement) and 3D noise
   `n` (second-difference statistics — see `docs/algorithms/tracking.md`).
2. `v_max` ≈ 3·s. `leaf_weight` = 1 with decent calibration, 0 without.
3. Run; count short tracks (cold starts) vs gaps. More gaps than tracks →
   raise `max_gap` to 3. More fragments at crossings → enable
   `allow_shared`.
4. Validate on synthetic ground truth with the same spacing/noise before
   trusting a production run (`tests/helpers/synthetic_scene.py` generates
   scenes; `scripts/proto_shared_validate.py` shows the scoring pattern).

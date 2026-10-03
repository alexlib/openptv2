# The two_phase tracker

Source: `src/openptv2/plugins/two_phase_tracking.py`.
Select it with `selected_tracking: two_phase` (GUI: Plugins page, tracking plugin).
All parameters are in the `track` section of the YAML file.
This document uses the ASD-STE100 style (Simplified Technical English).

## 1. Purpose

The tracker links 3D particle positions from frame to frame. The result is a set of trajectories.

The tracker uses two phases to find each link:

- **Phase 1** uses the 3D positions to find candidates.
- **Phase 2** uses the camera images to choose the best candidate.

In short, the 3D data proposes and the images decide.

## 2. Terms

| Term | Meaning |
|---|---|
| Track | A chain of particle points that the tracker follows in time. |
| Detection | A 3D point in a new frame. |
| Prediction | The expected position of a track in the new frame. |
| Candidate | A detection that is inside the search radius of a prediction. |
| Leaf | The 2D pixel position of a point in one camera. |
| Gap | A frame in which a track has no matching detection. |
| Cost | A number. A low cost means a good match. |
| Ghost | A false 3D point. Camera rays cross there, but no real particle exists. |

## 3. Overview of one step

For each pair of frames (t and t+1), the tracker does these steps:

1. Predict where each active track will be.
2. **Phase 1:** find the candidates around each prediction in 3D.
3. **Phase 2:** calculate the cost of each prediction–candidate pair from the 2D images.
4. Solve the assignment so that each detection is used once.
5. Update the tracks. Start new tracks from unused detections.

After the last frame, the tracker removes doubtful links (the confirmation step, section 8).

## 4. Prediction

The tracker keeps a position and a velocity for each track.

The prediction is: `position + velocity × steps × dt`. The `steps` value is the number of frames since the track was last seen.

- A new track has zero velocity (a "cold start").
- The tracker updates the velocity after each match: `(new position − old position) / (steps × dt)`.
- If `use_velocity` is off, the prediction equals the last position. Then every crossing of two tracks becomes a bounce. Keep this option on.

## 5. Phase 1: the 3D search

The tracker builds a KD-tree (`cKDTree`) of the detections in frame t+1. It then finds all detections within `v_max` mm of each prediction.

- A true link must be a candidate. If `v_max` is too small, the true link never becomes a candidate.
- If `v_max` is too big, many particles connect to each other. The groups become large and the tracker becomes slow.
- Tip: set `v_max` to about 3 times the typical step per frame.

Optional filters remove candidates at this stage:

- **Ghost block (`q_young`, `q_seed`).** A young track (fewer than `q_young` points) cannot continue onto a point with a ghost probability above `q_seed`.
- **Blob gate (`blob_gate`).** A real particle keeps its brightness from frame to frame. The tracker removes a candidate when the brightness change is too large.

## 6. Phase 2: the 2D cost and the assignment

### 6.1 Cost

With `cost_mode: projected`, the tracker projects each prediction into the cameras. It compares each projected position with the leaf of the candidate in each camera.

The cost is the mean pixel distance over the cameras that see both points. The tracker scales it by `C / n_valid`. If fewer cameras see the point, the cost increases. This is because fewer cameras give less reliable data.

The tracker uses 3D distance instead in these cases:

- `cost_mode` is `3d`.
- `leaf_weight` is 0.
- No projection function exists.

The tracker multiplies the cost by `leaf_weight`. If calibration is poor, lower this value.

### 6.2 Groups

The tracker builds a graph. Predictions and candidates are the nodes. Each candidate pair is an edge. It then finds the connected groups.

- **One edge in a group:** the tracker accepts the link directly.
- **Larger group:** the tracker uses the Hungarian algorithm (`linear_sum_assignment`). This gives the lowest total cost with each detection used once.
- **Group larger than `max_group_size` (default 128):** the tracker uses a greedy method. It takes the lowest-cost edge first. This prevents a stall in dense data.

## 7. Track update

After the assignment:

- **Matched tracks:** the tracker updates position and velocity.
- **Unmatched tracks:** the tracker increases their miss count. A track that misses more than `max_gap` frames ends. A track can cross a gap because it continues on its prediction.
- **Unused detections:** each one starts a new track. The exception is a doubtful ghost point (`q_seed`).

Note: `max_gap` counts steps. `max_gap=1` means no bridging. `max_gap=0` finds no links.

## 8. Confirmation (the main accuracy gain)

Benchmark results show that the two-hop confirmation is the reason two_phase works well. The 2D leaf cost is not.

The rule: a link stays only if the next link continues smoothly. A false link rarely has a smooth continuation.

- The velocity change ("kink") between two links must be below `confirm_tol` (mm/frame).
- With `confirm_ends` on, a track cannot end by stepping onto a stranger.
- With `confirm_auto` (default 8), the tolerance is `8 × median kink`. It follows the position noise.
- The tracker uses the automatic value only for sparse data. The data must have at most 7 neighbours within 5 mm, and the median kink must be at least 0.5 × the median step. Otherwise the tracker uses the fixed value of 0.3.
- An explicit `confirm_tol` number switches the automatic mode off. `confirm_tol: null` switches confirmation off. Do not do this.

The trade-off: a tighter tolerance gives lower velocity error but shorter tracks.

## 9. Optional modes

**Bidirectional (`bidirectional: true`).** The tracker runs forward and backward on reversed frames. It then merges the results with a "Forward-First" rule:

- All forward links stay.
- A backward link is added only if both of its ends are free.
- The tracker adds backward links in order of 3D distance.

`bwd_v_max` can give the backward pass a wider radius. `parallel_directions` runs the two passes in two processes.

**Shared observation (`allow_shared`).** This is for occlusion. In a group with more tracks than detections, a losing track can share the winner's detection.

- `max_shared` limits the consecutive shared frames (default 2).
- `share_tol` limits the cost of a shared claim (default 1.0). This stops a stranded track from taking a foreign detection.
- A shared point moves the position but never the velocity.

## 10. Parameters

| Parameter | Default | Effect |
|---|---|---|
| `v_max` (or `dvxmax`) | 5.0 mm/frame | 3D search radius |
| `max_gap` | 2 | Frames that a track can miss |
| `leaf_weight` | 1.0 | Weight of the 2D cost. 0 = pure 3D. |
| `use_velocity` | true | Use the prediction |
| `cost_mode` | `projected` | `projected` = 2D cost. `3d` = 3D distance. |
| `dt` | 1.0 | Time step for the velocity |
| `max_group_size` | 128 | Largest group that gets the Hungarian algorithm |
| `allow_shared`, `max_shared`, `share_tol` | false, 2, 1.0 | Occlusion handling |
| `q_seed`, `q_young`, `q_weight` | 0.2, 3, 0 | Ghost rules |
| `confirm_tol`, `confirm_auto`, `confirm_ends` | auto, 8, true | Link confirmation |
| `bidirectional`, `bwd_v_max`, `parallel_directions` | false, `v_max`, false | Forward plus backward pass |

The plugin sets the ghost and confirmation defaults when it loads. The dataclass `TwoPhaseTrackerConfig` has these rules off.

For values that suit your data, see [Tracking parameters](tracking_parameters_guide.md). Run `scripts/tracking_advice.py PATH_TO_RUN_FOLDER` to measure your data.

## 11. Use

**Batch or YAML:**

```yaml
plugins:
  selected_tracking: two_phase
track:
  v_max: 5.0
  leaf_weight: 1.0
  max_gap: 2
```

**Python:**

```python
from openptv2.plugins.two_phase_tracking import (
    TwoPhaseTracker, TwoPhaseTrackerConfig)

cfg = TwoPhaseTrackerConfig(v_max=5.0, leaf_weight=1.0, max_gap=2)
links = TwoPhaseTracker(cfg).track_frames(
    frame_particles,   # list of (N_i, 3) arrays, mm
    frame_leaves,      # list of (N_i, 2*C) arrays, px (optional)
    project_fn,        # (N,3) -> (N,2*C) re-projection (optional)
)
# links: list of (t0, row0, t1, row1)
# return_chains=True also gives the per-track histories.
```

## 12. Tuning procedure

1. Measure the typical step `s` (median linked displacement).
2. Set `v_max` to about 3 × `s`.
3. Set `leaf_weight` to 1 with good calibration. Set it to 0 without calibration.
4. Run the tracker. Count the short tracks and the gaps.
5. If gaps are more frequent than short tracks, set `max_gap` to 3.
6. If tracks break at crossings, set `allow_shared` to true.
7. Validate on synthetic data with the same spacing and noise (`tests/helpers/synthetic_scene.py`).

## 13. Known limits

- If particles move farther per frame than the typical spacing, every tracker fails. Check this before you change parameters.
- A maneuver inside a gap is lost, because the tracker continues on the prediction.
- Ghost points cause most of the wrong links that remain. In one benchmark, removal of ghosts from the input lowered the velocity error from 0.198 to 0.147.
- Speed is about linear in the number of particles. `max_group_size` limits the worst case.

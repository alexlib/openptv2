# Spatial Consensus and the Cold-Start Bootstrap

This page explains three new features of the `two_phase` tracker. The features
help the tracker when the data contains ghost points, and when the flow is not
uniform.

- **Spatial consensus.** The tracker compares each link with the links of the
  neighbouring particles.
- **Cold-start bootstrap.** The tracker gives the new tracks of the first frame
  a velocity, so that the spatial consensus also works in the first frame.
- **Affine consensus.** The tracker fits a local velocity gradient, so that a
  shear or a rotation does not spoil the comparison.

All three features are off by default. The tracker behaves as before until you
switch them on.

Source code: `src/openptv2/plugins/two_phase_tracking.py`.
This document uses the ASD-STE100 style (Simplified Technical English).

## 1. The problem

### 1.1 Ghost points

A 3D point is a **ghost** when the camera rays cross there, but no real
particle exists. A ghost looks like a real point to the tracker. It also looks
like a real point to the cameras, because the rays really do cross.

The tracker can only remove a ghost with information that does not come from a
single point. The spatial consensus uses the information of the neighbours.

### 1.2 The cold start

The tracker needs a velocity to predict where a track will be in the next
frame. A new track has no history, so it has no velocity. The tracker starts
such a track at rest.

This is a problem in the first frame of a sequence. All tracks are new. All
velocities are zero. The tracker cannot use the spatial consensus in the first
frame, because the consensus needs velocities.

The result is that a ghost can take over a new track in the first frame. In
one measurement, more than half of the false links that the spatial consensus
did not remove started in the first frame.

## 2. The physical idea

Real particles in a fluid move together. The particles in one small region of
the volume have almost the same velocity. The velocity changes slowly in space.

Two facts follow from this:

1. A true link has a displacement that is close to the displacement of the
   neighbouring particles.
2. A link to a ghost has a displacement that differs from the neighbours.

So the tracker can measure the velocity of the neighbourhood and use it as a
reference. A candidate that does not agree with the reference is probably
wrong.

The comparison needs a tolerance. A tight tolerance removes more ghosts, but it
also removes true links. Section 9 explains how to set the tolerance.

## 3. The spatial consensus

The tracker calls the reference velocity the **consensus** of a point.

The consensus of a point is the median of the velocities of the points around
it. The tracker uses these parameters:

- `vfc_radius` (default 10 mm) is the size of the neighbourhood.
- `vfc_min_neighbors` (default 3) is the minimum number of neighbours. A
  point with fewer neighbours gets no consensus, and the tracker does not use
  the consensus for that point.

The tracker has two ways to use the consensus.

### 3.1 Option A: the gate before the assignment

The tracker checks each candidate before it solves the assignment. It measures
the difference between the displacement of the candidate and the consensus.

- `vfc_gate` removes the candidate when the difference is larger than the gate
  (mm/frame). The gate is off when the value is `None`.
- `vfc_weight` increases the cost of a candidate instead of removing it. The
  cost grows with the difference, relative to the spread of the neighbourhood.
  The value 0 switches this off.

The gate is the stronger tool. It removes the ghost before the assignment, so
the assignment cannot choose it. It also makes the groups smaller. This can
make the tracker faster in dense data.

The soft weight is the weaker tool. It only changes the order of the
candidates. In one test it had almost no effect on the result.

### 3.2 Option B: the filter after the assignment

The tracker can also check the links after the assignment. It takes the
displacement of each link and removes the links that do not fit the
neighbourhood. The tracker uses the median and the median absolute deviation
(MAD) of the neighbourhood for this test.

- `vfc_post_k` (default 3) multiplies the MAD to give the tolerance.
- `vfc_post_min_tol` (default 0.5 mm/frame) is the minimum tolerance. The
  tolerance is the larger of the two values.
- `vfc_post_reject_cold_start` (default true) stops a removed detection from
  starting a new track.

The filter is fast. It only looks at the links that the assignment produced.
But it comes late: if the assignment gave a detection to the wrong track, the
filter removes the link, and the true particle is lost for that frame. The
track can find the particle again in a later frame, if the track stays alive
(see `max_gap`).

### 3.3 The mode

`vfc_mode` selects what the tracker does.

| Value | Meaning |
|---|---|
| `off` | No consensus. This is the default. |
| `pre` | Option A. The tracker uses the gate and the weight before the assignment. |
| `post` | Option B. The tracker filters the links after the assignment. |
| `both` | Option A and Option B. |

## 4. The cold-start bootstrap

The bootstrap gives the tracks of the first frame a velocity. The tracker then
can use the consensus in the first frame, and a ghost cannot take over a new
track there.

### 4.1 How the tracker builds the field

The tracker uses two passes.

1. The first pass tracks the sequence in the normal way. The new tracks of the
   first frame start at rest.
2. The tracker takes the tracks of the first pass. Each track gives the
   velocity of one step. The tracker computes the consensus of these
   velocities at the positions of the first frame. This is the **field**.
3. The second pass tracks the sequence again. The tracks of the first frame
   start with the velocity of the field.

The tracker leaves out the links of the first step when it builds the field.
The first step is the step without a velocity, so it is the least reliable
step.

### 4.2 Why two passes

The tracker can also estimate the field directly from the candidate points of
the first frame. It takes the displacements of all candidate pairs and looks
for the most common displacement. This method is simple, but it only works
when the flow is the same in the whole cell.

In a shear flow the velocity changes inside the cell. The true displacements
then lie along a line in the velocity space, and they spread over several
bins of the histogram. The most common bin is then a wrong bin. In a measured
case the true velocity was 0.76 mm/frame and the error of the field was
0.34 mm/frame.

The two-pass method does not have this problem. It uses the velocities of
tracks that the tracker has already found. These velocities are clean. The
same measurement gives an error of 0.00 mm/frame for the two-pass field.

### 4.3 The cost

The two-pass method tracks the sequence twice. The tracking time is about two
times the time of one pass. The bootstrap is off by default, so the cost only
applies to the runs that use it.

### 4.4 The guard

The field of the first pass is not always good. The first link of a track is
made without a velocity. If the particles move a long way in one frame, this
first link is more or less a random neighbour. The field is then mixed up, and
it is worse than the direct estimate.

The tracker measures this. It compares the displacement of the first links
with the distance between the particles.

- The displacement is the median length of the first links.
- The distance is the median distance between a particle and its nearest
  neighbour.

If the displacement is more than `vfc_bootstrap_shift_ratio` (default 0.5)
times the distance, the tracker does not use the field. It uses the direct
estimate instead.

The measured results show the same limit. The table gives the error of the
field in mm/frame. The two-pass method wins as long as the particles move less
than half of the distance between them.

| Displacement / distance | Accuracy of the first links | Direct estimate | Two passes |
|---|---|---|---|
| 0.07 | 99 % | 0.07 | **0.00** |
| 0.19 | 99 % | 0.18 | **0.00** |
| 0.37 | 92 % | 0.34 | **0.00** |
| 0.56 | 76 % | 0.51 | 0.15 |
| 0.94 | 53 % | 0.85 | 0.81 |
| 1.50 | 33 % | 1.42 | 2.35 |

## 5. The affine consensus

The consensus of section 3 is the median of the neighbour velocities. The
median cannot describe a shear. In a shear, the velocity changes across the
neighbourhood, and the median gives the velocity in the middle of the
neighbourhood. The comparison then has a large error at the edge of the
neighbourhood.

The tracker can fit a local velocity gradient instead. The model is

```text
v(x) = v0 + A (x - x0)
```

Here `v0` is the velocity at the centre `x0`, and `A` is the 3×3 matrix of the
velocity gradient. The tracker fits this model to the neighbour velocities by
a least-squares method.

For a fluid with constant density, the flow has no volume change. The
divergence of the velocity is zero. The tracker adds this condition to the
fit: `tr(A) = 0`. The model has 12 values (3 for `v0` and 9 for `A`). The
condition removes one value, so the fit has 11 free values.

The fit needs enough points.

- `vfc_affine` (default false) switches the affine fit on.
- `vfc_affine_min_neighbors` (default 6) is the minimum number of neighbours
  for the fit. Below this number the tracker uses the median.

The affine fit gives a much tighter comparison in a shear or in a rotation.
The gate can then be much smaller. Section 6 explains the result.

## 6. What the features do together

The features work together. The following table gives the result on a shear
flow. The flow is `vx = 0.12 y` with 96 particles in a 24 mm box. The gate is
on with `vfc_gate: 0.5` and the affine fit is on.

| Setting | Links kept | Correct links in the first frame |
|---|---|---|
| Direct estimate (one pass) | 136 of 288 | 45 % |
| Field of a first pass (two passes) | **288 of 288** | **100 %** |

The direct estimate does not only give a wrong velocity. The gate uses the
velocity as its reference. A wrong reference makes the gate remove the true
candidates. In this test, the gate removed more than half of the true links.

The affine fit is the reason the field of the first pass is exact. The fit
reproduces a linear shear exactly.

## 7. Parameters

| Parameter | Default | Effect |
|---|---|---|
| `vfc_mode` | `off` | `pre`, `post` or `both` |
| `vfc_radius` | 10.0 mm | Size of the neighbourhood of the consensus |
| `vfc_min_neighbors` | 3 | Minimum number of neighbours for a consensus |
| `vfc_gate` | `None` | Option A. Removes a candidate above this deviation (mm/frame) |
| `vfc_weight` | 0.0 | Option A. Increases the cost of a deviating candidate |
| `vfc_post_k` | 3.0 | Option B. Multiplier of the local MAD |
| `vfc_post_min_tol` | 0.5 mm/frame | Option B. Minimum tolerance |
| `vfc_post_reject_cold_start` | true | Option B. A removed detection does not start a track |
| `vfc_bootstrap` | false | Gives the first-frame tracks a velocity |
| `vfc_bootstrap_passes` | 2 | 1 = direct estimate, 2 = field of a first pass |
| `vfc_bootstrap_shift_ratio` | 0.5 | The guard of section 4.4 |
| `vfc_bootstrap_grid_size` | 20.0 mm | Cell size of the direct estimate |
| `vfc_affine` | false | Fits a local velocity gradient with `tr(A) = 0` |
| `vfc_affine_min_neighbors` | 6 | Minimum number of neighbours for the fit |

## 8. Use

**YAML:**

```yaml
plugins:
  selected_tracking: two_phase
track:
  vfc_mode: both
  vfc_radius: 10.0
  vfc_min_neighbors: 3
  vfc_gate: 0.5
  vfc_post_k: 3.0
  vfc_post_min_tol: 0.5
  vfc_bootstrap: true
  vfc_bootstrap_passes: 2
  vfc_affine: true
  vfc_affine_min_neighbors: 6
```

**Python:**

```python
from openptv2.plugins.two_phase_tracking import (
    TwoPhaseTracker, TwoPhaseTrackerConfig)

cfg = TwoPhaseTrackerConfig(
    v_max=5.0,
    vfc_mode="both",
    vfc_gate=0.5,
    vfc_bootstrap=True,
    vfc_affine=True,
)
links = TwoPhaseTracker(cfg).track_frames(frame_particles, frame_leaves, project_fn)
```

## 9. How to set the values

1. Measure the typical step of the particles. This is the median length of the
   links of a good run.
2. Measure the distance between the particles. This is the median distance
   from a particle to its nearest neighbour.
3. Set `vfc_radius` to about 3 times this distance. The neighbourhood must
   contain enough particles, but it must be small enough to describe one local
   flow.
4. Set `vfc_min_neighbors` to 3 or 4. A larger value gives no consensus in
   sparse data.
5. Set `vfc_gate` to 2 to 3 times the noise of the 3D positions. Start with
   the Option B filter (`vfc_mode: post`) because it is safer. Lower the
   tolerance only if the tracks are too short.
6. Switch on `vfc_bootstrap` and `vfc_affine` together. The affine fit is
   what makes the field of the first pass correct in a shear.
7. Check the result. The number of links and the length of the tracks must not
   go down.

Run `scripts/tracking_advice.py PATH_TO_RUN_FOLDER` to measure the data
first. Run `scripts/bench_two_phase_vfc.py` to compare the settings on
synthetic data with ghosts.

## 10. Limits

- The features help only when the flow is slow enough compared to the
  distance between the particles. If the particles move farther per frame than
  their distance from each other, every tracker fails. Check this first.
- The consensus needs a dense enough seeding. In sparse data the tracker has
  no neighbours, and the features do nothing.
- The bootstrap doubles the tracking time.
- The two-pass field uses the tracks of the first pass. It describes the flow
  of the later frames. It is wrong for a flow that changes quickly in time.
- The affine fit assumes a flow with constant density. A compressible flow
  gives a small bias.

## 11. Tests and measurements

The tests are in `tests/unit/test_two_phase_vfc.py` and
`tests/unit/test_two_phase_bootstrap.py`. The script
`scripts/bench_two_phase_vfc.py` measures the speed and the accuracy of the
options on synthetic data with ghosts. The script
`scripts/visualize_vfc_ghosts.py` draws the trajectories before and after the
consensus.

The measurements of this page come from the plan documents
`docs/plans/2026-10-12-spatial-neighborhood-consensus-vfc.md` and
`docs/plans/2026-10-12-ideal-tracker-bootstrap-and-affine-consensus.md`.

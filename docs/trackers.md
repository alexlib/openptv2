# Particle trackers in openptv2

**Three trackers, two switches, one add-on.** That is the whole setup. This page is the user guide; the
machine-readable version is `openptv list-trackers` (`src/openptv2/tracking_registry.py`), and the single place
that interprets a saved configuration is `src/openptv2/tracking_setup.py`. For measured numbers see
[Tracker Tutorials](tracker-tutorials.md); for files and pipeline see the [Tracking Guide](tracking_guide.md);
for the default tracker in depth, [Two-Phase Tracking](two-phase-tracking.md); for the add-on,
[Holistic tracking](holistic-tracking.md).

## Quick start

Do nothing and you get the recommended setup: `two_phase`, `direction: both`, no extras. For difficult data (overlapping
particles) `openptv recommend <run.zarr>` measures your points and says which extras are worth their run time
([Choosing](#choosing-which-tracker-which-extras)).

```yaml
track:
  tracker: two_phase      # two_phase | trackcorr | fast_3d          default: two_phase
  direction: both         # forward | both                           default: both
  align_4cam: true        # all trackers: fix 3-cam calibration jumps  default: true
  postprocess: true       # trackcorr, fast_3d: reciprocity + gaps   default: true
  holistic_rounds: 0      # two_phase add-on: merged-blob loop       default: 0 (off)
  shared_correction: false  # two_phase add-on: fix points in shared blobs  default: false
  dvxmin: -2.0            # search bounds, same for every tracker
  dvxmax: 2.0             #   (dvy*, dvz* likewise) mm/frame
  dacc: 0.4               # mm/frame^2
  angle: 120              # gon
```

| Where | How |
|---|---|
| GUI | Tracking Parameters: **Tracker**, **Direction**, Post-process, the bounds, and the add-on box *Merged-blob loop rounds*. Saved into the YAML above. |
| Batch | `openptv-batch parameters_Run1.yaml 1 500` uses the YAML. Override on the command line: `--tracker fast_3d`, `--direction forward`, `--holistic-rounds 2`, `--shared-correction`. The same flags exist in the parallel batch. |
| Python | `run_tracking_plugin("default", exp)` reads the YAML; `run_tracking_plugin("trackcorr", exp)` names the tracker. |
| Advice | `openptv recommend res/run.zarr`: measures the data, names the tracker and the extras, with reasons. |
| List | `openptv list-trackers` (table); `openptv list-trackers --show two_phase` (details and every setting). |

## Choosing: which tracker, which extras

Short answer: **`two_phase`, with nothing else, unless your data says otherwise.** The extras
(`ptv.shared_pass`, `track.shared_correction`) are for data where particles overlap in the camera images. They bring a
solid but not drastic improvement and cost run time, so they are never on by default.

Ask your own data: `openptv recommend <run.zarr>` measures six numbers on your points (no truth needed) and prints the
setup below with the reasons. The numbers: points per frame; spacing/step (how many steps apart neighbours are);
mutual-link share (how many points have an unambiguous partner in the next frame; low = dense); noise/step (the
velocity kink of a track against its step: 0 = exact data); partial views (share of points seen by fewer cameras
than the rig has); **shared-blob flag** (share of points with one view much brighter than their other views, the sign
of a blob holding two particles).

| Your data | Use | Why (benchmark, 13 cases; real wp1, wp2 for the descriptors) |
|---|---|---|
| Real or noisy data (any density, any step) | **`two_phase`** | fastest (4 to 5 s per 200 frames, against 25 to 176 s for `trackcorr`) and the most accurate: velocity error 1.3 to 7 times lower than `trackcorr` or `fast_3d`; the only one that copes when neighbours are fewer than 10 steps apart |
| Exact data (simulation, no noise: noise/step below 0.1, no merged blobs) | **`fast_3d`** | keeps every point (yield 0.995) where `two_phase`'s ghost rules drop 3 to 6 % of them; `trackcorr` gives the same result 2 to 3 times slower |
| You must reproduce a legacy (OpenPTV, C `trackcorr`) result | `trackcorr` | the legacy algorithm. It is never the fastest here, it is the least accurate on noisy data (velocity error 0.20 to 1.05 against 0.08 to 0.41 for `two_phase`), and with large steps it breaks (frame skip 4: 0.39 against 0.11; skip 8: 0.82 against 0.13, 176 s) |
| Shared-blob flag 10 % or more | `two_phase` + `shared_correction` (+ `shared_pass`) | see below; real recordings wp1 and wp2 have 15 % |
| Shared-blob flag below 8 % | `two_phase` alone | the extras change nothing (clean data) or a few percent |

**What the extras bring**, velocity error (long-track completeness) per tracker setup. Benchmark cases with truth;
the shared-blob flag is the number `openptv recommend` prints. Run time relative to `two_phase` alone:
about 1 (`two_phase`), +10 to 30 % of the matching time for the pass, 3 to 4 times the tracking time for the
correction.

| Case (flag) | `two_phase` | + pass | + correction | + both | `fast_3d` | `trackcorr` |
|---|---|---|---|---|---|---|
| exact, no noise, dense (0.00) | 0.0008 (0.92) | 0.0009 | 0.0008 | 0.0009 | 0.0009 (0.98) | 0.0009 (0.98) |
| noise only, no extra effects (0.13) | 0.160 (0.53) | 0.158 (0.56) | 0.096 (0.56) | 0.098 (0.60) | 0.204 (0.57) | 0.266 (0.52) |
| real-like, merge rate 0.6 (0.11) | 0.081 (0.70) | 0.078 (0.73) | 0.078 (0.70) | 0.071 (0.74) | 0.145 (0.73) | 0.201 (0.59) |
| real-like, merge rate 0.75 (0.14) | 0.105 (0.62) | 0.102 (0.66) | 0.092 (0.64) | 0.088 (0.69) | 0.155 (0.67) | 0.220 (0.54) |
| real-like, default merge rate (0.17) | 0.183 (0.50) | 0.183 (0.53) | 0.128 (0.53) | 0.128 (0.56) | 0.243 (0.56) | 0.297 (0.44) |
| density x2 (0.26) | 0.311 (0.34) | 0.319 (0.36) | 0.225 (0.36) | 0.226 (0.39) | 0.594 (0.39) | 0.508 (0.31) |
| density x4 (0.40) | 0.409 (0.16) | 0.420 (0.16) | 0.340 (0.17) | 0.345 (0.17) | 1.688 (0.18) | 1.050 (0.15) |
| every 4th frame (0.17) | 0.113 (0.52) | 0.112 (0.54) | 0.089 (0.55) | 0.096 (0.59) | 0.427 (0.54) | 0.388 (0.53) |
| every 8th frame (0.17) | 0.130 (0.48) | 0.124 (0.52) | 0.112 (0.51) | 0.110 (0.56) | 1.162 (0.40) | 0.816 (0.45) |
| clustered, every 4th frame (0.23) | 0.170 (0.32) | 0.173 (0.34) | 0.130 (0.34) | 0.131 (0.37) | 1.251 (0.34) | 0.758 (0.35) |

Reading it:
- The correction pays in proportion to the flag: about 4 % at 0.11, 12 % at 0.14, 28 to 40 % at 0.13 to 0.26 when
  the noise is low. At 0.17 to 0.26 it removes a quarter to a third of the error.
- The pass alone is a yield option: +2 points of yield and +3 to 4 of long-track completeness; it does not lower the
  velocity error, and in dense data (flag above 0.25) it raises it a little unless the correction is on too.
  Together they beat either alone when the flag is around 0.11 to 0.14.
- At density x4 and with large steps `fast_3d` and `trackcorr` fall apart (ghost points, wrong links); `two_phase`
  stays usable. Nothing makes the densest case good: yield stays at 0.23 to 0.25 (a particle in a crowd has no
  clean view).
- On the real recordings (flag 0.15, noise/step 0.67, density like the benchmark's x1) the same setup,
  `two_phase` with both extras, raised the share of points in tracks of 200 frames or more from 6 to 7 % to 14 to 16 %
  and the smoothness of the tracks by 15 %.

## What tracking is

A camera takes a picture every frame. In each picture, every particle is a
dot. In the next picture the dots have moved a little. **Tracking means
deciding which dot in picture 2 is the same particle as each dot in
picture 1.**

When particles are far apart and slow, this is easy: take the nearest dot.
It gets hard when particles cross each other, when one is hidden for a
frame, or when positions are noisy. Every tracker below is a different set
of rules for that decision. They differ in only four things:

1. **Guess** — where do you expect the particle next? (stay put, constant
   velocity, smoothed curve, look at future frames, bigger time step)
2. **Score** — how do you rank candidates? (distance, acceleration, angle,
   camera-image agreement)
3. **Conflicts** — two particles want the same dot: who wins? (first come,
   cheapest first, best total pairing, nobody)
4. **Evidence** — what do you look at? Only 3D points, or also the original
   camera images? Trackers that check the images survive bad 3D points;
   trackers that ignore them are faster.

Two facts shape everything on this page. First, measured on synthetic flow
(see `docs/tracking-benchmark-results.md`), most wrong links come from
**missing detections across gaps (~60%)** and **cold starts (~20%)** — not
from the scoring rule. So gap handling matters more than clever costs.
Second, particles only ever **enter and exit** the volume; a track that ends
mid-volume is almost always an occlusion or a dropout, not a real exit.

## The three trackers

| | `two_phase` (default) | `trackcorr` | `fast_3d` |
|---|---|---|---|
| In one line | 3D candidates, image-based cost, optimal pairing, confirmation | The legacy algorithm: search the 2D targets in each camera | Links the triangulated 3D points only |
| Looks at | 3D points and the 2D blob positions in every camera | 2D targets in the images and 3D | 3D points only (optional soft epipolar score) |
| Predicts | position + velocity × steps, velocity learned along the track | the last step repeated | the last step repeated (neighbours' velocity at track starts) |
| Picks links | best total pairing per group of conflicting candidates (Hungarian) | per particle, in the legacy order | cheapest link first, per pass (greedy) |
| Checks | a link must be continued smoothly by the next one (two-hop) | epipolar match in all cameras, acceleration and angle limits | acceleration limit |
| Gaps | carried inside the loop (`max_gap`) | bridged in post-processing | bridged in post-processing |
| Choose it for | quality and speed on the standard pipeline; the only one the add-on works with | comparison with the legacy software, old results | runs with triangulated points only; a 3D-only baseline |
| Needs | the run store (the standard pipeline) | the targets of every camera | triangulated points |

**How to choose.** Start with the default. Use `trackcorr` when you must reproduce the legacy software. Use `fast_3d`
when you have only triangulated points, or want a baseline that ignores the images. Each tracker has its own few extra
settings, listed below; do not carry them from one tracker to another.

### What we measured

One synthetic case with truth (1,220 particles per frame, 200 frames, real-like merge rate and brightness scatter;
`scripts/synth_bench.py`, case `L1_d1_c0_k1_n200_s0.08_a0.18_f0.05_m0.75`), same search bounds for all, run time on one
machine with nothing else running. Velocity error = RMS of the smoothed velocity error against the true flow, relative to
the rms speed (lower is better); link precision = share of links that are true.

| Tracker, direction | velocity error | link precision | long-track completeness | time |
|---|---|---|---|---|
| `two_phase`, both (default) | **0.105** | **0.971** | 0.62 | **4.0 s** |
| `two_phase`, forward | 0.116 | 0.973 | 0.59 | 4.2 s |
| `fast_3d`, both | 0.155 | 0.940 | 0.67 | 9.8 s |
| `fast_3d`, forward | 0.157 | 0.941 | 0.67 | 4.9 s |
| `trackcorr`, both | 0.220 | 0.940 | 0.54 | 24.2 s |
| `trackcorr`, forward | n/a | n/a | n/a | 18.2 s |

Notes. (1) Writing the results to the run store used to dominate `fast_3d` and `trackcorr` (every frame wrote several
zarr arrays; `fast_3d` took 22.6 s and `trackcorr` 34.6 s, with the kernel itself about 2 s for `fast_3d`). Tracking now
holds the linkage in memory and writes it once per pass, rewrites in place, and skips identical re-writes: `fast_3d` forward
11.3 s to 4.9 s, `trackcorr` both 34.6 s to 24.2 s, with identical link quality. What remains for `trackcorr` is its 2D
search itself. (2) With `fast_3d`, forward alone and both directions are close (velocity error 0.157 and 0.155 here; the other cases
below); `two_phase` gains more from both.

Direction, three cases (velocity error; `both` for `fast_3d` is the reversed pass, see below):

| Case | `two_phase` fwd / both | `fast_3d` fwd / both (reversed) / both (old C-style backward) | `trackcorr` fwd / both |
|---|---|---|---|
| default merge rate | 0.188 / **0.183** | 0.254 / **0.243** / 0.265 | 0.294 / 0.297 |
| merge scale 0.6 | 0.087 / **0.081** | **0.141** / 0.145 / 0.155 | 0.201 / 0.201 |
| merge scale 0.75 | 0.116 / **0.105** | 0.157 / **0.155** / 0.167 | 0.220 / 0.220 |

One more finding: on a clean dense case with no ghost points (`L0_d4_c0_k1_n150`, 4 times the seeding) `two_phase` keeps
94 % of the true points where `fast_3d` and `trackcorr` keep 99.6 %. The ghost rules (`q_seed`, `q_young`) are the cause:
with them off `two_phase` keeps 99.6 % too. With ghosts present (the other cases) they pay: link precision 0.971 against
0.944. So they stay on; switch them off (`q_seed: null`, `q_young: 0`) for data you know has no ghost points.

### Settings: shared, and per tracker

| Setting | `two_phase` | `trackcorr` | `fast_3d` |
|---|---|---|---|
| `dvx*`, `dvy*`, `dvz*` (mm/frame) | yes (as the radius `v_max`: the largest of the six) | yes | yes |
| `dacc` (mm/frame²), `angle` (gon) | no: it confirms links instead (`confirm_tol`, automatic) | yes | yes |
| `direction`, `postprocess` | `direction` yes; `postprocess` not used (it carries gaps and confirms inside the loop) | both | both |
| `holistic_rounds` | **yes (the add-on)** | no | no (tried, see below) |
| `leaf_weight`, `max_gap`, `confirm_*`, `q_*`, `blob_gate` | yes | no | no |
| `epi_weight`, `accel_gate` | no | no | yes (defaults 0 and off) |

`fast_3d` reads `dacc` and `angle` only when `accel_gate: true` is set. Measured: with a loose `dacc` (6) the velocity error
fell from 0.1665 to 0.1595 and link recall stayed at 0.999; with a tight `dacc` (0.4) link recall fell to 0.887, so
take `dacc` from a probe run. `epi_weight` (the old "hybrid" penalty) changed the velocity error by 0.0004 (0.1665 to
0.1661); the old "hybrid" never applied it at all (a bug, fixed), so its measured results were plain `fast_3d`.

## The two switches

**`direction: forward | both`.** Forward tracks the frames in order. `both` also tracks backward and keeps what is
consistent or fills gaps. It is two passes, so it takes longer, and it gives the better trajectories for `two_phase`.
How backward is done depends on the tracker:

- `two_phase` and `fast_3d`: **the same tracker on the reversed frame numbers** (backward is only a different way of
  numbering the frames), then the forward links stay and backward links are added where both ends are still free
  (Forward-First). For `fast_3d` this beat the old C-style backward on all three benchmark cases.
- `trackcorr`: the **legacy backward pass** (`trackback`), because the tracker is meant to match the legacy
  software; a reversed pass was measured and was worse for it (velocity error 0.327 against 0.297).

In the GUI the *Track back* menu item exists for `trackcorr` and `fast_3d` runs that were made forward-only.

**`postprocess: true | false`.** After the passes: reciprocity check (a link must be found from both ends) and gap
bridging. For `trackcorr` and `fast_3d`. `two_phase` does these inside its own loop, so the switch has no effect
there.

## The add-on: `holistic_rounds`

The merged-blob loop. Where two particles overlap in a camera image, the detector gives one blob with a shifted
centre. The loop estimates a centre for each particle from its own track, re-triangulates, and links again;
`holistic_rounds: 2` repeats that twice. It costs run time. It works with `two_phase`; asking for it with
another tracker is ignored with a log line. It was tried around `fast_3d` and made the velocity error worse (0.167 to
0.199: `fast_3d` has no link confirmation, and the extra points bring wrong links), so it is not offered there. Details and measurements: [Holistic tracking](holistic-tracking.md).

## The add-on: `shared_correction`

The part of the merged-blob loop that matters most, on its own and cheaper. A point that uses a blob another particle
also uses has a shifted centre in that camera, and its position is wrong by a correlated amount that smoothing cannot
remove (down-weighting such points in the smoothing gained 2 % even with perfect weights). With
`shared_correction: true` the linker does this:

1. link once (as usual);
2. align the 4 camera views onto a single zero-drift 3D gauge (`align_4cam=True`) so switching between 4 cameras, 3 cameras, or down-weighted merged views does not introduce a calibration jump (see [Camera-Subset Alignment](camera-subset-alignment.md));
3. for every point that touches a shared blob (or whose blob is too bright for its own particle), take the track's
   predicted position of the *other* members of the blob and compute this point's own share of the blob centre,
   `(centre - sum of the others' weights x predicted positions) / its own weight`; weights are the brightness
   signatures of the tracks. Re-triangulate purely from those aligned, corrected camera views (`prior=False`, no quadratic
   smoothing bias). Clean 4-camera points are untouched; clean 3-camera points are aligned onto the 4-camera gauge;
4. link again with the corrected positions. The second link is needed: it is what lets the linker connect the
   corrected points (link recall 0.974 to 0.984 on the benchmark); keeping the old links with new positions was
   measurably worse.

```yaml
track:
  shared_correction: true     # two_phase only; default false. Batch: --shared-correction
```

If no blob is shared by two points the step is skipped. It works best with `ptv.shared_pass` (the new points of the
pass are then corrected too and are no longer penalised by the ghost filter's brightness check). Cost: about 2.5 to 3.5
times the `two_phase` run time (two link passes plus the aligned refinement).

Measured (benchmark with truth; velocity error, long-track completeness, fragments per long track, yield):

| Case | `two_phase` | `shared_correction` | + `shared_pass` |
|---|---|---|---|
| rendered with cal offset (`syn1`, 40 fr) | 0.1670 / — / — / 0.706 | **0.0930** / — / — / 0.710 | **0.0921** / — / — / 0.730 |
| rendered (`L2`, 200 fr) | 0.1400 / 0.544 / 2.03 / 0.691 | **0.0913** / 0.569 / 1.86 / 0.695 | **0.0891** / 0.611 / 1.71 / 0.716 |
| real-like merge rate (`m0.75`) | 0.1046 / 0.623 / 1.68 / 0.754 | **0.0879** / 0.641 / 1.58 / 0.756 | **0.0824** / 0.692 / 1.36 / 0.775 |
| default merge rate | 0.1825 / 0.503 / 2.29 / 0.680 | 0.1283 / 0.528 / 2.07 / 0.686 | **0.1212** / 0.568 / 1.92 / 0.712 |
| 4x density | 0.433 / 0.182 / 0.77 / 0.227 | 0.361 / 0.189 / 0.77 / 0.232 | 0.365 / 0.196 / 0.78 / 0.240 |
| no merged blobs | no change (skipped) | | |

On the real recording wp2 (300 frames, no truth; `bench/real_check/shared_pass_real.py`): share of
points in tracks of 50 frames or more 42.2 % -> 51.1 % (55.3 % with the pass), of 200 or more 7.1 % -> 10.8 % (15.6 %),
roughness of the tracks (second difference, median) 0.0548 -> 0.0464 mm, short tracks (under 5 frames) 15.4 % -> 14.1 %,
tracks of 50 frames or more 1540 -> 1695 (1720). Tracking took 9 s -> 29 s for the 300 frames. wp1 (same
check): 40.0 % -> 49.6 % (54.4 %), 200 or more 6.0 % -> 10.2 % (14.1 %), roughness 0.0557 -> 0.0474 mm, tracks of 50
frames or more 1512 -> 1691 (1738).

## Before tracking: the shared-blob pass (`ptv.shared_pass`)

Two particles that overlap in a camera produce one blob. The stereo matcher uses every blob once, so the first
particle takes it and the second is left with its other cameras, often only two, which the matcher does not accept.
On the benchmark that costs a particle seen in 4 cameras with 2 of them shared about a third of its chances to be
found (64 % found, against 100 % when no blob is shared). The tracker cannot get back a point that was never made.

`shared_pass` is an option of the matching step (the `ptv` section of the parameter file, not `track`):

```yaml
ptv:
  shared_pass: true      # or a number, the brightness ratio (default 1.7); false / absent = off (default)
```

After the normal matching, the blobs that look shared are offered again together with the targets no point used, and
the matcher runs once more on that small set. A blob looks shared when it is brighter than the ratio times the
second-dimmest view of the same matched point (grey sums normalised by the camera's median blob). A new point needs
at least 3 cameras, at least 2 of them on blobs nobody else used, and its shared views count little in the position.
It costs about 10 % more matching time and changes nothing when no blob is shared (clean data).

Measured with `two_phase` (benchmark, ratio 1.7; yield, long-track completeness, velocity error):

| Case | yield | long tracks | velocity error |
|---|---|---|---|
| real-like merge rate (m0.75) | 0.754 -> 0.772 | 0.623 -> 0.658 | 0.1046 -> 0.1016 |
| default merge rate | 0.680 -> 0.700 | 0.503 -> 0.526 | 0.1825 -> 0.1832 |
| 4x density | 0.227 -> 0.236 | 0.182 -> 0.189 | 0.433 -> 0.445 |
| no merged blobs | no change | no change | no change |

The new points are noisier than ordinary ones (about 3 times the position error), so the gain is in yield and
continuity; at 4x density it costs a little velocity error and a few more ghost points, so use it for sparse to
moderate seeding. Combined with `holistic_rounds` it supplies most of what the births do, and the loop's position
refinement then lowers the error further. Measurements: `docs/plans/2026-11-21-tracker-setup.md`, sections 14 and 15.

## Before tracking: the light check (`ptv.light_check`)

A point matched in all cameras but one is the weakest kind the stereo matcher makes: a few rays that meet by chance (a
ghost) look like a particle seen in one camera less. A real particle leaves light in the camera that did not see it.
The light check projects every such point into its missing camera, reads the raw image there, and drops the point when
there is no light:

```yaml
ptv:
  light_check: true      # or a number, the minimum light in noise units (default 6); false / absent = off (default)
```

The light is the peak within 1.5 px of the projected position over the local background (median of the ring beyond
4 px), in units of the local noise (1.4826 times the median absolute deviation). A ghost shows noise there (median about
4), a real particle a clear peak (median about 90 on the rendered benchmark, about half that on real recordings). Only
points with exactly one camera missing are judged; a point whose window leaves the image is kept; a frame whose images
cannot be read is skipped with one warning and keeps all its points. It runs after the matching, needs the images (read
once per frame), and adds about 3 to 4 s per 300 frames of 4 cameras.

Measured with `two_phase` + `shared_correction` (docs/plans/2026-11-21-tracker-setup.md, sections 23 and 24):

| Case | pass only | pass + light check |
|---|---|---|
| rendered benchmark syn1, 400 frames: yield | 0.694 | 0.690 |
| ghost points | 3.58 % | 2.71 % |
| raw link precision | 0.9643 | 0.9724 |
| velocity error | 0.1188 | 0.1132 |
| real wp2, 300 frames: points in tracks of 50+ frames, per frame | 649 | 621 |
| singletons | 11.3 % | 8.8 % |
| real wp1, 300 frames: points in tracks of 50+ frames, per frame | 648 | 621 |
| singletons | 11.1 % | 8.7 % |

On real recordings the missing views carry less light than on the benchmark (a quarter of the 3-camera points show
none), so the check removes 8 to 10 % of the points; the roughness tail and the leave-one-camera-out residual do not
get worse, but some of the removed points are real. Lower the number (4) to keep more, raise it (10) to remove more.
It is meant to go with `shared_pass`, whose new points are 3-camera points and, left alone, add ghosts. Without
`shared_pass` it removes ghosts at a small cost in points.

## Old names still work (for one release)

A saved YAML or script that uses an old name is translated by `tracking_setup.resolve_tracking_setup`, and the log
says what it became. Old names will stop working after one release of warnings.

| Old | Now |
|---|---|
| `priority_segment_3d`, `fast`, `fast_3d`, `cython_3d*`, `openptv_fast_3d` | `fast_3d` |
| `trackcorr` (any engine key absent or `epipolar`), `cython_epipolar*`, `openptv_epipolar` | `trackcorr`, **the legacy algorithm only** |
| `trackcorr` with `trackcorr_engine: 3d_hybrid` (the old default) | `fast_3d` with `epi_weight: 1`, direction both |
| `full_multipass`, `two_directional` | `trackcorr`, direction both |
| `standard_forward` | `trackcorr`, direction forward |
| `holistic` | `two_phase`, `holistic_rounds: 2` |
| `unified`, `4be`, `hybrid_deltat_3d`, `splitter_tracking` | retired; shown as the nearest kept tracker |
| `run_backward`, `backward`, `bidirectional`, `direction: forward_backward` | `direction: forward \| both` |
| a plugin name of your own | runs your plugin, unchanged |

**What changed in behaviour.** (1) The default is now `two_phase` with `direction: both`; before, a YAML without a
choice ran the 3D-only tracker forward. (2) `trackcorr` now always means the legacy algorithm; before, it ran the
faster 3D "hybrid" unless `trackcorr_engine: epipolar` was set. To keep the old default behaviour use `tracker:
fast_3d` with `epi_weight: 1`. (3) New projects no longer get `preset: full_multipass` in the YAML.

## Upstream credit: MyPTV and proPTV

openptv2 borrows ideas from two outside projects — adapted concepts on
openptv2's own data structures, not modified copies of their code. The full
frameworks (triangulation pipelines, calibration, backtracking/repair,
smoothing toolboxes) live only in their own repositories — use those projects
directly if you need them. Both are permissively MIT-licensed.

- **MyPTV** by Ron Shnapp — open-source Python 3D-PTV library.
  Repository: <https://github.com/ronshnapp/MyPTV> ·
  Paper: Shnapp, R. (2022). *MyPTV: A Python Package for 3D Particle
  Tracking.* Journal of Open Source Software, 7(75), 4398.
  <https://doi.org/10.21105/joss.04398> ·
  What we adapted: kinematic prediction + assignment matching in 3D
  (the retired `hybrid_deltat_3d`).
- **proPTV** by Robin Barta and colleagues (DLR) — probabilistic PTV
  framework, Python.
  Repository: <https://github.com/RobinBarta/proPTV> ·
  Paper: Barta, R. et al. (2024). *proPTV: A probabilistic particle
  tracking velocimetry framework.* Journal of Computational Physics, 514,
  113212. <https://doi.org/10.1016/j.jcp.2024.113212> ·
  What we use: its benchmark ground-truth format and identity metrics.
- The classic engines descend from the OpenPTV/liboptv lineage
  (<http://www.openptv.net>).

If you publish with these methods, please cite the original authors above
in addition to openptv2.


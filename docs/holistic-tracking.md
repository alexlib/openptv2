# Holistic tracking (the add-on: `holistic_rounds` with `two_phase`)

!!! abstract "In short"
    The merged-blob loop: tracks and blobs resolve each other, so overlapping particles keep their own positions. Switch: `track.holistic_rounds: 2`.

    **Basics:** [Course page 6](course/06-track.md). **Do it:** [Dense seeding](recipes/dense-seeding.md).


Switch: `track.holistic_rounds: 2` (default 0 = off). See [Particle trackers](trackers.md) for the whole setup.

Detection, stereo matching and tracking normally run once each, in that order. The weak point is the **merged blob**:
two or more particles in one detected blob. The matcher gives the blob to one point, so the kept particle gets a
shifted centre, its partner is dropped, and the partner's light corrupts the kept particle's observation. On the
benchmark about 27 % of blob observations are merged (about half that on real wp2); with exact data the chain is
accurate to 0.0009, so this is the main loss.

The holistic loop (an option of `two_phase`, not a separate tracker) lets tracks and blobs resolve each other. Per round:

1. link with `two_phase`;
2. infer from the tracks which blobs hold several particles;
3. estimate each particle's brightness per camera (jointly with the blob counts);
4. give each particle its own effective pixel observation in a shared blob and triangulate again (a short
   path prior keeps it stable);
5. add particles hidden in other particles' blobs, found from the unexplained brightness, matched in 3D;
6. link again.

After the last round, births that no track took are removed (about half of them are wrong; linked births are 96 %
real on the benchmark) and the points are linked once more.

## Use

```yaml
plugins:
  selected_tracking: two_phase  # GUI: Tracking Parameters -> Tracker
track:
  holistic_rounds: 2          # 0 (default) = plain two_phase; GUI: "Merged-blob loop rounds"
  holistic_min_resid: 0.25    # smallest unexplained brightness that makes a birth
  holistic_partner_min_len: 10
  holistic_extend: 0          # passes of strict track-assisted extension after the loop
  noise_aware_confirm: true   # two_phase link confirmation widens for noisy points (sparse data only)
```

The old name `selected_tracking: holistic` still works: it is `two_phase` with `holistic_rounds` defaulting to 2.

Needs a zarr run store with correspondences. Output is the usual `ptv_is` linkage; the store also gets an
observation table (`obs/`) and per-point sigma (`psigma/`). Cost: about 0.6 s per frame per round (Python).

## Measured (details: `docs/plans/2026-11-20-holistic-ptv.md`)

| | `two_phase` | with `holistic_rounds: 2` |
|---|---|---|
| Rendered images, 200 frames: velocity error | 0.140 | 0.108 |
| Same: yield / long-track completeness | 0.691 / 0.544 | 0.741 / 0.670 |
| Dense (x4): velocity error | 0.433 | 0.382 |
| Real wp2, 300 frames: mean track length | 20.7 | 28.2 |
| Real wp2: jump share | 1.44 % | 0.47 % |

## Limits
- Dense data gains little: at x4 the yield is 0.25 against a ceiling of 0.85.
- It rests on `two_phase` tracks; if those are poor the blob inference is poor.
- Truth-free check on your own data: leave-one-camera-out (`openptv2.loco`) and `scripts/holistic_real.py`.

## Viewing the tracks

The loop writes the final tracks into `res/run.zarr/trajectories` (metres, velocity per frame), so openptv-cloud's
`03_results` marimo notebook (3D tracks coloured by speed, length histogram) opens a holistic run directly. For
runs made by another tracker, or edited afterwards: `uv run python scripts/export_trajectories.py <run folder>`.

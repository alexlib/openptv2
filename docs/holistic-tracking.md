# Holistic tracking (`holistic`)

Detection, stereo matching and tracking normally run once each, in that order. The weak point is the **merged blob**:
two or more particles in one detected blob. The matcher gives the blob to one point, so the kept particle gets a
shifted centre, its partner is dropped, and the partner's light corrupts the kept particle's observation. On the
benchmark about 27 % of blob observations are merged (about half that on real wp2); with exact data the chain is
accurate to 0.0009, so this is the main loss.

`holistic` lets tracks and blobs resolve each other. Per round:

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
  selected_tracking: holistic   # GUI: Plugins page, tracking plugin
track:
  holistic_rounds: 2          # default
  holistic_min_resid: 0.25    # smallest unexplained brightness that makes a birth
  holistic_partner_min_len: 10
  noise_aware_confirm: true   # two_phase link confirmation widens for noisy points (sparse data only)
```

Needs a zarr run store with correspondences. Output is the usual `ptv_is` linkage; the store also gets an
observation table (`obs/`) and per-point sigma (`psigma/`). Cost: about 0.6 s per frame per round (Python).

## Measured (details: `docs/plans/2026-11-20-holistic-ptv.md`)

| | `two_phase` | `holistic` |
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

## Using it from `unified`

`unified` delegates to the holistic loop when its `track:` section has `holistic: true` (default off; the other
`unified` post-passes are then not run, because the loop does its own linking and cleanup):

```yaml
track:
  holistic: true
  holistic_rounds: 2
```

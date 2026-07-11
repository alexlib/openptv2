# tracking_synthetic — synthetic ground-truth tracking fixture

A fully-known 4-camera scene for validating the tracking engines and mapping the
tracking-parameter envelope. Consumed by `tests/unit/test_tracking_synthetic.py`.

## What it is
- **12 particles**, **5 frames** (10001–10005), 4 cameras (test_cavity optics).
- Particles sit on a coarse, collision-free 4×3 grid (spacing 20×14 mm), so the
  correct correspondence is unambiguous.
- **Ground truth = identity:** particle `p` occupies row `p` of `rt_is` in every
  frame, so a correct forward link is `next[p] == p`. With 12 particles and 4
  transitions there are **48** correct links; any `next[p] != p` is a *wrong*
  (cross-particle) link.

## Designed motion signatures (to probe each gate)
| particle | motion | gate it stresses |
|----------|--------|------------------|
| p0 FAST  | extra x-velocity (~4 mm/frame) | `dvxmax` |
| p1 ACCEL | constant accel (~1.5 mm/frame²) | `dacc` |
| p2 TURN  | ~90° zig-zag direction change | `dangle` (and `dacc`) |
| p3–p11   | slow straight drift | none (always linkable) |

## Files
- `cal/camN.tif.ori`, `.addpar` — calibration (copied from test_cavity).
- `img/camN.FFFFF_targets` — 2D detections per camera/frame; `tnr` = particle id
  (the rt_is row it corresponds to). **This tnr↔rt_is consistency is essential**
  for the 2D epipolar tracker — the real pipeline maintains it via
  `correspondences()` + `write_targets()`.
- `res/rt_is.FFFFF` — 3D positions + per-camera correspondence indices.
- `parameters_Run1.yaml` — control/criteria/sequence/track params (`%d` bases).

## Regenerate
```
uv run python test_data/tracking_synthetic/generate.py
```

## What the test demonstrates
- At default params, **both** trackcorr and track3d recover all 48 links, 0 wrong.
- **trackcorr** enforces `dvxmax`, `dacc`, `dangle` and *fails safe* (tightening a
  gate drops the offending particle, never creates a wrong link).
- **track3d** gates on 3D proximity: it ignores `dacc`/`dangle` here, and under a
  too-tight `dvxmax` it can produce **wrong cross-links** — so keep `dvxmax`
  generous for the 3D engine and use `dacc`/`dangle` to reject bad motion in 2D.

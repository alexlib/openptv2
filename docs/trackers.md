# Particle trackers in openptv2

**Three trackers, two switches, one add-on.** That is the whole setup. This page is the user guide; the
machine-readable version is `openptv list-trackers` (`src/openptv2/tracking_registry.py`), and the single place
that interprets a saved configuration is `src/openptv2/tracking_setup.py`. For measured numbers see
[Tracker Tutorials](tracker-tutorials.md); for files and pipeline see the [Tracking Guide](tracking_guide.md);
for the default tracker in depth, [Two-Phase Tracking](two-phase-tracking.md); for the add-on,
[Holistic tracking](holistic-tracking.md).

## Quick start

Do nothing and you get the recommended setup: `two_phase`, `direction: both`, no holistic loop.

```yaml
track:
  tracker: two_phase      # two_phase | trackcorr | fast_3d          default: two_phase
  direction: both         # forward | both                           default: both
  postprocess: true       # trackcorr, fast_3d: reciprocity + gaps   default: true
  holistic_rounds: 0      # two_phase add-on: merged-blob loop       default: 0 (off)
  dvxmin: -2.0            # search bounds, same for every tracker
  dvxmax: 2.0             #   (dvy*, dvz* likewise) mm/frame
  dacc: 0.4               # mm/frame^2
  angle: 120              # gon
```

| Where | How |
|---|---|
| GUI | Tracking Parameters: **Tracker**, **Direction**, Post-process, the bounds, and the add-on box *Merged-blob loop rounds*. Saved into the YAML above. |
| Batch | `openptv-batch parameters_Run1.yaml 1 500` uses the YAML. Override on the command line: `--tracker fast_3d`, `--direction forward`, `--holistic-rounds 2`. The same flags exist in the parallel batch. |
| Python | `run_tracking_plugin("default", exp)` reads the YAML; `run_tracking_plugin("trackcorr", exp)` names the tracker. |
| List | `openptv list-trackers` (table); `openptv list-trackers --show two_phase` (details and every setting). |

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
| Choose it for | quality; the only one the add-on works with | comparison with the legacy software, old results | speed, quick looks, tuning the bounds |
| Needs | the run store (the standard pipeline) | the targets of every camera | triangulated points |

**How to choose.** Start with the default. Use `fast_3d` for a quick first look at a new run or when run time
matters most. Use `trackcorr` when you must reproduce the legacy software. Do not mix their settings: each tracker
has its own few extra settings, listed below.

### Settings: shared, and per tracker

| Setting | `two_phase` | `trackcorr` | `fast_3d` |
|---|---|---|---|
| `dvx*`, `dvy*`, `dvz*` (mm/frame) | yes (as the radius `v_max`: the largest of the six) | yes | yes |
| `dacc` (mm/frame²), `angle` (gon) | no: it confirms links instead (`confirm_tol`, automatic) | yes | yes |
| `direction`, `postprocess` | `direction` yes; `postprocess` not used (it carries gaps and confirms inside the loop) | both | both |
| `holistic_rounds` | **yes (the add-on)** | not yet | not yet |
| `leaf_weight`, `max_gap`, `confirm_*`, `q_*`, `blob_gate` | yes | no | no |
| `epi_weight` | no | no | yes (0 = pure 3D; 1.0 is what the old "hybrid" used) |

`dacc` and `angle` on `fast_3d` are read by the kernel only when its acceleration gate is switched on; this is
tracked in the plan (stage 2) and until then `fast_3d` limits links by the velocity box and its cost.

## The two switches

**`direction: forward | both`.** Forward tracks the frames in order. `both` also runs the same tracker on the
reversed frame order (backward is only a different way of numbering the frames) and keeps what is consistent or
fills gaps. It is two passes, so it takes longer, and it gives the better trajectories. Works for all three
trackers. In the GUI the *Track back* menu item exists for `trackcorr` and `fast_3d` runs that were made forward-only.

**`postprocess: true | false`.** After the passes: reciprocity check (a link must be found from both ends) and gap
bridging. For `trackcorr` and `fast_3d`. `two_phase` does these inside its own loop, so the switch has no effect
there.

## The add-on: `holistic_rounds`

The merged-blob loop. Where two particles overlap in a camera image, the detector gives one blob with a shifted
centre. The loop estimates a centre for each particle from its own track, re-triangulates, and links again;
`holistic_rounds: 2` repeats that twice. It costs run time. It works with `two_phase` today; asking for it with
another tracker is ignored with a log line. Details and measurements: [Holistic tracking](holistic-tracking.md).

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


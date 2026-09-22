# Particle Trackers in openptv2

Start here if you want to know which tracker to use and why. For measured
numbers on a reference dataset, see [Tracker Tutorials](tracker-tutorials.md);
for pipeline and file formats, see the [Tracking Guide](tracking_guide.md);
for Two-Phase usage in depth, see [Two-Phase Tracking](two-phase-tracking.md).

## 1. The basics: what tracking is

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

## 2. All trackers at a glance

Select with `plugins.selected_tracking` in your parameters (GUI: Plugins
page). The `name` column is the exact preset string.

| preset (`name`) | idea in one line | looks at |
|---|---|---|
| `default`, `standard_forward`, `full_multipass`, `two_directional` (trackcorr) | Guess forward, confirm in every camera, accept smooth links | 2D targets + 3D |
| `priority_segment_3d` (Fast 3D / 3MA) | Cheapest (smoothest) links first, globally | 3D only |
| `4be` | Peek at frame n+2 before accepting; conflicts link to nobody | 3D only |
| `nearest_hungarian_3d` (MyPTV 3D) | Best total pairing per frame pair (Hungarian), survives gaps | 3D only |
| `myptv_2d_tracking` (MyPTV 2D) | Each camera tracks its own images, then cameras vote | 2D per camera |
| `predictive_gmm_3d` (proPTV) | Fit a smooth curve through history, predict from it | 3D only |
| `two_phase` (Two-Phase) | 3D search for candidates, per-camera images for ranking | 3D + 2D |
| `hybrid_deltat_3d` (Hybrid) | Match every N-th frame where motion beats noise, fill between | 3D only |

Details per tracker live in `src/openptv2/tracking_registry.py`
(`TRACKER_REGISTRY`) — the machine-readable version of this page.

## 3. Each tracker: caveats, tips, tricks

### trackcorr (`default`, `standard_forward`, …) — the original engine
The safest default. Predicts forward, checks candidates in every camera
image, accepts links with small acceleration and turning angle, and can run
backward plus gap relinking. **Caveat:** slowest of the bunch, and three
interacting parameters (`dvxmax`, `dacc`, `angle`). **Tip:** on noisy slow
flow, switch the angle limit off — the measured turning angle is mostly
noise then. If a particle is hidden in one camera, trackcorr usually still
gets it through the others.

### Fast 3D / 3MA (`priority_segment_3d`) — the fast one
Hundreds of thousands of particles per second, simplest mental model
(smoothest join wins). **Caveat:** blind to the images, so a ghost particle
sitting where the guess expects is accepted without question. **Tip:** use
it for clean, dense data and high throughput; distrust it where ghosts are
likely (poor calibration, few cameras).

### 4BE (`4be`) — the careful one
Looks one frame into the future before committing, and refuses contested
dots outright. **Caveat:** built for sparse clean data; on noisy/dense data
it produces the worst accelerations of all engines and leaves gaps
deliberately (gap bridging is off for its preset on purpose — it would
rebuild exactly the links 4BE declined). **Tip:** sparse lab data with
reliable detection; not turbulence.

### MyPTV 3D (`nearest_hungarian_3d`) — the fair one
Nobody grabs the nearest dot first: it finds the pairing with the lowest
*total* distance, so nobody is paired badly. Tracks survive short gaps
(`max_gap`), code is plain readable Python. **Caveat:** one frame pair at a
time — it cannot use what happens next. **Tip:** good first alternative to
the default; easiest engine to modify (`src/openptv2/plugins/myptv_3d_tracking.py`).

### MyPTV 2D (`myptv_2d_tracking`) — the per-camera voter
Each camera tracks its own movie; links with the most camera votes win.
**Caveat:** a link needs only one vote, so a single confused camera can
still create a bad link. **Tip:** reaches for it when 3D triangulation is
unreliable but the raw images are clean.

### proPTV (`predictive_gmm_3d`) — the smoother
Fits smooth curves through each path, so speeds and accelerations stay
sensible under noise. **Caveat (read first):** this port predicts from the
*smoothed current position*, not from an extrapolated one, and its search
radius is the same with or without history — so it currently behaves closer
to smoothed nearest-neighbour than to the predictive scheme of the paper.
Also dense data gets expensive. **Tip:** check the plugin README before
trusting its "predictive" label; fix the extrapolation first if you build
on it.

### Two-Phase (`two_phase`) — the hybrid
3D search lists candidates, per-camera image distances rank them, Hungarian
per connected group decides. No motion model — immune to bad guesses, but
fails once motion outruns particle spacing. Reported +74% multi-frame
trajectories over the default on a poorly-conditioned aorta dataset.
**Caveat:** needs Cet 2D targets per camera in the store; falls back to
pure 3D (`leaf_weight=0`) without them. **Tips:** start with
`leaf_weight=1`, `v_max` at ~3× your typical step; see
[Two-Phase Tracking](two-phase-tracking.md) for the full parameter guide
including the shared-observation prototype flags.

### Hybrid multi-Δt (`hybrid_deltat_3d`) — the slow-flow specialist
When particles crawl, frame-to-frame steps drown in noise — so it matches
every N-th frame (where displacement beats noise) and fills the middle with
a smooth curve. **Caveat:** the only engine that changes the signal-to-noise
ratio instead of fighting it, but the smooth fill is wrong for fast or
curved motion. **Tip:** high frame rate + slow flow; set `stride` so the
coarse step clearly exceeds your 3D noise floor.

## 4. Upstream credit: MyPTV and proPTV are plugins, not forks

Two engines borrow ideas from outside projects. **We use them as plugins —
adapted concepts on openptv2's own data structures, not modified copies of
their code.** The full frameworks (triangulation pipelines, calibration,
backtracking/repair, smoothing toolboxes) live only in their own
repositories — use those projects directly if you need them. Both are
permissively MIT-licensed.

- **MyPTV** by Ron Shnapp — open-source Python 3D-PTV library.
  Repository: <https://github.com/ronshnapp/MyPTV> ·
  Paper: Shnapp, R. (2022). *MyPTV: A Python Package for 3D Particle
  Tracking.* Journal of Open Source Software, 7(75), 4398.
  <https://doi.org/10.21105/joss.04398> ·
  What we adapted: per-camera 2D image-space tracking with multi-camera
  consensus (`myptv_2d_tracking`), and kinematic prediction + assignment
  matching in 3D (`nearest_hungarian_3d`).
- **proPTV** by Robin Barta and colleagues (DLR) — probabilistic PTV
  framework, Python.
  Repository: <https://github.com/RobinBarta/proPTV> ·
  Paper: Barta, R. et al. (2024). *proPTV: A probabilistic particle
  tracking velocimetry framework.* Journal of Computational Physics, 514,
  113212. <https://doi.org/10.1016/j.jcp.2024.113212> ·
  What we adapted: the small pure-numpy core (Gaussian-mixture / basis
  approximation and Savitzky–Golay smoothing, vendored under
  `src/openptv2/plugins/proptv/`), wired into the `predictive_gmm_3d`
  plugin. The original's triangulation, probability model, backtracking
  and repair are *not* ported.
- The classic engines descend from the OpenPTV/liboptv lineage
  (<http://www.openptv.net>).

If you publish with these methods, please cite the original authors above
in addition to openptv2.

# Detection that keeps its doubts: blob hypotheses, conflict sites, and physical merge/split events

Date: 2026-11-21. Follows `2026-11-21-what-we-missed.md` (M1 becomes stage D1 here) and
`2026-11-21-holistic-next-phase.md` (the holistic loop becomes one of the users of the hypotheses).

## 0. Where we stand, and the main point

- **We are sparse.** Real wp2 has 0.0057 particles per pixel (ppp) and lv-multi wp4 has 0.0053 ppp, counting
  detected blobs only (`scratchpad/ppp.py`). The true density is somewhat higher, about 0.007 ppp, because merged
  and missed particles are not counted. Image-based Lagrangian particle tracking (Shake-The-Box and similar) runs at
  0.05-0.1 ppp, ten times denser.
- **Even so, centroids lose 7-15 % of particles to merging.** Random placement at 0.0057 ppp puts a neighbour within
  2-3 px of 1 - exp(-ρπd²) = 7-15 % of particles. That matches the roughly 13 % merged rate on wp2. The loss comes
  from geometry, not from detector settings, and at 0.05 ppp it would reach 50-75 %.
- **Our synthetic benchmark is denser than the real data** (27 % merged against about 13 %). It overweights merging,
  which is why gains on it shrink on real data.
- **At real density, merging is not the only loss.** About 11-15 % of true particles have no blob at all (too faint),
  and the calibration floor sets the position noise.
- **The main point.** Detection makes one hard decision per blob ("this is one particle at this centre") and then
  throws the image away. Every later step works on that summary and cannot recover what was lost. The two big losses,
  merged blobs and missing faint particles, are both detection doubts that were silently resolved the wrong way.

## 1. Goal and principles

1. **Quality first, then density.** First raise track quality at today's density (about 0.005 ppp). Then raise the
   seeding step by step (0.005 -> 0.01 -> 0.02 -> 0.04 ppp, synthetic first) and find where each part breaks.
2. **Learn from image-space methods; do not rebuild Shake-The-Box.** We take three ideas from them: a measured
   per-camera particle image shape (point-spread function, PSF), explaining a blob by the sum of the particles in it,
   and using the track prediction to resolve the image. We do not take the global loop that re-renders and shakes
   every particle in every frame.
3. **Keep the doubt, resolve it lazily.** Detection writes its best answer, exactly as today, plus the alternatives
   it was unsure about, each with a probability. Nothing downstream changes unless a later step needs an
   alternative. Only then is it used, and only at that place.
4. **One mechanism for tracers and for real physics.** "Two tracers overlap in one camera" and "two colloidal
   particles stick together" both look like two tracks entering one blob. They differ in 3D and in time, and only
   detection, stereo matching and tracking together can tell them apart. The same machinery should give continuous
   tracer tracks in the first case and a recorded event in the second.

## 2. Detection that keeps its doubts

### 2.1 What detection writes

For each blob, the primary target exactly as today (centre, pixel count, size, grey sum), plus a small hypothesis
record:

| Field | Meaning |
|---|---|
| `p_single` | probability that the blob holds one particle |
| `k2: (x1, y1, I1, x2, y2, I2, p)` | best two-particle explanation: two centres, two brightnesses, its probability |
| `k3: ... , p` | three-particle explanation, only when `k2` fits clearly better than one particle |
| `shape: (sxx, sxy, syy)` | brightness-weighted second moments (size and elongation) |

Plus a list of **faint candidates**: local maxima above a lower threshold (for example half of `gvthres`) that did
not make a target, with centre, brightness and a probability. They are what a track or a missing camera can ask
for.

The primary target list stays the blobs with `p_single` high or no better alternative. So matching and tracking see
the same input as today, and the hypotheses cost nothing unless used.

### 2.2 How it is computed, fast

All of this comes from the region-growing loop in `segmentation._vectorized_targ_rec` (one pass over the blob's
pixels), plus a few arithmetic operations per blob:

1. **Second moments in the same loop.** Add sums of `x²`, `xy`, `y²` weighted by grey value next to the existing
   weighted sums. The cost is three more additions per pixel.
2. **A per-camera PSF from the data.** Take isolated, unsaturated blobs (no neighbour within 6 px, mid brightness)
   and record their second moments. Their median is the camera's particle image size σ²_psf (and its spread). This is
   the PSF idea from image-space methods, done with numbers we already have and without an optical model.
3. **Split by moments (closed form).** A single particle has covariance close to σ²_psf in every direction. Two
   particles at distance d along an axis add d²·w₁w₂ to the variance along that axis (w are the brightness shares).
   So the excess variance along the major axis, λ₁ - σ²_psf, gives the separation, and the third moment along that
   axis gives the brightness ratio. That yields `k2` with no iteration. A short two-Gaussian least-squares fit on the
   blob's own pixels (about 10-30 pixels, 6 unknowns) refines it only for the blobs that are kept as doubtful.
4. **Peaks inside the blob.** Today the region growing absorbs a second local maximum when the dip between them is
   smaller than `discont`. Record those absorbed maxima instead of losing them. Two maxima are strong evidence for
   `k2`. A merged blob with a single maximum is the hard case that the moments handle.
5. **Probabilities.** A small calibrated model (logistic regression on: excess variance / σ²_psf, eccentricity,
   number of internal maxima, brightness against the local median, pixel count) trained on `syn1`, where truth
   tells how many particles each blob holds. On real data the same features are used, with σ²_psf and the brightness
   scale taken from the run itself.

Cost: a few more operations per pixel in a loop that runs anyway, and a fit only on the doubtful 10-25 % of blobs.

### 2.3 Where it is stored

New optional arrays in `RunStore` next to `targets/`: `target_hyp/cam_{c}/{frame}` (one row per blob, columns as in
2.1, keyed by the target's `pnr`) and `faint/cam_{c}/{frame}`. Old stores without them work as now. Writers and
readers live beside `write_targets` / `read_targets` in `storage/run_store.py`.

## 3. Using the doubts where they matter (conflict sites)

A later step asks for an alternative only at a **conflict site**. These are a few percent of all observations,
which keeps the method fast.

| Conflict site | What triggers it | What the alternative gives |
|---|---|---|
| **Stereo: a camera is missing** | a 3D candidate is seen in 2-3 cameras; in the remaining camera its projection falls on a blob already used, or on nothing | a `k2` centre on the epipolar line inside a used blob, or a faint candidate there; the point gets that camera with a larger uncertainty |
| **Tracking: a prediction lands in a taken blob** | two tracks predict into the same blob | the `k2` pair, assigned to the two tracks by their predictions; both tracks continue with their own centres instead of one ending |
| **Tracking: a prediction lands on nothing** | a good track has no blob within the search radius | a faint candidate near the prediction; the gap is filled instead of the track ending |
| **Identity swap** | a track zig-zags: a jump in one direction followed by a jump back, or acceleration far above its own level, next to another track | the `k2` record of the blobs along the jump; the two tracks are re-assigned so both become smooth |
| **Split or merge of tracks** | one track ends where another starts, or two tracks end where one starts (or the reverse) | the event test of section 4 |

The first two are what the holistic loop already does by inference from predictions (`docs/holistic-tracking.md`).
The hypotheses give it measured sub-blob centres to check against, instead of positions inferred only from the
tracks.

## 4. Physical merge and split versus optical overlap

### 4.1 The four explanations

When tracks meet a shared blob, four explanations compete:

| | What happens in 3D | What the cameras see |
|---|---|---|
| **O. Optical overlap** (tracers) | two particles pass each other at a distance larger than their size | merged blob in one or two cameras, at frames that differ between cameras; separate blobs in the others |
| **C. Coalescence** (agglomeration) | two particles touch and from then on move as one | one blob in every camera from the same frame on, for many frames; the new centre is the brightness-weighted mean of the two |
| **B. Breakup** | one object breaks into two (or three) | one blob in every camera, then two (three) from the same frame on, starting from the same 3D point and moving apart |
| **S. Over-segmentation** | one irregular object (an aggregate) | its blob splits into two at the threshold in some cameras or frames only |

### 4.2 What tells them apart

- **Brightness alone cannot.** In O and C the merged blob's brightness is the sum of the two. Brightness tells that
  a blob holds two particles, not whether they touch.
- **3D separation.** In O, the cameras where the two particles are still separate give two 3D points, and their
  distance stays larger than the particle size. In C it falls to the contact distance at the merge frame.
- **Same frame in every camera.** A physical event happens at one moment, so all cameras show it at the same frame.
  Optical overlaps happen at different frames in different cameras, because they depend on the viewing direction.
- **Persistence.** An optical overlap lasts as long as it takes the two blobs to pass each other on the sensor
  (blob size divided by relative image speed, a few frames). An agglomerate persists.
- **Conservation at the event.** At coalescence the merged centre is the brightness-weighted mean of the two parents
  at contact, and the merged brightness is their sum (within the run's own spread). At breakup the same holds in
  reverse: the children's brightness adds up to the parent's, and the children start from the parent's position.
- **Shape.** An agglomerate is a larger, compact blob in every camera. An optical overlap is elongated along the line
  between the two particles in the overlap camera only. Over-segmentation (S) shows splits that come and go between
  frames and disagree between cameras.

### 4.3 Finding the event moment by following tracks forward and backward

For a candidate breakup: take the two child tracks after the split and the parent track before it. Extrapolate the
children backward and the parent forward. For each candidate frame t₀ in a window, fit one model: parent path up to
t₀, children from the parent position at t₀ with free velocities, brightness conserved, with the observation model
of the holistic loop (a blob is explained by the particles that project into it). The best t₀ is the one with the
lowest cost. Coalescence is the same in reverse time. The overlap hypothesis O is the same fit with two independent
paths that never touch. The decision compares the costs, with a penalty for the extra event, and is written with its
probability.

This is a small, exact problem per site (2-3 tracks, about 20 frames, 4 cameras), solved only at the few sites where
tracks end and start next to each other.

### 4.4 Output

Trajectories become a **lineage graph**: track segments plus an event table (type O/C/B/S, frame, 3D position,
segments in, segments out, probability). For tracers, O events are resolved inside the tracker and the user simply
gets continuous tracks. For colloids, C and B events are the scientific output (breakup and aggregation rates,
fragment size distributions), and the segments carry brightness as a size proxy.

## 5. What we take from image-space methods, and what we leave

| Idea | Image-space methods (Shake-The-Box, IPR) | Here |
|---|---|---|
| Particle image shape | optical transfer function calibrated per camera and volume region | per-camera σ²_psf from isolated blobs, updated per run |
| Overlapping particles | fit the image of all particles jointly, every frame | split only doubtful blobs, from moments, refined by a small fit; used only at conflict sites |
| Track-guided detection | predicted particles are shaken in the image, the residual image is searched for new ones | the prediction asks for a stored alternative (sub-blob centre or faint candidate); no residual images |
| Ghost removal | ghosts fade because the image does not support their intensity | brightness accounting in blob space (holistic loop) |
| Physical merge and split | not a target of these methods | first-class events with a recorded probability |

The pixels are read once, at detection, in the loop that already runs. A later stage (D6) may re-read small windows
around predictions for faint particles, but there is never a global image loop.

## 6. Related work to read before building (from memory; check the papers)

- **u-track** (Jaqaman et al., Nature Methods 2008): single-particle tracking with explicit merge and split events,
  using an intensity cost (the merged intensity should equal the sum). The closest existing 2D formulation of our
  section 4.
- **Conservation tracking** (Schiegg et al., ICCV 2013; ilastik): a detection may hold several objects (a count
  variable), chosen jointly with tracking. Once the count is known, the blob is split by fitting that many Gaussians.
  That is "keep the doubt, resolve later" in a published, tested form.
- **Joint segmentation and tracking with segmentation hypotheses** (Jug et al., 2014, mother machine; Arteta et al.,
  MICCAI 2012, extremal regions): a tree of blobs at several thresholds; tracking picks the consistent set. Our
  internal-maxima record (2.2.4) is a small version of this tree.
- **Aggregate breakup by 3D-PTV** (ETH group, for example Saha et al., Langmuir 2016): the experimental side of the
  colloid case and the quantities users want out of it.
- **Shake-The-Box / IPR** (Schanz et al., Wieneke) and **OpenLPT** (open source): to compare with at the density
  ramp, and for the PSF practice.

## 7. Stages, each measured; stop rules are part of the plan

Scoring as in `2026-11-21-what-we-missed.md`: one scorecard (`scripts/synth_bench.py eval`, `bench/*.json`), the
same cases for every claim; a stage that gains only by dropping true points is reverted.

**D0. Truth and density cases (first; hours to a day)**
- Re-render `syn1` at the real merged rate (about 13 %) next to the current 27 %, so gains are measured where the
  real data is.
- Density ladder for later: 0.005, 0.01, 0.02, 0.04 ppp.
- Per-camera σ²_psf on `syn1` and on wp2 (scratch copy): does the real PSF behave like the rendered one?

**D1. The hypothesis detector, scored against truth (replaces M1)**
- Second moments, internal maxima, moment split, faint candidates, calibrated probabilities (section 2).
- Score per blob against `syn1/truth/truth.npz`: for merged blobs, the error of the `k2` centres against the true
  centres; for faint candidates, recall and false-candidate rate; calibration of `p_single` (reliability curve).
- Compare with the detectors on `feat/detection-prototypes-dask-trackpy-skimage-proptv` on the same scorecard.
- **Gate:** `k2` centres within 0.5 px of truth for at least half of the two-particle blobs; `p_single` calibrated
  within 0.1; faint candidates recover at least a third of the particles with no blob at fewer than 2 false
  candidates per true one. If the `k2` error is no better than the 0.27 px track prediction, the split adds nothing
  over inference and we stop at the faint candidates.

**D2. Stereo and tracking use the alternatives (conflict sites 1-3)**
- Missing-camera completion in matching; shared-blob claims and faint-candidate gap filling in `two_phase` and in
  the holistic loop.
- **Gate (benchmark at the real merged rate):** yield +5 points and long-track completeness +8 points over
  `two_phase` with holistic rounds, velocity error not higher; on wp2: leave-one-camera-out error not higher, tracks
  of 50 frames or more clearly more.

**D3. Identity-swap repair (conflict site 4)**
- Zig-zag detector plus re-assignment through the `k2` record.
- **Gate:** on truth, swaps per 1000 track frames down by half; on wp2, acceleration flatness (M6) lower.

**D4. Physical events (section 4)**
- Synthetic truth first: extend `benchmarking/scenario.py` with particles of different sizes that coalesce
  (brightness-weighted centre, brightness sum) and break up (children from the parent position), rendered through
  the same camera rig. Mix with optical overlaps of tracers.
- Event-window solver and lineage graph output.
- **Gate:** event type right for at least 90 % of C and B events and for at least 95 % of O sites (tracers must not
  be turned into fake aggregates); event frame within ±1 frame for at least 80 % of C and B.

**D5. Density ladder**
- Run detection with hypotheses, `two_phase` + holistic, and the event solver at 0.005 -> 0.04 ppp. Find the density
  at which yield or velocity error break down, and which part breaks first. That density is the recommended seeding
  limit, and the part that breaks is the next thing to fix.

**D6. Only if D5 shows it is needed: small-window image re-reads**
- Around a prediction, re-detect with a local threshold or fit the known PSF to a 9x9 window. Still no global
  image loop.

## 8. Speed budget

Detection: at most +20 % (moments in the existing loop; fits on doubtful blobs only). Matching and tracking: at most
+50 % over today, since alternatives are consulted only at conflict sites. Event solver: per site, run only where
tracks end and start next to each other. Total pipeline at most 1.5x today at 0.005 ppp; measured at every rung of
the density ladder.

## 9. Questions for the user

1. Is there a colloid or aggregate dataset (real or rendered) to aim D4 at? Particle size against pixel size, and
   whether aggregates are resolved (larger than the PSF) or unresolved, change the shape test in 4.2.
2. For colloids, should brightness be treated as a size proxy (with its scattering law), or only checked for
   conservation within the run's spread?
3. Order: D1 + D2 first for tracers (the near-term quality gain), with D4 after; or D4 synthetic truth in parallel
   because the colloid case is the scientific goal?

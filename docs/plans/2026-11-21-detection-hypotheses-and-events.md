# Detection that keeps its doubts: blob hypotheses for overlapping tracers

Date: 2026-11-21. Follows `2026-11-21-what-we-missed.md` (M1 becomes stage D1 here) and
`2026-11-21-holistic-next-phase.md` (the holistic loop becomes one of the users of the hypotheses).

**Scope.** Tracers only. The goal is a 3D-PTV tracker (not Shake-The-Box) that is clearly better than the older
libptv `trackcorr`-based solution. Colliding or breaking particles are only a picture that helps think about
overlaps; overlapping tracers do not behave like them. They pass each other and keep their own paths.

## 0. Where we stand, and the main point

- **We are sparse.** Real wp2 has 0.0057 particles per pixel (ppp) and lv-multi wp4 has 0.0053 ppp, counting
  detected blobs only. The true density is about 0.007 ppp, since merged and missed particles are not counted.
  Image-based tracking (Shake-The-Box and similar) runs at 0.05-0.1 ppp.
- **Even so, centroids lose 7-15 % of particles to overlap.** Random placement at 0.0057 ppp puts a neighbour within
  2-3 px of 1 - exp(-ρπd²) = 7-15 % of particles, which matches the roughly 13 % merged rate on wp2. This comes from
  geometry, not detector settings, and it grows quickly with seeding (50-75 % at 0.05 ppp).
- **Our synthetic benchmark is denser than the real data** (27 % merged against about 13 %), so it overweights
  overlaps, and gains measured on it shrink on real data.
- **At real density, overlap is not the only loss.** About 11-15 % of true particles have no blob (too faint), and the
  calibration floor sets the position noise.
- **The main point.** Detection makes one hard decision per blob ("one particle, this centre") and throws the image
  away. Overlapped tracers and faint tracers are both detection doubts that were silently resolved the wrong way,
  and no later step can undo that.

## 1. Goal and principles

1. **Better than `trackcorr`, measured.** Every stage is scored against legacy `trackcorr` (and our current best,
   `two_phase` with holistic rounds) on the same cases and metrics.
2. **Quality first, then density.** First raise track quality at today's density (about 0.005 ppp). Then raise the
   seeding step by step (0.005 -> 0.01 -> 0.02 -> 0.04 ppp, synthetic first) and find where each part breaks.
3. **Learn from image-space methods; do not rebuild Shake-The-Box.** We take a measured per-camera particle image
   shape, the idea that a blob is explained by the tracers inside it, and the use of track predictions to resolve the
   image. We do not take the global loop that re-renders and shakes every particle in every frame.
4. **Keep the doubt, resolve it lazily.** Detection writes its best answer exactly as today, plus the alternatives it
   was unsure about, each with a probability. Later steps use an alternative only where they need one.
5. **Only detection, stereo and time together decide.** Whether a blob is one tracer or two overlapping tracers is
   often undecidable in one image. Other cameras (where the two are apart) and the tracks before and after (two
   smooth paths entering and leaving) decide it.

## 2. Detection that keeps its doubts

### 2.1 What detection writes

For each blob, the primary target exactly as today, plus a small hypothesis record:

| Field | Meaning |
|---|---|
| `p_single` | probability that the blob holds one tracer |
| `k2: (x1, y1, x2, y2, share, p)` | best two-tracer explanation: two centres, the grey share of the first, its probability |
| `k3: ..., p` | three-tracer explanation, only when `k2` fits clearly better than one tracer |
| `shape: (sxx, sxy, syy)` | grey-weighted second moments (size and elongation) |
| `brightness` | mean intensity, `sumg / n` |

Plus a list of **faint candidates**: local maxima above a lower threshold (for example half of `gvthres`) that did
not become targets, with centre, brightness and a probability.

The primary target list does not change, so matching and tracking see the same input as today. The hypotheses
cost nothing unless a later step uses them.

### 2.2 Brightness and size

- **Brightness is mean intensity**, `sumg / n`. For one tracer it is a property of the tracer and its distance from the
  light sheet, so it changes slowly along a track and works as the tracer's signature in each camera.
- **Overlap shows mainly in size and shape.** Two overlapping tracer images give a blob with more pixels and an
  elongated shape, while its mean intensity stays close to that of a single tracer (a little higher where the two
  images add up).
- **The centre split needs the grey shares.** A blob's centre is the grey-weighted mean of the tracers inside it, so
  splitting it uses each tracer's share of the summed grey value (its mean intensity times its area), not the mean
  intensity alone.
- **Today the holistic loop uses the summed grey value** (`sumg`) as signature (`holistic.estimate_signatures`). D1
  measures which cue separates one tracer from two best on truth: mean intensity, pixel count, the second moments,
  or the summed value. The signature follows the result.

### 2.3 How it is computed, fast

All of this comes from the region-growing loop in `segmentation._vectorized_targ_rec`, which already visits every
pixel of every blob once:

1. **Second moments in the same loop.** Add grey-weighted sums of `x²`, `xy` and `y²` next to the existing
   weighted sums: three more additions per pixel.
2. **A per-camera particle image size from the data.** Take isolated, unsaturated blobs (no neighbour within 6 px)
   and record their second moments. Their median is the camera's single-tracer size σ²_psf, and their spread tells
   how much a single tracer varies. This is the image-space point-spread-function idea, done with numbers we
   already have.
3. **Split by moments (closed form).** Two tracers at distance d with grey shares w₁ and w₂ add w₁w₂d² to the
   variance along the line joining them. So the excess along the major axis, λ₁ - σ²_psf, gives the separation, and
   the third moment along that axis gives the share. That yields `k2` with no iteration. A short two-Gaussian fit on
   the blob's own pixels (10-30 pixels) refines it only for blobs kept as doubtful.
4. **Peaks inside the blob.** Today the region growing absorbs a second local maximum when the dip between them is
   smaller than `discont`. We record that maximum instead of losing it. Two maxima are strong evidence for `k2`. A
   merged blob with a single maximum is the hard case that the moments handle.
5. **Probabilities.** A small calibrated model (logistic regression on excess variance over σ²_psf, eccentricity,
   number of internal maxima, mean intensity against the tracer population, pixel count), trained on `syn1`, where
   truth tells how many tracers each blob holds. On real data σ²_psf and the brightness scale come from the run.

Cost: a few more operations per pixel in a loop that runs anyway, plus a fit for the doubtful 10-25 % of blobs.

### 2.4 Where it is stored

New optional arrays in `RunStore` next to `targets/`: `target_hyp/cam_{c}/{frame}` (one row per blob, keyed by the
target's `pnr`) and `faint/cam_{c}/{frame}`. Old stores without them work as now. Readers and writers sit beside
`write_targets` / `read_targets` in `storage/run_store.py`.

## 3. Using the doubts where they matter (conflict sites)

An alternative is consulted only at a **conflict site**. These are a few percent of all observations, which keeps
the method fast.

| Conflict site | What triggers it | What the alternative gives |
|---|---|---|
| **Stereo: a camera is missing** | a 3D candidate seen in 2-3 cameras projects, in the remaining camera, onto a blob already used, or onto nothing | a `k2` centre on the epipolar line inside the used blob, or a faint candidate there; the point gets that camera with a larger uncertainty |
| **Tracking: a prediction lands in a taken blob** | two tracks predict into the same blob | the `k2` pair, assigned to the two tracks by their predictions; both tracks continue with their own centres instead of one ending |
| **Tracking: a prediction lands on nothing** | a good track has no blob within the search radius | a faint candidate near the prediction; the gap is filled instead of the track ending |
| **Identity swap** | a track zig-zags (a jump and a jump back, or acceleration far above its own level) next to another track | the `k2` record of the blobs along the jump; the two tracks are re-assigned so both become smooth |
| **Overlap window** | tracks end, break or swap where they meet a shared blob | the overlap-window solver of section 4 |

The first two are what the holistic loop already does by inference from predictions (`docs/holistic-tracking.md`).
The hypotheses give it measured sub-blob centres to check against, instead of positions inferred from the tracks
only.

## 4. Resolving an overlap window

### 4.1 What can explain a shared blob

| Explanation | In 3D | What the cameras see |
|---|---|---|
| **Overlap** (the common case) | two tracers pass each other at a distance larger than their image size | a merged blob in one or two cameras, at frames that differ between cameras; separate blobs in the others; afterwards both continue on their own smooth paths |
| **Split image of one tracer** | one tracer | its blob breaks into two at the threshold in some frames or cameras only (noise, uneven illumination) |
| **Ghost** | no tracer | a stereo candidate that borrows a blob already explained by real tracers |

### 4.2 What tells them apart

- **Other cameras.** Where the two tracers are still apart, they give two 3D points whose distance stays larger
  than the image size. That settles "overlap" in most cases (cameras plus nearby frames resolved 76 % of merged
  observations on the benchmark, `2026-11-19-joint-tracker.md`).
- **Frames differ between cameras.** An overlap depends on the viewing direction, so it starts and ends at different
  frames in different cameras. A blob that is merged in every camera at once needs the tracks to decide.
- **Duration.** An overlap lasts about blob size divided by relative image speed: a few frames.
- **Tracks before and after.** Two smooth paths enter the shared blob and two leave it. Following them forward and
  backward fixes the first and last shared frames in each camera.
- **Size and shape.** In the overlap camera the blob is larger and elongated along the line between the predicted
  positions; its mean intensity stays near a single tracer's.

### 4.3 The solver

For each window (2-3 tracks, about 20 frames, all cameras): fit both tracers' 3D paths through the window, with the
observation model of the holistic loop (a blob is explained by the tracers projecting into it, its centre is their
grey-weighted mean) and the `k2` centres as extra measurements. Compare with "one tracer" and "one of the two is a
ghost", with a penalty for each extra tracer. Keep the best, write its probability, and pass the two continuous
tracks on.

This is a small exact problem, solved only at the few windows where tracks meet a shared blob.

## 5. What we take from image-space methods, and what we leave

| Idea | Image-space methods (Shake-The-Box, IPR) | Here |
|---|---|---|
| Particle image shape | optical transfer function calibrated per camera and volume region | per-camera σ²_psf from isolated blobs, per run |
| Overlapping tracers | fit the image of all particles jointly, every frame | split only doubtful blobs, from moments plus a small fit; used only at conflict sites |
| Track-guided detection | predicted particles shaken in the image; residual image searched for new ones | the prediction asks for a stored alternative (sub-blob centre or faint candidate); no residual images |
| Ghost removal | ghosts fade because the image does not support their intensity | blob accounting in blob space (holistic loop) |

The pixels are read once, at detection, in the loop that already runs. A later stage (D6) may re-read small windows
around predictions for faint tracers, but there is never a global image loop.

## 6. Related methods to read before building (from memory; check the papers)

- **u-track** (Jaqaman et al., Nature Methods 2008): 2D tracking with explicit merge and split events and an
  intensity cost; a tested formulation of tracks meeting in one blob.
- **Conservation tracking** (Schiegg et al., ICCV 2013; ilastik): a detection may hold several objects (a count
  variable), chosen jointly with tracking; once the count is known, the blob is split by fitting that many
  Gaussians. That is "keep the doubt, resolve it later" in published form.
- **Segmentation hypotheses** (Jug et al. 2014; Arteta et al., MICCAI 2012): a tree of blobs at several thresholds,
  from which tracking picks a consistent set. Our internal-maxima record is a small version of this tree.
- **Shake-The-Box / IPR** (Schanz et al., Wieneke) and **OpenLPT** (open source): comparators at the density ladder,
  and their point-spread-function practice.

## 7. Stages, each measured; stop rules are part of the plan

Scoring as in `2026-11-21-what-we-missed.md`: one scorecard (`scripts/synth_bench.py eval`, `bench/*.json`), the
same cases for every claim. Baselines in every table: legacy `trackcorr` and `two_phase` with holistic rounds. A
stage that gains only by dropping true points is reverted.

**D0. Truth cases and baselines (first; about a day)**
- Re-render `syn1` at the real merged rate (about 13 %) next to the current 27 %, so gains are measured where the
  real data is.
- Density ladder for later: 0.005, 0.01, 0.02, 0.04 ppp.
- Per-camera σ²_psf on `syn1` and on wp2 (scratch copy): does the real particle image behave like the rendered one?
- Legacy `trackcorr` scored on all cases, so "better than trackcorr" has numbers.

**D1. The hypothesis detector, scored against truth (replaces M1)**
- Second moments, internal maxima, moment split, faint candidates, calibrated probabilities (section 2), and the
  comparison of cues (mean intensity, pixel count, moments, summed grey value).
- Score per blob against `syn1/truth/truth.npz`: `k2` centre error against the true centres; recall and false rate of
  faint candidates; calibration of `p_single`.
- Compare with the detectors on `feat/detection-prototypes-dask-trackpy-skimage-proptv` on the same scorecard.
- **Gate:** `k2` centres within 0.5 px of truth for at least half of the two-tracer blobs; `p_single` calibrated
  within 0.1; faint candidates recover at least a third of the tracers with no blob, with fewer than 2 false
  candidates per true one. If the `k2` error is no better than the 0.27 px track prediction, the split adds nothing
  over inference, and we keep only the faint candidates.

**D2. Stereo and tracking use the alternatives (conflict sites 1-3)**
- Missing-camera completion in matching; shared-blob claims and faint-candidate gap filling in `two_phase` and the
  holistic loop.
- **Gate (benchmark at the real merged rate):** yield +5 points and long-track completeness +8 points over
  `two_phase` with holistic rounds, velocity error not higher; on wp2: leave-one-camera-out error not higher, and
  clearly more tracks of 50 frames or more.

**D3. Identity-swap repair and the overlap-window solver (sections 3-4)**
- Zig-zag detector, re-assignment through the `k2` record, and the window solver.
- **Gate:** on truth, swaps per 1000 track frames down by half and tracks through overlaps continuous in at least
  80 % of cases; on wp2, acceleration flatness (M6) lower.

**D4. Density ladder**
- Run the whole chain at 0.005 -> 0.04 ppp against `trackcorr` and `two_phase`. Find the density at which yield or
  velocity error break down, and which part breaks first. That density is the recommended seeding limit, and the
  part that breaks is the next thing to fix.

**D5. Real data**
- wp2 (300 frames, then all 1072) and lv-multi on scratch copies: leave-one-camera-out error, tracks of 50 frames or
  more, acceleration flatness, and the Eulerian statistics against the MATLAB reference.

**D6. Only if D4 shows it is needed: small-window image re-reads**
- Around a prediction, re-detect with a local threshold or fit the known particle image to a 9x9 window. Still no
  global image loop.

## 8. Speed budget

Detection at most +20 % (moments in the existing loop, fits on doubtful blobs only). Matching and tracking at most
+50 %, since alternatives are consulted only at conflict sites. Whole pipeline at most 1.5x today at 0.005 ppp,
measured at every rung of the density ladder.

## 9. Results

### D0. Truth cases and baselines (done 2026-11-21)

Case `L1_d1_c0_k1_n200_s0.08_a0.18_f0.05_m0.75`, 200 frames, built with `MERGE_SCALE=0.75`. Data in
`bench/d0_baselines_real_merge_rate_2026-11-21.json`.

- **Merge rate.** Share of particle views inside a merged blob: 22.6 % at the default scale (1.0), 13.4 % at 0.75,
  9.3 % at 0.6. The real wp2 rate is about 13 %, so **0.75 is the real-like case** and is the main benchmark from
  here on. The share with no blob is about 16 % in all of them.
- **Particle image size.** `scripts/psf_estimate.py` (isolated blobs, median grey-weighted second moment): rendered
  `syn1` 0.35-0.38 px² and 7-9 pixels per blob; real wp2 0.25-0.28 px² and 5-6 pixels. Real tracers are smaller than the
  rendered ones, which is why the real merge rate is lower, and it agrees with the scale 0.75. The rendered images
  therefore overstate the overlap problem, and `D1` thresholds must come from each run's own estimate.

| tracker | velocity error | acceleration error | ghosts | long-track completeness | pieces per track | yield | time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle (perfect linking of the input) | 0.078 | 5.1 | 0.001 | 0.67 | 1.37 | 0.76 | 10 |
| `two_phase` + holistic rounds=2 | **0.084** | **5.1** | 0.027 | **0.74** | **1.16** | **0.79** | 157 |
| `two_phase` | 0.105 | 5.8 | 0.029 | 0.62 | 1.68 | 0.75 | 8 |
| `trackcorr` (our port) | 0.167 | 13.2 | 0.060 | 0.67 | 1.41 | 0.76 | 42 |
| legacy C `trackcorr` (`optv`, forward only) | 0.393 | 27.5 | 0.065 | 0.57 | 2.45 | 0.77 | 19 |

What it says:
- **We are already far better than legacy `trackcorr`:** velocity error 0.105 against 0.393 (3.7x lower), and
  0.084 with the holistic loop (4.7x), with fewer pieces per track and fewer ghosts.
- **The oracle is not a ceiling for the holistic loop.** The oracle links the input points perfectly but keeps the
  merged-blob positions as they are. The holistic loop corrects those positions, so it ends below the oracle in
  acceleration error and above it in long-track completeness. The room left is in the *input*, which is what D1-D3 target.
- **Cost of the holistic loop is 20x** (157 s against 8 s for the same 200 frames), far outside the speed budget of
  section 8. Any gain from D2/D3 has to be got without that cost, or by making the loop cheaper first.
- The yield is 0.75-0.79 in every method: the 16 % of particle views with no blob cap it, as the plan said.

D0 also restored the legacy comparator: `scripts/legacy_export.py` now holds the ASCII exporter and trajectory reader
that the zarr-only change removed, so `synth_bench.py track --tracker optv_c` works again.

### D1. The hypothesis detector, scored against rendered truth (measured 2026-11-21; first pass)

Code: `src/openptv2/blob_hypotheses.py` (cues from each blob's own pixels, one- and two-Gaussian fits, a small
logistic regression), `scripts/d1_features.py`, `d1_split.py`, `d1_classify.py`, `d1_faint.py`. Tests in
`tests/unit/test_blob_hypotheses.py`. The reference segmentation (`segmentation._vectorized_targ_rec`) gained an optional
`pixels_out` to hand out each blob's pixels; it finds exactly the blobs of the production detector (1298 against 1298
in the check frame). The production Cython detector is not changed yet. Rendered `syn1`, 4 cameras, frames 1/5 for
training and 9/13 for testing; blobs are labelled by how many true tracers fall inside them.

**Which cue says "more than one tracer"** (single cues, area under the ROC curve; 0.5 = nothing): major-axis variance
0.82, eccentricity 0.77, pixel count 0.75, summed grey value 0.71, **mean intensity 0.63**, internal maxima 0.62, third
moment 0.52. So mean intensity is a weak cue for overlap, as expected: overlap shows in size and shape, and the
signature of a tracer should stay `sumg / n` only as a *brightness* identity, not as the overlap detector.
Combined with the gain of the two-Gaussian fit over the one-Gaussian fit, the logistic model reaches **0.87**.

**Calibration of `p_single`** on the test frames: predicted against actual share of "two or more" in six bins agrees
to within 0.05 in five of them; the top bin (predicted 0.94, 113 blobs) has 0.79 actually, 0.15 off. At p above 0.3
the model catches 67 % of two-tracer blobs with 9 % false alarms among single-tracer blobs; at p above 0.5, 48 % with
3.6 %.

**Centre of the split** (two-tracer blobs, true count given, single-tracer variance 0.4 px² from the fit on clean
blobs, where the Gaussian fit gives a 0.06 px centre error):

| true separation | blob centroid error (px, median) | fitted centre error | fitted within 0.5 px |
|---|---:|---:|---:|
| under 1 px | 0.28 | 0.35 | 0.67 |
| 1-2 px | 0.75 | **0.24** | 0.71 |
| 2-3 px | 1.17 | **0.42** | 0.54 |
| over 3 px | 1.68 | 0.69 | 0.42 |
| all (600 blobs) | 0.67 | **0.33** | **0.62** |

**Faint candidates** (peaks at a lower threshold that are not within 3 px of a standard blob; 1114 true tracers had no
standard blob in the test frames): threshold 3 recovers 35 % of them with 1.5 false candidates per recovered one;
threshold 5 recovers 27 % with 1.2; threshold 7 recovers 17 % with 1.0.

**Against the gate**
- Split centres within 0.5 px for at least half of the two-tracer blobs: **met** (0.62), with the true count given.
- `p_single` calibrated within 0.1: **met in five of six bins**, not in the top bin.
- Faint candidates: at least a third of the missing tracers with fewer than 2 false per recovered: **met at threshold 3**
  (0.35 and 1.5), narrowly. A faint candidate by itself is not trustworthy; it is only used where a prediction or
  an epipolar line asks for one (section 3), which removes most of the false ones.
- Split centre error better than the 0.27 px track prediction: **not met on its own** (median 0.33; better only for
  separations of 1-2 px). The split is an independent measurement, so combined with the prediction the error falls to
  about 0.2 px if the two errors are independent, and the blob centroid it replaces is off by 0.67 px. D2 is the real
  test of that, and its stop rule applies.

**Caveats.** All of this is on rendered images in which the tracer is a Gaussian of known width, so it is optimistic for
real data, where the tracer image is smaller (0.25-0.28 px², D0) and its shape is less regular. The classifier trained on
rendered data cannot be transferred as it is; on a real run the cue weights have to come from that run (the
tracks themselves give labels: a blob a good track passes through while a second track also predicts into it).
Pixel count and the fits depend on the threshold and the particle image size, so they are re-estimated per camera and run
(`scripts/psf_estimate.py`).

**Decision.** Go on to D2 with the split as an extra, weighted measurement at conflict sites and faint candidates only
where a prediction asks for them; D2's gate decides whether it stays.

# Plan: bring two_phase's good ideas into trackcorr

Date: 2026-10-09. Status: plan, nothing implemented yet.
Related: `docs/plans/2026-09-30-tracker-plan.md` (benchmark facts),
`docs/two-phase-tracking.md` (how two_phase works).

## 1. Goal, in one paragraph

`trackcorr` is the 3DPTV tracker (Willneff), ported via liboptv. It uses the camera
images while it tracks: it can add a particle from unused 2D blobs where a track is
expected, and it looks two frames ahead. `two_phase` is younger, faster, and more
accurate on our benchmarks. We want one tracker that keeps trackcorr's strengths and
gains two_phase's. We do this in small steps. Each step is measured. Each step is
switched by a flag, and the old behaviour stays the default until the benchmark says
otherwise.

## 2. What each tracker does best

"3DPTV" is the original algorithm (Tcl/Tk `track.c`, liboptv). "openptv2 trackcorr"
is the same algorithm with the extensions openptv2 already has; each one has a
`tracking_run` switch, and the 3DPTV value of each switch is given.

| Topic | 3DPTV trackcorr | openptv2 trackcorr today | two_phase |
|---|---|---|---|
| Candidate search | Per particle: project an 8-corner search box into every camera, search each image (y-band scan), keep the 4 nearest blobs per camera, count how many cameras agree | Same, optionally with a uniform 2D grid per image (`use_grid`, 3DPTV = 0; switched on automatically when targets are not y-sorted) | KD-tree in 3D around the prediction, all candidates inside `v_max` |
| Prediction | `2·X1 − X0`; damped quadratic for the look-ahead; no history: `X2 = X1` | Same, but no history: neighbours' mean velocity (`cold_start_neighbour`, 3DPTV = 0) | Position + velocity × steps since last seen |
| Gates | One box (`dv*`), `dacc`, `dangle` | Plus a per-particle scale (`gate_scale`) with a stricter rule outside the base box | `v_max` ball |
| Score | `(dl/lmax + acc/dacc + angle/dangle) / quality` | Plus optional brightness term (`app_weight`, 3DPTV = 0) | Pixel distance of projected prediction to candidate blobs, or 3D distance |
| Conflicts | First come wins; a better score steals; the loser is dropped | Losers retry their next choices among unclaimed candidates (`loser_retry`, 3DPTV = 0) | Global assignment per connected group (Hungarian; greedy above 128 nodes) |
| Wrong links | Look-ahead score only; the fallback branch links without a look-ahead | Same | **Two-hop confirmation:** a link stays only if the next link continues smoothly |
| Gaps | A track ends; a missing point can be created from 2D blobs (`add_particle`, ≥ 2 cameras) | Same, but a free blob is needed in every camera | The track coasts on its prediction for up to `max_gap` steps |
| Ghosts | No rule | No rule | Young tracks may not start or continue on likely ghosts (`q_seed`, `q_young`) |
| Backward pass | `trackback`, adds links where both ends are free | Same | Forward-first merge, same rule |

## 3. What the benchmark already told us

From `docs/plans/2026-09-30-tracker-plan.md` and the synthetic benchmark
(`scripts/synth_bench.py`):

1. **Confirmation is why two_phase wins.** Its 2D leaf cost does not matter:
   `leaf_weight = 0` gives the same result.
2. **Ghost points cause most of the remaining error** (about 60% of two_phase's gap to
   a perfect linker) and nearly all wrong links.
3. **Most tracks have about one candidate.** Only 3–6% face a contested link. So a
   better score or a better assignment helps a little; candidate quality and
   confirmation help a lot.
4. **trackcorr's sharp jumps came from gates far too wide for slow particles, and from
   its greedy conflicts.** A tight `dacc` cut its velocity error from 0.51 to 0.18 on
   the clustered case.
5. Confirmation already exists as a post-pass for any tracker:
   `tracking_postprocess.confirm_links()`.

So the order of work below follows the expected gain, not the order of the code.

## 4. Before starting: the base must be correct

These were found and fixed on 2026-10-09 (branch `fix/brown-affine-scx-x-only`). A
benchmark run before them is not a valid baseline for this plan:

- The LUT bounds in the three inlined tracking projections (16–35 px wrong at the top
  of the volume).
- Kernel projections without a LUT skipped refraction (up to 228 px).
- `trackback` passed `Ymin = Ymax = 0`, so the backward pass never added a particle.
- `trackback` now adds a particle only with a free blob in every camera, like the
  forward pass.
- Added particles carry `prio = 2` again (3DPTV/liboptv meaning: "added by the
  tracker"), so the benchmark can count them.

**Step 0.** Re-run the benchmark baseline for `trackcorr` and `two_phase` on the
standard cases (including the real-jitter case `L1_d1_c0_k1_n200_s0.08`) and on the
real CompleteTest-e2e data. Store the numbers in this file.

## 5. Where the code is

- Driver: `src/openptv2/algorithms/track.py` (`trackcorr_c_loop`, `trackback_c`).
- Kernels: `src/openptv2/algorithms/track_kernels_corr.py`
  (`_trackcorr_particle_fast`, `trackcorr_loop_fast`, `trackback_loop_fast`),
  `track_kernels_search.py` (search box, 2D search, frequency sort),
  `track_kernels_position.py` (`assess_new_position`, angle/acceleration).
- Frame buffer: `src/openptv2/algorithms/tracking_frame_buf.py` (SoA arrays per frame,
  writes frame `t` one step after its links are decided).
- Post-passes: `src/openptv2/tracking_postprocess.py` (`confirm_links`,
  `seed_cold_start`).
- two_phase: `src/openptv2/plugins/two_phase_tracking.py`
  (`_match_two_phase_frame`, `_ghost_probabilities`).
- Guard: `tests/unit/test_track3d.py::test_trackcorr_burgers_parity_with_cython`
  (link-for-link equality with liboptv). It runs with the openptv2 defaults
  (`loser_retry = 1`, `cold_start_neighbour = 1`) and still matches, so the burgers
  data is too sparse to exercise those switches. **Before step 1, add a stronger
  classic-mode guard:** all switches at their 3DPTV values, on denser data (cavity or
  synthetic), compared link for link with liboptv.

## 6. The steps

Every step adds one key to the `track:` section (default = current behaviour), one
test, and one benchmark row. Do not combine steps in one change.

### Step 1. Confirmation inside trackcorr (largest expected gain, least code)

**What.** First measure `trackcorr` + `confirm_links()` post-pass against `two_phase`.
If the gap closes, move the rule into the loop: the frame buffer writes frame `t` only
after the links `t+1 → t+2` are known, so at write time a link `t → t+1` can be cut
when the next link's velocity change exceeds `confirm_tol`. Same tolerance rules as
two_phase (`confirm_auto`, `confirm_ends`).

**Also.** The fallback branch that links without a look-ahead ("try to link if kk is
not found") makes exactly the unconfirmed links. Let confirmation decide on them; do
not delete the branch.

**Key:** `track.confirm_tol` (shared with two_phase).
**Test:** a synthetic crossing where the fallback branch makes a wrong link; with
confirmation it is cut.
**Success:** velocity error and link precision within 2% of two_phase on the
real-jitter case, mean track length not lower by more than 10%.

### Step 2. Global assignment of contested links

**What.** openptv2 already lets a loser retry its next choices (`loser_retry`, "Phase
3" at the end of `trackcorr_loop_fast`). That is greedy: the first claimant keeps a
candidate even when a swap would lower the total cost. First measure `loser_retry`
0 vs 1. Then, only if contested links still cost accuracy, replace phases 2–3 with:
collect all (particle, candidate, `rr`) pairs, find connected groups (union-find), and
solve each group with the Hungarian method (`scipy.optimize.linear_sum_assignment` in
Python; a small C routine inside the kernel), greedy above 128 nodes.

**Key:** `track.assignment: greedy | global` (greedy = today, with `loser_retry`).
**Test:** three particles where the greedy order blocks a swap that links all three;
global links all three.
**Success:** more links in dense cases, no loss of precision. Expect a small gain only
(point 3 above).

### Step 3. Candidate search in 3D (speed)

**What.** Put the 3D particles of each new frame in a uniform grid with cells about
`v_max` wide (or a KD-tree), and take all particles near the prediction. This replaces
the search-box projection, the per-camera 2D search capped at 4 nearest blobs, and the
frequency sort for **linking**. (`use_grid` only speeds up the existing 2D search; it
does not remove the projections or the cap.) Keep the 2D search for `add_particle`;
that is trackcorr's own strength.

**Key:** `track.search: image | volume`.
**Test:** on the cavity and burgers data, the candidate set in volume mode contains
every candidate that image mode finds inside the 3D box.
**Success:** at least 2× faster on CompleteTest density; accuracy unchanged or better.

### Step 4. Velocity state and miss counter

**What.** Store a velocity per particle (from the last link) instead of recomputing
`2·X1 − X0`; for a new particle keep today's neighbour-velocity guess
(`cold_start_neighbour`). Keep a miss counter. Gaps stay filled by `add_particle` from the images,
because `ptv_is` files can only link neighbouring frames; do not add "coasting".

**Key:** `track.velocity_state: false | true`.
**Test:** a particle with a one-frame detection gap; with an unused blob in every
camera it is re-created and linked.
**Success:** no change in classic mode; fewer broken tracks across single-frame drop-outs.

### Step 5. Ghost rule

**What.** Same rule as two_phase: a track with fewer than `q_young` points may not
continue onto a point whose ghost probability is above `q_seed`, and such a point may
not start a track. Use the same ghost model (`rcm_blob`: ray miss distance scaled per
run, number of cameras that saw the point, blob brightness), so the two trackers agree.

**Keys:** `track.q_seed`, `track.q_young` (shared with two_phase).
**Test:** a static ghost point near a real track; the young track does not jump to it.
**Success:** link precision up, with at most 1% fewer points kept.

### Step 6 (only if the data asks for it). Image cost

Two_phase's projected pixel cost did not help on the benchmark. Try it last, and only
if steps 1–5 leave a contested-link error that the images could resolve.

## 7. Rules for this work

- One step per change. Measure before and after. Write the numbers here.
- Classic mode stays the default and stays equal to liboptv on the parity test.
- Judge on the real-jitter synthetic case and on real data, not on the gentle cases.
- Gate sizes are not the lever (point 3). Do not tune gates to win a step.
- If a step gains only by dropping true points, revert it (see the A7 notes in the
  2026-09-30 plan).

## 8. Open questions

1. Should the merged tracker be `trackcorr` with flags, or a new name (for example
   `trackcorr2`) with its own defaults? Flags first; decide after step 3.
2. Should liboptv get the same steps? The C code has the same structure; step 1 fits
   its write lag in `trackcorr_c_loop` the same way.
3. Once confirmation runs inside trackcorr, the post-pass becomes redundant for it.
   Keep the post-pass for the other trackers.

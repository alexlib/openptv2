# One clear tracker setup: three trackers, two switches, one add-on

Date: 2026-11-21. Branch `plan/what-we-missed`. Follows the comparison of `3d_hybrid` and `two_phase`
(they are near-twins under different names) and the legacy comparison in
`2026-11-21-detection-hypotheses-and-events.md` section 9.

## 0. Status (2026-11-22)

Decisions A, B, C taken as recommended. **Stage 1 done** (commit f715c77e): `tracking_setup.py`, loader, plugins, GUI, batch
(`--tracker`, `--direction`, `--holistic-rounds`), registry, warm-up, docs (`docs/trackers.md`), tests. **Stage 2 done**: see
section 8. Stages 3 to 5: see section 8 for what the measurements say. Stage 6 waits one release (decision C).

## 1. The problem today

One idea, many names, and the same setting spelled several ways.

| What you want | Names that exist today |
|---|---|
| The legacy-style tracker (2D search in the images) | `trackcorr` (but by default it runs something else), `cython_epipolar_tracking`, `cython_epipolar`, `openptv_epipolar`, `full_multipass`, `standard_forward`, `two_directional` |
| The 3D-only tracker | `priority_segment_3d`, `fast`, `fast_3d`, `cython_3d`, `cython_3d_tracking`, `openptv_fast_3d` |
| "trackcorr" that is really 3D-only plus a soft epipolar penalty | `trackcorr_engine: 3d_hybrid` (the default of `trackcorr`); the real port is `trackcorr_engine: epipolar` |
| Two-phase | `two_phase`, and `unified` / `unified_tracking` / `holistic` as aliases or near-copies |
| Merged-blob loop | `holistic` (a plugin name) and `holistic_rounds` (a setting) |
| Retired but still loadable | `4be`, `four_be`, `hybrid_deltat_3d`, `multi_deltat_3d`, `splitter_tracking` |
| Forward or both directions | `run_backward`, `backward`, `direction`, `bidirectional` (two_phase), `parallel_directions`, and the preset name itself |

Consequences we hit: the `trackcorr` we benchmarked was not the legacy algorithm; `dacc` had no effect because the
hybrid ignores it unless an internal flag is set; the picker says `two_phase` cannot run backward while it can.

## 2. The target setup

Three trackers, two switches, one add-on. Nothing else is visible to the user.

```yaml
track:
  tracker: two_phase        # trackcorr | fast_3d | two_phase
  direction: both           # forward | both
  postprocess: true         # reciprocity + gap bridging
  holistic_rounds: 0        # 0 = off; 2 = merged-blob inference (add-on)
  # search bounds, same names, units and meaning for every tracker:
  dvxmin: -2.0  dvxmax: 2.0  dvymin: -2.0  dvymax: 2.0  dvzmin: -2.0  dvzmax: 2.0   # mm/frame
  dacc: 0.4                 # mm/frame^2
  angle: 120                # gon
```

### The three trackers

| Name | What it is | Looks at images? | Role |
|---|---|---|---|
| `trackcorr` | The faithful port of legacy `trackcorr`: predict the next position, project it into each camera, search the 2D targets there, confirm by epipolar match. Nothing else. | Yes | Reference and compatibility. Must reproduce the legacy C results at the same settings. |
| `fast_3d` | Links the already-triangulated 3D points only. Compiled cascade, cheapest links first. One optional setting, `epi_weight` (0 = pure 3D; above 0 adds the soft epipolar penalty). The old "3d_hybrid" is this tracker with `epi_weight` 1. | No | Speed. |
| `two_phase` | Candidates from 3D, cost from the 2D leaves, optimal assignment per group, two-hop confirmation, gap carrying, ghost rules. | Yes (blob leaves) | Accuracy. |

### The two switches (for every tracker)

- `direction: forward | both`. Backward is the same tracker run on reversed frame numbers; `both` then merges the two
  link sets. `bidirectional`, `run_backward`, `backward`, `parallel_directions` (stays as a speed option under
  `direction: both`) are no longer user settings.
- `postprocess: true | false`. Reciprocity check and gap bridging after the link step. For `two_phase`, which already
  carries gaps and confirms links inside the loop, this switch controls only the reciprocity merge.

### The add-on

`holistic_rounds: N`. Re-estimates the centres of merged blobs from the tracks, re-triangulates, and relinks, N times.
Works with `two_phase` now. Whether it also works with `fast_3d` and `trackcorr` is stage 5 (see below).

## 3. Old names keep working, through one translation

One function, `resolve_tracking_setup(track_cfg, plugins_cfg)`, replaces `infer_tracker`, `infer_direction`,
`infer_preset`, `apply_preset`, `apply_tracker`, and the plugin alias tables. It returns the three choices and logs one
line when it translated something ("`full_multipass` -> tracker `trackcorr`, direction both, postprocess true").

| Old | New |
|---|---|
| `priority_segment_3d`, `fast`, `fast_3d`, `cython_3d*`, `openptv_fast_3d` | `fast_3d` |
| `trackcorr` with engine `3d_hybrid` (the old default) | `fast_3d`, `epi_weight` 1, direction both |
| `trackcorr` with engine `epipolar`, `cython_epipolar*`, `openptv_epipolar` | `trackcorr` |
| `full_multipass`, `two_directional` | `trackcorr`, direction both |
| `standard_forward` | `trackcorr`, direction forward |
| `holistic` | `two_phase`, `holistic_rounds` 2 |
| `unified`, `unified_tracking` | `two_phase` (retired code; warning as today) |
| `4be`, `four_be`, `hybrid_deltat_3d`, `multi_deltat_3d`, `splitter_tracking` | retired; warning as today; nearest kept tracker |
| `run_backward`, `backward`, `bidirectional`, `direction` | `direction` |

The old default of `trackcorr` changes meaning (hybrid to the real port). The translation above keeps saved YAMLs that
named `3d_hybrid` explicitly running the same algorithm; a YAML that said only `trackcorr` and relied on the default
gets a one-time warning that says which of the two it now runs. Which one a bare `trackcorr` means is decision A below.

## 4. Which setting does which tracker read

| Setting | trackcorr | fast_3d | two_phase |
|---|---|---|---|
| `dv*` search box | yes (2D window derived from it) | yes (3D box) | yes (as `v_max`, the largest of the six) |
| `dacc`, `angle` | yes | yes (today silently ignored; to be honoured) | no (uses `confirm_tol` instead; documented) |
| `epi_weight` | no | yes | no |
| `confirm_*`, `leaf_weight`, `q_*`, `blob_gate`, `max_gap` | no | no | yes |
| `direction`, `postprocess` | yes | yes | yes |
| `holistic_rounds` | stage 5 | stage 5 | yes |

Tracker-specific settings sit under their own sub-key (`track.two_phase.confirm_tol`, `track.fast_3d.epi_weight`),
so a reader can see at once which tracker a setting belongs to. The flat old keys still load.

## 5. Stages

Each stage keeps the test suite green and changes one thing.

1. **Names and config only, no algorithm change.** Add `resolve_tracking_setup` with tests for every old name above;
   registry, GUI picker, CLI `list-trackers` and docs show three trackers; direction and postprocess become the two
   switches. Check: on the benchmark, every old name produces links identical to before (compare link hashes).
2. **`fast_3d` absorbs the hybrid.** `epi_weight` replaces `trackcorr_engine`; honour `dacc`/`dangle`
   (or document why not). Benchmark `epi_weight` 0 against 1 and pick the default from the numbers.
3. **`trackcorr` is only the legacy port.** Re-run the legacy comparison (C `trackcorr` against ours at the same
   `dv`, `dacc`); the port matches legacy accuracy but is about 2.5 times slower, so profile and speed it up only if
   it is the tracker we keep for compatibility.
4. **One direction mechanism.** Backward = same linker on reversed frame order, one merge rule for all three. Keep each
   tracker's current mechanism if a common merge is measurably worse.
5. **Holistic for any linker.** The loop needs only "points in, links out". Adapter for `fast_3d` first (its relink is
   the cheap one, so `fast_3d` + holistic may be the fast holistic we wanted); `trackcorr` after, since it works on the
   2D targets and the loop would have to feed it corrected blob centres. Each is shipped only if it beats the same
   tracker without the loop on the benchmark.
6. **Delete the retired code** (`unified_tracking.py` 1,114 lines, `four_be_tracking.py`, `hybrid_deltat_3d.py`,
   splitter alias, the `holistic` alias plugin) once stage 1 has shipped one release with warnings. Needs approval.

## 6. Decisions for the user

- **A.** A bare `trackcorr` in an old YAML: run the real port (matches the name, slower) or the old hybrid default
  (matches saved results)? Recommended: real port, with the warning.
- **B.** Which tracker is the default for new projects: `two_phase` (most accurate, 7.8 s on the benchmark) or
  `fast_3d` (fastest)? Recommended: `two_phase`, `direction: both`, `holistic_rounds: 0` until stage 5 shows the loop
  is cheap enough.
- **C.** Stage 6 deletion: now, or after one release of warnings? Recommended: after.

## 7. Known issue found while planning

The last full test run on this branch had three failures:
`test_cython_epipolar_direction.py` (two tests: forward-only must not run the backward pass; the epipolar plugin runs
`trackcorr`, not `track3d`) and `test_sequence_triangulation_method.py::test_lsq_triangulation_reprojects_better_than_pairs`.
The first two test exactly the direction logic this plan rewrites, so stage 1 must start by finding out whether they
fail on `main` too or only on this branch.

## 8. Results of stages 1 and 2 (2026-11-22)

Case `L1_d1_c0_k1_n200_s0.08_a0.18_f0.05_m0.75` (`scripts/synth_bench.py track` / `eval`), same bounds, run time with nothing
else running. Table in `docs/trackers.md`.

| Tracker, direction | velocity error | link precision | time |
|---|---|---|---|
| two_phase, both (new default) | 0.105 | 0.971 | 4.0 s |
| two_phase, forward | 0.116 | 0.973 | 4.2 s |
| fast_3d, both | 0.167 | 0.940 | 22.6 s |
| fast_3d, forward | 0.157 | 0.941 | 11.3 s |
| trackcorr (legacy port), both | 0.220 | 0.940 | 34.6 s |
| trackcorr, forward | not scored | | 23.1 s |

Findings that change the plan:

1. **The old "3d_hybrid" never applied its epipolar penalty.** `full_forward_3d()` builds a new run object, and the flags
   were set on the old one. Its "gain over fast 3D" was zero by construction; the earlier 0.124 against 0.155 comparison
   was bounds and direction, not the penalty. Fixed (flags live on the `Tracker`; test added). With the penalty really on,
   `epi_weight` 1.0 changes velocity error 0.1665 to 0.1661: negligible, default 0.
2. **`accel_gate`** (honouring `dacc`/`dangle` in `fast_3d`) is now a setting. Loose `dacc` 6: velocity error 0.1595 (4 %
   better), link recall 0.999. Tight `dacc` 0.4: link recall 0.887. Default off; needs `dacc` from a probe run.
3. **`fast_3d` is not fast.** Kernel about 2 s of 15 s; the rest is zarr output (per frame and camera: `write_targets` 800
   calls, 4.9 s; `create_array` 7.3 s; `write_linkage` 200 calls, 3.9 s). `two_phase` writes only links and takes 4 s.
   So the registry no longer calls `fast_3d` the fastest.
4. **Backward does not help `fast_3d` here** (0.157 forward against 0.167 both). It does help `two_phase` (0.116 to 0.105).
5. **Pre-existing, found on the way:** `two_phase` crashed on a frame with zero particles (`reshape(0, -1)`); fixed with a test.
   `tests/unit/test_sequence_triangulation_method.py::test_lsq_triangulation_reprojects_better_than_pairs` fails before and
   after (run store missing), unrelated.

## 9. Results of stages 3 to 5 (2026-11-23)

**Stage 3, output speed (done).** The cost of `fast_3d` and `trackcorr` was zarr output, not the kernel: every frame wrote
four target arrays, the points and the linkage, each write several array operations. Changes, all in `storage/run_store.py`
and `tracker.py` (pure Python, no Cython rebuild): the linkage of a pass is held in memory and written once as blocks
(`RunStore.deferred_linkage`); a write over an array of the same shape is done in place; an identical re-write is skipped.
`fast_3d` forward 11.3 s to 4.9 s, `fast_3d` both 22.6 s to about 10 s, `trackcorr` both 34.6 s to 24.2 s; link quality
identical (tests in `tests/unit/test_run_store_rewrites.py`).

**Stage 4, one direction mechanism (done, with an exception).** `reversed_pass.py`: the same tracker on a mirrored view of
the store (`ReversedStore`, writes in memory), links mapped back and merged Forward-First, as `two_phase` does. Used by
`two_phase` and `fast_3d`. For `fast_3d` it beat the C-style backward on all three cases (0.243/0.145/0.155 against
0.265/0.155/0.167). For `trackcorr` it was worse (0.327 against 0.297), so `trackcorr` keeps the legacy backward, which
fits its role. Table in `docs/trackers.md`.

**Stage 5, holistic for `fast_3d` (tried, rejected).** Velocity error 0.1986 against 0.1665 without the loop (and 0.0843
for `two_phase` with it), at 109 s against 11 s. The loop adds true particles (yield 0.760 to 0.802, long-track
completeness 0.67 to 0.78) but `fast_3d` has no link confirmation and links the extra points wrongly. Not shipped. The loop
itself costs about 100 s of the 109, whatever the linker; that is the cost to cut (profile in section 8 of
`2026-11-21-detection-hypotheses-and-events.md`: scalar Python per observation, vectorisable).

**Found on the way.** On a clean dense case without ghost points, `two_phase` keeps 94 % of the true points against 99.6 %
for the others; the ghost rules `q_seed`/`q_young` cause it (off: 99.6 %). With ghosts they win (link precision 0.971
against 0.944), so they stay on and the guide says when to switch them off.

## 10. Where the holistic loop spends its time (2026-11-23)

The loop now logs seconds per phase. Benchmark case m0.75, `two_phase` + 2 rounds (the run took 110 to 159 s depending on
machine load; `two_phase` alone 4.9 s), per round 1 / round 2:

| Phase | seconds |
|---|---|
| membership + signatures | 18.3 / 3.4 |
| **refine (per-blob Python loop around `split_blob_effective`, 3 inner rounds)** | **37.0 / 32.5** |
| births (`birth_targets`, `birth_points`) | 7.1 / 5.7 |
| write back to the store | 6.3 / 5.5 |
| relink (`two_phase`) | 4.7 / 4.2 |

Refine is over half of the loop. A cProfile run shows why: 384,000 calls of `split_blob_effective` (each a few array
operations on 2 to 3 members) plus 17,600 calls of `project_pixels` on tiny arrays. A grouped rewrite of
`birth_targets` was tried first and proved equivalent to the old loop, but changed the run time by about 3 %, so it was
not kept: the per-call overhead is spread over many functions and needs a batched design, not one function at a time.

## 11. Batched refine (2026-11-23)

Done, each step checked to give the same benchmark result (vErr 0.0932 with one round, 0.0843 with two):

| Step | one round (s) |
|---|---|
| before | 99 |
| all frames in one `triangulate_weighted` call (was 800 per-frame calls) | 77 |
| blobs split in batches by member count (`split_blobs_effective`, equal to the one-blob function in a test) | 64 |
| forward-difference Jacobian (4 projections per point instead of 7) | 60 |
| `stamp_tnr` vectorised, rewrites one column and only if it changed (was 4.7 ms per array) | 49 |
| measured slots of one camera share one projection (blob and prediction slots) | 45 |

Then `birth_targets` grouped per camera over all frames (equal to the old loop in a test; 6.7 s to 2.9 s per round) and
vectorised signatures and young-partner splits (about 1 s). Measured which of births and relink matters more: births
was 7 s of Python loops, now 3 s; relink is a full `two_phase` pass (4.5 s) plus 1.7 s of zarr reads, so it has no
cheap cut that keeps the result.

Two rounds: 108 s, now 76 s (round 2: membership 2.9, refine 10.9, births 2.8, write 5.1, relink 5.5). `two_phase`
alone is 4.9 s, so the loop is still about 15 times it, not the 2 to 3 times aimed at. What remains is spread out:
`project_pixels` for the Jacobian (about 6 s), births (now 3 s), relink (a full `two_phase` pass per round),
the per-camera write of observations. Each is 5 to 15 % of the loop; no single step is left that halves it.

## 12. What is left

- **Holistic loop speed (partly done, section 11).** Remaining: births, relink and the Jacobian cost. Original text: batch the refine step: group the blobs of a frame and camera by member count (2, 3, 4), run
  `split_blob_effective`, the gate test and the projections on whole groups at once (young members and hidden partners as
  masks), and project all points of a frame and camera in one call. Prove each step equal to the current loop on the
  benchmark before replacing it. Target: the loop at most 2 to 3 times the `two_phase` time (about 15 s), from about 110 s.
  This decides whether holistic can ever be a default.
- **Stage 6, deletion** of `unified`, `4be`, `hybrid_deltat_3d`, the splitter alias and `tracking_presets.py` after one
  release of warnings (decision C). Not before.

## 13. What `two_phase` is missing, and what holistic gives (2026-11-23)

Method: truth per particle-frame (`truth.npz` has `ndet`, the blob id per camera), match each tracker's output to it,
classify every true point by its blobs. Main case (m0.75), then the 4x density case as a check.

**Where the gap is.** Share of true points / found by `two_phase` / found by holistic (2 rounds) / position error (mm):

| True points | share | `two_phase` | holistic | error |
|---|---|---|---|---|
| all detected blobs clean | 47 % | 96.5 % | 96.5 % | 0.039 -> 0.032 |
| some blob shared with another particle | 38 % | 81.4 % | 90.7 % | 0.083 -> 0.055 |
| fewer than 2 cameras see it | 15 % | 0 % | 0 % | none can be found |

All 10,849 recovered points are in the shared-blob group (8,225 of them are particles with 2 clean and 2 shared
cameras). In clean situations holistic adds nothing. At 4x density 82 % of true points have a shared blob.

**Where the loss happens.** Not in the linker. The input points already lack them (input recall 0.762, final yield
0.754, so linking keeps about 99 % of what it gets). The matcher gives each blob to one particle; the mechanism is traced in section 14
(it is not the size and grey-sum criteria, as first assumed). `two_phase` has no way to find it: it links points, it does not create them.

**What each part of the loop does** (main case, 2 rounds / 4x density, 1 round; velocity error, long-track
completeness, fragments per long track, yield):

| Variant | m0.75 vErr | longC | frag | yield | 4x vErr | 4x yield |
|---|---|---|---|---|---|---|
| `two_phase` | 0.1046 | 0.623 | 1.68 | 0.754 | 0.433 | 0.227 |
| positions only (births off) | 0.0945 | 0.630 | 1.66 | 0.753 | 0.360 | 0.231 |
| births only (positions off) | 0.1019 | 0.691 | 1.38 | 0.785 | 0.457 | 0.240 |
| both | 0.0843 | 0.738 | 1.16 | 0.790 | 0.390 | 0.246 |

- **Positions** reduce noise on shared blobs, and also raise link precision at density (0.929 -> 0.944). They matter
  more the denser the data.
- **Births** (hidden partners from residual blob brightness; every birth needs at least one residual target) give the
  continuity and about 3 points of yield on sparse data. At density, births without corrected positions make things
  worse (vErr 0.457), so births depend on positions.
- Births do not need refined positions to be found: with one inner refine pass the births and the fragmentation are
  the same (yield 0.783, frag 1.38). Positions need the full three passes to reach their gain on sparse data (one pass:
  0.1054, no gain; at 4x one pass gives 0.376).

**Tried and rejected.**
- *Weight points in shared blobs less in the smoothing*, even with oracle weights: vErr 0.1046 -> 0.1024 (2 %). The
  errors are correlated shifts while two particles overlap, not noise that smoothing averages out. Only a corrected
  centre fixes them.
- *Drop unlinked births in place* (renumber the neighbouring links) instead of linking again: saves 4 s but vErr
  0.1010 against 0.0982 and aErr 6.10 against 5.61 (plain `two_phase`: 5.80). The second link does real work. Kept the
  relink.

**What this means for a smarter `two_phase`.** The shared-blob group is the whole difference. A tracker that does the
correction while it links would need, per point: its blob id and share count per camera (known), a brightness signature
per track and camera (running median of the clean observations), and the track's predicted pixel position (the linker
already has the predicted 3D position). With those, a linked point in a shared blob gets its effective observation
`(centre - sum_others w p) / w` instead of the raw centre, and the hidden partner of a blob whose grey sum exceeds its
members' typical sum is proposed as a candidate for the next frame. That is the loop's logic running inside the forward
and backward passes, with no store round trip and no second full pass. Not built yet.

## 14. Matcher trace: why the partner in a merged blob is missing (2026-11-23)

Scripts: `bench/matcher_trace/`. Main case (m0.75), every 7th frame (about 37,000 true particles), truth blob ids per
camera from `truth.npz`.

**Not the size and grey-sum criteria.** For every camera pair of every true particle the matcher's tests were applied:
`cn` 0.02, `csumg` 0.02, `cnx` 0.3, `cny` 0.3 are ratios (min/max), so a blob up to 50 times brighter passes. Pass rates
99.5 to 100 % for pairs of missing and of matched particles alike. A merged blob is not rejected for its size or grey sum.

**The epipolar band matters a little.** `eps0` 0.02 mm is 1 pixel here (`pix_x` 0.02). A clean-clean pair passes it
100 %, a clean-shared pair 87 % (matched particles) / 78 % (missing ones): the shifted centre of a merged blob
sometimes falls outside the band. This explains a minority.

**The mechanism is blob exclusivity.** Share found by the matcher, 4-camera particles by the number of cameras where
their blob is shared with another particle: 0 shared 100 %, 1 shared 99.7 %, **2 shared 64 %**, 3 shared 63 %, 4 shared
91 %. Of the missing particles with 2 shared cameras, 1,148 of 1,297 have exactly their 2 shared blobs used by another
matcher row (the partner took them) and their 2 clean blobs free. 401 of them have a full 4-camera clique of passing
pair edges and 763 a 3-camera one, so the edges exist; the targets are gone. The matcher takes cliques by descending
correlation and uses every target once (`take_best_candidates`, `tusage`). What is left to the partner are its 2 clean
cameras, and 2-camera pairs are off (`pair_flag` false) and, when on, accepted only if unambiguous (exactly one
candidate in the band: `consistent_pair_matching`, `n_arr == 1`).

**Pairs on is not the fix.** `pair_flag` 1: shared-in-2-cameras particles found 54.7 % -> 57.4 %, ghost rows 6.4 % -> 11.5 %.

**Letting a shared blob serve a second particle is.** A second matcher pass on the leftover targets plus the blobs
considered shared, keeping only new matches with at least 3 cameras and at least one leftover target:

| Which blobs count as shared | missing shared-in-2-cameras particles recovered | new rows correct |
|---|---|---|
| truth (oracle) | 62 % | 96.4 % |
| brighter than the same matcher row's other cameras by 1.4x (gain-normalised, vs its second-dimmest camera) | 48 % | 85 % |
| same, 1.7x | 37 % | 88 % |
| brighter than 1.4x the camera median | 30 % | 80 % |

The cross-camera test needs no tracks: a matched row sees one particle in several cameras, so a blob much brighter than
its row's other cameras is likely merged. The second pass costs one more matcher call on a reduced target set.

**At 4x density it is much weaker.** Oracle second pass: 9 % of the missing particles recovered at 61 % precision
(1.4x cross-camera: 6 % at 57 %). There the epipolar candidate lists are long and the cliques ambiguous, so the loss is
ambiguity as much as exclusivity.

**What follows.** A track-free "shared blob" second pass in the correspondence stage is a real, cheap lever at
sparse and moderate density (recovering about half of the missing partners at about 85 % precision before any tracker
ghost rule), and it would feed `two_phase` points it currently never sees. At high density the matcher itself is the
limit. Not built yet; the next step is to implement it in `correspondences.py` (wrapper level, not the compiled
`algorithms/`), switch it by a `shared_pass` key, and measure the tracker output end to end.

## 15. The shared-blob second pass, built and measured (2026-11-23)

`correspondences.shared_blob_pass` (called by `match_frame_correspondences(shared_pass=...)`; YAML `ptv.shared_pass`;
unit tests `tests/unit/test_shared_blob_pass.py`). Offers the blobs that look shared (brighter than `ratio` times the
second-dimmest view of the same matched point, per-camera gain) again, with the unused targets, to a second matcher
run; keeps new points with at least 3 cameras and at least 2 own (unused-blob) views; triangulates them with the
shared views at variance 1 px^2 against 0.01 px^2. Default off; ratio 1.7 when switched on.

End to end through `two_phase` (`scripts/synth_bench.py track ... --tracker two_phase+shared_pass=R`; the key re-matches
the stored targets before tracking). Main case m0.75:

| | vErr | vErr, points `two_phase` has | longC | frag | yield | ghost | extra pts |
|---|---|---|---|---|---|---|---|
| `two_phase` | 0.1046 | 0.0905 | 0.623 | 1.68 | 0.754 | 0.0286 | 0 |
| R 1.2 | 0.1037 | 0.0784 | 0.671 | 1.48 | 0.779 | 0.0287 | 3.3 % |
| R 1.4 | 0.1024 | 0.0787 | 0.669 | 1.48 | 0.776 | 0.0284 | 3.0 % |
| **R 1.7** | **0.1016** | 0.0809 | 0.658 | 1.53 | 0.772 | 0.0282 | 2.4 % |
| R 2.0 | 0.1018 | 0.0827 | 0.648 | 1.57 | 0.767 | 0.0283 | 1.8 % |

Default merge rate: R 1.7 gives vErr 0.1825 -> 0.1832, longC 0.503 -> 0.526, yield 0.680 -> 0.700. 4x density: yield
0.227 -> 0.236, vErr 0.433 -> 0.445, ghost 0.068 -> 0.071. Clean dense case: identical. Cost per matcher frame: +13 ms
on 110 ms (+12 %) on the main case, +1 % on the dense clean case.

Two things were learnt on the way:
- The first version (1 own view allowed) made velocity error worse (0.1111): rows with a single own view are poorly
  constrained in depth (position error 0.13 mm median against 0.039). Requiring 2 own views, plus down-weighting the
  shared views in the triangulation, fixed it. The points `two_phase` already had got better by 14 % on their own
  (0.0905 -> 0.0787 at R 1.4): the partner is no longer swallowed into the wrong point.
- With the holistic loop on top (m0.75, one round): second pass + positions only (no births) 0.0895 / long 0.691 /
  frag 1.37 / yield 0.777 at 38 s; second pass + full loop 0.0929 / 0.724 / 1.23 / 0.789 at 49 s; the loop alone
  0.0932 / 0.712 / 1.26 / 0.785 at 46 s. The second pass makes half of the births unnecessary (6.9k against 13.8k).

What is left: the new points are noisy (vErr 0.22 to 0.69 on the extra points). A position refinement for shared
blobs inside the linker (section 13, last paragraph) is what would bring them in line, and with it the second pass
could become a default. Not tried on real recordings yet.

## 16. Real recordings, and the position correction (2026-11-23)

**Real data, no truth.** `bench/real_check/shared_pass_real.py` copies `res/run.zarr` of a run folder to a scratch
directory (the source is only read), matches the first 300 frames again with and without the pass, tracks with the
run's own settings, and prints truth-free statistics. CompleteTest wp2 and wp1, 300 frames each (about 1,150 points per
frame, 40 % of them 3-camera points):

| | base | `shared_pass` | `shared_correction` | both |
|---|---|---|---|---|
| wp2: points in tracks >= 50 frames | 42.2 % | 44.2 % | 51.1 % | 55.3 % |
| wp2: points in tracks >= 200 frames | 7.1 % | 8.2 % | 10.8 % | 15.6 % |
| wp2: tracks >= 50 frames | 1540 | 1609 | 1695 | 1720 |
| wp2: roughness, median (mm) | 0.0548 | 0.0557 | 0.0464 | 0.0465 |
| wp2: roughness of the new points | | 0.105 | | 0.054 |
| wp2: short tracks (< 5 frames) | 15.4 % | 15.5 % | 14.1 % | 14.1 % |
| wp1: points in tracks >= 50 / >= 200 | 40.0 / 6.0 % | | 49.6 / 10.2 % | 54.4 / 14.1 % |
| wp1: roughness, median (mm) | 0.0557 | | 0.0474 | 0.0474 |
| tracking time, 300 frames (s) | 9 | 9 | 29 | 29 |

The pass alone adds 2.1 % points (7.2k and 7.4k in 300 frames), 84 to 88 % of them in tracks of 5 frames or more, and
changes nothing about the ordinary points (roughness 0.0548 -> 0.0554, singletons 11.7 % -> 11.9 %); the new points
are rough (0.105, about twice). The correction removes that: new points 0.054 against 0.0465 for the others, and it
makes the whole track set smoother (-15 %) and longer. The effect on the real recordings is larger than on the
benchmark.

**The position correction** is `track.shared_correction` (`holistic.run_holistic_experiment(..., correction_only=True)`):
link, re-estimate only the points that touch a shared blob (or whose blob is too bright for its own particle:
`refine_merged(shared_only=True, x_init=stored positions)`), link again. What was learnt building it:
- Re-estimating *only* the shared-blob points is better and faster than re-estimating all points (velocity error
  0.0876 against 0.0903, refine 4.6 s against 10.1 s): the temporal prior on the clean points was not helping.
- The second link is necessary. Writing the corrected positions into the existing links (no relink) gave velocity error
  0.163 (worse than no correction, 0.1046), and re-confirming the old links with the two-hop rule only got to 0.106;
  a real relink gives 0.0903. The corrected positions let the linker connect more links (link recall 0.9737 -> 0.9860).
- A forward-only first link saves 2 s and costs accuracy (0.0904 against 0.0876): not used.
- The observation table is written by the loop and read by nothing; the correction does not write it.

Benchmark, velocity error / long completeness / fragments / yield (all with truth):

| | `two_phase` | correction | pass + correction |
|---|---|---|---|
| m0.75 | 0.1046 / 0.623 / 1.68 / 0.754 | 0.0921 / 0.639 / 1.57 / 0.755 | 0.0876 / 0.685 / 1.38 / 0.773 |
| default merge rate | 0.1825 / 0.503 / 2.29 / 0.680 | 0.1283 / 0.528 / 2.07 / 0.686 | 0.1281 / 0.563 / 1.91 / 0.708 |
| 4x density | 0.4326 / 0.182 / 0.77 / 0.227 | 0.3605 / 0.189 / 0.77 / 0.232 | 0.3649 / 0.196 / 0.78 / 0.240 |
| no merged blobs | | identical (round skipped) | identical |

At the default merge rate acceleration error falls 11.5 -> 8.3 and ghost points 3.8 % -> 3.3 %; at 4x density ghost
points fall 6.8 % -> 5.0 % and link precision rises 0.929 -> 0.944. Cost: about 4 times the `two_phase` run time (two
link passes, membership and signatures 2.4 s, refine 4.6 s, write 2.7 s per 200 frames).

**Not done:** both options are off by default. The real-data evidence is truth-free (track length and roughness), and
the cost is 3 to 4 times the run time; a default would be a decision for the user. A single-pass version (correction
inside the forward and backward passes, no second link) was not built: the measurements say the second link is where
part of the gain comes from.

## 17. Choosing, and where the next step is (2026-11-24)

The user's brief: the extras (`shared_pass`, `shared_correction`) stay off by default and are for difficult data;
learn from the data which setup to recommend; find where the next significant quality gain is.

### 17.1 The sweep

10 benchmark cases (exact dense, noise only, three merge rates, density x2 and x4, frame skip 4 and 8, clustered with
skip 4) x `trackcorr`, `fast_3d`, `two_phase`, +pass, +correction, +both, one job at a time (clean timings), plus three
more cases with the three trackers only (exact sparse, two ghost-free); scored with the postptv repair and Savitzky-Golay at the best window. Raw numbers of the 10 cases: `bench/limits/sweep_results.json`.
The table is in `docs/trackers.md` ("Choosing"). Findings:

1. `two_phase` is the fastest *and* the most accurate on noisy data: 4 to 5 s per 200 frames against 25 s (`trackcorr`) and
   10 s (`fast_3d`); velocity error 1.3x to 7x lower. `trackcorr` is never the fastest in this implementation.
2. Large steps break the others: frame skip 4 / 8: `trackcorr` 0.39 / 0.82 (176 s), `fast_3d` 0.43 / 1.16, `two_phase`
   0.11 / 0.13. Same at density x4 (1.05 / 1.69 against 0.41).
3. On exact data the order flips: `fast_3d` and `trackcorr` yield 0.995 and long completeness 0.97 to 0.98,
   `two_phase` 0.94 to 0.96 / 0.92 to 0.93 (its ghost rules drop 3 to 6 % of true points). With noise but no ghosts
   `fast_3d` and `two_phase` tie in accuracy (0.139 against 0.141; 0.163 against 0.155) and `fast_3d` keeps 1 to 2 points more.
   (`trackcorr` found almost nothing on the "noghost" variants: harness artefact, those rows are not used.)
4. The correction pays in proportion to how many blobs are shared. The truth-free measure is the **shared-blob flag**
   (share of 3 to 4 camera points with a view brighter than 1.7x their second-dimmest view): flag 0.11 -4 % (-13 % with the
   pass), 0.14 -12 %, 0.17 -30 %, 0.13 on low-noise data -40 %, 0.26 -28 %, 0.40 -17 %, 0 none. The pass alone: +2 points
   of yield, +3 to 4 of long completeness, no velocity gain, +3 % velocity error when the data is dense (mutual-link share
   below 0.9) unless the correction is on too.
5. Real wp1/wp2 measure like the benchmark's density x1, noise 0.08 px cases (1,150 points/frame, spacing 29 steps,
   noise/step 0.67, partial views 0.38, flag 0.15).

`tracking_advice.py` (`openptv recommend`) turns six numbers into this advice; checked on all 13 cases and both real
recordings, it agrees with the best setup of the sweep in every case. (Its thresholds are those numbers: exact below
noise/step 0.1; correction at flag >= 0.10; pass at flag >= 0.10 and, when dense, only with the correction.)

### 17.2 Where is the headroom? (m0.75, the real-like case)

| | velocity error |
|---|---|
| `two_phase` | 0.1046 |
| + pass + correction (best now) | 0.0876 |
| perfect association, each merged blob centre to 0.3 px (ceiling E1c, section 13 of the joint-tracker plan) | about 0.060 |
| the same to 0.1 px | about 0.044 (sparse default case: 0.0436) |
| exact data | 0.0009 |

Three measurements decide where the gap is (scripts in `bench/limits/`):

- **De-merge accuracy** (`demerge_error.py`): views on a blob shared by two particles, error of the pixel position used
  against the truth. Clean views 0.082 px (median; 0.16 rms). Shared views, raw centre: median 0.60, 24 % above 1 px.
  After the correction: **median 0.30 px** (0.28 when the point has a trusted prediction, 0.53 without), but **12.5 % above
  1 px, rms 0.82 px** (p90 1.14, p99 3.14). The median is at the ceiling's 0.3 px; the tail is not. The fit knows: the
  views that end up more than 1 px off report a standard deviation of 1.15 px against 0.28 px for the rest (error/sd:
  median 0.93, p99 4.8), so they are down-weighted; the information in them is lost, not recovered. With three or more
  particles in a blob the median is 0.68 px.
- **Ray convergence** (`loco_compare.py`): leave-one-camera-out residual of 4-camera points. Synthetic 0.31 px median
  (p90 0.51); real wp1 and wp2 0.35 px (p90 0.68). Real data is 13 % worse in the median, 35 % in p90: calibration and
  optics are not a large hidden term.
- **Information in the stored blob extents**: on real wp2, views flagged as merged have 1.62x the pixel count and
  1.33x the x and y extent of the other views of the same point. The blob's shape carries information on the split that
  the correction does not use (it uses centre and grey sum only). The synthetic blob tables have constant extents, so
  this cannot be tested on the benchmark.

### 17.3 Candidates, ranked by evidence

| # | Idea | Evidence | Expected | Cost / risk |
|---|---|---|---|---|
| 1 | **Pixel-level de-merging**: fit the pixels of a shared blob with one PSF per member, seeded by the tracks' predictions (0.28 px), instead of splitting a centre by brightness. Store a small patch per target at detection (about 80 bytes) or re-read the images for the 15 % shared blobs | the tail (12.5 % above 1 px) is where centre + grey sum run out; real merged blobs differ in shape; ceiling at 0.1 px is 0.044 | velocity error -30 to -50 % from 0.0876; the tail of ambiguous splits goes; helps 3-particle blobs | needs the PSF (`scripts/psf_estimate.py` exists), image access or patches; a rendered case with truth to prove it first |
| 2 | Use the stored extents (nx, ny, n) as a second measurement of the separation | real merged views are 1.33x wider | possibly part of #1's gain for free | untestable on the benchmark (constant extents); needs a rendered case with truth |
| 3 | Tracks confirm 2-camera matches (pairs are ambiguous alone, the prediction picks the candidate) | 2.4 points of the 5 missing points of yield are particles seen by only 2 cameras | +2 to 3 points of yield, long completeness | moderate; ghost risk; the matcher stays untouched |
| 4 | Track-guided detection of faint particles (lower local threshold along a predicted track) | 11 % of true particles have no blob at all (the yield ceiling of blob space, 0.85) | yield beyond 0.85 | needs the images; this is the second half of Shake-the-Box |
| 5 | One-pass correction (no second link) | the second link is 40 % of the correction's time | run time 4x -> 2.5x, no accuracy change | measured against: the relink supplies part of the gain |
| not worth it | weights in the smoothing (2 % even with oracle weights), ghost removal for `two_phase` (0.5 to 2 %), stitching (harmful), a better linker (99 % of reachable links found), calibration (13 %) | | | |

**Where the breakthrough is:** #1, the step from blob space to pixel space for the shared blobs only, with the tracks as the
prior: the same idea as Shake-the-Box's residual fit, but local and cheap (15 % of the blobs, no image rendering of the
whole field). Everything else on the list is a few points. A tracker or a matcher change cannot reach it: the loss is in
the measurement of the blob, and the sweep shows linking already finds 99 % of the reachable links.

**First experiment (go / no-go before building anything):** take a rendered case with truth (`Synthetic-CompleteTest`
syn1 render, or render x1 and x2 with openptv-analysis `synthetic_experiment`), take the blobs the standard detection
merges, fit one PSF per member seeded with the true position plus 0.28 px noise, and measure the de-merge error as in
`demerge_error.py`. Go when the median is at most 0.15 px and the share above 1 px at most 3 % (now 0.30 px and 12.5 %).
Also on the rendered case: does a regression of the true separation on the stored extents beat the prediction alone
(test of #2).

## 18. Pixel-level de-merging: the go/no-go experiment (2026-11-24)

Script: `bench/limits/pixel_demerge.py` (reads images only). Case: `Synthetic-CompleteTest/syn1`, rendered images with truth
(1500 particles per frame, Gaussian PSF sigma 0.72 +- 0.15 px, 4 cameras). For each detected blob (the pipeline's own
high-pass and `target_recognition`) that holds two or more true particles, one Gaussian per member is fitted to the pixels,
seeded at truth + N(0, 0.28 px), which stands for the track's prediction (the real prediction error measured earlier).
Truth is used for the association of members to blobs and for the error only. The fit knows no truth position and no
amplitude. Members: 7,262 merged members in 4 frames (33 % of all blob members here, a denser case than the benchmark).

Setup that was tuned on frame 3 and then **checked on frames 60, 120, 200 (5,498 merged members, never seen when tuning)**:
raw image with a local plane as background, one common sigma per blob (free, 0.5 to 1.1), amplitudes free, a Gaussian prior
on the positions at the seed (weight 15 to 20 relative to the pixel residuals).

| held-out frames, merged members | median error | share > 1 px | share > 0.5 px |
|---|---|---|---|
| blob centre (a merged observation today) | 0.776 px | 42.7 % | 61 % |
| seed alone (the prediction) | 0.327 | 0.2 % | 20 % |
| brightness-weighted centre split (the `shared_correction` idea, with fitted amplitudes: generous) | 0.348 | 8.8 % | 33 % |
| **pixel fit, prior weight 15** | **0.153** | **2.7 %** | 12 % |
| **pixel fit, prior weight 20** | 0.159 | 1.6 % | 11 % |
| pixel fit, prior weight 20, seed error 0.40 px instead of 0.28 | 0.176 | 4.7 % | 18 % |
| single (unmerged) blobs, same fit | 0.05 | 0.8 % | 1.6 % |

What drove the numbers (frame 3, merged members, median / share above 1 px):
- fixed sigma per camera 0.188 / 6.6 %; **free common sigma** 0.156 / 4.5 %; raw image + plane instead of the high-passed
  image: 0.145 / 4.0 % (the high-pass has negative lobes that a Gaussian cannot follow);
- the prior weight trades the median against the tail: weight 3: 0.189 / 8.7 %; 15: 0.143 / 2.8 %; 30: 0.166 / 0.9 %;
  100: 0.261 / 0.1 %. At very strong weights the fit just returns the seed (0.33).
- by separation of the nearest co-member: below 0.75 px the fit is no better than the split (0.32 against 0.29: these two
  particles are one blob); from 1.25 px the gain is 2x (0.16 against 0.31); above 2 px 4x (0.11 against 0.43).
- 3 or more members: median 0.22 to 0.29, share above 1 px 2.5 to 4 %.

**Verdict: go, with two cautions.** The criterion (median at most 0.15 px, share above 1 px at most 3 %) is met on the tail
and is at the line on the median (0.153 to 0.159 on the held-out frames, 0.143 on the tuning frame). Against the centre
split on the same blobs it halves the median and cuts the share above 1 px by 3 to 5 times. A rough translation to the
velocity error, from the error budget of section 13 (ceiling 0.0436 with centres at 0.1 px, 0.060 at 0.3 px): a median of 0.15
px with a 2 % tail should land between them, about 0.05 to 0.06 at m0.75, against 0.0876 now. That is an estimate, not a
measurement.

Cautions: (1) the render uses the same circular Gaussian that the fit assumes. A real PSF is not exactly that; the fit
needs to be checked on a PSF it does not know (render with another PSF shape, or the real recordings, where the leave-one-
camera-out miss is the truth-free score). (2) The cost: one least-squares per shared blob, about 2.5 s for the 5,300 blobs
of a frame in this unoptimised script (about 1,800 merged members of which); at 15 % shared blobs in a real run this is
tracker-scale cost, and it needs the images (or a stored patch per target), which `shared_correction` does not.

Not done: the stored extents as a measurement (#2) were not tested; the free sigma already captures much of the shape.

Next step if continued: put the fit inside `refine_merged` as the measurement for shared views (seed from the track, return
a position and a variance per view, fall back to the centre split where images are unavailable), run it on syn1 through the
tracker, and compare the velocity error to truth against `shared_correction`.

## 19. The pixel fit inside the refine step: built, and it does not pay on the rendered case (2026-11-24)

Built: `src/openptv2/pixel_fit.py` (`fit_window`, `PixelFitter`), the hook `refine_merged(pixel_fit=...)`, the loop argument
`run_holistic(pixel_fit=...)` and the experiment key `track.holistic_pixel_fit` (with `shared_correction`). Off, not in the
setup, the GUI, the command line or `trackers.md`: it stays an experiment. The fit takes the track prediction as the seed,
reads the pixel noise from its own residual (so the prior needs no per-camera tuning), fits other blobs of the window along,
and reports a variance (covariance x 2.5: measured 2.6x too small). Tests: `tests/unit/test_pixel_fit.py`, the hook in
`test_merge_refine.py`. Harness: `bench/limits/syn1_pixel_fit.py` (tracker on syn1, truth score, oracle variant),
`bench/limits/syn1_obs_check.py` (observation level).

Observation level (15 frames, 5,005 shared views matched to truth, the real refine step with its own predictions):

| pixel position of a shared view | median error | share > 1 px |
|---|---|---|
| blob centre | 0.72 px | 34.6 % |
| centre split (`shared_correction` today) | 0.44 px | 23.3 % |
| pixel fit | 0.22 px | 13.2 % |

The fit does what it should, but less than in the isolated experiment of section 18 (0.15 px, 2 %): here the seeds are the
tracker's own predictions (wrong for some), the other blobs are seeded at their centres, and 3 or 4 members occur.

Tracker level (syn1, first 40 frames, truth score on the common set, velocity error with the 41-frame window):

| variant | velocity error | acc error | position error (bias removed) | time |
|---|---|---|---|---|
| `two_phase` | 0.1670 | 10.63 | 0.0775 | 1.1 s |
| + `shared_correction` | 0.1179 | 7.93 | 0.0559 | 4.2 s |
| + correction with the pixel fit | 0.1188 | 8.03 | 0.0543 | 207 s |
| + correction with the **true pixel positions** (+0.05 px) | 0.1146 | 8.12 | 0.0501 | 4.7 s |

**Result: no gain, and the oracle says why.** Perfect de-merging of every shared view improves the velocity error by 3 % over
the centre-split correction, and the pixel fit by 0 %, at 50 times the time (11 ms per blob, a numeric Jacobian and two
fits per blob; it could be made 5 to 10 times faster, but that is not the problem). On this case the correction has already
taken what shared blobs had to give (0.167 to 0.118); what is left (0.115) is something else. The 3D error of the points
that touch a shared blob against truth is 0.16 to 0.17 mm in all variants (stored positions, split, fit): a systematic
offset, larger than any de-merge difference. The likely floor is the case itself: the images were rendered with the
calibration perturbed by 0.12 to 0.34 px RMS per camera, so every camera sees the particle shifted by a smooth field that no
tracker can remove. That reading is not proved here (it would need a render with the processing calibration exactly).

What this changes in section 17: the "breakthrough" claim (velocity error 0.0876 to about 0.044 by closing the de-merge
error) came from blob-table cases with exact calibration. A rendered case with a realistic calibration error shows the
ceiling for this idea at 3 % beyond `shared_correction`. Real recordings have a calibration miss of about 0.35 px
(leave-one-camera-out, section 17.2), like syn1's. So on real data the pixel fit is expected to add little too, and it needs
the images and 50 times the time. **Recommendation: do not ship it, keep the code as an experiment.** What would reopen it: a
render with an exact calibration (isolates the de-merge from the calibration floor), or a real run whose truth-free score
(leave-one-camera-out) improves with it. The calibration floor is the next thing to look at: 0.12 to 0.34 px per camera is
the same size as the whole error budget, and self-calibration on particle tracks (`scripts/selfcal_synthetic.py`) attacks it.

## 20. Two corrections to section 19, and the `disco` sweep (2026-11-24)

**The calibration-floor reading in section 19 is wrong.** `bench/limits/calib_floor.py` triangulates syn1's truth pixel
positions (rendered with the perturbed calibration) with the processing calibration, perfect identities, and scores them with
the benchmark evaluator:

| input | velocity error (41) | acceleration error | 3D error (median) |
|---|---|---|---|
| truth pixels, exact | 0.0066 | 0.27 | 0.167 mm |
| truth pixels + 0.05 px noise | 0.0207 | 1.34 | 0.167 mm |
| control: truth 3D through the processing calibration | 0.0066 | 0.27 | 0.0001 mm |

The calibration mismatch is real (0.167 mm of position offset: the floor seen in section 19) but a smooth offset cancels along a
track: it costs 0.0066 of velocity error, not the 0.115 left after `shared_correction`. So the remaining error is elsewhere, and
"perfect de-merging adds 3 %" still stands (the oracle of section 19 is unaffected) but its explanation does not.

Where the error is (`bench/limits/error_sources.py`, tracked points against the truth triangulated with the processing
calibration, so the offset is gone; 40 frames after `shared_correction`): median 0.027 mm, rms 0.068 mm, p90 0.105 mm.
Points from 3 cameras (29 % of points) have median 0.054 mm and carry 53 % of the squared error; 4-camera points have median
0.021 mm and carry 47 % (a heavy tail). A 3-camera point is a point that lost a view, most often to blob exclusivity in the
matcher or to a missing detection. The stored blob ids hold one point per blob, so "touches a shared blob" cannot be read
from the store without `extend_membership`; that split is not in this table.

Not explained yet: why 0.115 of velocity error remains when single-blob centroids are accurate to 0.03 px (median) and exact
pixels give 0.007. The candidates are the 3-camera points, the 3 % wrong links (raw link precision 0.969), and the 3 % ghost
points. The next measurement is a velocity error split by those groups (which points enter the error), not another
measurement-model change.

**`targ_rec.disco`** (the discontinuity value; `bench/limits/disco_sweep.py`, syn1 frames 3 and 60, per true particle in a
camera view, all values with the same other thresholds):

| disco | blobs | own blob | merged | over-split | missed | fragments (blobs without a true particle) |
|---|---|---|---|---|---|---|
| 0 | 11,407 | 61.6 % | 26.1 % | 1.9 % | 10.9 % | 18.0 % |
| 5 | 10,901 | 62.0 % | 26.3 % | 0.6 % | 11.2 % | 16.0 % |
| 20 (syn1's value) | 10,598 | 61.4 % | 26.5 % | 0.4 % | 11.5 % | 14.4 % |
| 50 | 9,988 | 59.6 % | 27.5 % | 0.2 % | 12.4 % | 11.5 % |
| 100 | 9,333 | 56.6 % | 29.0 % | 0.1 % | 14.0 % | 9.0 % |
| 255 | 8,566 | 50.3 % | 30.0 % | 0.0 % | 18.8 % | 9.1 % |

`disco` is how far a pixel may rise above the pixel it grows from; small values stop at the valley between two peaks. The
merged share barely moves (26 % to 30 %): merged particles of this size are closer than about 1.5 px and form one intensity
peak, so there is no valley to cut. Values from 0 to 20 are equivalent (own blobs 61.4 to 62.1 %); only values above 50 hurt (own
blobs fall to 50 %, particles missed rise to 19 %, because separate particles are grown into one blob and one of them is
dropped). Advice: leave it at 5 to 20; it is not a lever for merges. A splitter at detection would need a model of the blob
(this section 18's fit run on every blob with a second peak or an excess size), not a threshold; whether that pays is the same
question as section 19 and the 3-camera error above.

## 21. Streaks: blobs followed in one camera through time (2026-11-24)

Question (user): merges are episodes (approach, merge, separate) and occlusion along the viewing axis is the same thing; follow the blob in 2D over time and use its brightness. Is brightness helpful, and do streaks help? Scripts: `bench/limits/streaks_syn1.py` (mass additivity, truth), `streak_lib.py` (truth-free streaks), `streaks_ablation.py` (ablation on syn1 and real wp2, frames 100-140).

A streak is a chain of 1-1 links between consecutive frames in one camera (Hungarian, gate 2 px; particles move 0.45 px per frame, blobs are 4.7 px apart, so linking is nearly unambiguous). Cues per blob: mass ratio to the streak's own unit mass (15th percentile of its mass within +-30 frames, no identities), and topology (another chain ended or started within 3 px in the 6 frames before or after).

**Brightness is additive (syn1, truth identities).** Two parents merging into one child: log(child / sum of parents) median -0.009. A single blob from one frame to the next: robust sd 0.033, 5-95 % within +-0.10. A second particle joining: median +0.40 (x1.5), tail to +2.2, because overlap is gradual. AUC of the one-frame jump for "a second particle joined": 0.888, without identities.

**Does the streak help, merged blobs found at 5 % false alarms:**

| Cues | syn1, truth: merged | syn1: geo label | real wp2: geo label |
|---|---|---|---|
| blob alone (n, grey sum) | AUC 0.763, recall 29 % | 0.784, 24 % | 0.693, 23 % |
| + streak mass ratio | 0.844, 56 % | 0.872, 52 % | 0.713, 34 % |
| + streak topology | 0.810, 42 % | 0.836, 42 % | 0.723, 32 % |
| + both | 0.851, 56 % | 0.874, 53 % | 0.725, 37 % |

- Geo label = a 3D point seen by three cameras projects within 1.5 px onto this blob in the fourth. Truth-free. On syn1 it is right 96 % of the time but finds only 18 % of the truly merged blobs (partner must be a clean 3-camera point).
- Real mass is noisier: robust sd of the per-frame change 0.090 against 0.032; 4.5 % of links jump above x1.3 against 1.4 %. The real lift is about a third of the synthetic one (AUC +0.03 against +0.09), and topology carries more of it than mass.
- The leave-one-camera-out label is a poor merge label (on syn1 AUC 0.68 against the truth shift; on real it also holds calibration and optics error: 26 % above 0.5 px against 12 %). It shows no lift on real (0.569 to 0.574) and should not be used for this question.
- Reference age (syn1, truth-clean frames): 36 % of merged blobs have one within +-3 frames, 82 % anywhere on their streak; 18-21 % are on streaks with no clean frame. Truth-free unit mass: AUC 0.64 (+-3 frames) rising to 0.78 (whole streak).

**Reading.** Brightness helps, strongly on rendered images and about a third as much on real data. Streaks add a clear gain over the single blob. What this measures is finding merged blobs, not resolving them; the earlier oracle (perfect observation after the correction) bought only 3 % of velocity error. The value of streaks, if any, is in yield and track continuity, which is the unmeasured part: what share of breaks and missing views lies inside an episode with clean frames on both sides.

## 22. Where the tracker loses particles, and how much a streak could bridge (2026-11-24)

Scripts: `bench/limits/loss_in_episodes.py` (syn1, truth), `loss_streaks_real.py` (truth-free, syn1 and real wp2). syn1 frames 1-40, rows from frames 7-34 with at least 3 cameras in view (42000 particle-frames), raw two_phase output. Per camera a truth particle is 'clean' (alone in its blob), 'shared' (blob holds 2 or more), or 'noblob'. A shared view is **bridgeable** when the particle has a clean view in the same camera within K = 6 frames on both sides (interpolation) or on one side only.

**Loss, in points of all particle-frames (base, then shared_pass + shared_correction):**

| | base | pass + corr |
|---|---|---|
| lost | 27.2 | 24.9 |
| held with 3 cameras | 21.2 | 23.2 |
| held with 4 cameras | 51.6 | 51.9 |
| lost, clean in 3+ cameras (matcher / linker, not a merge) | 0.7 | 0.6 |
| lost, streak bridge both sides | 2.7 | 2.0 |
| lost, one-sided bridge only | 5.1 | 4.2 |
| lost, shared with no clean frame within 6 (episode too long) | 7.3 | 6.7 |
| lost, fewer than 3 blobs (faint, outside) | 11.3 | 11.3 |

- The existing second pass (`shared_pass`) already takes back 2.3 points, almost all from the bridgeable classes (a third of the two-sided, about a sixth of the one-sided) and a little from the too-long episodes; nothing from the faint ones.
- What is left for a streak: 2.0 points (two-sided) up to 6.2 (with one-sided extrapolation), at best. The faint class (11.3) and the too-long episodes (6.7) are outside it; the second needs a per-frame resolution, not a bridge.
- Dose-response of loss with shared cameras per particle-frame (0, 1, 2, 3, 4): lost 12, 21, 47, 62, 62 %; held with 4 cameras 82, 45, 29, 23, 27 %. 67 % of all particle-frames have a shared blob in some camera, so "a merge is near" is not a cause on its own.
- Missing view of a 3-camera point (8890): shared blob 88 % (bridgeable both sides 21 %, one side 35 %, no clean frame within 6 frames 32 %), no blob 10 %, clean blob not used 2 %.

**Breaks (1145 gaps, about 900 id switches, 1509 particles):**
- 78 % of the gaps are 1-3 frames; the scored pipeline repairs those in 3D (`repair_arrays`, `max_gap=3`) with no streak. The 253 long gaps (22 %): 43 % bridgeable by a streak, 25 % too-long episodes, 29 % with a frame of fewer than 3 blobs.
- Id switches with every frame found (the points exist, the tracker does not link them): 849 base, 1043 with the pass (new points), 824 with pass + correction. 87 % have a shared blob at one of the two frames, against 63 % of correctly linked pairs; only 3 % are clean 4-camera points. This is the larger break class and it is merge-related, but a link problem of shifted centres, not a missing view.

**Truth-free counterpart (same script on syn1 and real wp2):**

| | syn1 | wp2 |
|---|---|---|
| missing view of a 3-camera point lands on a blob within 1.5 px / 1.5-3 px / none | 49 / 37 / 14 % | 53 / 24 / 23 % |
| points with 4 cameras: track ends / starts / mid-track | 39 / 35 / 77 % | 40 / 44 / 68 % |
| points with a flagged blob in some camera: ends / mid-track | 57 / 37 % | 72 / 66 % |
| flagged blobs at ends and starts, clean on both sides within 6 frames / one side / none | 21 / 51 / 27 % | 19 / 49 / 32 % |

- On real data, as on syn1, tracks end where views drop out, and most missing views land on an existing blob (77 % real, 86 % syn1). Real has more missing views with no blob there (23 % against 14 %).
- The streak flags lose their power on real data: 66 % of mid-track points already have a flagged blob (mass noise 2.8 times larger, detection flicker makes topology events), so the end-of-track enrichment is 1.09 against 1.5 on syn1.

**Reading.** Merges explain most of the loss that is not faint, and the dose-response is clear, but a streak bridge would add at most 2 to 6 points of raw yield on top of the shared pass, and fewer long track gaps than the id switches. The two larger parts of the loss need other tools: faint particles (no blob at all) need a lower local threshold on the image; episodes with no clean frame and the id switches need a per-frame resolution, which is what the pixel fit did at observation level (it fixes the points that exist, not the ones never matched).

## 23. Relaxing blob exclusivity in the matcher, and a pixel check of the candidates (2026-11-24)

Question: the losses left after sections 21-22 are shared views that no streak can bridge, and id switches. Does a looser second matcher pass recover them, and can the raw image decide which of the new candidates are real? Scripts: `bench/limits/relax_exclusivity.py` (syn1, truth), `new_points_quality.py`, `candidate_pixel_check.py`, `light_check_real.py`, `relax_real.py` (real wp2, truth-free). The pass is `correspondences.shared_blob_pass`; its ratio decides which matched blobs are offered again (brighter than ratio x the second-dimmest view of their point; default 1.7) and `min_own` how many unused blobs a new point needs (default 2). Offering every blob (ratio 0) reproduces the first pass, because the second pass is exclusive too: the lever is the selection.

**syn1, 40 frames, with `shared_correction` (truth):**

| | yield | ghost points | raw link precision | velocity error | error of the extra points | id switches |
|---|---|---|---|---|---|---|
| no second pass | 0.709 | 3.19 % | 0.9685 | 0.1099 | - | 709 |
| default pass (1.7, 2) | 0.724 | 3.27 % | 0.9679 | 0.1122 | 0.202 | 832 |
| ratio 1.05, 1 own view | 0.743 | 3.29 % | 0.9675 | 0.1166 | 0.268 | 1134 |
| ratio 1.05 + pixel fit | 0.742 | 3.29 % | 0.9677 | 0.1166 | 0.280 | 1134 |
| no pass + light filter z>=10 | 0.706 | 2.40 % | 0.9754 | 0.1090 | - | 685 |
| ratio 1.05 + light filter z>=6 | 0.742 | 2.54 % | 0.9742 | 0.1159 | 0.248 | 1079 |
| ratio 1.05 + light filter z>=10 | 0.740 | 2.42 % | 0.9752 | 0.1167 | 0.268 | 1069 |

(Without correction, ratio 1.05 gives yield 0.737 and velocity error 0.182 against 0.160 for the plain run.) Yield rises monotonically as the ratio falls and flattens at 1.0.

- The points the loose pass adds are 7 % of all points, 91 % of them with 3 cameras: 30 % ghosts, position error 0.102 mm median against 0.025 for the ordinary points, 61 % singletons. The cost in velocity error is the real but poor extra points (0.27 against 0.11 overall), not the ghosts, and the id switches rise 60 %.
- **Light check:** project a 3-camera point into its missing camera and measure the raw image there (peak within 1.5 px over the local background, in units of the local noise, z). For the new points the ghosts have median z 4.7, the real ones 91; AUC for ghost against real 0.876, against 0.627 for the blob-level cues (brightness of the views). Used as a filter after matching it takes the ghost share of the output from 3.2 % to 2.4 % and the raw link precision from 0.9685 to 0.9754, at no cost in yield (base: 0.709 to 0.706). It does not cure the extra points' error or the id switches.
- The pixel fit on the observations adds nothing here either (velocity 0.1166 both, 278 s against 4 s).

**Real wp2, first 40 frames, fresh matching and tracking, scratch copy, `shared_correction` on (truth-free):**

| | points/frame | singletons | in tracks of 30+ of 40 frames | points in those tracks per frame | roughness median | LOCO median |
|---|---|---|---|---|---|---|
| no second pass | 1178 | 11.8 % | 47.5 % | 560 | 0.0490 | 0.355 |
| + light filter z>=10 | 1080 | 8.9 % | 49.8 % | 538 | 0.0487 | 0.355 |
| ratio 1.05 | 1265 | 14.0 % | 52.6 % | 665 | 0.0499 | 0.357 |
| ratio 1.05 + z>=10 | 1138 | 9.6 % | 56.0 % | 638 | 0.0491 | 0.357 |
| ratio 1.05 + z>=20 | 1094 | 9.1 % | 55.8 % | 610 | 0.0484 | 0.357 |

- Missing views of 3-camera points carry less light on real data: 24.5 % have z < 10 (syn1: 11 %); median z 46 against 100. The light filter therefore removes more on real (8 % of the points at z>=10) and, alone, 4 % of the points that end in long tracks; with the relaxed pass the long-track points rise by 14 %, singletons fall below the plain run, roughness and the LOCO residual do not change.
- Track length inside the store's own linkage could not validate the filter on real data (72 % singletons even with strong light: that linkage is not comparable), hence the fresh runs above.

**Reading.** The second pass is a real yield lever (+3 points on syn1, +14 % points in long tracks on wp2) and the light check is a cheap, effective ghost filter (AUC 0.88 against 0.63). The price is accuracy: the extra points sit on shared views with two own cameras, their error is 2.5 times the average and the id switches rise, which the observation-level fixes (correction, pixel fit) do not change; velocity error on syn1 +6 % at the loosest setting. Not shipped; open: a 300-frame real run, the cost of reading images in the matching stage, and whether weighting or dropping the extra points' shared views in linking removes the extra id switches.

## 24. Light check: implementation and validation (2026-11-24)

Implementation: `src/openptv2/light_check.py` (`light_z`, `missing_view_light`, `light_keep`, `apply_light_check`, `light_check_from_params`), called after the matching in both paths of `gui/ptv.py::py_sequence_loop` (batch and per frame) when `ptv.light_check` is true or a number. Default threshold z >= 6. The `rcm` array is filtered with the points; unreadable images skip the frame with one warning. Tests: `tests/unit/test_light_check.py` (12: vectorised against the plain-loop definition, edges, scene, store round trip, empty frame, the real `py_sequence_loop` on `test_cavity`, unreadable images). Benchmarks: `bench/limits/light_check_bench.py` (real pipeline, scratch copies), `switch_diag.py`, `track_variants.py`.

**Why the track gate is not relaxed.** Re-tracking the relaxed-pass store with the point-quality gate (`q_seed`, `q_young`) off removes the id switches among the new points (609 to 108), but doubles the ghosts (3.3 to 7.5 %) and raises the velocity error from 0.117 to 0.163; the two-hop confirmation off is worse (0.161). The light check does not make the gate safe to drop (syn1, 100 frames, ratio 1.05 + light check: velocity error 0.114 with the gate, 0.132 without). The gate removes ghosts and real new points together; keep it.

**syn1, truth, real pipeline, `shared_correction` on:**

| 400 frames | yield | ghost | link precision | velocity error | id switches |
|---|---|---|---|---|---|
| base | 0.675 | 3.56 % | 0.9645 | 0.1189 | 10792 |
| light check only (10) | 0.669 | 2.58 % | 0.9736 | 0.1185 | 9713 |
| shared pass (1.7) | 0.694 | 3.58 % | 0.9643 | 0.1188 | 11846 |
| pass + light check z>=4 | 0.691 | 2.86 % | 0.9710 | 0.1174 | 11534 |
| pass + light check z>=6 | 0.690 | 2.71 % | 0.9724 | 0.1132 | 11116 |
| pass + light check z>=8 | 0.688 | 2.64 % | 0.9731 | 0.1133 | 10960 |
| pass + light check z>=10 | 0.687 | 2.59 % | 0.9736 | 0.1134 | 10804 |

(100 frames, same direction: pass + light check 0.1030 against 0.1102 base; loosening the ratio to 1.05 costs velocity error, 0.1140.) z>=6 is the default: velocity error as low as at 10, more yield, and on real data more long-track points.

**Real recordings, 300 frames, `shared_correction` on, truth-free** (points in tracks of 50+ frames per frame; singletons; roughness p90; LOCO median):

| | wp2 | wp1 |
|---|---|---|
| base | 587; 11.2 %; 0.1485; 0.347 | 579; 11.0 %; 0.1499; 0.349 |
| light check only (10) | 551; 8.6 %; 0.1456; 0.347 | 544; 8.5 %; 0.1468; 0.349 |
| pass | 649; 11.3 %; 0.1488; 0.348 | 648; 11.1 %; 0.1504; 0.350 |
| pass + z>=4 | 626; 9.1 %; 0.1478; 0.348 | 627; 9.0 %; 0.1494; 0.350 |
| pass + z>=6 | 621; 8.8 %; 0.1469; 0.348 | 621; 8.7 %; 0.1485; 0.350 |
| pass + z>=8 | 614; 8.7 %; 0.1463; 0.348 | 614; 8.6 %; 0.1478; 0.350 |
| pass + z>=10 | 611; 8.5 %; 0.1459; 0.348 | 610; 8.4 %; 0.1471; 0.350 |

The pass adds 11-12 % points in long tracks; the check takes 4-6 % of them back and removes the singletons (11 to 9 %) and the roughness tail; the leave-one-camera-out residual stays 0.347-0.351. Matching time 4.7 s to 8-10 s over 300 frames (tracking 28 s). On the other camera setup (lv-multi wp5, 20 frames) the check removes 20 % of the points at z>=10 (3-camera points 52 to 40 %) and 17 % at z>=6, mean track length 10.7 to 11.9 and roughness tail 0.1725 to 0.1606 (pass against pass + z>=10); its run store for wp4 holds no targets, so only wp5 was usable.

**Limits.** No truth on the real recordings: the removed points that are real cannot be counted, only bounded (hidden views of 4-camera points fall below z=6 in 2.6 % on wp2, 1 % on syn1). One synthetic case with images exists. The full unit suite (2132 passed, 94 skipped) has one failure, `test_sequence_triangulation_method::test_lsq_triangulation_reprojects_better_than_pairs` ('run.zarr does not exist'), which also fails with these changes stashed. The GUI has no checkbox for `shared_pass` or `light_check` (YAML only).

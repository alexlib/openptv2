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

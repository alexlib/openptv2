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

## 11. What is left

- **Holistic loop speed.** Batch the refine step: group the blobs of a frame and camera by member count (2, 3, 4), run
  `split_blob_effective`, the gate test and the projections on whole groups at once (young members and hidden partners as
  masks), and project all points of a frame and camera in one call. Prove each step equal to the current loop on the
  benchmark before replacing it. Target: the loop at most 2 to 3 times the `two_phase` time (about 15 s), from about 110 s.
  This decides whether holistic can ever be a default.
- **Stage 6, deletion** of `unified`, `4be`, `hybrid_deltat_3d`, the splitter alias and `tracking_presets.py` after one
  release of warnings (decision C). Not before.

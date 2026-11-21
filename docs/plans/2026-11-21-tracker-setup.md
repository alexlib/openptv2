# One clear tracker setup: three trackers, two switches, one add-on

Date: 2026-11-21. Branch `plan/what-we-missed`. Follows the comparison of `3d_hybrid` and `two_phase`
(they are near-twins under different names) and the legacy comparison in
`2026-11-21-detection-hypotheses-and-events.md` section 9.

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

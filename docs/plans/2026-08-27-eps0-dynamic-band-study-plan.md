# Plan: study porting 3dptv.exe's dynamic per-particle epipolar band into openptv2

Companion to `2026-08-27-track3d-beat-gt-plan.md`. Background and the
conversion formula derived so far live in
`docs/algorithms/correspondence-eps0-semantics.md` — read that first. This
plan is narrower: should openptv2 gain a dynamic (`find_candidate_plus`-style)
`eps0` scaling mode, and if so, how do we build and validate it without
breaking every existing dataset that already tunes a flat `eps0`.

## Why this is worth studying (not just closing the wp1 ticket)

wp1 turned out to be the *easy* case: its particles are small and uniform
enough (`max(nx,ny)` 2–8px) that 3dptv's dynamic formula collapses to a flat
0.06mm floor for 100% of detected targets — `flat_eps0_mm = 0.06` is exact,
not an approximation. That's a coincidence of this dataset's tracer size, not
a general property. Any other legacy 3dptv dataset with larger or
size-varied particles (dumbbells, bigger tracers, mixed populations) will
have `floor_fraction < 1.0`, where a single flat value is provably lossy: it
either starves recall for large particles or inflates false candidates for
small ones. Every future "convert a 3dptv dataset to openptv2" task hits this
same risk. Building the dynamic mode once removes the guesswork for all of
them.

## Study questions, in order

1. **How common is `floor_fraction < 1.0` in practice?** Run the
   `floor_fraction` diagnostic (formula in the eps0-semantics doc) across
   every legacy dataset we have `.par` + detected-target data for (wp1,
   burgers, any others in `test_data/`). If every real dataset we have access
   to also saturates the floor, the dynamic mode is a correctness nice-to-have
   with no measured benefit yet — deprioritize. If even one dataset shows a
   meaningfully lower `floor_fraction`, that's the validation case for the
   dynamic mode below.
2. **What's the actual point-count/precision effect of the dynamic formula
   vs. the best flat approximation**, on whichever dataset from (1) has the
   lowest `floor_fraction`? Build a standalone (non-compiled, plain
   numpy/Python) reimplementation of `_build_adjacency_for_pair`'s inner loop
   with the dynamic formula, run it against that dataset's stored 2D targets,
   and diff candidate counts / final 3D point counts against both the current
   flat-eps0 production path and (if available) ground truth. This validates
   the win *before* touching the compiled kernel.
3. **What's the performance cost** of the extra per-candidate
   multiply+compare in the innermost matching loop
   (`_build_adjacency_for_pair`, called O(N₁×N₂) times per camera pair per
   frame)? Benchmark the standalone reimplementation's overhead relative to
   the current flat comparison before deciding whether it needs to be
   precomputed per-source-particle-once (it can be — `tol_i` only depends on
   the source particle `i`, not the inner `j` loop, so it hoists out of the
   inner loop entirely; the cost should be O(N₁), not O(N₁×N₂)).

## Implementation sketch (only after questions 1-3 justify it)

- Add an opt-in switch, not a default-behavior change — e.g.
  `criteria.eps0_mode: flat | dynamic` (default `flat`, current behavior
  unchanged) or a `criteria.eps0_particle_scaled: bool`. Exact shape TBD
  during implementation; the constraint is that every existing dataset's YAML
  keeps working unmodified.
- Both call sites need the change to stay consistent:
  `openptv2/algorithms/epi.py::find_candidate` (used by the pure-Python
  fallback / some callers) and
  `openptv2/algorithms/correspondences.py::_build_adjacency_for_pair` (the
  actual production per-pair matching kernel `match_pairs` drives). Confirm
  there isn't a third call site before starting.
- The 0.06mm floor should be a parameter, not a hardcoded constant carried
  over from 3dptv's assumptions about its own sensor/lens setup — expose it
  (e.g. `criteria.eps0_floor_mm`, default 0.06 to match 3dptv when the
  dynamic mode is on).
- Requires a Cython rebuild (`uv run python setup.py build_ext --inplace`)
  and a full run of the correspondence/correlation parity test suite with the
  new mode *off* (must be byte-identical to today) before any dataset is
  switched to the new mode on.
- Add the `floor_fraction` diagnostic as a reusable helper (candidate home:
  `openptv2.calibration_diagnostics`, alongside the existing sight-line-angle
  and cross-camera-RCM checks — same spirit, a pre-flight sanity number for a
  dataset's config) rather than a one-off script, so future dataset
  conversions get it automatically via `calib.py run` / the `validate`
  command in `skills/openptv-params`.

## Out of scope for this plan

- Actually changing wp1's production `eps0` value further — already resolved
  (`flat_eps0_mm = 0.06`, exact for this dataset, see the semantics doc).
- Any other legacy-vs-openptv2 parameter semantics mismatch (e.g. `csumg`,
  `corrmin`) — only in scope if the same investigation pattern surfaces a
  similar unit/definition mismatch for another parameter; track those
  separately if found.

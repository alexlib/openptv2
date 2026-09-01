# Plan: find and fix the double-claim / long-jump bug in backward tracking + postprocess

> **Update 2026-09-01:** After zarr-only cutover (`archive/2026-09-01-zarr-only-final-cutover-plan.md`, `6a1e81aa`), diagnostics here should use `RunStore.read_linkage` (`linkage/ptv_is/frame_*`) not `res/ptv_is.*` files (already store-branched in `tracking_postprocess.py`).

## The bug, confirmed today

`trackcorr` with `direction: forward_backward` + `postprocess: true`
(`parameters_Run1.yaml`, wp1 dataset, `img_3dptv` images, frames
100001-100010) produces trajectories where **two different source particles
link to the same target point** — physically impossible; one 3D point can
only be the continuation of one particle.

Measured: **185 double-claimed target points** across the 10-frame sequence.
Traced one concretely (frame 100002, target index 1281): a legitimate link
existed (source 1289, 0.082mm away — a real, tight match) but a second
source (77) *also* claimed that same target from **26.143mm away** — over
13x the configured `dvxmax=1.9`mm velocity limit. That link should have
been rejected outright by the search box and wasn't.

**Confirmed fix (empirically, not yet root-caused):** switching to
`direction: forward` + `postprocess: false` (plain forward tracking only)
drops the double-claim count to **0** on the same dataset, same detections,
same correspondences. The bug is isolated to backward tracking and/or
postprocess — the core forward pass is clean.

## What's not yet known

`direction: forward_backward` triggers `full_forward()` then
`full_backward()` (which calls `trackback_c`, `openptv2/algorithms/track.py`).
`postprocess: true` then runs three separate sub-passes in sequence
(`Tracker.postprocess()`, `openptv2/tracker.py:209`, delegating to
`openptv2/tracking_postprocess.py`):
1. `seed_cold_start` — recovers the under-linked first transition using
   later frames' velocity field.
2. `enforce_reciprocity` — severs any non-bidirectional link.
3. `relink_trajectory_gaps` (`max_gap=2` default) — bridges occluded-particle
   gaps across skipped frames.

Any of these four passes (`trackback_c`, `seed_cold_start`,
`enforce_reciprocity`, `relink_trajectory_gaps`) could be the source — today
only proved "backward+postprocess together" vs "neither," not which one(s)
specifically. `relink_trajectory_gaps` is the prior suspect worth watching
first: it explicitly reconnects across *missing* frames, which is exactly
the kind of pass that would need its own distance/velocity check
independent of the normal single-frame search box — and if that check is
missing or uses the wrong tolerance (an existing pattern in this codebase,
per `cython_3d_tracking.py`'s own docstring noting a fixed instance of
`gap_relinking` being "handed `dvxmax`, a velocity gate, as an
acceleration-scale tolerance" for `track3d`'s equivalent pass), the same
class of bug in `trackcorr`'s version would explain a 26mm jump slipping
through a 1.9mm gate exactly like this.

## Diagnostic already built and validated

`double-claim count` (any target point in frame t+1 claimed as `next` by
more than one source in frame t) is now a proven, sensitive signal — 185 on
the buggy config, exactly 0 on the clean one, computed directly from
`RunStore.read_linkage`. Reuse this exact check (a ~15 line script, already
run twice this session) as the pass/fail gate for every step below, plus the
specific long-jump trace (source-to-target distance for each double-claim)
to characterize *how* wrong each one is, not just count them.

## Study steps, in order

1. **Bisect the four passes one at a time**, starting from the known-clean
   baseline (`direction: forward`, `postprocess: false`) and adding exactly
   one back at a time:
   - forward + `trackback_c` only (no postprocess) → double-claim count?
   - forward + postprocess only, `seed_cold_start` alone (disable the other
     two via `Tracker.postprocess(reciprocity=False, gap_relinking=False)`)
   - forward + postprocess, `enforce_reciprocity` alone
   - forward + postprocess, `relink_trajectory_gaps` alone
   - This isolates which single pass (or combination) introduces the
     double-claims, rather than assuming.
2. **For the guilty pass(es)**, read the actual implementation
   (`openptv2/algorithms/track.py::trackback_c` and/or
   `openptv2/tracking_postprocess.py`'s three functions) and find the
   specific missing/wrong distance check — likely candidates: a search
   radius that isn't being applied at all for this pass, or a parameter
   mix-up (an unrelated tolerance substituted for the real velocity/dacc
   gate, matching the `track3d` precedent noted above).
3. **Confirm the fix closes ALL 185 double-claims**, not just the one traced
   example — rerun the same diagnostic on the full sequence after the fix,
   same as the forward-only baseline already showed 0.
4. **Check whether uniqueness (no double-claims) was ever enforced anywhere**
   downstream, or whether every consumer of this linkage (trajectory
   building, this session's scoring scripts, any GUI display) silently
   assumes one-to-one links and produces a corrupted result whenever it
   isn't — if so, consider adding an assertion/validation step in
   `Tracker.postprocess()` or `RunStore.read_linkage` itself so a future
   regression here fails loudly instead of silently drawing bad trajectories
   (a visualization artifact was how this was first noticed, not a test
   failure — that gap is itself worth closing).
5. **Re-validate on wp1's original splitter-based dataset**
   (`parameters_wp1.yaml`) once fixed on `parameters_Run1.yaml` — confirm
   the fix isn't dataset-specific, since both configs exist and use the
   same underlying `trackcorr` code path.

## Out of scope for this plan

- Whether backward tracking / postprocess should be `true` by default —
  that's a separate design decision to make once the bug itself is fixed
  and verified safe, not before.
- The unrelated `track3d`/`dacc=0` jumpiness discussed earlier in this
  session (that one is understood: `dacc=0` intentionally disables the
  acceleration gate for `track3d`, not a bug) — do not conflate the two
  findings.

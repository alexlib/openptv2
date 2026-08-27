# Structural direction: tracking as one joint leaf-to-leaf forest

Companion to `2026-08-27-track3d-beat-gt-plan.md` (the near-term patch list:
Level 2 losers-retry, adaptive cold-start gate, Level 1 angle term). This
document is the longer-term redesign direction that plan's trajectory-tree
diff tool is actually building toward — not something to implement this
week, but the shape to keep steering the patches toward so they compose
into it rather than away from it.

## The reframing

Per frame, 2D detections in each camera are leaves; correspondence-solving
combines leaves across cameras into a 3D point — the root of a small
per-frame tree. Tracking then links roots across frames into trajectories.
Today these are two independent trees solved once each, in strict sequence:
build the per-frame root (correspondence), throw the leaves away, link roots
across time using 3D kinematics only (`track3d`) or re-derive candidate
leaves from scratch per step via fresh pixel-space search (`track.c` —
better information, but rebuilt per step rather than carried as one
structure). Neither treats the whole thing as one connected forest that
persists leaf identity through time.

## What this exposes, structurally

1. **Single-channel acceptance is the root cause of several findings so
   far.** A wrong link is often kinematically smooth in 3D (a ghost point
   sitting where the prediction expects) while its leaves — the per-camera
   2D detections that built it — don't persist smoothly in image space
   across that same step. `track3d` can't see this because correspondence
   discards the leaves once the 3D root is built. This is the same failure
   *shape* as the proven 4BE bug (`docs/holistic-3d-ptv-systems-research-program.md`
   §2): a single evidence channel can be won by coincidence; two independent
   channels that must both agree (3D kinematic consistency AND per-camera
   leaf-path consistency) can't be fooled the same way. It also explains,
   structurally, why `track.c` (pixel-space, keeps leaves) tracks closer to
   GT than `track3d` (pure 3D, leaves discarded) even though `track3d`'s
   cost function is not obviously worse — it has strictly less information
   to work with.

2. **The 3-level cascade (and `track.c`'s claim-in-cost-order conflict
   resolution) are both greedy, non-backtracking local tree-builds.**
   Yesterday's Level 2 finding — 43/76 FN because Level 1 claims first and
   exclusively, with no fallback for the loser — is a symptom of treating a
   joint assignment problem as a strict priority cascade instead of solving
   it as an assignment. The "losers retry" patch (already in
   `trackcorr_c_loop`, planned for `track3d`'s Level 2) is a hand-rolled
   approximation of running a real assignment solver (Hungarian/LAP) over a
   frame pair, or a short multi-frame window. Structural upgrade path: model
   linking explicitly as **windowed min-cost assignment** over the joint
   node set (3D roots with their leaf bundles), not a fixed 3-level greedy
   cascade with ad hoc backtracking bolted on case by case as each starvation
   pattern is found.

3. **A trajectory is really a bundle of persistent leaf-paths (one per
   camera) plus one root-path (3D), and the bundle gives a free invariant**:
   wherever a camera had visibility, that camera's 2D sub-path should also
   stay smooth. A 3D jump that is *also* a jump in every leaf is very likely
   a real event (occlusion, entry/exit at the volume edge). A 3D jump where
   the leaves stayed smooth is very likely a correspondence-stage error
   wrongly attributed to tracking. This reframes some of the "jump" defects
   `cold_start_gate` currently suppresses purely in 3D as detectable, and
   distinguishable from real events, using information the pipeline already
   computes at correspondence time and currently throws away.

4. **This reframes "which tracker family wins" as the wrong question.**
   `track.c` vs `track3d.c` is a proxy fight over who keeps more information
   (leaves) available at decision time. The real target is one joint linker
   that carries leaf identity through the whole forest and requires the 3D
   and 2D channels to agree, rather than choosing between a 2D-aware/heavy
   design (`track.c`) and a 2D-blind/fast design (`track3d`).

## How this should shape near-term work (not a rewrite mandate)

- The trajectory-tree diff tool already planned (extending
  `analyze_trajectory_trees.py`) is the right foundation: it already carries
  per-node 3D position + per-camera 2D target. Extend its *use* from "diff
  two trackers' decisions" to "check leaf-path smoothness alongside
  root-path smoothness for one tracker's own output" — same data structure,
  additional consistency check.
- Before building a real windowed-assignment solver, first measure how much
  headroom it would actually buy: instrument how often Level 1/2/3's greedy
  claim order (not cost ranking) is the actual reason the correct edge
  lost, vs. how often the correct edge simply wasn't the lowest-cost edge
  even with unlimited backtracking. The former justifies an assignment
  solver; the latter means the cost function (item 3 in the patch plan —
  angle term) is the real lever, and a solver would just optimize the wrong
  objective faster.
- Treat "does a 3D jump correspond to a leaf-path jump too" as a candidate
  *feature* for `cold_start_gate`-style decisions before treating it as a
  reason to merge the two tracker families into one implementation — it may
  be enough to add one leaf-consistency check to `track3d`'s existing
  cascade rather than rebuilding tracking as joint assignment from scratch.

## Open questions to resolve before committing engineering time

- Does wp1's GT store retain the original per-camera 2D targets anywhere
  (`res_ground_truth_backup/` — check for `*_targets` files), or would a
  leaf-consistency check have to run on our own re-detected targets (which
  reintroduces the detection-stage contamination this session already found
  and worked around once)?
- What is the actual cost of carrying leaf bundles through a multi-frame
  window in the compiled Cython kernels — is this a data-structure change
  only, or does it also mean giving up the current per-level vectorized
  edge-sort (`np.argsort` over a flat edge-cost buffer), which is where
  most of `track3d`'s speed advantage over `track.c` comes from?

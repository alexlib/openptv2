# Plan: study the 4-camera quad-uniqueness-pass gap between openptv2 and 3dptv.exe

Companion to `2026-08-27-track3d-beat-gt-plan.md` and
`2026-08-27-eps0-dynamic-band-study-plan.md`. Where those two are about
`eps0`, this one is about a second, independent correspondence-stage finding
from the same wp1 investigation — and the eps0 sweep (see
`wp1_10_images/scripts/point_cloud_diff.py`) now shows it's the **dominant**
of the two: no `eps0` value (tested 0.02–0.20) gets openptv2's total point
count within 35% of GT's, and openptv2 is still ~1.9x GT's *quadruplet* count
even at the tightest tested `eps0=0.02`.

## The finding

`3dptv/src_c/correspondences.c`'s greedy "take best quadruplets, skip if any
point already used" pass has a real quirk in its C:

```c
p1 = con0[i].p[0];  if (p1 > -1 && ++tim[0][p1] > 1) continue;
p2 = con0[i].p[1];  if (p2 > -1 && ++tim[1][p2] > 1) continue;
p3 = con0[i].p[2];  if (p3 > -1 && ++tim[2][p3] > 1) continue;
p4 = con0[i].p[3];  if (p4 > -1 && ++tim[3][p4] > 1) continue;
con[match++] = con0[i];
```

Each point's usage counter is incremented **as it's checked, in order** —
before the candidate is known to survive all four checks. A candidate that
fails at `p2` has *already* incremented `tim[0][p1]`, even though it never
gets committed to `con`. A later candidate that also uses that same `p1` then
sees `tim[0][p1]` already at 1 and gets rejected on the very first check —
even though the *first* candidate touching `p1` was itself never accepted.
This is an ordering artifact of the C loop, not a documented design choice —
nothing in the surrounding code or comments suggests it's intentional.

openptv2's `take_best_candidates` (`correspondences.py:823`) does the clean
version: check all four points are free first, only then commit all four.
Strictly more permissive than 3dptv's quirky pass — it accepts every quad
3dptv would have, plus some 3dptv's ordering bug incorrectly threw away.

## Why this matters more than it might sound

The wp1 point-cloud diff (`point_cloud_diff.py`) shows the practical scale:
at `eps0=0.06` (the corrected flat value), of ~1854 points/frame we produce,
only ~35% match a GT point within 0.3mm; of GT's ~1281 points/frame, only
~51% are matched. The "extra" points on our side are **not** dominated by
weak, low-camera-count matches — the breakdown is `{quad: 6939, trip: 4692,
pair: 355}` across all 10 frames, i.e. the excess is overwhelmingly
4-camera-consistent (or 3-camera-consistent) correspondences, exactly the
population this dedup pass governs. Symmetrically, GT has points we never
reproduce at all (`{quad: 2538, trip: 2944, pair: 778}`) — so this isn't just
"we keep everything GT keeps plus junk"; the *set* of which ambiguous
candidate wins differs, consistent with an ordering-dependent resolution
difference rather than a simple threshold difference.

One reassuring result from the same diff: for the ~6551 points that *do*
match a GT point, position residuals are tight (mean 0.112mm, median 0.089mm,
p95 0.273mm) — ray-tracing/triangulation (`point_positions`) agrees with
3dptv.exe wherever the two pipelines agree on *which* points to reconstruct.
The problem is entirely in candidate selection, not in the geometry math.

## Study questions, in order

1. **Faithfully replicate 3dptv's ordering quirk in a standalone (non-compiled)
   reimplementation** of the greedy pass — a direct one-to-one port of the
   `++tim[...] > 1` short-circuit sequence, applied to openptv2's own sorted
   quad candidate list (same input as `take_best_candidates` gets today).
   Compare its output point count and per-frame quad count against GT
   directly. This isolates whether the ordering quirk alone (not eps0, not
   detection differences) accounts for the ~1.9x quad excess, or only part of
   it.

   **Done, 2026-08-27** (`wp1_10_images/scripts/replicate_3dptv_uniqueness_pass.py`):
   ran both the clean pass (openptv2's actual `take_best_candidates` logic)
   and the ordering-quirked replica against the *same* raw candidate list
   from `four_camera_matching`, all 10 wp1 frames, `eps0=0.06`:

   ```
   gt_quad=7376  clean_quad=12562  buggy_quad=10830
   clean excess over GT: 5186 (70%)
   buggy excess over GT: 3454 (47%)
   ordering quirk explains 33% of the clean-vs-GT gap
   ```

   **Confirmed but partial**: the quirk is real and measurably contributes,
   but two-thirds of the gap survives even with an identical uniqueness pass.
   The raw candidate count feeding the uniqueness pass (8600–9080/frame here,
   vs. GT's final 703–780 quads/frame) means the dominant remaining factor is
   upstream — in what `match_pairs` generates as candidates *before* dedup,
   not in the dedup logic itself. Re-scope: don't invest further in the
   ordering-quirk theory alone; move to candidate-generation-volume causes
   (question 3, promoted to primary).

   **Resolved, 2026-08-27, with 3dptv's own raw detections**
   (`wp1_10_images/img_3dptv/Cam{1-4}.{frame}_targets`, provided by the user
   — 3dptv.exe's actual per-camera 2D target output, not openptv2's):
   `MAXCAND` (200 in both — ruled out) and raw target counts were checked;
   `match_pairs` at all frames: 3dptv detects 66,735 total raw targets across
   all cameras/frames, openptv2 detects 81,298 — a consistent **~22% excess**
   at the detection stage, threshold values identical
   (`targ_rec.par`/YAML match exactly). Feeding 3dptv's own real detections
   through openptv2's matcher (`verify_3dptv_detections.py`) with the CLEAN
   dedup already drops the excess from 70% to 16%; adding the ordering-quirk
   replica on top of 3dptv's real detections closes it to **-1%**
   (`gt_quad=7376` vs `buggy_quad=7287`, several individual frames matching
   exactly). **The two factors found this session (detection-stage volume +
   uniqueness-pass ordering) together fully account for the point-cloud
   excess** — no further correspondence-stage cause needs to be hunted.
   Remaining open items, now clearly scoped:
   - **Detection-stage fidelity** (why ~22% more raw targets at identical
     thresholds) is its own investigation — the underlying peak-finding /
     connected-component algorithm, not a parameter — out of scope for this
     plan; track separately if pursued.
   - **Whether to adopt the ordering-quirk in production** is still an open
     decision (see "Open question" below), now with much stronger evidence:
     even with clean, correct detections, the *clean* dedup logic still
     accepts 16% more quads than 3dptv's buggy one — worth deciding whether
     that's openptv2 legitimately doing better, or genuine over-acceptance,
     before touching production code.
2. **If it accounts for most of the gap**: decide whether to actually adopt
   the quirky behavior in production, or keep the clean version and instead
   verify the "extra" quads independently (do they hold up under a stricter
   physical check — e.g. reprojection residual across all 4 cameras, or
   Level-1/2/3 tracking-cascade classification from
   `classify_by_level.py` — are they trackable into consistent trajectories,
   or do they look like transient junk?). Don't replicate a C ordering bug
   without evidence the "bug" version is actually the more correct one for
   *this* purpose — see "Open question" below.
3. **If it accounts for only part of the gap**: the remainder needs its own
   investigation — likely candidates are differences in `quicksort_con`'s
   sort stability/tie-breaking (3dptv's C `qsort` is not guaranteed stable;
   openptv2's `np.argsort` also isn't by default — same-`corr` ties could
   resolve differently between the two, feeding different candidates into an
   otherwise-identical uniqueness pass) or a difference in how many candidate
   quads even get *generated* before the uniqueness pass runs (check
   `list[...].n` truncation via `maxcand`/`MAXCAND` — is 3dptv's per-pair
   candidate cap the same value as openptv2's?).

## Open question: is the "bug" actually better dropped?

3dptv.exe's ordering quirk makes its dedup *more* conservative than a correct
greedy algorithm — it's an accidental **precision-over-recall** bias, not
necessarily a defect to fix. openptv2's clean version is *more* internally
consistent (it implements the documented "skip if used" rule correctly) but
that alone doesn't mean the extra quads it accepts are junk — 3dptv might be
dropping perfectly good quads for accidental reasons. This plan's step 2
(reprojection / trackability check on the extra quads) is what actually
answers "should we chase GT-parity here, or is openptv2's version already
better and GT undercounts real particles" — don't assume either answer before
running it.

## Also worth checking in step 1's replication

- `maxcand`/`MAXCAND`: confirm openptv2's per-pair candidate cap matches
  3dptv's `maxcand` (both need the same value or the candidate lists feeding
  the uniqueness pass already differ before ordering even matters).
- Sort stability: `quicksort_con` (3dptv) vs `np.argsort` (openptv2,
  `take_best_candidates:834`) on tied `corr` values — test with synthetic
  tied-score input to confirm they don't silently diverge.

## Out of scope for this plan

- The `eps0` semantics gap — tracked separately
  (`2026-08-27-eps0-dynamic-band-study-plan.md`); the sweep here already
  shows `eps0` isn't the dominant lever, but it's not zero-effect either
  (point count did increase 0.02→0.10) — note also that
  `wp1_10_images/scripts/point_cloud_diff.py`'s sweep shows a suspicious
  **non-monotonic collapse** above `eps0=0.10` (18696 at 0.10 → 7542 at 0.15
  → 4211 at 0.20) that looks like a separate bug (numerical edge case,
  candidate-array truncation, or a sensor-boundary check misbehaving at wide
  tolerances) — flag for a quick look if anyone tunes `eps0` above ~0.10 on
  any dataset, but not otherwise in scope here.

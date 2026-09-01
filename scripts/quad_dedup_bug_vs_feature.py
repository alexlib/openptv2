"""Phase 2: is 3dptv's quad-uniqueness ordering quirk a BUG or a FEATURE?

docs/plans/2026-08-27-verified-pipeline-ghost-particle-study-plan.md

The quirk (3dptv/src_c/correspondences.c): the greedy "take best quads,
skip if any point already used" pass increments each point's usage counter
AS IT CHECKS IT, in order, and does NOT roll back when a later point fails.
So a candidate rejected at p3 has already burned p1 and p2 -- a later,
otherwise-valid quad using p1 is then wrongly rejected too.

openptv2's take_best_candidates checks all 4 free BEFORE committing any.

MECHANISM (proved by inspection, verified by gate 2 below): the quirk is
STRICTLY more restrictive -- it can only ever reject more, never accept
more. So "buggy has higher precision, lower recall" is NOT evidence it is
better; that is just a point on a precision/recall tradeoff curve, and any
matcher can be made more conservative by raising corrmin.

THE SHARP QUESTION, and what this script actually measures:
  Does the quirk's PARTICULAR way of being conservative (burning points of
  rejected candidates -- i.e. using conflict topology) beat the OBVIOUS
  alternative (just raise corrmin until you accept the same number)?

    buggy  >  clean-tightened-to-same-N  -> the quirk encodes real
        structural information; worth implementing DELIBERATELY (as a
        principle, not a quirk-port).
    buggy ~= clean-tightened-to-same-N  -> it is merely a randomized way of
        being pickier; openptv2's clean version + a tunable corrmin
        strictly dominates (simpler, monotone, explainable). Do nothing.
    buggy  <  clean-tightened-to-same-N  -> actively harmful; it discards on
        an ordering artifact rather than on evidence quality. Do nothing.

Ground truth is synthetic particle identity, so "correct quad" is exact:
all 4 targets must come from the same true particle.
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIX = HERE.parent / "test_data" / "tracking_synthetic_dense"
sys.path.insert(0, str(FIX))
from generate import BASE_FIXTURE, NCAM  # noqa: E402

from openptv2.algorithms.calibration import Calibration  # noqa: E402
from openptv2.algorithms.constants import TR_UNUSED  # noqa: E402
from openptv2.algorithms.correspondences import (  # noqa: E402
    allocate_adjacency_arrays,
    correct_frame,
    four_camera_matching,
    match_pairs,
)
from openptv2.algorithms.imgcoord import img_coord  # noqa: E402
from openptv2.algorithms.parameters import ControlPar, VolumePar  # noqa: E402
from openptv2.algorithms.tracking_frame_buf import Frame, Target  # noqa: E402
from openptv2.algorithms.trafo import metric_to_pixel  # noqa: E402

NMAX = 20000


def build_one_frame(n_particles, noise_px, seed, half_extent=(25.0, 20.0, 15.0)):
    """One synthetic frame: real projection + noise, real correspondence
    candidate generation. Returns everything needed to score identity.

    Mirrors build_fixture_with_correspondence's projection/target logic
    (test_data/tracking_synthetic_dense/generate.py) but stops BEFORE
    dedup, and hands back the pieces that function keeps internal
    (corrected, pnr_to_pid) so both dedup variants can be applied to the
    exact same candidate list and scored against particle identity.

    Particles are sampled uniformly INSIDE the search volume rather than via
    build_scene's spacing-based scatter. build_scene(spacing_mm=12) puts ~40%
    of particles outside this fixture's vpar volume (Z in [-20,20]); the
    matcher then correctly refuses to match them, which silently deflates
    recall against a denominator of unobservable particles. Sampling inside
    the volume makes the ground-truth denominator honest AND makes density
    -- the variable that actually drives correspondence ambiguity, and so
    the whole bug-vs-feature question -- directly controllable via
    n_particles at fixed extent.
    """
    cwd = os.getcwd()
    try:
        os.chdir(BASE_FIXTURE)
        cpar = ControlPar.from_yaml(os.path.join(BASE_FIXTURE, "parameters_Run1.yaml"))
        vpar = VolumePar.from_yaml(os.path.join(BASE_FIXTURE, "parameters_Run1.yaml"))
    finally:
        os.chdir(cwd)
    mm = cpar.mm

    cals = []
    for c in range(NCAM):
        cal = Calibration()
        cal.from_file(
            os.path.join(BASE_FIXTURE, "cal", f"cam{c + 1}.tif.ori"),
            os.path.join(BASE_FIXTURE, "cal", f"cam{c + 1}.tif.addpar"),
        )
        cals.append(cal)

    rng = np.random.default_rng(seed + 1)
    hx, hy, hz = half_extent
    P_all = np.column_stack(
        [
            rng.uniform(-hx, hx, n_particles),
            rng.uniform(-hy, hy, n_particles),
            rng.uniform(-hz, hz, n_particles),
        ]
    )

    # Keep only particles visible in ALL 4 sensors -- those are the ones a
    # quad matcher can legitimately find, so they are the honest ground-truth
    # denominator for recall.
    keep = []
    for p in range(n_particles):
        ok = True
        for c in range(NCAM):
            mx, my = img_coord(P_all[p], cals[c], mm)
            px, py = metric_to_pixel(mx, my, cpar)
            if not (0 <= px < cpar.imx and 0 <= py < cpar.imy):
                ok = False
                break
        if ok:
            keep.append(p)
    P = P_all[keep]
    n = len(P)

    pix = np.zeros((NCAM, n, 2))
    for c in range(NCAM):
        for p in range(n):
            mx, my = img_coord(P[p], cals[c], mm)
            pix[c, p] = metric_to_pixel(mx, my, cpar)
    pix += rng.normal(0.0, noise_px, pix.shape)

    frm = Frame(num_cams=NCAM, max_targets=n)
    pnr_to_pid = [dict() for _ in range(NCAM)]
    for c in range(NCAM):
        order = np.argsort(
            pix[c, :, 1], kind="stable"
        )  # y-sort, as real detection does
        frm.num_targets[c] = n
        for pnr, p in enumerate(order):
            x, y = pix[c, p]
            frm.targets[c][pnr] = Target(
                pnr=pnr, x=x, y=y, n=100, nx=10, ny=10, sumg=1000, tnr=TR_UNUSED
            )
            pnr_to_pid[c][pnr] = int(p)

    corrected = correct_frame(frm, cals, cpar, 0.0001)
    return frm, corrected, pnr_to_pid, cals, cpar, vpar, n


def candidate_quads(frm, corrected, cals, cpar, vpar, corrmin):
    """Raw 4-camera candidate list, pre-dedup, sorted by descending corr."""
    num_targets = list(frm.num_targets)
    p1_arr, n_arr, p2_arr, corr_arr, dist_arr = allocate_adjacency_arrays(
        NCAM, num_targets
    )
    match_pairs(
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr, corrected, frm, vpar, cpar, cals
    )

    con0_size = NCAM * NMAX
    con0_p = np.full((con0_size, NCAM), -1, dtype=np.int32)
    con0_corr = np.zeros(con0_size, dtype=np.float64)
    match0 = four_camera_matching(
        p1_arr,
        n_arr,
        p2_arr,
        corr_arr,
        dist_arr,
        num_targets[0],
        corrmin,
        con0_p,
        con0_corr,
        con0_size,
    )
    src_p = con0_p[:match0]
    src_corr = con0_corr[:match0]
    order = np.argsort(src_corr)[::-1]
    return src_p, order, max(num_targets)


def dedup_clean(order, src_p, max_targ):
    """openptv2's take_best_candidates: all 4 free BEFORE committing any."""
    tusage = np.zeros((NCAM, max_targ + 1), dtype=np.int32)
    taken = []
    for cand in order:
        if any(
            src_p[cand, c] > -1 and tusage[c, src_p[cand, c]] > 0 for c in range(NCAM)
        ):
            continue
        for c in range(NCAM):
            if src_p[cand, c] > -1:
                tusage[c, src_p[cand, c]] += 1
        taken.append(cand)
    return taken


def dedup_buggy(order, src_p, max_targ):
    """3dptv's ordering quirk: increment as you check, no rollback on failure."""
    tim = np.zeros((NCAM, max_targ + 1), dtype=np.int32)
    taken = []
    for cand in order:
        ok = True
        for c in range(NCAM):
            p = src_p[cand, c]
            if p > -1:
                tim[c, p] += 1  # incremented BEFORE knowing the rest pass
                if tim[c, p] > 1:
                    ok = False
                    break  # ... and NOT rolled back. This is the quirk.
        if ok:
            taken.append(cand)
    return taken


def score(taken, src_p, corrected, pnr_to_pid, n_particles):
    """Exact identity scoring against synthetic ground truth.

    con0_p[k,c] is an INDEX into corrected[c] (x-sorted), NOT a pnr --
    translate via corrected[c][idx].pnr before identity lookup. (Same
    convention documented and verified in generate.py.) Getting this wrong
    silently corrupts every number below.
    """
    correct_pids = set()
    n_correct = 0
    for cand in taken:
        pids = []
        for c in range(NCAM):
            idx = src_p[cand, c]
            if idx < 0:
                continue
            pnr = corrected[c][idx].pnr
            pids.append(pnr_to_pid[c][pnr])
        if len(pids) == NCAM and all(p == pids[0] for p in pids):
            n_correct += 1
            correct_pids.add(pids[0])
    n_acc = len(taken)
    return dict(
        accepted=n_acc,
        correct=n_correct,
        ghosts=n_acc - n_correct,
        precision=n_correct / n_acc if n_acc else float("nan"),
        recall=len(correct_pids) / n_particles,
    )


def run_case(n_particles, noise_px, seed, corrmin_base):
    frm, corrected, pnr_to_pid, cals, cpar, vpar, n = build_one_frame(
        n_particles, noise_px, seed
    )
    src_p, order, max_targ = candidate_quads(
        frm, corrected, cals, cpar, vpar, corrmin_base
    )
    clean = score(dedup_clean(order, src_p, max_targ), src_p, corrected, pnr_to_pid, n)
    buggy = score(dedup_buggy(order, src_p, max_targ), src_p, corrected, pnr_to_pid, n)

    # Matched operating point: raise corrmin for CLEAN until it accepts <= what
    # buggy accepted, then compare precision there. This is the real test.
    tightened = None
    if buggy["accepted"] < clean["accepted"]:
        lo, hi = corrmin_base, corrmin_base
        for _ in range(60):  # find an upper bracket
            hi = hi * 1.5 + 1.0
            sp, od, mt = candidate_quads(frm, corrected, cals, cpar, vpar, hi)
            if len(dedup_clean(od, sp, mt)) <= buggy["accepted"]:
                break
        for _ in range(40):  # bisect to the matched N
            mid = 0.5 * (lo + hi)
            sp, od, mt = candidate_quads(frm, corrected, cals, cpar, vpar, mid)
            t = dedup_clean(od, sp, mt)
            if len(t) > buggy["accepted"]:
                lo = mid
            else:
                hi = mid
                tightened = score(t, sp, corrected, pnr_to_pid, n)
        if tightened is not None:
            tightened["corrmin"] = hi
    return clean, buggy, tightened, n


def main():
    corrmin_base = 33.0  # wp1's value (parameters_wp1.yaml criteria.corrmin)
    noise_px = 1.0

    print("=" * 100)
    print("VALIDATION GATES (must pass before any result below is trustworthy)")
    print("=" * 100)

    # Gate 1: at very low density there is no ambiguity -> both variants must
    # agree, and both must be near-perfect. Catches harness/scoring bugs.
    clean, buggy, _t, n = run_case(15, 0.3, 0, corrmin_base)
    g1 = (
        clean["accepted"] == buggy["accepted"]
        and clean["precision"] > 0.98
        and clean["recall"] > 0.9
    )
    print(
        f"gate 1 (sparse scene -> variants agree & near-perfect): "
        f"n_obs={n} clean acc={clean['accepted']} prec={clean['precision']:.3f} rec={clean['recall']:.3f} | "
        f"buggy acc={buggy['accepted']} prec={buggy['precision']:.3f} -> {'PASS' if g1 else 'FAIL'}"
    )

    # Gate 2: the quirk is provably strictly-more-restrictive. If buggy ever
    # accepts MORE than clean, the harness is wrong.
    g2 = True
    for np_, sd in ((60, 1), (150, 2), (300, 3)):
        c, b, _t, _n = run_case(np_, noise_px, sd, corrmin_base)
        if b["accepted"] > c["accepted"]:
            g2 = False
    print(f"gate 2 (buggy <= clean accepted, always): {'PASS' if g2 else 'FAIL'}")

    # Gate 3: ghost rate must RISE with density -- if it does not, the scene
    # is not actually producing the ambiguity this experiment is about.
    ghost_rates = []
    for np_ in (60, 200, 500):
        c, _b, _t, _n = run_case(np_, noise_px, 7, corrmin_base)
        ghost_rates.append(1.0 - c["precision"])
    g3 = ghost_rates[-1] > ghost_rates[0]
    print(
        f"gate 3 (ghost rate rises with density): "
        f"{[f'{g:.3f}' for g in ghost_rates]} -> {'PASS' if g3 else 'FAIL'}"
    )
    g1 = g1 and g3

    if not (g1 and g2):
        print(
            "\nGATES FAILED -- results below are NOT trustworthy. Fix the harness first."
        )
        return

    print()
    print("=" * 100)
    print("RESULT: buggy(3dptv quirk) vs clean-TIGHTENED-to-the-same-accept-count")
    print("(comparing raw clean vs buggy would be meaningless -- see module docstring)")
    print("=" * 100)
    hdr = (
        f"{'n_part':>7} {'n_obs':>6} {'seed':>5} | {'clean_acc':>9} {'clean_P':>8} {'clean_R':>8} "
        f"| {'buggy_acc':>9} {'buggy_P':>8} {'buggy_R':>8} "
        f"| {'tight_P':>8} {'tight_R':>8} | verdict"
    )
    print(hdr)
    print("-" * len(hdr))

    wins = {"quirk": 0, "tie": 0, "corrmin": 0}
    for n_part in (60, 150, 300, 500):
        for seed in (1, 2, 3):
            clean, buggy, tight, n = run_case(n_part, noise_px, seed, corrmin_base)
            if tight is None:
                print(
                    f"{n_part:7d} {n:6d} {seed:5d} |  (buggy did not reduce count; no matched point)"
                )
                continue
            d = buggy["precision"] - tight["precision"]
            if abs(d) < 0.005:
                verdict, key = "tie", "tie"
            elif d > 0:
                verdict, key = f"QUIRK +{d:.3f}", "quirk"
            else:
                verdict, key = f"corrmin +{-d:.3f}", "corrmin"
            wins[key] += 1
            print(
                f"{n_part:7d} {n:6d} {seed:5d} "
                f"| {clean['accepted']:9d} {clean['precision']:8.3f} {clean['recall']:8.3f} "
                f"| {buggy['accepted']:9d} {buggy['precision']:8.3f} {buggy['recall']:8.3f} "
                f"| {tight['precision']:8.3f} {tight['recall']:8.3f} | {verdict}"
            )

    print()
    print(
        f"tally at matched operating point -> quirk better: {wins['quirk']}, "
        f"tie: {wins['tie']}, plain-corrmin better: {wins['corrmin']}"
    )


if __name__ == "__main__":
    main()

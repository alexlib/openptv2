"""Why did openptv2 drop this link? Replay the candidate shortlist per link.

Takes the links a reference run (legacy 3dptv) made and the run under test did
NOT, and replays the frame-2 candidate stage for each one using the shipped
reference implementations (searchquader + candsearch_in_pix from
openptv2.algorithms.track). For every dropped link it asks, in the order the
tracker asks:

  OUT_OF_BOX    the true partner's target is not inside the projected search
                window in ANY camera -> search volume / prediction problem
  RANK>4        it IS in the window but more than 4 targets sit closer to the
                predicted centre, so candsearch_in_pix evicts it before
                anything scores it -> shortlist depth problem
  FREQ<2        it survives the top-4 in fewer than 2 cameras, so
                sortwhatfound culls it (ttools.c:335) -> multi-camera problem
  SURVIVED      it reached the scoring stage, so the loss is downstream
                (two-hop lookahead, acc/angle gate, or conflict resolution)

Run from repo root:
    uv run python scripts/diag_candidate_autopsy.py \
        --ref <folder>/res_ground_truth_backup \
        --test <folder>/res_retry0 \
        --data <folder>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, TrackPar
from openptv2.algorithms.track import (
    angle_acc,
    candsearch_in_pix,
    point_to_pixel,
    searchquader,
)
from openptv2.algorithms.tracking_frame_buf import read_targets

MAX_SHORTLIST = 4


def read_ptv_is(path: Path):
    lines = path.read_text().strip().splitlines()
    n = int(lines[0])
    prev = np.empty(n, dtype=int)
    for i, line in enumerate(lines[1 : n + 1]):
        prev[i] = int(line.split()[0])
    return prev


def read_rt_is(path: Path):
    """Return (xyz, corres) — corres[i, cam] is the target index, -1 if none."""
    lines = path.read_text().strip().splitlines()
    n = int(lines[0])
    xyz = np.empty((n, 3))
    cor = np.full((n, 4), -1, dtype=int)
    for i, line in enumerate(lines[1 : n + 1]):
        p = line.split()
        xyz[i] = [float(p[1]), float(p[2]), float(p[3])]
        for c in range(4):
            cor[i, c] = int(p[4 + c])
    return xyz, cor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--first", type=int, default=100002)
    ap.add_argument("--last", type=int, default=100010)
    ap.add_argument("--targets", default="img_3dptv/Cam{cam}.")
    ap.add_argument("--limit", type=int, default=0, help="stop after N drops")
    ap.add_argument("--pred-from-test", action="store_true",
                    help="predict using the RUN-UNDER-TEST link history "
                         "(what the tracker actually had) instead of the "
                         "reference's -- isolates cascade effects")
    args = ap.parse_args()

    cpar = ControlPar.from_file(str(args.data / "parameters" / "ptv.par"))
    tpar = TrackPar.from_file(str(args.data / "parameters" / "track.par"))
    ncam = cpar.num_cams
    cals = []
    for c in range(1, ncam + 1):
        ori = add = None
        for stem in (f"cam_{c}.tif", f"cam{c}.tif", f"Cam{c}.tif", f"cam_{c}"):
            p = args.data / "cal" / f"{stem}.ori"
            if p.exists():
                ori = str(p)
                a = args.data / "cal" / f"{stem}.addpar"
                add = str(a) if a.exists() else None
                break
        if ori is None:
            raise SystemExit(f"no .ori found for camera {c} in {args.data / 'cal'}")
        cals.append(Calibration.from_file(ori, add))

    verdict = {"OUT_OF_BOX": 0, "RANK>4": 0, "FREQ<2": 0, "SURVIVED": 0}
    rank_hist = []
    step_of = {k: [] for k in verdict}
    n_drop = 0
    stage2 = {
        "checked": 0, "no_true_succ": 0, "DISPL_GATE": 0,
        "HOP2_NOT_FOUND": 0, "ACC_ANGLE_GATE": 0, "GATE_PASSED": 0,
    }
    gate_fail, hop2_step, gate_ok_step = [], [], []

    for f in range(args.first, args.last + 1):
        ref_prev = read_ptv_is(args.ref / f"ptv_is.{f}")
        test_prev = read_ptv_is(args.test / f"ptv_is.{f}")
        cur_xyz, cur_cor = read_rt_is(args.ref / f"rt_is.{f}")
        old_xyz, old_cor = read_rt_is(args.ref / f"rt_is.{f - 1}")
        otargs = [
            read_targets(str(args.data / args.targets.format(cam=c + 1)), f - 1)
            for c in range(ncam)
        ]
        # frame f-2, for the constant-velocity prediction
        older_xyz = None
        if f - 2 >= args.first - 1:
            p2 = args.ref / f"rt_is.{f - 2}"
            if p2.exists():
                older_xyz, _ = read_rt_is(p2)
                src = args.test if args.pred_from_test else args.ref
                oprev = read_ptv_is(src / f"ptv_is.{f - 1}")

        targs = []
        for c in range(ncam):
            base = str(args.data / args.targets.format(cam=c + 1))
            targs.append(read_targets(base, f))

        # frame f+1, for the two-hop lookahead stage
        nxt_xyz = nxt_cor = ntargs = None
        next_of = {}
        pn = args.ref / f"rt_is.{f + 1}"
        if pn.exists():
            nxt_xyz, nxt_cor = read_rt_is(pn)
            nprev = read_ptv_is(args.ref / f"ptv_is.{f + 1}")
            for j, pj in enumerate(nprev):
                if pj >= 0:
                    next_of[pj] = j
            ntargs = [
                read_targets(str(args.data / args.targets.format(cam=c + 1)), f + 1)
                for c in range(ncam)
            ]

        for i in range(len(ref_prev)):
            p_ref = ref_prev[i]
            p_test = test_prev[i] if i < len(test_prev) else -2
            if p_ref < 0 or p_test == p_ref:
                continue  # no reference link, or we matched it
            n_drop += 1
            if args.limit and n_drop > args.limit:
                break

            X1 = old_xyz[p_ref]
            X2 = X1
            if older_xyz is not None and p_ref < len(oprev) and oprev[p_ref] >= 0:
                X0 = older_xyz[oprev[p_ref]]
                X2 = 2 * X1 - X0

            step = float(np.linalg.norm(cur_xyz[i] - X1))
            xr, xl, yd, yu = searchquader(X2, tpar, cpar, cals)

            in_box = 0
            in_top4 = 0
            best_rank = None
            for c in range(ncam):
                tidx = cur_cor[i, c]
                if tidx < 0 or tidx >= len(targs[c]):
                    continue
                # The tracker centres the pixel search on the particle's own
                # MEASURED target (track.c:136), and only reprojects for a
                # camera with no correspondence. Using point_to_pixel here
                # instead silently shifts the window by the reprojection
                # residual -- which is what made an earlier version of this
                # script disagree with the tracker.
                oidx = old_cor[p_ref, c] if p_ref < len(old_cor) else -1
                if 0 <= oidx < len(otargs[c]):
                    cx, cy = otargs[c][oidx].x, otargs[c][oidx].y
                else:
                    cx, cy = point_to_pixel(X2, cals[c], cpar)
                tx = targs[c][tidx].x
                ty = targs[c][tidx].y
                if not (
                    cx - xl[c] < tx < cx + xr[c] and cy - yu[c] < ty < cy + yd[c]
                ):
                    continue
                in_box += 1
                # Rank by distance among every target inside the same window.
                d_true = np.hypot(cx - tx, cy - ty)
                closer = 0
                for k in range(len(targs[c])):
                    if targs[c][k].tnr == -1:
                        continue
                    kx, ky = targs[c][k].x, targs[c][k].y
                    if not (
                        cx - xl[c] < kx < cx + xr[c]
                        and cy - yu[c] < ky < cy + yd[c]
                    ):
                        continue
                    if np.hypot(cx - kx, cy - ky) < d_true:
                        closer += 1
                rank = closer + 1
                best_rank = rank if best_rank is None else min(best_rank, rank)
                # Confirm against the shipped shortlist implementation.
                p = candsearch_in_pix(
                    targs[c], len(targs[c]), cx, cy,
                    xl[c], xr[c], yu[c], yd[c], cpar,
                )
                if tidx in p:
                    in_top4 += 1

            if in_box == 0:
                v = "OUT_OF_BOX"
            elif in_top4 == 0:
                v = "RANK>4"
            elif in_top4 < 2:
                v = "FREQ<2"
            else:
                v = "SURVIVED"
            verdict[v] += 1
            step_of[v].append(step)
            if best_rank is not None:
                rank_hist.append(best_rank)

            # ---- stage 2: the two-hop lookahead (track.c lines 204-337) ----
            # The frame-2 candidate is only registered if a valid frame-3
            # continuation is also found. A perfect frame-2 candidate still
            # yields NO LINK when the frame-3 search comes up empty.
            if v != "SURVIVED" or nxt_xyz is None:
                continue
            stage2["checked"] += 1
            X3 = cur_xyz[i]
            X0v = None
            if older_xyz is not None and p_ref < len(oprev) and oprev[p_ref] >= 0:
                X0v = older_xyz[oprev[p_ref]]
            X5 = 2 * X3 - X1 if X0v is None else 0.5 * (5 * X3 - 4 * X1 + X0v)

            succ = next_of.get(i, -1)
            if succ < 0:
                stage2["no_true_succ"] += 1
                continue
            X4 = nxt_xyz[succ]

            d = X4 - X3
            if not (
                tpar.dvxmin < d[0] < tpar.dvxmax
                and tpar.dvymin < d[1] < tpar.dvymax
                and tpar.dvzmin < d[2] < tpar.dvzmax
            ):
                stage2["DISPL_GATE"] += 1
                continue

            xr5, xl5, yd5, yu5 = searchquader(X5, tpar, cpar, cals)
            hits = 0
            for c in range(ncam):
                tidx = nxt_cor[succ, c]
                if tidx < 0 or tidx >= len(ntargs[c]):
                    continue
                cx, cy = point_to_pixel(X5, cals[c], cpar)
                p = candsearch_in_pix(
                    ntargs[c], len(ntargs[c]), cx, cy,
                    xl5[c], xr5[c], yu5[c], yd5[c], cpar,
                )
                if tidx in p:
                    hits += 1
            if hits < 2:
                stage2["HOP2_NOT_FOUND"] += 1
                hop2_step.append(step)
                continue

            a1, ac1 = angle_acc(X3, X4, X5)
            if X0v is not None:
                a0, ac0 = angle_acc(X1, X2, X3)
            else:
                a0, ac0 = a1, ac1
            acc = (ac0 + ac1) / 2
            ang = (a0 + a1) / 2
            if (acc < tpar.dacc and ang < tpar.dangle) or acc < tpar.dacc / 10:
                stage2["GATE_PASSED"] += 1
                gate_ok_step.append(step)
            else:
                stage2["ACC_ANGLE_GATE"] += 1
                gate_fail.append((acc, ang, step))
        if args.limit and n_drop > args.limit:
            break

    total = sum(verdict.values())
    print(f"\ndropped links analysed: {total}\n")
    print(f"{'verdict':<14}{'count':>8}{'share':>9}{'mean step':>12}{'max step':>11}")
    for k in ("OUT_OF_BOX", "RANK>4", "FREQ<2", "SURVIVED"):
        s = step_of[k]
        share = verdict[k] / total * 100 if total else 0
        ms = np.mean(s) if s else 0.0
        xs = np.max(s) if s else 0.0
        print(f"{k:<14}{verdict[k]:>8}{share:>8.1f}%{ms:>12.4f}{xs:>11.4f}")

    if rank_hist:
        r = np.array(rank_hist)
        print(f"\nbest-camera distance rank of the true partner "
              f"(shortlist keeps {MAX_SHORTLIST}):")
        for k in range(1, 9):
            n = int((r == k).sum())
            if n:
                print(f"  rank {k:<2} {n:>6}  {'#' * min(60, n * 60 // len(r))}")
        n = int((r > 8).sum())
        if n:
            print(f"  rank >8 {n:>6}")
        print(f"  median rank {np.median(r):.1f}   "
              f"fraction beyond top-4: {(r > MAX_SHORTLIST).mean():.1%}")

    if stage2["checked"]:
        print(f"\n--- stage 2: two-hop lookahead, for the {stage2['checked']} "
              f"drops whose frame-2 candidate SURVIVED ---")
        order = ["no_true_succ", "DISPL_GATE", "HOP2_NOT_FOUND",
                 "ACC_ANGLE_GATE", "GATE_PASSED"]
        for k in order:
            share = stage2[k] / stage2["checked"] * 100
            print(f"  {k:<18}{stage2[k]:>7}{share:>8.1f}%")
        if hop2_step:
            print(f"\n  HOP2_NOT_FOUND mean step {np.mean(hop2_step):.4f} mm  "
                  f"max {np.max(hop2_step):.4f} mm")
        if gate_fail:
            a = np.array([g[0] for g in gate_fail])
            g = np.array([g[1] for g in gate_fail])
            print(f"  ACC_ANGLE_GATE  acc median {np.median(a):.3f} "
                  f"(dacc={tpar.dacc})   angle median {np.median(g):.1f} "
                  f"(dangle={tpar.dangle})")
        if gate_ok_step:
            print(f"\n  GATE_PASSED mean step {np.mean(gate_ok_step):.4f} mm "
                  f"-- these reached scoring, so they were lost in link "
                  f"resolution (lost a contest).")


if __name__ == "__main__":
    main()

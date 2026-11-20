"""Births recall: for every view of a missed true point that is hidden in a blob used by tracked points, at which
stage is it lost? (docs/plans/2026-11-20-holistic-ptv.md, H3)

    uv run python scripts/joint_birth_recall.py <case> [--min-resid 0.25]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_bounds as jb  # noqa: E402
import synth_bench as sb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--min-resid", type=float, default=0.25)
    ap.add_argument("--sig-quantile", type=float, default=0.5)
    a_ = ap.parse_args()
    from openptv2.loco import project_pixels
    from openptv2.merge_post import arrays_from_store, extend_membership, row_signatures
    from openptv2.merge_refine import refine_merged
    from openptv2.storage import RunStore

    case_dir, info, exp, _s, tr = jb.case_setup(a_.case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_out = info["n_out"]
    st = RunStore(sb.WORK / "runs" / a_.case / "two_phase" / "res" / "run.zarr", mode="r")
    a = arrays_from_store(st, "ptv_is", 1, n_out, 4)
    blob, centre, sumg, _ = extend_membership(a["frame"], a["pos"], a["blob"], a["centre"], a["sumg"], a["blobs"],
                                              cpar, cals)
    res = refine_merged(a["frame"], a["tid"], blob, centre, sumg, cpar, cals, vpar)
    sig = row_signatures(a["tid"], a["frame"], blob, sumg, a_.sig_quantile)
    fb = np.full(res.pred_px.shape, np.nan)
    okx = np.isfinite(res.x[:, 0])
    for c in range(4):
        m = okx & (blob[:, c] >= 0)
        fb[m, c] = project_pixels(res.x[m], cals[c], cpar)
    pred = np.where(np.isfinite(res.pred_px), res.pred_px, fb)
    users = {}
    for c in range(4):
        for r in np.flatnonzero(blob[:, c] >= 0):
            users.setdefault((int(a["frame"][r]), c, int(blob[r, c])), []).append(r)
    med = [np.nanmedian(sig[:, c]) for c in range(4)]
    # missed eligible truth points (not found by the two_phase output)
    pred_tp = dict(np.load(sb.WORK / "runs" / a_.case / "two_phase" / "pred.npz"))
    ev = sb.evaluate_pred(case_dir, pred_tp, sb.Flow())
    k_all = tr["particle_id"].astype(np.int64) * 100000 + tr["frame"]
    tb = tr["blob_tid"]
    miss = np.flatnonzero(((tb >= 0).sum(1) >= 3) & ~np.isin(k_all, ev["_keys"]))
    render = list(sb._experiment(case_dir, case_dir / "parameters_Run1.yaml", 1, n_out).cals)
    sb._render_cals(render, cpar)
    why = Counter()
    ratios = []
    per_point_ok = np.zeros(len(miss), int)
    per_point_free = np.zeros(len(miss), int)
    for k, i in enumerate(miss):
        f = int(tr["frame"][i])
        for c in range(4):
            b = int(tb[i, c])
            if b < 0:
                continue
            mem = users.get((f, c, b))
            if mem is None:
                per_point_free[k] += 1
                continue
            s_m = sig[mem, c]
            p_m = pred[mem, c]
            if not np.all(np.isfinite(s_m)):
                why["member signature unknown"] += 1
                continue
            if not np.all(np.isfinite(p_m)):
                why["member position unknown"] += 1
                continue
            cent, sg = a["blobs"][(f, c)]
            resid = sg[b] - s_m.sum()
            ratios.append(resid / med[c])
            if resid < a_.min_resid * med[c]:
                why["residual brightness below threshold"] += 1
                continue
            s = (sg[b] * cent[b] - (s_m[:, None] * p_m).sum(0)) / resid
            if np.linalg.norm(s - cent[b]) > 3.0:
                why["residual target too far from the blob (> 3 px)"] += 1
                continue
            true_px = sb._project(tr["pos_mm"][i][None], render[c], cpar)[0]
            err = np.linalg.norm(s - true_px)
            if err > 1.0:
                why["target placed > 1 px from the true spot"] += 1
                continue
            why["good residual target (<= 1 px)"] += 1
            per_point_ok[k] += 1
    tot = sum(why.values())
    print("missed eligible points: %d; hidden views: %d" % (len(miss), tot))
    for kk, v in why.most_common():
        print("  %-48s %6d  %.3f" % (kk, v, v / tot))
    rr = np.array(ratios)
    print("residual / median signature at hidden views: quantiles 10/25/50/75/90 %s" % np.round(np.quantile(rr, [.1, .25, .5, .75, .9]), 2))
    truth_b = []
    usable = (per_point_ok + per_point_free) >= 3
    print("missed points with >= 3 usable views (good residual targets + free blobs): %.3f" % usable.mean())
    print("  of those with >= 1 good residual target: %.3f" % (usable & (per_point_ok >= 1)).mean())


if __name__ == "__main__":
    main()

"""Where does H2 lose against perfect association? (docs/plans/2026-11-20-holistic-ptv.md)

Maps every row of a tracker run (run store kept by ``joint_bounds.py h2``) to the nearest truth particle of its
frame (<= 0.4 mm after removing the global bias), then:
  1. membership accuracy per camera: inferred blob (after ``extend_membership``) vs the truth particle's blob;
  2. oracle ablation: the same refinement with the truth blobs for the matched rows (links unchanged), written as
     ``runs/<case>/<tracker>+h2truthmem`` for ``synth_bench eval --ref <tracker>+h2base``.

    uv run python scripts/joint_h2_diag.py <case> [--tracker two_phase]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_bounds as jb  # noqa: E402
import synth_bench as sb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--tracker", default="two_phase")
    ap.add_argument("--radius", type=float, default=2.0)
    a_ = ap.parse_args()
    from openptv2.merge_post import arrays_from_store, extend_membership
    from openptv2.merge_refine import refine_merged, shared_blobs
    from openptv2.storage import RunStore

    case_dir, info, exp, _s, tr = jb.case_setup(a_.case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams, n_out = len(cals), info["n_out"]
    st = RunStore(sb.WORK / "runs" / a_.case / a_.tracker / "res" / "run.zarr", mode="r")
    a = arrays_from_store(st, "ptv_is", 1, n_out, n_cams)
    blob, centre, sumg, n_att = extend_membership(a["frame"], a["pos"], a["blob"], a["centre"], a["sumg"],
                                                  a["blobs"], cpar, cals, radius=a_.radius)
    # rows -> truth
    tfr, tpos, tblob = tr["frame"], tr["pos_mm"], tr["blob_tid"]
    match = np.full(len(a["frame"]), -1)
    d0 = []
    for f in range(1, n_out + 1):
        r = np.flatnonzero(a["frame"] == f)
        t = np.flatnonzero(tfr == f)
        if len(r) == 0 or len(t) == 0:
            continue
        d, j = cKDTree(tpos[t]).query(a["pos"][r])
        d0.append(a["pos"][r] - tpos[t][j])
    bias = np.median(np.vstack(d0), axis=0)
    for f in range(1, n_out + 1):
        r = np.flatnonzero(a["frame"] == f)
        t = np.flatnonzero(tfr == f)
        if len(r) == 0 or len(t) == 0:
            continue
        d, j = cKDTree(tpos[t]).query(a["pos"][r] - bias)
        ok = d < 0.4
        match[r[ok]] = t[j[ok]]
    m = match >= 0
    out = {"rows": int(len(m)), "matched_to_truth": float(m.mean())}
    tb = np.where(m[:, None], tblob[np.maximum(match, 0)], -2)       # truth blob of the matched particle
    share_true = np.zeros_like(blob)
    for f in range(1, n_out + 1):
        t = np.flatnonzero(tfr == f)
        share_true[np.flatnonzero(a["frame"] == f)] = 0
        tb_f = tblob[t]
        for c in range(n_cams):
            u, cnt = np.unique(tb_f[:, c][tb_f[:, c] >= 0], return_counts=True)
            rows = np.flatnonzero((a["frame"] == f) & m)
            tb_rows = tblob[match[rows], c]
            share_true[rows, c] = np.where(tb_rows >= 0, cnt[np.searchsorted(u, np.maximum(tb_rows, 0))] * 1, 0)
    for name, bb in (("matcher", a["blob"]), ("after_extension", blob)):
        cats = {}
        mm = m[:, None] & np.ones(n_cams, bool)
        truth_has = (tb >= 0) & mm
        cats["truth_detected_obs"] = int(truth_has.sum())
        cats["correct_blob"] = float(((bb == tb) & truth_has).sum() / truth_has.sum())
        cats["wrong_blob"] = float(((bb >= 0) & (bb != tb) & truth_has).sum() / truth_has.sum())
        cats["missing_view"] = float(((bb < 0) & truth_has).sum() / truth_has.sum())
        cats["view_where_truth_has_none"] = int(((bb >= 0) & (tb < 0) & mm).sum())
        ts = truth_has & (share_true >= 2)
        cats["truth_shared_obs"] = int(ts.sum())
        cats["truth_shared_obs_correct_blob"] = float(((bb == tb) & ts).sum() / max(ts.sum(), 1))
        inf_share = shared_blobs(a["frame"], bb)
        cats["inferred_shared_obs"] = int(((inf_share >= 2) & mm).sum())
        cats["inferred_shared_that_are_truly_shared"] = float(((inf_share >= 2) & ts).sum() / max(((inf_share >= 2) & mm).sum(), 1))
        out[name] = cats
    out["attachments"] = int(n_att)
    # oracle ablation: truth blobs for matched rows
    blob_t = blob.copy()
    centre_t, sumg_t = centre.copy(), sumg.copy()
    for c in range(n_cams):
        rows = np.flatnonzero(m & (tb[:, c] >= 0))
        blob_t[rows, c] = tb[rows, c]
        rows_none = np.flatnonzero(m & (tb[:, c] < 0))
        blob_t[rows_none, c] = -1
        centre_t[rows_none, c] = np.nan
        sumg_t[rows_none, c] = np.nan
        for f in np.unique(a["frame"][rows]):
            rr = rows[a["frame"][rows] == f]
            cent, sg = a["blobs"][(int(f), c)]
            centre_t[rr, c] = cent[blob_t[rr, c]]
            sumg_t[rr, c] = sg[blob_t[rr, c]]
    res = refine_merged(a["frame"], a["tid"], blob_t, centre_t, sumg_t, cpar, cals, vpar)
    new_pos = np.where(np.isfinite(res.x[:, :1]), res.x, a["pos"])
    keep = np.bincount(a["tid"])[a["tid"]] >= 2
    d = sb.WORK / "runs" / a_.case / f"{a_.tracker}+h2truthmem"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "pred.npz", trajid=a["tid"][keep], frame=a["frame"][keep], pos=new_pos[keep])
    (d / "run.json").write_text(json.dumps({"tracker": d.name, "case": a_.case, "time_s": 0.0, "status": "ok"}))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

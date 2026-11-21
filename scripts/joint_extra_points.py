"""Why are the points a variant adds (not in the reference output) noisier? Position error vs truth, velocity of their
tracks, track lengths: common vs extra points. (docs/plans/2026-11-20-holistic-ptv.md)

    uv run python scripts/joint_extra_points.py <case> <label> <ref case> <ref label>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_bench as sb  # noqa: E402


def main() -> None:
    case, label, rcase, rlabel = sys.argv[1:5]
    cd = sb.WORK / "cases" / case
    tr = np.load(cd / "truth.npz")
    by_frame = {}
    from scipy.spatial import cKDTree
    for f in np.unique(tr["frame"]):
        m = tr["frame"] == f
        by_frame[int(f)] = (tr["pos_mm"][m], tr["particle_id"][m], cKDTree(tr["pos_mm"][m]))
    ref = dict(np.load(sb.WORK / "runs" / rcase / rlabel / "pred.npz"))
    p = dict(np.load(sb.WORK / "runs" / case / label / "pred.npz"))
    out = {}
    for nm, q in (("ref", ref), ("var", p)):
        g0, _, off = sb._match(by_frame, q["frame"], q["pos"], 0.6)
        bias = np.median(off[g0 >= 0], axis=0)
        gid, d, _ = sb._match(by_frame, q["frame"], q["pos"] - bias, 0.4)
        out[nm] = (gid, d, q)
    rg = out["ref"][0]
    rkeys = rg[rg >= 0].astype(np.int64) * 100000 + out["ref"][2]["frame"][rg >= 0]
    gid, d, q = out["var"]
    key = gid.astype(np.int64) * 100000 + q["frame"]
    m = gid >= 0
    common = m & np.isin(key, rkeys)
    extra = m & ~np.isin(key, rkeys)
    L = np.bincount(np.unique(q["trajid"], return_inverse=True)[1])[np.unique(q["trajid"], return_inverse=True)[1]]
    print("matched points: common %d, extra %d, unmatched (ghost) %d" % (common.sum(), extra.sum(), (~m).sum()))
    for nm, s in (("common", common), ("extra", extra)):
        print("  %-6s position error rms %.3f mm median %.3f | track length median %d, share in tracks < 5: %.2f, >= 20: %.2f"
              % (nm, np.sqrt(np.mean(d[s] ** 2)), np.median(d[s]), np.median(L[s]), np.mean(L[s] < 5), np.mean(L[s] >= 20)))
    # extras inside the same tracks as common points vs tracks made only of extras
    tid = q["trajid"]
    has_common = np.isin(tid, np.unique(tid[common]))
    print("  extra points on tracks that also hold common points: %.2f" % np.mean(has_common[extra]))


if __name__ == "__main__":
    main()

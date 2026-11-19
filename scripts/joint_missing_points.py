"""Which true points does a tracker output miss, and was the particle found at neighbouring frames?
(docs/plans/2026-11-20-holistic-ptv.md, H3). uv run python scripts/joint_missing_points.py <case> <run label>"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
import synth_bench as sb

case, label = sys.argv[1], sys.argv[2]
cd = sb.WORK / "cases" / case
tr = np.load(cd / "truth.npz")
pred = dict(np.load(sb.WORK / "runs" / case / label / "pred.npz"))
flow = sb.Flow()
r = sb.evaluate_pred(cd, pred, flow)
keys = set(r["_keys"].tolist())
pid, fr, ndet = tr["particle_id"], tr["frame"], (tr["blob_tid"] >= 0).sum(1)
k_all = pid.astype(np.int64) * 100000 + fr
elig = ndet >= 3
found = np.isin(k_all, r["_keys"])
print("eligible truth points (>=3 cameras detected): %d, found by the tracker output: %.3f" % (elig.sum(), found[elig].mean()))
miss = elig & ~found
mk = k_all[miss]
for g in (1, 2, 3, 6):
    near = np.zeros(len(mk), bool)
    for d in range(-g, g + 1):
        if d:
            near |= np.isin(mk + d, r["_keys"])
    both = np.zeros(len(mk), bool)
    before = np.zeros(len(mk), bool)
    after = np.zeros(len(mk), bool)
    for d in range(1, g + 1):
        before |= np.isin(mk - d, r["_keys"])
        after |= np.isin(mk + d, r["_keys"])
    print("  missed points whose particle is found within +-%d frames: %.3f (both sides: %.3f)" % (g, near.mean(), (before & after).mean()))
pids_found = np.unique(pid[found])
print("  missed points of particles never found at all: %.3f" % (~np.isin(pid[miss], pids_found)).mean())
print("  share of eligible points that are missed: %.3f; of those the particle is found elsewhere: %.3f" % (miss.sum()/elig.sum(), np.isin(pid[miss], pids_found).mean()))

"""For the true points a tracker output misses: in how many cameras is their blob free, or hidden in a blob a
tracked point uses? (bound for births).  uv run python scripts/joint_birth_bound.py <case>
"""

import sys

import numpy as np

sys.path.insert(0, "scripts")
import joint_bounds as jb
import synth_bench as sb

from openptv2.merge_post import arrays_from_store, extend_membership
from openptv2.storage import RunStore

case = sys.argv[1]
cd, info, exp, _s, tr = jb.case_setup(case)
cpar, cals = exp.cpar, list(exp.cals)
n_out = info["n_out"]
st = RunStore(sb.WORK / "runs" / case / "two_phase" / "res" / "run.zarr", mode="r")
a = arrays_from_store(st, "ptv_is", 1, n_out, 4)
blob, *_ = extend_membership(
    a["frame"], a["pos"], a["blob"], a["centre"], a["sumg"], a["blobs"], cpar, cals
)
used = set()
for c in range(4):
    r = np.flatnonzero(blob[:, c] >= 0)
    used |= set(zip(a["frame"][r].tolist(), [c] * len(r), blob[r, c].tolist()))
pred = dict(np.load(sb.WORK / "runs" / case / "two_phase+h2base" / "pred.npz"))
res = sb.evaluate_pred(cd, pred, sb.Flow())
k_all = tr["particle_id"].astype(np.int64) * 100000 + tr["frame"]
tb = tr["blob_tid"]
elig = (tb >= 0).sum(1) >= 3
miss = elig & ~np.isin(k_all, res["_keys"])
free = np.zeros(len(tb), int)
hidden = np.zeros(len(tb), int)
for i in np.flatnonzero(miss):
    f = int(tr["frame"][i])
    for c in range(4):
        if tb[i, c] < 0:
            continue
        if (f, c, int(tb[i, c])) in used:
            hidden[i] += 1
        else:
            free[i] += 1
mi = np.flatnonzero(miss)
print("missed eligible points:", len(mi), "of", elig.sum())
for k in range(5):
    print(
        "  views free=%d: %.3f" % (k, np.mean(free[mi] == k)),
        "  hidden=%d: %.3f" % (k, np.mean(hidden[mi] == k)),
    )
tot = free[mi] + hidden[mi]
print(
    "  free+hidden >=3: %.3f ; >=2: %.3f ; free>=2 & hidden>=1: %.3f ; free>=3: %.3f"
    % (
        np.mean(tot >= 3),
        np.mean(tot >= 2),
        np.mean((free[mi] >= 2) & (hidden[mi] >= 1)),
        np.mean(free[mi] >= 3),
    )
)

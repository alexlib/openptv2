# ruff: noqa
"""Quality of the points a relaxed second pass adds, against truth: uv run python bench/limits/new_points_quality.py <base dir> <relaxed dir> <n_frames>"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb
from openptv2.merge_post import arrays_from_store
from openptv2.storage import RunStore
base, rel, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sb0 = RunStore(str(base / "res" / "run.zarr"), mode="r")
n_base = {f: len(sb0.read_correspondences(f)[0]) for f in range(1, N + 1)}
st = RunStore(str(rel / "res" / "run.zarr"), mode="r")
a = arrays_from_store(st, "ptv_is", 1, N, 4)
tr = np.load(sb.SRC / "truth" / "truth.npz"); k = tr["frame"] <= N
by = {int(f): (tr["pos_mm"][k][tr["frame"][k] == f], tr["particle_id"][k][tr["frame"][k] == f], cKDTree(tr["pos_mm"][k][tr["frame"][k] == f])) for f in np.unique(tr["frame"][k])}
g0, _, off = sb._match(by, a["frame"], a["pos"], 0.6); bias = np.median(off[g0 >= 0], axis=0)
gid, dist, off = sb._match(by, a["frame"], a["pos"] - bias, 0.4)
new = a["row"] >= np.array([n_base[int(f)] for f in a["frame"]])
ncam = (a["blob"] >= 0).sum(1)
from collections import Counter
ln = Counter(a["tid"]); plen = np.array([ln[t] for t in a["tid"]])
print(f"points {len(new)}, new {new.sum()} ({100 * new.mean():.1f}%)")
for nm, m in (("ordinary", ~new), ("new", new), ("new, 3 cameras", new & (ncam == 3)), ("new, 4 cameras", new & (ncam == 4))):
    ok = m & (gid >= 0); e = np.linalg.norm(off[ok], axis=1)
    print(f"  {nm:16s} n={m.sum():6d}  ghost {100 * (gid[m] < 0).mean():4.1f}%  position error median {np.median(e):.3f} p90 {np.percentile(e, 90):.3f} mm  in tracks of >=5 points {100 * (plen[m] >= 5).mean():.0f}%  singletons {100 * (plen[m] == 1).mean():.0f}%")

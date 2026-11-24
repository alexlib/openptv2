# ruff: noqa
"""Which id switches (both frames found, one frame apart, different track) involve the points a relaxed pass adds?

    uv run python bench/limits/switch_diag.py <base dir> <other dir> <n_frames>
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb
from openptv2.merge_post import arrays_from_store
from openptv2.storage import RunStore
base, other, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sb0 = RunStore(str(base / "res" / "run.zarr"), mode="r")
n_base = {f: len(sb0.read_correspondences(f)[0]) for f in range(1, N + 1)}
tr = np.load(sb.SRC / "truth" / "truth.npz"); k = tr["frame"] <= N
by = {int(f): (tr["pos_mm"][k][tr["frame"][k] == f], tr["particle_id"][k][tr["frame"][k] == f], cKDTree(tr["pos_mm"][k][tr["frame"][k] == f])) for f in np.unique(tr["frame"][k])}
tpos = {(int(i), int(f)): p for i, f, p in zip(tr["particle_id"][k], tr["frame"][k], tr["pos_mm"][k])}

def report(d, label):
    st = RunStore(str(d / "res" / "run.zarr"), mode="r")
    a = arrays_from_store(st, "ptv_is", 1, N, 4)
    g0, _, off = sb._match(by, a["frame"], a["pos"], 0.6); bias = np.median(off[g0 >= 0], axis=0)
    gid, _, _ = sb._match(by, a["frame"], a["pos"] - bias, 0.4)
    new = a["row"] >= np.array([n_base[int(f)] for f in a["frame"]])
    ncam = (a["blob"] >= 0).sum(1)
    pt = {}
    for i in np.flatnonzero(gid >= 0): pt[int(gid[i]), int(a["frame"][i])] = i
    cat = Counter(); linked_ok = Counter(); step_err = defaultdict(list)
    for (p, f), i in pt.items():
        j = pt.get((p, f + 1))
        if j is None: continue
        c = ("new" if new[i] else "old") + "->" + ("new" if new[j] else "old")
        c += f" ({min(ncam[i], ncam[j])}c)" if min(ncam[i], ncam[j]) < 4 else " (4c)"
        same = a["tid"][i] == a["tid"][j]
        (linked_ok if same else cat)[c] += 1
    tot = sum(cat.values()); ok = sum(linked_ok.values())
    print(f"[{label}] pairs of consecutive found frames {tot + ok}, id switches {tot} ({100 * tot / (tot + ok):.1f}%)")
    for c in sorted(set(cat) | set(linked_ok)):
        n = cat[c] + linked_ok[c]
        print(f"     {c:22s} pairs {n:6d}  switches {cat[c]:5d}  = {100 * cat[c] / max(n, 1):5.1f}% of its pairs")
report(base, "base"); report(other, "relaxed")

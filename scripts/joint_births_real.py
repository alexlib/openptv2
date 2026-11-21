"""Truth-free statistics of the births of a holistic run, and (if a benchmark case is given) their truth precision,
so the real-data numbers can be read against a case with known truth. (docs/plans/2026-11-20-holistic-ptv.md)

    uv run python scripts/joint_births_real.py <final run.zarr> <source run.zarr> <first> <last> [--case CASE]

Births = rows of the final store beyond the source store's rows in each frame (the loop appends them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("final")
    ap.add_argument("source")
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--case", default=None)
    a = ap.parse_args()
    from openptv2.merge_post import arrays_from_store
    from openptv2.storage import RunStore

    fin = RunStore(a.final, mode="r")
    src = RunStore(a.source, mode="r")
    arr = arrays_from_store(fin, "ptv_is", a.first, a.last, 4)
    is_birth = np.zeros(len(arr["frame"]), bool)
    for f in range(a.first, a.last + 1):
        n0 = len(src.read_correspondences(f)[0])
        rows = np.flatnonzero(arr["frame"] == f)
        is_birth[rows[n0:]] = True
    tid = arr["tid"]
    _u, inv, cnt = np.unique(tid, return_inverse=True, return_counts=True)
    L = cnt[inv]
    has_base = np.zeros(len(_u), bool)
    np.logical_or.at(has_base, inv, ~is_birth)
    out = {"births": int(is_birth.sum()), "birth_share_of_points": float(is_birth.mean())}
    for nm, m in (("births", is_birth), ("established", ~is_birth)):
        out[nm] = {"linked (track >= 2)": float(np.mean(L[m] >= 2)), "track >= 10": float(np.mean(L[m] >= 10)),
                   "track >= 30": float(np.mean(L[m] >= 30)), "median track length": float(np.median(L[m]))}
    out["births on tracks that also hold established points"] = float(np.mean(has_base[inv][is_birth]))
    if a.case:
        import synth_bench as sb
        from scipy.spatial import cKDTree

        tr = np.load(sb.WORK / "cases" / a.case / "truth.npz")
        ok = np.zeros(len(arr["frame"]), bool)
        d0 = []
        for f in range(a.first, a.last + 1, 10):
            r = np.flatnonzero((arr["frame"] == f) & ~is_birth)
            t = tr["pos_mm"][tr["frame"] == f]
            d, j = cKDTree(t).query(arr["pos"][r])
            d0.append((arr["pos"][r] - t[j])[d < 0.6])
        bias = np.median(np.vstack(d0), axis=0)
        for f in range(a.first, a.last + 1):
            r = np.flatnonzero(arr["frame"] == f)
            t = tr["pos_mm"][tr["frame"] == f]
            d, _ = cKDTree(t).query(arr["pos"][r] - bias)
            ok[r] = d < 0.4
        out["truth: births within 0.4 mm of a particle"] = float(ok[is_birth].mean())
        for k in (2, 10, 30):
            m = is_birth & (L >= k)
            out[f"truth: precision of births on tracks >= {k}"] = float(ok[m].mean()) if m.any() else None
        m = is_birth & (L < 2)
        out["truth: precision of unlinked births"] = float(ok[m].mean()) if m.any() else None
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

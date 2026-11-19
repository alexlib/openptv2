"""Truth-based bounds for the joint matching+tracking idea
(docs/plans/2026-11-19-joint-tracker.md, step 0, B1-B6).

The candidate pool is prototyped here from the existing matcher building blocks
(pair adjacency + clique finders, without ``take_best_candidates``); it moves
into ``algorithms/correspondences.py`` only if B1/B2 pass.

Cases need ``truth.npz['blob_tid']`` (rebuild with scripts/synth_bench.py build).

    uv run python scripts/joint_bounds.py b1 --case <case> [--frames 20] [--pairs]

Results are merged into bench/step23_joint_bounds_2026-11.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_bench as sb  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "bench" / "step23_joint_bounds_2026-11.json"


def case_setup(case: str):
    from openptv2.storage import RunStore

    case_dir = sb.WORK / "cases" / case
    info = json.loads((case_dir / "case.json").read_text())
    exp = sb._experiment(case_dir, case_dir / "parameters_Run1.yaml", 1, info["n_out"])
    store = RunStore(case_dir / "res" / "run.zarr", mode="r")
    tr = np.load(case_dir / "truth.npz")
    if "blob_tid" not in tr.files:
        raise SystemExit(f"{case}: truth.npz has no blob_tid; rebuild the case")
    return case_dir, info, exp, store, tr


def raw_targets(store, frame: int, n_cams: int) -> list[np.ndarray]:
    from openptv2.storage.run_store import _frame_key

    return [np.asarray(store.root[f"targets/cam_{c}/{_frame_key(frame)}"]) for c in range(n_cams)]


def pool_frame(rows: list[np.ndarray], cpar, cals, vpar, with_pairs: bool = False):
    """All blob combinations that pass the epipolar test, none discarded.

    Returns ``ids (M, n_cams)`` (target index per camera, -1 = unused) and
    ``corr (M,)``. 4-camera, 3-camera and (optionally) 2-camera combinations;
    a combination of a larger one is listed too (every combination, not a
    non-overlapping choice).
    """
    from openptv2.algorithms import correspondences as ac
    from openptv2.algorithms.tracking_frame_buf import Frame, Target

    n_cams = len(rows)
    frm = Frame(num_cams=n_cams, max_targets=max(1, max(len(r) for r in rows)))
    for c, r in enumerate(rows):
        frm.targets[c] = [
            Target(pnr=i, x=float(t[1]), y=float(t[2]), n=int(t[3]), nx=int(t[4]),
                   ny=int(t[5]), sumg=int(t[6]), tnr=-1)
            for i, t in enumerate(r)
        ]
        frm.num_targets[c] = len(r)
    corrected = ac.correct_frame(frm, cals, cpar, 1e-5)
    p1, n, p2, corr, dist = ac.allocate_adjacency_arrays(n_cams, frm.num_targets)
    ac.match_pairs(p1, n, p2, corr, dist, corrected, frm, vpar, cpar, cals)

    size = n_cams * ac.NMAX
    found_p, found_c = [], []

    def scratch():
        return (np.full((size, n_cams), -1, np.int32), np.zeros(size, np.float64))

    zero_use = np.zeros((n_cams, ac.NMAX), np.int32)
    if n_cams == 4:
        sp, sc = scratch()
        fcm = ac.four_camera_matching if ac.is_compiled() else ac._four_camera_matching_fast_py
        m = fcm(p1, n, p2, corr, dist, frm.num_targets[0], vpar.corrmin, sp, sc, size)
        found_p.append(sp[:m].copy())
        found_c.append(sc[:m].copy())
    if n_cams >= 3:
        sp, sc = scratch()
        m = ac.three_camera_matching(p1, n, p2, corr, dist, n_cams, frm.num_targets,
                                     vpar.corrmin, sp, sc, size, zero_use.copy())
        found_p.append(sp[:m].copy())
        found_c.append(sc[:m].copy())
    if with_pairs or n_cams == 2:
        sp, sc = scratch()
        m = ac.consistent_pair_matching(p1, n, p2, corr, dist, n_cams, frm.num_targets,
                                        vpar.corrmin, sp, sc, size, zero_use.copy())
        found_p.append(sp[:m].copy())
        found_c.append(sc[:m].copy())
    sat = int((np.asarray(n) >= ac.MAXCAND).sum())
    cp = np.vstack(found_p) if found_p else np.empty((0, n_cams), np.int32)
    cc = np.concatenate(found_c) if found_c else np.empty(0)
    # x-sorted index -> target number
    pnr = [np.array([c.pnr for c in corrected[cam]], np.int32) for cam in range(n_cams)]
    ids = np.full(cp.shape, -1, np.int32)
    for cam in range(n_cams):
        ok = cp[:, cam] >= 0
        ids[ok, cam] = pnr[cam][cp[ok, cam]]
    return ids, cc, sat


def purity(ids: np.ndarray, blob: np.ndarray, n_tid: int):
    """Match candidate rows to truth particles.

    ``blob (n_truth, n_cams)`` is the truth point -> target index map. A row is
    *pure for* truth particle p when every camera it uses holds p's own blob (a blob
    shared by merged particles serves all of them, so one row can be pure for
    several particles and two rows can be pure for particles that share a blob).

    Returns ``(row, truth)`` index pairs of all pure matches, ``ncam (M,)`` the
    number of cameras of each row, and ``ghost (M,)`` True for rows pure for nobody.
    """
    n_cams = ids.shape[1]
    ncam = (ids >= 0).sum(1)
    # per camera: sorted truth indices by blob id -> ranges
    order = [np.argsort(blob[:, c], kind="stable") for c in range(n_cams)]
    sorted_tid = [blob[order[c], c] for c in range(n_cams)]
    rows_out, truth_out = [], []
    used0 = np.argmax(ids >= 0, axis=1)  # first used camera of each row
    for r in range(len(ids)):
        c0 = int(used0[r])
        lo = np.searchsorted(sorted_tid[c0], ids[r, c0], "left")
        hi = np.searchsorted(sorted_tid[c0], ids[r, c0], "right")
        for p in order[c0][lo:hi]:
            if all(ids[r, c] < 0 or blob[p, c] == ids[r, c] for c in range(n_cams)):
                rows_out.append(r)
                truth_out.append(int(p))
    row = np.array(rows_out, np.int64)
    truth = np.array(truth_out, np.int64)
    ghost = np.ones(len(ids), bool)
    ghost[row] = False
    return row, truth, ncam, ghost


def b1(case: str, n_frames: int, with_pairs: bool, seed: int = 0) -> dict:
    case_dir, info, exp, store, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    frames_all = np.arange(1, info["n_out"] + 1)
    frames = frames_all[np.linspace(0, len(frames_all) - 1, n_frames).astype(int)]
    keys = ("truth_ge3", "greedy_ge3", "pool_ge3", "pool_exact", "greedy_exact",
            "lost_ge3", "recovered_ge3")
    agg = {f"{m}_{k}": 0 for m in ("all", "clean") for k in keys}
    agg.update(pool_rows=0, greedy_rows=0, saturated=0, greedy_not_in_pool=0)
    by_n = {k: {"pool_pure": 0, "pool_ghost": 0, "greedy_pure": 0, "greedy_ghost": 0}
            for k in range(2, n_cams + 1)}
    for f in frames:
        sel = np.flatnonzero(tr["frame"] == f)
        blob = tr["blob_tid"][sel]
        ndet = (blob >= 0).sum(1)
        # "clean" truth: no detected camera shares its blob with another particle
        clean = np.ones(len(sel), bool)
        for cam in range(n_cams):
            u, inv, cnt = np.unique(blob[:, cam], return_inverse=True, return_counts=True)
            shared = (blob[:, cam] >= 0) & (cnt[inv] > 1)
            clean &= ~shared
        rows = raw_targets(store, int(f), n_cams)
        n_tid = max(len(r) for r in rows) + 1
        ids, _corr, sat = pool_frame(rows, cpar, cals, vpar, with_pairs)
        g_pos, g_ids = store.read_correspondences(int(f))
        g_ids = np.asarray(g_ids, np.int32)[:, :n_cams]
        # sanity: every stored (greedy) row is a pool row
        pool_set = {tuple(r) for r in ids.tolist()}
        agg["greedy_not_in_pool"] += sum(tuple(r) not in pool_set for r in g_ids.tolist())
        res = {}
        for name, arr in (("pool", ids), ("greedy", g_ids)):
            prow, ptruth, ncam, ghost = purity(arr, blob, n_tid)
            best = np.zeros(len(sel), np.int64)
            np.maximum.at(best, ptruth, ncam[prow])
            res[name] = best
            for k in range(2, n_cams + 1):
                m = ncam == k
                by_n[k][f"{name}_pure"] += int((m & ~ghost).sum())
                by_n[k][f"{name}_ghost"] += int((m & ghost).sum())
        agg["pool_rows"] += len(ids)
        agg["greedy_rows"] += len(g_ids)
        agg["saturated"] += sat
        for m, mask in (("all", np.ones(len(sel), bool)), ("clean", clean)):
            g3 = (ndet >= 3) & mask
            lost = g3 & (res["greedy"] < 3)
            for k, v in (("truth_ge3", g3), ("greedy_ge3", g3 & (res["greedy"] >= 3)),
                         ("pool_ge3", g3 & (res["pool"] >= 3)),
                         ("greedy_exact", g3 & (res["greedy"] >= ndet)),
                         ("pool_exact", g3 & (res["pool"] >= ndet)),
                         ("lost_ge3", lost), ("recovered_ge3", lost & (res["pool"] >= 3))):
                agg[f"{m}_{k}"] += int(v.sum())
    out = {"frames": len(frames), "with_pairs": with_pairs, **agg,
           "pool_rows_per_frame": agg["pool_rows"] / len(frames),
           "greedy_rows_per_frame": agg["greedy_rows"] / len(frames), "by_cameras": by_n}
    for m in ("all", "clean"):
        t3 = max(agg[f"{m}_truth_ge3"], 1)
        out[f"{m}_greedy_recall_ge3"] = agg[f"{m}_greedy_ge3"] / t3
        out[f"{m}_pool_recall_ge3"] = agg[f"{m}_pool_ge3"] / t3
        out[f"{m}_lost_frac_ge3"] = agg[f"{m}_lost_ge3"] / t3
        out[f"{m}_pool_recovers_of_lost"] = agg[f"{m}_recovered_ge3"] / max(agg[f"{m}_lost_ge3"], 1)
    return out


def triangulate(store, frame: int, ids: np.ndarray, cpar, cals, vpar) -> np.ndarray:
    """lsq positions (mm) of ``ids`` rows, like ``match_frame_correspondences``."""
    from openptv2.correspondences import MatchedCoords
    from openptv2.orientation import point_positions

    n_cams = len(cals)
    mcs = [MatchedCoords(store.read_targets(c, frame), cpar, cals[c]) for c in range(n_cams)]
    flat = np.array([mc.get_by_pnrs(ids[:, c]) for c, mc in enumerate(mcs)])
    pos, _rcm = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar, method="lsq")
    return np.asarray(pos, float)


def b2(case: str) -> dict:
    """Build the B2 input variants of ``case`` (new case dirs under cases/):

    ``<case>_B2ng``    the greedy cloud with every ghost row removed (known bound)
    ``<case>_B2pool``  one best pure row per truth particle taken from the whole pool
                       (>= 3 cameras; truth picks, no ghosts)
    Track them with synth_bench (two_phase, unified, oracle) and compare.
    """
    import shutil

    from openptv2.storage import RunStore

    case_dir, info, exp, store, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    outs = {}
    for tag in ("B2ng", "B2pool"):
        d = sb.WORK / "cases" / f"{case}_{tag}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(case_dir, d)
        outs[tag] = RunStore(d / "res" / "run.zarr", mode="a")
    stats = {"B2ng_rows": 0, "B2pool_rows": 0, "greedy_rows": 0}
    for f in range(1, info["n_out"] + 1):
        sel = np.flatnonzero(tr["frame"] == f)
        blob = tr["blob_tid"][sel]
        rows = raw_targets(store, f, n_cams)
        n_tid = max(len(r) for r in rows) + 1
        g_pos, g_ids = store.read_correspondences(f)
        g_ids = np.asarray(g_ids, np.int32)[:, :n_cams]
        _r, _t, _n, ghost = purity(g_ids, blob, n_tid)
        outs["B2ng"].write_correspondences(f, np.asarray(g_pos)[~ghost], g_ids[~ghost])
        ids, corr, _ = pool_frame(rows, cpar, cals, vpar)
        prow, ptruth, ncam, _g = purity(ids, blob, n_tid)
        o = np.lexsort((corr[prow], ncam[prow], ptruth))  # per truth: best (ncam, corr) last
        last = np.r_[ptruth[o][1:] != ptruth[o][:-1], True]
        pick = np.unique(prow[o][last])
        pick = pick[ncam[pick] >= 3]
        sel_ids = ids[pick]
        pos = triangulate(store, f, sel_ids, cpar, cals, vpar) if len(pick) else np.empty((0, 3))
        outs["B2pool"].write_correspondences(f, pos, sel_ids)
        stats["B2ng_rows"] += int((~ghost).sum())
        stats["B2pool_rows"] += len(pick)
        stats["greedy_rows"] += len(g_ids)
    for tag in outs:
        sb.stamp_tnr(sb.WORK / "cases" / f"{case}_{tag}")
    return stats


def save(key: str, case: str, value: dict) -> None:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data.setdefault(key, {})[case] = value
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("b1")
    p.add_argument("--case", required=True)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--pairs", action="store_true")
    q = sub.add_parser("b2")
    q.add_argument("--case", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "b2":
        r = b2(a.case)
        save("B2_inputs", a.case, r)
        print(json.dumps(r, indent=2))
        return
    if a.cmd == "b1":
        r = b1(a.case, a.frames, a.pairs)
        save("B1b" if a.pairs else "B1", a.case, r)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()

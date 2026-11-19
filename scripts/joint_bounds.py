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


def shared_mask(blob: np.ndarray) -> np.ndarray:
    """(n_truth, n_cams) True where the particle's blob is shared with another particle."""
    sh = np.zeros(blob.shape, bool)
    for cam in range(blob.shape[1]):
        _u, inv, cnt = np.unique(blob[:, cam], return_inverse=True, return_counts=True)
        sh[:, cam] = (blob[:, cam] >= 0) & (cnt[inv] > 1)
    return sh


def ladder_inputs(case: str, min_cams: int = 3) -> dict:
    """Error-budget inputs with PERFECT association (no matcher, no ghosts):

    ``<case>_E1a``  every truth particle, rows from its clean (unshared) blobs only
    ``<case>_E1b``  every truth particle, rows from all its detected blobs (merged ones too)
    Both need >= ``min_cams`` cameras. Track with synth_bench ``oracle`` to get the accuracy
    floor of the measurements + estimator, with association made free.
    """
    import shutil

    from openptv2.storage import RunStore

    case_dir, info, exp, store, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    outs, stats = {}, {}
    for tag in ("E1a", "E1b"):
        d = sb.WORK / "cases" / f"{case}_{tag}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(case_dir, d)
        outs[tag] = RunStore(d / "res" / "run.zarr", mode="a")
        stats[tag] = 0
    for f in range(1, info["n_out"] + 1):
        sel = np.flatnonzero(tr["frame"] == f)
        blob = tr["blob_tid"][sel]
        sh = shared_mask(blob)
        for tag, use in (("E1a", (blob >= 0) & ~sh), ("E1b", blob >= 0)):
            ids = np.where(use, blob, -1).astype(np.int32)
            ids = ids[(ids >= 0).sum(1) >= min_cams]
            pos = triangulate(store, f, ids, cpar, cals, vpar) if len(ids) else np.empty((0, 3))
            outs[tag].write_correspondences(f, pos, ids)
            stats[tag] += len(ids)
    for tag in outs:
        sb.stamp_tnr(sb.WORK / "cases" / f"{case}_{tag}")
    return stats


def demerge_inputs(case: str, sigma_px: float, min_cams: int = 3) -> dict:
    """M2: perfect association AND ideal de-merging. Like ``E1b`` (all detected blobs), but a
    merged (shared) blob is replaced, for each of its particles, by that particle's own spot
    centre (projection of the true position with the render calibration) plus Gaussian noise of
    ``sigma_px``. Tracks as ``<case>_E1c<sigma>``; use synth_bench ``oracle``."""
    import shutil

    from openptv2.orientation import point_positions
    from openptv2.storage import RunStore
    from openptv2.transforms import convert_arr_pixel_to_metric, distorted_to_flat

    case_dir, info, exp, store, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    render = list(sb._experiment(case_dir, case_dir / "parameters_Run1.yaml", 1, info["n_out"]).cals)
    sb._render_cals(render, cpar)
    n_cams = len(cals)
    d = sb.WORK / "cases" / f"{case}_E1c{sigma_px:g}"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(case_dir, d)
    out = RunStore(d / "res" / "run.zarr", mode="a")
    rng = np.random.default_rng(1)
    total = replaced = 0
    for f in range(1, info["n_out"] + 1):
        sel = np.flatnonzero(tr["frame"] == f)
        blob = tr["blob_tid"][sel]
        sh = shared_mask(blob)
        rows = raw_targets(store, f, n_cams)
        keep = (blob >= 0).sum(1) >= min_cams
        px = np.full((len(sel), n_cams, 2), np.nan)
        for cam in range(n_cams):
            det = blob[:, cam] >= 0
            px[det, cam] = rows[cam][blob[det, cam], 1:3]
            m = det & sh[:, cam]
            if m.any():
                own = sb._project(tr["pos_mm"][sel[m]], render[cam], cpar)
                px[m, cam] = own + rng.normal(0, sigma_px, own.shape)
        replaced += int((sh & (blob >= 0))[keep].sum())
        total += int(((blob >= 0))[keep].sum())
        flat = np.full((n_cams, int(keep.sum()), 2), -1e10)
        for cam in range(n_cams):
            ok = keep & ~np.isnan(px[:, cam, 0])
            idx = np.flatnonzero(ok[keep])
            if len(idx):
                met = convert_arr_pixel_to_metric(px[ok, cam], cpar)
                flat[cam, idx] = distorted_to_flat(met, cals[cam])
        pos = np.asarray(point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar, method="lsq")[0], float)
        out.write_correspondences(f, pos, blob[keep].astype(np.int32))
    sb.stamp_tnr(d)
    return {"rows_blobs": total, "blobs_replaced": replaced, "share_replaced": replaced / max(total, 1)}


def b3(case: str, n_frames: int = 12, order: int = 3) -> dict:
    """Prediction error in PIXELS: predict a particle's 3D position at t from its ``order`` previous
    positions (lsq of its clean blobs, quadratic/linear extrapolation as ``track_assisted``), project
    into each camera where it has a clean blob at t and compare with that blob's centre. This is the
    error of the offset a track would use to de-merge a blob. Also by number of clean cameras at t-1..t-3
    (how well a position can be known)."""
    from openptv2.correspondences import MatchedCoords
    from openptv2.orientation import point_positions
    from openptv2.track_assisted import _predict_position

    case_dir, info, exp, store, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    f_list = np.linspace(order + 2, info["n_out"] - 2, n_frames).astype(int)
    errs, errs_cam, errs_dim = [], [], []
    for f in f_list:
        frs = [f - j for j in range(order, -1, -1)]  # f-order .. f
        sels = {g: np.flatnonzero(tr["frame"] == g) for g in frs}
        pid = {g: tr["particle_id"][sels[g]] for g in frs}
        common = set(pid[frs[0]].tolist())
        for g in frs[1:]:
            common &= set(pid[g].tolist())
        mcs = {g: [MatchedCoords(store.read_targets(c, int(g)), cpar, cals[c]) for c in range(n_cams)]
               for g in frs}
        rows_f = raw_targets(store, int(f), n_cams)
        loc = {g: {int(p): i for i, p in enumerate(pid[g])} for g in frs}
        clean = {g: ~shared_mask(tr["blob_tid"][sels[g]]) & (tr["blob_tid"][sels[g]] >= 0) for g in frs}
        for p in sorted(common):
            hist = []
            ok = True
            for g in frs[:-1]:
                i = loc[g][p]
                use = clean[g][i]
                if use.sum() < 3:
                    ok = False
                    break
                ids = np.where(use, tr["blob_tid"][sels[g]][i], -1).astype(np.int32)[None]
                flat = np.array([mc.get_by_pnrs(ids[:, c]) for c, mc in enumerate(mcs[g])])
                hist.append(np.asarray(point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar,
                                                       method="lsq")[0], float)[0])
            if not ok:
                continue
            pred = _predict_position(hist[::-1], "poly" if order >= 3 else "linear")
            i_t = loc[f][p]
            for c in range(n_cams):
                if not clean[f][i_t][c]:
                    continue
                px = sb._project(pred[None], cals[c], cpar)[0]
                meas = rows_f[c][tr["blob_tid"][sels[f]][i_t][c], 1:3]
                errs.append(np.linalg.norm(px - meas))
                errs_cam.append(c)
                errs_dim.append(px - meas)
    e = np.array(errs)
    d = np.array(errs_dim)
    return {"n": len(e), "rms_px": float(np.sqrt(np.mean(e**2))), "median_px": float(np.median(e)),
            "p90_px": float(np.percentile(e, 90)), "p99_px": float(np.percentile(e, 99)),
            "rms_x_px": float(np.sqrt(np.mean(d[:, 0] ** 2))), "rms_y_px": float(np.sqrt(np.mean(d[:, 1] ** 2))),
            "share_below_0.3px": float(np.mean(e < 0.3)), "share_below_1px": float(np.mean(e < 1.0))}


def h1(case: str, rounds: int = 3, window: int = 9, order: int = 2, var_c: float = 0.01,
       write: bool = False, trust: bool = True, gate: float = 4.0, trust_support: float = 3.0,
       prior: bool = True, prior_scale: float = 1.0) -> dict:
    """H1 of docs/plans/2026-11-20-holistic-ptv.md: the blob model on PERFECTLY ASSOCIATED data.

    Every truth particle is a track with known blob membership (truth). The refinement itself is
    :func:`openptv2.merge_refine.refine_merged` (truth-free); this driver builds its input from the case,
    scores the result against truth: 3D error by number of merged cameras per round, and the pixel error of
    the corrected centres against the true spot centres (render calibration), the raw blob centres and the
    predictions. With ``write`` the result goes into ``<case>_H1`` (correspondences + observation table) for
    the linkers.
    """
    import shutil

    from openptv2.merge_refine import refine_merged
    from openptv2.storage import RunStore

    case_dir, info, exp, store, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams, n_out = len(cals), info["n_out"]
    fr, pid, blob, truth = tr["frame"], tr["particle_id"], tr["blob_tid"], tr["pos_mm"]
    det = blob >= 0
    n_all = len(fr)
    cx = np.full((n_all, n_cams, 2), np.nan)
    sumg = np.full((n_all, n_cams), np.nan)
    shared = np.zeros((n_all, n_cams), bool)
    frame_idx = {}
    for f in range(1, n_out + 1):
        sel = np.flatnonzero(fr == f)
        frame_idx[f] = sel
        rows = raw_targets(store, f, n_cams)
        shared[sel] = shared_mask(blob[sel])
        for c in range(n_cams):
            d = det[sel, c]
            cx[sel[d], c] = rows[c][blob[sel[d], c], 1:3]
            sumg[sel[d], c] = rows[c][blob[sel[d], c], 6]
    res = refine_merged(fr, pid, blob, cx, sumg, cpar, cals, vpar, rounds=rounds, window=window, order=order,
                        var_c=var_c, trust=trust, trust_support=trust_support, gate=gate, prior=prior,
                        prior_scale=prior_scale)
    ev = det.sum(1) >= 3                                       # the population the output is judged on
    nm = shared.sum(1)

    def report(x, label):
        bias = np.median((x - truth)[ev], axis=0)
        err = np.linalg.norm(x - truth - bias, axis=1)
        out = {"all_rms": float(np.sqrt(np.mean(err[ev] ** 2))), "all_median": float(np.median(err[ev])),
               "label": label}
        for k, name in ((0, "0"), (1, "1"), (2, "2"), (3, "3+")):
            m = ev & ((nm == k) if k < 3 else (nm >= 3))
            out[f"merged_cams_{name}"] = {"n": int(m.sum()),
                                          "rms": float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else None}
        return out

    results = [report(res.x_rounds[0], "round0 raw blob centres (equal weights)")]
    for i, xr in enumerate(res.x_rounds[1:], 1):
        results.append(report(xr, f"round{i} merge-aware"))
    # pixel error of the final observations against the true spot centres (render calibration)
    render = list(sb._experiment(case_dir, case_dir / "parameters_Run1.yaml", 1, n_out).cals)
    sb._render_cals(render, cpar)
    eff, raw_e, pred_e, zz = [], [], [], []
    for c in range(n_cams):
        m = shared[:, c] & np.isfinite(res.obs_px[:, c, 0])
        if not m.any():
            continue
        own = sb._project(truth[m], render[c], cpar)
        e = np.linalg.norm(res.obs_px[m, c] - own, axis=1)
        eff.append(e)
        raw_e.append(np.linalg.norm(cx[m, c] - own, axis=1))
        zz.append(e**2 / res.obs_var[m, c] / 2)
        mp = m & np.isfinite(res.pred_px[:, c, 0])
        if mp.any():
            pred_e.append(np.linalg.norm(res.pred_px[mp, c] - sb._project(truth[mp], render[c], cpar), axis=1))
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)  # noqa: E731
    rms = lambda a: float(np.sqrt(np.mean(a**2))) if len(a) else None  # noqa: E731
    pix = {"merged_obs": int(len(cat(eff))), "raw_centre_rms_px": rms(cat(raw_e)),
           "effective_obs_rms_px": rms(cat(eff)), "effective_obs_median_px": float(np.median(cat(eff))),
           "prediction_rms_px": rms(cat(pred_e)), "variance_calibration_z2": float(np.mean(cat(zz))),
           "rounds": res.info}
    out = {"case": case, "rounds": results, "pixel_error_of_merged_observations": [pix]}
    if write:
        d = sb.WORK / "cases" / f"{case}_H1"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(case_dir, d)
        o = RunStore(d / "res" / "run.zarr", mode="a")
        for f in range(1, n_out + 1):
            sel = frame_idx[f]
            rows_k = sel[det[sel].sum(1) >= 3]
            ids = np.where(det[rows_k], blob[rows_k], -1).astype(np.int32)
            o.write_correspondences(f, res.x[rows_k], ids)
            ob = np.zeros((len(rows_k), n_cams, 4), np.float32)
            ob[:, :, 0:2] = res.obs_px[rows_k]
            ob[:, :, 2] = res.obs_var[rows_k]
            ob[:, :, 3] = np.where(det[rows_k], res.n_share[rows_k], 0)
            o.write_observations(f, ob)
        sb.stamp_tnr(d)
    return out


def h2(case: str, tracker: str = "two_phase", radius: float = 2.0, min_len: int = 2, iters: int = 0, tag: str = "",
       fill: bool = False, fill_radius: float = 1.5, fill_gap: int = 3, fill_growth: float = 0.0,
       fill_hermite: bool = False, mass: tuple | None = None, **opts) -> dict:
    """H2: merge-aware refinement of a REAL tracker run (no truth in the algorithm).

    Runs ``tracker`` on the case (keeping its store), finds shared blobs from the tracks
    (:mod:`openptv2.merge_post`), refines the positions (:func:`openptv2.merge_refine.refine_merged`) and
    scores raw and refined positions of the *same* track rows against truth (links unchanged).
    With ``iters`` > 0 the loop continues: the refined positions replace those of the case's point cloud, the
    tracker links again on them (case ``<case>_H2i<k>``), and the new tracks give new shared blobs and new
    positions (always computed from the raw blob centres, never from earlier refinements).
    Writes ``runs/<case>/<tracker>+h2base``, ``+h2`` and ``+h2i<k>`` (``pred.npz``) for ``synth_bench eval``.
    """
    import os
    import shutil

    from openptv2.merge_post import arrays_from_store, extend_membership, fill_gaps
    from openptv2.merge_refine import refine_merged, shared_blobs
    from openptv2.storage import RunStore

    case_dir, info, exp, store0, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams, n_out = len(cals), info["n_out"]
    os.environ["KEEP_STORE"] = "1"
    if fill and iters:
        raise ValueError("gap filling adds rows, which cannot be written back to the case yet: use iters=0")

    def tracked_arrays(run_case):
        run_dir = sb.WORK / "runs" / run_case / tracker
        if not (run_dir / "res" / "run.zarr").exists():
            sb.track(run_case, tracker)
        return arrays_from_store(RunStore(run_dir / "res" / "run.zarr", mode="r"), "ptv_is", 1, n_out, n_cams)

    def refine_from(a):
        blob, centre, sumg, n_att = extend_membership(a["frame"], a["pos"], a["blob"], a["centre"], a["sumg"],
                                                      a["blobs"], cpar, cals, radius=radius)
        res = refine_merged(a["frame"], a["tid"], blob, centre, sumg, cpar, cals, vpar, **opts)
        if fill:
            from openptv2.reconnect import reconnect_pieces

            path = np.where(np.isfinite(res.x[:, :1]), res.x, a["pos"])
            # re-join fragments on the refined path, then claim blobs where the path says the particle is
            a = {**a, "tid": reconnect_pieces(a["tid"], a["frame"], path, max_gap=fill_gap + 1, min_len=4)}
            new = fill_gaps(a, path, cpar, cals, radius=fill_radius, max_gap=fill_gap,
                            interp="hermite" if fill_hermite else "linear", radius_growth=fill_growth,
                            mass_range=mass)
            if len(new["frame"]):
                a = {**a, **{k: np.concatenate([a[k], new[k]]) for k in ("frame", "tid", "pos", "blob", "centre", "sumg")}}
                blob, centre, sumg = a["blob"], a["centre"], a["sumg"]
                blob, centre, sumg, n2 = extend_membership(a["frame"], a["pos"], blob, centre, sumg, a["blobs"],
                                                           cpar, cals, radius=radius)
                n_att += n2
                res = refine_merged(a["frame"], a["tid"], blob, centre, sumg, cpar, cals, vpar, **opts)
        return res, blob, n_att, a

    def write_pred(label, a, pos):
        keep = np.bincount(a["tid"])[a["tid"]] >= min_len
        d = sb.WORK / "runs" / case / f"{tracker}+{label}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "pred.npz", trajid=a["tid"][keep], frame=a["frame"][keep], pos=pos[keep])
        (d / "run.json").write_text(json.dumps({"tracker": f"{tracker}+{label}", "case": case, "time_s": 0.0,
                                                "status": "ok"}))
        return int(keep.sum())

    a = tracked_arrays(case)
    a0 = a
    res, blob, n_att, a = refine_from(a)
    new_pos = np.where(np.isfinite(res.x[:, :1]), res.x, a["pos"])
    base_a = a0
    tracked = write_pred("h2base", base_a, base_a["pos"])
    write_pred(("h2fill" + tag) if fill else "h2", a, new_pos)
    out = {"rows": int(len(a["frame"])), "tracked_rows": tracked, "attachments": int(n_att),
           "rows_with_a_shared_blob": float((shared_blobs(a["frame"], blob) >= 2).any(1).mean()),
           "info": res.info, "iterations": []}
    for k in range(1, iters + 1):
        run_case = f"{case}_H2i{k}"
        d = sb.WORK / "cases" / run_case
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(case_dir, d)
        st = RunStore(d / "res" / "run.zarr", mode="a")
        off = 0
        for f in range(1, n_out + 1):
            pos0, ids0 = st.read_correspondences(f)
            n = len(pos0)
            seg = new_pos[off: off + n]
            assert len(seg) == n and (a["frame"][off: off + n] == f).all(), "row alignment lost"
            st.write_correspondences(f, np.where(np.isfinite(seg[:, :1]), seg, pos0), ids0)
            off += n
        sb.stamp_tnr(d)
        shutil.rmtree(sb.WORK / "runs" / run_case / tracker, ignore_errors=True)
        sb.track(run_case, tracker)
        a = tracked_arrays(run_case)
        res, blob, n_att, a = refine_from(a)
        new_pos = np.where(np.isfinite(res.x[:, :1]), res.x, a["pos"])
        write_pred(f"h2i{k}", a, new_pos)
        out["iterations"].append({"k": k, "attachments": int(n_att),
                                  "rows_with_a_shared_blob": float((shared_blobs(a["frame"], blob) >= 2).any(1).mean())})
    return out


def shared_blobs_count(frame, blob):
    from openptv2.merge_refine import shared_blobs

    return shared_blobs(frame, blob)


def m3(case: str, n_part: int = 300, half: int = 10, deg: int = 3, clean_only: bool = True,
       sigma_depth_weighting: bool = False) -> dict:
    """Estimator bound with PERFECT association: velocity of true particles at the middle of a
    (2*half+1)-frame window from (A) per-frame lsq 3D points + a degree-``deg`` polynomial in time
    (what smoothing the tracker output does) and (B) one degree-``deg`` trajectory fitted directly to
    all blob centres of all cameras and frames (reprojection error in pixels)."""
    from scipy.optimize import least_squares

    case_dir, info, exp, store, tr = case_setup(case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams, k = len(cals), int(info["k"])
    flow = sb.Flow()
    f0 = info["n_out"] // 2
    frames = np.arange(f0 - half, f0 + half + 1)
    # particles detected (clean or not, per option) in >= 3 cameras in every frame of the window
    by_f = {f: np.flatnonzero(tr["frame"] == f) for f in frames}
    pid = {f: tr["particle_id"][by_f[f]] for f in frames}
    common = set(pid[frames[0]].tolist())
    for f in frames[1:]:
        common &= set(pid[f].tolist())
    rows = {f: raw_targets(store, int(f), n_cams) for f in frames}
    from openptv2.correspondences import MatchedCoords
    from openptv2.orientation import point_positions

    mcs = {f: [MatchedCoords(store.read_targets(c, int(f)), cpar, cals[c]) for c in range(n_cams)]
           for f in frames}

    def tri(f, ids):  # lsq position of one row, coordinate tables built once per frame
        flat = np.array([mc.get_by_pnrs(ids[:, c]) for c, mc in enumerate(mcs[f])])
        return np.asarray(point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar, method="lsq")[0], float)

    cand = []
    for p in sorted(common):
        ok = True
        info_f = {}
        for f in frames:
            i = by_f[f][np.flatnonzero(pid[f] == p)[0]]
            blob = tr["blob_tid"][i]
            if clean_only:
                shr = shared_mask(tr["blob_tid"][by_f[f]])[np.flatnonzero(pid[f] == p)[0]]
                use = (blob >= 0) & ~shr
            else:
                use = blob >= 0
            if use.sum() < 3:
                ok = False
                break
            info_f[f] = (i, use, blob)
        if ok:
            cand.append((p, info_f))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(cand), min(n_part, len(cand)), replace=False)
    t = frames - f0
    V = np.vander(t.astype(float), deg + 1, increasing=True)
    dV = np.zeros(deg + 1)
    dV[1] = 1.0  # d/dt at t=0 of the polynomial: coefficient 1
    errA, errB, u_sq = [], [], []
    for ci in pick:
        p, info_f = cand[ci]
        pts, obs = [], []
        for f in frames:
            i, use, blob = info_f[f]
            ids = np.where(use, blob, -1).astype(np.int32)[None]
            pts.append(tri(f, ids)[0])
            obs.append([rows[f][c][blob[c], 1:3] if use[c] else None for c in range(n_cams)])
        pts = np.array(pts)
        coef_a = np.linalg.lstsq(V, pts, rcond=None)[0]          # (deg+1, 3)
        pos_true = tr["pos_mm"][info_f[f0][0]]
        tn = 1.0 + (f0 - 1) * k
        u_true = flow.velocity(pos_true[None], tn)[0] * sb.FPS / k * k  # mm/s as in evaluate_pred
        va = coef_a[1] * sb.FPS / k
        def resid(x):
            c = x.reshape(deg + 1, 3)
            xyz = V @ c
            out = []
            for cam in range(n_cams):
                m = [j for j in range(len(frames)) if obs[j][cam] is not None]
                if not m:
                    continue
                pr = sb._project(xyz[m], cals[cam], cpar)
                out.append((pr - np.array([obs[j][cam] for j in m])).ravel())
            return np.concatenate(out)
        sol = least_squares(resid, coef_a.ravel(), method="lm", max_nfev=60)
        vb = sol.x.reshape(deg + 1, 3)[1] * sb.FPS / k
        errA.append(va - u_true)
        errB.append(vb - u_true)
        u_sq.append(np.sum(u_true**2))
    errA, errB = np.array(errA), np.array(errB)
    u_rms = float(np.sqrt(np.mean(u_sq)))
    rel = lambda e: float(np.sqrt(np.mean(np.sum(e**2, 1))) / u_rms)  # noqa: E731
    return {"n_particles": len(errA), "window": 2 * half + 1, "deg": deg, "clean_only": clean_only,
            "vErr_3d_points_polyfit": rel(errA), "vErr_image_space_fit": rel(errB),
            "rms_axis_3d": np.sqrt(np.mean(errA**2, 0)).tolist(),
            "rms_axis_image": np.sqrt(np.mean(errB**2, 0)).tolist(), "u_rms_mm_s": u_rms}


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
    q2 = sub.add_parser("ladder")
    q2.add_argument("--case", required=True)
    q4 = sub.add_parser("demerge")
    q4.add_argument("--case", required=True)
    q4.add_argument("--sigma", type=float, required=True)
    q6 = sub.add_parser("h1")
    q6.add_argument("--case", required=True)
    q6.add_argument("--rounds", type=int, default=3)
    q6.add_argument("--window", type=int, default=9)
    q6.add_argument("--order", type=int, default=2)
    q6.add_argument("--var-c", type=float, default=0.01)
    q6.add_argument("--write", action="store_true")
    q6.add_argument("--no-trust", action="store_true")
    q6.add_argument("--gate", type=float, default=4.0)
    q6.add_argument("--support", type=float, default=3.0)
    q6.add_argument("--no-prior", action="store_true")
    q6.add_argument("--prior-scale", type=float, default=1.0)
    q6.add_argument("--tag", default="")
    q7 = sub.add_parser("h2")
    q7.add_argument("--case", required=True)
    q7.add_argument("--tracker", default="two_phase")
    q7.add_argument("--radius", type=float, default=2.0)
    q7.add_argument("--rounds", type=int, default=3)
    q7.add_argument("--iters", type=int, default=0)
    q7.add_argument("--fill", action="store_true")
    q7.add_argument("--fill-radius", type=float, default=1.5)
    q7.add_argument("--fill-gap", type=int, default=3)
    q7.add_argument("--growth", type=float, default=0.0)
    q7.add_argument("--hermite", action="store_true")
    q7.add_argument("--mass", type=float, nargs=2, default=None)
    q7.add_argument("--tag", default="")
    q5 = sub.add_parser("b3")
    q5.add_argument("--case", required=True)
    q5.add_argument("--order", type=int, default=3)
    q3 = sub.add_parser("m3")
    q3.add_argument("--case", required=True)
    q3.add_argument("--n", type=int, default=300)
    q3.add_argument("--half", type=int, default=10)
    q3.add_argument("--deg", type=int, default=3)
    q3.add_argument("--all-detected", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "ladder":
        r = ladder_inputs(a.case)
        save("E1_inputs", a.case, r)
        print(json.dumps(r))
        return
    if a.cmd == "demerge":
        r = demerge_inputs(a.case, a.sigma)
        save("M2_inputs", f"{a.case}_sigma{a.sigma:g}", r)
        print(json.dumps(r))
        return
    if a.cmd == "h2":
        r = h2(a.case, a.tracker, a.radius, iters=a.iters, tag=a.tag, fill=a.fill, fill_radius=a.fill_radius,
               fill_gap=a.fill_gap, fill_growth=a.growth, fill_hermite=a.hermite,
               mass=tuple(a.mass) if a.mass else None, rounds=a.rounds)
        save("H2", f"{a.case}|{a.tracker}", r)
        print(json.dumps(r, indent=1))
        return
    if a.cmd == "h1":
        r = h1(a.case, a.rounds, a.window, a.order, a.var_c, a.write, not a.no_trust, a.gate, a.support, not a.no_prior, a.prior_scale)
        save("H1" + a.tag, a.case, r)
        print(json.dumps(r, indent=1))
        return
    if a.cmd == "b3":
        r = b3(a.case, order=a.order)
        save(f"B3_pred_error_px_order{a.order}", a.case, r)
        print(json.dumps(r))
        return
    if a.cmd == "m3":
        r = m3(a.case, a.n, a.half, a.deg, not a.all_detected)
        save("M3_" + ("all" if a.all_detected else "clean") + f"_w{2*a.half+1}_d{a.deg}", a.case, r)
        print(json.dumps(r, indent=1))
        return
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

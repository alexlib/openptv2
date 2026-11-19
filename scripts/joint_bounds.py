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
       write: bool = False, mode: str = "effective") -> dict:
    """H1 of docs/plans/2026-11-20-holistic-ptv.md: the blob model on PERFECTLY ASSOCIATED data.

    Every truth particle with >= 2 detected cameras is a track with known blob membership (truth). Starting
    from the raw blob centres (positions of ``E1b``), each round predicts every particle's pixel position in
    every camera from its other frames (leave-one-out local polynomial), gives every particle in a shared
    blob its effective observation (blob centre minus the other members' predictions, divided by its
    brightness share), and re-triangulates with weights 1/variance (Gauss-Newton on the reprojection error).
    Reports 3D error against truth by number of merged cameras, and the pixel error of the corrected centres
    against the true spot centres (render calibration), the raw blob centres and the predictions.
    """
    import shutil

    from openptv2.blob_model import (
        local_poly_predict,
        signature_from_history,
        split_blob_effective,
        triangulate_weighted,
    )
    from openptv2.loco import pixels_to_flat, project_pixels
    from openptv2.orientation import point_positions
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
    clean = det & ~shared
    # brightness signature per particle and camera: median grey sum over the frames where it is alone
    n_pid = int(pid.max()) + 1
    sig = np.full((n_pid, n_cams), np.nan)
    for c in range(n_cams):
        m = np.flatnonzero(clean[:, c])
        o = m[np.argsort(pid[m], kind="stable")]
        cuts = np.flatnonzero(np.r_[True, pid[o][1:] != pid[o][:-1], True])
        for a, b in zip(cuts[:-1], cuts[1:]):
            sig[pid[o[a]], c] = signature_from_history(sumg[o[a:b], c], np.ones(b - a, bool))
    pop = np.flatnonzero(det.sum(1) >= 2)                    # particles that can be triangulated
    pidx = np.full(n_all, -1)
    pidx[pop] = np.arange(len(pop))
    ev = det[pop].sum(1) >= 3                                 # the population the output is judged on
    n_pop = len(pop)
    fr_p, pid_p = fr[pop], pid[pop]
    f_rows = {f: np.flatnonzero(fr_p == f) for f in range(1, n_out + 1)}
    render = list(sb._experiment(case_dir, case_dir / "parameters_Run1.yaml", 1, n_out).cals)
    sb._render_cals(render, cpar)
    own = np.full((n_pop, n_cams, 2), np.nan)                # true spot centres (px), for diagnostics only
    for c in range(n_cams):
        d = det[pop, c]
        own[d, c] = sb._project(truth[pop][d], render[c], cpar)

    def triangulate_all(obs_px, obs_var, x_start=None):
        x = np.zeros((n_pop, 3)) if x_start is None else x_start.copy()
        for f, rr in f_rows.items():
            if len(rr) == 0:
                continue
            if x_start is None:
                flat = np.full((len(rr), n_cams, 2), -1e10)
                for c in range(n_cams):
                    v = ~np.isnan(obs_px[rr, c, 0])
                    flat[v, c] = pixels_to_flat(obs_px[rr[v], c], cals[c], cpar)
                x0 = np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)
            else:
                x0 = x_start[rr]
            x[rr] = triangulate_weighted(obs_px[rr], obs_var[rr], cpar, cals, x0, n_iter=3)
        return x

    def report(x, label):
        bias = np.median((x - truth[pop])[ev], axis=0)
        err = np.linalg.norm(x - truth[pop] - bias, axis=1)
        nm = shared[pop].sum(1)
        out = {"all_rms": float(np.sqrt(np.mean(err[ev] ** 2))), "all_median": float(np.median(err[ev]))}
        for k, name in ((0, "0"), (1, "1"), (2, "2"), (3, "3+")):
            m = ev & ((nm == k) if k < 3 else (nm >= 3))
            out[f"merged_cams_{name}"] = {"n": int(m.sum()), "rms": float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else None}
        out["label"] = label
        return out

    obs_raw = cx[pop]
    obs_var_raw = np.where(np.isnan(obs_raw[:, :, 0]), np.nan, var_c)
    x = triangulate_all(obs_raw, obs_var_raw)
    results = [report(x, "round0 raw blob centres (equal weights)")]
    px_diag = []
    for rnd in range(1, rounds + 1):
        pred3 = local_poly_predict(pid_p, fr_p, x, None, window=window, order=order)
        pred_px = np.full((n_pop, n_cams, 2), np.nan)
        for c in range(n_cams):
            ok = np.isfinite(pred3[:, 0]) & det[pop, c]
            if ok.any():
                pred_px[ok, c] = project_pixels(pred3[ok], cals[c], cpar)
        cl = clean[pop] & np.isfinite(pred_px[:, :, 0])
        resid2 = np.mean(((obs_raw - pred_px)[cl]) ** 2)
        var_p = max(float(resid2) - var_c, 0.01)           # prediction variance, learned from the clean blobs
        obs_px, obs_var = obs_raw.copy(), obs_var_raw.copy()
        n_groups = 0
        eff_err, raw_err, pred_err, zz = [], [], [], []
        for f in range(1, n_out + 1):
            sel = frame_idx[f]
            for c in range(n_cams):
                idx = sel[det[sel, c]]
                b = blob[idx, c]
                o = np.argsort(b, kind="stable")
                idx, b = idx[o], b[o]
                cuts = np.flatnonzero(np.r_[True, b[1:] != b[:-1], True])
                for a, e in zip(cuts[:-1], cuts[1:]):
                    if e - a < 2:
                        continue
                    mem = idx[a:e]
                    centre = cx[mem[0], c]
                    pm = pidx[mem]
                    pr = np.where((pm >= 0)[:, None], pred_px[np.maximum(pm, 0), c], np.nan)
                    unknown = ~np.isfinite(pr[:, 0])
                    pr[unknown] = centre
                    vp = np.where(unknown, 1.0, var_p)       # a member without a path: a blob radius of doubt
                    s_eff, v_eff = split_blob_effective(centre, pr, sig[pid[mem], c], var_c, vp)
                    n_groups += 1
                    for j in range(len(mem)):
                        if pm[j] < 0:
                            continue
                        obs_px[pm[j], c] = s_eff[j]
                        obs_var[pm[j], c] = v_eff[j]
                        t = own[pm[j], c]
                        eff_err.append(np.linalg.norm(s_eff[j] - t))
                        raw_err.append(np.linalg.norm(centre - t))
                        if not unknown[j]:
                            pred_err.append(np.linalg.norm(pr[j] - t))
                        zz.append(np.sum((s_eff[j] - t) ** 2) / v_eff[j] / 2)
        x = triangulate_all(obs_px, obs_var, x_start=x)
        results.append(report(x, f"round{rnd} effective observations"))
        rms = lambda a: float(np.sqrt(np.mean(np.square(a)))) if len(a) else None  # noqa: E731
        px_diag.append({"round": rnd, "var_p_px2": var_p, "merged_groups": n_groups,
                        "merged_obs": len(eff_err), "raw_centre_rms_px": rms(raw_err),
                        "effective_obs_rms_px": rms(eff_err), "effective_obs_median_px": float(np.median(eff_err)),
                        "prediction_rms_px": rms(pred_err), "variance_calibration_z2": float(np.mean(zz))})
    out = {"case": case, "rounds": results, "pixel_error_of_merged_observations": px_diag}
    if write:
        d = sb.WORK / "cases" / f"{case}_H1"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(case_dir, d)
        o = RunStore(d / "res" / "run.zarr", mode="a")
        for f in range(1, n_out + 1):
            sel = frame_idx[f]
            keep = det[sel].sum(1) >= 3
            rows_k = sel[keep]
            pi = pidx[rows_k]
            ids = np.where(det[rows_k], blob[rows_k], -1).astype(np.int32)
            o.write_correspondences(f, x[pi], ids)
            ob = np.zeros((len(rows_k), n_cams, 4), np.float32)
            ob[:, :, 0:2] = obs_px[pi]
            ob[:, :, 2] = obs_var[pi]
            ob[:, :, 3] = det[rows_k] + shared[rows_k]       # 0 unused, 1 alone, 2 shared
            o.write_observations(f, ob)
        sb.stamp_tnr(d)
    return out


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
    if a.cmd == "h1":
        r = h1(a.case, a.rounds, a.window, a.order, a.var_c, a.write)
        save("H1", a.case, r)
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

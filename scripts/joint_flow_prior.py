"""N2 feasibility: does a flow prior built from tracked neighbours separate true candidates from ghosts?

Candidate 3D points of frame t = every >= 2-camera combination that passes the epipolar test (the full pool, ``pool_frame``).
For each candidate the displacement t -> t+1 (and t-1 -> t) is read off the tracked neighbours (inverse-distance mean of
the k nearest tracked points' displacements), the candidate is moved by it, projected into every camera, and counted
where a detected blob lies within ``--tol`` px, at t+1 and at t-1.  A true particle has blobs at the predicted places
in the cameras that see it; a ghost mostly does not.  Truth tells real from ghost.

    uv run python scripts/joint_flow_prior.py --case L1_d4_c0_k1_n60_s0.08_a0.18_f0.05 --run holistic_diag
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "scripts")
import joint_bounds as jb  # noqa: E402
import synth_bench as sb  # noqa: E402


def all_pair_rows(rows, cpar, cals, vpar):
    """Every epipolar-consistent pair of blobs from two cameras (rows (M, n_cams) of target numbers, -1 = unused)."""
    from openptv2.algorithms import correspondences as ac
    from openptv2.algorithms.tracking_frame_buf import Frame, Target

    n_cams = len(rows)
    frm = Frame(num_cams=n_cams, max_targets=max(1, max(len(r) for r in rows)))
    for c, r in enumerate(rows):
        frm.targets[c] = [Target(pnr=i, x=float(t[1]), y=float(t[2]), n=int(t[3]), nx=int(t[4]), ny=int(t[5]),
                                 sumg=int(t[6]), tnr=-1) for i, t in enumerate(r)]
        frm.num_targets[c] = len(r)
    corrected = ac.correct_frame(frm, cals, cpar, 1e-5)
    p1, n, p2, corr, dist = ac.allocate_adjacency_arrays(n_cams, frm.num_targets)
    ac.match_pairs(p1, n, p2, corr, dist, corrected, frm, vpar, cpar, cals)
    pnr = [np.array([c.pnr for c in corrected[cam]], np.int32) for cam in range(n_cams)]
    out = []
    for c1 in range(n_cams - 1):
        for c2 in range(c1 + 1, n_cams):
            for i in range(frm.num_targets[c1]):
                k = int(n[c1, c2, i])
                if k <= 0:
                    continue
                cand = p2[c1, c2, i, :k]
                m = np.full((k, n_cams), -1, np.int32)
                m[:, c1] = pnr[c1][i]
                m[:, c2] = pnr[c2][cand]
                out.append(m)
    return np.vstack(out) if out else np.empty((0, n_cams), np.int32)


def sb_truth_next(tr, t):
    """{particle id: (position at t, position at t+1)} from the truth."""
    m0, m1 = tr["frame"] == t, tr["frame"] == t + 1
    p1 = dict(zip(tr["particle_id"][m1].tolist(), tr["pos_mm"][m1]))
    return {i: (pp, p1[i]) for i, pp in zip(tr["particle_id"][m0].tolist(), tr["pos_mm"][m0]) if i in p1}


def displacement_tables(pred: dict):
    tid, fr, pos = pred["trajid"], pred["frame"], pred["pos"]
    o = np.lexsort((fr, tid))
    tid, fr, pos = tid[o], fr[o], pos[o]
    same = (tid[1:] == tid[:-1]) & (fr[1:] == fr[:-1] + 1)
    a = np.flatnonzero(same)
    # (frame of the earlier point, position of the earlier point, displacement to the next frame)
    return fr[a], pos[a], pos[a + 1] - pos[a]


def flow_at(x: np.ndarray, tree: cKDTree, disp: np.ndarray, k: int, radius: float):
    d, j = tree.query(x, k=k, distance_upper_bound=radius)
    ok = np.isfinite(d)
    w = np.where(ok, 1.0 / (np.where(ok, d, 1.0) + 0.3), 0.0)
    jj = np.where(ok, j, 0)
    n = ok.sum(1)
    est = (w[:, :, None] * disp[jj]).sum(1) / np.maximum(w.sum(1), 1e-12)[:, None]
    return est, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--run", default="holistic_diag")
    ap.add_argument("--frames", type=int, nargs="+", default=[15, 30, 45])
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--pairs", action="store_true", help="all 2-camera epipolar pairs instead of the matcher pool")
    a = ap.parse_args()

    from openptv2.loco import project_pixels

    case_dir, info, exp, store, tr = jb.case_setup(a.case)
    cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
    n_cams = len(cals)
    pred = dict(np.load(sb.WORK / "runs" / a.case / a.run / "pred.npz"))
    f_t, p_t, d_t = displacement_tables(pred)
    agg: dict = {}
    diag: list = []
    for t in a.frames:
        sel = np.flatnonzero(tr["frame"] == t)
        blob = tr["blob_tid"][sel]
        tid_t = tr["particle_id"][sel]
        tpos = tr["pos_mm"][sel]
        rows = jb.raw_targets(store, t, n_cams)
        if a.pairs:
            ids = all_pair_rows(rows, cpar, cals, vpar)
        else:
            ids, _c, _sat = jb.pool_frame(rows, cpar, cals, vpar, True)
        ncam = (ids >= 0).sum(1)
        keep = ncam >= 2
        ids, ncam = ids[keep], ncam[keep]
        x = jb.triangulate(store, t, ids, cpar, cals, vpar)
        n_tid = max(len(r) for r in rows) + 1
        prow, ptruth, _n, ghost = jb.purity(ids, blob, n_tid)
        real_tid = np.full(len(ids), -1)
        real_tid[prow] = tid_t[ptruth]
        # which truth particles does the holistic output already hold at t?
        m = pred["frame"] == t
        tt = cKDTree(tpos)
        d0, j0 = tt.query(pred["pos"][m])
        bias = np.median(pred["pos"][m][d0 < 0.6] - tpos[j0[d0 < 0.6]], axis=0)
        d1, j1 = tt.query(pred["pos"][m] - bias, distance_upper_bound=0.4)
        found = set(tid_t[j1[np.isfinite(d1)]].tolist())
        # flow prior forward (t -> t+1) and backward (t-1 -> t)
        conf = {}
        for name, step in (("f", 1), ("b", -1)):
            tab = (f_t == t) if step == 1 else (f_t == t - 1)
            p0, d0_ = p_t[tab], d_t[tab]
            tree = cKDTree(p0)
            est, nn = flow_at(x, tree, d0_, a.k, a.radius)
            xs = x + est if step == 1 else x - est
            tgt = jb.raw_targets(store, t + step, n_cams)
            cnt = np.zeros(len(x), int)
            for c in range(n_cams):
                px = project_pixels(xs, cals[c], cpar)
                d, _ = cKDTree(tgt[c][:, 1:3]).query(px, distance_upper_bound=a.tol)
                cnt += np.isfinite(d)
            cnt[nn < 3] = -1                                  # no flow estimate there
            conf[name] = cnt
            if name == "f":
                # accuracy of the flow prior itself, on real candidates: predicted against true displacement
                nxt = sb_truth_next(tr, t)
                has = (real_tid >= 0) & (nn >= 3)
                tdisp = np.full((len(x), 3), np.nan)
                for r_i in np.flatnonzero(has):
                    q = nxt.get(int(real_tid[r_i]))
                    if q is not None:
                        tdisp[r_i] = q[1] - q[0]
                e = np.linalg.norm(est - tdisp, axis=1)
                e = e[np.isfinite(e)]
                diag.append(e)
        for label, sel_rows in (("all", np.ones(len(x), bool)),):
            pass
        real = real_tid >= 0
        # per truth particle: clean cameras, reachable by a pair, best temporal confirmation among its rows
        share = np.zeros(blob.shape, int)
        for c in range(n_cams):
            kk = np.where(blob[:, c] >= 0, blob[:, c], -1 - np.arange(len(blob)))
            u, inv_, cn = np.unique(kk, return_inverse=True, return_counts=True)
            share[:, c] = cn[inv_]
        nclean = ((blob >= 0) & (share == 1)).sum(1)
        best = np.full(len(sel), -2)
        mn = np.minimum(conf["f"], conf["b"])
        for r_i, p_i in zip(prow, ptruth):
            best[p_i] = max(best[p_i], mn[r_i])
        miss = ~np.isin(tid_t, list(found))
        for grp, gm in (("2+ clean", nclean >= 2), ("1 clean", nclean == 1), ("0 clean", nclean == 0)):
            g_ = gm & miss & ((blob >= 0).sum(1) >= 2)
            gg = agg.setdefault(("grp", grp), [0, 0, 0, 0])
            gg[0] += int(g_.sum())
            gg[1] += int((g_ & (best > -2)).sum())
            gg[2] += int((g_ & (best >= 2)).sum())
            gg[3] += int((g_ & (best >= 3)).sum())
        new_real = real & ~np.isin(real_tid, list(found))
        for n_lo in (2, 3):
            key = f"ncam{n_lo}"
            g = ncam == n_lo if n_lo == 2 else ncam >= 3
            for rule, ok in (("none", np.ones(len(x), bool)),
                             ("f>=2&b>=2", (conf["f"] >= 2) & (conf["b"] >= 2)),
                             ("f>=3&b>=3", (conf["f"] >= 3) & (conf["b"] >= 3)),
                             ("f+b>=5", (conf["f"] + conf["b"] >= 5) & (conf["f"] >= 0) & (conf["b"] >= 0))):
                acc = g & ok
                rec = agg.setdefault((key, rule), [0, 0, 0, 0, set()])
                rec[0] += int(acc.sum())
                rec[1] += int((acc & real).sum())
                rec[2] += int((acc & new_real).sum())
                rec[4] |= {(t, int(i)) for i in np.unique(real_tid[acc & new_real])}
        agg.setdefault("truth", [0])
        agg["truth"][0] += len(sel)
        agg.setdefault("truth_new_possible", [0])
        agg["truth_new_possible"][0] += int((~np.isin(tid_t, list(found))).sum())
    e = np.concatenate(diag)
    print(f"flow prior error on real candidates (mm): median {np.median(e):.3f}  p90 {np.percentile(e, 90):.3f} "
          f" p99 {np.percentile(e, 99):.3f}; true step median {np.median(np.linalg.norm(d_t, axis=1)):.3f}")
    print(f"pool rows (>= 2 cameras) in the last frame: {len(ids)}")
    print(f"truth points in the {len(a.frames)} frames: {agg['truth'][0]}; not in the holistic output: "
          f"{agg['truth_new_possible'][0]}")
    print(f"{'pool group':10s} {'rule':10s} {'accepted':>9s} {'real %':>7s} {'new real (unique)':>18s}")
    print("missing truth particles with blobs in >= 2 cameras, by clean cameras: total | reachable by a pair | "
          "confirmed (both directions >= 2 / >= 3 cameras)")
    for grp in ("2+ clean", "1 clean", "0 clean"):
        g = agg[("grp", grp)]
        print(f"  {grp:9s} {g[0]:6d} | {g[1]:6d} | {g[2]:6d} / {g[3]:6d}")
    for (key, rule), rec in sorted((k, v) for k, v in agg.items() if isinstance(k, tuple) and k[0] != "grp"):
        print(f"{key:10s} {rule:10s} {rec[0]:9d} {100 * rec[1] / max(rec[0], 1):7.1f} {len(rec[4]):18d}")


if __name__ == "__main__":
    main()

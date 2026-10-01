"""Answer-free tracker quality numbers for real data (and synthetic pred.npz).

Real data has no known answer, so these are the plan's real-data scores
(docs/plans/2026-09-30-tracker-plan.md, sections 1-2):

  jump_share     share of steps whose velocity changes by > 0.3 mm in depth
                 (median-based, lower is better; wrong links show up here)
  never4_share   share of long (>= 11 frames) trajectories never seen by all
                 4 cameras (best real-data hint of fake trajectories)
  jitter_*       scatter around an 11-frame quadratic fit (removes real
                 acceleration), median and worst-10% in depth
  mean_len etc.  trajectory length summary

Usage:
    uv run python scripts/realism_metrics.py RUN_ZARR [--last N] [--linkage ptv_is]
    uv run python scripts/realism_metrics.py --pred PRED_NPZ
    ... [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

W = 11
_T = np.arange(W) - W // 2
_V = np.vander(_T, 3)
_H = _V @ np.linalg.pinv(_V)


def chains_from_store(path: Path, last: int | None, linkage: str):
    from openptv2.storage import RunStore

    s = RunStore.open(Path(path), mode="r")
    frames = sorted(int(k.split("_")[1]) for k in s.root[f"linkage/{linkage}"].keys())
    f0, f1 = frames[0], frames[-1] if last is None else min(frames[-1], last)
    pos, nxt, prv, nc = {}, {}, {}, {}
    for f in range(f0, f1 + 1):
        p, n, q = s.read_linkage(f, linkage)
        _, cam = s.read_correspondences(f)
        pos[f], nxt[f], prv[f], nc[f] = q[:, :3], n, p, (cam >= 0).sum(1)
    out = []
    for f in range(f0, f1 + 1):
        for i in range(len(pos[f])):
            if prv[f][i] >= 0 and f > f0:
                continue
            c, k, ff, ii = [], [], f, i
            while True:
                c.append(pos[ff][ii])
                k.append(nc[ff][ii])
                j = nxt[ff][ii]
                if j < 0 or ff == f1:
                    break
                ff, ii = ff + 1, int(j)
            out.append((np.array(c), np.array(k)))
    return out, f1 - f0 + 1


def chains_from_pred(path: Path):
    p = np.load(path)
    tid, fr, x = p["trajid"], p["frame"], p["pos"]
    o = np.lexsort((fr, tid))
    tid, fr, x = tid[o], fr[o], x[o]
    cuts = np.flatnonzero(np.diff(tid)) + 1
    out = []
    for ff, xx in zip(np.split(fr, cuts), np.split(x, cuts)):
        br = np.flatnonzero(np.diff(ff) != 1) + 1
        out += [(xs, None) for xs in np.split(xx, br)]
    return out, int(fr.max() - fr.min() + 1)


def metrics(chains, n_frames: int) -> dict:
    L = np.array([len(c) for c, _ in chains])
    use = [c for c, _ in chains if len(c) >= 8]
    step = np.linalg.norm(np.vstack([np.diff(c, axis=0) for c in use]), axis=1)
    a = np.vstack([np.abs(np.diff(c, 2, axis=0)) for c in use])
    res = []
    for c, _ in chains:
        for s in range(0, len(c) - W + 1, W):
            seg = c[s : s + W]
            res.append(np.sqrt(((seg - _H @ seg) ** 2).sum(0) / (W - 3)))
    res = np.array(res)
    r = {
        "frames": n_frames,
        "trajectories": int((L >= 2).sum()),
        "mean_len": round(float(L[L >= 2].mean()), 2),
        "n_ge50": int((L >= 50).sum()),
        "median_step_mm": round(float(np.median(step)), 4),
        "median_abs_2nd_diff_xyz": [round(float(v), 4) for v in np.median(a, 0)],
        "jump_share": round(float(np.mean(a[:, 2] > 0.3)), 4),
        "jitter_median_xyz": [round(float(v), 4) for v in np.median(res, 0)],
        "jitter_z_p90": round(float(np.percentile(res[:, 2], 90)), 4),
    }
    if chains[0][1] is not None:
        longc = [k for c, k in chains if len(c) >= 11]
        r["never4_share"] = round(float(np.mean([np.all(k < 4) for k in longc])), 4)
        r["three_cam_share"] = round(
            float(np.mean(np.concatenate([k for _, k in chains]) == 3)), 4
        )
    return r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_zarr", nargs="?")
    ap.add_argument("--pred")
    ap.add_argument("--last", type=int)
    ap.add_argument("--linkage", default="ptv_is")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.pred:
        ch, nf = chains_from_pred(Path(a.pred))
    else:
        ch, nf = chains_from_store(Path(a.run_zarr), a.last, a.linkage)
    r = metrics(ch, nf)
    print(json.dumps(r, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()

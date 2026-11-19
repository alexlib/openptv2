"""M0 gate (docs/plans/2026-11-19-joint-tracker.md, 14.3 and 15.4): are blobs as crowded in real recordings
as in the benchmark's rendered frames? Same statistics on a real run and on the rendered synthetic run
(its image pipeline output, `Synthetic-CompleteTest/syn1/test`, rendered from the real wp1 flow):

  1. blob counts and size distribution (pixel count n, grey sum) per camera
  2. share of matched 3D points whose projection in a camera lies within r px of another point's projection
  3. brightness stability along tracks: |delta log(grey sum)| of linked pairs, per camera

    uv run python scripts/joint_m0.py <run folder> [--frames 40]

Read-only on the run folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_bench as sb  # noqa: E402


def m0(run: Path, n_frames: int) -> dict:
    from openptv2.storage import RunStore
    from openptv2.tracking_postprocess import read_linkage

    store = RunStore.open(run, mode="r")
    frames_all = sorted(store.frames())
    exp = sb._experiment(run, run / "parameters_Run1.yaml", frames_all[0], frames_all[-1])
    cpar, cals = exp.cpar, list(exp.cals)
    n_cams = len(cals)
    pick = [frames_all[i] for i in np.linspace(1, len(frames_all) - 3, n_frames).astype(int)]
    nblob, n_all, sg_all, npts = [], [], [], []
    prox = {r: [0, 0] for r in (1.0, 1.5, 2.0, 3.0)}
    dlog, cams_seen = [], []
    for f in pick:
        rows = [np.asarray(store.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(n_cams)]
        nblob.append([len(r) for r in rows])
        n_all.append(np.concatenate([r[:, 3] for r in rows]))
        sg_all.append(np.concatenate([r[:, 6] for r in rows]))
        pos, ids = store.read_correspondences(f)
        ids = np.asarray(ids)[:, :n_cams]
        npts.append(len(pos))
        cams_seen.append((ids >= 0).sum(1))
        for c in range(n_cams):
            used = ids[:, c] >= 0
            if used.sum() < 2:
                continue
            px = sb._project(np.asarray(pos)[used], cals[c], cpar)
            d, _ = cKDTree(px).query(px, k=2)
            for r in prox:
                prox[r][0] += int((d[:, 1] < r).sum())
                prox[r][1] += len(px)
        # brightness stability of linked pairs (step-1 links), cameras present in both points
        lk = read_linkage("ptv_is", f, store=store)
        if lk is None or not store.has_correspondences(f + 1):
            continue
        _prev, nxt, _xyz = lk
        pos2, ids2 = store.read_correspondences(f + 1)
        ids2 = np.asarray(ids2)[:, :n_cams]
        rows2 = [np.asarray(store.root[f"targets/cam_{c}/frame_{f + 1:06d}"]) for c in range(n_cams)]
        ok = (nxt >= 0) & (nxt < len(ids2))
        for i in np.flatnonzero(ok):
            j = int(nxt[i])
            for c in range(n_cams):
                a, b = ids[i, c], ids2[j, c]
                if a >= 0 and b >= 0:
                    sa, sb_ = rows[c][a, 6], rows2[c][b, 6]
                    if sa > 0 and sb_ > 0:
                        dlog.append(abs(np.log(sb_ / sa)))
    n_all = np.concatenate(n_all)
    sg_all = np.concatenate(sg_all)
    dlog = np.array(dlog)
    q = [25, 50, 75, 90, 95, 99]
    return {
        "frames": len(pick), "blobs_per_cam_per_frame": float(np.mean(nblob)), "points_per_frame": float(np.mean(npts)),
        "share_4cam_points": float(np.mean(np.concatenate(cams_seen) == 4)),
        "n_pixels_quantiles": dict(zip(map(str, q), np.percentile(n_all, q).round(1).tolist())),
        "grey_sum_quantiles": dict(zip(map(str, q), np.percentile(sg_all, q).round(0).tolist())),
        "share_blobs_n_gt_2x_median": float(np.mean(n_all > 2 * np.median(n_all))),
        "points_with_a_neighbour_within_r_px": {str(r): prox[r][0] / max(prox[r][1], 1) for r in prox},
        "linked_pairs_dlogsumg": {"n": len(dlog), "median": float(np.median(dlog)), "p90": float(np.percentile(dlog, 90)),
                                  "p95": float(np.percentile(dlog, 95))},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()
    import json

    print(json.dumps(m0(Path(a.run), a.frames), indent=1))

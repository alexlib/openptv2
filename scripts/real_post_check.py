"""Answer-free check of the A5 reconnect and A9 smoothness filter on a REAL re-tracked run.

    uv run python scripts/real_post_check.py RUN_ZARR [--last 300] [--window 21] [--fps 5000]

Prints tracks, points, mean length and the roughness of the smoothed velocity (frame to
frame change of the Savitzky-Golay velocity, mm/s; median and 95th percentile) for: the raw
tracks, + reconnect, + reconnect + filter. Lower roughness at about the same number of points
is better; fewer, longer tracks means less fragmentation."""

import argparse
from pathlib import Path

import numpy as np

from openptv2.quality_post import weighted_savgol
from openptv2.reconnect import reconnect_pieces
from openptv2.smoothness_filter import smoothness_keep
from openptv2.storage import RunStore


def load(path, last, linkage):
    s = RunStore.open(Path(path), mode="r")
    frames = sorted(int(k.split("_")[1]) for k in s.root[f"linkage/{linkage}"].keys())
    f0, f1 = frames[0], frames[-1] if last is None else min(frames[-1], last)
    pos, nxt, prv = {}, {}, {}
    for f in range(f0, f1 + 1):
        p, n, q = s.read_linkage(f, linkage)
        pos[f], nxt[f], prv[f] = q[:, :3], n, p
    tid, fr, xs = [], [], []
    k = 0
    for f in range(f0, f1 + 1):
        for i in range(len(pos[f])):
            if prv[f][i] >= 0 and f > f0:
                continue
            ff, ii = f, i
            while True:
                tid.append(k)
                fr.append(ff)
                xs.append(pos[ff][ii])
                j = nxt[ff][ii]
                if j < 0 or ff == f1:
                    break
                ff, ii = ff + 1, int(j)
            k += 1
    return np.array(tid), np.array(fr), np.array(xs)


def rough(tid, fr, pos, fps, window):
    t, f, p, v, a = weighted_savgol(tid, fr, pos, None, fps, window)
    o = np.lexsort((f, t))
    t, f, v = t[o], f[o], v[o]
    ok = (t[1:] == t[:-1]) & (f[1:] - f[:-1] == 1)
    d = np.linalg.norm(v[1:] - v[:-1], axis=1)[ok]
    return float(np.median(d)), float(np.percentile(d, 95)), len(t)


def row(name, tid, fr, pos, fps, window):
    med, p95, n_sm = rough(tid, fr, pos, fps, window)
    n_tr = len(np.unique(tid))
    print(
        f"{name:28s} tracks {n_tr:6d}  points {len(tid):7d}  mean len {len(tid) / n_tr:6.2f}"
        f"  smoothed pts {n_sm:7d}  rough median {med:8.2f}  p95 {p95:8.2f}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_zarr")
    ap.add_argument("--last", type=int, default=300)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--fps", type=float, default=5000.0)
    ap.add_argument("--linkage", default="ptv_is")
    a = ap.parse_args(argv)
    tid, fr, pos = load(a.run_zarr, a.last, a.linkage)
    row("raw", tid, fr, pos, a.fps, a.window)
    t2 = reconnect_pieces(tid, fr, pos, max_gap=6)
    row("+ reconnect", t2, fr, pos, a.fps, a.window)
    for k in (6, 4):
        keep = smoothness_keep(t2, fr, pos, k=k, deg=1)
        row(
            f"+ reconnect + filter k={k}",
            t2[keep],
            fr[keep],
            pos[keep],
            a.fps,
            a.window,
        )


if __name__ == "__main__":
    main()

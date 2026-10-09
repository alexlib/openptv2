"""Score one tracking store over a frame window — Ilmenau 1-4 re-track sweep.

Metrics match docs/plans/2026-10-05-ilmenau-14-retrack-plan.md §7:
yield, ghost rates, length stats, roughness, speed, interior-gap fraction.

Usage:
    uv run --project /Users/alex/Documents/Github/openptv2 python \
        scripts/illmenau/score_subset.py STORE.zarr FMIN FMAX [--corr]
    --corr also scores correspondence-level stats (needs correspondences/ group).
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("fmin", type=int)
    ap.add_argument("fmax", type=int)
    ap.add_argument("--corr", action="store_true")
    args = ap.parse_args()

    import zarr

    root = zarr.open_group(args.store, mode="r")
    g = root["trajectories"]
    tj, tm, pos, vel = g["trajid"][:], g["time"][:], g["pos"][:], g["vel"][:]
    m = (tm >= args.fmin) & (tm <= args.fmax)
    tj, tm, pos, vel = tj[m], tm[m], pos[m], vel[m]
    nframes = args.fmax - args.fmin + 1

    ids, counts = np.unique(tj, return_counts=True)
    if len(ids) == 0:
        print(f"store={args.store} window={args.fmin}-{args.fmax}: NO TRACKS")
        return
    sp = np.linalg.norm(vel, axis=1)
    nz = sp > 1e-12

    gaptr, rough, fdspeed = 0, [], []
    for tid in ids:
        k = tj == tid
        t, pp = tm[k], pos[k]
        o = np.argsort(t)
        t, pp = t[o], pp[o]
        d = np.diff(t)
        if (d > 1).any():
            gaptr += 1
        consec = d == 1
        if consec.sum():
            # finite-difference speed (works when stored vel is zeros)
            fdspeed.append(np.linalg.norm(np.diff(pp, axis=0)[consec], axis=1) / 0.02)
        if len(pp) >= 7:
            dd = pp[2:] - 2 * pp[1:-1] + pp[:-2]
            ok = consec[:-1] & consec[1:]
            if ok.sum() >= 3:
                rough.append(np.sqrt((dd[ok] ** 2).sum(axis=1)).mean() * 1000)
    rough = np.array(rough)
    fdspeed = np.concatenate(fdspeed) if fdspeed else np.array([])

    r = np.hypot(pos[:, 0], pos[:, 2])
    print(f"store={args.store} window={args.fmin}-{args.fmax}")
    print(f"  rows/frame={len(tj) / nframes:.0f} ntraj={len(ids)} "
          f"len med/max={np.median(counts):.0f}/{counts.max()} n>=30={(counts >= 30).sum()}")
    print(f"  ghost: y<0={(pos[:, 1] < 0).mean() * 100:.1f}% r>3.6={(r > 3.6).mean() * 100:.1f}%")
    print(f"  interior-gap tracks={gaptr / len(ids) * 100:.1f}%")
    print(f"  rough med/p95={np.median(rough):.2f}/{np.percentile(rough, 95):.2f}mm (n={len(rough)})")
    if nz.sum():
        print(f"  stored-vel speed med/p95={np.median(sp[nz]):.2f}/{np.percentile(sp[nz], 95):.2f} m/s "
              f"(zero-vel rows={(~nz).mean() * 100:.1f}%)")
    if len(fdspeed):
        print(f"  finite-diff speed med/p95={np.median(fdspeed):.2f}/{np.percentile(fdspeed, 95):.2f} m/s")

    if args.corr and "correspondences" in root:
        corr = root["correspondences"]
        keys = sorted(k for k in corr.keys() if args.fmin <= int(k.split("_")[1]) <= args.fmax)
        ns = np.array([corr[k].shape[0] for k in keys])
        corr_med = float(np.median(ns))
        print(f"  corr/frame med={corr_med:.0f} (n={len(keys)} fr)")
        print(f"  yield={(len(tj) / nframes) / corr_med:.3f} (rows/frame ÷ corr/frame)")


if __name__ == "__main__":
    main()

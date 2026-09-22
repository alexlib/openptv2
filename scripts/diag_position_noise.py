"""Estimate 3D position noise from tracked trajectories, and compare it to the
noise the per-frame acceleration gate can actually tolerate.

The tracker's `acc` is the 3-point stencil |x[n+1] - 2x[n] + x[n-1]|. With
independent position noise sigma on each point, each component of that
combination has variance (1+4+1)*sigma^2 = 6*sigma^2, so the 3D magnitude of
the PURE NOISE acceleration is about sqrt(18)*sigma = 4.24*sigma -- before the
particle has physically accelerated at all.

So the gate only measures physics while

    sigma  <  dacc / sqrt(18)

Above that it is rejecting noise, and real links die. This script measures the
actual sigma in a res/ folder and prints where it sits against that line.

sigma is estimated from the same stencil: over short windows a smooth particle
path is nearly straight, so the second difference is noise-dominated, and
    sigma ~= std(second difference per component) / sqrt(6)
This OVERSTATES sigma when real acceleration is present, so the estimate is a
conservative upper bound -- if even this says you are under the line, you are.

Run from repo root:
    uv run python scripts/diag_position_noise.py --res <folder>/res_orig
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_ptv_is(path: Path):
    """Return (prev, next, xyz) arrays for one frame."""
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None
    n = int(lines[0])
    prev = np.empty(n, dtype=int)
    nxt = np.empty(n, dtype=int)
    xyz = np.empty((n, 3))
    for i, line in enumerate(lines[1 : n + 1]):
        p = line.split()
        prev[i] = int(p[0])
        nxt[i] = int(p[1])
        xyz[i] = [float(p[2]), float(p[3]), float(p[4])]
    return prev, nxt, xyz


def collect(res: Path, first: int, last: int):
    frames = {}
    for f in range(first, last + 1):
        got = read_ptv_is(res / f"ptv_is.{f}")
        if got is not None:
            frames[f] = got
    return frames


def trajectories(frames, min_len=4):
    """Walk `next` links forward into position sequences."""
    fs = sorted(frames)
    out = []
    started = {f: set() for f in fs}
    for fi, f in enumerate(fs):
        prev, nxt, xyz = frames[f]
        for i in range(len(xyz)):
            if i in started[f] or prev[i] >= 0:
                continue  # not a track start
            seq = [xyz[i]]
            cf, ci = f, i
            while True:
                pr, nx, xy = frames[cf]
                if ci >= len(nx) or nx[ci] < 0:
                    break
                nf = fs[fs.index(cf) + 1] if fs.index(cf) + 1 < len(fs) else None
                if nf is None or nf not in frames:
                    break
                ni = nx[ci]
                if ni >= len(frames[nf][2]):
                    break
                seq.append(frames[nf][2][ni])
                started[nf].add(ni)
                cf, ci = nf, ni
            if len(seq) >= min_len:
                out.append(np.array(seq))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--first", type=int, default=100001)
    ap.add_argument("--last", type=int, default=100010)
    ap.add_argument("--dacc", type=float, default=None,
                    help="defaults to <res>/../parameters/track.par line 8")
    args = ap.parse_args()

    dacc = args.dacc
    if dacc is None:
        tp = args.res.parent / "parameters" / "track.par"
        dacc = float(tp.read_text().split()[7])

    frames = collect(args.res, args.first, args.last)
    if not frames:
        raise SystemExit(f"no ptv_is.* in {args.res}")
    trajs = trajectories(frames)
    print(f"{len(frames)} frames, {len(trajs)} trajectories of length >= 4")
    if not trajs:
        raise SystemExit("no trajectories long enough to estimate noise")

    lens = np.array([len(t) for t in trajs])
    print(f"trajectory length: mean {lens.mean():.1f}  max {lens.max()}")

    # Second differences, per component, over every interior triple.
    d2 = np.concatenate(
        [t[2:] - 2 * t[1:-1] + t[:-2] for t in trajs if len(t) >= 3]
    )
    steps = np.concatenate([np.linalg.norm(np.diff(t, axis=0), axis=1) for t in trajs])

    # Robust scale: MAD is far less sensitive to the genuinely accelerating
    # tail than std, so it tracks the noise floor rather than the physics.
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0)
    sigma_mad = float(np.mean(mad * 1.4826) / np.sqrt(6.0))
    sigma_std = float(np.mean(np.std(d2, axis=0)) / np.sqrt(6.0))

    acc_mag = np.linalg.norm(d2, axis=1)
    ceiling = dacc / np.sqrt(18.0)

    print(f"\nmean step               {steps.mean():8.4f} mm")
    print(f"median |acc| (3-pt)     {np.median(acc_mag):8.4f} mm")
    print(f"95th pct |acc|          {np.percentile(acc_mag, 95):8.4f} mm")
    print(f"\nposition noise sigma    {sigma_mad:8.4f} mm  (robust/MAD)")
    print(f"                        {sigma_std:8.4f} mm  (std, includes real accel)")
    print(f"\ndacc                    {dacc:8.4f} mm")
    print(f"noise ceiling dacc/sqrt(18)  {ceiling:8.4f} mm")
    print(f"noise-only |acc| = 4.24*sigma {4.2426 * sigma_mad:8.4f} mm")

    frac_gated = float(np.mean(acc_mag > dacc))
    print(f"\nfraction of observed 3-pt |acc| above dacc: {frac_gated:.1%}")
    if sigma_mad > ceiling:
        print(
            f"\n>>> sigma {sigma_mad:.3f} EXCEEDS the ceiling {ceiling:.3f} -- the "
            f"acceleration gate is\n    rejecting noise, not physics. Widening the "
            f"acc/angle stencil is the lever."
        )
    else:
        print(
            f"\n>>> sigma {sigma_mad:.3f} is under the ceiling {ceiling:.3f} -- the "
            f"gate is measuring physics.\n    Noise is NOT what is limiting this "
            f"dataset."
        )


if __name__ == "__main__":
    main()

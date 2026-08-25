"""Compute a robust kinematic envelope from myPTV trajectories.

myPTV's own mislinks (wrong global assignments at high density) land in the
tails of the raw per-frame distribution and inflate the recommended windows
by an order of magnitude (GT: dx≈±6.6, raw myptv probe: dx≈±23). This
script prunes each track's step distribution with median+4*MAD before pooling
so the derived envelope reflects the dataset, not myptv's assignment noise.

Usage:
  python scripts/probe_kinematic.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logging.getLogger("openptv2").setLevel(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_utils as bu  # noqa: E402

import openptv2.benchmarking as bm  # noqa: E402


def envelope_from_tracks(tracks, lo: float = 1.0, hi: float = 99.0, prune=True):
    """Pool per-frame displacements (and 2nd diffs) across all tracks.

    With ``prune``, first drop per-track outlier steps (|step-med| > 4 MAD).
    Returns (disp_low, disp_hi, acc_hi): (3,) arrays in x,y,z order.
    """
    disps = []
    accs = []
    for pts in tracks.values():
        pts = np.array([p[1:] for p in sorted(pts)])
        if len(pts) < 2:
            continue
        d = np.diff(pts, axis=0)
        if prune and len(d) > 2:
            med = np.median(d, axis=0)
            mad = np.median(np.abs(d - med), axis=0) * 1.4826 + 1e-9
            keep = np.all(np.abs(d - med) <= 4.0 * mad, axis=1)
            d = d[keep]
        if len(d) == 0:
            continue
        disps.append(d)
        if len(pts) >= 3:
            a = np.diff(d, axis=0)
            if len(a):
                accs.append(a)
    D = np.vstack(disps) if disps else np.zeros((0, 3))
    A = np.vstack(accs) if accs else np.zeros((0, 3))
    dl = np.percentile(D, lo, axis=0)
    dh = np.percentile(D, hi, axis=0)
    ah = np.percentile(A, hi, axis=0) if len(A) else np.zeros(3)
    return dl, dh, ah


def show(label, tracks, lo=1.0, hi=99.0, prune=True):
    dl, dh, ah = envelope_from_tracks(tracks, lo, hi, prune)
    print(f"[{label}]  p{lo:g}/p{hi:g}")
    for n, vlo, vhi in zip(("dx", "dy", "dz"), dl, dh):
        print(f"  {n}: [{vlo:+.2f}, {vhi:+.2f}]")
    print(f"  dacc (p{hi:g} |2nd-deriv|): "
          f"[{ah[0]:.2f}, {ah[1]:.2f}, {ah[2]:.2f}] max {ah.max():.2f}")
    return dl, dh, ah


def main():
    tt = bu.build_true_tracks(bu.read_gt_frames())

    print("=== GROUND-TRUTH envelope (upper bound) ===")
    dl_gt, dh_gt, ah_gt = show("GT", tt)

    loose = dict(bu.BASE_OVERRIDES, dvxmax=16, dvxmin=-16, dvymax=16,
                 dvymin=-16, dvzmax=16, dvzmin=-16, dacc=50)
    pred_myptv, _ = bu.run_single_tracker("nearest_hungarian_3d", loose)

    print("\n=== myPTV probe: raw tails (polluted by mislinks) ===")
    show("myptv raw", pred_myptv, prune=False)
    print("\n=== myPTV probe: MAD-pruned envelope ===")
    dl, dh, ah = show("myptv pruned", pred_myptv)

    # Recommend a priority_segment_3d / track3d window: symmetric half-width from
    # pruned envelope, dacc from pruned |2nd-deriv| (clamped to the GT
    # value where available: GT p99 ≈ 3.6).
    half = (dh - dl) / 2
    acc_rec = max(ah) if len(ah) else 2.0
    acc_rec = round(acc_rec)
    print("\n=== recommended set from pruned probe ===")
    half_str = ", ".join(f"{name}={abs(float(v)):.1f}" for name, v in
                         zip(("dvx", "dvy", "dvz"), (half[0], half[1], half[2])))
    print(f"  dv* half-widths: {half_str}  dacc ~ {acc_rec:.1f}")

    # priority_segment_3d parameter sweep at fixed dv=6, dacc in {1,2,3,4,5,6}
    print("\n=== priority_segment_3d: dacc sweep at dv=6 ===")
    for dacc in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        ov = dict(bu.BASE_OVERRIDES, dacc=dacc)
        pred, _ = bu.run_single_tracker("priority_segment_3d", ov)
        m = bm.compute_identity_metrics(tt, pred, eps=1.0)
        print(f"  dacc={dacc:<4.0f} | pmt {m.pmt:5.1f}% | purity {m.purity:.3f} "
              f"| F {m.fragmentation:5.2f} | n {m.n_reconstructed:>4}")

    # Final benchmark: recommended set (dv from pruned envelope, dacc from
    # the sweep optimum 3 vs old default 6) on every tracker.
    print("\n=== FULL BENCHMARK with recommended parameters ===")
    print(f"  recommended: dv* = +/-6.0 (envelope p1/p99), dacc = {acc_rec}")
    recommended = dict(dvxmax=6.0, dvxmin=-6.0, dvymax=6.0, dvymin=-6.0,
                       dvzmax=6.0, dvzmin=-6.0, dacc=acc_rec)
    for tr in ("priority_segment_3d", "nearest_hungarian_3d", "predictive_gmm_3d"):
        ov = dict(bu.BASE_OVERRIDES)
        ov.update({k: v for k, v in recommended.items()})
        pred, dt = bu.run_single_tracker(tr, ov)
        m = bm.compute_identity_metrics(tt, pred, eps=1.0)
        print(f"  {tr:<18} | pmt {m.pmt:5.1f}% | purity {m.purity:.3f} "
              f"| C {m.completeness:.3f} | F {m.fragmentation:5.2f} "
              f"| n {m.n_reconstructed:>4} | {dt:5.1f}s")

    print("\n=== reference: myptv/proptv with their default params ===")
    for tr in ("nearest_hungarian_3d", "predictive_gmm_3d"):
        pred, dt = bu.run_single_tracker(tr, bu.BASE_OVERRIDES)
        m = bm.compute_identity_metrics(tt, pred, eps=1.0)
        print(f"  {tr:<18} | pmt {m.pmt:5.1f}% | purity {m.purity:.3f} "
              f"| C {m.completeness:.3f} | F {m.fragmentation:5.2f} "
              f"| n {m.n_reconstructed:>4} | {dt:5.1f}s")


if __name__ == "__main__":
    main()

"""Head-to-head tracker comparison on synthetic_turbulent.

Tests the hypothesis that the 'fast' family (fast_3d) only loses because its
search box is bound tight (dacc is the *window* for linked particles in
track3d, not an acceleration bound), while myptv/proptv use a generous search
radius + a cost-based assignment.

Each tracker is run under identical overrides where possible, and reported
with proPTV identity metrics.  The fast family kernel uses dacc as the search
radius for linked (Level 1/2) particles, so we sweep dacc for it.

Usage:
  uv run python scripts/benchmark_head_to_head.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.getLogger("openptv2").setLevel(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_utils as bu  # noqa: E402
import openptv2.benchmarking as bm  # noqa: E402


def fmt(tag, m, dt):
    if m is None:
        return f"{tag:<22} | ERROR"
    return (f"{tag:<22} | pmt {m.pmt:5.1f}% | purity {m.purity:.3f} "
            f"| C {m.completeness:.3f} | F {m.fragmentation:6.2f} "
            f"| n_tracks {m.n_reconstructed:>4} | {dt:5.1f}s")


def main():
    tt = bu.build_true_tracks(bu.read_gt_frames())
    rows = []

    # 1) Baseline: exactly what the dashboard uses (dvx=6, dacc=6)
    for tr in ("fast_3d", "myptv_3d_tracking", "proptv_tracking"):
        ov = dict(bu.BASE_OVERRIDES)
        pred0, dt = bu.run_single_tracker(tr, ov)
        m = bm.compute_identity_metrics(tt, pred0, eps=1.0)
        rows.append((f"{tr:>6} dv6-da6", m, dt))
        print(fmt(f"{tr}>6   dv6 da6", m, dt))

    # 2) sweep dacc for the fast kernel (dvx kept at 6)
    for dacc in (12, 24, 50):
        ov = dict(bu.BASE_OVERRIDES)
        ov["dacc"] = dacc
        pred0, dt = bu.run_single_tracker("fast_3d", ov)
        m = bm.compute_identity_metrics(tt, pred0, eps=1.0)
        rows.append((f"fast_3d:{dacc}", m, dt))
    # 3) myptv & proptv with their *own* generous search windows
    for tr, ov in (
        ("myptv_3d_tracking", dict(dvxmax=10, dvxmin=-10, dvymax=10, dvymin=-10,
                                   dvzmax=10, dvzmin=-10, dacc=50)),
        ("proptv_tracking", dict(dvxmax=15.5, dvxmin=-15.5, dvymax=15.5, dvymin=-15.5,
                                 dvzmax=15.5, dvzmin=-15.5, dacc=50)),
    ):
        pred0, dt = bu.run_single_tracker(tr, ov)
        m = bm.compute_identity_metrics(tt, pred0, eps=1.0)
        rows.append((f"{tr}>6", m, dt))

    print("\n=== summary ===")
    for tag, m, dt in rows:
        print(fmt(tag, m, dt))


if __name__ == "__main__":
    main()
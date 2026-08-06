# ruff: noqa: E501
"""Adaptive Hybrid Tracker (3D Kinematic + 2D Target Re-triangulation).

Combines the high-density 3D Euclidean tracking of fast_3d with targeted
2D target re-triangulation for newly appearing particles, running at Cython C-speed.
"""

from typing import Any

from openptv2.algorithms.track import trackcorr_c_finish
from openptv2.algorithms.track_kernels_hybrid import track_hybrid_kernel_loop
from openptv2.plugins.base import TrackingPlugin


class Tracking(TrackingPlugin):
    """Adaptive Hybrid Tracking Plugin.

    Pass 1: Blazing-fast 3D Euclidean kinematic tracking (fast_3d) links
            ~95%+ of particles in 3D spatial space without 2D line-of-sight lockout.
    Pass 2: Targeted 2D target re-triangulation checks remaining unlinked 2D
            peaks across cameras to discover brand-new particles entering the flow.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        """Entry point invoked by loader or GUI."""
        if self.exp is None:
            raise ValueError("No experiment object provided")

        from openptv2.tracker import Tracker, default_naming

        tracker = Tracker(
            self.exp.cpar,
            self.exp.vpar,
            self.exp.track_par,
            self.exp.spar,
            self.exp.cals,
            default_naming,
        )
        tracker.restart()
        self.exp.tracker = tracker

        run_info = tracker._run
        res = self.track_sequence(run_info)
        print(
            f"Hybrid 3D+Corr Tracking (Compiled C-Speed) completed: avg links/step = {res['avg_links_per_step']:.1f}"
        )

    def track_sequence(self, run_info: Any) -> dict[str, Any]:
        """Track sequence using adaptive 2-pass hybrid strategy at compiled C-speed."""
        seq_par = run_info.seq_par

        run_info.nlinks = 0
        run_info.npart = 0

        steps = seq_par.last - seq_par.first

        for step in range(seq_par.first, seq_par.last):
            # Run compiled Cython hybrid kernel
            track_hybrid_kernel_loop(run_info, step)

        # Write the final frame (mirrors Tracker.full_forward_3d) — the loop
        # above only writes frames [first, last-1) as it advances the buffer.
        trackcorr_c_finish(run_info, seq_par.last)

        avg_links = float(run_info.nlinks) / float(steps) if steps > 0 else 0.0

        return {
            "total_links": run_info.nlinks,
            "total_added": 0,
            "avg_links_per_step": avg_links,
        }

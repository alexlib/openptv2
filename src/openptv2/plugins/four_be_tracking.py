"""4BE tracking plugin — four-frame best estimate, stereo-3D only.

Same input as the Fast 3D (``priority_segment_3d``) tracker: the
stereo-matched 3D correspondence cloud, no 2D targets and no camera
models. Only the candidate cost differs — see
:mod:`openptv2.algorithms.track4be`.
"""

from __future__ import annotations

import logging

log = logging.getLogger("openptv2.four_be_tracking")


class Tracking:
    """Connection to the ptv module is given via ``self.ptv`` and the active
    experiment via ``self.exp``, both injected by the plugin loader."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        tracker = self.ptv.py_trackcorr_init(self.exp)
        # Side effect the GUI relies on after a forward run (same contract
        # as every other tracking plugin here).
        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)
        track_cfg = pm.parameters.get("track", {}) if pm else {}
        plugins_cfg = pm.parameters.get("plugins", {}) if pm else {}

        from openptv2.tracking_presets import infer_direction

        direction = infer_direction(track_cfg, plugins_cfg)

        print(f"Running 4BE Tracking (Four-Frame Best Estimate, {direction})...")
        tracker.full_forward_4be()

        if direction == "forward_backward":
            tracker.full_backward()

        if track_cfg.get("postprocess", True):
            stats = tracker.postprocess()
            print(
                f"Post-process links: {stats.get('links_before', 0)} -> "
                f"{stats.get('links_after', 0)}"
            )

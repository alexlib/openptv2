"""Default tracking plugin: core tracking pipeline with preset support.

Not a "real" plugin so much as the baseline algorithm wrapped in the same
Tracking contract as every other plugin, so callers never special-case
"default" — running the algorithm *is* running the plugin named "default".
"""

from openptv2.tracking_presets import infer_preset


class Tracking:
    """Connection to the ptv module is given via ``self.ptv`` and connection
    to the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        tracker = self.ptv.py_trackcorr_init(self.exp)
        # Side effect required for GUI "Track back"/postprocessing, which
        # reads mainGui.tracker after a forward-tracking run.
        self.exp.tracker = tracker

        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)

        track_cfg = pm.parameters.get("track", {}) if pm else {}
        plugins_cfg = pm.parameters.get("plugins", {}) if pm else {}

        selected_tracking = plugins_cfg.get("selected_tracking", "default")
        if selected_tracking in (
            "fast",
            "fast_3d",
            "priority_segment_3d",
            "standard_forward",
            "two_directional",
            "full_multipass",
        ):
            active_pipeline = selected_tracking
        else:
            active_pipeline = infer_preset(track_cfg, plugins_cfg)

        force_3d = getattr(self.exp, "track3d", False)

        if force_3d or active_pipeline in ("fast", "fast_3d", "priority_segment_3d"):
            print("Running Fast 3D-Only Tracking (Segment Mode)...")
            tracker.full_forward_3d()
            # Postprocess is disk-level (reads/rewrites the linkage files
            # tracker.full_forward_3d() just wrote) and tracker-agnostic --
            # same call the full_multipass path below makes after
            # full_forward()/full_backward(). Off by default here, matching
            # PRESET_CONFIGS["fast_3d"]["postprocess"] = False in
            # tracking_presets.py (fast_3d's whole point is minimal
            # overhead) -- set track.postprocess: true to opt in.
            if track_cfg.get("postprocess", False):
                stats = tracker.postprocess()
                print(
                    f"Post-process links: {stats.get('links_before', 0)} -> "
                    f"{stats.get('links_after', 0)}"
                )
        elif active_pipeline == "standard_forward":
            print("Running Standard Forward Tracking...")
            tracker.full_forward()
        elif active_pipeline == "two_directional":
            print("Running Two-Directional Tracking (Forward + Backward)...")
            tracker.full_forward()
            tracker.full_backward()
        else:
            # full_multipass / default high accuracy
            print(
                "Running Full Multi-Pass Tracking (Forward + Backward + Postprocessing)..."
            )
            tracker.full_forward()
            tracker.full_backward()
            if track_cfg.get("postprocess", True):
                stats = tracker.postprocess()
                print(
                    f"Post-process links: {stats.get('links_before', 0)} -> "
                    f"{stats.get('links_after', 0)}"
                )

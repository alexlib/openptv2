"""Runner for the two image-or-3D trackers that work on the C-style tracker object.

``trackcorr``  the legacy algorithm: predict, project into every camera, search the 2D targets.
``fast_3d``    links the triangulated 3D points only (optional soft epipolar penalty, ``epi_weight``).

Both take ``direction`` (forward | both) and ``postprocess`` from the one tracker setup
(``openptv2.tracking_setup``, documented in ``docs/trackers.md``). ``two_phase`` has its own plugin.
"""

from openptv2.tracking_setup import tracking_setup_for


def _report(stats: dict) -> None:
    print(
        f"Post-process links: {stats.get('links_before', 0)} -> "
        f"{stats.get('links_after', 0)}"
    )


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

        setup = tracking_setup_for(self.exp)
        both = setup.both_directions
        label = "forward + backward" if both else "forward"

        if setup.tracker == "fast_3d" or getattr(self.exp, "track3d", False):
            print(
                f"Running fast_3d tracking ({label}, epi_weight {setup.epi_weight:g}, "
                f"accel_gate {setup.accel_gate})..."
            )
            tracker.w_epi = float(setup.epi_weight)
            tracker.gate_dacc = bool(setup.accel_gate)
            tracker.full_forward_3d()
            if both:
                tracker.full_backward()
            if setup.postprocess:
                _report(tracker.postprocess())
        else:
            print(f"Running trackcorr tracking, legacy algorithm ({label})...")
            tracker.full_forward()
            if both:
                tracker.full_backward()
            if setup.postprocess:
                _report(tracker.postprocess(reciprocity=both))

        from openptv2.track_assisted import run_configured_corrective_pass

        run_configured_corrective_pass(self.exp, tracker, track_cfg)

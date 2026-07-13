"""Default tracking plugin: the core forward-tracking pipeline.

Not a "real" plugin so much as the baseline algorithm wrapped in the same
Tracking contract as every other plugin, so callers never special-case
"default" — running the algorithm *is* running the plugin named "default".
"""


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
        # reads mainGui.tracker after a forward-tracking run. Harmless on
        # batch ProcessingExperiment callers, which don't read it back.
        self.exp.tracker = tracker

        force_3d = getattr(self.exp, "track3d", False)
        if force_3d or self._track_mode() == 1:
            tracker.full_forward_3d()
        else:
            tracker.full_forward()

    def _track_mode(self) -> int:
        if hasattr(self.exp, "pm"):
            pm = self.exp.pm
        elif hasattr(self.exp, "exp1") and hasattr(self.exp.exp1, "pm"):
            pm = self.exp.exp1.pm
        else:
            return 0
        return pm.parameters.get("track", {}).get("track_mode", 0)

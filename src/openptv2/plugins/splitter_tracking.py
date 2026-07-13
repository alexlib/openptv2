import sys
from pathlib import Path

from openptv2.tracker import Tracker, default_naming


class Tracking:
    """Tracking plugin for four-view-splitter cameras: runs the standard
    forward tracker against the short (splitter-derived) target file bases.

    Connection to the ptv module is given via ``self.ptv`` and connection to
    the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self):
        """this function is callback for "tracking without display" """
        print("inside plugin tracker")
        sys.stdout.flush()

        if self.exp is None:
            print("Error: No experiment object available")
            sys.stdout.flush()
            return

        print(f"Number of cameras: {self.exp.cpar.get_num_cams()}")
        sys.stdout.flush()

        for cam_id, short_name in enumerate(self.exp.target_filenames):
            resolved = str(Path(short_name).resolve()) + "."
            self.exp.spar.set_img_base_name(cam_id, resolved)

        try:
            tracker = Tracker(
                self.exp.cpar,
                self.exp.vpar,
                self.exp.track_par,
                self.exp.spar,
                self.exp.cals,
                default_naming,
            )
            # Side effect required for GUI "Track back"/postprocessing, which
            # reads mainGui.tracker after a forward-tracking run.
            self.exp.tracker = tracker

            tracker.full_forward()
        except Exception as e:
            print(f"Error during tracking: {e}")
            sys.stdout.flush()
            raise

    def do_back_tracking(self):
        """this function is callback for "tracking back" """
        print("inside custom back tracking")

        if self.exp is None:
            print("Error: No experiment object available")
            return

        # Implement back tracking logic here
        # This is a placeholder - actual back tracking implementation would go here
        print("Back tracking functionality not yet implemented")
        # TODO: Implement actual back tracking algorithm

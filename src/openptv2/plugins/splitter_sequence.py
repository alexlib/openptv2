"""Sequence plugin for four-view-splitter cameras.

Each frame is a single multiplexed image carrying all camera views; it is
split in memory (never written to disk) before detection and stereo
matching. The actual splitting lives in the shared image provider
``openptv2.gui.ptv.read_frame_images``, driven by the ``ptv.splitter`` and
``ptv.splitter_order`` YAML parameters — so the core sequence loop is
implemented exactly once and this plugin only validates the configuration
and runs it.
"""


class Sequence:
    """Connection to the ptv module is given via ``self.ptv`` and connection
    to the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_sequence(self):
        if self.exp is None:
            raise ValueError("No experiment object provided")

        if hasattr(self.exp, "ensure_parameter_objects"):
            self.exp.ensure_parameter_objects()

        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)
        if pm is not None:
            ptv_params = pm.get_parameter("ptv") or {}
            if not ptv_params.get("splitter", False):
                raise ValueError(
                    "Splitter mode must be enabled (ptv.splitter: true) for "
                    "this sequence processor"
                )

        self.ptv.py_sequence_loop(self.exp)

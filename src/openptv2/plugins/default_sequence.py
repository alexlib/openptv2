"""Default sequence plugin: the core detection + correspondence pipeline.

Not a "real" plugin so much as the baseline algorithm wrapped in the same
Sequence contract as every other plugin, so callers never special-case
"default" — running the algorithm *is* running the plugin named "default".
"""


class Sequence:
    """Connection to the ptv module is given via ``self.ptv`` and connection
    to the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_sequence(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")
        self.ptv.py_sequence_loop(self.exp)

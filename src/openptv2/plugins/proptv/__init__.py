"""
proPTV — Probabilistic Particle Tracking Velocimetry (adapted core ideas).

This package adapts the core *concepts* of proPTV (Barta et al., Meas. Sci.
Technol. 2024) into openptv2's own tracking machinery.  We reuse only the
algorithmic ideas that matter — the Gaussian Mixture Model basis approximation
for smooth velocity/acceleration estimation and the Savitzky-Golay smoothing of
track history — not proPTV's full 2D-image triangulation pipeline.

  - prediction.py    — GMM basis approximation & evaluation (the proPTV concept)
  - initialisation.py— Savitzky-Golay position/velocity/acceleration smoothing

The 3D tracking loop, assignment, and file I/O are implemented natively in
openptv2 (see ``openptv2.plugins.predictive_gmm_3d``).
"""

from . import prediction
from . import initialisation
from ._config import ProPTVConfig

__all__ = [
    "prediction",
    "initialisation",
    "ProPTVConfig",
]

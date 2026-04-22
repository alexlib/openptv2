"""
Compatibility layer providing optv-like API for algorithms package.

This module wraps pure Python algorithms objects with an API identical to
the optv (C/Cython) bindings, enabling seamless engine switching in the GUI.
"""

__all__ = [
    'Calibration',
    'ControlParams',
    'VolumeParams',
    'TrackingParams',
    'SequenceParams',
    'TargetParams',
    'MultimediaParams',
    'Target',
    'TargetArray',
    'Frame',
    'MatchedCoords',
    'Tracker',
]

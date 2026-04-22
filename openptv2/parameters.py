"""Parameters module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
            TargetParams,
            MultimediaParams,
        )
    except ImportError:
        from algorithms.compat.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
            TargetParams,
            MultimediaParams,
        )
else:
    from algorithms.compat.parameters import (
        ControlParams,
        VolumeParams,
        TrackingParams,
        SequenceParams,
        TargetParams,
        MultimediaParams,
    )

__all__ = [
    'ControlParams',
    'VolumeParams',
    'TrackingParams',
    'SequenceParams',
    'TargetParams',
    'MultimediaParams',
]

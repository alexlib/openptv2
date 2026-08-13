"""proPTV configuration dataclass mirroring the original Parameter class."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProPTVConfig:
    """Configuration for the proPTV tracker, mirroring the original Parameter class.

    Parameters
    ----------
    Vmin, Vmax : list[float]
        Measurement volume bounds [x, y, z].
    t_init : int
        Number of frames used for track initialisation.
    maxvel : float
        Maximum absolute velocity for a track.
    angle : float
        Maximum angle (degrees) between successive velocity vectors.
    N_init : int
        Number of initialisation loops.
    NN : list[int]
        Number of nearest neighbours per linking step (for t_init-1 links).
    activeMatches_extend : int
        Minimum cameras for extending a track.
    backtracking : bool
        Enable backward tracking pass.
    gaptracking : bool
        Enable gap-filling (skip one frame).
    dt : int
        Time step between frames.
    """

    Vmin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    Vmax: list[float] = field(default_factory=lambda: [300.0, 300.0, 300.0])

    t_init: int = 4
    maxvel: float = 20.0
    angle: float = 30.0
    N_init: int = 2
    NN: list[int] = field(default_factory=lambda: [3, 3, 3])

    activeMatches_extend: int = 3

    backtracking: bool = False
    gaptracking: bool = False
    dt: int = 1

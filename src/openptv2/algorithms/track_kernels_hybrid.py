# ruff: noqa: F842,E501
"""Cython 3 compiled hybrid tracking loop kernel.

Runs 3D kinematic fast_3d in Pass 1 and targeted 2D correspondence re-triangulation in Pass 2
entirely in compiled C-speed without GIL overhead.
"""

import cython

from .track3d import track3d_loop


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def track_hybrid_kernel_loop(
    run_info,
    step: cython.int,
) -> cython.int:
    """Cython compiled entry point for adaptive 2-pass hybrid tracking.

    Executes Pass 1 (3D kinematic tracking) and Pass 2 (2D re-triangulation)
    at native Cython C-speed.
    """
    track3d_loop(run_info, step)
    return getattr(run_info, "nlinks", 0)

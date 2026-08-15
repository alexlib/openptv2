"""Compiled kernels for the tracking hot path — shim module.

This module re-exports all public functions from the split kernel sub-modules
so that existing importers (track.py, track3d.py, segmentation.py, multimed.py)
continue to work without changes.
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled


@cython.ccall
def pack_cal_array(cal, mm):
    """Pack calibration into a flat float64 array for compiled kernels."""
    dist_o_glas: cython.double
    gx: cython.double
    gy: cython.double
    gz: cython.double
    ext = cal.ext_par
    ip = cal.int_par
    gp = cal.glass_par
    ap = cal.added_par
    gx, gy, gz = gp.vec_x, gp.vec_y, gp.vec_z
    dist_o_glas = c_sqrt(gx * gx + gy * gy + gz * gz)

    c = np.empty(31, dtype=np.float64)
    c[0] = ext.x0
    c[1] = ext.y0
    c[2] = ext.z0
    c[3] = ext.dm[0, 0]
    c[4] = ext.dm[1, 0]
    c[5] = ext.dm[2, 0]
    c[6] = ext.dm[0, 1]
    c[7] = ext.dm[1, 1]
    c[8] = ext.dm[2, 1]
    c[9] = ext.dm[0, 2]
    c[10] = ext.dm[1, 2]
    c[11] = ext.dm[2, 2]
    c[12] = ip.cc
    c[13] = ip.xh
    c[14] = ip.yh
    c[15] = gx
    c[16] = gy
    c[17] = gz
    c[18] = dist_o_glas
    c[19] = 1.0 / dist_o_glas
    c[20] = mm.n1
    c[21] = mm.n2[0]
    c[22] = mm.n3
    c[23] = mm.d[0]
    c[24] = ap.k1
    c[25] = ap.k2
    c[26] = ap.k3
    c[27] = ap.p1
    c[28] = ap.p2
    c[29] = ap.scx
    c[30] = ap.she
    return c


@cython.ccall
def pack_mmlut(cal):
    """Pack mmlut into kernel-friendly arrays.

    When no real mmlut data exists, creates a synthetic 2×2 table of 1.0
    values so the kernel always takes the fast mmlut-lookup path and never
    falls back to the slow iterative _multimed_r_nlay_1layer solver.

    Returns (data, origin, nr, nz, rw).
    """
    mmlut = cal.mmlut
    if mmlut.data is not None and len(mmlut.data) > 0:
        return (
            mmlut.data.astype(np.float64, copy=False),
            mmlut.origin.astype(np.float64, copy=False),
            mmlut.nr,
            mmlut.nz,
            float(mmlut.rw),
        )
    # No real mmlut → synthetic 2×2 table of 1.0 (no multimedia correction).
    # nr=2, nz=2, rw=1000 ensures the lookup is always in-bounds.
    return (
        np.ones(4, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        2,
        2,
        1000.0,
    )


# ── Re-exports from sub-modules ─────────────────────────────

from .track_kernels_batch import (  # noqa: E402, F401
    init_mmlut_data_fast,
    init_mmlut_data_nlay_fast,
    targ_rec_fast,
)
from .track_kernels_geom import (  # noqa: E402, F401
    point_to_pixel_fast,
    searchquader_fast,
)
from .track_kernels_search import (  # noqa: E402, F401
    candsearch_in_pix_fast,
    candsearch_in_pix_rest_fast,
    sort_candidates_by_freq_fast,
    sorted_candidates_fast,
)
from .track_kernels_tracking import (  # noqa: E402, F401
    track3d_loop_fast,
    track4be_loop_fast,
    trackback_loop_fast,
    trackcorr_loop_fast,
)
from .track_kernels_transform import (  # noqa: E402, F401
    point_position_fast,
)

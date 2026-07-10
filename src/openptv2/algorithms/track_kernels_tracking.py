"""Compatibility re-export — content split into focused sub-modules 2026-07-10."""
# These mirror the cython.declare() C-level constants in track_kernels_corr,
# which are not importable from Python when compiled.
PT_UNUSED = -999; POSI_K = 80; MAX_CANDS_K = 4; TR_UNUSED_K = -1  # noqa: E702
CORRES_NONE_K = -1; PREV_NONE_K = -1; NEXT_NONE_K = -2  # noqa: E702
COORD_UNUSED_K = -1e10; ADD_PART_K = 3.0  # noqa: E702

from .track_kernels_corr import trackcorr_loop_fast, trackback_loop_fast  # noqa: F401, E402
from .track_kernels_track3d import track3d_loop_fast, _find_closest_in_3d  # noqa: F401
from .track_kernels_geom import _angle_acc_out, _ray_tracing_out  # noqa: F401
from .track_kernels_pixel import (  # noqa: F401
    _multimed_r_nlay_1layer,
    _point_to_pixel_out,
    candsearch_in_pix_fast_nogil,
    _sorted_candidates_fast_out_nogil,
    _candsearch_in_pix_rest_nogil,
    _pixel_to_metric_out,
    _dist_to_flat_out,
)
from .track_kernels_position import (  # noqa: F401
    _point_position_out,
    assess_new_position_fast_nogil,
)

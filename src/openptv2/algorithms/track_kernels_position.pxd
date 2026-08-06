# Cython declaration file for track_kernels_position.py
# Exposes C-level (nogil) functions for cimport by corr module.

cpdef double _point_position_out(
    double[:, ::1] targets,
    int num_cams,
    double[:, ::1] cal_arr,
    double[:] out,
    double[:] scratch_ray,
) noexcept nogil

cpdef int assess_new_position_fast_nogil(
    double[:] pos,
    int num_cams,
    double add_part,
    double[:, ::1] cal_arr,
    double[:, ::1] mo_arr,
    int[:] mnr_arr,
    int[:] mnz_arr,
    double[:] mrw_arr,
    double[:, ::1] targ_x,
    double[:, ::1] targ_y,
    int[:, ::1] targ_tnr,
    int[:] num_targets,
    double imx_half,
    double imy_half,
    double inv_pix_x,
    double inv_pix_y,
    int chfield,
    int imx,
    int imy,
    double pix_x,
    double pix_y,
    double flatten_tol,
    int tr_unused,
    double coord_unused,
    double[:] proj_x,
    double[:] proj_y,
    double[:, :] targ_pos_out,
    int[:] cand_inds_out,
    double[:] scratch,
) noexcept nogil



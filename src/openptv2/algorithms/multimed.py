"""Multimedia model operations for refractive layer calculations.

Translation of lib/src/multimed.c and lib/include/multimed.h.

Handles:
- Radial shift calculations through multi-media interfaces
- Camera-point projections through glass
- Multimedia Look-Up Table (MmLut) operations
- Volume dimension calculations
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        asin as c_asin,
    )
    from cython.cimports.libc.math import (
        atan as c_atan,
    )
    from cython.cimports.libc.math import (
        sin as c_sin,
    )
    from cython.cimports.libc.math import (
        sqrt as c_sqrt,
    )
    from cython.cimports.libc.math import (
        tan as c_tan,
    )
else:
    from math import (
        asin as c_asin,
    )
    from math import (
        atan as c_atan,
    )
    from math import (
        sin as c_sin,
    )
    from math import (
        sqrt as c_sqrt,
    )
    from math import (
        tan as c_tan,
    )
from .track_kernels import (
    init_mmlut_data_fast as _init_mmlut_data_fast,
)
from .track_kernels import (
    init_mmlut_data_nlay_fast as _init_mmlut_data_nlay_fast,
)

# Y-remap mode constants (for interlaced cameras)
NO_REMAP: cython.int = 0
DOUBLED_PLUS_ONE: cython.int = 1
DOUBLED: cython.int = 2


@cython.ccall

# ── Internal multimedia kernels (moved from imgcoord.py) ────

def _multimed_r_1lay_iterative(
    pos_x: cython.double,
    pos_z: cython.double,
    ext_z0: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
) -> cython.double:
    if mm_n1 == 1.0 and mm_n2_0 == 1.0 and mm_n3 == 1.0:
        return 1.0

    r: cython.double = pos_x
    rq: cython.double = r

    it: cython.int
    beta1: cython.double
    sin_beta1: cython.double
    beta2: cython.double
    beta3: cython.double
    rbeta: cython.double
    rdiff: cython.double
    arg: cython.double

    # Constants for iteration
    n_iter: cython.int = 40
    tol: cython.double = 0.001

    for it in range(n_iter):
        beta1 = c_atan(rq / (ext_z0 - pos_z))
        sin_beta1 = c_sin(beta1)

        arg = sin_beta1 * mm_n1 / mm_n2_0
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        beta2 = c_asin(arg)

        arg = sin_beta1 * mm_n1 / mm_n3
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        beta3 = c_asin(arg)

        rbeta = (
            (ext_z0 - mm_d0) * c_tan(beta1)
            - pos_z * c_tan(beta3)
            + mm_d0 * c_tan(beta2)
        )
        rdiff = r - rbeta
        rq += rdiff

        if abs(rdiff) < tol:
            break
    else:
        return 1.0

    if r != 0.0:
        return rq / r
    else:
        return 1.0


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def _multimed_nlay_core(
    pos_x: cython.double,
    pos_z: cython.double,
    ext_z0: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
    mmf: cython.double,
) -> tuple:
    radial_shift: cython.double = 1.0
    if mmf > 0.0 and mmf != 1.0:
        radial_shift = mmf
    else:
        radial_shift = _multimed_r_1lay_iterative(
            pos_x,
            pos_z,
            ext_z0,
            mm_n1,
            mm_n2_0,
            mm_n3,
            mm_d0,
        )

    Xq: cython.double = pos_x * radial_shift
    Yq: cython.double = 0.0

    return Xq, Yq


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def multimed_nlay(
    pos_x: cython.double,
    pos_y: cython.double,
    pos_z: cython.double,
    ext_x0: cython.double,
    ext_y0: cython.double,
    ext_z0: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
    mm_nlay: cython.int = 1,
    mmf: cython.double = 1.0,
    mm_n2=None,
    mm_d=None,
) -> tuple:
    """Compute radial-shifted Xq, Yq positions.

    Args:
        pos_x, pos_y, pos_z: 3D particle position.
        ext_x0, ext_y0, ext_z0: camera center.
        mm_n1, mm_n2_0, mm_n3: refractive indices.
        mm_d0: glass thickness.
        mm_nlay: number of layers.
        mmf: multimedia factor (pre-computed from LUT, 1.0 means no LUT).

    Returns:
        (Xq, Yq) 2D position on glass surface.
    """
    radial_shift: cython.double = 1.0
    if mmf > 0 and mmf != 1.0:
        radial_shift = mmf
    else:
        radial_shift = multimed_r_nlay_iterative(
            pos_x,
            pos_y,
            pos_z,
            ext_x0,
            ext_y0,
            ext_z0,
            mm_n1,
            mm_n2_0,
            mm_n3,
            mm_d0,
            mm_nlay,
            mm_n2=mm_n2,
            mm_d=mm_d,
        )

    Xq: cython.double = ext_x0 + (pos_x - ext_x0) * radial_shift
    Yq: cython.double = ext_y0 + (pos_y - ext_y0) * radial_shift

    return Xq, Yq


@cython.ccall
@cython.locals(
    zout=cython.double,
    dx=cython.double,
    dy=cython.double,
    r=cython.double,
    rq=cython.double,
    it=cython.int,
    beta1=cython.double,
    sin_beta1=cython.double,
    arg=cython.double,
    beta3=cython.double,
    rbeta=cython.double,
    rdiff=cython.double,
    i=cython.int,
)
@cython.ccall
@cython.exceptval(check=False)
def multimed_r_nlay_iterative(
    pos_x: cython.double,
    pos_y: cython.double,
    pos_z: cython.double,
    ext_x0: cython.double,
    ext_y0: cython.double,
    ext_z0: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
    mm_nlay: cython.int = 1,
    mm_n2=None,
    mm_d=None,
    n_iter: cython.int = 40,
    tol: cython.double = 0.001,
) -> cython.double:
    """Iteratively compute radial shift factor.

    Args:
        pos_x, pos_y, pos_z: 3D particle position.
        ext_x0, ext_y0, ext_z0: camera center.
        mm_n1, mm_n2_0, mm_n3: refractive indices.
        mm_d0: glass thickness.
        mm_nlay: number of layers.
        mm_d: list of layer thicknesses (defaults to [mm_d0]).
        n_iter: maximum iterations.
        tol: convergence tolerance.

    Returns:
        Radial shift factor (default 1.0 if no solution).
    """
    if mm_n1 == 1.0 and mm_nlay == 1 and mm_n2_0 == 1.0 and mm_n3 == 1.0:
        return 1.0

    if mm_n2 is None:
        mm_n2 = [mm_n2_0]
    if mm_d is None:
        mm_d = [mm_d0]

    zout = pos_z
    for i in range(1, mm_nlay):
        zout += mm_d[i]

    dx = pos_x - ext_x0
    dy = pos_y - ext_y0
    r = c_sqrt(dx * dx + dy * dy)
    rq = r

    for it in range(n_iter):
        beta1 = c_atan(rq / (ext_z0 - pos_z))
        sin_beta1 = c_sin(beta1)

        arg = sin_beta1 * mm_n1 / mm_n3
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        beta3 = c_asin(arg)

        rbeta = (ext_z0 - mm_d0) * c_tan(beta1) - zout * c_tan(beta3)
        for i in range(mm_nlay):
            arg_i = sin_beta1 * mm_n1 / mm_n2[i]
            if arg_i > 1.0:
                arg_i = 1.0
            elif arg_i < -1.0:
                arg_i = -1.0
            rbeta += mm_d[i] * c_tan(c_asin(arg_i))

        rdiff = r - rbeta
        rq += rdiff

        if abs(rdiff) < tol:
            break
    else:
        return 1.0

    if r != 0:
        return rq / r
    else:
        return 1.0


@cython.ccall
@cython.locals(
    gx=cython.double,
    gy=cython.double,
    gz=cython.double,
    dist_o_glas=cython.double,
    inv_dog=cython.double,
    dot_cam=cython.double,
    dist_cam_glas=cython.double,
    dot_pos=cython.double,
    dist_point_glas=cython.double,
    s_cam=cython.double,
    s_pt=cython.double,
    ext_t_z0=cython.double,
    s_d=cython.double,
    ag_x=cython.double,
    ag_y=cython.double,
    ag_z=cython.double,
    tmp_x=cython.double,
    tmp_y=cython.double,
    tmp_z=cython.double,
)
def trans_cam_point(
    pos: cython.double[:],
    ext_x0: cython.double,
    ext_y0: cython.double,
    ext_z0: cython.double,
    glass_vec_x: cython.double,
    glass_vec_y: cython.double,
    glass_vec_z: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
) -> tuple:
    """Project global-coordinate points through glass surface."""
    gx, gy, gz = glass_vec_x, glass_vec_y, glass_vec_z
    dist_o_glas = c_sqrt(gx * gx + gy * gy + gz * gz)
    inv_dog = 1.0 / dist_o_glas

    dot_cam = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
    dist_cam_glas = dot_cam * inv_dog - dist_o_glas - mm_d0

    dot_pos = pos[0] * gx + pos[1] * gy + pos[2] * gz
    dist_point_glas = dot_pos * inv_dog - dist_o_glas

    s_cam = dist_cam_glas * inv_dog
    cross_c = np.array(
        [ext_x0 - gx * s_cam, ext_y0 - gy * s_cam, ext_z0 - gz * s_cam],
        dtype=np.float64,
    )

    s_pt = dist_point_glas * inv_dog
    cross_p = np.array(
        [pos[0] - gx * s_pt, pos[1] - gy * s_pt, pos[2] - gz * s_pt], dtype=np.float64
    )

    ext_t_z0 = dist_cam_glas + mm_d0

    s_d = mm_d0 * inv_dog
    ag_x = cross_c[0] - gx * s_d
    ag_y = cross_c[1] - gy * s_d
    ag_z = cross_c[2] - gz * s_d
    tmp_x = cross_p[0] - ag_x
    tmp_y = cross_p[1] - ag_y
    tmp_z = cross_p[2] - ag_z

    pos_t = np.array(
        [c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z), 0.0, dist_point_glas],
        dtype=np.float64,
    )

    return pos_t, cross_p, cross_c, ext_t_z0


@cython.ccall
@cython.locals(
    gx=cython.double,
    gy=cython.double,
    gz=cython.double,
    n_gl=cython.double,
    inv_ngl=cython.double,
    s_d=cython.double,
    ag_x=cython.double,
    ag_y=cython.double,
    ag_z=cython.double,
    tmp_x=cython.double,
    tmp_y=cython.double,
    tmp_z=cython.double,
    n_ve=cython.double,
    s_z=cython.double,
    px=cython.double,
    py=cython.double,
    pz=cython.double,
    s_x=cython.double,
)
def back_trans_point(
    pos_t: cython.double[:],
    cross_p: cython.double[:],
    cross_c: cython.double[:],
    glass_vec_x: cython.double,
    glass_vec_y: cython.double,
    glass_vec_z: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
) -> object:
    """Transform from local coordinates back to global 3D space."""
    gx, gy, gz = glass_vec_x, glass_vec_y, glass_vec_z
    n_gl = c_sqrt(gx * gx + gy * gy + gz * gz)
    inv_ngl = 1.0 / n_gl

    s_d = mm_d0 * inv_ngl
    ag_x = cross_c[0] - gx * s_d
    ag_y = cross_c[1] - gy * s_d
    ag_z = cross_c[2] - gz * s_d

    tmp_x = cross_p[0] - ag_x
    tmp_y = cross_p[1] - ag_y
    tmp_z = cross_p[2] - ag_z
    n_ve = c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)

    s_z = -pos_t[2] * inv_ngl
    px = ag_x - gx * s_z
    py = ag_y - gy * s_z
    pz = ag_z - gz * s_z

    if n_ve > 0:
        s_x = -pos_t[0] / n_ve
        px -= tmp_x * s_x
        py -= tmp_y * s_x
        pz -= tmp_z * s_x

    return np.array([px, py, pz], dtype=np.float64)


@cython.ccall
def move_along_ray(
    glob_Z: cython.double,
    vertex: cython.double[:],
    direct: cython.double[:],
) -> object:
    """Find point along ray at given global Z value."""
    x: cython.double = vertex[0] + (glob_Z - vertex[2]) * direct[0] / direct[2]
    y: cython.double = vertex[1] + (glob_Z - vertex[2]) * direct[1] / direct[2]
    return np.array([x, y, glob_Z], dtype=np.float64)


@cython.ccall
@cython.locals(
    tx=cython.double,
    ty=cython.double,
    tz=cython.double,
    sz=cython.double,
    iz=cython.int,
    R=cython.double,
    sr=cython.double,
    ir=cython.int,
    v=cython.int,
    mmf=cython.double,
)
def get_mmf_from_mmlut(
    pos: cython.double[:],
    mmlut_origin: cython.double[:],
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
    mmlut_data: cython.double[:],
) -> cython.double:
    """Get multimedia factor from look-up table via bilinear interpolation."""
    tx = pos[0] - mmlut_origin[0]
    ty = pos[1] - mmlut_origin[1]
    tz = pos[2] - mmlut_origin[2]
    sz = tz / mmlut_rw
    iz = int(sz)
    sz -= iz

    R = c_sqrt(tx * tx + ty * ty)
    sr = R / mmlut_rw
    ir = int(sr)
    sr -= ir

    # Bilinear interpolation needs the 4 cell corners
    # (ir,iz)..(ir+1,iz+1) to be valid grid indices, i.e. ir+1 <= nr-1 and
    # iz+1 <= nz-1. data has exactly nr*nz elements (indices 0..nr*nz-1);
    # the old check `v4 > nr*nz` let the exact far corner read one element
    # past the end. Points outside the LUT cells return 0.0 so the caller
    # falls back to the iterative solve.
    # Keep in sync with imgcoord._get_mmf_from_mmlut_core.
    if ir < 0 or ir + 1 > mmlut_nr - 1:
        return 0.0
    if iz < 0 or iz + 1 > mmlut_nz - 1:
        return 0.0

    # Get vertices of box for bilinear interpolation
    v4_0: cython.int = ir * mmlut_nz + iz
    v4_1: cython.int = ir * mmlut_nz + (iz + 1)
    v4_2: cython.int = (ir + 1) * mmlut_nz + iz
    v4_3: cython.int = (ir + 1) * mmlut_nz + (iz + 1)

    # Bilinear interpolation
    mmf = (
        mmlut_data[v4_0] * (1 - sr) * (1 - sz)
        + mmlut_data[v4_1] * (1 - sr) * sz
        + mmlut_data[v4_2] * sr * (1 - sz)
        + mmlut_data[v4_3] * sr * sz
    )

    return mmf


def volumedimension(vpar, cpar, cal):
    """Find measurement volume limits in 3D space.

    Matches C volumedimension exactly.

    Args:
        vpar: VolumePar with X_lay, Zmin_lay, Zmax_lay.
        cpar: ControlPar with imx, imy, pix_x, pix_y, chfield, mm.
        cal: list of Calibration objects.

    Returns:
        (xmax, xmin, ymax, ymin, zmax, zmin) volume bounds.
    """
    from .ray_tracing import ray_tracing
    from .trafo import correct_brown_affin, pixel_to_metric

    xc = [0.0, float(cpar.imx)]
    yc = [0.0, float(cpar.imy)]

    Zmin = vpar.Zmin_lay[0]
    Zmax = vpar.Zmax_lay[0]
    if vpar.Zmin_lay[1] < Zmin:
        Zmin = vpar.Zmin_lay[1]
    if vpar.Zmax_lay[1] > Zmax:
        Zmax = vpar.Zmax_lay[1]

    xmin = xmax = 0.0
    ymin = ymax = 0.0
    first = True

    for i_cam in range(cpar.num_cams):
        c = cal[i_cam]
        ap = c.added_par
        mm = cpar.mm

        for i in range(2):
            for j in range(2):
                x, y = pixel_to_metric(xc[i], yc[j], cpar)
                x -= c.int_par.xh
                y -= c.int_par.yh

                x, y = correct_brown_affin(
                    x, y, ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she
                )

                pos, a = ray_tracing(
                    x,
                    y,
                    c.ext_par.dm,
                    c.ext_par.x0,
                    c.ext_par.y0,
                    c.ext_par.z0,
                    c.int_par.cc,
                    c.glass_par.vec_x,
                    c.glass_par.vec_y,
                    c.glass_par.vec_z,
                    mm.n1,
                    mm.n2[0],
                    mm.n3,
                    mm.d[0],
                )

                for Z in [Zmin, Zmax]:
                    X = pos[0] + (Z - pos[2]) * a[0] / a[2]
                    Y = pos[1] + (Z - pos[2]) * a[1] / a[2]

                    # A camera whose pose puts a corner ray past the glass/water
                    # critical angle (total internal reflection) makes
                    # ray_tracing legitimately return NaN for that corner. Skip
                    # it rather than let it seed xmin/xmax/ymin/ymax: with the
                    # old "first" flag, a NaN seed poisons every later
                    # comparison (X > xmax is False when either is NaN), so one
                    # bad camera silently wipes out the volume even when the
                    # others are fine.
                    if not (np.isfinite(X) and np.isfinite(Y)):
                        continue

                    if first:
                        xmin = xmax = X
                        ymin = ymax = Y
                        first = False
                    else:
                        if X > xmax:
                            xmax = X
                        if X < xmin:
                            xmin = X
                        if Y > ymax:
                            ymax = Y
                        if Y < ymin:
                            ymin = Y

    return xmax, xmin, ymax, ymin, Zmax, Zmin


def init_mmlut(vpar, cpar, cal):
    """Initialize multimedia look-up table for a single camera.

    Translates C init_mmlut from lib/src/multimed.c.

    Args:
        vpar: VolumePar with Zmin_lay, Zmax_lay.
        cpar: ControlPar with imx, imy, pix_x, pix_y, chfield, mm.
        cal: Calibration object. Modified in-place (mmlut populated).

    Returns:
        The modified Calibration object.
    """
    from .ray_tracing import ray_tracing
    from .trafo import correct_brown_affin, pixel_to_metric

    rw = 2.0

    xc = [0.0, float(cpar.imx)]
    yc = [0.0, float(cpar.imy)]

    Zmin = vpar.Zmin_lay[0]
    Zmax = vpar.Zmax_lay[0]
    if vpar.Zmin_lay[1] < Zmin:
        Zmin = vpar.Zmin_lay[1]
    if vpar.Zmax_lay[1] > Zmax:
        Zmax = vpar.Zmax_lay[1]

    Zmin -= Zmin % rw
    Zmax += rw - Zmax % rw

    Zmin_t = Zmin
    Zmax_t = Zmax
    Rmax = 0.0

    # cal_t starts as copy of cal's exterior
    cal_t_x0 = cal.ext_par.x0
    cal_t_y0 = cal.ext_par.y0
    cal_t_z0 = cal.ext_par.z0

    for i in range(2):
        for j in range(2):
            x, y = pixel_to_metric(
                xc[i],
                yc[j],
                cpar.imx,
                cpar.imy,
                cpar.pix_x,
                cpar.pix_y,
                cpar.chfield,
            )
            x -= cal.int_par.xh
            y -= cal.int_par.yh

            x, y = correct_brown_affin(
                x,
                y,
                cal.added_par.k1,
                cal.added_par.k2,
                cal.added_par.k3,
                cal.added_par.p1,
                cal.added_par.p2,
                cal.added_par.scx,
                cal.added_par.she,
            )

            pos, a = ray_tracing(
                x,
                y,
                cal.ext_par.dm,
                cal.ext_par.x0,
                cal.ext_par.y0,
                cal.ext_par.z0,
                cal.int_par.cc,
                cal.glass_par.vec_x,
                cal.glass_par.vec_y,
                cal.glass_par.vec_z,
                cpar.mm.n1,
                cpar.mm.n2[0],
                cpar.mm.n3,
                cpar.mm.d[0],
            )

            xyz = move_along_ray(Zmin, pos, a)
            xyz_t, cross_p, cross_c, ext_t_z0 = trans_cam_point(
                xyz,
                cal.ext_par.x0,
                cal.ext_par.y0,
                cal.ext_par.z0,
                cal.glass_par.vec_x,
                cal.glass_par.vec_y,
                cal.glass_par.vec_z,
                cpar.mm.n1,
                cpar.mm.n2[0],
                cpar.mm.n3,
                cpar.mm.d[0],
            )
            cal_t_x0 = 0.0
            cal_t_y0 = 0.0
            cal_t_z0 = ext_t_z0

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = c_sqrt((xyz_t[0] - cal_t_x0) ** 2 + (xyz_t[1] - cal_t_y0) ** 2)
            if R > Rmax:
                Rmax = R

            xyz = move_along_ray(Zmax, pos, a)
            xyz_t, cross_p, cross_c, ext_t_z0 = trans_cam_point(
                xyz,
                cal.ext_par.x0,
                cal.ext_par.y0,
                cal.ext_par.z0,
                cal.glass_par.vec_x,
                cal.glass_par.vec_y,
                cal.glass_par.vec_z,
                cpar.mm.n1,
                cpar.mm.n2[0],
                cpar.mm.n3,
                cpar.mm.d[0],
            )
            cal_t_x0 = 0.0
            cal_t_y0 = 0.0
            cal_t_z0 = ext_t_z0

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = c_sqrt((xyz_t[0] - cal_t_x0) ** 2 + (xyz_t[1] - cal_t_y0) ** 2)
            if R > Rmax:
                Rmax = R

    Rmax += rw - (Rmax % rw)

    nr = int(Rmax / rw + 1)
    nz = int((Zmax_t - Zmin_t) / rw + 1)

    cal.mmlut.origin = np.array([cal_t_x0, cal_t_y0, Zmin_t], dtype=np.float64)
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = rw

    if cal.mmlut.data is None:
        if cpar.mm.nlay == 1:
            data = _init_mmlut_data_fast(
                nr,
                nz,
                rw,
                cal_t_x0,
                cal_t_y0,
                cal_t_z0,
                Zmin_t,
                cpar.mm.n1,
                cpar.mm.n2[0],
                cpar.mm.n3,
                cpar.mm.d[0],
            )
        else:
            n2_arr = np.ascontiguousarray(
                cpar.mm.n2[: cpar.mm.nlay], dtype=np.float64
            )
            d_arr = np.ascontiguousarray(
                cpar.mm.d[: cpar.mm.nlay], dtype=np.float64
            )
            data = _init_mmlut_data_nlay_fast(
                nr,
                nz,
                rw,
                cal_t_x0,
                cal_t_y0,
                cal_t_z0,
                Zmin_t,
                cpar.mm.n1,
                cpar.mm.n3,
                n2_arr,
                d_arr,
                cpar.mm.nlay,
            )

        cal.mmlut.data = data

    return cal


def prepare_mmluts(vpar, cpar, cals) -> None:
    """Build the multimedia LUT for every camera that lacks one.

    This is the explicit "init at run start" that lets the whole pipeline
    (correspondences, 3D determination, sequence, tracking) use the fast
    bilinear LUT instead of the iterative Snell solve on every projection.

    - No-op for all-air setups (n1 == n2 == n3 == 1.0): the iterative solve
      already short-circuits to 1.0, so a LUT would only add overhead.
    - Idempotent: cameras whose LUT is already initialized are skipped, so it
      is safe to call at several pipeline entry points.
    """
    mm = cpar.mm
    n2_0 = mm.n2[0] if hasattr(mm.n2, "__getitem__") else mm.n2
    if mm.n1 == 1.0 and n2_0 == 1.0 and mm.n3 == 1.0:
        return
    for cal in cals:
        if not cal.mmlut.is_initialized:
            init_mmlut(vpar, cpar, cal)


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled

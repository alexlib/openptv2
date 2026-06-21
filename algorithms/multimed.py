"""Multimedia model operations for refractive layer calculations.

Translation of lib/src/multimed.c and lib/include/multimed.h.

Handles:
- Radial shift calculations through multi-media interfaces
- Camera-point projections through glass
- Multimedia Look-Up Table (MmLut) operations
- Volume dimension calculations
"""

import math
import cython
import numpy as np
from .vec_utils import (
    vec_set, vec_norm, vec_dot, vec_scalar_mul, vec_add, vec_subt, unit_vector
)

from .track_kernels import init_mmlut_data_jit as _init_mmlut_data_jit


# Y-remap mode constants (for interlaced cameras)
NO_REMAP: cython.int = 0
DOUBLED_PLUS_ONE: cython.int = 1
DOUBLED: cython.int = 2


@cython.ccall
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
            pos_x, pos_y, pos_z,
            ext_x0, ext_y0, ext_z0,
            mm_n1, mm_n2_0, mm_n3, mm_d0, mm_nlay,
            mm_n2=mm_n2, mm_d=mm_d,
        )

    Xq: cython.double = ext_x0 + (pos_x - ext_x0) * radial_shift
    Yq: cython.double = ext_y0 + (pos_y - ext_y0) * radial_shift

    return Xq, Yq


@cython.ccall
@cython.locals(
    zout=cython.double, dx=cython.double, dy=cython.double, r=cython.double, rq=cython.double,
    it=cython.int, beta1=cython.double, sin_beta1=cython.double, arg=cython.double,
    beta3=cython.double, rbeta=cython.double, rdiff=cython.double, i=cython.int
)
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
    r = math.sqrt(dx * dx + dy * dy)
    rq = r

    for it in range(n_iter):
        beta1 = math.atan(rq / (ext_z0 - pos_z))
        sin_beta1 = math.sin(beta1)

        beta2_vals = [0.0] * mm_nlay
        for i in range(mm_nlay):
            arg = sin_beta1 * mm_n1 / mm_n2[i]
            if arg > 1.0:
                arg = 1.0
            elif arg < -1.0:
                arg = -1.0
            beta2_vals[i] = math.asin(arg)

        arg = sin_beta1 * mm_n1 / mm_n3
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        beta3 = math.asin(arg)

        rbeta = (ext_z0 - mm_d0) * math.tan(beta1) - zout * math.tan(beta3)
        for i in range(mm_nlay):
            rbeta += mm_d[i] * math.tan(beta2_vals[i])

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
    gx=cython.double, gy=cython.double, gz=cython.double,
    dist_o_glas=cython.double, inv_dog=cython.double,
    dot_cam=cython.double, dist_cam_glas=cython.double,
    dot_pos=cython.double, dist_point_glas=cython.double,
    s_cam=cython.double, s_pt=cython.double, ext_t_z0=cython.double,
    s_d=cython.double, ag_x=cython.double, ag_y=cython.double, ag_z=cython.double,
    tmp_x=cython.double, tmp_y=cython.double, tmp_z=cython.double
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
    dist_o_glas = math.sqrt(gx * gx + gy * gy + gz * gz)
    inv_dog = 1.0 / dist_o_glas

    dot_cam = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
    dist_cam_glas = dot_cam * inv_dog - dist_o_glas - mm_d0

    dot_pos = pos[0] * gx + pos[1] * gy + pos[2] * gz
    dist_point_glas = dot_pos * inv_dog - dist_o_glas

    s_cam = dist_cam_glas * inv_dog
    cross_c = np.array([ext_x0 - gx * s_cam, ext_y0 - gy * s_cam, ext_z0 - gz * s_cam], dtype=np.float64)

    s_pt = dist_point_glas * inv_dog
    cross_p = np.array([pos[0] - gx * s_pt, pos[1] - gy * s_pt, pos[2] - gz * s_pt], dtype=np.float64)

    ext_t_z0 = dist_cam_glas + mm_d0

    s_d = mm_d0 * inv_dog
    ag_x = cross_c[0] - gx * s_d
    ag_y = cross_c[1] - gy * s_d
    ag_z = cross_c[2] - gz * s_d
    tmp_x = cross_p[0] - ag_x
    tmp_y = cross_p[1] - ag_y
    tmp_z = cross_p[2] - ag_z

    pos_t = np.array([math.sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z),
                      0.0, dist_point_glas], dtype=np.float64)

    return pos_t, cross_p, cross_c, ext_t_z0


@cython.ccall
@cython.locals(
    gx=cython.double, gy=cython.double, gz=cython.double,
    n_gl=cython.double, inv_ngl=cython.double,
    s_d=cython.double, ag_x=cython.double, ag_y=cython.double, ag_z=cython.double,
    tmp_x=cython.double, tmp_y=cython.double, tmp_z=cython.double,
    n_ve=cython.double, s_z=cython.double,
    px=cython.double, py=cython.double, pz=cython.double,
    s_x=cython.double
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
) -> np.ndarray:
    """Transform from local coordinates back to global 3D space."""
    gx, gy, gz = glass_vec_x, glass_vec_y, glass_vec_z
    n_gl = math.sqrt(gx * gx + gy * gy + gz * gz)
    inv_ngl = 1.0 / n_gl

    s_d = mm_d0 * inv_ngl
    ag_x = cross_c[0] - gx * s_d
    ag_y = cross_c[1] - gy * s_d
    ag_z = cross_c[2] - gz * s_d

    tmp_x = cross_p[0] - ag_x
    tmp_y = cross_p[1] - ag_y
    tmp_z = cross_p[2] - ag_z
    n_ve = math.sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)

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
) -> np.ndarray:
    """Find point along ray at given global Z value."""
    x: cython.double = vertex[0] + (glob_Z - vertex[2]) * direct[0] / direct[2]
    y: cython.double = vertex[1] + (glob_Z - vertex[2]) * direct[1] / direct[2]
    return np.array([x, y, glob_Z], dtype=np.float64)


@cython.ccall
@cython.locals(
    tx=cython.double, ty=cython.double, tz=cython.double,
    sz=cython.double, iz=cython.int,
    R=cython.double, sr=cython.double, ir=cython.int,
    v=cython.int, mmf=cython.double
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

    R = math.sqrt(tx * tx + ty * ty)
    sr = R / mmlut_rw
    ir = int(sr)
    sr -= ir

    # Check if point is inside LUT bounds
    if ir > mmlut_nr:
        return 0.0
    if iz < 0 or iz > mmlut_nz:
        return 0.0

    # Get vertices of box for bilinear interpolation
    v4_0: cython.int = ir * mmlut_nz + iz
    v4_1: cython.int = ir * mmlut_nz + (iz + 1)
    v4_2: cython.int = (ir + 1) * mmlut_nz + iz
    v4_3: cython.int = (ir + 1) * mmlut_nz + (iz + 1)

    max_v: cython.int = mmlut_nr * mmlut_nz
    if v4_0 < 0 or v4_0 > max_v: return 0.0
    if v4_1 < 0 or v4_1 > max_v: return 0.0
    if v4_2 < 0 or v4_2 > max_v: return 0.0
    if v4_3 < 0 or v4_3 > max_v: return 0.0

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
    from .trafo import pixel_to_metric, correct_brown_affin
    from .ray_tracing import ray_tracing

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
                    x, y, ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she)

                pos, a = ray_tracing(
                    x, y,
                    c.ext_par.dm, c.ext_par.x0, c.ext_par.y0, c.ext_par.z0,
                    c.int_par.cc,
                    c.glass_par.vec_x, c.glass_par.vec_y, c.glass_par.vec_z,
                    mm.n1, mm.n2[0], mm.n3, mm.d[0],
                )

                for Z in [Zmin, Zmax]:
                    X = pos[0] + (Z - pos[2]) * a[0] / a[2]
                    Y = pos[1] + (Z - pos[2]) * a[1] / a[2]

                    if first:
                        xmin = xmax = X
                        ymin = ymax = Y
                        first = False
                    else:
                        if X > xmax: xmax = X
                        if X < xmin: xmin = X
                        if Y > ymax: ymax = Y
                        if Y < ymin: ymin = Y

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
    from .trafo import pixel_to_metric, correct_brown_affin
    from .ray_tracing import ray_tracing

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
                xc[i], yc[j], cpar.imx, cpar.imy,
                cpar.pix_x, cpar.pix_y, cpar.chfield,
            )
            x -= cal.int_par.xh
            y -= cal.int_par.yh

            x, y = correct_brown_affin(
                x, y,
                cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
                cal.added_par.p1, cal.added_par.p2,
                cal.added_par.scx, cal.added_par.she,
            )

            pos, a = ray_tracing(
                x, y,
                cal.ext_par.dm,
                cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
                cal.int_par.cc,
                cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
                cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0],
            )

            xyz = move_along_ray(Zmin, pos, a)
            xyz_t, cross_p, cross_c, ext_t_z0 = trans_cam_point(
                xyz,
                cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
                cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
                cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0],
            )
            cal_t_x0 = 0.0
            cal_t_y0 = 0.0
            cal_t_z0 = ext_t_z0

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = math.sqrt((xyz_t[0] - cal_t_x0) ** 2 + (xyz_t[1] - cal_t_y0) ** 2)
            if R > Rmax:
                Rmax = R

            xyz = move_along_ray(Zmax, pos, a)
            xyz_t, cross_p, cross_c, ext_t_z0 = trans_cam_point(
                xyz,
                cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
                cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
                cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0],
            )
            cal_t_x0 = 0.0
            cal_t_y0 = 0.0
            cal_t_z0 = ext_t_z0

            if xyz_t[2] < Zmin_t:
                Zmin_t = xyz_t[2]
            if xyz_t[2] > Zmax_t:
                Zmax_t = xyz_t[2]

            R = math.sqrt((xyz_t[0] - cal_t_x0) ** 2 + (xyz_t[1] - cal_t_y0) ** 2)
            if R > Rmax:
                Rmax = R

    Rmax += rw - (Rmax % rw)

    nr = int(Rmax / rw + 1)
    nz = int((Zmax_t - Zmin_t) / rw + 1)

    cal.mmlut.origin = np.array([cal_t_x0, cal_t_y0, Zmin_t], dtype=np.float64)
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = int(rw)

    if cal.mmlut.data is None:
        if cpar.mm.nlay == 1:
            data = _init_mmlut_data_jit(
                nr, nz, rw,
                cal_t_x0, cal_t_y0, cal_t_z0,
                Zmin_t,
                cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0],
            )
        else:
            Ri = np.arange(nr) * rw
            Zi = Zmin_t + np.arange(nz) * rw
            data = np.zeros(nr * nz, dtype=np.float64)
            for i in range(nr):
                for j in range(nz):
                    xyz = np.array([Ri[i] + cal_t_x0, cal_t_y0, Zi[j]], dtype=np.float64)
                    data[i * nz + j] = multimed_r_nlay_iterative(
                        xyz[0], xyz[1], xyz[2],
                        cal_t_x0, cal_t_y0, cal_t_z0,
                        cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0],
                        cpar.mm.nlay,
                        mm_n2=cpar.mm.n2,
                        mm_d=cpar.mm.d,
                    )

        cal.mmlut.data = data

    return cal


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled

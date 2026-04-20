"""Multimedia model operations for refractive layer calculations.

Translation of lib/src/multimed.c and lib/include/multimed.h.

Handles:
- Radial shift calculations through multi-media interfaces
- Camera-point projections through glass
- Multimedia Look-Up Table (MmLut) operations
- Volume dimension calculations
"""

import numpy as np
from .vec_utils import (
    vec_set, vec_norm, vec_dot, vec_scalar_mul, vec_add, vec_subt, unit_vector
)


def multimed_nlay(
    pos_x: float,
    pos_y: float,
    pos_z: float,
    ext_x0: float,
    ext_y0: float,
    ext_z0: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
    mm_nlay: int = 1,
    mmf: float = 1.0,
) -> tuple[float, float]:
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
    if mmf > 0 and mmf != 1.0:
        radial_shift = mmf
    else:
        radial_shift = multimed_r_nlay_iterative(
            pos_x, pos_y, pos_z,
            ext_x0, ext_y0, ext_z0,
            mm_n1, mm_n2_0, mm_n3, mm_d0, mm_nlay,
        )

    Xq = ext_x0 + (pos_x - ext_x0) * radial_shift
    Yq = ext_y0 + (pos_y - ext_y0) * radial_shift

    return Xq, Yq


def multimed_r_nlay_iterative(
    pos_x: float,
    pos_y: float,
    pos_z: float,
    ext_x0: float,
    ext_y0: float,
    ext_z0: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
    mm_nlay: int = 1,
    mm_d: list[float] | None = None,
    n_iter: int = 40,
    tol: float = 0.001,
) -> float:
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
    # 1-medium case: no refraction
    if mm_n1 == 1.0 and mm_nlay == 1 and mm_n2_0 == 1.0 and mm_n3 == 1.0:
        return 1.0

    if mm_d is None:
        mm_d = [mm_d0]

    # Extra layers protrude into water side
    zout = pos_z
    for i in range(1, mm_nlay):
        zout += mm_d[i]

    r = np.sqrt((pos_x - ext_x0) ** 2 + (pos_y - ext_y0) ** 2)
    rq = r

    for it in range(n_iter):
        beta1 = np.arctan(rq / (ext_z0 - pos_z))

        beta2 = []
        for i in range(mm_nlay):
            arg = np.sin(beta1) * mm_n1 / mm_n2_0
            if arg < -1.0 - tol or arg > 1.0 + tol:
                raise ValueError(
                    f"Total internal reflection: arcsin argument out of bounds ({arg}). "
                    f"Check geometry and refractive indices. "
                    f"Parameters: pos_x={pos_x}, pos_y={pos_y}, pos_z={pos_z}, ext_x0={ext_x0}, ext_y0={ext_y0}, ext_z0={ext_z0}, mm_n1={mm_n1}, mm_n2_0={mm_n2_0}, mm_n3={mm_n3}, mm_d0={mm_d0}, mm_nlay={mm_nlay}, i={i}, beta1={beta1}, arg={arg}"
                )
            arg_clipped = np.clip(arg, -1.0, 1.0)
            beta2.append(np.arcsin(arg_clipped))

        beta3 = np.arcsin(np.sin(beta1) * mm_n1 / mm_n3)

        rbeta = (ext_z0 - mm_d0) * np.tan(beta1) - zout * np.tan(beta3)
        for i in range(mm_nlay):
            rbeta += mm_d[i] * np.tan(beta2[i])

        rdiff = r - rbeta
        rq += rdiff

        if abs(rdiff) < tol:
            break
    else:
        # Did not converge
        return 1.0

    if r != 0:
        return rq / r
    else:
        return 1.0


def trans_cam_point(
    pos: np.ndarray,
    ext_x0: float,
    ext_y0: float,
    ext_z0: float,
    glass_vec_x: float,
    glass_vec_y: float,
    glass_vec_z: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Project global-coordinate points through glass surface.

    Projects camera center and observed point onto the glass surface.

    Returns:
        (pos_t, cross_p, cross_c, ext_t_z0) where:
        - pos_t: transformed position in camera-local coords
        - cross_p: observed point projection on glass
        - cross_c: camera position projection on glass
        - ext_t_z0: transformed camera z0 (dist_cam_glas + mm_d0)
    """
    glass_dir = np.array([glass_vec_x, glass_vec_y, glass_vec_z], dtype=np.float64)
    primary_pt = np.array([ext_x0, ext_y0, ext_z0], dtype=np.float64)

    dist_o_glas = vec_norm(glass_dir)
    dist_cam_glas = vec_dot(primary_pt, glass_dir) / dist_o_glas - dist_o_glas - mm_d0
    dist_point_glas = vec_dot(pos, glass_dir) / dist_o_glas - dist_o_glas

    # Camera projection
    renorm = vec_scalar_mul(glass_dir, dist_cam_glas / dist_o_glas)
    cross_c = vec_subt(primary_pt, renorm)

    # Point projection
    renorm = vec_scalar_mul(glass_dir, dist_point_glas / dist_o_glas)
    cross_p = vec_subt(pos, renorm)

    # Transformed exterior z0
    ext_t_z0 = dist_cam_glas + mm_d0

    # Transformed position
    renorm = vec_scalar_mul(glass_dir, mm_d0 / dist_o_glas)
    temp = vec_subt(cross_c, renorm)
    temp = vec_subt(cross_p, temp)

    pos_t = np.array([vec_norm(temp), 0.0, dist_point_glas], dtype=np.float64)

    return pos_t, cross_p, cross_c, ext_t_z0


def back_trans_point(
    pos_t: np.ndarray,
    cross_p: np.ndarray,
    cross_c: np.ndarray,
    glass_vec_x: float,
    glass_vec_y: float,
    glass_vec_z: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
) -> np.ndarray:
    """Transform from local coordinates back to global 3D space.

    Opposite direction transfer from X_t, Y_t, Z_t to X, Y, Z.

    Args:
        pos_t: transformed position (x_t, y_t, z_t).
        cross_p: point projection on glass.
        cross_c: camera projection on glass.
        glass_vec_x, glass_vec_y, glass_vec_z: glass normal.
        mm_n1, mm_n2_0, mm_n3: refractive indices (unused here).
        mm_d0: glass thickness.

    Returns:
        Global 3D position.
    """
    glass_dir = np.array([glass_vec_x, glass_vec_y, glass_vec_z], dtype=np.float64)
    n_gl = vec_norm(glass_dir)

    renorm = vec_scalar_mul(glass_dir, mm_d0 / n_gl)
    after_glass = vec_subt(cross_c, renorm)
    temp = vec_subt(cross_p, after_glass)

    n_ve = vec_norm(temp)

    renorm = vec_scalar_mul(glass_dir, -pos_t[2] / n_gl)
    pos = vec_subt(after_glass, renorm)

    if n_ve > 0:
        renorm = vec_scalar_mul(temp, -pos_t[0] / n_ve)
        pos = vec_subt(pos, renorm)

    return pos


def move_along_ray(
    glob_Z: float,
    vertex: np.ndarray,
    direct: np.ndarray,
) -> np.ndarray:
    """Find point along ray at given global Z value.

    Args:
        glob_Z: target Z coordinate.
        vertex: ray origin (3,).
        direct: ray direction unit vector (3,).

    Returns:
        Point on ray at Z = glob_Z.
    """
    x = vertex[0] + (glob_Z - vertex[2]) * direct[0] / direct[2]
    y = vertex[1] + (glob_Z - vertex[2]) * direct[1] / direct[2]
    return np.array([x, y, glob_Z], dtype=np.float64)


def get_mmf_from_mmlut(
    pos: np.ndarray,
    mmlut_origin: np.ndarray,
    mmlut_nr: int,
    mmlut_nz: int,
    mmlut_rw: float,
    mmlut_data: np.ndarray,
) -> float:
    """Get multimedia factor from look-up table via bilinear interpolation.

    Args:
        pos: 3D position.
        mmlut_origin: LUT grid origin (x0, y0, z0).
        mmlut_nr: number of radial grid points.
        mmlut_nz: number of axial grid points.
        mmlut_rw: grid spacing.
        mmlut_data: 1D array of size nr * nz.

    Returns:
        Multimedia factor (0 if outside LUT bounds).
    """
    temp = pos - mmlut_origin
    sz = temp[2] / mmlut_rw
    iz = int(sz)
    sz -= iz

    R = np.sqrt(temp[0] ** 2 + temp[1] ** 2)
    sr = R / mmlut_rw
    ir = int(sr)
    sr -= ir

    # Check if point is inside LUT bounds
    if ir > mmlut_nr:
        return 0.0
    if iz < 0 or iz > mmlut_nz:
        return 0.0

    # Get vertices of box for bilinear interpolation
    v4 = [
        ir * mmlut_nz + iz,
        ir * mmlut_nz + (iz + 1),
        (ir + 1) * mmlut_nz + iz,
        (ir + 1) * mmlut_nz + (iz + 1),
    ]

    # Check bounds
    for v in v4:
        if v < 0 or v > mmlut_nr * mmlut_nz:
            return 0.0

    # Bilinear interpolation
    mmf = (
        mmlut_data[v4[0]] * (1 - sr) * (1 - sz)
        + mmlut_data[v4[1]] * (1 - sr) * sz
        + mmlut_data[v4[2]] * sr * (1 - sz)
        + mmlut_data[v4[3]] * sr * sz
    )

    return float(mmf)


def volumedimension(
    vpar_X_lay: tuple[float, float],
    vpar_Zmin_lay: tuple[float, float],
    vpar_Zmax_lay: tuple[float, float],
    cal_list: list[dict],
    imx: int,
    imy: int,
    pix_x: float,
    pix_y: float,
    chfield: int,
    int_xh: list[float],
    int_yh: list[float],
    added_par_list: list[dict],
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
) -> tuple[float, float, float, float, float, float]:
    """Find measurement volume limits in 3D space.

    Args:
        vpar_X_lay: (X_left, X_right) boundaries.
        vpar_Zmin_lay: (Zmin_left, Zmin_right) boundaries.
        vpar_Zmax_lay: (Zmax_left, Zmax_right) boundaries.
        cal_list: list of calibration dicts (one per camera).
        imx, imy: image dimensions.
        pix_x, pix_y: pixel sizes.
        chfield: y-remap mode.
        int_xh, int_yh: principal points per camera.
        added_par_list: distortion params per camera.
        mm_n1, mm_n2_0, mm_n3, mm_d0: multimedia params.

    Returns:
        (xmax, xmin, ymax, ymin, zmax, zmin) volume bounds.
    """
    from .trafo import pixel_to_metric, correct_brown_affin
    from .ray_tracing import ray_tracing

    xc = [0.0, float(imx)]
    yc = [0.0, float(imy)]

    Zmin = min(vpar_Zmin_lay)
    Zmax = max(vpar_Zmax_lay)

    xmin = xmax = 0.0
    ymin = ymax = 0.0
    first = True

    for cam_cal in cal_list:
        for i in range(2):
            for j in range(2):
                x, y = pixel_to_metric(
                    xc[i], yc[j], imx, imy, pix_x, pix_y, chfield
                )
                x -= int_xh[cam_cal["idx"]]
                y -= int_yh[cam_cal["idx"]]

                x, y = correct_brown_affin(
                    x, y,
                    cam_cal["k1"], cam_cal["k2"], cam_cal["k3"],
                    cam_cal["p1"], cam_cal["p2"],
                    cam_cal["scx"], cam_cal["she"],
                )

                pos, a = ray_tracing(
                    x, y,
                    cam_cal["dm"],
                    cam_cal["x0"], cam_cal["y0"], cam_cal["z0"],
                    cam_cal["cc"],
                    cam_cal["gx"], cam_cal["gy"], cam_cal["gz"],
                    mm_n1, mm_n2_0, mm_n3, mm_d0,
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

            R = np.sqrt((xyz_t[0] - cal_t_x0) ** 2 + (xyz_t[1] - cal_t_y0) ** 2)
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

            R = np.sqrt((xyz_t[0] - cal_t_x0) ** 2 + (xyz_t[1] - cal_t_y0) ** 2)
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
                )

        cal.mmlut.data = data

    return cal

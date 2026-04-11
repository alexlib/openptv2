from typing import List, Tuple

import math

import numpy as np
from numba import njit

from .calibration import Calibration
from .parameters import (
    ControlPar,
    MultimediaPar,
    VolumePar,
)
from .ray_tracing import ray_tracing
from .trafo import correct_brown_affine, pixel_to_metric
from .vec_utils import vec_norm


def multimed_nlay(
    cal: Calibration, mm: MultimediaPar, pos: np.ndarray
) -> Tuple[float, float]:
    """Create the Xq,Yq points for each X,Y point in the image space.

    using radial shift from the multimedia model
    """
    radial_shift = multimed_r_nlay(cal, mm, pos)
    Xq = cal.ext_par.x0 + (pos[0] - cal.ext_par.x0) * radial_shift
    Yq = cal.ext_par.y0 + (pos[1] - cal.ext_par.y0) * radial_shift
    return Xq, Yq


def multimed_r_nlay(cal: Calibration, mm: MultimediaPar, pos: np.ndarray) -> float:
    """Calculate the radial shift for the multimedia model."""
    # 1-medium case
    if mm.n1 == 1 and mm.nlay == 1 and mm.n2[0] == 1 and mm.n3 == 1:
        return 1.0

    #  interpolation using the existing mmlut
    if cal.mmlut_data.shape != (0, 0):
        # print("going into get_mmf_from_mmlut\n")
        mmf = get_mmf_from_mmlut(cal, pos)
        if mmf > 0:
            # print(f"mmf from data = {mmf}")
            return mmf

    n2_arr = mm.n2 if isinstance(mm.n2, np.ndarray) else np.asarray(mm.n2, dtype=np.float64)
    d_arr = mm.d if isinstance(mm.d, np.ndarray) else np.asarray(mm.d, dtype=np.float64)

    mmf = fast_multimed_r_nlay(
        mm.nlay,
        mm.n1,
        n2_arr,
        mm.n3,
        d_arr,
        cal.ext_par.x0,
        cal.ext_par.y0,
        cal.ext_par.z0,
        pos,
    )

    return mmf


@njit(fastmath=True, cache=True, nogil=True)
def fast_get_mmf_from_mmlut_raw(
    rw: int, origin: np.ndarray, data: np.ndarray, nz: int, nr: int, pos: np.ndarray
) -> float:
    """Numba-friendly raw MMLUT lookup."""
    temp = pos - origin
    sz = temp[2] / rw
    iz = int(sz)
    sz -= iz

    R = float(np.sqrt(temp[0] * temp[0] + temp[1] * temp[1]))
    sr = R / rw
    ir = int(sr)
    sr -= ir

    if ir > nr:
        return 0.0
    if iz < 0 or iz > nz:
        return 0.0

    v4_0 = ir * nz + iz
    v4_1 = ir * nz + (iz + 1)
    v4_2 = (ir + 1) * nz + iz
    v4_3 = (ir + 1) * nz + (iz + 1)

    for v in (v4_0, v4_1, v4_2, v4_3):
        if v < 0 or v > nr * nz:
            return 0.0

    return (
        data[v4_0] * (1 - sr) * (1 - sz)
        + data[v4_1] * (1 - sr) * sz
        + data[v4_2] * sr * (1 - sz)
        + data[v4_3] * sr * sz
    )


@njit(fastmath=True, cache=True, nogil=True)
def fast_multimed_r_nlay(
    nlay: int,
    n1: float,
    n2: np.ndarray,
    n3: float,
    d: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    pos: np.ndarray,
) -> float:
    """Faster multimedia model calculation — matches C multimed_r_nlay."""
    n_iter = 40
    X = pos[0]
    Y = pos[1]
    Z = pos[2]

    # Extra layers protrude into water side
    zout = Z
    for i in range(1, nlay):
        zout += d[i]

    dx = X - x0
    dy = Y - y0
    r = math.sqrt(dx * dx + dy * dy)
    rq = r

    it = 0
    rdiff = 0.1
    beta2 = np.empty(nlay, dtype=np.float64)

    while (rdiff > 0.001 or rdiff < -0.001) and it < n_iter:
        beta1 = math.atan(rq / (z0 - Z))
        sin_beta1 = math.sin(beta1)

        for layer in range(nlay):
            beta2[layer] = math.asin(sin_beta1 * n1 / n2[layer])

        beta3 = math.asin(sin_beta1 * n1 / n3)

        rbeta = (z0 - d[0]) * math.tan(beta1) - zout * math.tan(beta3)
        for layer in range(nlay):
            rbeta += d[layer] * math.tan(beta2[layer])

        rdiff = r - rbeta
        rq += rdiff
        it += 1

    if it >= n_iter:
        return 1.0

    if r != 0.0:
        return rq / r
    else:
        return 1.0


def trans_cam_point(
    ex: np.ndarray, mm: MultimediaPar, glass_dir: np.ndarray, pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Transform the camera and point coordinates to the glass coordinates.

    ex = Exterior(x0=ex_x, y0=ex_y, z0=ex_z)
    mm = MultimediaPar(d=mm_d)
    glass = Glass(vec_x=gl_vec_x, vec_y=gl_vec_y, vec_z=gl_vec_z)
    pos = np.array([pos_x, pos_y, pos_z])

    pos_t, cross_p, cross_c = trans_cam_point(ex, mm, glass, pos, ex_t)
    """
    origin = np.r_[ex.x0, ex.y0, ex.z0]  # type: ignore
    pos = pos.astype(np.float64)

    return fast_trans_cam_point(origin, mm.d[0], glass_dir, pos)


@njit(fastmath=True, cache=True, nogil=True)
def fast_trans_cam_point(
    primary_point: np.ndarray, d: float, glass_dir: np.ndarray, pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Derive translation of camera point."""
    dist_o_glass = float(np.linalg.norm(glass_dir))  # vector length
    if dist_o_glass == 0.0:
        dist_o_glass = 1.0

    dist_cam_glas = primary_point.dot(glass_dir)
    dist_cam_glas /= dist_o_glass
    dist_cam_glas -= dist_o_glass
    dist_cam_glas -= d

    dist_point_glass = pos.dot(glass_dir)
    dist_point_glass /= dist_o_glass
    dist_point_glass -= dist_o_glass

    renorm_glass = glass_dir * (dist_cam_glas / dist_o_glass)
    cross_c = primary_point - renorm_glass

    renorm_glass = glass_dir * (dist_point_glass / dist_o_glass)
    cross_p = pos - renorm_glass

    z0 = dist_cam_glas + d

    renorm_glass = glass_dir * (d / float(dist_o_glass))
    temp = cross_c - renorm_glass
    temp = cross_p - temp
    pos_t = np.array([np.linalg.norm(temp), 0, dist_point_glass])

    return pos_t, cross_p, cross_c, float(z0)


def back_trans_point(
    pos_t: np.ndarray,
    mm: MultimediaPar,
    glass: np.ndarray,
    cross_p: np.ndarray,
    cross_c: np.ndarray,
) -> np.ndarray:
    """
    Transform the point coordinates from the glass to the camera coordinates.

    Args:
    ----
        pos_t: A numpy array representing the position of the point in the glass coordinate system.
        mm: SomeType (TODO: specify type). A parameter used to scale the glass direction vector.
        G: A Glass object representing the glass coordinate system.
        cross_p: A numpy array representing the position of the point in the pixel coordinate system.
        cross_c: A numpy array representing the position of the point in the camera coordinate system.

    Returns
    -------
        A numpy array representing the position of the point in the camera coordinate system.
    """
    return fast_back_trans_point(glass, mm.d[0], cross_c, cross_p, pos_t)


@njit(fastmath=True, cache=True, nogil=True)
def fast_back_trans_point(
    glass_direction: np.ndarray,
    d: float,
    cross_c: np.ndarray,
    cross_p: np.ndarray,
    pos_t: np.ndarray,
) -> np.ndarray:
    """Run numba faster version of back projection."""
    # Calculate the glass direction vector

    norm_glass_direction = float(np.linalg.norm(glass_direction))

    # Normalize the glass direction vector
    renorm_glass = glass_direction * (d / norm_glass_direction)

    # Calculate the position of the point after passing through the glass
    after_glass = cross_c - renorm_glass

    # Calculate the vector between the point in the glass and the point after passing through the glass
    temp = cross_p - after_glass

    # Calculate the norm of the vector temp
    norm_temp = np.linalg.norm(temp)

    # Calculate the position of the point in the camera coordinate system
    renorm_glass = glass_direction * (-pos_t[2] / norm_glass_direction)
    pos = after_glass - renorm_glass

    # If the norm of the vector temp is greater than zero, adjust the position
    # of the point in the camera coordinate system
    if norm_temp > 0.0:  # type: ignore
        renorm_temp = temp * (-pos_t[0] / norm_temp)
        pos = pos - renorm_temp

    return pos


@njit(fastmath=True, cache=True, nogil=True)
def fast_flat_image_coord_raw(
    orig_pos: np.ndarray,
    ex_pos: np.ndarray,
    ex_dm: np.ndarray,
    int_cc: float,
    glass_par: np.ndarray,
    mm_d: np.ndarray,
    mm_n1: float,
    mm_n2: np.ndarray,
    mm_n3: float,
    mmlut_origin: np.ndarray,
    mmlut_data: np.ndarray,
    mmlut_nz: int,
    mmlut_nr: int,
    mmlut_rw: int,
    mm_nlay: int = 1,
) -> Tuple[float, float]:
    """Raw-array version of flat_image_coord for batch use."""
    pos_t, cross_p, cross_c, z0_t = fast_trans_cam_point(ex_pos, mm_d[0], glass_par, orig_pos)

    # Use transformed calibration: x0=0, y0=0, z0=z0_t (matches C cal_t)
    mmf = fast_get_mmf_from_mmlut_raw(
        mmlut_rw, mmlut_origin, mmlut_data, mmlut_nz, mmlut_nr, pos_t
    )
    if mmf <= 0.0:
        mmf = fast_multimed_r_nlay(
            mm_nlay, mm_n1, mm_n2, mm_n3, mm_d,
            0.0, 0.0, z0_t,  # transformed cal: x0=0, y0=0, z0=z0_t
            pos_t,
        )

    # multimed_nlay: Xq = x0 + (pos[0] - x0) * mmf, with x0=0, y0=0
    x_t = pos_t[0] * mmf
    y_t = pos_t[1] * mmf

    pos_t2 = np.array([x_t, y_t, pos_t[2]])
    pos = fast_back_trans_point(glass_par, mm_d[0], cross_c, cross_p, pos_t2)

    dp0 = pos[0] - ex_pos[0]
    dp1 = pos[1] - ex_pos[1]
    dp2 = pos[2] - ex_pos[2]

    deno = ex_dm[0, 2] * dp0 + ex_dm[1, 2] * dp1 + ex_dm[2, 2] * dp2
    if deno == 0.0:
        deno = 1.0

    x = -int_cc * (ex_dm[0, 0] * dp0 + ex_dm[1, 0] * dp1 + ex_dm[2, 0] * dp2) / deno
    y = -int_cc * (ex_dm[0, 1] * dp0 + ex_dm[1, 1] * dp1 + ex_dm[2, 1] * dp2) / deno
    return x, y


@njit(fastmath=True, cache=True, nogil=True)
def fast_point_to_pixel(
    point: np.ndarray,
    ex_pos: np.ndarray,
    ex_dm: np.ndarray,
    int_cc: float,
    int_xh: float,
    int_yh: float,
    added_par: np.ndarray,
    glass_par: np.ndarray,
    mm_d: np.ndarray,
    mm_n1: float,
    mm_n2: np.ndarray,
    mm_n3: float,
    mm_nlay: int,
    mmlut_origin: np.ndarray,
    mmlut_data: np.ndarray,
    mmlut_nz: int,
    mmlut_nr: int,
    mmlut_rw: int,
    imx: int,
    imy: int,
    pix_x: float,
    pix_y: float,
) -> Tuple[float, float]:
    """Full 3D-to-pixel pipeline in numba: flat_image_coord + flat_to_dist + metric_to_pixel."""
    # --- flat_image_coord ---
    fx, fy = fast_flat_image_coord_raw(
        point, ex_pos, ex_dm, int_cc, glass_par,
        mm_d, mm_n1, mm_n2, mm_n3,
        mmlut_origin, mmlut_data, mmlut_nz, mmlut_nr, mmlut_rw,
        mm_nlay,
    )

    # --- flat_to_dist: shift by principal point then apply Brown distortion ---
    fx += int_xh
    fy += int_yh
    if fx != 0.0 or fy != 0.0:
        r = math.sqrt(fx * fx + fy * fy)
        r2 = r * r
        r4 = r2 * r2
        r6 = r4 * r2
        k1, k2, k3 = added_par[0], added_par[1], added_par[2]
        p1, p2 = added_par[3], added_par[4]
        scx, she = added_par[5], added_par[6]
        radial = k1 * r2 + k2 * r4 + k3 * r6
        dx = fx * radial + p1 * (r2 + 2.0 * fx * fx) + 2.0 * p2 * fx * fy
        dy = fy * radial + p2 * (r2 + 2.0 * fy * fy) + 2.0 * p1 * fx * fy
        fx += dx
        fy += dy
        x_dist = scx * fx - math.sin(she) * fy
        y_dist = math.cos(she) * fy
    else:
        x_dist = 0.0
        y_dist = 0.0

    # --- metric_to_pixel ---
    px = x_dist / pix_x + float(imx) / 2.0
    py = float(imy) / 2.0 - y_dist / pix_y
    return px, py


class CalibRawArrays:
    """Pre-extracted raw arrays from Calibration for fast numba calls."""

    __slots__ = (
        'ex_pos', 'ex_dm', 'int_cc', 'int_xh', 'int_yh', 'added_par',
        'glass_par', 'mm_d', 'mm_n1', 'mm_n2', 'mm_n3', 'mm_nlay',
        'mmlut_origin', 'mmlut_data', 'mmlut_nz', 'mmlut_nr', 'mmlut_rw',
        'imx', 'imy', 'pix_x', 'pix_y',
    )

    def __init__(self, cal, cpar):
        """Extract arrays from Calibration and ControlPar for numba use.

        Raises ValueError if any required parameter is missing or invalid.
        """
        # --- Exterior parameters ---
        ext = cal.ext_par
        if ext is None:
            raise ValueError("CalibRawArrays: cal.ext_par is None")
        self.ex_pos = np.array([ext.x0, ext.y0, ext.z0], dtype=np.float64)
        self.ex_dm = np.ascontiguousarray(ext.dm, dtype=np.float64)
        if self.ex_dm.shape != (3, 3):
            raise ValueError(
                f"CalibRawArrays: ext_par.dm has shape {self.ex_dm.shape}, expected (3, 3)"
            )

        # --- Interior parameters ---
        ip = cal.int_par
        if ip is None:
            raise ValueError("CalibRawArrays: cal.int_par is None")
        self.int_cc = float(ip.cc)
        if self.int_cc == 0.0:
            raise ValueError("CalibRawArrays: int_par.cc (camera constant) is 0")
        self.int_xh = float(ip.xh)
        self.int_yh = float(ip.yh)

        # --- Additional (distortion) parameters ---
        if cal.added_par is None:
            raise ValueError("CalibRawArrays: cal.added_par is None")
        self.added_par = np.ascontiguousarray(cal.added_par, dtype=np.float64)
        if self.added_par.shape != (7,):
            raise ValueError(
                f"CalibRawArrays: added_par has shape {self.added_par.shape}, expected (7,)"
            )

        # --- Glass parameters ---
        if cal.glass_par is None:
            raise ValueError("CalibRawArrays: cal.glass_par is None")
        self.glass_par = np.ascontiguousarray(cal.glass_par, dtype=np.float64)

        # --- Multimedia parameters ---
        mm = cpar.mm
        if mm is None:
            raise ValueError("CalibRawArrays: cpar.mm (MultimediaPar) is None")
        self.mm_d = np.asarray(mm.d, dtype=np.float64) if not isinstance(mm.d, np.ndarray) else mm.d
        self.mm_n1 = float(mm.n1)
        self.mm_n2 = np.asarray(mm.n2, dtype=np.float64) if not isinstance(mm.n2, np.ndarray) else mm.n2
        self.mm_n3 = float(mm.n3)
        self.mm_nlay = int(mm.nlay)
        if self.mm_n1 <= 0.0:
            raise ValueError(f"CalibRawArrays: mm.n1 = {self.mm_n1}, must be > 0")
        if self.mm_n3 <= 0.0:
            raise ValueError(f"CalibRawArrays: mm.n3 = {self.mm_n3}, must be > 0")

        # --- MMLUT ---
        if cal.mmlut is None:
            raise ValueError("CalibRawArrays: cal.mmlut is None")
        self.mmlut_nz = int(cal.mmlut.nz)
        self.mmlut_nr = int(cal.mmlut.nr)
        self.mmlut_rw = int(cal.mmlut.rw)
        if self.mmlut_nz == 0 or self.mmlut_nr == 0 or self.mmlut_rw == 0:
            raise ValueError(
                f"CalibRawArrays: MMLUT not initialized "
                f"(nz={self.mmlut_nz}, nr={self.mmlut_nr}, rw={self.mmlut_rw}). "
                f"Call init_mmlut() before creating CalibRawArrays."
            )
        self.mmlut_origin = np.ascontiguousarray(cal.mmlut.origin.ravel(), dtype=np.float64)
        if cal.mmlut_data is None:
            raise ValueError("CalibRawArrays: cal.mmlut_data is None")
        self.mmlut_data = cal.mmlut_data.ravel()
        if self.mmlut_data.size == 0:
            raise ValueError("CalibRawArrays: cal.mmlut_data is empty")

        # --- Image / pixel parameters ---
        self.imx = int(cpar.imx)
        self.imy = int(cpar.imy)
        if self.imx <= 0 or self.imy <= 0:
            raise ValueError(
                f"CalibRawArrays: image size invalid (imx={self.imx}, imy={self.imy})"
            )
        self.pix_x = float(cpar.pix_x)
        self.pix_y = float(cpar.pix_y)
        if self.pix_x <= 0.0 or self.pix_y <= 0.0:
            raise ValueError(
                f"CalibRawArrays: pixel size invalid (pix_x={self.pix_x}, pix_y={self.pix_y})"
            )

    def project(self, point):
        """Project a 3D point to pixel coordinates."""
        return fast_point_to_pixel(
            point, self.ex_pos, self.ex_dm, self.int_cc, self.int_xh, self.int_yh,
            self.added_par, self.glass_par, self.mm_d, self.mm_n1, self.mm_n2,
            self.mm_n3, self.mm_nlay, self.mmlut_origin, self.mmlut_data,
            self.mmlut_nz, self.mmlut_nr, self.mmlut_rw,
            self.imx, self.imy, self.pix_x, self.pix_y,
        )


@njit(fastmath=True, cache=True, nogil=True)
def move_along_ray(glob_z: float, vertex: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """Move along the ray to the global z plane.

    move_along_ray() calculates the position of a point in a global Z value
    along a ray whose vertex and direction are given.

    Arguments:
    ---------
    double glob_z - the Z value of the result point in the global
        coordinate system.
    vec3d vertex - the ray vertex.
    vec3d direct - the ray direction, a unit vector.
    vec3d out - result buffer.

    """
    out = np.zeros(3, dtype=np.float64)
    if direct[2] == 0:
        direct[2] = 1  # avoid division by zero

    out[0] = vertex[0] + (glob_z - vertex[2]) * direct[0] / direct[2]
    out[1] = vertex[1] + (glob_z - vertex[2]) * direct[1] / direct[2]
    out[2] = glob_z

    return out


def init_mmlut(vpar: VolumePar, cpar: ControlPar, cal: Calibration) -> Calibration:
    """Initialize the multilayer lookup table."""
    rw = 2
    Rmax = 0.0

    # image corners
    xc = [0.0, float(cpar.imx)]
    yc = [0.0, float(cpar.imy)]

    # find extrema of imaged object volume
    z_min = min(vpar.z_min_lay)
    z_max = max(vpar.z_max_lay)

    z_min -= np.fmod(z_min, rw)
    z_max += rw - np.fmod(z_max, rw)

    z_min_t = z_min
    z_max_t = z_max

    # intersect with image vertices rays
    cal_t = Calibration(mmlut=cal.mmlut.copy())

    for i in range(2):
        for j in range(2):
            x, y = pixel_to_metric(xc[i], yc[j], cpar)
            x -= cal.int_par.xh
            y -= cal.int_par.yh
            x, y = correct_brown_affine(x, y, cal.added_par)
            pos, a = ray_tracing(x, y, cal, cpar.mm)
            xyz = move_along_ray(z_min, pos, a)
            xyz_t, _, _, cal_t.ext_par.z0 = trans_cam_point(
                cal.ext_par, cpar.mm, cal.glass_par, xyz
            )

            if xyz_t[2] < z_min_t:
                z_min_t = xyz_t[2]
            if xyz_t[2] > z_max_t:
                z_max_t = xyz_t[2]

            R = vec_norm(
                np.r_[xyz_t[0] - cal_t.ext_par.x0, xyz_t[1] - cal_t.ext_par.y0, 0]
            )

            if R > Rmax:
                Rmax = R

            xyz = move_along_ray(z_max, pos, a)
            xyz_t, _, _, cal_t.ext_par.z0 = trans_cam_point(
                cal.ext_par, cpar.mm, cal.glass_par, xyz
            )

            if xyz_t[2] < z_min_t:
                z_min_t = xyz_t[2]
            if xyz_t[2] > z_max_t:
                z_max_t = xyz_t[2]

            R = vec_norm(
                np.r_[xyz_t[0] - cal_t.ext_par.x0, xyz_t[1] - cal_t.ext_par.y0, 0]
            )

            if R > Rmax:
                Rmax = R

    # round values (-> enlarge)
    Rmax += rw - np.fmod(Rmax, rw)

    # get # of rasterlines in r, z
    nr = int(Rmax / rw + 1)
    nz = int((z_max_t - z_min_t) / rw + 1)

    # create two dimensional mmlut structure
    cal.mmlut.origin = np.r_[cal_t.ext_par.x0, cal_t.ext_par.y0, z_min_t]
    cal.mmlut.nr = nr
    cal.mmlut.nz = nz
    cal.mmlut.rw = rw

    if cal.mmlut_data.shape == (0, 0):
        cal.mmlut_data = np.empty((nr, nz), dtype=np.float64)
        Ri = np.arange(nr) * rw
        Zi = np.arange(nz) * rw + z_min_t

        for i in range(nr):
            for j in range(nz):
                xyz = np.r_[Ri[i] + cal_t.ext_par.x0, cal_t.ext_par.y0, Zi[j]]
                cal.mmlut_data.flat[i * nz + j] = multimed_r_nlay(cal_t, cpar.mm, xyz)

        # print(f"filled mmlut data with {data}")
        # cal.mmlut_data = data

    return cal


def get_mmf_from_mmlut(cal: Calibration, pos: np.ndarray) -> float:
    """Get the refractive index of the medium at a given position."""
    rw = cal.mmlut.rw
    origin = cal.mmlut.origin
    data = cal.mmlut_data.ravel()  # view, no copy
    nz = int(cal.mmlut.nz)
    nr = int(cal.mmlut.nr)

    return fast_get_mmf_from_mmlut(rw, origin, data, nz, nr, pos)


@njit(fastmath=True, cache=True, nogil=True)
def fast_get_mmf_from_mmlut(
    rw: int, origin: np.ndarray, data: np.ndarray, nz: int, nr: int, pos: np.ndarray
) -> float:
    """Get the refractive index of the medium at a given position."""
    temp = pos - origin
    sz = temp[2] / rw
    iz = int(sz)
    sz -= iz

    R = float(np.sqrt(temp[0] * temp[0] + temp[1] * temp[1]))
    sr = R / rw
    ir = int(sr)
    sr -= ir

    if ir > nr:
        return 0.0
    if iz < 0 or iz > nz:
        return 0.0

    # bilinear interpolation in r/z box
    # get vertices of box
    v4_0 = ir * nz + iz
    v4_1 = ir * nz + (iz + 1)
    v4_2 = (ir + 1) * nz + iz
    v4_3 = (ir + 1) * nz + (iz + 1)

    # 2. check wther point is inside camera's object volume
    # important for epipolar line computation
    for v in (v4_0, v4_1, v4_2, v4_3):
        if v < 0 or v > nr * nz:
            return 0.0

    # interpolate
    mmf = (
        data[v4_0] * (1 - sr) * (1 - sz)
        + data[v4_1] * (1 - sr) * sz
        + data[v4_2] * sr * (1 - sz)
        + data[v4_3] * sr * sz
    )

    return mmf


def volumedimension(
    xmax: float,
    xmin: float,
    ymax: float,
    ymin: float,
    z_max: float,
    z_min: float,
    vpar: VolumePar,
    cpar: ControlPar,
    cal: List[Calibration],
) -> Tuple[float, float, float, float, float, float]:
    """Calculate the volume dimensions."""
    xc = [0.0, cpar.imx]
    yc = [0.0, cpar.imy]

    z_min = vpar.z_min_lay[0]
    z_max = vpar.z_max_lay[0]

    if vpar.z_min_lay[1] < z_min:
        z_min = vpar.z_min_lay[1]
    if vpar.z_max_lay[1] > z_max:
        z_max = vpar.z_max_lay[1]

    for i_cam in range(cpar.num_cams):
        for i in range(2):
            for j in range(2):
                x, y = pixel_to_metric(xc[i], yc[j], cpar)

                x -= cal[i_cam].int_par.xh
                y -= cal[i_cam].int_par.yh

                x, y = correct_brown_affine(x, y, cal[i_cam].added_par)

                pos, a = ray_tracing(x, y, cal[i_cam], cpar.mm)

                # Guard against division by zero when a[2] is zero/near-zero
                if abs(a[2]) > 1e-10:
                    X = pos[0] + (z_min + pos[2]) * a[0] / a[2]
                    Y = pos[1] + (z_min + pos[2]) * a[1] / a[2]

                    if X > xmax:
                        xmax = X
                    if X < xmin:
                        xmin = X
                    if Y > ymax:
                        ymax = Y
                    if Y < ymin:
                        ymin = Y

                if abs(a[2]) > 1e-10:
                    X = pos[0] + (z_max - pos[2]) * a[0] / a[2]
                    Y = pos[1] + (z_max - pos[2]) * a[1] / a[2]

                    if X > xmax:
                        xmax = X
                    if X < xmin:
                        xmin = X
                    if Y > ymax:
                        ymax = Y
                    if Y < ymin:
                        ymin = Y

    return (xmax, xmin, ymax, ymin, z_max, z_min)

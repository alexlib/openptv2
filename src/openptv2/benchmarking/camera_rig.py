"""Simple 4-camera rig for ground-truth tracking benchmarks.

Builds a small, self-consistent camera setup aimed at a measurement volume,
with optional multimedia (glass window + water) refraction.  The rig is defined
entirely in code (no calibration files required) and uses camera orientations
validated against the model so the volume projects onto the sensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Exterior,
    Glass,
    Interior,
)
from openptv2.algorithms.imgcoord import flat_image_coord_batch, img_coord_batch
from openptv2.algorithms.multimed import prepare_mmluts
from openptv2.algorithms.parameters import ControlPar, MmNp, VolumePar
from openptv2.algorithms.trafo import metric_to_pixel_batch

# Realistic refractive indices.  Convention (same as the imaging model):
#   n1 = air (camera side)
#   n2 = glass window (the refractive layer of thickness ``d``)
#   n3 = water (the medium the particles are in)
N_AIR = 1.0
N_GLASS = 1.46
N_WATER = 1.33

# Signed glass-normal magnitude (the model divides by |glass_vec|, sign just
# selects the interface orientation relative to the viewing direction).
_GLASS_MAG = 125.0


@dataclass
class CameraRig:
    """The result of :func:`make_standard_rig`.

    Attributes
    ----------
    cals : list[Calibration]
        One calibration per camera.
    cpar : ControlPar
        Control parameters (image size, pixel size, multimedia).
    vpar : VolumePar
        Volume parameters (measurement volume bounds).
    refract : bool
        Whether multimedia refraction was enabled.
    """

    cals: list[Calibration]
    cpar: ControlPar
    vpar: VolumePar
    refract: bool = False


# Proven base camera orientations: 2 cameras looking in from the -Z side and
# 2 from the +Z side, spread in X, around a volume centred at the origin.
# These are the synthetic-test rig orientations that are known to project a
# spread of points correctly onto the sensor.
_BASE_CAMERAS = [
    # (x, y, z, omega, phi, kappa) matching the proven rotations.  The angles
    # reproduce the rotation matrix exactly (compute_rotation_matrix), so the
    # rig can be serialised to .ori files and round-trip faithfully.
    dict(pos=(80.996, 13.130, -569.756), angle=(-56.54108642, 2.97742655, 56.53124852)),
    dict(
        pos=(-123.459, 23.996, -575.191), angle=(0.02718932, -2.92335731, -0.01854668)
    ),
    dict(
        pos=(-110.557, 73.466, 584.364), angle=(-0.11212347, -0.19805209, -0.02811924)
    ),
    dict(pos=(126.369, 67.935, 573.047), angle=(-0.11906977, 0.23974137, 0.00947221)),
]

# Reference rotation matrices (rows = x, y, z optical axes) that are known to
# project the volume correctly.  Sourced from the synthetic test rig.
_BASE_ROTATIONS = [
    np.array(
        [
            [-0.9864, -0.0172, 0.1634],
            [-0.0162, 0.9998, 0.0075],
            [-0.1635, 0.0047, -0.9865],
        ]
    ),
    np.array(
        [
            [-0.9761, -0.0181, -0.2165],
            [-0.0244, 0.9993, 0.0265],
            [0.2159, 0.0312, -0.9759],
        ]
    ),
    np.array(
        [
            [0.9801, 0.0276, -0.1968],
            [-0.0059, 0.9939, 0.1097],
            [0.1986, -0.1063, 0.9743],
        ]
    ),
    np.array(
        [
            [0.9714, -0.0092, 0.2375],
            [-0.0188, 0.9931, 0.1154],
            [-0.2369, -0.1166, 0.9645],
        ]
    ),
]


def make_standard_rig(
    num_cams: int = 4,
    volume: tuple[float, float, float] = (100.0, 100.0, 100.0),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    image_size: tuple[int, int] = (1280, 1024),
    pixel_size: tuple[float, float] = (0.012, 0.012),
    cc: float = 70.0,
    refract: bool = True,
    glass_thickness: float = 6.0,
    glass_vec: tuple[float, float, float] = (0.0, 0.0, -1.0),
    use_mmlut: bool = True,
) -> CameraRig:
    """Build a simple camera rig around a measurement volume.

    Uses proven camera orientations so the volume projects onto the sensors.
    Two cameras view from each side of the volume along the Z axis, spread in X.

    Parameters
    ----------
    num_cams : int
        Number of cameras (default 4).
    volume : tuple[float, float, float]
        Full measurement volume size [mm] along X, Y, Z.
    center : tuple[float, float, float]
        Centre of the measurement volume.
    image_size : tuple[int, int]
        Sensor size in pixels (width, height).
    pixel_size : tuple[float, float]
        Pixel pitch [mm].
    cc : float
        Camera constant / focal length [mm].
    refract : bool
        If True, enable glass + water multimedia refraction.  Otherwise the
        cameras see an all-air volume (n1=n2=n3=1).
    glass_thickness : float
        Thickness of the glass window [mm].
    glass_vec : tuple[float, float, float]
        Normal of the glass interface.  Must be a non-zero vector along Z
        (e.g. (0,0,1) or (0,0,-1)); a zero vector breaks the imaging model.
    use_mmlut : bool
        If True and refraction is on, build the multimedia lookup table.

    Returns
    -------
    CameraRig
    """
    center_arr = np.array(center, dtype=np.float64)
    n = num_cams
    if n > len(_BASE_CAMERAS):
        raise ValueError(
            f"Only {len(_BASE_CAMERAS)} base camera orientations available"
        )

    # Multimedia parameters.
    if refract:
        mm = MmNp(nlay=1, n1=N_AIR, n2=[N_GLASS], d=[glass_thickness], n3=N_WATER)
    else:
        mm = MmNp(nlay=1, n1=1.0, n2=[1.0], d=[0.0], n3=1.0)

    cpar = ControlPar(
        num_cams=n,
        img_base_name=[""] * n,
        cal_img_base_name=[""] * n,
        allCam_flag=0,
        hp_flag=1,
        chfield=0,
        tiff_flag=1,
        pix_x=pixel_size[0],
        pix_y=pixel_size[1],
        imx=image_size[0],
        imy=image_size[1],
        mm=mm,
    )

    vx, vy, vz = volume
    vpar = VolumePar(
        X_lay=np.array([center_arr[0] - vx / 2.0, center_arr[0] + vx / 2.0]),
        Zmin_lay=np.array([center_arr[2] - vz / 2.0, center_arr[2] - vz / 2.0]),
        Zmax_lay=np.array([center_arr[2] + vz / 2.0, center_arr[2] + vz / 2.0]),
        cnx=0.0,
        cny=0.0,
        cn=0.0,
        csumg=0.0,
        corrmin=0.0,
        eps0=0.0,
    )

    # Glass vector: recompute sign from the viewing direction so refraction is
    # physical, while keeping a non-zero +Z/-Z vector as required.
    cals: list[Calibration] = []
    for cam in range(n):
        base = _BASE_CAMERAS[cam]
        rot = _BASE_ROTATIONS[cam].copy()
        pos = (
            base["pos"][0] + center_arr[0],
            base["pos"][1] + center_arr[1],
            base["pos"][2] + center_arr[2],
        )
        om, ph, ka = base["angle"]

        ext = Exterior(x0=pos[0], y0=pos[1], z0=pos[2], omega=om, phi=ph, kappa=ka)

        cal = Calibration(
            ext_par=ext,
            int_par=Interior(xh=0.0, yh=0.0, cc=cc),
            added_par=AddedPar(
                k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, scx=1.0, she=0.0, field=0
            ),
            glass_par=Glass(
                vec_x=0.0,
                vec_y=0.0,
                vec_z=_GLASS_MAG,
                n1=N_AIR,
                n2=N_GLASS,
                n3=N_WATER,
                d=glass_thickness,
            ),
        )

        # __post_init__ recomputed dm from the stored angles above; since the
        # angles reproduce the base rotation exactly this matches, but enforce
        # it explicitly to be safe.
        cal.ext_par.dm = rot.copy()

        # Align the glass normal along the camera's viewing direction so that
        # refraction is physical (non-zero +Z/-Z vector maintained).
        view_z = rot[2, :]
        gv = np.array(glass_vec, dtype=np.float64)
        gv_mag = np.linalg.norm(gv)
        if gv_mag < 1e-9:
            gv = np.array([0.0, 0.0, -1.0])
            gv_mag = 1.0
        glass_n = gv / gv_mag
        if np.dot(view_z, glass_n) < 0:
            glass_n = -glass_n
        cal.glass_par.vec_x = glass_n[0] * _GLASS_MAG
        cal.glass_par.vec_y = glass_n[1] * _GLASS_MAG
        cal.glass_par.vec_z = glass_n[2] * _GLASS_MAG

        cals.append(cal)

    rig = CameraRig(cals=cals, cpar=cpar, vpar=vpar, refract=refract)

    if refract and use_mmlut:
        prepare_mmluts(vpar, cpar, cals)

    return rig


def project_to_pixels(
    rig: CameraRig, points3d: np.ndarray, flat: bool = False
) -> list[np.ndarray]:
    """Project 3D points to per-camera pixel coordinates.

    Parameters
    ----------
    rig : CameraRig
    points3d : ndarray (N, 3)
        3D world coordinates [mm].
    flat : bool
        If True use the flat (undistorted) projection; otherwise the distorted
        metric path (which also honours the mmlut multimedia correction).

    Returns
    -------
    list[np.ndarray]
        One (N, 2) pixel-coordinate array per camera.
    """
    pts = np.ascontiguousarray(points3d, dtype=np.float64)
    out = []
    for cal in rig.cals:
        if flat:
            metric = flat_image_coord_batch(pts, cal, rig.cpar.mm)
        else:
            metric = img_coord_batch(pts, cal, rig.cpar.mm)
        out.append(metric_to_pixel_batch(metric, rig.cpar))
    return out


__all__ = [
    "CameraRig",
    "make_standard_rig",
    "project_to_pixels",
    "N_AIR",
    "N_GLASS",
    "N_WATER",
]

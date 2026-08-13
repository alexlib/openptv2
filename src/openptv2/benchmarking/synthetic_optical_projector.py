"""Render synthetic multi-camera PTV images from 3D particle positions.

Given a :class:`~openptv2.benchmarking.camera_rig.CameraRig` (which already
handles pinhole projection, lens distortion, and air-glass-water multimedia
refraction -- see :mod:`openptv2.algorithms.imgcoord` /
:mod:`openptv2.algorithms.multimed`) and a frame of 3D particle positions,
renders one 2D intensity image per camera: Gaussian point-spread-function
blobs, laser-sheet Gaussian intensity attenuation away from the sheet plane,
Gaussian sensor noise, and spurious ghost particles.

Prototyped interactively in a live marimo notebook (PSF splatting, laser
sheet falloff, ghosts, noise) before landing here wired to the real
camera model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openptv2.benchmarking.camera_rig import CameraRig, project_to_pixels


@dataclass
class RenderConfig:
    """Rendering knobs for :func:`render_frame`."""

    peak_intensity: float = 220.0
    psf_sigma: float = 1.3
    patch_radius: int = 4
    sheet_center: float = 0.0
    sheet_sigma: float = 5.0
    noise_sigma: float = 3.0
    ghost_ratio: float = 0.0
    seed: int = 0


def laser_sheet_attenuation(z: np.ndarray, sheet_center: float = 0.0, sheet_sigma: float = 5.0) -> np.ndarray:
    """Gaussian laser-sheet intensity falloff away from the sheet plane."""
    return np.exp(-0.5 * ((np.asarray(z) - sheet_center) / sheet_sigma) ** 2)


def render_particles(
    px: np.ndarray,
    py: np.ndarray,
    intensity: np.ndarray,
    image_size: tuple[int, int],
    psf_sigma: float = 1.3,
    patch_radius: int = 4,
) -> np.ndarray:
    """Splat 2D pixel-plane particles onto an image via Gaussian PSF.

    Parameters
    ----------
    px, py, intensity : ndarray (N,)
        Pixel coordinates and peak intensity (already laser-sheet scaled).
    image_size : (width, height)

    Returns
    -------
    ndarray (height, width)
    """
    w, h = image_size
    img = np.zeros((h, w), dtype=np.float64)
    r = patch_radius
    for x, y, amp in zip(px, py, intensity):
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += amp * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * psf_sigma**2)
        )
    return img


def add_ghost_particles(
    px: np.ndarray,
    py: np.ndarray,
    intensity: np.ndarray,
    image_size: tuple[int, int],
    ghost_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append spurious particles at random pixel locations (no 3D truth)."""
    n_ghost = int(len(px) * ghost_ratio)
    if n_ghost == 0:
        return px, py, intensity
    w, h = image_size
    gx = rng.uniform(0, w, n_ghost)
    gy = rng.uniform(0, h, n_ghost)
    gi = (
        rng.uniform(intensity.min(), intensity.max(), n_ghost)
        if len(intensity)
        else np.full(n_ghost, 150.0)
    )
    return np.concatenate([px, gx]), np.concatenate([py, gy]), np.concatenate([intensity, gi])


def add_sensor_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = img + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255)


def render_frame(
    rig: CameraRig,
    points3d: np.ndarray,
    config: RenderConfig = RenderConfig(),
) -> list[np.ndarray]:
    """Render one synthetic multi-camera frame from 3D particle positions.

    Projects ``points3d`` through the rig's real camera model (distortion +
    multimedia refraction), applies laser-sheet attenuation by particle depth
    (``z`` in world coordinates), splats Gaussian PSF blobs, adds ghost
    particles and sensor noise -- independently per camera.

    Returns
    -------
    list[ndarray]
        One ``(height, width)`` intensity image per camera.
    """
    rng = np.random.default_rng(config.seed)
    pts = np.ascontiguousarray(points3d, dtype=np.float64)
    z = pts[:, 2]
    amp = config.peak_intensity * laser_sheet_attenuation(z, config.sheet_center, config.sheet_sigma)

    pixel_coords = project_to_pixels(rig, pts)
    image_size = (rig.cpar.imx, rig.cpar.imy)

    images = []
    for cam_px in pixel_coords:
        px, py, cam_amp = add_ghost_particles(
            cam_px[:, 0], cam_px[:, 1], amp, image_size, config.ghost_ratio, rng
        )
        img = render_particles(px, py, cam_amp, image_size, config.psf_sigma, config.patch_radius)
        if config.noise_sigma > 0:
            img = add_sensor_noise(img, config.noise_sigma, rng)
        images.append(img)
    return images


__all__ = [
    "RenderConfig",
    "laser_sheet_attenuation",
    "render_particles",
    "add_ghost_particles",
    "add_sensor_noise",
    "render_frame",
]

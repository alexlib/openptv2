"""Tests for plate detection and labeling."""

import numpy as np
import pytest

from openptv2.detect_plate import PlateDetectionResult, _classify_coded
from openptv2.plate_labeler import label_coded_6x7, label_plate, label_uncoded_grid


@pytest.mark.unit
def test_label_coded_6x7_synthetic():
    """Verify label_coded_6x7 on synthetic 6x7 grid with 3 coded dots."""
    nx, ny = 6, 7
    pitch = 120.0

    # Create grid in world (X, Y)
    grid_xy = []
    for iy in range(ny):
        for ix in range(nx):
            grid_xy.append([ix * pitch, iy * pitch])
    grid_xy = np.array(grid_xy, dtype=float)

    # Synthetic perspective / projective warp
    H = np.array([
        [1.2, 0.1, 500.0],
        [-0.05, 1.1, 400.0],
        [0.0001, 0.00005, 1.0],
    ])

    # Project grid to image
    gh = np.column_stack([grid_xy, np.ones(len(grid_xy))])
    proj = (gh @ H.T)
    img_pts = proj[:, :2] / proj[:, 2:3]

    # Place coded dots at (ix=3, iy=3), (ix=3, iy=2), (ix=1, iy=3)
    coded_mask = np.zeros(len(img_pts), dtype=bool)
    for i, (ix, iy) in enumerate([(ix, iy) for iy in range(ny) for ix in range(nx)]):
        if (ix, iy) in [(3, 3), (3, 2), (1, 3)]:
            coded_mask[i] = True

    assert np.sum(coded_mask) == 3

    img_labelled, ref_labelled, idx = label_coded_6x7(
        img_pts,
        coded_mask,
        pitch_x=pitch,
        pitch_y=pitch,
        nx=nx,
        ny=ny,
        y_sign=1,
    )

    assert len(img_labelled) == 42
    assert len(ref_labelled) == 42
    assert len(idx) == 42
    # Verify index spans entire grid
    assert len(set(map(tuple, idx))) == 42


@pytest.mark.unit
def test_label_uncoded_grid_synthetic():
    """Verify label_uncoded_grid on regular 25x19 grid."""
    nx, ny = 25, 19
    pitch = 40.0

    grid = []
    for iy in range(ny):
        for ix in range(nx):
            grid.append([ix * pitch + 100.0, iy * pitch + 100.0])
    pts = np.array(grid, dtype=float)

    img_pts, ref_pts, idx = label_uncoded_grid(pts, pitch_x=pitch, pitch_y=pitch, nx=nx, ny=ny)
    assert len(img_pts) == nx * ny
    assert len(ref_pts) == nx * ny

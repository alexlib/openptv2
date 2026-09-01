"""Tests for plate detection and labeling."""

import numpy as np
import pytest

from openptv2.algorithms.tracking_frame_buf import Target
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
    H = np.array(
        [
            [1.2, 0.1, 500.0],
            [-0.05, 1.1, 400.0],
            [0.0001, 0.00005, 1.0],
        ]
    )

    # Project grid to image
    gh = np.column_stack([grid_xy, np.ones(len(grid_xy))])
    proj = gh @ H.T
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

    img_pts, ref_pts, idx = label_uncoded_grid(
        pts, pitch_x=pitch, pitch_y=pitch, nx=nx, ny=ny
    )
    assert len(img_pts) == nx * ny
    assert len(ref_pts) == nx * ny


@pytest.mark.unit
def test_detect_plate_targets_keeps_pnr1_when_multiple_found(monkeypatch):
    """Regression for sentinel bug: real pnr==1 must not be dropped when ≥2 found.

    target_recognition returns pnr 0..n-1; pnr==1 is the second real dot.
    The old filter ``pnr != 1 or len==1`` dropped it whenever n≥2.
    """
    from unittest.mock import Mock

    import openptv2.detect_plate as dp

    # Two real targets, pnr 0 and 1 — neither is the dummy (x=y=1,n=1 is dummy)
    t0 = Target(pnr=0, x=100.0, y=100.0, n=10, nx=5, ny=5, sumg=1000, tnr=0)
    t1 = Target(pnr=1, x=200.0, y=100.0, n=12, nx=6, ny=6, sumg=1100, tnr=0)

    monkeypatch.setattr(
        "openptv2.segmentation.target_recognition", lambda *a, **kw: [t0, t1]
    )
    monkeypatch.setattr(
        "openptv2.image_scaling.to_uint8",
        lambda img, *a, **kw: np.zeros((20, 20), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "openptv2.image_processing.preprocess_image", lambda img, *a, **kw: img
    )
    monkeypatch.setattr(dp, "find_plate_roi", lambda *a, **kw: (1, 10, 1, 10))
    monkeypatch.setattr(dp, "_classify_coded", lambda *a, **kw: np.zeros(2, dtype=bool))

    cpar = Mock()
    cpar.negative = False
    cpar.hp_flag = 1
    cpar.get_hp_flag = Mock(return_value=1)
    tpar = Mock()
    img = np.zeros((20, 20), dtype=np.uint8)
    res = dp.detect_plate_targets(
        img, tpar, cpar, cam=0, use_roi=False, scaling={"mode": "stretch"}
    )
    assert len(res.targets) == 2
    assert [t.pnr for t in res.targets] == [0, 1]


@pytest.mark.unit
def test_detect_plate_targets_drops_single_dummy(monkeypatch):
    """Single dummy (n_found==0 sentinel pnr=1,x=y=1,n=1) must become empty."""
    import openptv2.detect_plate as dp

    dummy = Target(pnr=1, x=1.0, y=1.0, n=1, nx=1, ny=1, sumg=1, tnr=-1)
    monkeypatch.setattr(
        "openptv2.segmentation.target_recognition", lambda *a, **kw: [dummy]
    )
    monkeypatch.setattr(
        "openptv2.image_scaling.to_uint8",
        lambda img, *a, **kw: np.zeros((20, 20), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "openptv2.image_processing.preprocess_image", lambda img, *a, **kw: img
    )
    monkeypatch.setattr(dp, "find_plate_roi", lambda *a, **kw: (1, 10, 1, 10))
    monkeypatch.setattr(dp, "_classify_coded", lambda *a, **kw: np.zeros(0, dtype=bool))

    from unittest.mock import Mock

    cpar = Mock()
    cpar.negative = False
    cpar.hp_flag = 1
    tpar = Mock()
    img = np.zeros((20, 20), dtype=np.uint8)
    res = dp.detect_plate_targets(
        img, tpar, cpar, cam=0, use_roi=False, scaling={"mode": "stretch"}
    )
    assert len(res.targets) == 0

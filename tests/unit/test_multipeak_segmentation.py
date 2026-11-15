"""Unit tests for multi-peak 2D target detection splitting in segmentation."""

import numpy as np

from openptv2.algorithms.segmentation import CORRES_NONE, targ_rec


def _make_gaussian_image(
    shape: tuple[int, int],
    centers: list[tuple[float, float, float]],
    sigma: float = 1.2,
) -> np.ndarray:
    """Render 2D uint8 image with one or more Gaussian blobs.

    Each entry in ``centers`` is ``(x_center, y_center, amplitude)`` in pixel-center
    coordinates (where pixel ``(ix, iy)`` has center ``(ix + 0.5, iy + 0.5)``).
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    xc = xx.astype(np.float64) + 0.5
    yc = yy.astype(np.float64) + 0.5
    acc = np.zeros((h, w), dtype=np.float64)
    for cx, cy, amp in centers:
        acc += amp * np.exp(-((xc - cx) ** 2 + (yc - cy) ** 2) / (2.0 * sigma**2))
    return np.clip(np.rint(acc), 0, 255).astype(np.uint8)


def test_single_gaussian_blob_identical_count_and_centroid():
    """Isolated single Gaussian blob gives 1 target with multipeak off and on."""
    cx, cy = 16.3, 14.7
    img = _make_gaussian_image((32, 32), [(cx, cy, 220.0)], sigma=1.2)

    params = dict(
        gvthres=30,
        discont=5,
        nnmin=2,
        nnmax=100,
        nxmin=1,
        nxmax=20,
        nymin=1,
        nymax=20,
        sumg_min=50,
    )

    t_off = targ_rec(img, **params, multipeak=False)
    t_on = targ_rec(img, **params, multipeak=True)

    assert len(t_off) == 1
    assert len(t_on) == 1
    assert t_on[0].tnr == CORRES_NONE
    assert abs(t_on[0].x - cx) < 0.15
    assert abs(t_on[0].y - cy) < 0.15
    assert abs(t_on[0].x - t_off[0].x) < 0.1
    assert abs(t_on[0].y - t_off[0].y) < 0.1


def test_overlapping_gaussian_blobs_split_into_two_targets():
    """Two touching/overlapping Gaussian blobs (3.0 px apart) split when multipeak=True."""
    c1 = (14.5, 16.5, 220.0)
    c2 = (17.5, 16.5, 210.0)  # 3.0 px separation (> 2*sigma -> 2 maxima, merged above gvthres)
    img = _make_gaussian_image((32, 32), [c1, c2], sigma=1.15)

    params = dict(
        gvthres=30,
        discont=80,
        nnmin=2,
        nnmax=200,
        nxmin=1,
        nxmax=30,
        nymin=1,
        nymax=30,
        sumg_min=50,
    )

    # Default single-component detection merges touching blobs into 1 target.
    t_default = targ_rec(img, **params)
    assert len(t_default) == 1

    # Multi-peak splitting resolves both blobs with accurate sub-pixel centroids.
    t_split = targ_rec(img, **params, multipeak=True)
    assert len(t_split) == 2

    detected = sorted([(t.x, t.y) for t in t_split], key=lambda p: p[0])
    assert abs(detected[0][0] - c1[0]) < 0.35
    assert abs(detected[0][1] - c1[1]) < 0.2
    assert abs(detected[1][0] - c2[0]) < 0.35
    assert abs(detected[1][1] - c2[1]) < 0.2


def test_diagonal_overlapping_blobs_split():
    """Two diagonally touching Gaussian blobs (~3.1 px apart) split cleanly."""
    c1 = (14.2, 14.2, 220.0)
    c2 = (16.4, 16.4, 215.0)  # ~3.11 px separation (> 2*sigma -> 2 maxima, merged above gvthres)
    img = _make_gaussian_image((32, 32), [c1, c2], sigma=1.15)

    params = dict(
        gvthres=30,
        discont=80,
        nnmin=2,
        nnmax=200,
        nxmin=1,
        nxmax=30,
        nymin=1,
        nymax=30,
        sumg_min=50,
    )

    t_default = targ_rec(img, **params, multipeak=False)
    assert len(t_default) == 1

    t_split = targ_rec(img, **params, multipeak=True)
    assert len(t_split) == 2

    detected = sorted([(t.x, t.y) for t in t_split], key=lambda p: p[0])
    assert abs(detected[0][0] - c1[0]) < 0.4
    assert abs(detected[0][1] - c1[1]) < 0.4
    assert abs(detected[1][0] - c2[0]) < 0.4
    assert abs(detected[1][1] - c2[1]) < 0.4


def test_noise_prominence_robustness_no_spurious_split():
    """Single noisy Gaussian blob is not spuriously split into multiple targets."""
    rng = np.random.default_rng(42)
    base = _make_gaussian_image((32, 32), [(16.5, 16.5, 200.0)], sigma=2.2).astype(
        np.float64
    )
    # Add small pixel-level noise around the broad peak
    noise = rng.normal(loc=0.0, scale=4.0, size=base.shape)
    noisy = np.clip(np.rint(base + noise), 0, 255).astype(np.uint8)

    params = dict(
        gvthres=40,
        discont=5,
        nnmin=3,
        nnmax=200,
        nxmin=1,
        nxmax=30,
        nymin=1,
        nymax=30,
        sumg_min=100,
    )

    t_on = targ_rec(noisy, **params, multipeak=True)
    assert len(t_on) == 1
    assert abs(t_on[0].x - 16.5) < 0.3
    assert abs(t_on[0].y - 16.5) < 0.3


def test_backward_compatibility_default_unchanged():
    """Calling targ_rec without multipeak matches multipeak=False identically."""
    img = _make_gaussian_image(
        (32, 32),
        [(10.5, 10.5, 210.0), (13.5, 10.5, 200.0), (22.0, 22.0, 230.0)],
        sigma=1.0,
    )
    params = dict(
        gvthres=30,
        discont=5,
        nnmin=2,
        nnmax=200,
        nxmin=1,
        nxmax=30,
        nymin=1,
        nymax=30,
        sumg_min=50,
    )

    t_default = targ_rec(img, **params)
    t_explicit_off = targ_rec(img, **params, multipeak=False)

    assert len(t_default) == len(t_explicit_off) == 3
    for a, b in zip(t_default, t_explicit_off):
        assert a.pnr == b.pnr
        assert a.x == b.x
        assert a.y == b.y
        assert a.n == b.n
        assert a.nx == b.nx
        assert a.ny == b.ny
        assert a.sumg == b.sumg
        assert a.tnr == b.tnr

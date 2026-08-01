import numpy as np

from openptv2.algorithms.segmentation import CORRES_NONE, targ_rec


def test_peak_fit():
    img = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            255,
            255,
            255,
            0,
            0,
            255,
            255,
            255,
            0,
            0,
            255,
            255,
            255,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint8,
    ).reshape(5, 5)
    # Parameters matching the C test
    gvthres = 250
    discont = 5
    nnmin = 1
    nnmax = 10
    nxmin = 1
    nxmax = 10
    nymin = 1
    nymax = 10
    sumg_min = 12
    # Run detection
    from openptv2.algorithms.segmentation import peak_fit

    targets = peak_fit(
        img, gvthres, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min
    )
    assert len(targets) == 1
    assert targets[0].n == 9

    # Two objects
    img1 = np.array(
        [0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 251, 0, 0, 0, 0, 0, 0],
        dtype=np.uint8,
    ).reshape(5, 5)

    targets2 = peak_fit(
        img1, gvthres, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min
    )
    assert len(targets2) == 2

    # Change threshold to exclude second object
    gvthres2 = 252
    targets3 = peak_fit(
        img1, gvthres2, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min
    )
    assert len(targets3) == 1


def test_targ_rec():
    img = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            255,
            255,
            255,
            0,
            0,
            255,
            255,
            255,
            0,
            0,
            255,
            255,
            255,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint8,
    ).reshape(5, 5)
    # Parameters matching the C test
    gvthres = 250
    discont = 5
    nnmin = 1
    nnmax = 10
    nxmin = 1
    nxmax = 10
    nymin = 1
    nymax = 10
    sumg_min = 12

    # Run detection
    targets = targ_rec(
        img, gvthres, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min
    )
    assert len(targets) == 1
    assert targets[0].n == 9
    assert targets[0].tnr == CORRES_NONE

    # Two objects
    img1 = np.array(
        [0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 251, 0, 0, 0, 0, 0, 0],
        dtype=np.uint8,
    ).reshape(5, 5)

    targets2 = targ_rec(
        img1, gvthres, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min
    )
    assert len(targets2) == 2

    # Change threshold to exclude second object
    gvthres2 = 252
    targets3 = targ_rec(
        img1, gvthres2, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min
    )
    assert len(targets3) == 1


def test_targ_rec_matches_c_dummy_target_when_no_particle_found():
    img = np.zeros((5, 5), dtype=np.uint8)
    targets = targ_rec(img, 250, 5, 1, 10, 1, 10, 1, 10, 12)

    assert len(targets) == 1
    assert targets[0].pnr == 1
    assert targets[0].x == 1
    assert targets[0].y == 1

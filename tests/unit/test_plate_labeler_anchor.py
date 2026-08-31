"""The coded-plate labeller must anchor on the coded corner, not on what it saw.

Regression test for the failure that wasted half the Illmenau calibration
frames: without ``corner_index`` the grid is re-anchored to ``min(ix)`` over the
dots a given view happened to detect, so any view missing the leftmost column or
the bottom row labels every dot one step off.  The result is wrong yet perfectly
self-consistent -- a per-camera PnP fit of the rigid plate to those labels has a
sub-pixel residual -- so it survives every single-view check and only surfaces
much later as a cross-camera disagreement.
"""
import numpy as np
import pytest

from openptv2.plate_labeler import _identify_L, label_coded_6x7

NX, NY, PITCH = 6, 7, 120.0
CORNER = (2, 3)          # grid index of the coded L corner on the Illmenau plate


def synth_plate(*, drop_left_column=False, scale=7.0, origin=(400.0, 1300.0)):
    """A 6x7 grid in image space, +Y up the plate mapping to -y in pixels."""
    pts, idx = [], []
    for iy in range(NY):
        for ix in range(NX):
            if drop_left_column and ix == 0:
                continue
            pts.append([origin[0] + ix * scale * PITCH / 10.0,
                        origin[1] - iy * scale * PITCH / 10.0])
            idx.append((ix, iy))
    pts = np.array(pts, float)
    idx = np.array(idx, int)
    # the three coded dots: corner, +1 pitch in +Y, +2 pitch in +X
    coded = np.zeros(len(pts), bool)
    for want in (CORNER, (CORNER[0], CORNER[1] + 1), (CORNER[0] + 2, CORNER[1])):
        (k,) = np.where((idx[:, 0] == want[0]) & (idx[:, 1] == want[1]))
        coded[k[0]] = True
    return pts, coded, idx


def label(pts, coded, **kw):
    _, _, out = label_coded_6x7(pts, coded, pitch_x=PITCH, pitch_y=PITCH,
                                nx=NX, ny=NY, y_sign=1, **kw)
    return out


def test_full_grid_labels_the_same_either_way():
    """With every dot visible, the detected extent and the corner agree."""
    pts, coded, idx = synth_plate()
    got = label(pts, coded, corner_index=CORNER)
    assert len(got) == NX * NY
    np.testing.assert_array_equal(np.sort(got, axis=0), np.sort(idx, axis=0))
    # the fallback anchoring happens to be right here -- that is why the bug hid
    np.testing.assert_array_equal(np.sort(label(pts, coded), axis=0),
                                  np.sort(got, axis=0))


def test_missing_left_column_shifts_every_index_without_corner_index():
    """The bug: a view that cannot see column 0 relabels the whole plate."""
    pts, coded, idx = synth_plate(drop_left_column=True)
    fallback = label(pts, coded)
    assert len(fallback) == len(idx)
    # every ix comes back one too small -- self-consistent, and completely wrong
    assert fallback[:, 0].min() == 0
    assert idx[:, 0].min() == 1
    assert not np.array_equal(np.sort(fallback, axis=0), np.sort(idx, axis=0))


def test_corner_index_survives_a_missing_column():
    """The fix: anchoring on the coded dot ignores which dots were detected."""
    pts, coded, idx = synth_plate(drop_left_column=True)
    got = label(pts, coded, corner_index=CORNER)
    assert len(got) == len(idx)
    np.testing.assert_array_equal(np.sort(got, axis=0), np.sort(idx, axis=0))


@pytest.mark.parametrize("hint", [(0.0, -1.0), (0.0, 1.0), (0.3, -0.95)])
def test_up_hint_decides_which_way_is_up(hint):
    """Whatever the hint, the chosen +Y lies in the hint's half-plane.

    That is the whole contract.  The three coded dots alone do not settle it:
    taking the ``+1*pitch`` dot as the corner also gives legs at a near-1:2
    ratio and a near-right angle, so the geometric score has a second, spurious
    optimum with +Y reversed.  Under strong perspective that one can win, which
    rotates the entire grid.  A hint from the calibration removes the choice.
    """
    pts, coded, _ = synth_plate()
    h = np.array(hint, float)
    h /= np.linalg.norm(h)
    _, _, e_y = _identify_L(pts[coded], PITCH, up_hint=h)
    assert float(np.dot(e_y, h)) > 0.0


def test_up_hint_agreeing_with_the_geometry_changes_nothing():
    """The true +Y runs up the plate (towards -y in pixels); a hint along it is
    redundant and must leave the unhinted answer exactly as it was."""
    pts, coded, _ = synth_plate()
    _, _, e_y = _identify_L(pts[coded], PITCH)
    assert e_y[1] < 0
    _, _, e_y_hint = _identify_L(pts[coded], PITCH, up_hint=np.array([0.0, -1.0]))
    np.testing.assert_allclose(e_y_hint, e_y, atol=1e-9)

"""Headless checks for the Tk/matplotlib migration pilot (Phase 0).

Verifies the machine-checkable parts of the pilot: the figure builds with the
image + overlay artists, and the click->pixel transform round-trips exactly.
The interactive click/visual-alignment part is a human checkpoint (P0).
"""
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"


@pytest.mark.gui
def test_pilot_selftest_builds_and_maps_pixels():
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    if not (FIX / "cal" / "cam1.tif").exists():
        pytest.skip("test_cavity calibration image not available")

    import matplotlib
    matplotlib.use("Agg")
    from openptv2.gui_tk.pilot import _build_figure, _load_scene

    img, det, matched, res = _load_scene(FIX)
    assert img.ndim == 2 and img.size > 0
    assert len(det) > 0
    assert len(matched) == len(res)

    fig, ax = _build_figure(img, det, matched, res)
    assert len(ax.get_images()) == 1          # the camera image
    assert len(ax.collections) >= 1           # detected/matched/quiver overlays

    # click->pixel: imshow default extent makes data coords == image pixels, so
    # a display round-trip must return the same pixel (what the click handler uses)
    disp = ax.transData.transform(det[:20])
    back = ax.transData.inverted().transform(disp)
    assert np.abs(back - det[:20]).max() < 1e-6

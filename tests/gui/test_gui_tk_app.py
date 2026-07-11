"""Headless (Xvfb) smoke tests for the Tk app skeleton.

Run under a virtual display, e.g.:
    xvfb-run -a uv run pytest tests/gui/test_gui_tk_app.py

Verifies the machine-checkable wiring: window/tree/grid build, cross-view click
propagation via the bus, and the parameter edit→save round-trip. Visual look/feel
and overlay alignment remain the human checkpoint (C1).
"""
import os
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"


@pytest.fixture
def app():
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio")
    tk = pytest.importorskip("tkinter")
    if not os.environ.get("DISPLAY"):
        pytest.skip("no DISPLAY (run under xvfb-run)")
    if not (FIX / "parameters_Run1.yaml").exists():
        pytest.skip("test_cavity fixture not available")
    import matplotlib
    matplotlib.use("Agg")  # figure backend; Tk provides the window
    from openptv2.gui_tk.app import MainWindow

    root = tk.Tk()
    win = MainWindow(root, str(FIX))
    root.update_idletasks()
    yield win
    root.destroy()


def test_window_builds_with_camera_grid_and_tree(app):
    assert len(app.views) == app.num_cams == 4
    assert app.tree.get_children()          # parameter sections present
    assert set(app._menus) >= {"File", "Run", "Calibration", "Windows"}


def test_camera_click_propagates_to_other_views(app):
    from openptv2.gui_tk.events import CameraClick

    app.bus.publish(CameraClick(cam=0, x=123.0, y=45.0))
    # clicked view + all others got exactly one overlay marker
    for cam, view in app.views.items():
        assert len(view._overlays) == 1, f"cam{cam} overlay missing"
    assert "123" in app.status.cget("text")


def test_param_edit_and_save_roundtrip(app):
    from openptv2.gui_tk.events import ParamsChanged

    got = []
    app.bus.subscribe(ParamsChanged, got.append)
    # select the 'track' section node and open its editor
    for node, sec in app._tree_section.items():
        if sec == "track":
            app.tree.selection_set(node)
            break
    app._edit_selected_section()
    form = app._last_form
    vals = form.get_values()
    assert "dvxmax" in vals
    vals_saved = form.save()
    assert got and got[0].section == "track"
    assert app.pm.parameters["track"]["dvxmax"] == vals_saved["dvxmax"]


def test_calibration_and_3d_windows_open(app):
    app.open_calibration()
    app.open_view3d()
    # status reflects success (not an exception message)
    assert "opened" in app.status.cget("text") or "3D" in app.status.cget("text")

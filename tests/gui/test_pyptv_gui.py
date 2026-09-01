"""Characterization tests for openptv2.gui.pyptv_gui (MainGUI, TreeMenuHandler,
CameraWindow).

Like calibration_gui.py, this file had zero direct test coverage despite
being the main application window (2200+ lines, 9 bug-fixes in 6 months,
60 file-level dependents on gui/ptv.py which it drives). MainGUI is
instantiable headless -- __init__ only needs a yaml_file Path and an
Experiment, and traits/chaco objects don't need a live GUI toolkit until
actually rendered -- and TreeMenuHandler's action methods only touch
``info.object`` (the MainGUI instance) for the actions exercised here, so a
plain object standing in for traitsui's UIInfo is enough to drive them
without a live UI.

calib_action/detection_gui_action/open_action are NOT exercised here: they
call .configure_traits(), which opens a real GUI window and would hang a
headless test run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openptv2.gui.experiment import Experiment
from openptv2.gui.parameter_manager import ParameterManager
from openptv2.gui.pyptv_gui import CameraWindow, MainGUI, TreeMenuHandler

CAVITY_SRC = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    # Exclude res/: it's gitignored and accumulates real run output (run.zarr,
    # rt_is.*) across manual/test runs against the checked-out fixture, so a
    # verbatim copy makes fixture behavior depend on whatever stale state
    # happens to be sitting there. MainGUI creates a fresh res/ on init.
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("res"))


class _Info:
    """Minimal stand-in for traitsui's UIInfo: TreeMenuHandler action methods
    only ever read info.object (the model) and, on the error path, info.ui.control."""

    def __init__(self, obj):
        self.object = obj
        self.ui = MagicMock()


@pytest.fixture
def cavity_gui(tmp_path):
    """A real MainGUI wired to a throwaway copy of test_cavity."""
    if not CAVITY_SRC.exists():
        pytest.skip("test_cavity fixture not present")
    work = tmp_path / "test_cavity"
    _copy_tree(CAVITY_SRC, work)

    original_cwd = Path.cwd()
    try:
        yaml_path = (work / "parameters_Run1.yaml").resolve()
        pm = ParameterManager()
        pm.from_yaml(yaml_path)
        exp = Experiment(pm=pm)
        exp.populate_runs(work, active_yaml=yaml_path)
        os.chdir(work)
        gui = MainGUI(yaml_path, exp)
        yield gui
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_main_gui_instantiation(cavity_gui):
    """Construction loads the experiment, seeds num_cams, and creates one
    CameraWindow per camera with orig_images pre-allocated."""
    assert cavity_gui.num_cams == 4
    assert len(cavity_gui.camera_list) == 4
    assert len(cavity_gui.orig_images) == 4
    assert cavity_gui.pass_init is False
    assert cavity_gui.plugins is not None


@pytest.mark.unit
def test_main_gui_rejects_non_yaml_path(tmp_path):
    if not CAVITY_SRC.exists():
        pytest.skip("test_cavity fixture not present")
    work = tmp_path / "test_cavity"
    _copy_tree(CAVITY_SRC, work)
    original_cwd = Path.cwd()
    try:
        pm = ParameterManager()
        pm.from_yaml((work / "parameters_Run1.yaml").resolve())
        exp = Experiment(pm=pm)
        not_yaml = work / "cal" / "cam1.tif"
        assert not_yaml.exists()
        with pytest.raises(ValueError):
            MainGUI(not_yaml, exp)
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
class TestCameraWindowGeometry:
    """CameraWindow.remove_short_lines / _clip_line_to_rect are pure
    geometry helpers with no chaco/GUI dependency."""

    def test_remove_short_lines_drops_lines_under_threshold(self):
        # (100,100)->(100,102): dx=0,dy=2, not > the dx=2,dy=2 threshold -> dropped.
        # (200,200)->(200,210): dy=10 > 2 -> kept.
        # (300,300)->(320,300): dx=20 > 2 -> kept.
        x1, y1, x2, y2 = CameraWindow.remove_short_lines(
            [100, 200, 300], [100, 200, 300], [100, 200, 320], [102, 210, 300]
        )
        assert x1 == [200, 300]
        assert y1 == [200, 300]
        assert x2 == [200, 320]
        assert y2 == [210, 300]

    def test_remove_short_lines_empty_input(self):
        assert CameraWindow.remove_short_lines([], [], [], []) == ([], [], [], [])

    def test_remove_short_lines_keeps_all_long_lines(self):
        x1, y1, x2, y2 = CameraWindow.remove_short_lines([0], [0], [50], [50])
        assert (x1, y1, x2, y2) == ([0], [0], [50], [50])

    def test_clip_line_to_rect_fully_inside_is_unchanged(self):
        result = CameraWindow._clip_line_to_rect(10, 10, 50, 50, width=100, height=100)
        assert result == (10, 10, 50, 50)

    def test_clip_line_to_rect_clips_to_bounds(self):
        # Line from inside to far outside the right edge must clip at width-1.
        result = CameraWindow._clip_line_to_rect(50, 50, 500, 50, width=100, height=100)
        assert result is not None
        nx1, ny1, nx2, ny2 = result
        assert nx1 == pytest.approx(50)
        assert nx2 == pytest.approx(99)
        assert ny1 == pytest.approx(50)
        assert ny2 == pytest.approx(50)

    def test_clip_line_to_rect_fully_outside_returns_none(self):
        # Entirely to the right of a 100x100 image.
        result = CameraWindow._clip_line_to_rect(
            200, 50, 300, 50, width=100, height=100
        )
        assert result is None

    def test_clip_line_to_rect_zero_size_returns_none(self):
        assert (
            CameraWindow._clip_line_to_rect(0, 0, 10, 10, width=0, height=100) is None
        )
        assert (
            CameraWindow._clip_line_to_rect(0, 0, 10, 10, width=100, height=0) is None
        )


@pytest.mark.unit
def test_init_action_loads_images_and_params(cavity_gui):
    """init_action is the foundation every other menu action depends on:
    loads calibration images, resolves cpar/spar/vpar/tpar/cals/epar via
    ptv.py_start_proc_c, and populates target_filenames."""
    handler = TreeMenuHandler()
    info = _Info(cavity_gui)

    handler.init_action(info)

    assert cavity_gui.pass_init is True
    assert cavity_gui.cpar is not None
    assert cavity_gui.spar is not None
    assert cavity_gui.cals is not None
    assert len(cavity_gui.cals) == cavity_gui.num_cams
    assert cavity_gui.target_filenames is not None
    assert len(cavity_gui.target_filenames) == cavity_gui.num_cams
    assert (cavity_gui.exp_path / "res").is_dir()


@pytest.mark.unit
def test_sequence_action_runs_full_pipeline(cavity_gui):
    """sequence_action drives ptv.run_sequence_plugin(mainGui), the same
    "default" plugin path exercised by CLI/batch runs -- writes correspondences
    to the run.zarr store for every frame in the configured range (the
    ptv.py_sequence_loop determination step calls store.write_correspondences
    directly; it has not written legacy rt_is.* ASCII files since the
    RunStore migration -- see docs/plans/2026-08-14-storage-formats-as-built.md)."""
    from openptv2.storage import RunStore

    handler = TreeMenuHandler()
    info = _Info(cavity_gui)

    handler.init_action(info)
    handler.sequence_action(info)

    first = cavity_gui.spar.get_first()
    last = cavity_gui.spar.get_last()
    assert last > first  # sanity: a real multi-frame range
    store = RunStore.open(cavity_gui.exp_path, mode="r")
    for frame in range(first, last + 1):
        assert store.has_correspondences(frame)
        pos_3d, cam_ids = store.read_correspondences(frame)
        assert pos_3d.shape[0] > 0


@pytest.mark.unit
def test_track_no_disp_action_runs_and_sets_tracker(cavity_gui):
    """track_no_disp_action drives ptv.run_tracking_plugin(mainGui), which
    must set mainGui.tracker as a side effect -- track_back_action (and
    postprocessing) reads it after a forward-tracking run."""
    handler = TreeMenuHandler()
    info = _Info(cavity_gui)

    handler.init_action(info)
    handler.sequence_action(info)
    handler.track_no_disp_action(info)

    assert getattr(cavity_gui, "tracker", None) is not None

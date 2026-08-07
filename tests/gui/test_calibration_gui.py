"""Characterization tests for openptv2.gui.calibration_gui.CalibrationGUI.

CalibrationGUI had zero test coverage anywhere in the repo despite being a
919+ line class with 7 bug-fixes in the last 6 months (repowise health scan).
These tests pin down its current, real behavior against the test_cavity
fixture -- both to catch regressions and to make future refactoring of the
flagged brain method (_button_fine_orient_fired) safe.

CalibrationGUI.__init__ chdir()s into the dataset directory and its button
handlers write .ori/.addpar files in place, so every test works on a tmp_path
copy of the fixture and restores the original cwd afterward.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.gui.calibration_gui import CalibrationGUI

CAVITY_SRC = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@pytest.fixture
def cavity_gui(tmp_path):
    """A CalibrationGUI instantiated against a throwaway copy of test_cavity."""
    if not CAVITY_SRC.exists():
        pytest.skip("test_cavity fixture not present")
    work = tmp_path / "test_cavity"
    _copy_tree(CAVITY_SRC, work)

    original_cwd = Path.cwd()
    try:
        gui = CalibrationGUI(work / "parameters_Run1.yaml")
        yield gui
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_calibration_gui_instantiation(cavity_gui):
    """Construction loads the experiment, seeds num_cams, and creates one
    PlotWindow per camera -- the baseline every button handler depends on."""
    assert cavity_gui.num_cams == 4
    assert len(cavity_gui.camera) == 4
    assert cavity_gui.experiment is not None
    assert cavity_gui.detections is None
    assert cavity_gui._cal_splitter is False


@pytest.mark.unit
class TestParseExcludeIds:
    """_parse_exclude_ids: 'cam:id,cam:id' -> {0-indexed cam: {ids}},
    silently skipping malformed entries (runs on every keystroke)."""

    def test_empty_text_returns_empty_dict(self, cavity_gui):
        cavity_gui.exclude_ids_text = " "
        assert cavity_gui._parse_exclude_ids() == {}

    def test_single_pair(self, cavity_gui):
        cavity_gui.exclude_ids_text = "2:53"
        assert cavity_gui._parse_exclude_ids() == {1: {53}}

    def test_multiple_pairs_same_camera_accumulate(self, cavity_gui):
        cavity_gui.exclude_ids_text = "2:53,2:94"
        assert cavity_gui._parse_exclude_ids() == {1: {53, 94}}

    def test_multiple_cameras(self, cavity_gui):
        cavity_gui.exclude_ids_text = "2:53,4:94"
        assert cavity_gui._parse_exclude_ids() == {1: {53}, 3: {94}}

    def test_newlines_are_treated_as_separators(self, cavity_gui):
        cavity_gui.exclude_ids_text = "2:53\n4:94"
        assert cavity_gui._parse_exclude_ids() == {1: {53}, 3: {94}}

    def test_whitespace_around_pairs_is_stripped(self, cavity_gui):
        cavity_gui.exclude_ids_text = " 2 : 53 , 4:94 "
        assert cavity_gui._parse_exclude_ids() == {1: {53}, 3: {94}}

    @pytest.mark.parametrize(
        "text",
        ["not_a_pair", "2", ":53", "cam:notanumber", "2:53:extra"],
    )
    def test_malformed_entries_are_silently_skipped(self, cavity_gui, text):
        cavity_gui.exclude_ids_text = text
        # Must not raise -- the field is parsed on every Sortgrid click,
        # including mid-edit.
        result = cavity_gui._parse_exclude_ids()
        assert isinstance(result, dict)

    def test_malformed_entry_does_not_block_valid_siblings(self, cavity_gui):
        cavity_gui.exclude_ids_text = "garbage,2:53"
        assert cavity_gui._parse_exclude_ids() == {1: {53}}


@pytest.mark.unit
def test_backup_and_restore_ori_files_roundtrip(cavity_gui):
    """_backup_ori_files then restore_ori_files must bring BOTH .ori and
    .addpar back to their pre-mutation content.

    Regression test for a real bug: restore_ori_files' addpar line used to
    copy live -> .bck (the backup direction) instead of .bck -> live (the
    restore direction), so restoring never actually restored distortion
    parameters and destroyed the only backup of them in the process.
    """
    img_ori = cavity_gui.get_parameter("cal_ori")["img_ori"][: cavity_gui.num_cams]
    addpar_files = [f.replace("ori", "addpar") for f in img_ori]

    original_ori = [Path(f).read_bytes() for f in img_ori]
    original_addpar = [Path(f).read_bytes() for f in addpar_files]

    cavity_gui._backup_ori_files()

    # Mutate both files, as a real calibration attempt would before a restore.
    for f in img_ori:
        Path(f).write_text("mutated ori content\n")
    for f in addpar_files:
        Path(f).write_text("mutated addpar content\n")

    cavity_gui.restore_ori_files()

    for f, orig in zip(img_ori, original_ori):
        assert Path(f).read_bytes() == orig, f"{f} was not restored"
    for f, orig in zip(addpar_files, original_addpar):
        assert Path(f).read_bytes() == orig, f"{f} was not restored"


@pytest.mark.unit
def test_full_calibration_pipeline_button_sequence(cavity_gui):
    """End-to-end through the exact button sequence a user drives manually:
    load images -> detect -> load manual-orientation points from YAML ->
    sortgrid -> raw orientation -> fine orientation (the flagged 337-line,
    CCN-84 brain method). Exercises the real openptv2 calibration math on
    the real test_cavity fixture, which already has man_ori_coordinates.
    """
    gui = cavity_gui
    man_ori_coords = gui.experiment.pm.parameters.get("man_ori_coordinates")
    if not man_ori_coords:
        pytest.skip("test_cavity fixture has no man_ori_coordinates")

    gui._button_showimg_fired()
    assert gui.pass_init is True
    assert len(gui.cal_images) == gui.num_cams

    gui._button_detection_fired()
    assert gui.detections is not None
    assert len(gui.detections) == gui.num_cams
    assert all(len(cam_targs) > 0 for cam_targs in gui.detections)

    gui._button_file_orient_fired()
    for cam in gui.camera:
        assert len(cam._x) == 4
        assert len(cam._y) == 4

    gui._button_sort_grid_fired()
    assert gui.pass_sortgrid is True
    assert len(gui.sorted_targs) == gui.num_cams

    gui._button_raw_orient_fired()
    assert gui.pass_raw_orient is True

    # Cameras before fine orientation, for a before/after RMS comparison.
    cals_after_raw = [
        (cal.get_pos().copy(), cal.get_angles().copy()) for cal in gui.cals
    ]

    gui._button_fine_orient_fired()
    assert gui.status_text == "Orientation finished."

    # Fine orientation must actually move at least one camera's parameters
    # (a full_calibration run that changes nothing is a silent no-op bug).
    moved = any(
        not np.allclose(before_pos, cal.get_pos())
        or not np.allclose(before_ang, cal.get_angles())
        for (before_pos, before_ang), cal in zip(cals_after_raw, gui.cals)
    )
    assert moved, "fine orientation did not change any camera's parameters"

    # .ori files on disk must reflect the fitted calibration (_write_ori is
    # called per-camera inside _button_fine_orient_fired on success).
    img_ori = gui.get_parameter("cal_ori")["img_ori"][: gui.num_cams]
    for f in img_ori:
        assert Path(f).exists()
        assert Path(f).stat().st_size > 0

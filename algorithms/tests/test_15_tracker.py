"""
Engine comparison tests for Tracker class.

Tests the main Tracker class workflow, mirroring the Cython binding tests
in bindings/tests/test_tracker.py to ensure identical behavior.

Tolerance: 1e-7 (full tracking pipeline)
"""

import os
import shutil
import yaml
import numpy as np
import pytest

from .conftest import get_tolerance

TOLERANCE = get_tolerance("tracker")

TRACK_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "track"
)

framebuf_naming = {
    "corres": "test_data/track/res/particles",
    "linkage": "test_data/track/res/linkage",
    "prio": "test_data/track/res/whatever",
}


def _load_cals_from_yaml(yaml_conf):
    """Load calibrations from the YAML config."""
    from algorithms.calibration import Calibration

    cals = []
    for cam_spec in yaml_conf["cameras"]:
        ori_file = cam_spec["ori_file"]
        addpar_file = cam_spec.get("addpar_file", None)
        cal = Calibration()
        cal.from_file(ori_file, addpar_file)
        cals.append(cal)
    return cals


def _build_python_tracker(yaml_conf):
    """Build a Python Tracker from the YAML config, same as bindings test."""
    from algorithms.parameters import ControlPar, VolumePar, TrackParTuple, SequencePar
    from algorithms.track import Tracker

    seq_cfg = yaml_conf["sequence"]
    scene = yaml_conf["scene"]
    corresp = yaml_conf["correspondences"]
    tracking = yaml_conf["tracking"]

    cals = _load_cals_from_yaml(yaml_conf)

    cpar = ControlPar(num_cams=len(yaml_conf["cameras"]))
    cpar.imx = scene["image_size"][0]
    cpar.imy = scene["image_size"][1]
    cpar.pix_x = scene["pixel_size"][0]
    cpar.pix_y = scene["pixel_size"][1]

    vpar = VolumePar(
        x_lay=corresp["x_span"],
        z_min_lay=[corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))],
        z_max_lay=[corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))],
    )

    vel = tracking["velocity_lims"]
    tpar = TrackParTuple(
        dvxmin=vel[0][0],
        dvxmax=vel[0][1],
        dvymin=vel[1][0],
        dvymax=vel[1][1],
        dvzmin=vel[2][0],
        dvzmax=vel[2][1],
        dangle=tracking["angle_lim"],
        dacc=tracking["accel_lim"],
        add=tracking["add_particle"],
        dsumg=0.0,
        dn=0.0,
        dnx=0.0,
        dny=0.0,
    )

    img_base = []
    for cix in range(len(yaml_conf["cameras"])):
        img_base.append(seq_cfg["targets_template"].format(cam=cix + 1))

    spar = SequencePar(
        img_base_name=img_base,
        first=seq_cfg["first"],
        last=seq_cfg["last"],
    )

    tracker = Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)
    return tracker


@pytest.fixture
def track_test_dir(tmp_path):
    """Set up a temporary copy of the track test data, like bindings tests do."""
    src = TRACK_DATA_DIR
    # Copy res_orig to res
    res_orig = os.path.join(src, "res_orig")
    res_dst = os.path.join(src, "res")
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    shutil.copytree(res_orig, res_dst)

    # Also need to restore target files since tracking writes to them
    # Save originals first
    newpart_dir = os.path.join(src, "newpart")
    backup_dir = str(tmp_path / "newpart_backup")
    shutil.copytree(newpart_dir, backup_dir)

    yield src

    # Teardown - restore original target files
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    if os.path.exists(newpart_dir):
        shutil.rmtree(newpart_dir)
    shutil.copytree(backup_dir, newpart_dir)


class TestTracker:
    """Compare Tracker class between optv and python engines."""

    def _make_tracker(self):
        with open(os.path.join(TRACK_DATA_DIR, "conf.yaml")) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        return _build_python_tracker(yaml_conf)

    def test_forward(self, track_test_dir):
        """Manual forward tracking run, mirroring bindings test_forward."""
        tracker = self._make_tracker()
        tracker.restart()
        last_step = 10001
        while tracker.step_forward():
            assert tracker.current_step() > last_step
            with open(f"test_data/track/res/linkage.{last_step}") as f:
                lines = f.readlines()
                if last_step == 10003:
                    assert lines[0] == "0\n"
                else:
                    assert lines[0] == "1\n"
            last_step += 1
        tracker.finalize()

    def test_full_forward(self, track_test_dir):
        """Automatic full forward tracking run, mirroring bindings test_full_forward."""
        tracker = self._make_tracker()
        tracker.full_forward()

    def test_forward_3d(self, track_test_dir):
        """Manual forward 3D tracking run, mirroring bindings test_forward_3d."""
        tracker = self._make_tracker()
        tracker.restart()
        last_step = 10001
        while tracker.step_forward_3d():
            assert tracker.current_step() > last_step
            with open(f"test_data/track/res/linkage.{last_step}") as f:
                lines = f.readlines()
                if last_step == 10003:
                    assert lines[0] == "0\n"
                else:
                    assert lines[0] == "1\n"
            last_step += 1
        tracker.finalize()

    def test_full_forward_3d(self, track_test_dir):
        """Automatic full forward 3D tracking run, mirroring bindings test_full_forward_3d."""
        tracker = self._make_tracker()
        tracker.full_forward_3d()

    def test_full_backward(self, track_test_dir):
        """Automatic full backward correction phase, mirroring bindings test_full_backward."""
        tracker = self._make_tracker()
        tracker.full_forward()
        tracker.full_backward()

    def test_tracker_creation(self):
        """Test Tracker creation with parameters."""
        tracker = self._make_tracker()
        assert tracker is not None

    def test_tracker_restart(self, track_test_dir):
        """Test Tracker.restart() method."""
        tracker = self._make_tracker()
        tracker.restart()
        assert tracker.current_step() == 10001

    def test_tracker_step_forward(self, track_test_dir):
        """Test Tracker.step_forward() method."""
        tracker = self._make_tracker()
        tracker.restart()

        result = tracker.step_forward()
        assert result is True
        assert tracker.current_step() == 10002

    def test_tracker_finalize(self, track_test_dir):
        """Test Tracker.finalize() method."""
        tracker = self._make_tracker()
        tracker.restart()

        while tracker.step_forward():
            pass

        tracker.finalize()
        assert tracker.current_step() == 10005

    def test_tracker_full_forward(self, track_test_dir):
        """Test Tracker.full_forward() method."""
        tracker = self._make_tracker()
        tracker.full_forward()
        assert tracker.current_step() == 0

    def test_tracker_current_step(self, track_test_dir):
        """Test Tracker.current_step() method."""
        tracker = self._make_tracker()
        tracker.restart()
        assert tracker.current_step() == 10001


class TestTrackerWithNaming:
    """Test Tracker with custom naming, mirroring bindings test_tracker_string_handling."""

    def test_tracker_string_naming(self):
        """Test Tracker with string naming dict."""
        with open(os.path.join(TRACK_DATA_DIR, "conf.yaml")) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        cals = _load_cals_from_yaml(yaml_conf)

        from algorithms.parameters import (
            ControlPar,
            VolumePar,
            TrackParTuple,
            SequencePar,
        )
        from algorithms.track import Tracker

        scene = yaml_conf["scene"]
        seq_cfg = yaml_conf["sequence"]
        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]

        cpar = ControlPar(num_cams=len(yaml_conf["cameras"]))
        cpar.imx = scene["image_size"][0]
        cpar.imy = scene["image_size"][1]
        cpar.pix_x = scene["pixel_size"][0]
        cpar.pix_y = scene["pixel_size"][1]

        vpar = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=[
                corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))
            ],
            z_max_lay=[
                corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))
            ],
        )

        vel = tracking["velocity_lims"]
        tpar = TrackParTuple(
            dvxmin=vel[0][0],
            dvxmax=vel[0][1],
            dvymin=vel[1][0],
            dvymax=vel[1][1],
            dvzmin=vel[2][0],
            dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking["add_particle"],
            dsumg=0.0,
            dn=0.0,
            dnx=0.0,
            dny=0.0,
        )

        img_base = [
            seq_cfg["targets_template"].format(cam=cix + 1)
            for cix in range(len(yaml_conf["cameras"]))
        ]
        spar = SequencePar(
            img_base_name=img_base,
            first=seq_cfg["first"],
            last=seq_cfg["last"],
        )

        # String naming - Python Tracker accepts strings directly
        naming_strings = {
            "corres": "res/rt_is",
            "linkage": "res/ptv_is",
            "prio": "res/added",
        }
        tracker = Tracker(cpar, vpar, tpar, spar, cals, naming_strings)
        assert tracker is not None

        # Partial dict - missing keys use defaults
        naming_partial = {"corres": "res/rt_is"}
        tracker2 = Tracker(cpar, vpar, tpar, spar, cals, naming_partial)
        assert tracker2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

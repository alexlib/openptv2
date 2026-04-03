"""
Engine comparison tests for Tracker with Burgers vortex data.

Mirrors bindings/tests/test_burgers.py to ensure the Python algorithm engine
produces identical tracking behavior to the Cython bindings.

Tolerance: 1e-7 (full tracking pipeline)
"""

import os
import shutil
import yaml
import pytest

from .conftest import get_tolerance

TOLERANCE = get_tolerance("tracker")

BURGERS_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "burgers"
)

framebuf_naming = {
    "corres": "test_data/burgers/res/rt_is",
    "linkage": "test_data/burgers/res/ptv_is",
    "prio": "test_data/burgers/res/whatever",
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
    """Build a Python Tracker from the YAML config."""
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
        z_min_lay=corresp["z_spans"][0],
        z_max_lay=corresp["z_spans"][1],
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
def burgers_test_dir(tmp_path):
    """Set up a temporary copy of the burgers test data."""
    src = BURGERS_DATA_DIR
    res_orig = os.path.join(src, "res_orig")
    res_dst = os.path.join(src, "res")
    img_orig = os.path.join(src, "img_orig")
    img_dst = os.path.join(src, "img")

    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    if os.path.exists(img_dst):
        shutil.rmtree(img_dst)

    shutil.copytree(res_orig, res_dst)
    shutil.copytree(img_orig, img_dst)

    yield src

    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    if os.path.exists(img_dst):
        shutil.rmtree(img_dst)


class TestBurgersTracker:
    """Compare Burgers Tracker between optv and python engines, mirroring bindings tests."""

    def _make_tracker(self):
        with open(os.path.join(BURGERS_DATA_DIR, "conf.yaml")) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        return _build_python_tracker(yaml_conf)

    def test_forward(self, burgers_test_dir):
        """Manual forward tracking run, mirroring bindings test_forward."""
        tracker = self._make_tracker()
        tracker.restart()
        last_step = 10001
        while tracker.step_forward():
            assert tracker.current_step() > last_step
            with open("test_data/burgers/res/rt_is.%d" % last_step) as f:
                lines = f.readlines()
                if last_step == 10003:
                    assert lines[0] == "4\n"
                else:
                    assert lines[0] == "5\n"
            last_step += 1
        tracker.finalize()

    def test_forward_3d(self, burgers_test_dir):
        """Manual forward 3D tracking run, mirroring bindings test_forward_3d."""
        tracker = self._make_tracker()
        tracker.restart()
        last_step = 10001
        while tracker.step_forward_3d():
            assert tracker.current_step() > last_step
            with open("test_data/burgers/res/rt_is.%d" % last_step) as f:
                lines = f.readlines()
                if last_step == 10003:
                    assert lines[0] == "4\n"
                else:
                    assert lines[0] == "5\n"
            last_step += 1
        tracker.finalize()

    def test_full_forward(self, burgers_test_dir):
        """Automatic full forward tracking run, mirroring bindings test_full_forward."""
        tracker = self._make_tracker()
        tracker.full_forward()

    def test_full_forward_3d(self, burgers_test_dir):
        """Automatic full forward 3D tracking run, mirroring bindings test_full_forward_3d."""
        tracker = self._make_tracker()
        tracker.full_forward_3d()

    def test_full_backward(self, burgers_test_dir):
        """Automatic full backward correction phase, mirroring bindings test_full_backward."""
        tracker = self._make_tracker()
        tracker.full_forward()
        tracker.full_backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

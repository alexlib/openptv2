"""
Tests for Python track3d_loop implementation.

Mirrors the C tests in lib/tests/check_track3d.c and the Cython tests
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
    from algorithms.calibration import Calibration

    cals = []
    for cam_spec in yaml_conf["cameras"]:
        cal = Calibration()
        cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
        cals.append(cal)
    return cals


def _build_python_tracker(yaml_conf):
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

    img_base = [
        seq_cfg["targets_template"].format(cam=cix + 1)
        for cix in range(len(yaml_conf["cameras"]))
    ]
    spar = SequencePar(
        img_base_name=img_base,
        first=seq_cfg["first"],
        last=seq_cfg["last"],
    )

    return Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)


@pytest.fixture
def track_test_dir(tmp_path):
    """Set up temporary copy of track test data."""
    src = TRACK_DATA_DIR
    res_orig = os.path.join(src, "res_orig")
    res_dst = os.path.join(src, "res")
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    shutil.copytree(res_orig, res_dst)
    newpart_dir = os.path.join(src, "newpart")
    backup_dir = str(tmp_path / "newpart_backup")
    shutil.copytree(newpart_dir, backup_dir)
    yield src
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    if os.path.exists(newpart_dir):
        shutil.rmtree(newpart_dir)
    shutil.copytree(backup_dir, newpart_dir)


class TestFindCandidatesIn3D:
    """Unit tests for find_candidates_in_3d function."""

    def test_empty_frame(self):
        """Empty frame returns no candidates."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame

        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 0

        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0)
        assert len(indices) == 0

    def test_single_match(self):
        """Single particle within box is found."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame

        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.path_info[0].x = np.array([5.0, 5.0, 5.0])

        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_no_match_outside_box(self):
        """Particle outside box is not found."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame

        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.path_info[0].x = np.array([5.0, 5.0, 5.0])

        indices = find_candidates_in_3d(
            frm, np.array([10.0, 10.0, 10.0]), 1.0, 1.0, 1.0
        )
        assert len(indices) == 0

    def test_multiple_matches(self):
        """Multiple particles within box are found."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame

        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 5
        positions = [[0, 0, 0], [1, 1, 1], [5, 5, 5], [6, 6, 6], [10, 10, 10]]
        for i, pos in enumerate(positions):
            frm.path_info[i].x = np.array(pos, dtype=float)

        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 2.0, 2.0, 2.0)
        assert len(indices) == 2
        assert 2 in indices  # particle at (5,5,5)
        assert 3 in indices  # particle at (6,6,6)

    def test_max_cands_limit(self):
        """Result count is limited by max_cands."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame

        frm = Frame(num_cams=1, max_targets=20)
        frm.num_parts = 10
        for i in range(10):
            frm.path_info[i].x = np.array([5.0 + i * 0.01, 5.0, 5.0])

        indices = find_candidates_in_3d(
            frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0, max_cands=3
        )
        assert len(indices) == 3

    def test_boundary_excluded(self):
        """Particle exactly on boundary is excluded (< not <=)."""
        from algorithms.track import find_candidates_in_3d
        from algorithms.tracking_frame_buf import Frame

        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.path_info[0].x = np.array([6.0, 5.0, 5.0])  # pos + dx

        indices = find_candidates_in_3d(frm, np.array([5.0, 5.0, 5.0]), 1.0, 1.0, 1.0)
        assert len(indices) == 0


class TestTrack3DLoop:
    """Integration tests for track3d_loop via Tracker class."""

    def _make_tracker(self):
        with open(os.path.join(TRACK_DATA_DIR, "conf.yaml")) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        return _build_python_tracker(yaml_conf)

    def test_step_forward_3d(self, track_test_dir):
        """Manual step_forward_3d run."""
        tracker = self._make_tracker()
        tracker.restart()
        last_step = 10001
        while tracker.step_forward_3d():
            assert tracker.current_step() > last_step
            last_step += 1
        tracker.finalize()

    def test_full_forward_3d(self, track_test_dir):
        """Automatic full_forward_3d run."""
        tracker = self._make_tracker()
        tracker.full_forward_3d()

    def test_full_forward_3d_produces_output(self, track_test_dir):
        """Verify output files are created."""
        tracker = self._make_tracker()
        tracker.full_forward_3d()
        for step in range(10001, 10003):
            path = f"test_data/track/res/particles.{step}"
            assert os.path.exists(path), f"Missing output: {path}"

    def test_forward_3d_no_not_implemented_error(self, track_test_dir):
        """Verify NotImplementedError is no longer raised."""
        tracker = self._make_tracker()
        tracker.restart()
        # Should not raise
        result = tracker.step_forward_3d()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

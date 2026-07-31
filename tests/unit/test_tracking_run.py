"""Tests for TrackingRun initialization and frame reading."""

import os

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.constants import NEXT_NONE, POSI, PREV_NONE, PT_UNUSED
from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
    convert_track_par_to_tuple,
)
from openptv2.algorithms.tracking_frame_buf import (
    Corres,
    Frame,
    FrameBuf,
    Pathinfo,
    Target,
    read_path_frame,
    read_targets,
    write_path_frame,
    write_targets,
)
from openptv2.algorithms.tracking_run import TrackingRun, tr_new

EPS = 1e-5


def read_all_calibration(num_cams, base_path="test_data/track"):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


class TestTrackingRunInit:
    def test_tr_new_from_file_paths(self):
        """tr_new should accept file paths and create a valid TrackingRun."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            calib = read_all_calibration(cpar.num_cams, base_path=".")

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"),
                TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"),
                ControlPar.from_yaml("parameters.yaml"),
                4,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                calib,
                0.0001,
            )

            assert isinstance(run, TrackingRun)
            assert isinstance(run.fb, FrameBuf)
            assert run.fb.buf_len == 4
            assert run.fb.num_cams == cpar.num_cams
            assert len(run.fb.buf) == 4
            for frame in run.fb.buf:
                assert isinstance(frame, Frame)

            assert run.npart == 0
            assert run.nlinks == 0
            assert run.flatten_tol == 0.0001
            assert run.lmax > 0
            assert hasattr(run, "ymin")
            assert hasattr(run, "ymax")
        finally:
            os.chdir(original)

    def test_tr_new_from_objects(self):
        """tr_new should accept parameter objects directly."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            seq_par = SequencePar.from_yaml("parameters.yaml", cpar.num_cams)
            tpar = TrackPar.from_yaml("parameters.yaml")
            tpar = convert_track_par_to_tuple(tpar)
            vpar = VolumePar.from_yaml("parameters.yaml")
            calib = read_all_calibration(cpar.num_cams, base_path=".")

            run = tr_new(
                seq_par,
                tpar,
                vpar,
                cpar,
                4,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                calib,
                0.0001,
            )

            assert isinstance(run, TrackingRun)
            assert isinstance(run.fb, FrameBuf)
            assert run.seq_par.first == 10095
            assert run.seq_par.last == 10105
        finally:
            os.chdir(original)

    def test_lmax_computation(self):
        """lmax should be the Euclidean norm of the tracking volume diagonal."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            calib = read_all_calibration(cpar.num_cams, base_path=".")

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"),
                TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"),
                ControlPar.from_yaml("parameters.yaml"),
                4,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                calib,
                0.0001,
            )

            expected = np.linalg.norm(
                [
                    run.tpar.dvxmin - run.tpar.dvxmax,
                    run.tpar.dvymin - run.tpar.dvymax,
                    run.tpar.dvzmin - run.tpar.dvzmax,
                ]
            )
            assert abs(run.lmax - expected) < EPS
            assert run.lmax > 0
        finally:
            os.chdir(original)

    def test_volumedimension_sets_ymin_ymax(self):
        """volumedimension should compute ymin/ymax from camera views."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            calib = read_all_calibration(cpar.num_cams, base_path=".")

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"),
                TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"),
                ControlPar.from_yaml("parameters.yaml"),
                4,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                calib,
                0.0001,
            )

            assert run.ymin != 0.0 or run.ymax != 0.0
            assert run.ymin < run.ymax
        finally:
            os.chdir(original)

    def test_fb_target_file_base(self):
        """FrameBuf should use seq_par.img_base_name as target_file_base."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            calib = read_all_calibration(cpar.num_cams, base_path=".")

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"),
                TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"),
                ControlPar.from_yaml("parameters.yaml"),
                4,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                calib,
                0.0001,
            )

            assert run.fb.target_file_base == run.seq_par.img_base_name
            assert run.fb.corres_file_base == "res_orig/rt_is"
            assert run.fb.linkage_file_base == "res_orig/ptv_is"
            assert run.fb.prio_file_base == "res_orig/added"
        finally:
            os.chdir(original)


class TestFrameReading:
    def test_read_targets(self):
        """read_targets should parse target files correctly."""
        targets = read_targets("test_data/track/img_orig/cam1.", 10095)
        assert len(targets) > 0
        t = targets[0]
        assert isinstance(t, Target)
        assert t.pnr == 0
        assert abs(t.x - 1053.3689) < 0.001
        assert abs(t.y - 696.799) < 0.001

    def test_read_path_frame(self):
        """read_path_frame should parse correspondence files."""
        cor_list, path_list = read_path_frame(
            "test_data/track/res_orig/rt_is", "", "", 10095
        )
        assert len(cor_list) > 0
        assert len(path_list) > 0
        assert len(cor_list) == len(path_list)

        c = cor_list[0]
        assert isinstance(c, Corres)
        assert c.nr == 1

        p = path_list[0]
        assert isinstance(p, Pathinfo)
        assert p.prev == PREV_NONE
        assert p.next_idx == NEXT_NONE
        assert p.prio == 4
        assert p.finaldecis == 1000000.0
        assert p.inlist == 0
        assert len(p.decis) == POSI
        assert len(p.linkdecis) == POSI
        assert all(d == 0.0 for d in p.decis)
        assert all(ld == PT_UNUSED for ld in p.linkdecis)

    def test_frame_read(self):
        """Frame.read should load both correspondence and target data."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            frm = Frame(cpar.num_cams, 20000)
            ok = frm.read(
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                ["img_orig/cam1.", "img_orig/cam2."],
                10095,
            )

            assert ok is True
            assert frm.num_parts > 0

            for cam in range(cpar.num_cams):
                assert frm.num_targets[cam] > 0
                t = frm.targets[cam][0]
                assert isinstance(t, Target)
                assert t.pnr >= 0
        finally:
            os.chdir(original)

    def test_framebuf_read_frame_at_end(self):
        """FrameBuf.read_frame_at_end should read a frame into the last buffer slot."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            seq_par = SequencePar.from_yaml("parameters.yaml", cpar.num_cams)

            fb = FrameBuf(
                4,
                cpar.num_cams,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                seq_par.img_base_name,
            )

            ok = fb.read_frame_at_end(10095, read_links=True)
            assert ok is True

            last_frame = fb.buf[fb.buf_len - 1]
            assert last_frame.num_parts > 0
            for cam in range(cpar.num_cams):
                assert last_frame.num_targets[cam] > 0
        finally:
            os.chdir(original)

    def test_framebuf_fb_next_rotates(self):
        """fb_next should rotate the ring buffer."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            seq_par = SequencePar.from_yaml("parameters.yaml", cpar.num_cams)

            fb = FrameBuf(
                4,
                cpar.num_cams,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                seq_par.img_base_name,
            )

            fb.read_frame_at_end(10095, read_links=True)
            last_before = fb.buf[3]

            fb.fb_next()

            # fb_next moves buf[0] to end; old buf[3] shifts to buf[2]
            assert fb.buf[2] is last_before
        finally:
            os.chdir(original)

    def test_framebuf_fill_buffer(self):
        """Should be able to fill the entire buffer with consecutive frames."""
        original = os.getcwd()
        try:
            os.chdir("test_data/track")
            cpar = ControlPar.from_yaml("parameters.yaml")
            seq_par = SequencePar.from_yaml("parameters.yaml", cpar.num_cams)

            fb = FrameBuf(
                4,
                cpar.num_cams,
                20000,
                "res_orig/rt_is",
                "res_orig/ptv_is",
                "res_orig/added",
                seq_par.img_base_name,
            )

            for i in range(4):
                ok = fb.read_frame_at_end(seq_par.first + i, read_links=True)
                assert ok is True, f"Failed to read frame {seq_par.first + i}"
                if i < 3:
                    fb.fb_next()

            for i in range(4):
                assert fb.buf[i].num_parts > 0
        finally:
            os.chdir(original)


class TestTargetIO:
    def test_write_read_roundtrip(self, tmp_path):
        """Targets written and read back should match."""
        targets = [
            Target(pnr=0, x=100.5, y=200.3, n=50, nx=5, ny=5, sumg=1000, tnr=1),
            Target(pnr=1, x=300.2, y=400.1, n=30, nx=3, ny=4, sumg=800, tnr=2),
        ]
        base = str(tmp_path / "test_cam.")
        write_targets(targets, 2, base, 42)
        read_back = read_targets(base, 42)

        assert len(read_back) == 2
        for orig, rb in zip(targets, read_back):
            assert orig.pnr == rb.pnr
            assert abs(orig.x - rb.x) < 0.01
            assert abs(orig.y - rb.y) < 0.01
            assert orig.n == rb.n
            assert orig.nx == rb.nx
            assert orig.ny == rb.ny
            assert orig.sumg == rb.sumg
            assert orig.tnr == rb.tnr

    def test_write_read_path_frame_roundtrip(self, tmp_path):
        """Path frame written and read back should match."""
        cor_buf = [
            Corres(nr=1, p=np.array([0, 1, -1, -1], dtype=np.int32)),
            Corres(nr=2, p=np.array([1, 0, 2, -1], dtype=np.int32)),
        ]
        path_buf = [
            Pathinfo(x=np.array([10.0, 20.0, 30.0]), prev=-1, next_idx=-2),
            Pathinfo(x=np.array([40.0, 50.0, 60.0]), prev=0, next_idx=1),
        ]
        base_corres = str(tmp_path / "rt_is")
        base_linkage = str(tmp_path / "ptv_is")

        write_path_frame(cor_buf, path_buf, 2, base_corres, base_linkage, None, 100)

        cor_read, path_read = read_path_frame(base_corres, base_linkage, "", 100)
        assert len(cor_read) == 2
        assert len(path_read) == 2

        for i in range(2):
            np.testing.assert_array_equal(cor_read[i].p, cor_buf[i].p)
            np.testing.assert_allclose(path_read[i].x, path_buf[i].x, atol=0.01)

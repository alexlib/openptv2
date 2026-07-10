# -*- coding: utf-8 -*-
"""Pure-Python coverage tests for openptv2.algorithms.parameters.

Run with the ppsrc override for coverage:
    COVERAGE_FILE=/tmp/.cov_params uv run pytest tests/unit/test_parameters_coverage.py \
      -o pythonpath=/tmp/ppsrc -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 --cov-config=/tmp/covrc \
      --cov-report=term-missing -q \
      2>&1 | grep -E "(parameters|TOTAL|PASSED|FAILED|ERROR|passed|failed|error)"
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.parameters import (
    CalibrationPar,
    ControlPar,
    ExaminePar,
    MmNp,
    MultimediaPar,
    MultiPlanesPar,
    OrientPar,
    PftVersionPar,
    SequencePar,
    TargetPar,
    TrackPar,
    TrackParTuple,
    VolumePar,
    # aliases
    ControlParams,
    MultimediaParams,
    SequenceParams,
    TargetParams,
    TrackingParams,
    VolumeParams,
    _clean_name_list,
    _load_yaml_params,
    convert_track_par_to_tuple,
    is_compiled,
    read_control_par,
    read_sequence_par,
    read_track_par,
    read_volume_par,
)

# ---------------------------------------------------------------------------
# Absolute paths to canned test data
# ---------------------------------------------------------------------------
TD = Path("/home/user/Documents/GitHub/openptv2/test_data")
PARAMS_YAML = TD / "parameters.yaml"
PARAMS_DIR = TD / "parameters"
VOLUME_PAR = TD / "volume_parameters" / "volume.par"
TARG_PAR = TD / "target_parameters" / "targ_rec.par"
CTRL_PAR = TD / "corresp" / "control.par"
SEQ_PAR = TD / "sequence_parameters" / "sequence.par"
TRACK_PAR = PARAMS_DIR / "track.par"
ORIENT_PAR = PARAMS_DIR / "orient.par"
BURGERS_DIR = TD / "burgers" / "parameters"


# ===========================================================================
# _load_yaml_params
# ===========================================================================


def test_load_yaml_params_returns_dict():
    d = _load_yaml_params(PARAMS_YAML)
    assert isinstance(d, dict)
    assert "num_cams" in d


def test_load_yaml_params_rejects_non_dict(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="not a valid parameter YAML"):
        _load_yaml_params(bad)


# ===========================================================================
# _clean_name_list
# ===========================================================================


def test_clean_name_list_with_real_names():
    result = _clean_name_list(["cam1", "cam2"], 2)
    assert result == ["cam1", "cam2"]


def test_clean_name_list_strips_placeholders():
    result = _clean_name_list(["---", "cam1", "--", "cam2"], 2)
    assert result == ["cam1", "cam2"]


def test_clean_name_list_empty_with_num_cams():
    result = _clean_name_list([], 3)
    assert result == ["", "", ""]


def test_clean_name_list_none_with_num_cams():
    result = _clean_name_list(None, 2)
    assert result == ["", ""]


def test_clean_name_list_empty_no_cams():
    result = _clean_name_list([], 0)
    assert result == []


def test_clean_name_list_all_placeholders_no_cams():
    result = _clean_name_list(["---", "--", ""], 0)
    assert result == []


# ===========================================================================
# TrackParTuple namedtuple
# ===========================================================================


def test_track_par_tuple_fields():
    t = TrackParTuple(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    assert t.dvxmin == 1
    assert t.dvxmax == 2
    assert t.dny == 13


# ===========================================================================
# convert_track_par_to_tuple
# ===========================================================================


def test_convert_track_par_to_tuple_basic():
    tp = TrackPar(dvxmin=1.0, dvxmax=2.0, dvymin=3.0, dvymax=4.0,
                  dvzmin=5.0, dvzmax=6.0, dangle=7.0, dacc=8.0, add=1)
    result = convert_track_par_to_tuple(tp)
    assert isinstance(result, TrackParTuple)
    assert result.dvxmin == 1.0
    assert result.dangle == 7.0
    assert result.add == 1


def test_convert_track_par_to_tuple_dsumg_dn_dnx_dny():
    tp = TrackPar()
    tp.dsumg = 0.5
    tp.dn = 0.1
    tp.dnx = 0.2
    tp.dny = 0.3
    result = convert_track_par_to_tuple(tp)
    assert result.dsumg == 0.5
    assert result.dn == 0.1
    assert result.dnx == 0.2
    assert result.dny == 0.3


# ===========================================================================
# SequencePar
# ===========================================================================


class TestSequencePar:
    def test_defaults(self):
        sp = SequencePar()
        assert sp.num_cams == 0
        assert sp.img_base_name == []
        assert sp.first == 0
        assert sp.last == 0

    def test_explicit_args(self):
        sp = SequencePar(num_cams=2, img_base_name=["a", "b"], first=1, last=100)
        assert sp.num_cams == 2
        assert sp.img_base_name == ["a", "b"]
        assert sp.first == 1
        assert sp.last == 100

    def test_image_base_alias(self):
        sp = SequencePar(num_cams=2, image_base=["x", "y"])
        assert sp.img_base_name == ["x", "y"]

    def test_frame_range_alias(self):
        sp = SequencePar(frame_range=(10, 200))
        assert sp.first == 10
        assert sp.last == 200

    def test_empty_img_base_padded_for_num_cams(self):
        sp = SequencePar(num_cams=3)
        assert sp.img_base_name == ["", "", ""]

    def test_from_file(self):
        sp = SequencePar.from_file(SEQ_PAR, 4)
        assert sp.num_cams == 4
        assert sp.first == 497
        assert sp.last == 597
        assert len(sp.img_base_name) == 4

    def test_from_file_too_short(self, tmp_path):
        f = tmp_path / "seq.par"
        f.write_text("cam1\n1000\n")  # only 2 lines, needs >=6 for 4 cams
        with pytest.raises(ValueError, match="Expected at least"):
            SequencePar.from_file(f, 4)

    def test_from_yaml(self):
        sp = SequencePar.from_yaml(PARAMS_YAML)
        assert sp.first == 1000
        assert sp.last == 2000
        assert sp.num_cams == 4

    def test_from_yaml_with_explicit_num_cams(self):
        sp = SequencePar.from_yaml(PARAMS_YAML, num_cams=2)
        assert sp.num_cams == 2

    def test_getters_setters(self):
        sp = SequencePar(first=5, last=50)
        assert sp.get_first() == 5
        sp.set_first(10)
        assert sp.get_first() == 10
        assert sp.get_last() == 50
        sp.set_last(100)
        assert sp.get_last() == 100

    def test_get_img_base_name_valid(self):
        sp = SequencePar(num_cams=2, img_base_name=["a", "b"])
        assert sp.get_img_base_name(0) == "a"
        assert sp.get_img_base_name(1) == "b"

    def test_get_img_base_name_out_of_range(self):
        sp = SequencePar(num_cams=1, img_base_name=["a"])
        assert sp.get_img_base_name(5) == ""

    def test_set_img_base_name_extend(self):
        sp = SequencePar()
        sp.set_img_base_name(2, "cam3")
        assert sp.img_base_name[2] == "cam3"
        assert len(sp.img_base_name) == 3

    def test_read_sequence_par_inplace(self, tmp_path):
        f = tmp_path / "seq.par"
        f.write_text("cam_a\ncam_b\n10\n20\n")
        sp = SequencePar()
        sp.read_sequence_par(f, 2)
        assert sp.num_cams == 2
        assert sp.first == 10
        assert sp.last == 20


# ===========================================================================
# TrackPar
# ===========================================================================


class TestTrackPar:
    def test_defaults(self):
        tp = TrackPar()
        assert tp.dvxmin == 0.0
        assert tp.dvxmax == 0.0
        assert tp.add == 0
        assert tp.track_mode == 0
        assert tp.dsumg == 0.0
        assert tp.dn == 0.0
        assert tp.dnx == 0.0
        assert tp.dny == 0.0

    def test_explicit_args(self):
        tp = TrackPar(dvxmin=1.0, dvxmax=2.0, dvymin=3.0, dvymax=4.0,
                      dvzmin=5.0, dvzmax=6.0, dangle=7.0, dacc=8.0,
                      add=1, track_mode=1)
        assert tp.dvxmin == 1.0
        assert tp.track_mode == 1

    def test_from_file_9_lines(self, tmp_path):
        f = tmp_path / "track.par"
        f.write_text("0.4\n120.0\n2.0\n-2.0\n2.0\n-2.0\n2.0\n-2.0\n1\n")
        tp = TrackPar.from_file(f)
        assert tp.dvxmin == 0.4
        assert tp.add == 1
        assert tp.track_mode == 0  # default when only 9 lines

    def test_from_file_10_lines(self, tmp_path):
        f = tmp_path / "track.par"
        f.write_text("0.4\n120.0\n2.0\n-2.0\n2.0\n-2.0\n2.0\n-2.0\n1\n2\n")
        tp = TrackPar.from_file(f)
        assert tp.track_mode == 2

    def test_from_file_bad_track_mode(self, tmp_path):
        """10th line non-integer falls back to 0."""
        f = tmp_path / "track.par"
        f.write_text("0.4\n120.0\n2.0\n-2.0\n2.0\n-2.0\n2.0\n-2.0\n1\nbad\n")
        tp = TrackPar.from_file(f)
        assert tp.track_mode == 0

    def test_from_file_too_short(self, tmp_path):
        f = tmp_path / "track.par"
        f.write_text("0.4\n1.0\n")
        with pytest.raises(ValueError, match="Expected 9 lines"):
            TrackPar.from_file(f)

    def test_from_file_real(self):
        tp = TrackPar.from_file(TRACK_PAR)
        assert tp.dvxmin == 0.4
        assert tp.dvxmax == 120.0

    def test_from_yaml_with_add(self, tmp_path):
        y = tmp_path / "p.yaml"
        y.write_text(
            "track:\n  dvxmin: 1.0\n  dvxmax: 2.0\n  dvymin: 3.0\n"
            "  dvymax: 4.0\n  dvzmin: 5.0\n  dvzmax: 6.0\n"
            "  dangle: 7.0\n  dacc: 8.0\n  add: 1\n  track_mode: 0\n"
        )
        tp = TrackPar.from_yaml(y)
        assert tp.dvxmin == 1.0
        assert tp.add == 1

    def test_from_yaml_with_flagNewParticles(self, tmp_path):
        y = tmp_path / "p.yaml"
        y.write_text(
            "track:\n  dvxmin: 0.0\n  dvxmax: 0.0\n  dvymin: 0.0\n"
            "  dvymax: 0.0\n  dvzmin: 0.0\n  dvzmax: 0.0\n"
            "  angle: 2.0\n  dacc: 0.0\n  flagNewParticles: true\n"
        )
        tp = TrackPar.from_yaml(y)
        assert tp.add == 1
        assert tp.dangle == 2.0

    def test_from_yaml_real_file(self):
        tp = TrackPar.from_yaml(PARAMS_YAML)
        assert tp.dvxmin == 0.4
        assert tp.dangle == 2.0
        assert tp.add == 1

    def test_getters_setters(self):
        tp = TrackPar()
        for attr, val in [("dvxmin", 1.1), ("dvxmax", 2.2), ("dvymin", 3.3),
                          ("dvymax", 4.4), ("dvzmin", 5.5), ("dvzmax", 6.6),
                          ("dangle", 7.7), ("dacc", 8.8)]:
            getattr(tp, f"set_{attr}")(val)
            assert abs(getattr(tp, f"get_{attr}")() - val) < 1e-9

        tp.set_add(1)
        assert tp.get_add() == 1
        tp.set_track_mode(2)
        assert tp.get_track_mode() == 2
        tp.set_dsumg(0.5)
        assert tp.get_dsumg() == 0.5
        tp.set_dn(0.1)
        assert tp.get_dn() == 0.1
        tp.set_dnx(0.2)
        assert tp.get_dnx() == 0.2
        tp.set_dny(0.3)
        assert tp.get_dny() == 0.3

    def test_read_track_par_inplace(self, tmp_path):
        f = tmp_path / "track.par"
        f.write_text("1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n8.0\n0\n")
        tp = TrackPar()
        tp.read_track_par(f)
        assert tp.dvxmin == 1.0
        assert tp.dvxmax == 2.0


# ===========================================================================
# VolumePar
# ===========================================================================


class TestVolumePar:
    def test_defaults(self):
        vp = VolumePar()
        np.testing.assert_array_equal(vp.X_lay, [0.0, 0.0])
        np.testing.assert_array_equal(vp.Zmin_lay, [0.0, 0.0])
        np.testing.assert_array_equal(vp.Zmax_lay, [0.0, 0.0])
        assert vp.cnx == 0.0
        assert vp.eps0 == 0.0

    def test_explicit_args(self):
        vp = VolumePar(X_lay=[-250, 250], Zmin_lay=[-100, -100],
                       Zmax_lay=[100, 100], cnx=0.3, cny=0.3,
                       cn=0.01, csumg=0.01, corrmin=33.0, eps0=1.0)
        np.testing.assert_array_equal(vp.X_lay, [-250.0, 250.0])
        assert vp.corrmin == 33.0

    def test_from_file(self):
        vp = VolumePar.from_file(VOLUME_PAR)
        assert vp.X_lay[0] == 111.111
        assert vp.corrmin == 1111.1111
        assert vp.eps0 == 1212.1212

    def test_from_file_too_short(self, tmp_path):
        f = tmp_path / "vol.par"
        f.write_text("1.0\n2.0\n")
        with pytest.raises(ValueError, match="Expected 12 lines"):
            VolumePar.from_file(f)

    def test_from_yaml(self):
        vp = VolumePar.from_yaml(PARAMS_YAML)
        np.testing.assert_array_equal(vp.X_lay, [-250.0, 250.0])
        assert vp.corrmin == 33.0
        assert vp.eps0 == 1.0

    def test_get_X_lay_copy(self):
        vp = VolumePar(X_lay=[1.0, 2.0])
        arr = vp.get_X_lay(copy=True)
        arr[0] = 99.0
        assert vp.X_lay[0] == 1.0  # original unchanged

    def test_get_X_lay_no_copy(self):
        vp = VolumePar(X_lay=[1.0, 2.0])
        arr = vp.get_X_lay(copy=False)
        assert arr is vp.X_lay

    def test_get_Zmin_lay_copy_and_no_copy(self):
        vp = VolumePar(Zmin_lay=[-10.0, -20.0])
        arr_copy = vp.get_Zmin_lay(copy=True)
        arr_nocopy = vp.get_Zmin_lay(copy=False)
        assert arr_nocopy is vp.Zmin_lay
        arr_copy[0] = 99.0
        assert vp.Zmin_lay[0] == -10.0

    def test_get_Zmax_lay_copy_and_no_copy(self):
        vp = VolumePar(Zmax_lay=[10.0, 20.0])
        arr_copy = vp.get_Zmax_lay(copy=True)
        arr_nocopy = vp.get_Zmax_lay(copy=False)
        assert arr_nocopy is vp.Zmax_lay
        arr_copy[0] = 99.0
        assert vp.Zmax_lay[0] == 10.0

    def test_setters(self):
        vp = VolumePar()
        vp.set_X_lay([1.0, 2.0])
        np.testing.assert_array_equal(vp.X_lay, [1.0, 2.0])
        vp.set_Zmin_lay([-5.0, -5.0])
        np.testing.assert_array_equal(vp.Zmin_lay, [-5.0, -5.0])
        vp.set_Zmax_lay([5.0, 5.0])
        np.testing.assert_array_equal(vp.Zmax_lay, [5.0, 5.0])
        vp.set_cn(0.1)
        assert vp.get_cn() == 0.1
        vp.set_cnx(0.2)
        assert vp.get_cnx() == 0.2
        vp.set_cny(0.3)
        assert vp.get_cny() == 0.3
        vp.set_csumg(0.4)
        assert vp.get_csumg() == 0.4
        vp.set_eps0(0.5)
        assert vp.get_eps0() == 0.5
        vp.set_corrmin(30.0)
        assert vp.get_corrmin() == 30.0

    def test_read_volume_par_inplace(self):
        vp = VolumePar()
        vp.read_volume_par(VOLUME_PAR)
        assert vp.X_lay[0] == 111.111


# ===========================================================================
# MmNp
# ===========================================================================


class TestMmNp:
    def test_defaults(self):
        mm = MmNp()
        assert mm.nlay == 1
        assert mm.n1 == 1.0
        assert mm.n3 == 1.0
        np.testing.assert_array_equal(mm.n2, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(mm.d, [0.0, 0.0, 0.0])

    def test_explicit_args(self):
        mm = MmNp(nlay=2, n1=1.5, n2=[1.49, 1.0, 1.0],
                  d=[5.0, 0.0, 0.0], n3=1.33)
        assert mm.nlay == 2
        assert mm.n1 == 1.5
        assert mm.n3 == 1.33

    def test_copy_constructor_via_mm(self):
        src = MmNp(nlay=2, n1=1.5, n2=[1.49, 1.0, 1.0],
                   d=[5.0, 0.0, 0.0], n3=1.33)
        dst = MmNp(_mm=src)
        assert dst.nlay == src.nlay
        assert dst.n1 == src.n1
        assert dst.n3 == src.n3

    def test_get_nlay(self):
        mm = MmNp(nlay=3)
        assert mm.get_nlay() == 3

    def test_get_n1_and_set(self):
        mm = MmNp(n1=1.0)
        mm.set_n1(1.5)
        assert mm.get_n1() == 1.5

    def test_get_n2_copy(self):
        mm = MmNp(n2=[1.49, 1.0, 1.0])
        arr = mm.get_n2(copy=True)
        arr[0] = 99.0
        assert mm.n2[0] == 1.49  # original unchanged

    def test_get_n2_no_copy(self):
        mm = MmNp()
        arr = mm.get_n2(copy=False)
        assert arr is mm.n2

    def test_get_d_copy(self):
        mm = MmNp(d=[5.0, 0.0, 0.0])
        arr = mm.get_d(copy=True)
        arr[0] = 99.0
        assert mm.d[0] == 5.0

    def test_get_d_no_copy(self):
        mm = MmNp()
        arr = mm.get_d(copy=False)
        assert arr is mm.d

    def test_set_layers(self):
        mm = MmNp()
        mm.set_layers([1.49, 1.33], [5.0, 3.0])
        assert mm.nlay == 2
        np.testing.assert_array_equal(mm.n2, [1.49, 1.33])
        np.testing.assert_array_equal(mm.d, [5.0, 3.0])

    def test_get_n3_and_set(self):
        mm = MmNp(n3=1.33)
        assert mm.get_n3() == 1.33
        mm.set_n3(1.5)
        assert mm.get_n3() == 1.5


# ===========================================================================
# ControlPar
# ===========================================================================


class TestControlPar:
    def test_defaults(self):
        cp = ControlPar()
        assert cp.num_cams == 0
        assert cp.img_base_name == []
        assert cp.cal_img_base_name == []
        assert cp.hp_flag == 0
        assert cp.allCam_flag == 0
        assert cp.tiff_flag == 0
        assert cp.imx == 0
        assert cp.imy == 0
        assert cp.pix_x == 0.0
        assert cp.pix_y == 0.0
        assert cp.chfield == 0
        assert isinstance(cp.mm, MmNp)

    def test_all_cam_flag_alias(self):
        cp = ControlPar(all_cam_flag=1)
        assert cp.allCam_flag == 1

    def test_allCam_flag_direct(self):
        cp = ControlPar(allCam_flag=1)
        assert cp.allCam_flag == 1

    def test_empty_img_padded(self):
        cp = ControlPar(num_cams=2)
        assert cp.img_base_name == ["", ""]
        assert cp.cal_img_base_name == ["", ""]

    def test_img_base_name_provided(self):
        cp = ControlPar(num_cams=2,
                        img_base_name=["a", "b"],
                        cal_img_base_name=["ca", "cb"])
        assert cp.img_base_name == ["a", "b"]
        assert cp.cal_img_base_name == ["ca", "cb"]

    def test_mm_provided(self):
        mm = MmNp(n1=1.5)
        cp = ControlPar(mm=mm)
        assert cp.mm.n1 == 1.5

    def test_from_file_empty(self, tmp_path):
        f = tmp_path / "empty.par"
        f.write_text("")
        with pytest.raises(ValueError, match="Empty control parameter file"):
            ControlPar.from_file(f)

    def test_from_file(self):
        cp = ControlPar.from_file(CTRL_PAR)
        assert cp.num_cams == 4
        assert cp.imx == 1280
        assert cp.imy == 1024
        assert abs(cp.pix_x - 0.017) < 1e-6
        assert cp.hp_flag == 1
        assert cp.tiff_flag == 1
        assert cp.mm.n1 == 1.0
        assert cp.mm.n3 == 1.33

    def test_from_yaml(self):
        cp = ControlPar.from_yaml(PARAMS_YAML)
        assert cp.num_cams == 4
        assert cp.imx == 1280
        assert cp.imy == 1024
        assert cp.hp_flag == 1
        assert cp.tiff_flag == 1
        assert abs(cp.mm.n2[0] - 1.49) < 1e-6

    def test_from_yaml_explicit_num_cams(self):
        cp = ControlPar.from_yaml(PARAMS_YAML, num_cams=2)
        assert cp.num_cams == 2

    def test_get_num_cams(self):
        cp = ControlPar(num_cams=3)
        assert cp.get_num_cams() == 3

    def test_get_set_image_size(self):
        cp = ControlPar(imx=640, imy=480)
        assert cp.get_image_size() == (640, 480)
        cp.set_image_size((1280, 1024))
        assert cp.imx == 1280
        assert cp.imy == 1024

    def test_get_set_pixel_size(self):
        cp = ControlPar(pix_x=0.01, pix_y=0.02)
        assert cp.get_pixel_size() == (0.01, 0.02)
        cp.set_pixel_size((0.017, 0.017))
        assert abs(cp.pix_x - 0.017) < 1e-9

    def test_get_set_hp_flag(self):
        cp = ControlPar()
        cp.set_hp_flag(1)
        assert cp.get_hp_flag() == 1

    def test_get_set_allCam_flag(self):
        cp = ControlPar()
        cp.set_allCam_flag(1)
        assert cp.get_allCam_flag() == 1

    def test_get_set_tiff_flag(self):
        cp = ControlPar()
        cp.set_tiff_flag(1)
        assert cp.get_tiff_flag() == 1

    def test_get_set_chfield(self):
        cp = ControlPar()
        cp.set_chfield(2)
        assert cp.get_chfield() == 2

    def test_get_multimedia_params(self):
        mm = MmNp(n1=1.3)
        cp = ControlPar(mm=mm)
        assert cp.get_multimedia_params().n1 == 1.3

    def test_get_img_base_name_valid(self):
        cp = ControlPar(num_cams=2, img_base_name=["a", "b"])
        assert cp.get_img_base_name(0) == "a"
        assert cp.get_img_base_name(1) == "b"

    def test_get_img_base_name_out_of_range(self):
        cp = ControlPar(num_cams=1, img_base_name=["a"])
        assert cp.get_img_base_name(5) == ""

    def test_set_img_base_name_extend(self):
        cp = ControlPar()
        cp.set_img_base_name(2, "cam3")
        assert cp.img_base_name[2] == "cam3"

    def test_get_cal_img_base_name_valid(self):
        cp = ControlPar(num_cams=2, cal_img_base_name=["ca", "cb"])
        assert cp.get_cal_img_base_name(0) == "ca"

    def test_get_cal_img_base_name_out_of_range(self):
        cp = ControlPar(num_cams=1, cal_img_base_name=["ca"])
        assert cp.get_cal_img_base_name(10) == ""

    def test_set_cal_img_base_name_extend(self):
        cp = ControlPar()
        cp.set_cal_img_base_name(1, "cal2")
        assert cp.cal_img_base_name[1] == "cal2"

    def test_read_control_par_inplace(self):
        cp = ControlPar()
        cp.read_control_par(CTRL_PAR)
        assert cp.num_cams == 4
        assert cp.imx == 1280


# ===========================================================================
# TargetPar
# ===========================================================================


class TestTargetPar:
    def test_defaults(self):
        tp = TargetPar()
        np.testing.assert_array_equal(tp.gvthres, [0, 0, 0, 0])
        assert tp.discont == 0
        assert tp.nnmin == 0
        assert tp.nnmax == 0

    def test_explicit_args(self):
        tp = TargetPar(gvthres=[10, 20, 30, 40], discont=5,
                       nnmin=2, nnmax=50, nxmin=3, nxmax=100,
                       nymin=2, nymax=80, sumg_min=100, cr_sz=3)
        assert tp.discont == 5
        assert tp.nnmin == 2
        assert tp.sumg_min == 100

    def test_gvthresh_alias(self):
        tp = TargetPar(gvthresh=[5, 10, 15, 20])
        np.testing.assert_array_equal(tp.gvthres, [5, 10, 15, 20])

    def test_pixel_count_bounds_alias(self):
        tp = TargetPar(pixel_count_bounds=(2, 50))
        assert tp.nnmin == 2
        assert tp.nnmax == 50

    def test_xsize_bounds_alias(self):
        tp = TargetPar(xsize_bounds=(3, 100))
        assert tp.nxmin == 3
        assert tp.nxmax == 100

    def test_ysize_bounds_alias(self):
        tp = TargetPar(ysize_bounds=(2, 80))
        assert tp.nymin == 2
        assert tp.nymax == 80

    def test_min_sum_grey_alias(self):
        tp = TargetPar(min_sum_grey=150)
        assert tp.sumg_min == 150

    def test_cross_size_alias(self):
        tp = TargetPar(cross_size=5)
        assert tp.cr_sz == 5

    def test_from_file(self):
        tp = TargetPar.from_file(TARG_PAR)
        # targ_rec.par tokens: 3 2 2 3 5 3 100 1 20 1 20 3 2
        # gvthres=[3,2,2,3], discont=5, nnmin=3, nnmax=100
        assert tp.discont == 5
        assert tp.nnmin == 3
        assert tp.nnmax == 100
        assert tp.cr_sz == 2

    def test_from_file_no_cr_sz(self, tmp_path):
        """12 tokens only — cr_sz defaults to 0."""
        f = tmp_path / "targ.par"
        f.write_text("3 2 2 3 5 3 100 1 20 1 20 3")
        tp = TargetPar.from_file(f)
        assert tp.cr_sz == 0

    def test_from_file_with_cr_sz(self, tmp_path):
        """13 tokens — cr_sz is set."""
        f = tmp_path / "targ.par"
        f.write_text("3 2 2 3 5 3 100 1 20 1 20 3 7")
        tp = TargetPar.from_file(f)
        assert tp.cr_sz == 7

    def test_from_file_too_short(self, tmp_path):
        f = tmp_path / "targ.par"
        f.write_text("1 2 3 4 5")
        with pytest.raises(ValueError, match="Expected at least 12"):
            TargetPar.from_file(f)

    def test_to_file_round_trip(self, tmp_path):
        tp = TargetPar(gvthres=[3, 2, 2, 3], discont=5,
                       nnmin=3, nnmax=100, nxmin=1, nxmax=20,
                       nymin=1, nymax=20, sumg_min=3, cr_sz=2)
        f = tmp_path / "targ_out.par"
        tp.to_file(f)
        tp2 = TargetPar.from_file(f)
        np.testing.assert_array_equal(tp2.gvthres, tp.gvthres)
        assert tp2.discont == tp.discont
        assert tp2.nnmin == tp.nnmin
        assert tp2.cr_sz == tp.cr_sz

    def test_getters(self):
        tp = TargetPar(gvthres=[1, 2, 3, 4], discont=5, nnmin=2, nnmax=50,
                       nxmin=3, nxmax=100, nymin=2, nymax=80,
                       sumg_min=100, cr_sz=3)
        arr = tp.get_grey_thresholds()
        np.testing.assert_array_equal(arr, [1, 2, 3, 4])
        arr_nocopy = tp.get_grey_thresholds(copy=False)
        assert arr_nocopy is tp.gvthres
        assert tp.get_max_discontinuity() == 5
        assert tp.get_pixel_count_bounds() == (2, 50)
        assert tp.get_xsize_bounds() == (3, 100)
        assert tp.get_ysize_bounds() == (2, 80)
        assert tp.get_min_sum_grey() == 100
        assert tp.get_cross_size() == 3

    def test_setters(self):
        tp = TargetPar()
        tp.set_grey_thresholds([5, 6, 7, 8])
        np.testing.assert_array_equal(tp.gvthres, [5, 6, 7, 8])
        tp.set_max_discontinuity(10)
        assert tp.discont == 10
        tp.set_pixel_count_bounds((3, 60))
        assert tp.nnmin == 3 and tp.nnmax == 60
        tp.set_xsize_bounds((4, 90))
        assert tp.nxmin == 4 and tp.nxmax == 90
        tp.set_ysize_bounds((5, 70))
        assert tp.nymin == 5 and tp.nymax == 70
        tp.set_min_sum_grey(200)
        assert tp.sumg_min == 200
        tp.set_cross_size(6)
        assert tp.cr_sz == 6

    def test_read_inplace(self):
        tp = TargetPar()
        tp.read(TARG_PAR)
        assert tp.nnmin == 3


# ===========================================================================
# OrientPar
# ===========================================================================


class TestOrientPar:
    def test_defaults(self):
        op = OrientPar()
        assert op.useflag == 0
        assert op.interfflag == 0

    def test_explicit(self):
        op = OrientPar(useflag=1, ccflag=1, xhflag=0, yhflag=0,
                       k1flag=1, k2flag=0, k3flag=0, p1flag=0,
                       p2flag=0, scxflag=0, sheflag=0, interfflag=1)
        assert op.useflag == 1
        assert op.interfflag == 1

    def test_from_file(self):
        op = OrientPar.from_file(ORIENT_PAR)
        # orient.par has 12 zeros
        assert op.useflag == 0
        assert op.interfflag == 0

    def test_from_file_short(self, tmp_path):
        """Fewer than 12 lines — pads with zeros."""
        f = tmp_path / "orient.par"
        f.write_text("1\n0\n1\n")  # only 3 flags
        op = OrientPar.from_file(f)
        assert op.useflag == 1
        assert op.xhflag == 1
        assert op.yhflag == 0  # padded


# ===========================================================================
# MultimediaPar
# ===========================================================================


class TestMultimediaPar:
    def test_defaults(self):
        mp = MultimediaPar()
        assert mp.n1 == 1.0
        assert mp.n3 == 1.0
        assert mp.nlay == 1
        np.testing.assert_array_equal(mp.n2, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(mp.d, [0.0, 0.0, 0.0])

    def test_explicit_args(self):
        mp = MultimediaPar(n1=1.5, n2=[1.49, 1.0, 1.0],
                           d=[5.0, 0.0, 0.0], n3=1.33, nlay=2)
        assert mp.n1 == 1.5
        assert mp.nlay == 2

    def test_n2_none_defaults(self):
        mp = MultimediaPar(n2=None)
        np.testing.assert_array_equal(mp.n2, [1.0, 1.0, 1.0])

    def test_d_none_defaults(self):
        mp = MultimediaPar(d=None)
        np.testing.assert_array_equal(mp.d, [0.0, 0.0, 0.0])


# ===========================================================================
# CalibrationPar
# ===========================================================================


class TestCalibrationPar:
    def test_defaults(self):
        cp = CalibrationPar()
        assert cp.fixp_name == ""
        assert cp.img_cal_name == []
        assert cp.img_ori == []
        assert cp.tiff_flag == 0
        assert cp.pair_flag == 0
        assert cp.chfield == 0

    def test_explicit(self):
        cp = CalibrationPar(fixp_name="target.txt",
                            img_cal_name=["cam1.tif", "cam2.tif"],
                            img_ori=["cam1.ori", "cam2.ori"],
                            tiff_flag=1, pair_flag=1, chfield=0)
        assert cp.fixp_name == "target.txt"
        assert len(cp.img_cal_name) == 2

    def test_from_file(self):
        cal_par = BURGERS_DIR / "cal_ori.par"
        cp = CalibrationPar.from_file(cal_par, 4)
        assert cp.fixp_name == "cal/target_file.txt"
        assert len(cp.img_cal_name) == 4
        assert len(cp.img_ori) == 4
        assert cp.tiff_flag == 1


# ===========================================================================
# MultiPlanesPar
# ===========================================================================


class TestMultiPlanesPar:
    def test_defaults(self):
        mp = MultiPlanesPar()
        assert mp.num_planes == 0
        assert mp.filename == []

    def test_explicit(self):
        mp = MultiPlanesPar(num_planes=2, filename=["plane1.txt", "plane2.txt"])
        assert mp.num_planes == 2

    def test_from_file(self):
        f = BURGERS_DIR / "multi_planes.par"
        mp = MultiPlanesPar.from_file(f)
        assert mp.num_planes == 3
        assert len(mp.filename) == 3


# ===========================================================================
# ExaminePar
# ===========================================================================


class TestExaminePar:
    def test_defaults(self):
        ep = ExaminePar()
        assert ep.examine_flag is False
        assert ep.combine_flag is False

    def test_explicit(self):
        ep = ExaminePar(examine_flag=True, combine_flag=True)
        assert ep.examine_flag is True
        assert ep.combine_flag is True

    def test_from_file_false(self):
        f = BURGERS_DIR / "examine.par"
        ep = ExaminePar.from_file(f)
        assert ep.examine_flag is False
        assert ep.combine_flag is False

    def test_from_file_true(self, tmp_path):
        f = tmp_path / "examine.par"
        f.write_text("1\n1\n")
        ep = ExaminePar.from_file(f)
        assert ep.examine_flag is True
        assert ep.combine_flag is True


# ===========================================================================
# PftVersionPar
# ===========================================================================


class TestPftVersionPar:
    def test_defaults(self):
        pv = PftVersionPar()
        assert pv.existing_target_flag is False

    def test_explicit(self):
        pv = PftVersionPar(existing_target_flag=True)
        assert pv.existing_target_flag is True

    def test_from_file_false(self):
        f = BURGERS_DIR / "pft_version.par"
        pv = PftVersionPar.from_file(f)
        assert pv.existing_target_flag is False

    def test_from_file_true(self, tmp_path):
        f = tmp_path / "pft.par"
        f.write_text("1\n")
        pv = PftVersionPar.from_file(f)
        assert pv.existing_target_flag is True


# ===========================================================================
# Module-level aliases and read_* functions
# ===========================================================================


def test_aliases_are_same_classes():
    assert ControlParams is ControlPar
    assert VolumeParams is VolumePar
    assert TargetParams is TargetPar
    assert TrackingParams is TrackPar
    assert SequenceParams is SequencePar
    assert MultimediaParams is MmNp


def test_read_control_par_alias():
    cp = read_control_par(CTRL_PAR)
    assert cp.num_cams == 4


def test_read_volume_par_alias():
    vp = read_volume_par(VOLUME_PAR)
    assert vp.X_lay[0] == 111.111


def test_read_sequence_par_alias():
    sp = read_sequence_par(SEQ_PAR, 4)
    assert sp.num_cams == 4


def test_read_track_par_alias():
    tp = read_track_par(TRACK_PAR)
    assert tp.dvxmin == 0.4


# ===========================================================================
# is_compiled
# ===========================================================================


def test_is_compiled_returns_bool():
    result = is_compiled()
    assert isinstance(result, bool)


# ===========================================================================
# YAML round-trip: VolumePar from_yaml uses PARAMS_YAML criteria
# ===========================================================================


def test_volume_par_yaml_criteria_values():
    vp = VolumePar.from_yaml(PARAMS_YAML)
    assert vp.cn == pytest.approx(0.01)
    assert vp.cnx == pytest.approx(0.3)
    assert vp.cny == pytest.approx(0.3)
    assert vp.csumg == pytest.approx(0.01)


# ===========================================================================
# ControlPar from_yaml — allcam_flag false -> 0
# ===========================================================================


def test_control_par_from_yaml_allcam_false():
    cp = ControlPar.from_yaml(PARAMS_YAML)
    assert cp.allCam_flag == 0


# ===========================================================================
# Edge-case: SequencePar.from_yaml with missing sequence section
# ===========================================================================


def test_sequence_par_from_yaml_missing_section(tmp_path):
    y = tmp_path / "empty.yaml"
    y.write_text("num_cams: 0\n")
    sp = SequencePar.from_yaml(y)
    assert sp.first == 0
    assert sp.last == 0
    assert sp.num_cams == 0


# ===========================================================================
# TrackPar.from_yaml — missing track section falls back to defaults
# ===========================================================================


def test_track_par_from_yaml_missing_section(tmp_path):
    y = tmp_path / "empty.yaml"
    y.write_text("num_cams: 0\n")
    tp = TrackPar.from_yaml(y)
    assert tp.dvxmin == 0.0
    assert tp.add == 0


# ===========================================================================
# VolumePar.from_yaml — missing criteria section falls back to defaults
# ===========================================================================


def test_volume_par_from_yaml_missing_section(tmp_path):
    y = tmp_path / "empty.yaml"
    y.write_text("num_cams: 0\n")
    vp = VolumePar.from_yaml(y)
    np.testing.assert_array_equal(vp.X_lay, [0.0, 0.0])
    assert vp.corrmin == 0.0


# ===========================================================================
# TargetPar to_file then from_file with different values
# ===========================================================================


def test_target_par_roundtrip_no_cr_sz(tmp_path):
    tp = TargetPar(gvthres=[1, 2, 3, 4], discont=0,
                   nnmin=1, nnmax=10, nxmin=1, nxmax=10,
                   nymin=1, nymax=10, sumg_min=50, cr_sz=0)
    f = tmp_path / "t.par"
    tp.to_file(f)
    content = f.read_text()
    # cr_sz=0 should still be written
    assert "0" in content

"""
Pure-Python line-coverage tests for parameter_converters.py.

No skip guard needed — parameter_converters.py has 0 @cython.cfunc decorators;
tests run fine in both compiled and pure-Python mode.

Coverage command (from repo root):
    COVERAGE_FILE=/tmp/.cov_parameter_converters uv run pytest \
      tests/unit/test_parameter_converters_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q 2>&1 | grep -E "(algorithms/parameter_converters\\.|TOTAL|passed|failed|error)"

Fixed bugs (2026-07-10):
    get_calibration_par() now passes img_cal_name= and img_ori= (was img_name=/img_ori0=).
    get_multiplanes_par() now guards None with `or {}` (was crashing on None value).
"""


import numpy as np
import pytest

from openptv2.algorithms.parameter_converters import (
    _check_required,
    _get_section,
    _merge_with_defaults,
    convert_optv_calibrations,
    get_all_params,
    get_calibration_par,
    get_control_par,
    get_examine_par,
    get_multimedia_par,
    get_multiplanes_par,
    get_orient_par,
    get_pft_version_par,
    get_sequence_par,
    get_target_par,
    get_track_par_tuple,
    get_volume_par,
)

# ===========================================================================
# Helper: minimal valid param dict for tests requiring all required params
# ===========================================================================

def _base_params():
    return {
        "num_cams": 4,
        "ptv": {
            "imx": 1024,
            "imy": 1024,
            "pix_x": 0.010,
            "pix_y": 0.010,
        },
        "sequence": {
            "first": 1,
            "last": 100,
            "base_name": ["img/cam1.", "img/cam2.", "img/cam3.", "img/cam4."],
        },
        "criteria": {
            "X_lay": [-10.0, 10.0],
        },
        "cal_ori": {
            "img_cal_name": ["cal/cam1.tif", "cal/cam2.tif"],
            "img_ori": ["cal/cam1.ori", "cal/cam2.ori"],
        },
    }


# ===========================================================================
# _get_section
# ===========================================================================

class TestGetSection:
    def test_first_key_present(self):
        params = {"criteria": {"X_lay": [1, 2]}}
        result = _get_section(params, "criteria", "volume")
        assert result == {"X_lay": [1, 2]}

    def test_second_key_present_when_first_missing(self):
        params = {"volume": {"X_lay": [0, 1]}}
        result = _get_section(params, "criteria", "volume")
        assert result == {"X_lay": [0, 1]}

    def test_no_key_returns_empty_dict(self):
        result = _get_section({}, "criteria", "volume")
        assert result == {}

    def test_key_present_but_falsy_returns_empty_dict(self):
        # None value should be treated as empty dict
        result = _get_section({"criteria": None}, "criteria")
        assert result == {}

    def test_single_key(self):
        params = {"track": {"dvxmin": -5}}
        result = _get_section(params, "track")
        assert result == {"dvxmin": -5}


# ===========================================================================
# _check_required
# ===========================================================================

class TestCheckRequired:
    def test_all_required_present(self):
        defaults = {"imx": None, "imy": None}
        file_dict = {"imx": 1024, "imy": 768}
        missing = _check_required(file_dict, defaults, "ptv")
        assert missing == []

    def test_missing_required_key(self):
        defaults = {"imx": None, "imy": None}
        file_dict = {"imx": 1024}
        missing = _check_required(file_dict, defaults, "ptv")
        assert "imy" in missing

    def test_optional_key_not_checked(self):
        # Default value is not None → key is optional, not checked
        defaults = {"imx": None, "hp_flag": True}
        file_dict = {"imx": 1024}  # hp_flag missing but optional
        missing = _check_required(file_dict, defaults, "ptv")
        assert missing == []

    def test_empty_list_counts_as_missing(self):
        defaults = {"X_lay": None}
        file_dict = {"X_lay": []}
        missing = _check_required(file_dict, defaults, "criteria")
        assert "X_lay" in missing

    def test_case_insensitive_key_lookup(self):
        defaults = {"imx": None}
        # Uppercase key in file, lowercase in defaults
        file_dict = {"IMX": 1024}
        missing = _check_required(file_dict, defaults, "ptv")
        assert missing == []

    def test_multiple_missing_keys(self):
        defaults = {"a": None, "b": None, "c": None}
        file_dict = {}
        missing = _check_required(file_dict, defaults, "test")
        assert set(missing) == {"a", "b", "c"}


# ===========================================================================
# _merge_with_defaults
# ===========================================================================

class TestMergeWithDefaults:
    def test_file_values_override_defaults(self):
        defaults = {"hp_flag": True, "chfield": 0}
        file_dict = {"hp_flag": False}
        result = _merge_with_defaults(file_dict, defaults)
        assert result["hp_flag"] is False
        assert result["chfield"] == 0

    def test_none_values_in_file_are_skipped(self):
        defaults = {"hp_flag": True}
        file_dict = {"hp_flag": None}
        result = _merge_with_defaults(file_dict, defaults)
        # None in file dict should NOT override the default
        assert result["hp_flag"] is True

    def test_extra_file_keys_included(self):
        defaults = {"a": 1}
        file_dict = {"b": 2}
        result = _merge_with_defaults(file_dict, defaults)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_empty_file_dict_returns_defaults(self):
        defaults = {"a": 1, "b": 2}
        result = _merge_with_defaults({}, defaults)
        assert result == {"a": 1, "b": 2}

    def test_zero_and_false_are_kept_from_file(self):
        defaults = {"a": 99, "b": True}
        file_dict = {"a": 0, "b": False}
        result = _merge_with_defaults(file_dict, defaults)
        assert result["a"] == 0
        assert result["b"] is False


# ===========================================================================
# get_multimedia_par
# ===========================================================================

class TestGetMultimediaPar:
    def test_defaults_when_no_ptv(self):
        from openptv2.algorithms.parameters import MultimediaPar
        mm = get_multimedia_par({})
        assert isinstance(mm, MultimediaPar)
        assert mm.n1 == 1.0
        assert mm.n3 == 1.46

    def test_ptv_section_overrides(self):
        params = {"ptv": {"mmp_n1": 1.5, "mmp_n2": 1.4, "mmp_d": 8.0, "mmp_n3": 1.6}}
        mm = get_multimedia_par(params)
        assert mm.n1 == 1.5
        assert mm.n3 == 1.6

    def test_ptv_none_uses_defaults(self):
        # ptv key exists but is None
        mm = get_multimedia_par({"ptv": None})
        assert mm.n1 == 1.0

    def test_n2_and_d_are_lists(self):
        mm = get_multimedia_par({})
        assert hasattr(mm, "n2")
        assert hasattr(mm, "d")


# ===========================================================================
# get_control_par
# ===========================================================================

class TestGetControlPar:
    def test_valid_params(self):
        from openptv2.algorithms.parameters import ControlPar
        p = _base_params()
        cpar = get_control_par(p)
        assert isinstance(cpar, ControlPar)
        assert cpar.imx == 1024
        assert cpar.imy == 1024
        assert cpar.pix_x == pytest.approx(0.010)
        assert cpar.pix_y == pytest.approx(0.010)
        assert cpar.num_cams == 4

    def test_missing_required_raises_value_error(self):
        params = {"ptv": {"imx": 1024}}  # missing imy, pix_x, pix_y
        with pytest.raises(ValueError, match="Missing required ptv parameters"):
            get_control_par(params)

    def test_missing_ptv_section_raises(self):
        with pytest.raises(ValueError, match="Missing required ptv parameters"):
            get_control_par({})

    def test_ptv_none_raises(self):
        with pytest.raises(ValueError):
            get_control_par({"ptv": None})

    def test_hp_flag_true(self):
        p = _base_params()
        p["ptv"]["hp_flag"] = True
        cpar = get_control_par(p)
        assert cpar.hp_flag == 1

    def test_hp_flag_false(self):
        p = _base_params()
        p["ptv"]["hp_flag"] = False
        cpar = get_control_par(p)
        assert cpar.hp_flag == 0

    def test_allcam_flag_true(self):
        p = _base_params()
        p["ptv"]["allcam_flag"] = True
        cpar = get_control_par(p)
        assert cpar.allCam_flag == 1

    def test_tiff_flag_false(self):
        p = _base_params()
        p["ptv"]["tiff_flag"] = False
        cpar = get_control_par(p)
        assert cpar.tiff_flag == 0

    def test_chfield(self):
        p = _base_params()
        p["ptv"]["chfield"] = 2
        cpar = get_control_par(p)
        assert cpar.chfield == 2

    def test_multimedia_params_attached(self):
        cpar = get_control_par(_base_params())
        assert cpar.mm is not None

    def test_num_cams_defaults_to_4(self):
        p = _base_params()
        del p["num_cams"]
        cpar = get_control_par(p)
        assert cpar.num_cams == 4

    def test_custom_num_cams(self):
        p = _base_params()
        p["num_cams"] = 2
        cpar = get_control_par(p)
        assert cpar.num_cams == 2


# ===========================================================================
# get_sequence_par
# ===========================================================================

class TestGetSequencePar:
    def test_valid_params(self):
        from openptv2.algorithms.parameters import SequencePar
        p = _base_params()
        spar = get_sequence_par(p)
        assert isinstance(spar, SequencePar)
        assert spar.first == 1
        assert spar.last == 100

    def test_missing_first_raises(self):
        params = {"sequence": {"last": 100, "base_name": ["cam1"]}}
        with pytest.raises(ValueError, match="Missing required sequence parameters"):
            get_sequence_par(params)

    def test_missing_last_raises(self):
        params = {"sequence": {"first": 1, "base_name": ["cam1"]}}
        with pytest.raises(ValueError, match="Missing required sequence parameters"):
            get_sequence_par(params)

    def test_missing_base_name_raises(self):
        params = {"sequence": {"first": 1, "last": 100}}
        with pytest.raises(ValueError, match="Missing required sequence parameters"):
            get_sequence_par(params)

    def test_missing_sequence_section_raises(self):
        with pytest.raises(ValueError, match="Missing required sequence parameters"):
            get_sequence_par({})

    def test_sequence_none_raises(self):
        with pytest.raises(ValueError):
            get_sequence_par({"sequence": None})

    def test_img_base_name(self):
        p = _base_params()
        spar = get_sequence_par(p)
        assert spar.img_base_name == ["img/cam1.", "img/cam2.", "img/cam3.", "img/cam4."]


# ===========================================================================
# get_volume_par
# ===========================================================================

class TestGetVolumePar:
    def test_valid_criteria_key(self):
        from openptv2.algorithms.parameters import VolumePar
        p = _base_params()
        vpar = get_volume_par(p)
        assert isinstance(vpar, VolumePar)
        np.testing.assert_array_equal(vpar.X_lay, [-10.0, 10.0])

    def test_volume_key_alias(self):
        """Accept 'volume' as alternative to 'criteria'."""
        from openptv2.algorithms.parameters import VolumePar
        params = {"volume": {"X_lay": [0.0, 5.0]}}
        vpar = get_volume_par(params)
        assert isinstance(vpar, VolumePar)

    def test_x_lay_lowercase_alias(self):
        """Accept 'x_lay' as alternative to 'X_lay'."""
        params = {"criteria": {"x_lay": [1.0, 2.0]}}
        vpar = get_volume_par(params)
        np.testing.assert_array_equal(vpar.X_lay, [1.0, 2.0])

    def test_missing_x_lay_raises(self):
        params = {"criteria": {"cn": 0.5}}  # X_lay missing
        with pytest.raises(ValueError, match="Missing required criteria parameters"):
            get_volume_par(params)

    def test_zmin_lay_uppercase(self):
        params = {"criteria": {"X_lay": [0, 1], "Zmin_lay": [-100], "Zmax_lay": [100]}}
        vpar = get_volume_par(params)
        assert vpar.Zmin_lay[0] == -100.0
        assert vpar.Zmax_lay[0] == 100.0

    def test_zmin_lay_lowercase_alias(self):
        params = {"criteria": {"X_lay": [0, 1], "z_min_lay": [-50], "z_max_lay": [50]}}
        vpar = get_volume_par(params)
        assert vpar.Zmin_lay[0] == -50.0
        assert vpar.Zmax_lay[0] == 50.0

    def test_optional_fields_defaults(self):
        params = {"criteria": {"X_lay": [0, 1]}}
        vpar = get_volume_par(params)
        assert vpar.cn == 0.0
        assert vpar.cnx == 0.0
        assert vpar.cny == 0.0
        assert vpar.csumg == 0.0
        assert vpar.eps0 == 0.0
        assert vpar.corrmin == 0.0

    def test_optional_fields_custom(self):
        params = {"criteria": {
            "X_lay": [0, 1],
            "cn": 0.5,
            "cnx": 1.0,
            "cny": 2.0,
            "csumg": 3.0,
            "eps0": 0.1,
            "corrmin": 0.3,
        }}
        vpar = get_volume_par(params)
        assert vpar.cn == pytest.approx(0.5)
        assert vpar.cnx == pytest.approx(1.0)


# ===========================================================================
# get_track_par_tuple
# ===========================================================================

class TestGetTrackParTuple:
    def test_all_defaults_empty_params(self):
        from openptv2.algorithms.parameters import TrackParTuple
        tpar = get_track_par_tuple({})
        assert isinstance(tpar, TrackParTuple)
        assert tpar.dvxmin == -20
        assert tpar.dvxmax == 20

    def test_track_key(self):
        params = {"track": {"dvxmin": -5, "dvxmax": 5}}
        tpar = get_track_par_tuple(params)
        assert tpar.dvxmin == -5
        assert tpar.dvxmax == 5

    def test_tracking_key_alias(self):
        """Accept 'tracking' as alternative to 'track'."""
        params = {"tracking": {"dvxmin": -3}}
        tpar = get_track_par_tuple(params)
        assert tpar.dvxmin == -3

    def test_angle_key(self):
        """'angle' in track section takes precedence over dangle."""
        params = {"track": {"angle": 25}}
        tpar = get_track_par_tuple(params)
        assert tpar.dangle == 25

    def test_dangle_key(self):
        """'dangle' key also accepted."""
        params = {"track": {"dangle": 15}}
        tpar = get_track_par_tuple(params)
        assert tpar.dangle == 15

    def test_neither_angle_nor_dangle_uses_default(self):
        params = {"track": {"dvxmin": -1}}
        tpar = get_track_par_tuple(params)
        assert tpar.dangle == 10  # default from DEFAULT_TRACK

    def test_all_fields_present(self):
        params = {"track": {
            "dvxmin": -1, "dvxmax": 1,
            "dvymin": -2, "dvymax": 2,
            "dvzmin": -3, "dvzmax": 3,
            "angle": 5, "dacc": 1, "add": 1,
            "dsumg": 0.5, "dn": 0.3, "dnx": 0.1, "dny": 0.2,
        }}
        tpar = get_track_par_tuple(params)
        assert tpar.dvxmin == -1
        assert tpar.dvymin == -2
        assert tpar.dvzmin == -3
        assert tpar.dangle == 5
        assert tpar.add == 1


# ===========================================================================
# get_target_par
# ===========================================================================

class TestGetTargetPar:
    def test_all_defaults(self):
        from openptv2.algorithms.parameters import TargetPar
        targ = get_target_par({})
        assert isinstance(targ, TargetPar)
        assert targ.discont == 100
        assert targ.nnmin == 4
        assert targ.nnmax == 500

    def test_targ_rec_key(self):
        params = {"targ_rec": {"discont": 50}}
        targ = get_target_par(params)
        assert targ.discont == 50

    def test_targ_key_alias(self):
        """'targ' is an alternative key."""
        params = {"targ": {"discont": 25, "nnmin": 2}}
        targ = get_target_par(params)
        assert targ.discont == 25
        assert targ.nnmin == 2

    def test_detect_plate_key(self):
        params = {
            "targ_rec": {},
            "detect_plate": {"gvth_1": 80, "gvth_2": 90},
        }
        targ = get_target_par(params)
        assert targ.gvthres[0] == 80
        assert targ.gvthres[1] == 90

    def test_plate_key_alias(self):
        """'plate' is an alternative to 'detect_plate'."""
        params = {"plate": {"gvth_1": 60}}
        targ = get_target_par(params)
        assert targ.gvthres[0] == 60

    def test_gvthres_default_four_entries(self):
        targ = get_target_par({})
        assert len(targ.gvthres) == 4
        assert targ.gvthres[0] == 40  # default

    def test_full_targ_rec_fields(self):
        params = {"targ_rec": {
            "discont": 5,
            "nnmin": 1,
            "nnmax": 200,
            "nxmin": 1,
            "nxmax": 50,
            "nymin": 1,
            "nymax": 50,
            "sumg_min": 100,
            "cr_sz": 3,
        }}
        targ = get_target_par(params)
        assert targ.discont == 5
        assert targ.cr_sz == 3


# ===========================================================================
# get_calibration_par
# ===========================================================================

class TestGetCalibrationPar:
    def test_basic_success(self):
        from openptv2.algorithms.parameters import CalibrationPar
        params = {
            "cal_ori": {
                "img_cal_name": ["cam1.tif"],
                "img_ori": ["cam1.ori"],
            }
        }
        result = get_calibration_par(params)
        assert isinstance(result, CalibrationPar)
        assert result.img_cal_name == ["cam1.tif"]
        assert result.img_ori == ["cam1.ori"]

    def test_calib_key_alias(self):
        """'calib' is an alternative key to 'cal_ori'."""
        from openptv2.algorithms.parameters import CalibrationPar
        params = {
            "calib": {
                "img_cal_name": ["cam1.tif"],
                "img_ori": ["cam1.ori"],
            }
        }
        result = get_calibration_par(params)
        assert isinstance(result, CalibrationPar)

    def test_missing_required_raises_value_error(self):
        params = {"cal_ori": {"img_ori": ["cam1.ori"]}}
        with pytest.raises(ValueError, match="Missing required cal_ori parameters"):
            get_calibration_par(params)

    def test_missing_img_ori_raises_value_error(self):
        params = {"cal_ori": {"img_cal_name": ["cam1.tif"]}}
        with pytest.raises(ValueError, match="Missing required cal_ori parameters"):
            get_calibration_par(params)

    def test_empty_cal_ori_raises_value_error(self):
        with pytest.raises(ValueError):
            get_calibration_par({"cal_ori": {}})

    def test_missing_section_raises_value_error(self):
        with pytest.raises(ValueError):
            get_calibration_par({})

    def test_optional_defaults(self):
        from openptv2.algorithms.parameters import CalibrationPar
        params = {
            "cal_ori": {
                "img_cal_name": ["cam.tif"],
                "img_ori": ["cam.ori"],
                "fixp_name": "fixp.txt",
                "tiff_flag": False,
                "pair_flag": True,
                "chfield": 1,
            }
        }
        result = get_calibration_par(params)
        assert isinstance(result, CalibrationPar)
        assert result.tiff_flag == 0
        assert result.pair_flag == 1
        assert result.chfield == 1


# ===========================================================================
# get_orient_par
# ===========================================================================

class TestGetOrientPar:
    def test_all_defaults(self):
        from openptv2.algorithms.parameters import OrientPar
        opar = get_orient_par({})
        assert isinstance(opar, OrientPar)
        assert opar.useflag == 0

    def test_with_orient_section(self):
        params = {"orient": {"useflag": 1, "ccflag": 1, "xhflag": 1}}
        opar = get_orient_par(params)
        assert opar.useflag == 1
        assert opar.ccflag == 1
        assert opar.xhflag == 1

    def test_orient_none_uses_defaults(self):
        opar = get_orient_par({"orient": None})
        assert opar.useflag == 0

    def test_all_flags(self):
        flags = {
            "useflag": 1, "ccflag": 1, "xhflag": 1, "yhflag": 1,
            "k1flag": 1, "k2flag": 1, "k3flag": 1,
            "p1flag": 1, "p2flag": 1, "scxflag": 1,
            "sheflag": 1, "interfflag": 1,
        }
        opar = get_orient_par({"orient": flags})
        assert opar.k1flag == 1
        assert opar.interfflag == 1


# ===========================================================================
# get_multiplanes_par
# ===========================================================================

class TestGetMultiplanesPar:
    def test_defaults(self):
        from openptv2.algorithms.parameters import MultiPlanesPar
        mpar = get_multiplanes_par({})
        assert isinstance(mpar, MultiPlanesPar)
        assert mpar.num_planes == 0

    def test_with_multi_planes_section(self):
        params = {"multi_planes": {"n_planes": 2, "plane_name": ["p1", "p2"]}}
        mpar = get_multiplanes_par(params)
        assert mpar.num_planes == 2
        assert mpar.filename == ["p1", "p2"]

    def test_none_value_falls_back_to_defaults(self):
        """None value is now treated as missing — falls back to defaults."""
        from openptv2.algorithms.parameters import MultiPlanesPar
        result = get_multiplanes_par({"multi_planes": None})
        assert isinstance(result, MultiPlanesPar)


# ===========================================================================
# get_examine_par
# ===========================================================================

class TestGetExaminePar:
    def test_defaults(self):
        from openptv2.algorithms.parameters import ExaminePar
        epar = get_examine_par({})
        assert isinstance(epar, ExaminePar)
        assert epar.examine_flag is False
        assert epar.combine_flag is False

    def test_with_examine_section(self):
        params = {"examine": {"Examine_Flag": True, "Combine_Flag": True}}
        epar = get_examine_par(params)
        assert epar.examine_flag is True
        assert epar.combine_flag is True

    def test_none_examine_uses_defaults(self):
        epar = get_examine_par({"examine": None})
        assert epar.examine_flag is False


# ===========================================================================
# get_pft_version_par
# ===========================================================================

class TestGetPftVersionPar:
    def test_defaults(self):
        from openptv2.algorithms.parameters import PftVersionPar
        ppar = get_pft_version_par({})
        assert isinstance(ppar, PftVersionPar)
        assert ppar.existing_target_flag == 0

    def test_with_pft_version_section(self):
        params = {"pft_version": {"Existing_Target": 1}}
        ppar = get_pft_version_par(params)
        assert ppar.existing_target_flag == 1

    def test_none_pft_version_uses_defaults(self):
        ppar = get_pft_version_par({"pft_version": None})
        assert ppar.existing_target_flag == 0


# ===========================================================================
# get_all_params
# ===========================================================================

class TestGetAllParams:
    def test_full_params(self):
        p = _base_params()
        result = get_all_params(p)

        assert "cpar" in result
        assert "spar" in result
        assert "vpar" in result
        assert "tpar" in result
        assert "targpar" in result
        assert "calpar" in result
        assert "orientpar" in result
        assert "multiplanespar" in result
        assert "examinepar" in result
        assert "pftversionpar" in result

    def test_propagates_missing_param_error(self):
        with pytest.raises(ValueError):
            get_all_params({})


# ===========================================================================
# convert_optv_calibrations
# ===========================================================================

class TestConvertOptvCalibrations:
    def test_empty_list_returns_empty_list(self):
        result = convert_optv_calibrations([])
        assert result == []

    def test_mock_cal_falls_through_to_default_on_attribute_error(self):
        """
        A calibration object that triggers AttributeError on
        _create_default_exterior() causes the except branch to fire,
        returning a default Calibration().
        """
        from openptv2.algorithms.calibration import Calibration

        class MockCal:
            def get_pos(self): return [0.0, 0.0, 100.0]
            def get_angles(self): return [0.0, 0.0, 0.0]
            def get_primary_point(self): return [0.0, 0.0, 50.0]
            def get_glass_vec(self): return np.zeros(3)
            def get_radial_distortion(self): return [0.0, 0.0, 0.0]
            def get_decentering(self): return [0.0, 0.0]
            def get_affine(self): return [1.0, 0.0]

        result = convert_optv_calibrations([MockCal()])
        # Should return one entry (from the except fallback)
        assert len(result) == 1
        assert isinstance(result[0], Calibration)

    def test_multiple_cals_all_fallback(self):
        class FailCal:
            def get_pos(self): raise RuntimeError("fail")

        result = convert_optv_calibrations([FailCal(), FailCal()])
        assert len(result) == 2

    def test_success_path_with_mocked_calibration_class(self, monkeypatch):
        """
        Monkeypatch openptv2.algorithms.calibration.Calibration so that
        _create_default_exterior and _create_default_interior exist, allowing
        the happy-path code inside convert_optv_calibrations to run.
        """
        import openptv2.algorithms.calibration as cal_mod

        # Build mock ext_par: a dict where values are 0-d numpy arrays
        def make_mock_ext_par():
            d = {}
            for k in ("x0", "y0", "z0", "omega", "phi", "kappa"):
                d[k] = np.zeros(())
            return d

        # Build mock int_par: object with xh, yh, cc as 0-d numpy arrays
        class MockIntPar:
            xh = np.zeros(())
            yh = np.zeros(())
            cc = np.zeros(())

        class MockPyCal:
            @staticmethod
            def _create_default_exterior():
                return make_mock_ext_par()

            @staticmethod
            def _create_default_interior():
                return MockIntPar()

            def __init__(self, ext_par=None, int_par=None, glass_par=None,
                         added_par=None, **kwargs):
                pass

        def mock_rotation_matrix(ext_par):
            pass  # no-op

        monkeypatch.setattr(cal_mod, "Calibration", MockPyCal)
        monkeypatch.setattr(cal_mod, "rotation_matrix", mock_rotation_matrix, raising=False)

        class MockOptvCal:
            def get_pos(self): return [1.0, 2.0, 300.0]
            def get_angles(self): return [0.01, 0.02, 0.03]
            def get_primary_point(self): return [0.5, -0.3, 50.0]
            def get_glass_vec(self): return np.array([0.0, 0.0, 60.0])
            def get_radial_distortion(self): return [1e-5, 2e-7, 0.0]
            def get_decentering(self): return [1e-6, 2e-6]
            def get_affine(self): return [1.0, 0.0]

        result = convert_optv_calibrations([MockOptvCal()])
        assert len(result) == 1
        assert isinstance(result[0], MockPyCal)

    def test_success_path_with_two_cals(self, monkeypatch):
        """Two cals on success path; verify both are returned."""
        import openptv2.algorithms.calibration as cal_mod

        def make_mock_ext_par():
            return {k: np.zeros(()) for k in ("x0", "y0", "z0", "omega", "phi", "kappa")}

        class MockIntPar:
            xh = np.zeros(())
            yh = np.zeros(())
            cc = np.zeros(())

        class MockPyCal2:
            @staticmethod
            def _create_default_exterior(): return make_mock_ext_par()
            @staticmethod
            def _create_default_interior(): return MockIntPar()
            def __init__(self, **kwargs): pass

        def mock_rm(ext_par): pass

        monkeypatch.setattr(cal_mod, "Calibration", MockPyCal2)
        monkeypatch.setattr(cal_mod, "rotation_matrix", mock_rm, raising=False)

        class MockOptvCal2:
            def get_pos(self): return [0.0, 0.0, 100.0]
            def get_angles(self): return [0.0, 0.0, 0.0]
            def get_primary_point(self): return [0.0, 0.0, 50.0]
            def get_glass_vec(self): return np.zeros(3)
            def get_radial_distortion(self): return [0.0, 0.0, 0.0]
            def get_decentering(self): return [0.0, 0.0]
            def get_affine(self): return [1.0, 0.0]

        result = convert_optv_calibrations([MockOptvCal2(), MockOptvCal2()])
        assert len(result) == 2

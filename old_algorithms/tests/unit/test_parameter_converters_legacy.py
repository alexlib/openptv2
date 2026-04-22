"""Tests for parameter_converters module.

Tests conversion from YAML parameters (with key variations) to algorithm dataclasses.
"""

import pytest
from algorithms.parameter_converters import (
    get_control_par,
    get_sequence_par,
    get_volume_par,
    get_track_par_tuple,
    get_multimedia_par,
    get_target_par,
    get_calibration_par,
    get_orient_par,
    get_multiplanes_par,
    get_examine_par,
    get_pft_version_par,
    get_all_params,
)


class TestVolumeParConverter:
    """Tests for VolumePar conversion with key variations."""

    def test_volume_par_criteria_key(self):
        """Test conversion using 'criteria' key."""
        yaml_params = {
            "criteria": {
                "X_lay": [0, 100],
                "Zmin_lay": [-50],
                "Zmax_lay": [50],
            }
        }
        vpar = get_volume_par(yaml_params)
        assert vpar.x_lay == [0, 100]
        assert vpar.z_min_lay == [-50]
        assert vpar.z_max_lay == [50]

    def test_volume_par_volume_key(self):
        """Test conversion using alternative 'volume' key."""
        yaml_params = {
            "volume": {
                "X_lay": [10, 200],
                "Zmin_lay": [-100],
                "Zmax_lay": [100],
            }
        }
        vpar = get_volume_par(yaml_params)
        assert vpar.x_lay == [10, 200]
        assert vpar.z_min_lay == [-100]
        assert vpar.z_max_lay == [100]

    def test_volume_par_lowercase_keys(self):
        """Test conversion using lowercase key variants."""
        yaml_params = {
            "criteria": {
                "x_lay": [20, 150],
                "z_min_lay": [-30],
                "z_max_lay": [30],
            }
        }
        vpar = get_volume_par(yaml_params)
        assert vpar.x_lay == [20, 150]
        assert vpar.z_min_lay == [-30]
        assert vpar.z_max_lay == [30]

    def test_volume_par_defaults(self):
        """Test default values when optional params not specified."""
        yaml_params = {
            "criteria": {
                "X_lay": [0, 100],
            }
        }
        vpar = get_volume_par(yaml_params)
        assert vpar.z_min_lay == [-50]
        assert vpar.z_max_lay == [50]
        assert vpar.cn == 0.0

    def test_volume_par_missing_required(self):
        """Test ValueError when required X_lay is missing."""
        yaml_params = {"criteria": {}}
        with pytest.raises(ValueError, match="Missing required criteria parameters"):
            get_volume_par(yaml_params)

    def test_volume_par_empty_criteria(self):
        """Test ValueError when criteria section is empty."""
        yaml_params = {}
        with pytest.raises(ValueError, match="Missing required criteria parameters"):
            get_volume_par(yaml_params)


class TestTrackParTupleConverter:
    """Tests for TrackParTuple conversion with key variations."""

    def test_track_par_track_key(self):
        """Test conversion using 'track' key."""
        yaml_params = {
            "track": {
                "dvxmin": -10,
                "dvxmax": 10,
                "dvymin": -20,
                "dvymax": 20,
                "dvzmin": -30,
                "dvzmax": 30,
                "dangle": 15,
                "dacc": 5,
                "add": 1,
                "dsumg": 2,
                "dn": 3,
                "dnx": 4,
                "dny": 5,
            }
        }
        tpar = get_track_par_tuple(yaml_params)
        assert tpar.dvxmin == -10
        assert tpar.dvxmax == 10
        assert tpar.dangle == 15

    def test_track_par_tracking_key(self):
        """Test conversion using alternative 'tracking' key."""
        yaml_params = {
            "tracking": {
                "dvxmin": -5,
                "dvxmax": 5,
                "dvymin": -5,
                "dvymax": 5,
                "dvzmin": -5,
                "dvzmax": 5,
                "dangle": 20,
                "dacc": 3,
                "add": 0,
                "dsumg": 0,
                "dn": 1,
                "dnx": 0,
                "dny": 0,
            }
        }
        tpar = get_track_par_tuple(yaml_params)
        assert tpar.dvxmin == -5
        assert tpar.dangle == 20

    def test_track_par_angle_key_variation(self):
        """Test conversion using 'angle' as alternative to 'dangle'."""
        yaml_params = {
            "track": {
                "dvxmin": -20,
                "dvxmax": 20,
                "dvymin": -20,
                "dvymax": 20,
                "dvzmin": -20,
                "dvzmax": 20,
                "angle": 25,
                "dacc": 2,
                "add": 0,
                "dsumg": 0,
                "dn": 1,
                "dnx": 0,
                "dny": 0,
            }
        }
        tpar = get_track_par_tuple(yaml_params)
        assert tpar.dangle == 25

    def test_track_par_defaults(self):
        """Test default values when optional params not specified."""
        yaml_params = {
            "track": {
                "dvxmin": -20,
                "dvxmax": 20,
                "dvymin": -20,
                "dvymax": 20,
                "dvzmin": -20,
                "dvzmax": 20,
                "dacc": 2,
                "add": 0,
                "dsumg": 0,
                "dn": 1,
                "dnx": 0,
                "dny": 0,
            }
        }
        tpar = get_track_par_tuple(yaml_params)
        assert tpar.dangle == 10
        assert tpar.add == 0


class TestControlParConverter:
    """Tests for ControlPar conversion."""

    def test_control_par_required_params(self):
        """Test conversion with required parameters."""
        yaml_params = {
            "ptv": {
                "imx": 1280,
                "imy": 1024,
                "pix_x": 0.012,
                "pix_y": 0.012,
            },
            "num_cams": 4,
        }
        cpar = get_control_par(yaml_params)
        assert cpar.imx == 1280
        assert cpar.imy == 1024
        assert cpar.pix_x == 0.012
        assert cpar.pix_y == 0.012
        assert cpar.num_cams == 4

    def test_control_par_optional_params(self):
        """Test optional parameters use defaults."""
        yaml_params = {
            "ptv": {
                "imx": 2048,
                "imy": 2048,
                "pix_x": 0.005,
                "pix_y": 0.005,
            }
        }
        cpar = get_control_par(yaml_params)
        assert cpar.hp_flag == 1
        assert cpar.tiff_flag == 1
        assert cpar.chfield == 0

    def test_control_par_missing_required(self):
        """Test ValueError when required params missing."""
        yaml_params = {
            "ptv": {
                "imx": 1280,
            }
        }
        with pytest.raises(ValueError, match="Missing required ptv parameters"):
            get_control_par(yaml_params)


class TestSequenceParConverter:
    """Tests for SequencePar conversion."""

    def test_sequence_par_required_params(self):
        """Test conversion with required parameters."""
        yaml_params = {
            "sequence": {
                "first": 1,
                "last": 100,
                "base_name": "test/img",
            }
        }
        spar = get_sequence_par(yaml_params)
        assert spar.first == 1
        assert spar.last == 100
        assert spar.img_base_name == "test/img"

    def test_sequence_par_missing_required(self):
        """Test ValueError when required params missing."""
        yaml_params = {
            "sequence": {
                "first": 1,
            }
        }
        with pytest.raises(ValueError, match="Missing required sequence parameters"):
            get_sequence_par(yaml_params)


class TestMultimediaParConverter:
    """Tests for MultimediaPar conversion."""

    def test_multimedia_par_defaults(self):
        """Test default multimedia parameters."""
        yaml_params = {"ptv": {}}
        mm = get_multimedia_par(yaml_params)
        assert mm.n1 == 1.0
        assert mm.n2 == [1.33]
        assert mm.d == [6.0]
        assert mm.n3 == 1.46

    def test_multimedia_par_custom_values(self):
        """Test custom multimedia parameters."""
        yaml_params = {
            "ptv": {
                "mmp_n1": 1.5,
                "mmp_n2": 1.4,
                "mmp_d": 10.0,
                "mmp_n3": 1.5,
            }
        }
        mm = get_multimedia_par(yaml_params)
        assert mm.n1 == 1.5
        assert mm.n2 == [1.4]
        assert mm.d == [10.0]
        assert mm.n3 == 1.5


class TestGetAllParams:
    """Tests for get_all_params function."""

    def test_get_all_params_complete(self):
        """Test getting all parameters at once."""
        yaml_params = {
            "ptv": {
                "imx": 1280,
                "imy": 1024,
                "pix_x": 0.012,
                "pix_y": 0.012,
            },
            "num_cams": 4,
            "sequence": {
                "first": 1,
                "last": 100,
                "base_name": "test",
            },
            "criteria": {
                "X_lay": [0, 100],
            },
            "track": {
                "dvxmin": -20,
                "dvxmax": 20,
                "dvymin": -20,
                "dvymax": 20,
                "dvzmin": -20,
                "dvzmax": 20,
                "dacc": 2,
                "add": 0,
                "dsumg": 0,
                "dn": 1,
                "dnx": 0,
                "dny": 0,
            },
            "targ_rec": {
                "discont": 100,
                "nnmin": 4,
                "nnmax": 500,
            },
            "cal_ori": {
                "img_cal_name": "cal",
                "img_ori": "ori",
            },
            "orient": {
                "useflag": 0,
            },
            "examine": {
                "Examine_Flag": False,
            },
            "pft_version": {
                "Existing_Target": 0,
            },
        }
        params = get_all_params(yaml_params)
        assert "cpar" in params
        assert "spar" in params
        assert "vpar" in params
        assert "tpar" in params

"""
Automated unit tests for TraitsUI parameter GUI models.
Verifies that Main_Params, Calib_Params, and Tracking_Params correctly
initialize from ParameterManager and save changes back.
"""

import tempfile
from pathlib import Path

import pytest

from openptv2.gui.experiment import Experiment
from openptv2.gui.parameter_gui import (
    CalHandler,
    Calib_Params,
    Main_Params,
    ParamHandler,
    TrackHandler,
    Tracking_Params,
)
from openptv2.gui.parameter_manager import ParameterManager


class DummyInfo:
    """Mock info object passed to Handler.closed."""

    def __init__(self, obj):
        self.object = obj


@pytest.fixture
def temp_experiment():
    """Fixture to create a temporary YAML file and load an Experiment with standard parameters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        yaml_file = tmp_path / "parameters_test.yaml"

        # Create a basic valid set of parameters
        pm = ParameterManager()
        pm.num_cams = 4
        pm.parameters = {
            "num_cams": 4,
            "ptv": {
                "img_name": ["cam1", "cam2", "cam3", "cam4"],
                "img_cal": ["cal1", "cal2", "cal3", "cal4"],
                "hp_flag": True,
                "allcam_flag": False,
                "tiff_flag": True,
                "imx": 1280,
                "imy": 1024,
                "pix_x": 0.012,
                "pix_y": 0.012,
                "chfield": 0,
                "mmp_n1": 1.0,
                "mmp_n2": 1.5,
                "mmp_n3": 1.33,
                "mmp_d": 5.0,
                "splitter": False,
                "negative": False,
            },
            "cal_ori": {
                "fixp_name": "fixpoint.txt",
                "img_cal_name": ["cal1", "cal2", "cal3", "cal4"],
                "img_ori": ["ori1", "ori2", "ori3", "ori4"],
                "cal_splitter": False,
            },
            "targ_rec": {
                "gvthres": [100, 110, 120, 130],
                "disco": 10,
                "nnmin": 10,
                "nnmax": 100,
                "nxmin": 2,
                "nxmax": 20,
                "nymin": 2,
                "nymax": 20,
                "sumg_min": 500,
                "cr_sz": 3,
            },
            "pft_version": {"Existing_Target": 1},
            "man_ori": {"nr": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
            "sequence": {
                "base_name": ["seq1", "seq2", "seq3", "seq4"],
                "first": 10000,
                "last": 10005,
            },
            "criteria": {
                "X_lay": [-100, 100],
                "Zmin_lay": [-50, -50],
                "Zmax_lay": [50, 50],
                "cnx": 0.5,
                "cny": 0.5,
                "cn": 0.8,
                "csumg": 1000,
                "corrmin": 0.3,
                "eps0": 0.1,
            },
            "masking": {"mask_flag": False, "mask_base_name": "mask"},
            "track": {
                "dvxmin": -10.0,
                "dvxmax": 10.0,
                "dvymin": -10.0,
                "dvymax": 10.0,
                "dvzmin": -10.0,
                "dvzmax": 10.0,
                "angle": 45.0,
                "dacc": 5.0,
                "flagNewParticles": True,
            },
            "detect_plate": {
                "gvth_1": 50,
                "gvth_2": 50,
                "gvth_3": 50,
                "gvth_4": 50,
                "tol_dis": 5,
                "min_npix": 5,
                "max_npix": 500,
                "min_npix_x": 1,
                "max_npix_x": 100,
                "min_npix_y": 1,
                "max_npix_y": 100,
                "sum_grey": 200,
                "size_cross": 2,
            },
            "examine": {"Examine_Flag": False, "Combine_Flag": False},
            "orient": {
                "pnfo": 10,
                "cc": 1,
                "xh": 1,
                "yh": 1,
                "k1": 1,
                "k2": 1,
                "k3": 1,
                "p1": 1,
                "p2": 1,
                "scale": 1,
                "shear": 1,
                "interf": 1,
            },
            "shaking": {
                "shaking_first_frame": 10000,
                "shaking_last_frame": 10005,
                "shaking_max_num_points": 1000,
                "shaking_max_num_frames": 10,
            },
            "dumbbell": {
                "dumbbell_eps": 0.1,
                "dumbbell_scale": 1.0,
                "dumbbell_gradient_descent": 0.1,
                "dumbbell_penalty_weight": 1.0,
                "dumbbell_step": 1,
                "dumbbell_niter": 100,
                "dumbbell_fixed_camera": -1,
            },
        }

        # Write to YAML to allow experiment to read it
        pm.to_yaml(yaml_file)

        # Create experiment and set active
        exp = Experiment(pm)
        exp.addParamset("test", yaml_file)
        exp.set_active(0)

        yield exp


def test_main_params_binding(temp_experiment):
    """Test Main_Params initialization and saving through ParamHandler."""
    exp = temp_experiment

    # Create Main_Params
    main_params = Main_Params(exp)

    # Verify attributes initialized correctly from ParameterManager
    assert main_params.Num_Cam == 4
    assert main_params.imx == 1280
    assert main_params.imy == 1024
    assert main_params.HighPass is True
    assert main_params.Refr_Air == 1.0
    assert main_params.Refr_Glass == 1.5
    assert main_params.Refr_Water == 1.33
    assert main_params.Xmin == -100
    assert main_params.Xmax == 100

    # Modify values on Main_Params
    main_params.Num_Cam = 2
    main_params.imx = 1000
    main_params.HighPass = False
    main_params.Xmin = -200

    # Trigger ParamHandler.closed simulating clicking OK
    handler = ParamHandler()
    info = DummyInfo(main_params)
    handler.closed(info, is_ok=True)

    # Verify changes wrote back to the ParameterManager
    assert exp.pm.parameters["num_cams"] == 2
    assert exp.pm.parameters["ptv"]["imx"] == 1000
    assert exp.pm.parameters["ptv"]["hp_flag"] is False
    assert exp.pm.parameters["criteria"]["X_lay"][0] == -200


def test_calib_params_binding(temp_experiment):
    """Test Calib_Params initialization and saving through CalHandler."""
    exp = temp_experiment

    # Create Calib_Params
    calib_params = Calib_Params(exp)

    # Verify attributes initialized correctly
    assert calib_params.h_image_size == 1280
    assert calib_params.v_image_size == 1024
    assert calib_params.dumbbell_eps == 0.1
    assert calib_params.dumbbell_niter == 100

    # Modify values
    calib_params.h_image_size = 1920
    calib_params.dumbbell_eps = 0.05

    # Trigger CalHandler.closed simulating clicking OK
    handler = CalHandler()
    info = DummyInfo(calib_params)
    handler.closed(info, is_ok=True)

    # Verify changes wrote back to the ParameterManager
    assert exp.pm.parameters["ptv"]["imx"] == 1920
    assert exp.pm.parameters["dumbbell"]["dumbbell_eps"] == 0.05


def test_tracking_params_binding(temp_experiment):
    """Test Tracking_Params initialization and saving through TrackHandler."""
    exp = temp_experiment

    # Create Tracking_Params
    track_params = Tracking_Params(exp)

    # Verify attributes initialized correctly
    assert track_params.dvxmin == -10.0
    assert track_params.dvxmax == 10.0
    assert track_params.angle == 45.0

    # Modify values
    track_params.dvxmin = -5.0
    track_params.angle = 30.0

    # Trigger TrackHandler.closed simulating clicking OK
    handler = TrackHandler()
    info = DummyInfo(track_params)
    handler.closed(info, is_ok=True)

    # Verify changes wrote back to the ParameterManager
    assert exp.pm.parameters["track"]["dvxmin"] == -5.0
    assert exp.pm.parameters["track"]["angle"] == 30.0


# --- Step 3: descriptive KeyError on missing sections ---


def test_tracking_params_missing_section_descriptive_error(temp_experiment):
    """Step 3: missing 'track' section gives descriptive KeyError, not bare key."""
    exp = temp_experiment
    del exp.pm.parameters["track"]
    with pytest.raises(KeyError, match="not found"):
        Tracking_Params(exp)


def test_main_params_missing_section_descriptive_error(temp_experiment):
    """Step 3: missing 'ptv' section gives descriptive KeyError."""
    exp = temp_experiment
    del exp.pm.parameters["ptv"]
    with pytest.raises(KeyError, match="not found"):
        Main_Params(exp)


# --- Step 4: float-valued int fields must not raise TraitError ---


def test_main_params_accepts_float_int_values(temp_experiment):
    """Step 4: Int traits must accept float-valued YAML ints without TraitError."""
    exp = temp_experiment
    exp.pm.parameters["ptv"]["imx"] = 1280.0
    exp.pm.parameters["ptv"]["imy"] = 1024.0
    exp.pm.parameters["ptv"]["chfield"] = 0.0
    exp.pm.parameters["targ_rec"]["nnmin"] = 5.0
    exp.pm.parameters["sequence"]["first"] = 1.0
    exp.pm.parameters["sequence"]["last"] = 100.0
    mp = Main_Params(exp)
    assert mp.imx == 1280
    assert mp.imy == 1024
    assert mp.Seq_First == 1
    assert mp.Seq_Last == 100


def test_calib_params_accepts_float_int_values(temp_experiment):
    """Step 4: Calib_Params Int traits must accept float-valued YAML ints."""
    exp = temp_experiment
    exp.pm.parameters["ptv"]["imx"] = 1280.0
    exp.pm.parameters["detect_plate"]["gvth_1"] = 100.0
    exp.pm.parameters["dumbbell"]["dumbbell_step"] = 10.0
    exp.pm.parameters["dumbbell"]["dumbbell_niter"] = 100.0
    cp = Calib_Params(exp)
    assert cp.h_image_size == 1280
    assert cp.grey_value_treshold_1 == 100
    assert cp.dumbbell_step == 10
    assert cp.dumbbell_niter == 100

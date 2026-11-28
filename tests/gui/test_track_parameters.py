from pathlib import Path

import pytest

from openptv2.gui.experiment import Experiment
from openptv2.gui.parameter_manager import ParameterManager

TEST_DATA = Path(__file__).parent.parent.parent / "test_data"


def get_track_params_from_yaml(yaml_path):
    pm = ParameterManager()
    experiment = Experiment()
    experiment.populate_runs(Path(yaml_path).parent)
    pm.from_yaml(yaml_path)
    return pm.parameters.get(
        "track"
    )  # Use direct dict access if get_parameter is not available


def get_track_params_from_dir(par_dir):
    pm = ParameterManager()
    pm.from_directory(par_dir)
    return pm.parameters.get("track")


REQUIRED_TRACK_PARAMS = [
    "dvxmin",
    "dvxmax",
    "dvymin",
    "dvymax",
    "dvzmin",
    "dvzmax",
    "angle",
    "dacc",
    "flagNewParticles",
]


@pytest.mark.parametrize(
    "yaml_path",
    [
        TEST_DATA / "test_cavity" / "parameters_Run1.yaml",
        # Add more YAML files as needed
    ],
)
def test_track_params_in_yaml(yaml_path):
    track = get_track_params_from_yaml(yaml_path)
    assert track is not None, f"No 'track' section in {yaml_path}"
    for key in REQUIRED_TRACK_PARAMS:
        assert key in track, f"Missing '{key}' in 'track' section of {yaml_path}"
        assert track[key] is not None, (
            f"'{key}' is None in 'track' section of {yaml_path}"
        )


@pytest.mark.parametrize(
    "par_dir",
    [
        TEST_DATA / "test_cavity" / "parameters",
        # Add more parameter directories as needed
    ],
)
def test_track_params_in_par_dir(par_dir):
    par_dir_path = Path(par_dir)
    experiment = Experiment()
    experiment.populate_runs(par_dir_path.parent)
    track = get_track_params_from_dir(par_dir)
    assert track is not None, f"No 'track' section in {par_dir}"
    for key in REQUIRED_TRACK_PARAMS:
        assert key in track, f"Missing '{key}' in 'track' section of {par_dir}"
        assert track[key] is not None, (
            f"'{key}' is None in 'track' section of {par_dir}"
        )


def test_tracking_params_gui_postprocess(tmp_path):
    from openptv2.gui.experiment import Experiment
    from openptv2.gui.parameter_gui import Tracking_Params
    from openptv2.gui.parameter_manager import ParameterManager

    pm = ParameterManager()
    pm.parameters = {
        "num_cams": 1,
        "track": {
            "dvxmin": -10.0,
            "dvxmax": 10.0,
            "dvymin": -10.0,
            "dvymax": 10.0,
            "dvzmin": -10.0,
            "dvzmax": 10.0,
            "angle": 120.0,
            "dacc": 5.0,
            "flagNewParticles": True,
            "track_mode": 0,
            "postprocess": True,
        },
    }
    exp = Experiment(pm=pm)
    tp = Tracking_Params(experiment=exp)
    assert tp.postprocess is True
    assert tp.vfc_mode == "off"
    assert tp.vfc_radius == 10.0
    assert tp.vfc_gate == 0.0


def test_tracking_params_gui_vfc_roundtrip(tmp_path):
    from types import SimpleNamespace

    from openptv2.gui.experiment import Experiment
    from openptv2.gui.parameter_gui import TrackHandler, Tracking_Params
    from openptv2.gui.parameter_manager import ParameterManager

    pm = ParameterManager()
    pm.parameters = {
        "num_cams": 4,
        "track": {
            "dvxmin": -5.0,
            "dvxmax": 5.0,
            "dvymin": -5.0,
            "dvymax": 5.0,
            "dvzmin": -5.0,
            "dvzmax": 5.0,
            "angle": 120.0,
            "dacc": 2.0,
            "flagNewParticles": True,
            "track_mode": 0,
            "postprocess": True,
            "vfc_mode": "both",
            "vfc_radius": 12.0,
            "vfc_min_neighbors": 4,
            "vfc_gate": 0.4,
            "vfc_post_min_tol": 0.4,
            "vfc_bootstrap": True,
            "vfc_affine": True,
        },
        "plugins": {"selected_tracking": "two_phase"},
    }
    exp = Experiment(pm=pm)
    exp.save_active = lambda: None
    tp = Tracking_Params(experiment=exp)
    assert tp.vfc_mode == "both"
    assert tp.vfc_radius == 12.0
    assert tp.vfc_min_neighbors == 4
    assert tp.vfc_gate == 0.4
    assert tp.vfc_post_min_tol == 0.4
    assert tp.vfc_bootstrap is True
    assert tp.vfc_affine is True

    # Modify in dialog and click OK (TrackHandler.closed)
    tp.vfc_mode = "post"
    tp.vfc_gate = 0.0  # 0.0 -> None in YAML
    tp.vfc_radius = 15.0
    TrackHandler().closed(SimpleNamespace(object=tp), is_ok=True)
    saved = exp.pm.parameters["track"]
    assert saved["vfc_mode"] == "post"
    assert saved["vfc_gate"] is None
    assert saved["vfc_radius"] == 15.0
    assert saved["vfc_bootstrap"] is True
    assert saved["vfc_affine"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Tests for Pydantic v2 parameter models."""
import pytest
from pydantic import ValidationError

from openptv2.gui.parameter_models import (
    AllParams,
    CalOriParams,
    PtvParams,
)
from openptv2.gui.parameter_manager import ParameterManager


# --- CalOriParams ---

def test_cal_ori_rejects_empty_img_cal_name():
    with pytest.raises(ValidationError, match="img_cal_name must not be empty"):
        CalOriParams(img_cal_name=[], img_ori=[])


def test_cal_ori_rejects_mismatched_lengths():
    with pytest.raises(ValidationError, match="equal length"):
        CalOriParams(img_cal_name=["a.tif"], img_ori=["a.ori", "b.ori"])


def test_cal_ori_valid():
    p = CalOriParams(img_cal_name=["a.tif"], img_ori=["a.tif.ori"])
    assert p.img_cal_name == ["a.tif"]


# --- AllParams ---

def _minimal_all_params(num_cams=2):
    names = [f"cam{i}.tif" for i in range(num_cams)]
    return AllParams(
        num_cams=num_cams,
        cal_ori=CalOriParams(img_cal_name=names, img_ori=[f"{n}.ori" for n in names]),
        ptv=PtvParams(img_name=names),
    )


def test_all_params_valid():
    p = _minimal_all_params(2)
    assert p.num_cams == 2


def test_all_params_cam_list_mismatch():
    with pytest.raises(ValidationError, match="num_cams"):
        AllParams(
            num_cams=2,
            cal_ori=CalOriParams(img_cal_name=["a.tif", "b.tif", "c.tif"],
                                 img_ori=["a.ori", "b.ori", "c.ori"]),
            ptv=PtvParams(img_name=["a.tif", "b.tif"]),
        )


# --- ParameterManager.validated() ---

def test_validated_round_trips_yaml(tmp_path):
    pm = ParameterManager()
    pm.from_yaml("test_data/test_cavity/parameters_Run1.yaml")
    v = pm.validated()
    assert v.num_cams == 4
    assert len(v.cal_ori.img_cal_name) == 4
    assert v.cal_ori.img_cal_name[0] == "cal/cam1.tif"


def test_validated_round_trips_directory():
    pm = ParameterManager()
    pm.from_directory("test_data/test_cavity/parameters")
    v = pm.validated()
    assert v.num_cams == 4
    assert len(v.cal_ori.img_ori) == 4


def test_validated_rejects_empty_cal_ori(tmp_path):
    """Simulate the historical bad-YAML bug: empty img_cal_name should fail validated()."""
    import yaml

    bad = {
        "num_cams": 4,
        "cal_ori": {
            "img_cal_name": [],
            "img_ori": [],
            "fixp_name": "cal/target.txt",
            "chfield": 0,
            "pair_flag": False,
            "tiff_flag": True,
            "cal_splitter": False,
        },
        "ptv": {
            "img_name": ["cam1", "cam2", "cam3", "cam4"],
            "img_cal": ["cal/cam1.tif"] * 4,
            "imx": 1280, "imy": 1024,
            "mmp_d": 6.0, "mmp_n1": 1.0, "mmp_n2": 1.33, "mmp_n3": 1.46,
            "pix_x": 0.012, "pix_y": 0.012,
            "tiff_flag": True, "splitter": False,
            "allcam_flag": False, "chfield": 0, "hp_flag": True,
        },
    }
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(yaml.safe_dump(bad))

    pm = ParameterManager()
    pm.from_yaml(yaml_file)  # loads without crash (warns only)

    with pytest.raises(ValidationError):
        pm.validated()

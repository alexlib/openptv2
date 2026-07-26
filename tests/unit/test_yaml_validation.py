"""Tests for YAML parameter validation, consistency checks, and legacy
.par-to-YAML translation.

Validates that OpenPTV2's YAML parameter files:
  1. Parse correctly into Pydantic models (AllParams).
  2. Detect missing or incomplete fields.
  3. Detect contradictory parameter ranges (e.g. dvxmin > dvxmax,
     first_frame > last_frame).
  4. Detect physically invalid parameters (e.g. non-positive pixel size,
     image size, num_cams).
  5. Support translating legacy .par directories into valid YAML parameter sets.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openptv2.gui.parameter_manager import ParameterManager
from openptv2.gui.parameter_models import AllParams

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_FIXTURES = [
    REPO_ROOT / "test_data" / "parameters.yaml",
    REPO_ROOT / "test_data" / "test_cavity" / "parameters_Run1.yaml",
    REPO_ROOT / "test_data" / "test_splitter" / "parameters_Run1.yaml",
]


@pytest.mark.parametrize("yaml_path", YAML_FIXTURES)
def test_valid_yaml_fixtures_load_and_validate(yaml_path: Path):
    """Ensure standard repository YAML fixtures validate without errors."""
    assert yaml_path.exists(), f"Fixture missing: {yaml_path}"
    data = yaml.safe_load(yaml_path.read_text())
    params = AllParams.model_validate(data)
    assert params.num_cams > 0
    assert params.ptv.imx > 0
    assert params.ptv.imy > 0


def test_missing_num_cams_raises_validation_error():
    """Missing or invalid top-level num_cams should fail validation."""
    data = {
        "cal_ori": {"img_cal_name": ["cam1"], "img_ori": ["cam1.ori"]},
        "ptv": {"img_name": ["cam1"]},
    }
    with pytest.raises(ValidationError, match="num_cams"):
        AllParams.model_validate(data)


def test_mismatched_camera_counts_raises_error():
    """Camera list lengths that do not match num_cams should be rejected."""
    data = {
        "num_cams": 4,
        "cal_ori": {
            "img_cal_name": ["cam1", "cam2"],  # only 2 names for 4 cameras
            "img_ori": ["cam1.ori", "cam2.ori"],
        },
        "ptv": {
            "img_name": ["c1", "c2", "c3", "c4"],
        },
    }
    with pytest.raises(ValidationError, match="img_cal_name length"):
        AllParams.model_validate(data)


def test_non_positive_num_cams_rejected():
    """num_cams <= 0 should raise a validation error."""
    base_data = yaml.safe_load(YAML_FIXTURES[0].read_text())
    base_data["num_cams"] = 0

    with pytest.raises(ValidationError, match="num_cams must be positive"):
        AllParams.model_validate(base_data)


def test_contradictory_sequence_frames_rejected():
    """first frame > last frame should raise a validation error."""
    base_data = yaml.safe_load(YAML_FIXTURES[0].read_text())
    base_data["sequence"]["first"] = 10010
    base_data["sequence"]["last"] = 10000  # contradictory: first > last

    with pytest.raises(ValidationError, match="sequence first frame"):
        AllParams.model_validate(base_data)


def test_non_positive_pixel_size_rejected():
    """pix_x <= 0 or pix_y <= 0 should raise a validation error."""
    base_data = yaml.safe_load(YAML_FIXTURES[0].read_text())
    base_data["ptv"]["pix_x"] = -0.012

    with pytest.raises(ValidationError, match="pixel sizes"):
        AllParams.model_validate(base_data)


def test_non_positive_refractive_indices_rejected():
    """Refractive index <= 0 should raise a validation error."""
    base_data = yaml.safe_load(YAML_FIXTURES[0].read_text())
    base_data["ptv"]["mmp_n1"] = 0.0

    with pytest.raises(ValidationError, match="refractive indices"):
        AllParams.model_validate(base_data)


def test_legacy_par_directory_to_yaml_translation():
    """Verify loading legacy .par directory and dumping to valid YAML."""
    par_dir = REPO_ROOT / "test_data" / "parameters"
    if not par_dir.exists():
        pytest.skip("Legacy parameters directory not present")

    pm = ParameterManager()
    pm.from_directory(par_dir)
    assert pm.num_cams > 0

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pm.to_yaml(tmp_path)
        assert tmp_path.exists()
        loaded = yaml.safe_load(tmp_path.read_text())
        model = AllParams.model_validate(loaded)
        assert model.num_cams == pm.num_cams
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

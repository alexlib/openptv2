"""Tests for Experiment.save_active() — the single save entry point."""
import yaml
import pytest
from pathlib import Path
from openptv2.gui.experiment import Experiment, Paramset
from openptv2.gui.parameter_manager import ParameterManager


def test_save_active_writes_to_active_yaml(tmp_path):
    pm = ParameterManager()
    pm.from_yaml("test_data/test_cavity/parameters_Run1.yaml")
    exp = Experiment(pm=pm)
    yaml_out = tmp_path / "params.yaml"
    ps = Paramset(name="Run1", yaml_path=yaml_out)
    exp.paramsets = [ps]
    exp.active_params = ps

    exp.save_active()

    data = yaml.safe_load(yaml_out.read_text())
    assert data["num_cams"] == 4


def test_save_active_no_op_when_no_active(tmp_path):
    pm = ParameterManager()
    pm.from_yaml("test_data/test_cavity/parameters_Run1.yaml")
    exp = Experiment(pm=pm)
    exp.active_params = None
    # Must not raise
    exp.save_active()


def test_save_parameters_removed():
    """save_parameters was the old API — must be gone."""
    assert not hasattr(Experiment, 'save_parameters'), \
        "save_parameters must be removed; use save_active()"


def test_rename_paramset(tmp_path):
    """test_rename_paramset renames the YAML file and updates Paramset metadata."""
    old_yaml = tmp_path / "parameters_Run1.yaml"
    old_yaml.write_text("num_cams: 4\n")

    pm = ParameterManager()
    exp = Experiment(pm=pm)
    ps = Paramset(name="Run1", yaml_path=old_yaml)
    exp.paramsets = [ps]

    exp.rename_paramset("Run1", "Run2")

    new_yaml = tmp_path / "parameters_Run2.yaml"
    assert not old_yaml.exists()
    assert new_yaml.exists()
    assert ps.name == "Run2"
    assert ps.yaml_path == new_yaml


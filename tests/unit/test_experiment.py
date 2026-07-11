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

"""Editing one parameter set must not change any other set's YAML.

Drives the real dialogs (Main / Calibration / Tracking parameters) and their
real OK handlers through the tree menu's _open_param_dialog, for the active
set and for inactive ones, before and after switching the active set. Only
edit_traits (the window itself) is replaced, by "change a value, press OK".
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("traitsui")

from openptv2.gui.experiment import Experiment  # noqa: E402
from openptv2.gui.parameter_gui import (  # noqa: E402
    CalHandler,
    Calib_Params,
    Main_Params,
    ParamHandler,
    TrackHandler,
    Tracking_Params,
)
from openptv2.gui.pyptv_gui import TreeMenuHandler  # noqa: E402
from tests._support import find_test_data_root  # noqa: E402

CAVITY = find_test_data_root() / "test_cavity"
RUNS = ("Run1", "Run2", "Run3")


@pytest.fixture
def experiment(tmp_path):
    work = tmp_path / "exp"
    work.mkdir()
    shutil.copytree(CAVITY / "cal", work / "cal")
    base = yaml.safe_load((CAVITY / "parameters_Run1.yaml").read_text())
    for i, name in enumerate(RUNS):
        p = yaml.safe_load(yaml.safe_dump(base))
        p["ptv"]["imx"] = 1280 + i
        p["track"]["dacc"] = 0.5 + i
        (work / f"parameters_{name}.yaml").write_text(yaml.safe_dump(p))
    exp = Experiment()
    exp.populate_runs(work, active_yaml=work / "parameters_Run1.yaml")
    return exp, work


def _values(work: Path) -> dict:
    out = {}
    for name in RUNS:
        p = yaml.safe_load((work / f"parameters_{name}.yaml").read_text())
        out[name] = (p["ptv"]["imx"], p["track"]["dacc"])
    return out


DIALOGS = {
    "main": (Main_Params, ParamHandler, "Main_Params_View", "imx", 100),
    "cal": (Calib_Params, CalHandler, "Calib_Params_View", None, 0),
    "track": (Tracking_Params, TrackHandler, "Tracking_Params_View", "dacc", 10.0),
}


def _press_ok(monkeypatch, exp, name, kind):
    cls, handler, view, attr, delta = DIALOGS[kind]
    shown = {}

    def fake_edit(self, view=None, kind=None):
        if attr:
            shown["value"] = getattr(self, attr)
            setattr(self, attr, getattr(self, attr) + delta)
        handler().closed(SimpleNamespace(object=self), is_ok=True)
        return True

    monkeypatch.setattr(cls, "edit_traits", fake_edit)
    paramset = next(p for p in exp.paramsets if p.name == name)
    editor = SimpleNamespace(get_parent=lambda obj: exp)
    TreeMenuHandler()._open_param_dialog(editor, paramset, cls, view, kind)
    return shown.get("value")


@pytest.mark.parametrize("kind", ["main", "cal", "track"])
@pytest.mark.parametrize("active", RUNS)
@pytest.mark.parametrize("target", RUNS)
def test_edit_changes_only_its_own_set(experiment, monkeypatch, kind, active, target):
    exp, work = experiment
    editor = SimpleNamespace(get_parent=lambda obj: exp)
    TreeMenuHandler().set_active(
        editor, next(p for p in exp.paramsets if p.name == active)
    )
    before = _values(work)

    shown = _press_ok(monkeypatch, exp, target, kind)

    after = _values(work)
    changed = {name for name in RUNS if after[name] != before[name]}
    assert changed <= {target}, f"editing {target} changed {changed - {target}}"
    if DIALOGS[kind][3] == "imx":
        assert shown == before[target][0], "dialog showed another set's values"
    if DIALOGS[kind][3] == "dacc":
        assert shown == before[target][1], "dialog showed another set's values"
    # the active set's in-memory parameters still belong to the active set
    assert exp.active_params.name == active
    assert exp.pm.parameters["ptv"]["imx"] == after[active][0]

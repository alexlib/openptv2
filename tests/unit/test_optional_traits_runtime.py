import builtins
import importlib
import sys
import tomllib
from pathlib import Path


def test_experiment_imports_without_traits(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "traits.api" or name.startswith("traits."):
            raise ModuleNotFoundError("No module named 'traits'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "openptv2.gui.experiment", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    experiment = importlib.import_module("openptv2.gui.experiment")

    paramset = experiment.Paramset(name="Run1", yaml_path=Path("parameters_Run1.yaml"))
    exp = experiment.Experiment()

    assert paramset.name == "Run1"
    assert paramset.yaml_path == Path("parameters_Run1.yaml")
    assert exp.paramsets == []
    assert exp.active_params is None


def test_traits_is_gui_only_dependency():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = data["project"]["dependencies"]
    gui_dependencies = data["project"]["optional-dependencies"]["gui"]

    assert "traits>=6.4.0" not in dependencies
    assert "traits>=6.4.0" in gui_dependencies

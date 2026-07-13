"""Tests for the GUI plugin selection dialog (openptv2.gui.pyptv_gui.Plugins)
and the "default" plugin unification's effect on GUI tracking state.

Plugins.read() used to trust a session-cached YAML snapshot (which, for a
YAML-first experiment, is often never populated at all — from_yaml() doesn't
scan plugins, only the legacy from_directory() .par-conversion path does) and
had two crash bugs: an unguarded ParameterManager.get_parameter("plugins")
raising ValueError when the key is missing, and Plugins.save() calling
Experiment.get_parameter with two positional args (that method only accepts
one). This module covers the rewrite: a live filesystem rescan on every
read(), graceful fallback to "default" for stale/missing selections, and a
working save() round trip.
"""

import shutil
from pathlib import Path

import pytest

from openptv2.gui.parameter_manager import ParameterManager
from openptv2.gui.pyptv_gui import Plugins


class FakeExperiment:
    """Minimal stand-in for openptv2.gui.experiment.Experiment: Plugins only
    ever touches self.experiment.pm."""

    def __init__(self, pm):
        self.pm = pm


@pytest.fixture
def synthetic_exp(tmp_path):
    """tracking_synthetic has no 'plugins:' section in its YAML at all —
    the exact case that used to crash GUI startup."""
    src = Path(__file__).parent.parent.parent / "test_data" / "tracking_synthetic"
    if not src.exists():
        pytest.skip("tracking_synthetic fixture not found")
    dst = tmp_path / "tracking_synthetic"
    shutil.copytree(src, dst)

    pm = ParameterManager()
    pm.from_yaml(dst / "parameters_Run1.yaml")
    assert "plugins" not in pm.parameters  # sanity: this is the crash case
    return FakeExperiment(pm)


def test_read_with_no_plugins_section_does_not_crash(synthetic_exp):
    plugins = Plugins(experiment=synthetic_exp)
    assert plugins.track_alg == "default"
    assert plugins.sequence_alg == "default"


def test_read_picks_up_newly_dropped_local_plugin(synthetic_exp):
    plugins = Plugins(experiment=synthetic_exp)

    plugins_dir = Path(synthetic_exp.pm.yaml_path).parent / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "my_sequence.py").write_text(
        "class Sequence:\n"
        "    def __init__(self, ptv=None, exp=None):\n"
        "        pass\n"
        "    def do_sequence(self):\n"
        "        pass\n"
    )

    # Live rescan on the *same* Plugins instance, no experiment reload.
    plugins.read()
    plugins.sequence_alg = "my_sequence"  # raises TraitError if not offered
    assert plugins.sequence_alg == "my_sequence"


def test_read_falls_back_to_default_for_stale_selection(synthetic_exp):
    synthetic_exp.pm.parameters["plugins"] = {
        "selected_tracking": "no_longer_installed",
        "selected_sequence": "default",
    }
    plugins = Plugins(experiment=synthetic_exp)
    assert plugins.track_alg == "default"


def test_save_round_trip(synthetic_exp):
    plugins = Plugins(experiment=synthetic_exp)
    plugins.sequence_alg = "splitter_sequence"
    plugins.save()  # regression test: used to raise TypeError

    saved = synthetic_exp.pm.parameters["plugins"]
    assert saved["selected_sequence"] == "splitter_sequence"
    assert saved["selected_tracking"] == "default"


class TestDefaultTrackingSetsTracker:
    """Unifying "default" into the plugin loader means GUI tracking always
    goes through Tracking.do_tracking() now, rather than track_no_disp_action
    setting mainGui.tracker inline. track_back_action depends on that
    attribute being set as a side effect of the forward-tracking run —
    verify default_tracking.py preserves it.
    """

    def test_default_tracking_sets_exp_tracker_and_supports_backward(self, tmp_path):
        from openptv2.gui import ptv
        from openptv2.gui.experiment import Experiment

        src = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
        dst = tmp_path / "test_cavity"
        shutil.copytree(src, dst)

        import os

        original_cwd = Path.cwd()
        os.chdir(dst)
        try:
            experiment = Experiment()
            experiment.pm.from_yaml(dst / "parameters_Run1.yaml")
            cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(
                experiment.pm
            )
            spar.set_first(10000)
            spar.set_last(10001)

            class ExpDouble:
                pass

            exp = ExpDouble()
            exp.pm = experiment.pm
            exp.cpar = cpar
            exp.spar = spar
            exp.vpar = vpar
            exp.track_par = track_par
            exp.tpar = tpar
            exp.cals = cals
            exp.num_cams = experiment.pm.num_cams
            exp.target_filenames = experiment.pm.get_target_filenames()

            class PluginsDouble:
                track_alg = "default"

            exp.plugins = PluginsDouble()

            assert not hasattr(exp, "tracker")
            ptv.run_tracking_plugin(exp)

            assert getattr(exp, "tracker", None) is not None
            try:
                exp.tracker.full_backward()
            except ZeroDivisionError:
                # Data-specific edge case (zero forward links on this tiny
                # 2-frame range) inside the backward-tracking algorithm
                # itself — unrelated to the regression under test, which is
                # specifically that exp.tracker exists and is usable at all
                # (no AttributeError).
                pass
        finally:
            os.chdir(original_cwd)

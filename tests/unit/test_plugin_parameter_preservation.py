"""Tests verifying that tracking plugins strictly preserve user-configured kinematic search parameters

and that built-in plugins (fast_3d, standard_forward, full_multipass) resolve correctly.
"""

from unittest.mock import MagicMock

import pytest

from openptv2.plugins.loader import (
    BUILTIN_TRACKING_PLUGINS,
    resolve_plugin_module,
)
from openptv2.tracking_presets import apply_preset


def test_builtin_plugins_registration():
    """Verify that all standard tracking pipelines are registered in BUILTIN_TRACKING_PLUGINS."""
    expected_plugins = ["default", "full_multipass", "standard_forward", "two_directional", "fast_3d", "splitter_tracking"]
    for plugin_name in expected_plugins:
        assert plugin_name in BUILTIN_TRACKING_PLUGINS
        mod = resolve_plugin_module(plugin_name, BUILTIN_TRACKING_PLUGINS)
        assert mod is not None
        assert hasattr(mod, "Tracking")


@pytest.mark.parametrize("preset_name", ["fast_3d", "standard_forward", "full_multipass", "splitter_tracking"])
def test_apply_preset_preserves_kinematic_bounds(preset_name):
    """Verify that applying a preset or plugin choice does not touch or overwrite kinematic search limits."""
    custom_track = {
        "dvxmin": -42.5,
        "dvxmax": 88.0,
        "dvymin": -12.3,
        "dvymax": 34.5,
        "dvzmin": -99.9,
        "dvzmax": 100.1,
        "angle": 135.0,
        "dacc": 9.5,
    }
    plugins_cfg = {}

    updated_track, updated_plugins = apply_preset(preset_name, custom_track, plugins_cfg)

    # Check that every single custom kinematic limit is preserved without modification
    assert updated_track["dvxmin"] == -42.5
    assert updated_track["dvxmax"] == 88.0
    assert updated_track["dvymin"] == -12.3
    assert updated_track["dvymax"] == 34.5
    assert updated_track["dvzmin"] == -99.9
    assert updated_track["dvzmax"] == 100.1
    assert updated_track["angle"] == 135.0
    assert updated_track["dacc"] == 9.5

    # Check that selected_tracking is set correctly in plugins section
    assert updated_plugins["selected_tracking"] == preset_name


def test_custom_plugin_name_preserves_kinematic_bounds():
    """Verify that custom plugin names are applied cleanly without touching kinematic limits."""
    custom_track = {
        "dvxmin": -5.0,
        "dvxmax": 5.0,
        "dvymin": -5.0,
        "dvymax": 5.0,
        "dvzmin": -5.0,
        "dvzmax": 5.0,
        "angle": 90.0,
        "dacc": 2.0,
    }
    plugins_cfg = {}

    updated_track, updated_plugins = apply_preset(
        "custom_plugin", custom_track, plugins_cfg, custom_plugin_name="my_research_plugin"
    )

    assert updated_plugins["selected_tracking"] == "my_research_plugin"
    assert updated_track["dvxmin"] == -5.0
    assert updated_track["dvxmax"] == 5.0


def test_plugin_execution_preserves_exp_parameters():
    """Verify that running default_tracking plugin initializes py_trackcorr_init directly with exp

    and never mutates or overwrites exp.pm.parameters['track'] kinematic bounds.
    """
    mock_exp = MagicMock()
    mock_tracker = MagicMock()
    mock_ptv = MagicMock()
    mock_ptv.py_trackcorr_init.return_value = mock_tracker

    initial_track_params = {
        "dvxmin": -15.0,
        "dvxmax": 15.0,
        "dvymin": -15.0,
        "dvymax": 15.0,
        "dvzmin": -15.0,
        "dvzmax": 15.0,
        "angle": 110.0,
        "dacc": 4.0,
    }
    mock_exp.pm.parameters = {
        "track": dict(initial_track_params),
        "plugins": {"selected_tracking": "fast_3d"},
    }

    # Load and execute fast_3d plugin
    mod = resolve_plugin_module("fast_3d", BUILTIN_TRACKING_PLUGINS)
    plugin = mod.Tracking(ptv=mock_ptv, exp=mock_exp)
    plugin.do_tracking()

    # 1. py_trackcorr_init must have been called with mock_exp
    mock_ptv.py_trackcorr_init.assert_called_once_with(mock_exp)

    # 2. full_forward_3d must have been called for fast_3d
    mock_tracker.full_forward_3d.assert_called_once()

    # 3. Kinematic bounds in mock_exp must be completely untouched
    current_track_params = mock_exp.pm.parameters["track"]
    for key, val in initial_track_params.items():
        assert current_track_params[key] == val

# ruff: noqa: E501
"""Unit tests for multi-density synthetic tracking and tracker selection defaults."""

import pytest

from openptv2.plugins.loader import BUILTIN_TRACKING_PLUGINS, LEGACY_ALIASES
from openptv2.tracking_presets import PRESET_CHOICES, TrackingPreset, infer_preset


def test_default_tracker_is_fast_3d():
    """Verify fast_3d is the primary default tracking plugin."""
    assert BUILTIN_TRACKING_PLUGINS["default"] == "openptv2.plugins.default_tracking"
    assert BUILTIN_TRACKING_PLUGINS["fast"] == "openptv2.plugins.default_tracking"
    assert BUILTIN_TRACKING_PLUGINS["fast_3d"] == "openptv2.plugins.default_tracking"


def test_fast_alias_mapping():
    """Verify fast is registered as an alias to fast_3d."""
    assert LEGACY_ALIASES["fast"] == "fast_3d"
    assert BUILTIN_TRACKING_PLUGINS["fast"] == "openptv2.plugins.default_tracking"
    assert BUILTIN_TRACKING_PLUGINS["fast_3d"] == "openptv2.plugins.default_tracking"


def test_preset_order():
    """Verify preset choice list starts with fast_3d."""
    preset_keys = [key for key, _ in PRESET_CHOICES]
    assert preset_keys[0] == "fast_3d"
    assert preset_keys[1] == "fast"


def test_infer_preset_default_fallback():
    """Verify infer_preset defaults to fast_3d."""
    res = infer_preset({}, {})
    assert res == TrackingPreset.FAST_3D.value

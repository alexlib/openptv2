# ruff: noqa: E501
"""Unit tests for multi-density synthetic tracking and tracker selection defaults."""

import pytest

from openptv2.plugins.loader import BUILTIN_TRACKING_PLUGINS, LEGACY_ALIASES
from openptv2.tracking_presets import PRESET_CHOICES, TrackingPreset, infer_preset


def test_default_tracker_is_priority_segment_3d():
    """Verify priority_segment_3d is the primary default tracking plugin.

    Since "feat: Make OpenPTV Fast 3D the default tracker" (31c5ad5), the
    compiled Cython3DTracker plugin serves these aliases directly --
    default_tracking.py is orphaned (no registry entry points to it).
    """
    assert BUILTIN_TRACKING_PLUGINS["default"] == "openptv2.plugins.cython_3d_tracking"
    assert BUILTIN_TRACKING_PLUGINS["fast"] == "openptv2.plugins.cython_3d_tracking"
    assert BUILTIN_TRACKING_PLUGINS["fast_3d"] == "openptv2.plugins.cython_3d_tracking"
    assert BUILTIN_TRACKING_PLUGINS["priority_segment_3d"] == "openptv2.plugins.cython_3d_tracking"


def test_fast_alias_mapping():
    """Verify fast is registered as an alias to priority_segment_3d."""
    assert LEGACY_ALIASES["fast"] == "priority_segment_3d"
    assert BUILTIN_TRACKING_PLUGINS["fast"] == "openptv2.plugins.cython_3d_tracking"
    assert BUILTIN_TRACKING_PLUGINS["fast_3d"] == "openptv2.plugins.cython_3d_tracking"


def test_preset_order():
    """Verify preset choice list starts with priority_segment_3d."""
    preset_keys = [key for key, _ in PRESET_CHOICES]
    assert preset_keys[0] == "priority_segment_3d"


def test_infer_preset_default_fallback():
    """Verify infer_preset defaults to priority_segment_3d."""
    res = infer_preset({}, {})
    assert res == TrackingPreset.PRIORITY_SEGMENT_3D.value

import pytest
from openptv2.tracking_presets import (
    TrackingPreset,
    infer_preset,
    apply_preset,
    PRESET_CONFIGS,
)


def test_infer_preset_explicit():
    track_cfg = {"preset": "fast_3d", "track_mode": 1}
    assert infer_preset(track_cfg) == "fast_3d"


def test_infer_preset_from_custom_plugin():
    track_cfg = {"track_mode": 0, "postprocess": True}
    plugins_cfg = {"selected_tracking": "my_custom_plugin"}
    assert infer_preset(track_cfg, plugins_cfg) == TrackingPreset.CUSTOM_PLUGIN.value


def test_infer_preset_from_track_mode():
    track_cfg = {"track_mode": 1, "flagNewParticles": False, "postprocess": False}
    assert infer_preset(track_cfg) == TrackingPreset.FAST_3D.value


def test_infer_preset_from_postprocess():
    track_cfg = {"track_mode": 0, "flagNewParticles": True, "postprocess": True}
    assert infer_preset(track_cfg) == TrackingPreset.FULL_MULTIPASS.value


def test_infer_preset_standard_forward():
    track_cfg = {"track_mode": 0, "flagNewParticles": True, "postprocess": False}
    assert infer_preset(track_cfg) == TrackingPreset.STANDARD_FORWARD.value


def test_apply_preset_full_multipass():
    track_in = {"dvxmin": -10.0, "dvxmax": 10.0}
    t_out, p_out = apply_preset("full_multipass", track_in)

    assert t_out["preset"] == "full_multipass"
    assert t_out["track_mode"] == 0
    assert t_out["flagNewParticles"] is True
    assert t_out["postprocess"] is True
    assert t_out["dvxmin"] == -10.0
    assert p_out["selected_tracking"] == "full_multipass"


def test_apply_preset_fast_3d():
    track_in = {"dvxmin": -5.0}
    t_out, p_out = apply_preset("fast_3d", track_in)

    assert t_out["preset"] == "fast_3d"
    assert t_out["track_mode"] == 1
    assert t_out["flagNewParticles"] is False
    assert t_out["postprocess"] is False
    assert t_out["dvxmin"] == -5.0
    assert p_out["selected_tracking"] == "fast_3d"

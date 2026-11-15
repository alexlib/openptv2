"""Unit tests for TrackingDecision (Milestone F / F0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openptv2.storage.run_store import RunStore
from openptv2.tracking_decision import DECISION_FILENAME, TrackingDecision


def _sample_decision() -> TrackingDecision:
    return TrackingDecision(
        resolved_config={
            "v_max": 12.5,
            "use_2d": True,
            "confirm_link": True,
            "delta_t": 1,
        },
        regime_flags={"sparse": False, "dense": True, "fast_flow": True},
        conditioning_ratio=0.72,
        probe_metrics=[
            {"window": [0, 5], "reciprocity_rate": 0.94, "mean_track_len": 4.8},
            {"window": [10, 15], "reciprocity_rate": 0.91, "mean_track_len": 4.5},
        ],
        grid_search_table=[
            {"v_max": 10.0, "score": 0.85},
            {"v_max": 12.5, "score": 0.93},
        ],
        chosen_config={"v_max": 12.5, "score": 0.93},
        report_metrics={"reciprocity_rate": 0.92, "jump_step_rate": 0.01},
        git_version="abc1234",
        timestamp="2026-11-14T16:00:00Z",
        data_hashes={"targets": "sha256:deadbeef", "calibrations": "sha256:cafebabe"},
    )


def test_round_trip_json_and_dict() -> None:
    decision = _sample_decision()

    as_dict = decision.to_dict()
    assert TrackingDecision.from_dict(as_dict) == decision

    json_text = decision.to_json()
    parsed = json.loads(json_text)
    assert parsed["conditioning_ratio"] == pytest.approx(0.72)
    assert parsed["resolved_config"]["v_max"] == 12.5
    assert TrackingDecision.from_json(json_text) == decision


def test_save_and_load_directory_and_file_path(tmp_path: Path) -> None:
    decision = _sample_decision()

    # Directory path -> writes <dir>/tracking_decision.json
    dir_out = tmp_path / "run_dir"
    saved_path = decision.save(dir_out)
    assert saved_path == dir_out / DECISION_FILENAME
    assert saved_path.is_file()
    assert TrackingDecision.load(dir_out) == decision
    assert TrackingDecision.load(saved_path) == decision

    # Explicit .json file path
    explicit_path = tmp_path / "nested" / "custom_decision.json"
    returned_path = decision.save(explicit_path)
    assert returned_path == explicit_path
    assert explicit_path.is_file()
    assert TrackingDecision.load(explicit_path) == decision


def test_run_store_persistence(tmp_path: Path) -> None:
    decision = _sample_decision()
    store_path = tmp_path / "res" / "run.zarr"
    store = RunStore(store_path, mode="a")

    saved_path = decision.save(store)
    assert saved_path == store_path / DECISION_FILENAME
    assert saved_path.is_file()

    # Attribute persisted in RunStore meta
    assert "tracking_decision" in store.meta
    assert store.meta["tracking_decision"] == decision.to_dict()
    assert store.root["meta"].attrs["tracking_decision"] == decision.to_dict()

    # Reopen store in read-only mode and verify both attr and load()
    reopened = RunStore(store_path, mode="r")
    assert reopened.meta["tracking_decision"] == decision.to_dict()
    assert TrackingDecision.load(reopened) == decision

    # Even if the sidecar JSON is removed, load(store) falls back to meta attr
    saved_path.unlink()
    assert TrackingDecision.load(reopened) == decision


def test_load_missing_from_run_store_raises(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "empty.zarr", mode="a")
    with pytest.raises(FileNotFoundError):
        TrackingDecision.load(store)


def test_missing_and_default_fields() -> None:
    empty = TrackingDecision()
    assert empty.resolved_config == {}
    assert empty.regime_flags == {}
    assert empty.conditioning_ratio is None
    assert empty.probe_metrics == []
    assert empty.grid_search_table == []
    assert empty.chosen_config == {}
    assert empty.report_metrics == {}
    assert empty.git_version == ""
    assert empty.timestamp == ""
    assert empty.data_hashes == {}

    assert TrackingDecision.from_dict({}) == empty

    partial = TrackingDecision.from_dict(
        {
            "resolved_config": {"v_max": 5.0},
            "conditioning_ratio": 1,
        }
    )
    assert partial.resolved_config == {"v_max": 5.0}
    assert partial.regime_flags == {}
    assert partial.conditioning_ratio == 1.0
    assert isinstance(partial.conditioning_ratio, float)
    assert partial.probe_metrics == []
    assert partial.git_version == ""

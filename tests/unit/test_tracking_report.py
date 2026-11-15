"""Unit tests for post-run tracking self-consistency report (Milestone D1-D2)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE
from openptv2.storage.run_store import RunStore
from openptv2.tracking_decision import TrackingDecision
from openptv2.tracking_report import (
    compute_self_consistency_metrics,
    compute_trajectory_quality_scores,
    generate_report,
)


def _open_store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "run.zarr", mode="w")


def test_clean_smooth_tracks_report(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    # 4 smooth linear tracks over frames 0..9 (length 10 each),
    # with one track having a 1-frame bridged gap between frame 4 and frame 6.
    n_frames = 10
    for f in range(n_frames):
        if f == 5:
            # Particle 3 is missing at frame 5 (bridged 4 -> 6).
            idx_map = [0, 1, 2]
            n = 3
            prev = np.array([0, 1, 2], dtype=np.int32)
            nxt = np.array([0, 1, 2], dtype=np.int32)
        else:
            idx_map = [0, 1, 2, 3]
            n = 4
            prev = np.full(n, PREV_NONE, dtype=np.int32) if f == 0 else np.arange(n, dtype=np.int32)
            nxt = np.full(n, NEXT_NONE, dtype=np.int32) if f == n_frames - 1 else np.arange(n, dtype=np.int32)
            if f == 4:
                # Particle 3 bridges across frame 5 to index 3 in frame 6.
                nxt[3] = 3
            if f == 6:
                prev[3] = 3

        xyz = np.zeros((n, 3), dtype=np.float64)
        for row, track_id in enumerate(idx_map):
            xyz[row] = [float(f) * 1.5, float(track_id) * 5.0, 0.0]
        store.write_linkage(f, prev, nxt, xyz)

    buf = io.StringIO()
    metrics = generate_report(store, 0, 9, print_report=True, stream=buf)

    assert metrics["reciprocity_rate"] == pytest.approx(1.0)
    assert metrics["jump_step_rate"] == pytest.approx(0.0)
    assert metrics["estimated_ghost_rate"] == pytest.approx(0.0)
    assert metrics["anomaly_flags"] == []

    tls = metrics["track_length_stats"]
    assert tls["count"] == 4
    assert tls["median"] == pytest.approx(10.0)  # sorted: [9, 10, 10, 10]
    assert tls["max"] == 10
    assert tls["p10"] >= 9.0
    assert tls["p90"] == pytest.approx(10.0)

    gaps = metrics["gap_distribution"]
    assert gaps["gap_1"] == 1
    assert gaps["broken_forward"] == 0
    assert gaps["broken_backward"] == 0
    assert "Anomaly warnings:     none" in buf.getvalue()


def test_broken_kinky_ghost_tracks_trigger_anomalies(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    # Build 6 frames (0..5) with:
    # - 1 short kinky 3-frame track (frames 0..2, index 0) with a 180-deg reversal at frame 1
    # - 1 broken one-sided forward link at frame 0 (index 1 -> frame 1 index 1, where prev is -1)
    # - 4 isolated ghost particles per frame (indices 2..5)
    for f in range(6):
        n = 6
        prev = np.full(n, PREV_NONE, dtype=np.int32)
        nxt = np.full(n, NEXT_NONE, dtype=np.int32)
        xyz = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            xyz[i] = [float(f), float(i) * 10.0, 0.0]

        if f == 0:
            nxt[0] = 0
            nxt[1] = 1  # one-sided forward link (frame 1 index 1 has prev == -1)
        elif f == 1:
            prev[0] = 0
            nxt[0] = 0
            xyz[0] = [5.0, 0.0, 0.0]  # sharp forward jump then reversal back at f=2
        elif f == 2:
            prev[0] = 0
            xyz[0] = [0.0, 0.0, 0.0]

        store.write_linkage(f, prev, nxt, xyz)

    metrics = compute_self_consistency_metrics(store, 0, 5)

    # 3 forward links total (f=0: 0->0, 1->1; f=1: 0->0); 2 are reciprocal -> 2/3 < 0.7
    assert metrics["reciprocity_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["reciprocity_rate"] < 0.7

    # Single track of length 3 -> median < 4
    assert metrics["track_length_stats"]["count"] == 1
    assert metrics["track_length_stats"]["median"] == pytest.approx(3.0)

    # Single 2-step transition reverses direction -> jump_step_rate == 1.0 > 0.1
    assert metrics["jump_step_rate"] == pytest.approx(1.0)

    # Isolated particles:
    # f=0: 4 (indices 2..5); f=1: 5 (1..5); f=2: 5 (1..5); f=3,4,5: 6 each -> 32/36
    assert metrics["estimated_ghost_rate"] == pytest.approx(32.0 / 36.0)
    assert metrics["estimated_ghost_rate"] > 0.4

    assert metrics["gap_distribution"]["broken_forward"] == 1

    flags_text = "\n".join(metrics["anomaly_flags"]).lower()
    assert len(metrics["anomaly_flags"]) == 4
    assert "reciprocity" in flags_text
    assert "track length" in flags_text
    assert "jump-step" in flags_text
    assert "ghost rate" in flags_text


def test_tracking_decision_integration(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    for f in range(5):
        prev = np.array([PREV_NONE if f == 0 else 0], dtype=np.int32)
        nxt = np.array([NEXT_NONE if f == 4 else 0], dtype=np.int32)
        xyz = np.array([[float(f), 0.0, 0.0]], dtype=np.float64)
        store.write_linkage(f, prev, nxt, xyz)

    decision = TrackingDecision(
        resolved_config={"v_max": 2.0},
        chosen_config={"v_max": 2.0},
    )
    metrics = generate_report(
        store,
        print_report=False,
        decision=decision,
        save_decision=True,
    )

    assert decision.report_metrics == metrics
    loaded = TrackingDecision.load(store)
    assert loaded.report_metrics == metrics
    assert loaded.report_metrics["reciprocity_rate"] == pytest.approx(1.0)
    assert loaded.report_metrics["track_length_stats"]["median"] == pytest.approx(5.0)


def test_empty_store_returns_zero_metrics(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    metrics = generate_report(store, print_report=False)
    assert metrics["n_frames"] == 0
    assert metrics["total_particles"] == 0
    assert metrics["reciprocity_rate"] == 0.0
    assert metrics["track_length_stats"]["count"] == 0
    assert metrics["anomaly_flags"] == []


def test_trajectory_quality_scores_and_prio_persistence(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    # 6 frames:
    # idx 0: clean smooth 6-frame track
    # idx 1: kinky 4-frame track (frames 0..3 with 180-deg reversal at frame 2)
    # idx 2: short 2-frame track (frames 0..1)
    # idx 3 (frame 0 only): isolated particle
    for k in range(6):
        if k == 0:
            prv = np.array([PREV_NONE, PREV_NONE, PREV_NONE, PREV_NONE], dtype=np.int32)
            nxt = np.array([0, 1, 2, NEXT_NONE], dtype=np.int32)
            pos = np.array(
                [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 20.0, 0.0], [0.0, 30.0, 0.0]],
                dtype=np.float64,
            )
        elif k == 1:
            prv = np.array([0, 1, 2], dtype=np.int32)
            nxt = np.array([0, 1, NEXT_NONE], dtype=np.int32)
            pos = np.array(
                [[1.0, 0.0, 0.0], [1.0, 10.0, 0.0], [1.0, 20.0, 0.0]],
                dtype=np.float64,
            )
        elif k in (2, 3):
            prv = np.array([0, 1], dtype=np.int32)
            nxt = np.array([0, 1 if k == 2 else NEXT_NONE], dtype=np.int32)
            x_kink = -1.0 if k == 2 else 0.5
            pos = np.array(
                [[float(k), 0.0, 0.0], [x_kink, 10.0, 0.0]],
                dtype=np.float64,
            )
        else:
            prv = np.array([0], dtype=np.int32)
            nxt = np.array([0 if k < 5 else NEXT_NONE], dtype=np.int32)
            pos = np.array([[float(k), 0.0, 0.0]], dtype=np.float64)
        store.write_linkage(k, prv, nxt, pos)

    scores = compute_trajectory_quality_scores(store, write_prio=True)

    clean_score = scores[(0, 0)]
    kinky_score = scores[(0, 1)]
    short_score = scores[(0, 2)]
    isolated_score = scores[(0, 3)]

    assert clean_score == pytest.approx(1.0)
    assert isolated_score == pytest.approx(0.0)
    assert clean_score > kinky_score
    assert clean_score > short_score

    prio0 = store.read_prio(0)
    assert prio0 is not None
    assert prio0.dtype == np.int32
    assert list(prio0) == [
        int(round(clean_score * 1000)),
        int(round(kinky_score * 1000)),
        int(round(short_score * 1000)),
        0,
    ]
    # Linkage arrays remain intact.
    prv0, nxt0, _pos0 = store.read_linkage(0)
    assert list(nxt0) == [0, 1, 2, NEXT_NONE]
    assert list(prv0) == [PREV_NONE, PREV_NONE, PREV_NONE, PREV_NONE]

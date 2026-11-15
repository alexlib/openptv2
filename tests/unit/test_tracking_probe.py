"""Unit tests for openptv2.tracking_probe (Step 3a, C1-C4)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from openptv2.storage.run_store import RunStore
from openptv2.tracking_estimator import make_synthetic_exp
from openptv2.tracking_probe import (
    confirm_probe_selection,
    probe_window,
    score_probe_metrics,
    select_best_config,
)


def _make_store_with_correspondences(path: Path, first: int = 1, last: int = 8) -> RunStore:
    store = RunStore.open(path, mode="w")
    for fn in range(first, last + 1):
        pos = np.array([[float(fn), 0.0, 0.0], [float(fn), 5.0, 0.0]], dtype=np.float64)
        ids = np.array([[0, 0], [1, 1]], dtype=np.int32)
        store.write_correspondences(fn, pos, ids)
    return store


def test_probe_window_isolated_does_not_mutate_source_store(tmp_path: Path):
    store = _make_store_with_correspondences(tmp_path / "src.zarr", 1, 6)
    # Write a sentinel linkage in source store
    store.write_linkage(
        1,
        np.array([-1, -1], dtype=np.int32),
        np.array([-2, -2], dtype=np.int32),
        np.zeros((2, 3), dtype=np.float64),
        name="ptv_is",
    )
    exp = make_synthetic_exp(store, 1, 6)

    def _runner(probe_exp, win, cfg, probe_store):
        w0, w1 = win
        for fn in range(w0, w1 + 1):
            pos, _ = probe_store.read_correspondences(fn)
            prv = np.array([-1, -1], dtype=np.int32) if fn == w0 else np.array([0, 1], dtype=np.int32)
            nxt = np.array([-2, -2], dtype=np.int32) if fn == w1 else np.array([0, 1], dtype=np.int32)
            probe_store.write_linkage(fn, prv, nxt, pos, name="ptv_is")

    metrics = probe_window(exp, (2, 5), {"v_max": 2.0}, runner=_runner)
    assert metrics["median_length"] == 4.0
    assert metrics["reciprocity"] == 1.0
    # Source store sentinel linkage untouched
    prv, nxt, _ = store.read_linkage(1, name="ptv_is")
    assert np.all(nxt == -2)
    assert not store.has_linkage(2, name="ptv_is")


def test_select_best_config_coordinate_descent_improves_v_max(tmp_path: Path):
    store = _make_store_with_correspondences(tmp_path / "cd.zarr", 1, 10)
    exp = make_synthetic_exp(store, 1, 10)

    def _runner(probe_exp, win, cfg, probe_store):
        w0, w1 = win
        v_max = float(cfg.get("v_max") or 1.0)
        # Only link when v_max >= 1.5
        for fn in range(w0, w1 + 1):
            pos, _ = probe_store.read_correspondences(fn)
            if v_max >= 1.5:
                prv = np.array([-1, -1], dtype=np.int32) if fn == w0 else np.array([0, 1], dtype=np.int32)
                nxt = np.array([-2, -2], dtype=np.int32) if fn == w1 else np.array([0, 1], dtype=np.int32)
            else:
                prv = np.array([-1, -1], dtype=np.int32)
                nxt = np.array([-2, -2], dtype=np.int32)
            probe_store.write_linkage(fn, prv, nxt, pos, name="ptv_is")

    best_cfg, per_win, table = select_best_config(
        exp,
        [(1, 8)],
        {"v_max": 1.0, "slow_flow": False, "use_2d": True, "bidirectional": True},
        runner=_runner,
        min_acceptable_length=5.0,
    )
    assert best_cfg["v_max"] >= 1.5
    assert len(table) >= 2
    assert score_probe_metrics({"median_length": 8.0, "yield": 0.875, "jump_rate": 0.0}) > 0.0


def test_select_best_config_respects_budget_cap(tmp_path: Path):
    store = _make_store_with_correspondences(tmp_path / "budget.zarr", 1, 6)
    exp = make_synthetic_exp(store, 1, 6)
    calls = 0

    def _slow_runner(probe_exp, win, cfg, probe_store):
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        w0, w1 = win
        for fn in range(w0, w1 + 1):
            pos, _ = probe_store.read_correspondences(fn)
            probe_store.write_linkage(
                fn,
                np.array([-1, -1], dtype=np.int32),
                np.array([-2, -2], dtype=np.int32),
                pos,
                name="ptv_is",
            )

    best_cfg, _, table = select_best_config(
        exp,
        [(1, 6)],
        {"v_max": 1.0, "slow_flow": True, "use_2d": False, "bidirectional": False},
        runner=_slow_runner,
        probe_budget_seconds=0.01,
    )
    # Initial probe runs once, then budget is exhausted so search stops immediately
    assert calls == 1
    assert len(table) == 1
    assert best_cfg["v_max"] == 1.0


def test_confirm_probe_selection_non_tty_and_decline(monkeypatch):
    best = {"v_max": 3.0}
    init = {"v_max": 2.0}
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def _fail_if_called(_prompt):
        raise AssertionError("input() must not be called in non-TTY mode")

    out = confirm_probe_selection(best, [], [{"step": "initial"}], initial_config=init, probe_confirm="auto", input_fn=_fail_if_called)
    assert out == best

    # Explicit prompt=True with user declining restores initial_config
    out_no = confirm_probe_selection(best, [], [{"step": "initial"}], initial_config=init, probe_confirm=True, input_fn=lambda _p: "n")
    assert out_no == init

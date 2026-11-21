"""Multi-window probe and coordinate-descent parameter search (Step 3a, C1-C4).

Runs short sub-window tracking probes in an isolated temporary ``RunStore``
so probing never mutates or clobbers the experiment's real ``ptv_is`` linkage.
When probe quality is poor, runs a budgeted coordinate-descent search over
``v_max``, ``delta_t`` (when ``slow_flow`` is enabled), and ``use_2d`` /
``bidirectional`` (when reciprocity remains low).
"""

from __future__ import annotations

import builtins
import copy
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from openptv2.storage.run_store import RunStore
from openptv2.tracking_report import compute_self_consistency_metrics

log = logging.getLogger("openptv2.tracking_probe")

DEFAULT_MIN_ACCEPTABLE_LENGTH = 5.0
DEFAULT_MAX_JUMP_RATE = 0.1
DEFAULT_RECIPROCITY_THRESHOLD = 0.7
DEFAULT_SCORE_LAMBDA = 0.5
DEFAULT_PROBE_BUDGET_SECONDS = 30.0


def score_probe_metrics(
    metrics: Mapping[str, Any],
    *,
    lam: float = DEFAULT_SCORE_LAMBDA,
) -> float:
    """Compute scalar probe score: ``median_length * yield - lam * jump_rate``."""
    median_length = float(metrics.get("median_length") or 0.0)
    yield_val = float(metrics.get("yield") or 0.0)
    jump_rate = float(metrics.get("jump_rate") or 0.0)
    return median_length * yield_val - float(lam) * jump_rate


def _unwrap_source_store(exp: Any, ptv: Any = None) -> Any:
    store = getattr(exp, "store", None)
    if store is None:
        store = getattr(getattr(exp, "tracker", None), "_store", None)
    if store is None and hasattr(exp, "exp1"):
        store = getattr(exp.exp1, "store", None)
    if store is None and ptv is not None:
        store = getattr(ptv, "store", None)
    return store


def _copy_window_inputs(
    src_store: RunStore,
    dst_store: RunStore,
    w_start: int,
    w_end: int,
) -> None:
    """Copy correspondences and targets for ``[w_start, w_end]`` into ``dst_store``."""
    target_cams: list[int] = []
    if hasattr(src_store, "target_cameras"):
        try:
            target_cams = list(src_store.target_cameras())
        except Exception:
            target_cams = []

    for fn in range(int(w_start), int(w_end) + 1):
        if hasattr(src_store, "has_correspondences") and src_store.has_correspondences(fn):
            pos, cam_ids = src_store.read_correspondences(fn)
            dst_store.write_correspondences(fn, pos, cam_ids)
        for cam in target_cams:
            if src_store.has_targets(cam, fn):
                targets = src_store.read_targets(cam, fn)
                dst_store.write_targets(cam, fn, targets)


class _IsolatedSparProxy:
    """Sequence parameter proxy overriding ``first``/``last`` for a probe window."""

    def __init__(self, base_spar: Any, first: int, last: int) -> None:
        self._base = base_spar
        self._first = int(first)
        self._last = int(last)

    def get_first(self) -> int:
        return self._first

    def get_last(self) -> int:
        return self._last

    @property
    def first(self) -> int:
        return self._first

    @property
    def last(self) -> int:
        return self._last

    def __getattr__(self, name: str) -> Any:
        if self._base is None:
            raise AttributeError(name)
        return getattr(self._base, name)


class _IsolatedPtvProxy:
    """PTV module/mock proxy pointing ``store`` to the isolated probe store."""

    def __init__(self, base_ptv: Any, probe_store: RunStore) -> None:
        self._base = base_ptv
        self.store = probe_store

    def py_trackcorr_init(self, probe_exp: Any) -> Any:
        tracker = self._base.py_trackcorr_init(probe_exp)
        if tracker is not None and getattr(tracker, "_store", None) is not self.store:
            # Keep mocked trackers isolated to the probe store as well.
            if hasattr(tracker, "__dict__"):
                tracker._store = self.store
        return tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _clone_track_par(tpar: Any) -> Any:
    if tpar is None:
        return None
    try:
        return copy.copy(tpar)
    except Exception:
        attrs = {}
        for name in ("dvxmin", "dvxmax", "dvymin", "dvymax", "dvzmin", "dvzmax"):
            getter = f"get_{name}"
            if hasattr(tpar, getter):
                attrs[name] = getattr(tpar, getter)()
            elif hasattr(tpar, name):
                attrs[name] = getattr(tpar, name)
        return SimpleNamespace(**attrs)


def _build_probe_exp(
    exp: Any,
    probe_store: RunStore,
    tmp_dir: Path,
    w_start: int,
    w_end: int,
    cfg: Mapping[str, Any],
) -> SimpleNamespace:
    """Build an isolated experiment proxy bound to ``probe_store``."""
    base_pm = getattr(exp, "pm", None)
    if base_pm is None and hasattr(exp, "exp1"):
        base_pm = getattr(exp.exp1, "pm", None)

    base_params = copy.deepcopy(getattr(base_pm, "parameters", None) or {})
    seq = dict(base_params.get("sequence") or {})
    seq["first"] = int(w_start)
    seq["last"] = int(w_end)
    base_params["sequence"] = seq

    probe_track = dict(cfg)
    probe_track["probe"] = False
    base_params["track"] = probe_track

    base_spar = getattr(exp, "spar", None)
    if base_spar is None and hasattr(exp, "exp1"):
        base_spar = getattr(exp.exp1, "spar", None)

    cals = getattr(exp, "cals", None)
    if cals is None and hasattr(exp, "exp1"):
        cals = getattr(exp.exp1, "cals", None)
    cpar = getattr(exp, "cpar", None)
    if cpar is None and hasattr(exp, "exp1"):
        cpar = getattr(exp.exp1, "cpar", None)
    vpar = getattr(exp, "vpar", None)
    if vpar is None and hasattr(exp, "exp1"):
        vpar = getattr(exp.exp1, "vpar", None)
    tpar = getattr(exp, "track_par", None)
    if tpar is None and hasattr(exp, "exp1"):
        tpar = getattr(exp.exp1, "track_par", None)

    return SimpleNamespace(
        pm=SimpleNamespace(parameters=base_params),
        spar=_IsolatedSparProxy(base_spar, w_start, w_end),
        store=probe_store,
        exp_path=str(tmp_dir),
        exp_dir=str(tmp_dir),
        res_dir=str(tmp_dir / "res"),
        cals=cals,
        cpar=cpar,
        vpar=vpar,
        track_par=_clone_track_par(tpar),
        target_filenames=list(getattr(exp, "target_filenames", []) or []),
    )


def _count_forward_links(
    store: RunStore,
    w_start: int,
    w_end: int,
    linkage_name: str,
) -> int:
    links = 0
    for fn in range(int(w_start), int(w_end)):
        if store.has_linkage(fn, linkage_name):
            _prv, nxt, _xyz = store.read_linkage(fn, linkage_name)
            nxt_arr = np.asarray(nxt, dtype=np.int32)
            links += int(np.sum(nxt_arr >= 0))
    return links


def probe_window(
    exp: Any,
    window: tuple[int, int],
    cfg: Mapping[str, Any],
    *,
    ptv: Any = None,
    linkage_name: str = "ptv_is",
    runner: Callable[[Any, tuple[int, int], dict[str, Any], RunStore], None] | None = None,
) -> dict[str, Any]:
    """Run tracking on ``[w_start, w_end]`` in an isolated temporary ``RunStore``.

    Never mutates the source experiment's ``store`` or ``ptv_is`` linkage.
    """
    w_start, w_end = int(window[0]), int(window[1])
    src_store = _unwrap_source_store(exp, ptv=ptv)
    if src_store is None:
        raise ValueError("probe_window requires an experiment with an attached RunStore")

    with tempfile.TemporaryDirectory(prefix="openptv2_probe_") as tmp_name:
        tmp_dir = Path(tmp_name)
        probe_store = RunStore.open(tmp_dir, mode="w")
        _copy_window_inputs(src_store, probe_store, w_start, w_end)

        probe_cfg = dict(cfg)
        probe_cfg["probe"] = False
        if runner is not None:
            probe_exp = _build_probe_exp(exp, probe_store, tmp_dir, w_start, w_end, probe_cfg)
            runner(probe_exp, (w_start, w_end), probe_cfg, probe_store)
        else:
            from openptv2.plugins.unified_tracking import Tracking

            # Keep individual probes fast by disabling 2-pass peeling/parallelism
            # and defaulting to cython_3d (use_2d=False) unless explicitly requested.
            probe_cfg.setdefault("peeling", False)
            probe_cfg.setdefault("parallel_workers", 0)
            probe_cfg.setdefault("postprocess", False)
            probe_exp = _build_probe_exp(exp, probe_store, tmp_dir, w_start, w_end, probe_cfg)

            base_ptv = ptv
            if base_ptv is None:
                from openptv2.gui import ptv as default_ptv

                base_ptv = default_ptv
            probe_ptv = _IsolatedPtvProxy(base_ptv, probe_store)
            Tracking(ptv=probe_ptv, exp=probe_exp).do_tracking()

        report = compute_self_consistency_metrics(
            probe_store,
            w_start,
            w_end,
            linkage_name=linkage_name,
        )
        total_particles = int(report.get("total_particles") or 0)
        n_links = _count_forward_links(probe_store, w_start, w_end, linkage_name)
        yield_val = float(n_links) / float(total_particles) if total_particles > 0 else 0.0
        gaps = report.get("gap_distribution") or {}
        gap_count = int(
            int(gaps.get("gap_1", 0))
            + int(gaps.get("gap_2", 0))
            + int(gaps.get("gap_gt_2", 0))
        )
        tls = report.get("track_length_stats") or {}

        return {
            "window": [w_start, w_end],
            "total_particles": total_particles,
            "n_links": n_links,
            "yield": float(yield_val),
            "median_length": float(tls.get("median") or 0.0),
            "reciprocity": float(report.get("reciprocity_rate") or 0.0),
            "jump_rate": float(report.get("jump_step_rate") or 0.0),
            "gap_count": gap_count,
            "v_max": float(cfg["v_max"]) if cfg.get("v_max") is not None else None,
        }


def _aggregate_metrics(per_window: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not per_window:
        return {
            "yield": 0.0,
            "median_length": 0.0,
            "reciprocity": 0.0,
            "jump_rate": 0.0,
            "gap_count": 0,
        }
    from openptv2.plugins.unified_tracking import Tracking

    # Exclude non-scalar fields like "window" from median/max aggregation.
    scalar_rows = [
        {k: v for k, v in row.items() if k != "window"}
        for row in per_window
    ]
    agg = Tracking.aggregate_probe_metrics(scalar_rows)
    if "gap_count" in agg and agg["gap_count"] is not None:
        agg["gap_count"] = int(round(float(agg["gap_count"])))
    if "median_length" not in agg or agg["median_length"] is None:
        agg["median_length"] = float(
            statistics.median(float(r.get("median_length") or 0.0) for r in per_window)
        )
    return agg


def probe_windows(
    exp: Any,
    windows: Sequence[tuple[int, int]],
    cfg: Mapping[str, Any],
    *,
    ptv: Any = None,
    linkage_name: str = "ptv_is",
    runner: Callable[[Any, tuple[int, int], dict[str, Any], RunStore], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run ``probe_window`` across all ``windows`` and aggregate metrics."""
    per_window = [
        probe_window(
            exp,
            w,
            cfg,
            ptv=ptv,
            linkage_name=linkage_name,
            runner=runner,
        )
        for w in windows
    ]
    return _aggregate_metrics(per_window), per_window


def _config_key(cfg: Mapping[str, Any]) -> tuple[Any, ...]:
    v_max = cfg.get("v_max")
    v_key = round(float(v_max), 8) if v_max is not None else None
    return (
        v_key,
        int(cfg.get("delta_t", 1)),
        bool(cfg.get("use_2d", True)),
        bool(cfg.get("bidirectional", True)),
    )


def _record_trial(
    table: list[dict[str, Any]],
    step: str,
    cfg: Mapping[str, Any],
    agg: Mapping[str, Any],
    score: float,
) -> None:
    table.append(
        {
            "step": step,
            "v_max": float(cfg["v_max"]) if cfg.get("v_max") is not None else None,
            "delta_t": int(cfg.get("delta_t", 1)),
            "use_2d": bool(cfg.get("use_2d", True)),
            "bidirectional": bool(cfg.get("bidirectional", True)),
            "slow_flow": bool(cfg.get("slow_flow", False)),
            "yield": float(agg.get("yield") or 0.0),
            "median_length": float(agg.get("median_length") or 0.0),
            "reciprocity": float(agg.get("reciprocity") or 0.0),
            "jump_rate": float(agg.get("jump_rate") or 0.0),
            "gap_count": int(agg.get("gap_count") or 0),
            "score": float(score),
        }
    )


def select_best_config(
    exp: Any,
    windows: Sequence[tuple[int, int]],
    base_cfg: Mapping[str, Any],
    *,
    ptv: Any = None,
    linkage_name: str = "ptv_is",
    runner: Callable[[Any, tuple[int, int], dict[str, Any], RunStore], None] | None = None,
    min_acceptable_length: float = DEFAULT_MIN_ACCEPTABLE_LENGTH,
    max_jump_rate: float = DEFAULT_MAX_JUMP_RATE,
    reciprocity_threshold: float = DEFAULT_RECIPROCITY_THRESHOLD,
    lam: float = DEFAULT_SCORE_LAMBDA,
    probe_budget_seconds: float = DEFAULT_PROBE_BUDGET_SECONDS,
    probe_fast_3d: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Probe ``windows`` and run budgeted coordinate descent if quality is poor.

    When ``runner is None`` and ``probe_fast_3d=True``, parameter sweeps run with
    ``use_2d=False`` (Cython 3D engine) for faster convergence before restoring
    any explicit ``use_2d`` preference.

    Returns ``(best_config, probe_metrics, grid_search_table)``.
    """
    start_time = time.monotonic()
    requested_use_2d = base_cfg.get("use_2d")
    best_cfg = dict(base_cfg)
    if runner is None and probe_fast_3d:
        best_cfg["use_2d"] = False
    best_agg, best_per_window = probe_windows(
        exp,
        windows,
        best_cfg,
        ptv=ptv,
        linkage_name=linkage_name,
        runner=runner,
    )
    best_score = score_probe_metrics(best_agg, lam=lam)

    grid_table: list[dict[str, Any]] = []
    _record_trial(grid_table, "initial", best_cfg, best_agg, best_score)

    needs_search = (
        float(best_agg.get("median_length") or 0.0) < float(min_acceptable_length)
        or float(best_agg.get("jump_rate") or 0.0) > float(max_jump_rate)
    )
    if not needs_search:
        if runner is None and probe_fast_3d and requested_use_2d is not None:
            best_cfg["use_2d"] = requested_use_2d
        return best_cfg, best_per_window, grid_table

    evaluated: dict[tuple[Any, ...], tuple[dict[str, Any], list[dict[str, Any]], float]] = {
        _config_key(best_cfg): (best_agg, best_per_window, best_score)
    }

    def _budget_remaining() -> bool:
        return (time.monotonic() - start_time) < float(probe_budget_seconds)

    def _eval_candidate(step: str, cand_cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], float] | None:
        key = _config_key(cand_cfg)
        if key in evaluated:
            return evaluated[key]
        if not _budget_remaining():
            return None
        agg, per_win = probe_windows(
            exp,
            windows,
            cand_cfg,
            ptv=ptv,
            linkage_name=linkage_name,
            runner=runner,
        )
        sc = score_probe_metrics(agg, lam=lam)
        evaluated[key] = (agg, per_win, sc)
        _record_trial(grid_table, step, cand_cfg, agg, sc)
        return agg, per_win, sc

    # 1. Sweep v_max * {1.0, 1.5, 2.0}
    base_v_max = best_cfg.get("v_max")
    if base_v_max is not None and float(base_v_max) > 0.0:
        v0 = float(base_v_max)
        for mult in (1.0, 1.5, 2.0):
            cand = dict(best_cfg)
            cand["v_max"] = v0 * mult
            res = _eval_candidate("v_max", cand)
            if res is None:
                break
            agg, per_win, sc = res
            if sc > best_score:
                best_cfg, best_agg, best_per_window, best_score = cand, agg, per_win, sc

    # 2. Sweep delta_t in {1, 3, 5} only if slow_flow is True
    if bool(best_cfg.get("slow_flow", False)):
        anchor_cfg = dict(best_cfg)
        for dt in (1, 3, 5):
            cand = dict(anchor_cfg)
            cand["delta_t"] = dt
            res = _eval_candidate("delta_t", cand)
            if res is None:
                break
            agg, per_win, sc = res
            if sc > best_score:
                best_cfg, best_agg, best_per_window, best_score = cand, agg, per_win, sc

    # 3. Toggle use_2d / bidirectional only if reciprocity remains poor
    if float(best_agg.get("reciprocity") or 0.0) < float(reciprocity_threshold):
        anchor_cfg = dict(best_cfg)
        for toggle_key in ("use_2d", "bidirectional"):
            cand = dict(anchor_cfg)
            cand[toggle_key] = not bool(anchor_cfg.get(toggle_key, True))
            res = _eval_candidate(f"toggle_{toggle_key}", cand)
            if res is None:
                break
            agg, per_win, sc = res
            if sc > best_score:
                best_cfg, best_agg, best_per_window, best_score = cand, agg, per_win, sc
                anchor_cfg = dict(cand)

    return best_cfg, best_per_window, grid_table


def format_grid_search_table(table: Sequence[Mapping[str, Any]]) -> str:
    """Format the coordinate-descent probe table for logging or interactive display."""
    if not table:
        return "(no probe trials)"
    lines = [
        "step | v_max | delta_t | use_2d | bidir | yield | med_len | recip | jump | score",
    ]
    for row in table:
        v_str = f"{row['v_max']:.3g}" if row.get("v_max") is not None else "auto"
        lines.append(
            f"{row.get('step', ''):>12} | {v_str:>6} | {row.get('delta_t', 1):>7} | "
            f"{str(row.get('use_2d', True)):>6} | {str(row.get('bidirectional', True)):>5} | "
            f"{float(row.get('yield') or 0.0):>5.3f} | {float(row.get('median_length') or 0.0):>7.2f} | "
            f"{float(row.get('reciprocity') or 0.0):>5.3f} | {float(row.get('jump_rate') or 0.0):>5.3f} | "
            f"{float(row.get('score') or 0.0):>6.3f}"
        )
    return "\n".join(lines)


def _should_prompt(probe_confirm: Any) -> bool:
    if isinstance(probe_confirm, bool):
        return probe_confirm
    if probe_confirm is None:
        return sys.stdin.isatty()
    val = str(probe_confirm).strip().lower()
    if val in ("auto", ""):
        return sys.stdin.isatty()
    if val in ("1", "true", "yes", "on"):
        return True
    return False


def confirm_probe_selection(
    best_config: Mapping[str, Any],
    probe_metrics: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    grid_search_table: Sequence[Mapping[str, Any]],
    *,
    initial_config: Mapping[str, Any] | None = None,
    probe_confirm: Any = "auto",
    input_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Confirm probe-selected config interactively, or auto-pick in non-TTY/batch."""
    chosen = dict(best_config)
    if not _should_prompt(probe_confirm):
        return chosen

    summary = format_grid_search_table(grid_search_table)
    print(f"[unified-probe]\n{summary}")
    fn = input_fn or builtins.input
    try:
        reply = fn("Accept probe-selected config? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return chosen

    if reply in ("n", "no") and initial_config is not None:
        return dict(initial_config)
    return chosen

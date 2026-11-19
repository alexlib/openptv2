"""Unified tracker — the single tracker. End users do not pick a tracker; this
one adapts to the data.

Goal: most accurate trajectories possible for the given data, with speed
coming from the parallel-run option rather than from dropping accuracy.

Adaptation is heuristic and (optionally) validated on a small probe run:
  1. User (or pipeline) runs a small subset / probe and reports metrics.
  2. Unified tracker reads the data characteristics (density, displacement vs.
     noise, conditioning) and enables only the passes that help accuracy.

Track YAML options (all default to "auto" unless noted):
  v_max, max_cands, dt, cold_start_gate  - motion/search bounds (physics).
  bidirectional: auto|true|false
      Run forward+backward passes. Enabled when density is not extreme and
      the data shows one-directional false links (auto). Disable to save time.
  use_2d: auto|true|false
      2D leaf re-ranking for ambiguous 3D. Enabled on dense/ambiguous data.
  lookahead: true|false
      4BE n+2-frame tie-breaker in the candidate cost. On by default.
  delta_t: auto|1..N
      Coarse-to-fine step for slow high-FPS flows. auto -> 5 if slow flow
      (displacement < noise), else 1.
  new_particles: true|false  - seed new particles mid-sequence. On by default.
  postprocess: true|false    - gap relink + cold start + reciprocity. On.
  confirm_link: auto|number|null
      Two-hop confirmation to cut jump-steps. auto -> 0.3 when v_max small.
  quality_post: false|true   - smoothness/reconnect/trim. Off by default
      (extra cost); enable for final production runs.
  slow_flow: false|true      - force coarse-to-fine. auto-detected otherwise.
  parallel_workers: 0       - 0 sequential; >1 chunked/distributed (speed).
  density_estimate: 0       - optional hint (particles/frame) for heuristics.

Legacy tracker names are kept as aliases that map to this tracker with fixed
config bundles, so existing experiments keep working.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("openptv2.unified_tracking")


class Tracking:
    """Unified tracking plugin (experiment-level)."""

    def __init__(self, ptv: Any = None, exp: Any = None):
        self.ptv = ptv
        self.exp = exp
        self._resolved: dict[str, Any] = {}
        self._decision: Any = None

    # ------------------------------------------------------------------
    # regime detection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _track_cfg(exp: Any) -> dict:
        pm = getattr(exp, "pm", None)
        if pm is None and hasattr(exp, "exp1"):
            pm = getattr(exp.exp1, "pm", None)
        return (pm.parameters.get("track", {}) if pm else {}) or {}

    @staticmethod
    def _plugins_cfg(exp: Any) -> dict:
        pm = getattr(exp, "pm", None)
        if pm is None and hasattr(exp, "exp1"):
            pm = getattr(exp.exp1, "pm", None)
        return (pm.parameters.get("plugins", {}) if pm else {}) or {}

    @staticmethod
    def _resolve_bool(val: Any, default: bool) -> bool:
        if val is None:
            return default
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("auto", ""):
                return default
            return v in ("1", "true", "yes", "on")
        return bool(val)

    @staticmethod
    def _is_auto_value(val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, str):
            return val.strip().lower() in ("auto", "")
        return False

    def _resolve_store_and_range(self, exp: Any) -> tuple[Any, Any, int | None, int | None]:
        """Extract ``(pm, store, first, last)`` from an experiment object."""
        from openptv2.tracking_estimator import _resolve_frame_range, _unwrap_exp

        pm, spar, store, _cals, _cpar = _unwrap_exp(exp)
        if store is None and self.ptv is not None:
            store = getattr(self.ptv, "store", None)
        first, last = _resolve_frame_range(spar, pm, store)
        return pm, store, first, last

    def _auto_regime(self, exp: Any, track_cfg: dict) -> dict:
        """Best-effort regime detection; returns a dict of resolved flags."""
        est: dict[str, Any] = {}
        if exp is not None:
            from openptv2.tracking_estimator import _unwrap_exp, estimate_params

            _, _, store, _, _ = _unwrap_exp(exp)
            if store is not None:
                est = estimate_params(exp, track_cfg)

        if est:
            slow_flow = bool(est.get("slow_flow", False))
            density = float(est.get("density", 0.0) or 0.0)
        else:
            slow_flow = bool(track_cfg.get("slow_flow", False))
            density = float(track_cfg.get("density_estimate", 0.0) or 0.0)
            if track_cfg.get("slow_flow") is True:
                slow_flow = True

        # Bidirectional + reciprocity: on by default for accuracy, but disable
        # on extreme density where the backward pass becomes the bottleneck and
        # the forward pass already saturates accuracy.
        bidirectional = not (density > 5000)

        # Engine selection: dense scenes benefit from the 3D priority cascade
        # (priority_segment_3d) which is more robust to 2D ghost ambiguity.
        # Sparse/clean scenes use the epipolar engine (trackcorr) for speed.
        # use_2d=True means "run epipolar 2D engine"; False means "3D engine".
        ghost_rate = float(est.get("ghost_rate", 0.0) or 0.0) if est else 0.0
        if density == 0:
            use_2d = True  # unknown -> epipolar (safe, fast)
        elif ghost_rate > 0.2 or density >= 500:
            use_2d = False  # ghosty/dense -> 3D engine (priority_segment_3d)
        else:
            use_2d = True  # clean/sparse -> epipolar

        # Honour explicit overrides (auto -> resolved above).
        bidirectional = self._resolve_bool(track_cfg.get("bidirectional", "auto"), bidirectional)
        use_2d = self._resolve_bool(track_cfg.get("use_2d", "auto"), use_2d)

        raw_delta_t = track_cfg.get("delta_t", "auto")
        if self._is_auto_value(raw_delta_t):
            delta_t = 5 if slow_flow else 1
        else:
            delta_t = int(raw_delta_t)

        confirm = track_cfg.get("confirm_link", "auto")
        if self._is_auto_value(confirm):
            confirm = None

        regime = {
            "bidirectional": bidirectional,
            "use_2d": use_2d,
            "slow_flow": slow_flow,
            "delta_t": delta_t,
            "confirm": confirm,
        }
        self._resolved = {
            **est,
            "density": density,
            "slow_flow": slow_flow,
            "bidirectional": bidirectional,
            "use_2d": use_2d,
            "delta_t": delta_t,
            "confirm_link": confirm,
        }
        return regime

    # ------------------------------------------------------------------
    # multi-window probing for long / non-stationary sequences
    # ------------------------------------------------------------------
    @staticmethod
    def sample_probe_windows(first: int, last: int, n_windows: int = 5, window: int = 30) -> list[tuple[int, int]]:
        """Sample several frame windows spread across [first, last].

        Data characteristics (density, motion, noise) can drift over thousands
        of frames, so a single 20-40 frame probe is not enough. We take
        ``n_windows`` windows of ``window`` frames, spaced evenly, and
        aggregate (median for typical values, max for safety bounds like
        v_max) so the chosen parameters cover the worst case along the run.
        """
        total = last - first + 1
        if total <= window:
            return [(first, last)]
        n_windows = max(1, min(n_windows, total // window))
        step = total / (n_windows + 1)
        out = []
        for i in range(1, n_windows + 1):
            c = int(first + i * step)
            s = max(first, c - window // 2)
            e = min(last, s + window - 1)
            s = max(first, e - window + 1)
            out.append((s, e))
        # de-duplicate
        uniq = []
        for w in out:
            if w not in uniq:
                uniq.append(w)
        return uniq

    @staticmethod
    def aggregate_probe_metrics(per_window: list[dict]) -> dict:
        """Aggregate per-window metrics: median for typical, max for safety."""
        if not per_window:
            return {}
        import statistics
        keys = per_window[0].keys()
        agg = {}
        for k in keys:
            vals = [w[k] for w in per_window if k in w and w[k] is not None]
            if not vals:
                agg[k] = None
                continue
            if k in ("v_max", "density", "jump_rate"):
                agg[k] = max(vals)  # safety bound
            else:
                agg[k] = statistics.median(vals)
        # regime drift flag
        densities = [w.get("density") for w in per_window if w.get("density") is not None]
        if densities and (max(densities) - min(densities)) > 0.5 * (sum(densities) / len(densities) + 1e-9):
            agg["regime_drift"] = True
        else:
            agg["regime_drift"] = False
        return agg

    def _apply_conditioning_gate(self, regime: dict, track_cfg: dict) -> None:
        """Apply B1 conditioning gate warnings and safe regime overrides."""
        conditioning_ratio = self._resolved.get("conditioning_ratio")
        if conditioning_ratio is None:
            return
        ratio = float(conditioning_ratio)
        if ratio < 0.2:
            msg = (
                f"[unified] STRONG WARNING: conditioning_ratio={ratio:.3f} < 0.2 "
                "(displacement is far below Z noise floor). Calibration improvements "
                "(camera geometry, stereo calibration, detection accuracy) are strongly "
                "recommended; proceeding with slow_flow and 2D disambiguation."
            )
            log.warning(msg)
            print(msg)
        elif ratio < 0.5:
            msg = (
                f"[unified] WARNING: conditioning_ratio={ratio:.3f} < 0.5 "
                "(displacement is below Z noise floor). Recommending slow_flow "
                "coarse-to-fine tracking (delta_t=5) and 2D disambiguation."
            )
            log.warning(msg)
            print(msg)

        if ratio < 0.5:
            if self._is_auto_value(track_cfg.get("slow_flow")):
                regime["slow_flow"] = True
                self._resolved["slow_flow"] = True
                if self._is_auto_value(track_cfg.get("delta_t", "auto")):
                    regime["delta_t"] = 5
                    self._resolved["delta_t"] = 5
            raw_use_2d = track_cfg.get("use_2d", "auto")
            if self._is_auto_value(raw_use_2d) or self._resolve_bool(raw_use_2d, True):
                regime["use_2d"] = True
                self._resolved["use_2d"] = True

    @staticmethod
    def _apply_v_max_to_tpar(exp: Any, v_max: float | None) -> None:
        """Apply resolved v_max to ``exp.track_par`` in-memory."""
        tpar = getattr(exp, "track_par", None)
        if tpar is None or v_max is None:
            return
        v_val = float(v_max)
        for setter, val in (
            ("set_dvxmin", -v_val),
            ("set_dvxmax", v_val),
            ("set_dvymin", -v_val),
            ("set_dvymax", v_val),
            ("set_dvzmin", -v_val),
            ("set_dvzmax", v_val),
        ):
            if hasattr(tpar, setter):
                getattr(tpar, setter)(val)
            else:
                setattr(tpar, setter[4:], val)

    def _run_hybrid_deltat(self, pm: Any, track_cfg: dict, regime: dict) -> bool:
        """Delegate to ``hybrid_deltat_3d`` with temporary in-memory track_cfg."""
        from openptv2.plugins import hybrid_deltat_3d

        if not hasattr(hybrid_deltat_3d, "Tracking"):
            return False

        params = getattr(pm, "parameters", None) if pm is not None else None
        had_track = isinstance(params, dict) and "track" in params
        orig_track = params.get("track") if isinstance(params, dict) else None
        if isinstance(params, dict):
            mem_track = dict(track_cfg)
            v_max = self._resolved.get("v_max")
            if v_max is not None:
                v_val = float(v_max)
                for k in ("dvxmax", "dvymax", "dvzmax"):
                    mem_track[k] = v_val
                for k in ("dvxmin", "dvymin", "dvzmin"):
                    mem_track[k] = -v_val
            if self._is_auto_value(track_cfg.get("stride")):
                dt = int(regime["delta_t"])
                mem_track["stride"] = dt if dt > 1 else 5
            params["track"] = mem_track

        try:
            hybrid_deltat_3d.Tracking(ptv=self.ptv, exp=self.exp).do_tracking()
            return True
        finally:
            if isinstance(params, dict):
                if had_track:
                    params["track"] = orig_track
                else:
                    params.pop("track", None)

    def _save_tracking_decision(
        self,
        store: Any,
        first: int | None,
        last: int | None,
        regime: dict,
        chosen_config: dict[str, Any],
        *,
        probe_metrics: list[dict[str, Any]] | None = None,
        grid_search_table: list[dict[str, Any]] | None = None,
    ) -> None:
        """Populate and persist ``TrackingDecision`` (including post-run report)."""
        from openptv2.tracking_decision import TrackingDecision

        raw_ratio = self._resolved.get("conditioning_ratio")
        decision = TrackingDecision(
            resolved_config=dict(self._resolved),
            regime_flags=dict(regime),
            conditioning_ratio=float(raw_ratio) if raw_ratio is not None else None,
            probe_metrics=list(probe_metrics or []),
            grid_search_table=list(grid_search_table or []),
            chosen_config=dict(chosen_config),
        )
        if store is not None:
            if hasattr(store, "linkage_frames"):
                try:
                    from openptv2.tracking_report import generate_report

                    generate_report(
                        store,
                        first=first,
                        last=last,
                        print_report=False,
                        decision=decision,
                        save_decision=False,
                    )
                except Exception as exc:  # pragma: no cover - never break tracking
                    log.warning("[unified] report generation skipped: %r", exc)
            decision.save(store)
        self._decision = decision

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        track_cfg = dict(self._track_cfg(self.exp))
        # Enable track_assisted corrective extension by default to repair fragments.
        track_cfg.setdefault("corrective_passes", 2)
        regime = self._auto_regime(self.exp, track_cfg)
        self._apply_conditioning_gate(regime, track_cfg)

        pm, store, first, last = self._resolve_store_and_range(self.exp)
        probe_metrics: list[dict[str, Any]] = []
        grid_search_table: list[dict[str, Any]] = []

        probe_opt = track_cfg.get("probe", False)
        do_probe = (
            probe_opt in (True, "true", "yes", "on")
            or (
                isinstance(probe_opt, str)
                and probe_opt.strip().lower() == "auto"
                and first is not None
                and last is not None
                and (last - first + 1) >= 60
            )
        )
        if do_probe and store is not None and first is not None and last is not None and last >= first:
            try:
                from openptv2.tracking_probe import confirm_probe_selection, select_best_config

                wins = self.sample_probe_windows(
                    first,
                    last,
                    n_windows=int(track_cfg.get("probe_windows", 5)),
                    window=int(track_cfg.get("probe_window_frames", 30)),
                )
                base_probe_cfg = {
                    "v_max": self._resolved.get("v_max"),
                    "slow_flow": bool(regime["slow_flow"]),
                    "delta_t": int(regime["delta_t"]),
                    "use_2d": bool(regime["use_2d"]),
                    "bidirectional": bool(regime["bidirectional"]),
                    "confirm_link": regime["confirm"],
                }
                best_cfg, probe_metrics, grid_search_table = select_best_config(
                    self.exp,
                    wins,
                    base_probe_cfg,
                    ptv=self.ptv,
                    probe_budget_seconds=float(track_cfg.get("probe_budget_seconds", 30.0)),
                )
                selected = confirm_probe_selection(
                    best_cfg,
                    probe_metrics,
                    grid_search_table,
                    initial_config=base_probe_cfg,
                    probe_confirm=track_cfg.get("probe_confirm", "auto"),
                )
                for key in ("slow_flow", "delta_t", "use_2d", "bidirectional"):
                    if key in selected:
                        regime[key] = selected[key]
                if selected.get("v_max") is not None:
                    self._resolved["v_max"] = selected["v_max"]
            except Exception as exc:  # pragma: no cover - never break tracking
                log.warning("[unified] probe skipped: %r", exc)

        v_max = self._resolved.get("v_max")
        chosen_config: dict[str, Any] = {
            "v_max": v_max,
            "bidirectional": bool(regime["bidirectional"]),
            "use_2d": bool(regime["use_2d"]),
            "slow_flow": bool(regime["slow_flow"]),
            "delta_t": int(regime["delta_t"]),
            "confirm_link": regime["confirm"],
        }

        ran_hybrid = False
        # Slow flow -> delegate to hybrid_deltat_3d plugin (coarse-to-fine).
        if regime["slow_flow"] or int(regime["delta_t"]) > 1:
            try:
                ran_hybrid = self._run_hybrid_deltat(pm, track_cfg, regime)
            except Exception as exc:  # pragma: no cover - fallback
                log.warning("hybrid delegate failed: %r; falling back", exc)

        ran_parallel = False
        raw_workers = track_cfg.get("parallel_workers", 0)
        explicit_parallel = not self._is_auto_value(raw_workers) and int(raw_workers or 0) > 1
        if not ran_hybrid and explicit_parallel:
            workers = int(raw_workers or 0)
            if (
                workers > 1
                and store is not None
                and hasattr(self.exp, "cpar")
                and hasattr(self.exp, "vpar")
                and hasattr(self.exp, "track_par")
                and hasattr(self.exp, "spar")
                and hasattr(self.exp, "cals")
            ):
                try:
                    from openptv2.tracking_chunked import track_sequence_chunked_parallel

                    self._apply_v_max_to_tpar(self.exp, v_max)
                    track_sequence_chunked_parallel(
                        self.exp.cpar,
                        self.exp.vpar,
                        self.exp.track_par,
                        self.exp.spar,
                        self.exp.cals,
                        store=store,
                        n_workers=workers,
                        mode="corr" if regime["use_2d"] else "3d",
                        postprocess=bool(track_cfg.get("postprocess", True)),
                    )
                    ran_parallel = True
                except Exception as exc:  # pragma: no cover - fallback to sequential
                    log.warning("[unified] parallel chunked tracking skipped: %r", exc)

        ran_peeling = False
        if not ran_hybrid and not ran_parallel:
            peel_opt = track_cfg.get("peeling", True)
            ghost_est = float(self._resolved.get("ghost_rate") or 0.0)
            if isinstance(peel_opt, str) and peel_opt.strip().lower() == "auto":
                do_peeling = ghost_est > 0.25
            else:
                do_peeling = self._resolve_bool(peel_opt, True)
            if (
                do_peeling
                and store is not None
                and hasattr(self.exp, "cpar")
                and hasattr(self.exp, "vpar")
                and hasattr(self.exp, "cals")
            ):
                try:
                    from openptv2.layered_tracking import peel_and_track

                    self._apply_v_max_to_tpar(self.exp, v_max)
                    peel_tiers = int(track_cfg.get("peel_tiers", 3 if ghost_est > 0.35 else 2))
                    raw_rcm = track_cfg.get("rcm_threshold")
                    rcm_thresh = float(raw_rcm) if raw_rcm is not None else None
                    peel_and_track(
                        self.exp.cpar,
                        self.exp.vpar,
                        self.exp.track_par,
                        self.exp.spar,
                        self.exp.cals,
                        store,
                        allow_shared_targets=bool(track_cfg.get("allow_shared_targets", False)),
                        engine=str(track_cfg.get("peel_engine", "3d")),
                        tiers=peel_tiers,
                        rcm_threshold=rcm_thresh,
                    )
                    ran_peeling = True
                except Exception as exc:  # pragma: no cover - fallback
                    log.warning("[unified] peeling skipped: %r", exc)

        if not ran_hybrid and not ran_peeling and not ran_parallel:
            if self._is_auto_value(raw_workers):
                workers = 0
                if first is not None and last is not None and (last - first + 1) >= 24:
                    import os

                    workers = min(os.cpu_count() or 1, 8)
                if (
                    workers > 1
                    and store is not None
                    and hasattr(self.exp, "cpar")
                    and hasattr(self.exp, "vpar")
                    and hasattr(self.exp, "track_par")
                    and hasattr(self.exp, "spar")
                    and hasattr(self.exp, "cals")
                ):
                    try:
                        from openptv2.tracking_chunked import track_sequence_chunked_parallel

                        self._apply_v_max_to_tpar(self.exp, v_max)
                        track_sequence_chunked_parallel(
                            self.exp.cpar,
                            self.exp.vpar,
                            self.exp.track_par,
                            self.exp.spar,
                            self.exp.cals,
                            store=store,
                            n_workers=workers,
                            mode="corr" if regime["use_2d"] else "3d",
                            postprocess=bool(track_cfg.get("postprocess", True)),
                        )
                        ran_parallel = True
                    except Exception as exc:  # pragma: no cover - fallback to sequential
                        log.warning("[unified] parallel chunked tracking skipped: %r", exc)

        if not ran_hybrid and not ran_peeling and not ran_parallel:
            # Optional early ghost filter (low-support correspondences) before tracking.
            gf = track_cfg.get("ghost_filter", "auto")
            do_gf = gf in (True, "true", "yes", "on") or gf == "auto"
            if do_gf and store is not None and first is not None and last is not None and last >= first:
                try:
                    from openptv2.ghost_filter import filter_low_support

                    dropped = filter_low_support(
                        store,
                        list(range(first, last + 1)),
                        min_cameras=int(track_cfg.get("ghost_min_cameras", 3)),
                    )
                    total = sum(dropped.values())
                    if total:
                        log.info("[unified] ghost filter dropped %d low-support correspondences", total)
                except Exception as exc:  # pragma: no cover - never break tracking
                    log.warning("[unified] ghost filter skipped: %r", exc)

            # Frame-quality awareness: interpolate low-quality frames.
            try:
                from openptv2.frame_quality import interpolate_frame_correspondences, low_quality_frames

                if store is not None and first is not None and last is not None and last >= first:
                    lq = low_quality_frames(store, first, last, threshold=float(track_cfg.get("frame_quality_threshold", 0.5)))
                    n_interp = 0
                    for f in lq:
                        n_interp += interpolate_frame_correspondences(store, f, first, last)
                    if n_interp:
                        log.info("[unified] frame-quality: interpolated %d points in %d low-quality frames", n_interp, len(lq))
            except Exception as exc:  # pragma: no cover - never break tracking
                log.warning("[unified] frame-quality skipped: %r", exc)

            self._apply_v_max_to_tpar(self.exp, v_max)
            tracker = self.ptv.py_trackcorr_init(self.exp)
            self.exp.tracker = tracker
            if store is None:
                store = getattr(tracker, "_store", None)

            log.info(
                "Unified tracking: 2d=%s bidir=%s delta_t=%s confirm=%s",
                regime["use_2d"], regime["bidirectional"], regime["delta_t"], regime["confirm"],
            )

            # Backbone is trackcorr (multi-camera epipolar engine) when use_2d=True;
            # 3D-only pass when use_2d=False.
            if regime["use_2d"] or not hasattr(tracker, "full_forward_3d"):
                tracker.full_forward()
            else:
                tracker.full_forward_3d()

            if regime["bidirectional"]:
                tracker.full_backward()

            confirm_tol = regime["confirm"]
            if confirm_tol is not None and float(confirm_tol) > 0:
                try:
                    from openptv2.tracking_postprocess import confirm_links

                    linkage_base = getattr(tracker, "_naming", {}).get("linkage", "")
                    if linkage_base and first is not None and last is not None and last >= first:
                        confirm_links(
                            linkage_base,
                            first,
                            last,
                            tol=float(confirm_tol),
                            store=store,
                        )
                except Exception as exc:  # pragma: no cover - never break tracking
                    log.warning("[unified] confirm_links skipped: %r", exc)

            if track_cfg.get("postprocess", True):
                max_gap = int(track_cfg.get("max_gap", 2))
                stats = tracker.postprocess(max_gap=max_gap)
                log.info(
                    "Post-process links: %s -> %s (max_gap=%d)",
                    stats.get("links_before", 0),
                    stats.get("links_after", 0),
                    max_gap,
                )

            from openptv2.track_assisted import run_configured_corrective_pass

            run_configured_corrective_pass(self.exp, tracker, track_cfg)

        # Trajectory-level ghost suppression (temporal coherence + target uniqueness).
        try:
            from openptv2.temporal_smoothing import (
                trajectory_level_ghost_filter,
                resolve_smoothing_config,
            )

            _ctx = dict(regime)
            _ctx.update(self._resolved or {})
            linkage_name = (getattr(self.ptv, "_naming", {}) or {}).get("linkage", "ptv_is") if self.ptv else "ptv_is"
            if regime.get("use_2d", True) and resolve_smoothing_config(track_cfg, _ctx) and store is not None and first is not None and last is not None and last >= first:
                stats = trajectory_level_ghost_filter(
                    store, first, last,
                    residual_threshold=float(track_cfg.get("smooth_residual_threshold", 2.0)),
                    linkage_name=linkage_name,
                    point_level=bool(track_cfg.get("point_level_smoothing", True)),
                )
                log.info("[unified] trajectory-level ghost filter: %s", stats)
        except Exception as exc:  # pragma: no cover - never break tracking
            log.warning("[unified] trajectory-level ghost filter skipped: %r", exc)

        # Multi-D search: only for epipolar engine.
        try:
            if not regime.get("use_2d", True):
                raise RuntimeError("skip rerank in 3D mode")
            rerank_opt = track_cfg.get("epipolar_rerank", "auto")
            do_rerank = rerank_opt in (True, "true", "yes", "on") or rerank_opt == "auto"
            if do_rerank and store is not None and first is not None and last is not None and last >= first:
                from openptv2.epipolar_rerank import rerank_trajectories

                rstats = rerank_trajectories(
                    store, first, last,
                    linkage_name=linkage_name,
                    min_consistency=float(track_cfg.get("epipolar_min_consistency", 0.5)),
                )
                log.info("[unified] epipolar re-rank: %s", rstats)
        except Exception as exc:  # pragma: no cover - never break tracking
            log.warning("[unified] epipolar re-rank skipped: %r", exc)

        # Multi-frame stitching: only for epipolar engine (3D engine has its own gap bridging).
        try:
            if not regime.get("use_2d", True):
                raise RuntimeError("skip stitch in 3D mode")
            stitch_opt = track_cfg.get("stitch", "auto")
            do_stitch = stitch_opt in (True, "true", "yes", "on") or stitch_opt == "auto"
            if do_stitch and store is not None and first is not None and last is not None and last >= first:
                from openptv2.trajectory_stitch import stitch_trajectories

                st = stitch_trajectories(
                    store, first, last,
                    max_gap=int(track_cfg.get("max_gap", 2)),
                    spatial_tol=float(track_cfg.get("stitch_spatial_tol", 5.0)),
                    kink_angle_deg=float(track_cfg.get("stitch_kink_angle", 60.0)),
                    linkage_name=linkage_name,
                )
                log.info("[unified] trajectory stitch: %s", st)
        except Exception as exc:  # pragma: no cover - never break tracking
            log.warning("[unified] trajectory stitch skipped: %r", exc)

        # Sliding-window MHT: re-optimize links over windows of frames.
        try:
            mht_opt = track_cfg.get("mht", "auto")
            do_mht = mht_opt in (True, "true", "yes", "on")  # auto-disabled: MHT over-merges at current tuning
            if do_mht and store is not None and first is not None and last is not None and last >= first:
                from openptv2.mht_linker import mht_relink

                mstats = mht_relink(
                    store, first, last,
                    window=int(track_cfg.get("mht_window", 3)),
                    max_gap=int(track_cfg.get("max_gap", 2)),
                    spatial_tol=float(track_cfg.get("mht_spatial_tol", 2.0)),
                    linkage_name=linkage_name,
                )
                log.info("[unified] MHT relink: %s", mstats)
        except Exception as exc:  # pragma: no cover - never break tracking
            log.warning("[unified] MHT relink skipped: %r", exc)

        self._save_tracking_decision(
            store,
            first,
            last,
            regime,
            chosen_config,
            probe_metrics=probe_metrics,
            grid_search_table=grid_search_table,
        )

    def track_frames(self, frame_particles: list) -> list:
        """Array-level API: delegate to Cython 3D engine for now."""
        # Reuse the priority-segment engine via the existing plugin.
        from openptv2.plugins.cython_3d_tracking import Cython3DTracker

        t = Cython3DTracker(ptv=self.ptv, exp=self.exp)
        # Build a minimal tracker instance is not possible without ptv/exp;
        # fall back to the array-level Cython3DTracker.track_frames.
        return Cython3DTracker().track_frames(frame_particles)

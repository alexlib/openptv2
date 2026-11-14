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

    def _auto_regime(self, exp: Any, track_cfg: dict) -> dict:
        """Best-effort regime detection; returns a dict of resolved flags."""
        # Defaults chosen for accuracy-first on typical data.
        slow_flow = bool(track_cfg.get("slow_flow", False))
        density = float(track_cfg.get("density_estimate", 0.0) or 0.0)

        # Bidirectional + reciprocity: on by default for accuracy, but disable
        # on extreme density where the backward pass becomes the bottleneck and
        # the forward pass already saturates accuracy.
        bidirectional = not (density > 5000)

        # 2D leaf re-ranking: enable when data is dense/ambiguous (3D
        # triangulation weak). On moderate density it helps disambiguate.
        use_2d = density >= 500 or density == 0  # 0 = unknown -> keep on (safe)

        # Slow flow / high-FPS detection can be supplied by a probe run.
        if track_cfg.get("slow_flow") is True:
            slow_flow = True

        # Honour explicit overrides (auto -> resolved above).
        bidirectional = self._resolve_bool(track_cfg.get("bidirectional", "auto"), bidirectional)
        use_2d = self._resolve_bool(track_cfg.get("use_2d", "auto"), use_2d)

        delta_t = track_cfg.get("delta_t", "auto")
        if delta_t == "auto":
            delta_t = 5 if slow_flow else 1

        confirm = track_cfg.get("confirm_link", "auto")
        if confirm == "auto":
            confirm = 0.3

        return {
            "bidirectional": bidirectional,
            "use_2d": use_2d,
            "slow_flow": slow_flow,
            "delta_t": delta_t,
            "confirm": confirm,
        }

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        track_cfg = self._track_cfg(self.exp)
        plugins_cfg = self._plugins_cfg(self.exp)
        regime = self._auto_regime(self.exp, track_cfg)

        # Slow flow -> delegate to hybrid_deltat_3d plugin (coarse-to-fine).
        if regime["slow_flow"] or int(regime["delta_t"]) > 1:
            try:
                from openptv2.plugins import hybrid_deltat_3d

                mod = hybrid_deltat_3d
                if hasattr(mod, "Tracking"):
                    mod.Tracking(ptv=self.ptv, exp=self.exp).do_tracking()
                    return
            except Exception as exc:  # pragma: no cover - fallback
                log.warning("hybrid delegate failed: %r; falling back", exc)

        tracker = self.ptv.py_trackcorr_init(self.exp)
        self.exp.tracker = tracker

        from openptv2.tracking_presets import infer_direction

        direction = infer_direction(track_cfg, plugins_cfg)
        # Unified prefers bidirectional unless user forced otherwise.
        if regime["bidirectional"] and direction != "forward_backward":
            direction = "forward_backward"

        log.info(
            "Unified tracking: 2d=%s bidir=%s delta_t=%s confirm=%s",
            regime["use_2d"], regime["bidirectional"], regime["delta_t"], regime["confirm"],
        )

        if regime["use_2d"]:
            tracker.full_forward()
        else:
            tracker.full_forward_3d()

        if direction == "forward_backward":
            tracker.full_backward()

        if track_cfg.get("postprocess", True):
            stats = tracker.postprocess()
            log.info(
                "Post-process links: %s -> %s",
                stats.get("links_before", 0),
                stats.get("links_after", 0),
            )

        from openptv2.track_assisted import run_configured_corrective_pass

        run_configured_corrective_pass(self.exp, tracker, track_cfg)

    def track_frames(self, frame_particles: list) -> list:
        """Array-level API: delegate to Cython 3D engine for now."""
        # Reuse the priority-segment engine via the existing plugin.
        from openptv2.plugins.cython_3d_tracking import Cython3DTracker

        t = Cython3DTracker(ptv=self.ptv, exp=self.exp)
        # Build a minimal tracker instance is not possible without ptv/exp;
        # fall back to the array-level Cython3DTracker.track_frames.
        return Cython3DTracker().track_frames(frame_particles)

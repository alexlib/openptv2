"""Tracking plugin ``holistic``: two_phase linking inside the holistic loop (:func:`openptv2.holistic.run_holistic`).

YAML (``track:`` section, besides the two_phase keys): ``holistic_rounds`` (default 2), ``holistic_min_resid``
(default 0.25), ``holistic_partner_min_len`` (default 10), ``holistic_extend`` (default 0: passes of strict
track-assisted extension after the loop; helps at moderate density, not at very dense data).
"""

from __future__ import annotations

from pathlib import Path


class Tracking:
    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        from openptv2.holistic import run_holistic
        from openptv2.plugins.two_phase_tracking import track_store

        if self.exp is None:
            raise ValueError("No experiment object provided")
        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)
        track_cfg = pm.parameters.get("track", {}) if pm else {}
        store = getattr(self.exp, "_store", None)
        if store is None:
            from openptv2.storage import RunStore
            from openptv2.storage.run_store import find_existing_store

            zarr_path = find_existing_store(Path.cwd())
            if zarr_path is None:
                raise RuntimeError("holistic tracking needs a RunStore with targets and correspondences")
            store = RunStore(zarr_path, mode="a")
        frames = sorted(store.frames())
        seq = pm.parameters.get("sequence", {}) if pm else {}
        if "first" in seq and "last" in seq:
            frames = [f for f in frames if int(seq["first"]) <= f <= int(seq["last"])]
        if len(frames) > 1:                                   # first contiguous block, as two_phase
            block = [frames[0]]
            for f in frames[1:]:
                if f != block[-1] + 1:
                    break
                block.append(f)
            frames = block
        cals, cpar = getattr(self.exp, "cals", None), getattr(self.exp, "cpar", None)
        vpar = getattr(self.exp, "vpar", None)

        def relink():
            return track_store(store, frames, track_cfg, cals, cpar)

        run_holistic(store, frames, cpar, list(cals), vpar, relink,
                     rounds=int(track_cfg.get("holistic_rounds", 2)),
                     min_resid=float(track_cfg.get("holistic_min_resid", 0.25)),
                     partner_min_len=int(track_cfg.get("holistic_partner_min_len", 10)))
        passes = int(track_cfg.get("holistic_extend", 0))
        if passes > 0:                                        # strict track-assisted extension on the final tracks
            from openptv2.storage import RunStore
            from openptv2.track_assisted import run_corrective_pass

            store = RunStore(store.store_path, mode="a")          # fresh object: no cached linkage
            spar = self.exp.spar
            spar.first, spar.last = frames[0], frames[-1]
            run_corrective_pass(cpar, vpar, self.exp.track_par, spar, list(cals), store, max_passes=passes,
                                forward=True, release_ghosts=False, radius_px=1.2, max_ray_miss_mm=0.15,
                                max_pos_err_mm=0.3)

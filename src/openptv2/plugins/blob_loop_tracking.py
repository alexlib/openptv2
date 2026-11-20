"""Tracking plugin: two_phase wrapped in the blob-ownership outer loop.

Registered as ``blob_loop`` (alias ``two_phase_blob_loop``) in
:mod:`openptv2.plugins.loader`. Reads targets and initial correspondences from
the RunStore, runs :func:`openptv2.blob_ownership.run_blob_ownership_loop`,
and writes the refined correspondences and ``ptv_is`` linkage back to the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openptv2.blob_ownership import BlobLoopConfig, run_blob_ownership_loop
from openptv2.plugins.two_phase_tracking import _links_to_linkage


def _cfg_from_track_dict(track_cfg: dict[str, Any]) -> BlobLoopConfig:
    """Extract BlobLoopConfig fields from ``track:`` YAML overrides."""
    defaults = BlobLoopConfig()
    kwargs: dict[str, Any] = {}
    for field_name in defaults.__dataclass_fields__:
        if field_name in track_cfg:
            val = track_cfg[field_name]
            default_val = getattr(defaults, field_name)
            if isinstance(default_val, bool):
                kwargs[field_name] = bool(val)
            elif isinstance(default_val, int):
                kwargs[field_name] = int(val)
            elif isinstance(default_val, float):
                kwargs[field_name] = float(val)
            else:
                kwargs[field_name] = val
    return BlobLoopConfig(**kwargs)


def track_store(
    store: Any,
    frames: list[int],
    track_cfg: dict[str, Any],
    cals: Any = None,
    cpar: Any = None,
    vpar: Any = None,
    *,
    linkage_name: str = "ptv_is",
) -> list[tuple[int, int, int, int]]:
    """Run blob-ownership outer loop on ``store`` over ``frames`` and write
    updated correspondences and linkage back into ``store``."""
    from openptv2.point_quality import read_frame_arrays

    cals_list = list(cals or [])
    if not (cals_list and cpar):
        raise ValueError(
            "blob_loop tracking requires calibrated camera models (cals, cpar)"
        )
    num_cams = len(cals_list)

    frame_pos: list[np.ndarray] = []
    frame_cam_ids: list[np.ndarray] = []
    raw_rows: list[list[np.ndarray | None]] = []
    for f in frames:
        pos_3d, cam_ids, rows = read_frame_arrays(store, f, num_cams)
        frame_pos.append(np.asarray(pos_3d, dtype=np.float64))
        frame_cam_ids.append(np.asarray(cam_ids, dtype=np.int32))
        raw_rows.append(rows)

    cfg = _cfg_from_track_dict(track_cfg)
    refined_pos, refined_cids, links, stats = run_blob_ownership_loop(
        frame_pos,
        frame_cam_ids,
        raw_rows,
        cals_list,
        cpar,
        vpar,
        track_cfg,
        cfg,
    )

    # Write updated correspondences and linkage back into the store
    for f, pos_f, cids_f in zip(frames, refined_pos, refined_cids):
        store.write_correspondences(f, pos_f, cids_f)

    linkage = _links_to_linkage(links, frames, [len(p) for p in refined_pos])
    store.write_linkage_many(frames, linkage, refined_pos, name=linkage_name)

    print(
        f"BlobLoopTracker: {len(links)} links across {len(frames)} frames "
        f"(iters={len(stats['iterations'])})"
    )
    return links


class Tracking:
    """Plugin interface matching :class:`openptv2.plugins.two_phase_tracking.Tracking`."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
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
                raise RuntimeError(
                    "BlobLoopTracker requires a RunStore (zarr) with targets and correspondences."
                )
            store = RunStore(zarr_path, mode="a")

        frames = sorted(store.frames())

        _seq_first = _seq_last = None
        _spar = getattr(self.exp, "spar", None) or getattr(self.exp, "seq_par", None)
        if _spar is not None:
            try:
                _seq_first = int(_spar.get_first())
                _seq_last = int(_spar.get_last())
            except Exception:
                try:
                    _seq_first = int(getattr(_spar, "first"))
                    _seq_last = int(getattr(_spar, "last"))
                except Exception:
                    pass
        if _seq_first is None:
            _seq_cfg = pm.parameters.get("sequence", {}) if pm else {}
            if "first" in _seq_cfg and "last" in _seq_cfg:
                _seq_first, _seq_last = int(_seq_cfg["first"]), int(_seq_cfg["last"])
        if _seq_first is not None and _seq_last is not None:
            frames = [f for f in frames if _seq_first <= f <= _seq_last]

        if len(frames) > 1:
            contiguous = [frames[0]]
            for i in range(1, len(frames)):
                if frames[i] == contiguous[-1] + 1:
                    contiguous.append(frames[i])
                else:
                    break
            frames = contiguous

        if not frames:
            raise RuntimeError("BlobLoopTracker: no frames to process.")

        track_store(
            store,
            frames,
            track_cfg,
            getattr(self.exp, "cals", None),
            getattr(self.exp, "cpar", None),
            getattr(self.exp, "vpar", None),
        )

"""Re-run the correspondence step on detections already in a RunStore.

After a new calibration or new ``criteria:`` the detections need not be found
again: the targets stored in ``res/run.zarr`` are re-matched across cameras and
re-triangulated with the current camera models, without reading any image.
The result goes into a separate store, so the original run stays as it was,
and can then be tracked with
:func:`openptv2.plugins.two_phase_tracking.track_store`.

Every setting comes from the experiment's parameter file (``ptv:`` for the
sensor and media, ``criteria:`` for the observation volume and epipolar band,
``cal_ori:`` for the camera files), through the same loader the batch run uses.

    from openptv2.recorrespond import load_parameters, stored_frames, \\
        correspond_frame, write_store
    cpar, vpar, cals, params = load_parameters("exp/parameters_Run1.yaml")
    src = RunStore("exp/res/run.zarr", mode="r")
    results = {f: correspond_frame(src, f, cpar, cals, vpar)
               for f in stored_frames(src, len(cals), 10001, 10500)}
    write_store(src, RunStore("exp/res/run_new.zarr", mode="w"), results)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_parameters(yaml_path: str | Path) -> tuple[Any, Any, list, dict]:
    """(cpar, vpar, cals, parameters) of an experiment, read from its YAML file.

    Calibration files are resolved against the YAML file's folder, so this
    works from any working directory.
    """
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c

    pm = ParameterManager()
    pm.from_yaml(Path(yaml_path))
    cpar, _spar, vpar, _track_par, _tpar, cals, _epar = py_start_proc_c(pm)
    return cpar, vpar, list(cals), pm.parameters


def stored_frames(
    store: Any, num_cams: int, first: int | None = None, last: int | None = None
) -> list[int]:
    """Frames of ``store`` in [first, last] that have targets in every camera."""
    return [
        f
        for f in sorted(store.frames())
        if (first is None or f >= first)
        and (last is None or f <= last)
        and all(store.has_targets(c, f) for c in range(num_cams))
    ]


def correspond_frame(
    store: Any, frame: int, cpar: Any, cals: list, vpar: Any, min_cams: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(pos, cam_target_ids) of one frame, matched from the stored targets.

    ``min_cams`` > 0 keeps only the points seen by at least that many cameras.
    """
    from openptv2.correspondences import match_frame_correspondences

    det = [store.read_targets(c, frame) for c in range(len(cals))]
    pos, ids = match_frame_correspondences(det, cpar, cals, vpar)
    pos, ids = np.asarray(pos, float), np.asarray(ids, np.int32)
    if min_cams and len(ids):
        keep = (ids >= 0).sum(1) >= min_cams
        pos, ids = pos[keep], ids[keep]
    return pos, ids


def write_store(
    src: Any, dst: Any, results: dict[int, tuple[np.ndarray, np.ndarray]]
) -> None:
    """Write ``results`` ({frame: (pos, cam_target_ids)}) into the RunStore
    ``dst``, together with the targets of those frames copied from ``src``, so
    ``dst`` is a complete store for the tracker."""
    for frame in sorted(results):
        pos, cam_ids = results[frame]
        for c in range(cam_ids.shape[1]):
            key = f"targets/cam_{c}/frame_{frame:06d}"
            if key in src.root:
                dst.write_targets(c, frame, np.asarray(src.root[key]))
        dst.write_correspondences(frame, pos, cam_ids)

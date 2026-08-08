"""Run openptv2 trackers on a generated experiment and collect trajectories.

Drives the real pipeline in-process: builds a
:class:`~openptv2.batch.pyptv_batch.ProcessingExperiment` from the generated
``parameters.yaml`` and invokes a tracking plugin via
:func:`openptv2.plugins.run_tracking_plugin`, then reads the resulting
``ptv_is.#`` linkage files back into ``{track_id: [(frame,x,y,z)]}`` form for
metric evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from openptv2.algorithms.tracking_frame_buf import Frame

# Tracker names recognised as presets by the default_tracking plugin.
_CORE_PRESETS = {
    "fast", "fast_3d", "standard_forward", "two_directional", "full_multipass",
}


def _read_path_info(res_dir: str | Path, first: int, last: int, num_cams: int):
    """Read linkage (ptv_is) frames and return per-frame path arrays.

    Returns
    -------
    list of dicts with keys 'prev' (list) and 'next' (list) aligning to each
    particle slot in that frame, plus 'x' (N,3) positions.
    """
    frames = []
    for fn in range(first, last + 1):
        fpath = Path(res_dir) / f"ptv_is.{fn}"
        if not fpath.exists():
            frames.append(None)
            continue
        with open(fpath) as fh:
            n = int(fh.readline().strip())
            prev = []
            nxt = []
            x = np.empty((n, 3))
            for i in range(n):
                parts = fh.readline().split()
                p, nx = int(parts[0]), int(parts[1])
                prev.append(p if p >= 0 else -1)
                nxt.append(nx if nx >= 0 else -1)
                x[i] = (float(parts[2]), float(parts[3]), float(parts[4]))
        frames.append({"prev": prev, "next": nxt, "x": x})
    return frames


def read_trajectories(
    res_dir: str | Path, first: int, last: int, num_cams: int = 4
) -> dict[int, list[tuple[int, float, float, float]]]:
    """Reconstruct trajectories from ptv_is.# linkage files.

    Follows ``next`` links from each particle that has no predecessor.

    Returns
    -------
    dict[int, list[(frame, x, y, z)]]
        Reconstructed trajectories, one list per assigned track id.
    """
    frames = _read_path_info(res_dir, first, last, num_cams)
    tracks: dict[int, list[tuple[int, float, float, float]]] = {}
    next_id = 0

    for fi in range(len(frames)):
        fr = frames[fi]
        if fr is None:
            continue
        frame_num = first + fi
        for slot in range(len(fr["next"])):
            if fr["prev"][slot] < 0:  # start of a track
                # walk forward
                track_points = [(frame_num, fr["x"][slot][0], fr["x"][slot][1], fr["x"][slot][2])]
                cur_frame = fi
                cur_slot = slot
                while True:
                    nx = frames[cur_frame]["next"][cur_slot]
                    if nx < 0 or cur_frame + 1 >= len(frames):
                        break
                    nxt_fr = frames[cur_frame + 1]
                    if nxt_fr is None:
                        break
                    cur_frame += 1
                    cur_slot = nx
                    if cur_slot >= len(nxt_fr["next"]):
                        break
                    track_points.append(
                        (first + cur_frame, nxt_fr["x"][cur_slot][0],
                         nxt_fr["x"][cur_slot][1], nxt_fr["x"][cur_slot][2])
                    )
                if len(track_points) >= 1:
                    tracks[next_id] = track_points
                    next_id += 1

    return tracks


def run_tracker(
    yaml_path: str | Path,
    tracker: str = "default",
    track_overrides: dict[str, float | bool | int] | None = None,
    postprocess: bool | None = None,
    cwd: str | Path | None = None,
) -> dict[int, list[tuple[int, float, float, float]]]:
    """Run one tracker on a generated experiment and return trajectories.

    Parameters
    ----------
    yaml_path : str | Path
        The ``parameters_Run1.yaml`` written by :func:`write_experiment`.
    tracker : str
        Tracker plugin name / preset (e.g. ``"fast_3d"``, ``"full_multipass"``,
        ``"myptv_3d_tracking"``, ``"proptv_tracking"``).
    track_overrides : dict, optional
        ``dvxmin``..``dvzmax``, ``dacc``, ``angle``/``dangle``,
        ``flagNewParticles``/``add`` overrides applied before running.
    postprocess : bool, optional
        Force the postprocess flag (for ``full_multipass`` presets).
    cwd : str | Path, optional
        Directory to run in (defaults to the YAML's parent; the batch layer
        uses cwd-relative paths).

    Returns
    -------
    dict[int, list[(frame, x, y, z)]]
    """
    yaml_path = Path(yaml_path)
    yaml_dir = yaml_path.parent
    run_dir = Path(cwd) if cwd is not None else yaml_dir

    from openptv2.batch.pyptv_batch import build_processing_experiment
    from openptv2.plugins import run_tracking_plugin

    # Read sequence bounds and camera count from the YAML.
    import yaml as _yaml
    data = _yaml.safe_load(yaml_path.read_text())
    seq_first = int(data["sequence"]["first"])
    seq_last = int(data["sequence"]["last"])
    num_cams = int(data["num_cams"])

    prev_cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        exp = build_processing_experiment(yaml_path, seq_first, seq_last)
    finally:
        os.chdir(prev_cwd)

    # Apply track parameter overrides.
    if track_overrides:
        tp = exp.track_par
        _map = {
            "dvxmin": "set_dvxmin", "dvxmax": "set_dvxmax",
            "dvymin": "set_dvymin", "dvymax": "set_dvymax",
            "dvzmin": "set_dvzmin", "dvzmax": "set_dvzmax",
            "angle": "set_dangle", "dacc": "set_dacc",
        }
        for key, val in track_overrides.items():
            method = _map.get(key)
            if method and hasattr(tp, method):
                getattr(tp, method)(val)
            elif hasattr(tp, key):
                setattr(tp, key, val)
        # Also update the YAML-backed pm for plugin paths that read it.
        try:
            pm_track = exp.pm.parameters.get("track", {})
            for key, val in track_overrides.items():
                if key in ("dvxmin", "dvxmax", "dvymin", "dvymax", "dvzmin", "dvzmax",
                           "dacc", "angle", "flagNewParticles"):
                    pm_track[key] = val
        except Exception:
            pass

    # Force track3d mode for fast_3d preset automatically.
    if tracker in ("fast", "fast_3d"):
        exp.track3d = True

    # Honor the requested preset even when the YAML says "selected_tracking:
    # default" (otherwise default_tracking.infer_preset would silently force
    # fast_3d and ``standard_forward`` would never run the trackcorr path).
    if tracker in _CORE_PRESETS:
        try:
            exp.pm.parameters.setdefault("plugins", {})["selected_tracking"] = tracker
            exp.pm.parameters.setdefault("track", {})
        except Exception:
            pass

    prev_cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        run_tracking_plugin(tracker, exp)
    finally:
        os.chdir(prev_cwd)

    return read_trajectories(Path(run_dir) / "res", seq_first, seq_last, num_cams)


__all__ = ["run_tracker", "read_trajectories"]

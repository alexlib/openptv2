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

from openptv2.tracking_postprocess import link_step

# Tracker names recognised as presets by the default_tracking plugin.
_CORE_PRESETS = {
    "fast", "fast_3d", "priority_segment_3d", "trackcorr",
    "standard_forward", "two_directional", "full_multipass",
    "cython_epipolar_tracking", "openptv_epipolar",
    "4be", "four_be",
}


def _read_path_info(res_dir: str | Path, first: int, last: int, num_cams: int):
    """Read linkage (ptv_is) frames and return per-frame path arrays.

    The RunStore is tried first when it has an entry for a frame --
    py_trackcorr_init's tracking path is store-only now (see
    docs/plans/2026-08-15-zarr-only-transition-plan.md), and ASCII
    ptv_is.<frame> files can be stale scaffolding pre-written by
    write_experiment ("initially unlinked") rather than real tracker output.
    ASCII is the fallback, for callers with no store at all (e.g.
    Quality3DTracker.track_directory, which writes real ptv_is.<frame> with
    store=None).

    Returns
    -------
    list of dicts with keys 'prev' (list) and 'next' (list) aligning to each
    particle slot in that frame, plus 'x' (N,3) positions.
    """
    from openptv2.storage import RunStore, RunStoreError, find_existing_store

    store = None
    try:
        store_path = find_existing_store(res_dir)
        if store_path is not None:
            store = RunStore(store_path, mode="r")
    except RunStoreError:
        store = None

    frames = []
    for fn in range(first, last + 1):
        if store is not None and store.has_linkage(fn, "ptv_is"):
            prev, nxt, x = store.read_linkage(fn, "ptv_is")
            frames.append({
                "prev": [int(p) for p in prev],
                "next": [int(n) for n in nxt],
                "x": np.asarray(x, dtype=np.float64),
            })
            continue

        fpath = Path(res_dir) / f"ptv_is.{fn}"
        if fpath.exists():
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
        else:
            frames.append(None)
    return frames


def _link_step(frames, fi: int, slot: int, nx: int) -> int:
    """Frame step of the forward link ``next[fi][slot] == nx`` (see
    ``tracking_postprocess.link_step``); 0 when nothing reciprocates it."""
    return link_step(
        lambda m: frames[m]["prev"] if 0 <= m < len(frames) and frames[m] else None,
        fi,
        slot,
        nx,
    )


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
                    if nx < 0:
                        break
                    # A gap-bridged link points >1 frame ahead; recover the step
                    # from the reciprocal `prev`. Fall back to 1 so non-reciprocal
                    # links (no postprocess pass) walk as they always did.
                    step = _link_step(frames, cur_frame, cur_slot, nx) or 1
                    if cur_frame + step >= len(frames):
                        break
                    nxt_fr = frames[cur_frame + step]
                    if nxt_fr is None:
                        break
                    cur_frame += step
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
        Tracker plugin name / preset (e.g. ``"priority_segment_3d"``, ``"full_multipass"``,
        ``"nearest_hungarian_3d"``, ``"predictive_gmm_3d"``).
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

    # Read sequence bounds and camera count from the YAML.
    import yaml as _yaml

    from openptv2.batch.pyptv_batch import build_processing_experiment
    from openptv2.plugins import run_tracking_plugin
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
                           "dacc", "angle", "flagNewParticles", "postprocess"):
                    pm_track[key] = val
        except Exception:
            pass

    # Force track3d mode for priority_segment_3d preset automatically.
    if tracker in ("fast", "fast_3d", "priority_segment_3d"):
        exp.track3d = True

    # Honor the requested preset even when the YAML says "selected_tracking:
    # default" (otherwise default_tracking.infer_preset would silently force
    # priority_segment_3d and ``standard_forward`` would never run the trackcorr path).
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

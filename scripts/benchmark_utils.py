"""Shared helpers to run tracker benchmarks and collect tracks.

Used by:
  * scripts/bench_trackers.py         (CLI benchmark; single entry point)
  * notebooks/tracking_dashboard.py   (interactive marimo dashboard)

Defaults to test_data/synthetic_turbulent (220 particles/frame), but every
entry point takes a `src` dataset dir + frame range so the same helpers drive
the density-sweep variants too. Runs each tracker on an isolated copy with
the same tracking parameters and returns, per tracker:
  * predicted trajectories  {track_id: [(frame, x, y, z), ...]}
  * proPTV-style identity metrics + link-level metrics (see combined_metrics)
and the ground-truth trajectories.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import openptv2.benchmarking as bm
from openptv2.benchmarking.metrics import e_track
from openptv2.tracking_metrics import calculate_tracking_metrics

SRC = Path("test_data/synthetic_turbulent")
FIRST = 10001
N_FRAMES = 30
LAST = FIRST + N_FRAMES - 1
TRACKERS = [
    "priority_segment_3d",
    "trackcorr",
    "nearest_hungarian_3d",
    "predictive_gmm_3d",
]

BASE_OVERRIDES = dict(
    dvxmax=6.0, dvxmin=-6.0, dvymax=6.0, dvymin=-6.0, dvzmax=6.0, dvzmin=-6.0, dacc=6.0
)


def has_liboptv() -> bool:
    """Whether the ``optv`` package (compiled Cython bindings to the
    original C liboptv, https://github.com/alexlib/openptv, that this
    project was translated from) is importable."""
    try:
        import optv.tracker  # noqa: F401

        return True
    except ImportError:
        return False


def read_gt_frames(
    src: Path = SRC,
    first: int = FIRST,
    n_frames: int = N_FRAMES,
) -> dict[int, list[tuple[int, float, float, float]]]:
    """Reconstruct per-frame ground truth from origin_*.txt (proPTV-style)."""
    frames: dict[int, list] = {}
    for fn in range(first, first + n_frames):
        p = src / "res" / f"origin_{fn}.txt"
        if not p.exists():
            continue
        rows = []
        for line in p.read_text().strip().splitlines()[1:]:
            parts = line.split(",")
            pid = int(parts[0])
            rows.append((pid, float(parts[1]), float(parts[2]), float(parts[3])))
        frames[fn] = rows
    return frames


def build_true_tracks(
    frames: dict[int, list[tuple[int, float, float, float]]],
    first: int = FIRST,
) -> dict[int, list[tuple[int, float, float, float]]]:
    """Ground-truth tracks {pid: [(frame,x,y,z)]} (frames 0-based)."""
    tt: dict[int, list] = {}
    for fn, rows in frames.items():
        for pid, x, y, z in rows:
            if pid < 0:
                continue
            tt.setdefault(pid, []).append((fn - first, x, y, z))
    return {k: sorted(list(v)) for k, v in tt.items()}


def build_ghost_frames(
    frames: dict[int, list[tuple[int, float, float, float]]],
    first: int = FIRST,
) -> dict[int, np.ndarray]:
    """Ghost (pid < 0) positions per frame (0-based), for
    ``compute_identity_metrics(..., ghost_pos_by_frame=...)``."""
    out: dict[int, np.ndarray] = {}
    for fn, rows in frames.items():
        ghosts = [(x, y, z) for pid, x, y, z in rows if pid < 0]
        if ghosts:
            out[fn - first] = np.array(ghosts)
    return out


def combined_metrics(
    tt: dict[int, list[tuple[int, float, float, float]]],
    pred0: dict[int, list[tuple[int, float, float, float]]],
    eps: float = 1.0,
    ghosts: dict[int, np.ndarray] | None = None,
) -> dict:
    """One flat row merging the three independent metric systems computed
    from the same run: proPTV-style identity metrics (F/C/purity/pmt/ghost
    capture, position-matched), link-level metrics (yield/precision/FCR/
    gap-recovery, matched on both endpoints of a link), and Ouellette's
    track-level ``e_track`` with its failure breakdown. This is a plain dict
    merge -- no new metric is invented here. The only overlapping key is
    ``n_true_tracks``, which both systems define as ``len(tt)``, so which one
    wins the merge does not matter.

    Note ``pmt`` and ``e_track`` are not two views of the same thing and must
    not be substituted for one another: ``pmt`` is computed over PREDICTED
    tracks and rises when a tracker fragments, ``e_track`` is computed over
    TRUE tracks and requires each to be reproduced exactly. ``e_track`` is
    only informative with gap bridging enabled -- see its docstring.
    """
    identity = bm.compute_identity_metrics(
        tt, pred0, eps=eps, ghost_pos_by_frame=ghosts
    )
    link = calculate_tracking_metrics(tt, pred0, distance_tolerance=eps)
    track = e_track(tt, pred0, eps=eps)
    return {**identity.to_dict(), **link.to_dict(), **track.to_dict()}


def _isolate_run_dir(src: Path = SRC) -> tuple[Path, Path]:
    run_dir = Path(tempfile.mkdtemp())
    for sub in ("cal", "res", "img"):
        shutil.copytree(src / sub, run_dir / sub)
    yaml_run = run_dir / "parameters_Run1.yaml"
    shutil.copy(src / "parameters_Run1.yaml", yaml_run)
    return run_dir, yaml_run


# liboptv's compiled optv==0.3.2 Tracker.full_forward() (the trackcorr/2D+3D
# engine) crashes on synthetic_turbulent above trivial density. Root cause,
# confirmed by reading the generated Cython C: optv.tracker.Tracker.__init__
# calls the underlying C tr_new(..., TR_BUFSPACE, MAX_TARGETS, ...) with
# compile-time constants baked into the compiled wheel -- not something a
# caller can configure, and not fixed in 0.3.2, the latest PyPI release.
# full_forward_3d() (fast3d) does not go through this path and is unaffected
# at any density tested (up to 225 particles/frame, full 30-frame range).
#
# The exact crash boundary is NOT a clean function of particle count or
# frame count alone: e.g. (3 particles/frame, 5 frames) reliably succeeds,
# but (3, 8), (4, 5), and (2, 10) all crash. Our density-capping (below)
# truncates each frame to its first N rows independently, which breaks
# temporal coherence between frames (frame-to-frame "particle 0" isn't the
# same physical particle) -- real trajectory data's natural smoothness
# normally keeps liboptv's internal candidate/correspondence buffers small;
# arbitrary, temporally-incoherent points don't get that for free, and
# appear to trigger the same overflow at much lower counts than real data
# would. This is why the burgers fixture (~5 particles/frame, REAL coherent
# trajectories) has always been safe for trackcorr parity testing, while an
# artificially truncated slice of synthetic_turbulent at a similar particle
# count is not reliably safe -- it is capturing the same bug, just via data
# that doesn't have burgers' natural coherence protecting it.
#
# These two values are an empirically verified, but non-general, safe
# operating point -- not a formula. Treat a crash even at these values as
# expected on some frame ranges/datasets, not a bug in this module.
LIBOPTV_TRACKCORR_MAX_PARTICLES = 3
LIBOPTV_TRACKCORR_MAX_FRAMES = 5


def make_density_capped_copy(
    src: Path,
    max_particles: int,
    first: int,
    last: int,
) -> tuple[Path, Path]:
    """Isolated copy of `src` with every frame's rt_is.#/*_targets truncated
    to at most `max_particles`, index-remapped consistently so every
    rt_is<->target cross-reference stays valid. Needed to get liboptv's
    full_forward() to run at all above LIBOPTV_TRACKCORR_MAX_PARTICLES (see
    its docstring) -- both the openptv2 tracker and the liboptv reference
    must run on the SAME capped copy for a trajectory-by-trajectory
    comparison to mean anything.
    """
    run_dir, yaml_run = _isolate_run_dir(src)
    num_cams = len(list((run_dir / "cal").glob("*.ori")))
    for fn in range(first, last + 1):
        rt_path = run_dir / "res" / f"rt_is.{fn}"
        if not rt_path.exists():
            continue
        lines = rt_path.read_text().splitlines()
        n = int(lines[0])
        rows = [line.split() for line in lines[1 : 1 + n]][:max_particles]

        cam_referenced_ids: list[dict[int, None]] = [{} for _ in range(num_cams)]
        for row in rows:
            for c in range(num_cams):
                idx = int(row[4 + c])
                if idx >= 0:
                    cam_referenced_ids[c].setdefault(idx, None)

        cam_id_maps: list[dict[int, int]] = []
        for c in range(num_cams):
            tpath = run_dir / "img" / f"cam{c + 1}.{fn}_targets"
            tlines = tpath.read_text().splitlines()
            old_ids = sorted(cam_referenced_ids[c])
            id_map = {old: new for new, old in enumerate(old_ids)}
            kept = [tlines[1 + old] for old in old_ids]
            tpath.write_text(
                f"{len(kept)}\n" + "\n".join(kept) + ("\n" if kept else "")
            )
            cam_id_maps.append(id_map)

        new_rows = []
        for row in rows:
            new_row = list(row)
            for c in range(num_cams):
                idx = int(row[4 + c])
                new_row[4 + c] = str(cam_id_maps[c][idx]) if idx >= 0 else "-1"
            new_rows.append(" ".join(new_row))
        rt_path.write_text(
            f"{len(new_rows)}\n" + "\n".join(new_rows) + ("\n" if new_rows else "")
        )

    return run_dir, yaml_run


def run_single_tracker(
    tracker: str,
    track_overrides: dict | None = None,
    src: Path = SRC,
    first: int = FIRST,
) -> tuple[dict, float]:
    """Run one tracker, return ({track_id: [(frame,x,y,z)]} 0-based, time_s)."""
    _, yaml_run = _isolate_run_dir(src)
    t0 = time.perf_counter()
    pred = bm.run_tracker(yaml_run, tracker, track_overrides=track_overrides)
    dt = time.perf_counter() - t0
    pred0 = {k: [(f - first, x, y, z) for (f, x, y, z) in v] for k, v in pred.items()}
    return pred0, dt


def run_liboptv_tracker(
    mode: str = "fast3d",
    track_overrides: dict | None = None,
    src: Path = SRC,
    first: int = FIRST,
    n_frames: int = N_FRAMES,
) -> tuple[dict, float]:
    """Run the real liboptv C/Cython tracker (the ``optv`` package -- the
    original openptv, https://github.com/alexlib/openptv, this project was
    translated from) forward-only, on an isolated copy of ``src``.

    ``mode``:
      "fast3d"    -- Tracker.full_forward_3d(), liboptv's counterpart of our
                     priority_segment_3d/nearest_hungarian_3d/
                     predictive_gmm_3d (3D-only linking
                     over already-triangulated rt_is.# points).
      "trackcorr" -- Tracker.full_forward(), liboptv's counterpart of our
                     trackcorr engine (multi-camera 2D+3D epipolar search).

    Same return contract as run_single_tracker, so results drop straight
    into the same comparison tables/metrics.
    """
    from optv.calibration import Calibration as CCalibration
    from optv.parameters import (
        ControlParams,
        SequenceParams,
        TrackingParams,
        VolumeParams,
    )
    from optv.tracker import Tracker as CTracker

    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c

    run_dir, yaml_run = _isolate_run_dir(src)
    overrides = track_overrides or {}
    prev_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        pm = ParameterManager()
        pm.from_yaml(yaml_run.name)
        num_cams = pm.num_cams
        cpar_py, spar_py, vpar_py, track_py, _tpar_py, _cals_py, _epar = (
            py_start_proc_c(pm)
        )

        cpar = ControlParams(num_cams)
        cpar.set_image_size(cpar_py.get_image_size())
        cpar.set_pixel_size(cpar_py.get_pixel_size())
        cpar.set_hp_flag(cpar_py.get_hp_flag())
        cpar.set_allCam_flag(cpar_py.get_allCam_flag())
        cpar.set_tiff_flag(cpar_py.get_tiff_flag())
        cpar.set_chfield(cpar_py.get_chfield())
        mm_py = cpar_py.get_multimedia_params()
        mm = cpar.get_multimedia_params()
        mm.set_n1(mm_py.get_n1())
        mm.set_layers(list(mm_py.get_n2()), list(mm_py.get_d()))
        mm.set_n3(mm_py.get_n3())
        cal_bases = []
        for i in range(num_cams):
            base = cpar_py.get_cal_img_base_name(i)
            cpar.set_cal_img_base_name(i, base)
            cal_bases.append(base)

        spar = SequenceParams(num_cams=num_cams)
        spar.set_first(first)
        spar.set_last(first + n_frames - 1)
        for i, short_name in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(i, str(Path(short_name).resolve()) + ".")

        vpar = VolumeParams()
        vpar.set_X_lay(vpar_py.get_X_lay())
        vpar.set_Zmin_lay(vpar_py.get_Zmin_lay())
        vpar.set_Zmax_lay(vpar_py.get_Zmax_lay())
        vpar.set_eps0(vpar_py.get_eps0())
        vpar.set_cn(vpar_py.get_cn())
        vpar.set_cnx(vpar_py.get_cnx())
        vpar.set_cny(vpar_py.get_cny())
        vpar.set_csumg(vpar_py.get_csumg())
        vpar.set_corrmin(vpar_py.get_corrmin())

        tpar = TrackingParams()
        tpar.set_dvxmin(float(overrides.get("dvxmin", track_py.get_dvxmin())))
        tpar.set_dvxmax(float(overrides.get("dvxmax", track_py.get_dvxmax())))
        tpar.set_dvymin(float(overrides.get("dvymin", track_py.get_dvymin())))
        tpar.set_dvymax(float(overrides.get("dvymax", track_py.get_dvymax())))
        tpar.set_dvzmin(float(overrides.get("dvzmin", track_py.get_dvzmin())))
        tpar.set_dvzmax(float(overrides.get("dvzmax", track_py.get_dvzmax())))
        tpar.set_dangle(float(overrides.get("angle", track_py.get_dangle())))
        tpar.set_dacc(float(overrides.get("dacc", track_py.get_dacc())))
        tpar.set_add(int(overrides.get("flagNewParticles", track_py.get_add())))

        cals = []
        for base in cal_bases:
            cc = CCalibration()
            cc.from_file(f"{base}.ori", f"{base}.addpar")
            cals.append(cc)

        naming = {"corres": "res/rt_is", "linkage": "res/ptv_is", "prio": "res/added"}
        tracker = CTracker(cpar, vpar, tpar, spar, cals, naming)

        t0 = time.perf_counter()
        if mode == "trackcorr":
            tracker.full_forward()
        else:
            tracker.full_forward_3d()
        dt = time.perf_counter() - t0

        tracks = bm.read_trajectories(
            Path("res"), first, first + n_frames - 1, num_cams
        )
    finally:
        os.chdir(prev_cwd)

    pred0 = {k: [(f - first, x, y, z) for (f, x, y, z) in v] for k, v in tracks.items()}
    return pred0, dt


def per_tracker_overrides(
    trackers: list[str],
    src: Path = SRC,
    first: int = FIRST,
    n_frames: int = N_FRAMES,
    base: dict | None = None,
) -> dict[str, dict]:
    """Recommended kinematic-bound overrides per tracker, derived from this
    dataset's own rt_is.# displacement/acceleration statistics via
    openptv2.tracking_recommender -- one shared BASE_OVERRIDES dict applied
    to every tracker hides real quality differences behind parameters that
    were only tuned for one engine (e.g. myptv/proptv use different
    parameter names/scales entirely -- see tracking_registry.py's
    ParameterGuide per tracker).
    """
    from openptv2.algorithms.tracking_frame_buf import Frame
    from openptv2.storage import RunStore
    from openptv2.tracking_recommender import _suggest_params, compute_dataset_stats
    from openptv2.tracking_registry import TRACKER_REGISTRY

    res_dir = Path(src) / "res"
    store = None
    if (res_dir / "run.zarr").exists():
        store = RunStore.open(Path(src), mode="r")

    frame_particles = []
    corres_base = str(res_dir / "rt_is")
    for fn in range(first, first + n_frames):
        if store is not None and store.has_correspondences(fn):
            pos, _ids = store.read_correspondences(fn)
            frame_particles.append(np.asarray(pos, dtype=np.float64))
            continue
        if not (res_dir / f"rt_is.{fn}").exists():
            frame_particles.append(np.empty((0, 3)))
            continue
        frame = Frame(num_cams=4, max_targets=20000)
        frame.read(corres_base, "", target_file_base="", frame_num=fn)
        frame_particles.append(frame.positions())

    stats = compute_dataset_stats(frame_particles)
    out = {}
    for tr in trackers:
        info = TRACKER_REGISTRY.get(tr)
        overrides = dict(base or BASE_OVERRIDES)
        if info is not None:
            overrides.update(_suggest_params(info, stats))
        out[tr] = overrides
    return out


def run_all_trackers(
    trackers: list[str] | None = None,
    track_overrides: dict | None = None,
    silent: bool = True,
    src: Path = SRC,
    first: int = FIRST,
    n_frames: int = N_FRAMES,
) -> dict[str, dict]:
    """Run all trackers on the dataset at ``src``; return {tracker: {...}}.

    Each entry has:
        tracks: {track_id: [(frame,x,y,z)]}
        metrics: IdentityMetrics (F/C/purity/pmt/ghost-capture)
        row: dict merging `metrics` with link-level yield/precision/FCR/
            gap-recovery (see combined_metrics)
        time_s: float
    """
    trackers = trackers or TRACKERS
    overrides = track_overrides or BASE_OVERRIDES
    frames = read_gt_frames(src, first, n_frames)
    tt = build_true_tracks(frames, first)
    ghosts = build_ghost_frames(frames, first)
    results: dict[str, dict] = {}
    for tr in trackers:
        try:
            pred0, dt = run_single_tracker(
                tr, track_overrides=overrides, src=src, first=first
            )
            m = bm.compute_identity_metrics(
                tt, pred0, eps=1.0, ghost_pos_by_frame=ghosts
            )
            row = {
                **m.to_dict(),
                **calculate_tracking_metrics(
                    tt, pred0, distance_tolerance=1.0
                ).to_dict(),
            }
            results[tr] = {"tracks": pred0, "metrics": m, "row": row, "time_s": dt}
        except Exception as e:  # surface error, keep going for other trackers
            results[tr] = {
                "tracks": {},
                "metrics": None,
                "row": None,
                "time_s": 0.0,
                "error": str(e),
            }
        if not silent:
            m = results[tr].get("metrics")
            if m:
                r = results[tr]["row"]
                print(
                    f"{tr:<22} | pmt {m.pmt:5.1f}% | purity {m.purity:.2f} | "
                    f"yield {r['yield_recall']:.2f} | precision {r['precision']:.2f} | "
                    f"ghost {m.ghost_capture_rate:.2%} | {results[tr]['time_s']:.1f}s"
                )
            else:
                print(f"{tr:<22} | ERROR {results[tr].get('error')}")
    return results


def remap_gt_to_tracker_space(tt, gt_ids_to_show=None):
    """Optional: select a subset of ground-truth ids."""
    if gt_ids_to_show is None:
        return tt
    return {k: v for k, v in tt.items() if k in gt_ids_to_show}


def trajectory_shape_stats(
    tracks: dict[int, list[tuple[int, float, float, float]]],
) -> dict:
    """Length and smoothness statistics computed directly from a predicted
    trajectory set -- no ground truth needed, so this always runs even
    where GT identity metrics don't apply (e.g. liboptv's own output).

    Smoothness is the mean angular deviation (degrees) between consecutive
    velocity vectors along each trajectory with >= 3 points: 0 deg is a
    perfectly straight/constant-velocity path, larger values mean sharper
    frame-to-frame direction changes (jitter, or genuinely tracking through
    a spurious neighbour). Only trajectories long enough to have two
    velocity vectors contribute.
    """
    lengths = [len(pts) for pts in tracks.values()]
    if not lengths:
        return {
            "n_tracks": 0,
            "mean_length": 0.0,
            "median_length": 0.0,
            "max_length": 0,
            "min_length": 0,
            "frac_short_lived": 0.0,
            "mean_smoothness_deg": float("nan"),
            "n_smoothness_samples": 0,
        }

    lengths_arr = np.array(lengths, dtype=float)
    short_thresh = 5  # frames; a trajectory this short is a fragment, not a real track
    frac_short = float(np.mean(lengths_arr < short_thresh))

    angle_devs = []
    for pts in tracks.values():
        if len(pts) < 3:
            continue
        xyz = np.array([(x, y, z) for _f, x, y, z in pts])
        vels = np.diff(xyz, axis=0)
        for k in range(len(vels) - 1):
            v1, v2 = vels[k], vels[k + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle_devs.append(np.degrees(np.arccos(cosang)))

    return {
        "n_tracks": len(lengths),
        "mean_length": float(np.mean(lengths_arr)),
        "median_length": float(np.median(lengths_arr)),
        "max_length": int(np.max(lengths_arr)),
        "min_length": int(np.min(lengths_arr)),
        "frac_short_lived": frac_short,
        "mean_smoothness_deg": float(np.mean(angle_devs))
        if angle_devs
        else float("nan"),
        "n_smoothness_samples": len(angle_devs),
    }


__all__ = [
    "SRC",
    "FIRST",
    "LAST",
    "N_FRAMES",
    "TRACKERS",
    "BASE_OVERRIDES",
    "read_gt_frames",
    "build_true_tracks",
    "build_ghost_frames",
    "combined_metrics",
    "run_single_tracker",
    "run_all_trackers",
    "remap_gt_to_tracker_space",
    "trajectory_shape_stats",
]

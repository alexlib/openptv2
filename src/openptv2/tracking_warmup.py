"""Standalone warmup auto-calibration (Stage 1,
docs/plans/2026-08-15-tracking-quality-overhaul.md).

Warmup is a separate step the user runs BEFORE tracking, as many times as
they like -- it is NOT invoked implicitly by the tracker. It measures a
small window of frames [first, first+frames-1] with forward+backward
tracking, uses forward/backward link AGREEMENT as a ground-truth-free
quality signal (works on real data, where no particle-identity ground
truth exists) to tune the velocity/acceleration search box and estimate
positional noise, then picks the better-performing engine between the two
programmatic Tracker methods (track3d via full_forward_3d, trackcorr via
full_forward+full_backward). Results are persisted to the RunStore and
written back into the run's tracking YAML config; the production tracking
run then just reads that config -- it never re-triggers warmup.

Scope cut (deliberate): algorithm selection only compares
priority_segment_3d (track3d) and full_multipass (trackcorr) -- the two
engines directly reachable through Tracker without the full plugin/
experiment machinery. The other TRACKER_REGISTRY entries are plugin-based
and need a constructed experiment object (pm, target files, etc.), which is
a materially bigger integration; add them here if warmup's pick needs to
extend beyond these two engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openptv2.algorithms.parameters import SequencePar, TrackPar
from openptv2.tracker import Tracker
from openptv2.tracking_postprocess import read_linkage
from openptv2.tracking_recommender import compute_dataset_stats, recommend_tracker

_CANDIDATE_ENGINES = ("priority_segment_3d", "full_multipass")


@dataclass
class WarmupResult:
    tracker: str
    track_par: dict[str, float]
    noise_estimate_mm: float
    agreement_rate: float
    frames: tuple[int, int]
    cycles: int
    engine_scores: dict[str, float] = field(default_factory=dict)


def _window_spar(spar: SequencePar, first: int, n_frames: int) -> SequencePar:
    last = min(first + n_frames - 1, spar.last)
    return SequencePar(
        num_cams=spar.num_cams, img_base_name=spar.img_base_name, first=first, last=last
    )


def _track_par_dict(tpar: TrackPar) -> dict[str, float]:
    return {
        "dvxmin": tpar.dvxmin, "dvxmax": tpar.dvxmax,
        "dvymin": tpar.dvymin, "dvymax": tpar.dvymax,
        "dvzmin": tpar.dvzmin, "dvzmax": tpar.dvzmax,
        "dangle": tpar.dangle, "dacc": tpar.dacc, "add": tpar.add,
    }


def _forward_backward_agreement(cpar, vpar, tpar, spar_window, cals, store, linkage_name):
    """Run trackcorr forward+backward on a scratch linkage group, measure
    reciprocity (link agreement -- a ground-truth-free quality signal) and
    the displacement distribution of the CONFIRMED (reciprocal) links, which
    is an empirical positional-noise estimate: a confirmed link's forward
    and backward predictions agreeing means the underlying 3D positions are
    consistent to within the tracker's own measurement noise.

    Returns (agreement_rate, displacements_mm array over confirmed links).
    """
    naming = {"corres": "warmup", "linkage": f"warmup/{linkage_name}", "prio": "warmup"}
    tracker = Tracker(cpar, vpar, tpar, spar_window, cals, naming=naming, store=store)
    tracker.full_forward()
    tracker.full_backward()
    stats = tracker.postprocess(cold_start=False, gap_relinking=False, reciprocity=True)
    links_before = stats["links_before"]
    links_after = stats["links_after"]
    agreement_rate = (links_after / links_before) if links_before else 0.0

    displacements = []
    first, last = spar_window.first, spar_window.last
    for f in range(first, last):
        r0 = read_linkage(naming["linkage"], f, store=store)
        r1 = read_linkage(naming["linkage"], f + 1, store=store)
        if r0 is None or r1 is None:
            continue
        _prev0, nxt0, xyz0 = r0
        _prev1, _nxt1, xyz1 = r1
        for i, j in enumerate(nxt0):
            if j >= 0 and j < len(xyz1):
                displacements.append(float(np.linalg.norm(xyz1[j] - xyz0[i])))
    return agreement_rate, np.asarray(displacements, dtype=np.float64)


def _tune_from_displacements(tpar: TrackPar, displacements: np.ndarray, margin: float = 3.0) -> TrackPar:
    """dv box set from the p99 of confirmed-link displacements (matches
    tracking_recommender._suggest_params's own p95/p99-not-max rationale:
    a single nearest-neighbour mismatch shouldn't set the whole search
    cone). Falls back to the seed tpar unchanged if there's no data yet."""
    if displacements.size < 4:
        return tpar
    half = max(float(np.percentile(displacements, 99)) * margin, 0.05)
    return TrackPar(
        dvxmin=-half, dvxmax=half, dvymin=-half, dvymax=half, dvzmin=-half, dvzmax=half,
        dangle=tpar.dangle, dacc=half, add=tpar.add, track_mode=tpar.track_mode,
    )


def _mean_track_length(linkage_name: str, first: int, last: int, store) -> float:
    """Ground-truth-free trajectory-quality proxy: mean length of the
    trajectories a forward-only run produced, computed straight from
    prev/next chains (no identity/pid info needed, unlike the benchmark
    harness's trajectory_shape_stats which expects an assembled tracks
    dict)."""
    frames = {}
    for f in range(first, last + 1):
        r = read_linkage(linkage_name, f, store=store)
        if r is not None:
            frames[f] = r
    if not frames:
        return 0.0

    lengths = []
    visited = set()
    for f in sorted(frames):
        prev, _nxt, _xyz = frames[f]
        for i in range(len(prev)):
            if (f, i) in visited or prev[i] >= 0:
                continue  # only start counting at a track head
            length = 1
            cf, ci = f, i
            while True:
                visited.add((cf, ci))
                if cf not in frames:
                    break
                _p, nxt, _x = frames[cf]
                if ci >= len(nxt) or nxt[ci] < 0 or (cf + 1) not in frames:
                    break
                cf, ci = cf + 1, int(nxt[ci])
                length += 1
            lengths.append(length)
    return float(np.mean(lengths)) if lengths else 0.0


def run_warmup(
    cpar, vpar, tpar, spar, cals, store,
    frames: int = 25, max_cycles: int = 3, plateau_tol: float = 0.01,
) -> WarmupResult:
    """The warmup loop. Callers (CLI, tests) supply an already-loaded
    experiment's params/calibration and an open RunStore whose
    correspondences (from a prior sequence run) already cover the window.
    """
    spar_window = _window_spar(spar, spar.first, frames)
    first, last = spar_window.first, spar_window.last

    frame_particles = []
    for f in range(first, last + 1):
        try:
            pos, _cam_ids = store.read_correspondences(f)
        except Exception:
            pos = np.empty((0, 3))
        frame_particles.append(pos)
    stats = compute_dataset_stats(frame_particles)
    rec = recommend_tracker(stats)

    cur_tpar = TrackPar(
        dvxmin=rec.suggested_params.get("dvxmin", tpar.dvxmin),
        dvxmax=rec.suggested_params.get("dvxmax", tpar.dvxmax),
        dvymin=rec.suggested_params.get("dvymin", tpar.dvymin),
        dvymax=rec.suggested_params.get("dvymax", tpar.dvymax),
        dvzmin=rec.suggested_params.get("dvzmin", tpar.dvzmin),
        dvzmax=rec.suggested_params.get("dvzmax", tpar.dvzmax),
        dangle=tpar.dangle, dacc=rec.suggested_params.get("dacc", tpar.dacc),
        add=tpar.add, track_mode=tpar.track_mode,
    )

    prev_agreement = -1.0
    agreement = 0.0
    displacements = np.empty(0)
    cycle = 0
    for cycle in range(1, max_cycles + 1):
        agreement, displacements = _forward_backward_agreement(
            cpar, vpar, cur_tpar, spar_window, cals, store, f"cycle{cycle}",
        )
        if agreement - prev_agreement < plateau_tol and cycle > 1:
            break
        prev_agreement = agreement
        cur_tpar = _tune_from_displacements(cur_tpar, displacements)

    noise_estimate = float(np.std(displacements)) if displacements.size else 0.0

    engine_scores: dict[str, float] = {}
    for engine in _CANDIDATE_ENGINES:
        naming = {"corres": "warmup", "linkage": f"warmup/select_{engine}", "prio": "warmup"}
        tracker = Tracker(cpar, vpar, cur_tpar, spar_window, cals, naming=naming, store=store)
        if engine == "priority_segment_3d":
            tracker.full_forward_3d()
        else:
            tracker.full_forward()
        engine_scores[engine] = _mean_track_length(naming["linkage"], first, last, store)
    best_engine = max(engine_scores, key=engine_scores.get)

    result = WarmupResult(
        tracker=best_engine,
        track_par=_track_par_dict(cur_tpar),
        noise_estimate_mm=noise_estimate,
        agreement_rate=agreement,
        frames=(first, last),
        cycles=cycle,
        engine_scores=engine_scores,
    )
    _persist(store, result)
    return result


def _persist(store, result: WarmupResult) -> None:
    """RunStore has no generic key/value stats group (write_stats has a
    fixed tracking-telemetry schema) -- store the warmup result as plain
    JSON-serializable data on the meta group's own attrs, next to
    schema_version/sealed."""
    payload: dict[str, Any] = {
        "tracker": result.tracker,
        "track_par": result.track_par,
        "noise_estimate_mm": result.noise_estimate_mm,
        "agreement_rate": result.agreement_rate,
        "frames": list(result.frames),
        "cycles": result.cycles,
        "engine_scores": result.engine_scores,
    }
    store.root["meta"].attrs["warmup"] = payload


def write_result_to_yaml(result: WarmupResult, yaml_path: str | Path) -> None:
    """Write the chosen tracker + tuned TrackPar back into the run's YAML,
    the way a user's manual edit would -- so a plain `openptv track` run
    afterward picks it up with no warmup-awareness of its own."""
    import yaml

    path = Path(yaml_path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    track = data.setdefault("track", {})
    track["dvxmin"] = result.track_par["dvxmin"]
    track["dvxmax"] = result.track_par["dvxmax"]
    track["dvymin"] = result.track_par["dvymin"]
    track["dvymax"] = result.track_par["dvymax"]
    track["dvzmin"] = result.track_par["dvzmin"]
    track["dvzmax"] = result.track_par["dvzmax"]
    track["angle"] = result.track_par["dangle"]
    track["dacc"] = result.track_par["dacc"]

    plugins = data.setdefault("plugins", {})
    plugins["selected_tracking"] = result.tracker

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


__all__ = ["WarmupResult", "run_warmup", "write_result_to_yaml"]

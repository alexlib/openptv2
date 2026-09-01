"""Tracking quality metrics, including proPTV-style identity metrics.

In addition to the standard yield/precision/RMS set, this module implements
the multi-object-tracking metrics used by proPTV (after Chenouard et al.,
"Objective comparison of particle tracking methods", Nat. Methods 2014):

  * F  — fragmentation: number of distinct reconstructed tracks that match
         one true trajectory (ideal = 1).
  * C  — completeness: fraction of a true trajectory's frames that were
         reconstructed (coverage).
  * Cr — purity / correctness ratio of a fragment: fraction of its points
         that really belong to the true particle (false-link measure).
  * pmt— percentage of correct tracks: a track is correct if at least ~2/3
         of its points map to one and the same true particle (majority ID).

``pmt`` is **not** a track-quality rate, despite the name. It is computed
over *predicted* tracks, and a 2-point fragment satisfies the 2/3 majority
automatically, so a tracker that fragments more scores higher. ``1 - pmt`` is
not an error rate and must not be read as one. For a track-level measure use
:func:`e_track` (Ouellette), which is defined over *true* tracks and requires
each one to be reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from openptv2.plugins._assignment import match_within_radius


@dataclass
class IdentityMetrics:
    """proPTV-style identity-aware tracking metrics."""

    fragmentation: float = 0.0  # mean F over true tracks
    completeness: float = 0.0  # mean C over true tracks
    purity: float = 0.0  # mean Cr over fragments
    pmt: float = 0.0  # percentage of correct tracks
    n_true_tracks: int = 0
    n_reconstructed: int = 0
    n_correct_tracks: int = 0
    # Fraction of predicted points that matched a ghost (spurious detection,
    # pid < 0) rather than a real particle. Only meaningful when
    # ``ghost_pos_by_frame`` was supplied to compute_identity_metrics.
    ghost_capture_rate: float = 0.0
    n_ghost_captures: int = 0

    def to_dict(self) -> dict:
        return {
            "fragmentation": self.fragmentation,
            "completeness": self.completeness,
            "purity": self.purity,
            "pmt": self.pmt,
            "n_true_tracks": self.n_true_tracks,
            "n_reconstructed": self.n_reconstructed,
            "n_correct_tracks": self.n_correct_tracks,
            "ghost_capture_rate": self.ghost_capture_rate,
            "n_ghost_captures": self.n_ghost_captures,
        }


def _frames_dict(
    tracks: Dict[int, List[Tuple[int, float, float, float]]],
) -> Dict[int, Dict[int, np.ndarray]]:
    """Convert {track_id: [(frame,x,y,z)]} to {frame: {track_id: pos}}."""
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for tid, pts in tracks.items():
        for frame, x, y, z in pts:
            out.setdefault(frame, {})[tid] = np.array([x, y, z])
    return out


def _true_frame_lookup(
    true_tracks: Dict[int, List[Tuple[int, float, float, float]]],
) -> Dict[int, Dict[int, np.ndarray]]:
    """{frame: {true_id: pos}}."""
    return _frames_dict(true_tracks)


def _pred_frame_lookup(
    pred_tracks: Dict[int, List[Tuple[int, float, float, float]]],
) -> Dict[int, Dict[int, np.ndarray]]:
    """{frame: {pred_id: pos}}."""
    return _frames_dict(pred_tracks)


def _match_frame(
    true_pos: Dict[int, np.ndarray],
    pred_pos: Dict[int, np.ndarray],
    eps: float,
) -> Dict[int, int]:
    """Within one frame, assign each predicted point to at most one true
    particle within ``eps``, minimising total displacement (one-to-one:
    two predicted points can never claim the same true particle).
    Returns {pred_id: true_id}."""
    if not true_pos or not pred_pos:
        return {}
    true_ids = list(true_pos.keys())
    pred_ids = list(pred_pos.keys())
    true_pts = np.array([true_pos[i] for i in true_ids])
    pred_pts = np.array([pred_pos[i] for i in pred_ids])
    rows, cols = match_within_radius(pred_pts, true_pts, eps)
    return {pred_ids[r]: true_ids[c] for r, c in zip(rows, cols)}


def _ghost_captures_in_frame(
    pred_pos: Dict[int, np.ndarray],
    matched_pred_ids: set,
    ghost_pts: Optional[np.ndarray],
    eps: float,
) -> int:
    """Count predicted points (not already matched to a real particle) that
    land within ``eps`` of a ghost (spurious) detection, one-to-one."""
    if ghost_pts is None or len(ghost_pts) == 0:
        return 0
    remaining_ids = [pid for pid in pred_pos if pid not in matched_pred_ids]
    if not remaining_ids:
        return 0
    remaining_pts = np.array([pred_pos[pid] for pid in remaining_ids])
    rows, _ = match_within_radius(remaining_pts, ghost_pts, eps)
    return len(rows)


def compute_identity_metrics(
    true_tracks: Dict[int, List[Tuple[int, float, float, float]]],
    pred_tracks: Dict[int, List[Tuple[int, float, float, float]]],
    eps: float = 0.10,
    correct_fraction: float = 2.0 / 3.0,
    ghost_pos_by_frame: Optional[Dict[int, np.ndarray]] = None,
) -> IdentityMetrics:
    """Compute proPTV-style identity metrics against ground truth.

    Parameters
    ----------
    true_tracks : dict[int, list[(frame, x, y, z)]]
        Ground-truth trajectories (real particles only, no ghosts).
    pred_tracks : dict[int, list[(frame, x, y, z)]]
        Reconstructed trajectories.
    eps : float
        Spatial matching tolerance.
    correct_fraction : float
        Minimum fraction of a track's points that must map to the same true
        particle for it to be considered "correct" (proPTV uses ~2/3).
    ghost_pos_by_frame : dict[int, (n,3) ndarray], optional
        Positions of spurious (ghost) detections per frame, if known. When
        given, predicted points that match a ghost instead of a real
        particle are reported as ``ghost_capture_rate``.

    Returns
    -------
    IdentityMetrics
    """
    true_frames = _true_frame_lookup(true_tracks)
    pred_frames = _pred_frame_lookup(pred_tracks)
    all_frames = set(true_frames) | set(pred_frames)

    if not true_tracks:
        return IdentityMetrics()

    # Per-frame matching: {frame: {pred_id: true_id}}
    frame_match: Dict[int, Dict[int, int]] = {}
    n_ghost_captures = 0
    n_pred_points = 0
    for frame in all_frames:
        pred_here = pred_frames.get(frame, {})
        frame_match[frame] = _match_frame(true_frames.get(frame, {}), pred_here, eps)
        n_pred_points += len(pred_here)
        ghost_pts = (
            None if ghost_pos_by_frame is None else ghost_pos_by_frame.get(frame)
        )
        n_ghost_captures += _ghost_captures_in_frame(
            pred_here, set(frame_match[frame].keys()), ghost_pts, eps
        )
    ghost_capture_rate = n_ghost_captures / n_pred_points if n_pred_points else 0.0

    # ------------------------------------------------------------------
    # Per-true-track: F (fragmentation), C (completeness)
    # ------------------------------------------------------------------
    frag_scores: List[float] = []
    comp_scores: List[float] = []
    for true_id, pts in true_tracks.items():
        n_true_frames = len(pts)
        covered_frames = set()
        fragments = set()
        for frame, _, _, _ in pts:
            pred_here = frame_match.get(frame, {})
            for pid, tidv in pred_here.items():
                if tidv == true_id:
                    fragments.add(pid)
                    covered_frames.add(frame)
        comp_scores.append(len(covered_frames) / max(1, n_true_frames))
        frag_scores.append(len(fragments))

    # ------------------------------------------------------------------
    # Per-fragment: Cr (purity)
    # ------------------------------------------------------------------
    purities: List[float] = []
    frag_points: Dict[int, List[Tuple[int, int]]] = {}  # pred_id -> [(true_id,count)]
    for frame in frame_match:
        for pid, tidv in frame_match[frame].items():
            frag_points.setdefault(pid, {})
            frag_points[pid][tidv] = frag_points[pid].get(tidv, 0) + 1
    for pid, counts in frag_points.items():
        total = sum(counts.values())
        dominant_true = max(counts, key=counts.get)
        dom_count = counts[dominant_true]
        # purity = fraction of this pred track's matched points belonging to
        # the dominant true particle
        purities.append(dom_count / total)

    # ------------------------------------------------------------------
    # pmt — percentage of correct tracks
    # ------------------------------------------------------------------
    n_correct = 0
    n_total = len(pred_tracks)
    for pid, pts in pred_tracks.items():
        # count matched true ids across this pred track's frames
        counts: Dict[int, int] = {}
        total_frames = len(pts)
        for frame, _, _, _ in pts:
            tidv = frame_match.get(frame, {}).get(pid)
            if tidv is not None:
                counts[tidv] = counts.get(tidv, 0) + 1
        if counts:
            majority = max(counts.values())
            if majority / max(1, total_frames) >= correct_fraction:
                n_correct += 1

    pmt = 100.0 * n_correct / n_total if n_total else 0.0

    return IdentityMetrics(
        fragmentation=float(np.mean(frag_scores)) if frag_scores else 0.0,
        completeness=float(np.mean(comp_scores)) if comp_scores else 0.0,
        purity=float(np.mean(purities)) if purities else 0.0,
        pmt=float(pmt),
        n_true_tracks=len(true_tracks),
        n_reconstructed=n_total,
        n_correct_tracks=n_correct,
        ghost_capture_rate=float(ghost_capture_rate),
        n_ghost_captures=n_ghost_captures,
    )


@dataclass
class TrackErrorMetrics:
    """Ouellette-style track-level error, plus why the failures failed.

    ``e_track`` alone saturates near 1.0 on data whose ground truth contains
    detection gaps, so the breakdown is part of the result rather than an
    optional extra -- without it the metric reports "almost nothing is
    perfect" and gives no way to tell an improving tracker from a stuck one.
    The four failure counts partition the non-perfect true tracks.
    """

    e_track: float = 1.0  # fraction of true tracks NOT reproduced exactly
    n_true_tracks: int = 0
    n_perfect: int = 0
    # Failure breakdown; these four sum to n_true_tracks - n_perfect.
    n_fragmented: int = 0  # covered by >1 predicted track
    n_contaminated: int = 0  # its one fragment also holds foreign/unmatched points
    n_incomplete: int = 0  # clean single fragment, but missing frames
    n_missed: int = 0  # no predicted track matched it at all

    def to_dict(self) -> dict:
        return {
            "e_track": self.e_track,
            "n_true_tracks": self.n_true_tracks,
            "n_perfect": self.n_perfect,
            "n_fragmented": self.n_fragmented,
            "n_contaminated": self.n_contaminated,
            "n_incomplete": self.n_incomplete,
            "n_missed": self.n_missed,
        }


def e_track(
    true_tracks: Dict[int, List[Tuple[int, float, float, float]]],
    pred_tracks: Dict[int, List[Tuple[int, float, float, float]]],
    eps: float,
) -> TrackErrorMetrics:
    """Ouellette's track-level error: the fraction of TRUE trajectories not
    reproduced perfectly.

    Reference: Ouellette, Xu & Bodenschatz, *A quantitative study of
    three-dimensional Lagrangian particle tracking algorithms*, Exp. Fluids
    40:301-313 (2006).

    A true track counts as reproduced perfectly only when exactly one
    predicted track matches it, that predicted track contains **no** other
    points (neither points belonging to another particle nor points matching
    nothing), and it spans **exactly** the true track's frames -- no missing
    points and none added.

    Why all three conditions matter
    -------------------------------
    Dropping the coverage condition -- checking only "every point maps to one
    true particle, and the start frame agrees", which is the form this metric
    was first written in -- makes a 2-point fragment score as a perfect
    reproduction of a 30-point trajectory. That is the same inflation that
    makes ``pmt`` unusable as a track-quality rate (a fragmenting tracker
    scores *better*), and it is not a theoretical worry: on
    ``test_data/synthetic_turbulent`` the loose form ranked a configuration
    producing 3054 tracks for 236 true ones as the best of its sweep, at a
    link yield of 0.55.

    Interpreting the result
    -----------------------
    This is a strict, all-or-nothing measure, so read ``e_track`` together
    with the failure breakdown. In particular, evaluate it with **gap
    bridging enabled**: where ground truth contains detection gaps (91% of
    trajectories in ``synthetic_turbulent`` contain at least one), a tracker
    that cannot bridge a gap can never reproduce those trajectories whole, so
    ``e_track`` pins near 1.0 for every configuration and discriminates
    nothing. The breakdown still does.

    Parameters
    ----------
    true_tracks, pred_tracks : dict[int, list[(frame, x, y, z)]]
    eps : float
        Spatial matching tolerance, in the same units as the coordinates.
        Deliberately has no default -- the sibling
        :func:`compute_identity_metrics` defaults to 0.10 while callers in
        this repo use 1.0, and silently inheriting the wrong one would
        change the result without any visible signal.

    Returns
    -------
    TrackErrorMetrics
    """
    if not true_tracks:
        return TrackErrorMetrics(e_track=1.0, n_true_tracks=0)

    true_frames = _true_frame_lookup(true_tracks)
    pred_frames = _pred_frame_lookup(pred_tracks)

    # {frame: {pred_id: true_id}}, one-to-one within each frame.
    frame_match: Dict[int, Dict[int, int]] = {}
    for frame in set(true_frames) | set(pred_frames):
        frame_match[frame] = _match_frame(
            true_frames.get(frame, {}), pred_frames.get(frame, {}), eps
        )

    # Per predicted track: which true ids it touches, how many of its points
    # matched nothing, and which frames it spans.
    pred_true_ids: Dict[int, set] = {}
    pred_unmatched: Dict[int, int] = {}
    pred_frame_set: Dict[int, set] = {}
    for pid, pts in pred_tracks.items():
        ids = set()
        unmatched = 0
        frames = set()
        for frame, _x, _y, _z in pts:
            frames.add(frame)
            tidv = frame_match.get(frame, {}).get(pid)
            if tidv is None:
                unmatched += 1
            else:
                ids.add(tidv)
        pred_true_ids[pid] = ids
        pred_unmatched[pid] = unmatched
        pred_frame_set[pid] = frames

    # Which predicted tracks touch each true track.
    fragments_of: Dict[int, set] = {tid: set() for tid in true_tracks}
    for pid, ids in pred_true_ids.items():
        for tid in ids:
            if tid in fragments_of:
                fragments_of[tid].add(pid)

    n_perfect = n_fragmented = n_contaminated = n_incomplete = n_missed = 0
    for tid, pts in true_tracks.items():
        frags = fragments_of[tid]
        if not frags:
            n_missed += 1
            continue
        if len(frags) > 1:
            n_fragmented += 1
            continue
        pid = next(iter(frags))
        clean = pred_true_ids[pid] == {tid} and pred_unmatched[pid] == 0
        if not clean:
            n_contaminated += 1
            continue
        if pred_frame_set[pid] != {f for f, _x, _y, _z in pts}:
            n_incomplete += 1
            continue
        n_perfect += 1

    n_true = len(true_tracks)
    return TrackErrorMetrics(
        e_track=1.0 - n_perfect / n_true,
        n_true_tracks=n_true,
        n_perfect=n_perfect,
        n_fragmented=n_fragmented,
        n_contaminated=n_contaminated,
        n_incomplete=n_incomplete,
        n_missed=n_missed,
    )


def ghost_positions_from_frame_gt(
    frame_gt: Dict[int, List[Tuple[int, float, float, float]]],
) -> Dict[int, np.ndarray]:
    """Extract per-frame ghost (pid < 0) positions from a scenario's
    ``frame_gt``, for use as ``compute_identity_metrics``'s
    ``ghost_pos_by_frame``."""
    out: Dict[int, np.ndarray] = {}
    for frame, pts in frame_gt.items():
        ghosts = [(x, y, z) for pid, x, y, z in pts if pid < 0]
        if ghosts:
            out[frame] = np.array(ghosts)
    return out


@dataclass
class PhysicsMetrics:
    """Lagrangian-turbulence physics quality signals (Stage 5 part 1,
    docs/plans/2026-08-15-tracking-quality-overhaul.md), from
    docs/lagrangian_turbulence_quality_guide.md sections A and B. Catch
    ghost contamination that link-level precision/recall can miss: a
    tracker can score well on precision while still corrupting the
    acceleration statistics turbulence research actually needs, because a
    handful of wrong links produce outlier accelerations regardless of how
    rare they are.
    """

    mean_track_length: float = 0.0
    frac_tracks_over_10: float = 0.0
    frac_tracks_over_30: float = 0.0
    n_tracks: int = 0
    acceleration_kurtosis: float = float("nan")
    n_acceleration_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "mean_track_length": self.mean_track_length,
            "frac_tracks_over_10": self.frac_tracks_over_10,
            "frac_tracks_over_30": self.frac_tracks_over_30,
            "n_tracks": self.n_tracks,
            "acceleration_kurtosis": self.acceleration_kurtosis,
            "n_acceleration_samples": self.n_acceleration_samples,
        }


def track_lifetime_distribution(
    tracks: Dict[int, List[Tuple[int, float, float, float]]],
) -> dict:
    """Section A: mean track duration and the fraction of tracks spanning
    more than 10 / 30 frames. Splitting one long physical trajectory into
    many short fragments (a real failure mode -- see
    docs/plans/2026-08-15-tracking-quality-overhaul.md's ghost-inclusive
    benchmark) doesn't show up in precision/recall but destroys these
    fractions."""
    lengths = np.array([len(pts) for pts in tracks.values()], dtype=float)
    if lengths.size == 0:
        return {
            "mean_track_length": 0.0,
            "frac_tracks_over_10": 0.0,
            "frac_tracks_over_30": 0.0,
            "n_tracks": 0,
        }
    return {
        "mean_track_length": float(np.mean(lengths)),
        "frac_tracks_over_10": float(np.mean(lengths > 10)),
        "frac_tracks_over_30": float(np.mean(lengths > 30)),
        "n_tracks": int(lengths.size),
    }


def acceleration_kurtosis(
    tracks: Dict[int, List[Tuple[int, float, float, float]]],
    dt: float = 1.0,
) -> tuple[float, int]:
    """Section B: K_a = <a^4> / <a^2>^2, the flatness factor of the
    acceleration PDF, pooled over every velocity COMPONENT of every track
    with >= 3 points (the standard turbulence-literature statistic, not a
    per-vector-magnitude one -- isotropic turbulence treats x/y/z
    components as statistically equivalent, so pooling them is standard
    practice, not an approximation).

    A single false link produces an acceleration outlier regardless of how
    rare the link error is (a ~= 1/dt^2 jump), which the 4th-moment-driven
    K_a is specifically sensitive to (Gaussian K_a = 3; real turbulence
    K_a ~= 10-50; spurious swaps inflate it further). Returns (K_a, n
    samples) -- NaN when there's not enough data (< 2 acceleration
    samples), matching the guide's "kurtosis needs a real sample" caveat
    rather than reporting a meaningless number from a handful of points.
    """
    accs = []
    for pts in tracks.values():
        if len(pts) < 3:
            continue
        sorted_pts = sorted(pts, key=lambda p: p[0])
        pos = np.array([[x, y, z] for _f, x, y, z in sorted_pts])
        frames = np.array([f for f, _x, _y, _z in sorted_pts])
        # only consecutive-frame triples give a valid finite-difference
        # acceleration -- a gap-bridged jump would fake an outlier.
        consecutive = (frames[2:] - frames[:-2]) == 2
        a = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) / (dt * dt)
        accs.append(a[consecutive])

    if not accs:
        return float("nan"), 0
    a = np.concatenate(accs).ravel()  # pool x, y, z components together
    if a.size < 2:
        return float("nan"), 0
    m2 = float(np.mean(a**2))
    if m2 <= 0:
        return float("nan"), int(a.size)
    m4 = float(np.mean(a**4))
    return m4 / (m2 * m2), int(a.size)


def compute_physics_metrics(
    tracks: Dict[int, List[Tuple[int, float, float, float]]],
    dt: float = 1.0,
) -> PhysicsMetrics:
    """Convenience wrapper bundling track_lifetime_distribution and
    acceleration_kurtosis into one PhysicsMetrics result, matching this
    module's IdentityMetrics/compute_identity_metrics pattern."""
    lifetime = track_lifetime_distribution(tracks)
    k_a, n_a = acceleration_kurtosis(tracks, dt=dt)
    return PhysicsMetrics(
        mean_track_length=lifetime["mean_track_length"],
        frac_tracks_over_10=lifetime["frac_tracks_over_10"],
        frac_tracks_over_30=lifetime["frac_tracks_over_30"],
        n_tracks=lifetime["n_tracks"],
        acceleration_kurtosis=k_a,
        n_acceleration_samples=n_a,
    )


__all__ = [
    "IdentityMetrics",
    "PhysicsMetrics",
    "TrackErrorMetrics",
    "compute_identity_metrics",
    "e_track",
    "ghost_positions_from_frame_gt",
    "track_lifetime_distribution",
    "acceleration_kurtosis",
    "compute_physics_metrics",
]

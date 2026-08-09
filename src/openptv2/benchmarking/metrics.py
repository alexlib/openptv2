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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from openptv2.plugins._assignment import match_within_radius


@dataclass
class IdentityMetrics:
    """proPTV-style identity-aware tracking metrics."""

    fragmentation: float = 0.0          # mean F over true tracks
    completeness: float = 0.0           # mean C over true tracks
    purity: float = 0.0                 # mean Cr over fragments
    pmt: float = 0.0                    # percentage of correct tracks
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
        ghost_pts = None if ghost_pos_by_frame is None else ghost_pos_by_frame.get(frame)
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


__all__ = [
    "IdentityMetrics",
    "compute_identity_metrics",
    "ghost_positions_from_frame_gt",
]

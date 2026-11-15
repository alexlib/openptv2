"""Post-run self-consistency report for RunStore linkage (Milestone D1-D2).

Computes trajectory self-consistency metrics directly from a ``RunStore``
linkage stream (default ``"ptv_is"``), formats a human-readable summary,
flags common tracking anomalies with suggested fixes, and optionally updates
a :class:`~openptv2.tracking_decision.TrackingDecision` record.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Mapping, TextIO

import numpy as np

from openptv2.tracking_postprocess import MAX_LINK_STEP, back_link_step, link_step

if TYPE_CHECKING:
    from openptv2.storage.run_store import RunStore
    from openptv2.tracking_decision import TrackingDecision

__all__ = [
    "RECIPROCITY_WARN_THRESHOLD",
    "MEDIAN_LENGTH_WARN_THRESHOLD",
    "JUMP_RATE_WARN_THRESHOLD",
    "GHOST_RATE_WARN_THRESHOLD",
    "compute_self_consistency_metrics",
    "format_report",
    "generate_report",
]

RECIPROCITY_WARN_THRESHOLD = 0.7
MEDIAN_LENGTH_WARN_THRESHOLD = 4.0
JUMP_RATE_WARN_THRESHOLD = 0.1
GHOST_RATE_WARN_THRESHOLD = 0.4


def _load_linkage_frames(
    store: "RunStore",
    first: int | None,
    last: int | None,
    linkage_name: str,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], int | None, int | None]:
    available = store.linkage_frames(linkage_name)
    if not available:
        return {}, first, last

    lo = available[0] if first is None else int(first)
    hi = available[-1] if last is None else int(last)

    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in range(lo, hi + 1):
        if store.has_linkage(k, linkage_name):
            prev, nxt, xyz = store.read_linkage(k, linkage_name)
            frames[k] = (
                np.asarray(prev, dtype=np.int32),
                np.asarray(nxt, dtype=np.int32),
                np.asarray(xyz, dtype=np.float64),
            )
    return frames, lo, hi


def _is_velocity_jump(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    dt1: int,
    dt2: int,
    jump_angle_deg: float,
    jump_ratio: float,
) -> bool:
    v1 = (p1 - p0) / float(dt1)
    v2 = (p2 - p1) / float(dt2)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    mean_speed = 0.5 * (n1 + n2)
    if mean_speed <= 1e-9:
        return False

    rel_diff = float(np.linalg.norm(v2 - v1)) / mean_speed
    if rel_diff > jump_ratio:
        return True

    if n1 > 1e-9 and n2 > 1e-9:
        cos_theta = float(np.dot(v1, v2)) / (n1 * n2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = math.degrees(math.acos(cos_theta))
        if angle_deg > jump_angle_deg:
            return True
    return False


def _evaluate_anomalies(
    *,
    reciprocity_rate: float,
    track_length_stats: Mapping[str, float | int],
    jump_step_rate: float,
    estimated_ghost_rate: float,
    total_particles: int,
    total_forward_links: int,
) -> list[str]:
    flags: list[str] = []
    if (total_forward_links > 0 or total_particles > 0) and (
        reciprocity_rate < RECIPROCITY_WARN_THRESHOLD
    ):
        flags.append(
            f"Low reciprocity rate ({reciprocity_rate:.2f} < {RECIPROCITY_WARN_THRESHOLD:.2f}): "
            "forward and backward passes disagree. "
            "Suggested fix: tighten v_max/dv_max, enable confirm_link or enforce_reciprocity, "
            "or reduce particle density."
        )

    median_len = float(track_length_stats["median"])
    if total_particles > 0 and median_len < MEDIAN_LENGTH_WARN_THRESHOLD:
        flags.append(
            f"Short median track length ({median_len:.1f} < {MEDIAN_LENGTH_WARN_THRESHOLD:.0f}): "
            "trajectories are fragmented. "
            "Suggested fix: increase v_max or max_gap / relink_trajectory_gaps, or check calibration."
        )

    if jump_step_rate > JUMP_RATE_WARN_THRESHOLD:
        flags.append(
            f"High jump-step rate ({jump_step_rate:.2f} > {JUMP_RATE_WARN_THRESHOLD:.2f}): "
            "many 2-step transitions show velocity kinks or sudden speed jumps. "
            "Suggested fix: tighten dv_max / velocity smoothness or enable confirm_link."
        )

    if estimated_ghost_rate > GHOST_RATE_WARN_THRESHOLD:
        flags.append(
            f"High estimated ghost rate ({estimated_ghost_rate:.2f} > {GHOST_RATE_WARN_THRESHOLD:.2f}): "
            "many detections have neither prev nor next links. "
            "Suggested fix: tighten stereo correspondence thresholds or increase v_max if valid particles are unlinked."
        )
    return flags


def compute_self_consistency_metrics(
    store: "RunStore",
    first: int | None = None,
    last: int | None = None,
    *,
    linkage_name: str = "ptv_is",
    max_step: int = MAX_LINK_STEP,
    jump_angle_deg: float = 90.0,
    jump_ratio: float = 1.0,
) -> dict[str, Any]:
    """Compute post-run self-consistency metrics from ``RunStore`` linkage.

    Parameters
    ----------
    store:
        Open ``RunStore`` containing linkage under ``linkage_name``.
    first, last:
        Inclusive frame range to inspect. Defaults to the full range present
        in ``store.linkage_frames(linkage_name)``.
    linkage_name:
        Linkage stream name (default ``"ptv_is"``).
    max_step:
        Maximum frame step searched when verifying reciprocal links.
    jump_angle_deg:
        Velocity turning-angle threshold (degrees) for counting a jump step.
    jump_ratio:
        Relative velocity-difference threshold ``|v2 - v1| / mean(|v1|, |v2|)``
        for counting a jump step.
    """
    frames, lo, hi = _load_linkage_frames(store, first, last, linkage_name)
    if not frames or lo is None or hi is None:
        empty_stats = {
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "max": 0,
            "count": 0,
        }
        empty_gaps = {
            "step_1": 0,
            "gap_1": 0,
            "gap_2": 0,
            "gap_gt_2": 0,
            "unlinked_starts": 0,
            "unlinked_ends": 0,
            "broken_forward": 0,
            "broken_backward": 0,
        }
        return {
            "first": lo,
            "last": hi,
            "n_frames": 0,
            "total_particles": 0,
            "reciprocity_rate": 0.0,
            "track_length_stats": empty_stats,
            "jump_step_rate": 0.0,
            "gap_distribution": empty_gaps,
            "estimated_ghost_rate": 0.0,
            "anomaly_flags": [],
        }

    def prev_of(frame: int) -> np.ndarray | None:
        entry = frames.get(frame)
        return entry[0] if entry is not None else None

    def next_of(frame: int) -> np.ndarray | None:
        entry = frames.get(frame)
        return entry[1] if entry is not None else None

    total_particles = 0
    isolated_particles = 0
    total_forward_links = 0
    reciprocal_forward_links = 0
    broken_forward = 0
    broken_backward = 0
    unlinked_starts = 0
    unlinked_ends = 0
    step_1 = 0
    gap_1 = 0
    gap_2 = 0
    gap_gt_2 = 0

    # Map (k, i) -> (k + s, j) for reciprocal forward links.
    reciprocal_next: dict[tuple[int, int], tuple[int, int]] = {}
    has_reciprocal_prev: set[tuple[int, int]] = set()

    for k in sorted(frames):
        prv, nxt, _xyz = frames[k]
        n = len(prv)
        total_particles += n

        isolated_particles += int(np.sum((prv < 0) & (nxt < 0)))
        if k > lo:
            unlinked_starts += int(np.sum((prv < 0) & (nxt >= 0)))
        if k < hi:
            unlinked_ends += int(np.sum((nxt < 0) & (prv >= 0)))

        if k < hi:
            for i in range(n):
                j = int(nxt[i])
                if j < 0:
                    continue
                total_forward_links += 1
                s = link_step(prev_of, k, i, j, max_step=max_step)
                if s > 0:
                    reciprocal_forward_links += 1
                    reciprocal_next[(k, i)] = (k + s, j)
                    has_reciprocal_prev.add((k + s, j))
                    if s == 1:
                        step_1 += 1
                    elif s == 2:
                        gap_1 += 1
                    elif s == 3:
                        gap_2 += 1
                    else:
                        gap_gt_2 += 1
                else:
                    broken_forward += 1

        if k > lo:
            for j in range(n):
                i = int(prv[j])
                if i < 0:
                    continue
                if back_link_step(next_of, k, j, i, max_step=max_step) == 0:
                    broken_backward += 1

    reciprocity_rate = (
        float(reciprocal_forward_links) / float(total_forward_links)
        if total_forward_links > 0
        else 0.0
    )
    estimated_ghost_rate = (
        float(isolated_particles) / float(total_particles)
        if total_particles > 0
        else 0.0
    )

    # Walk reciprocal tracks to compute track lengths and 2-step velocity jumps.
    visited: set[tuple[int, int]] = set()
    track_lengths: list[int] = []
    two_step_transitions = 0
    jump_transitions = 0

    for k in sorted(frames):
        n = len(frames[k][0])
        for i in range(n):
            start = (k, i)
            if start in has_reciprocal_prev or start in visited:
                continue
            if start not in reciprocal_next:
                continue
            chain = [start]
            visited.add(start)
            cur = start
            while cur in reciprocal_next:
                nxt_node = reciprocal_next[cur]
                if nxt_node in visited:
                    break
                visited.add(nxt_node)
                chain.append(nxt_node)
                cur = nxt_node
            if len(chain) >= 2:
                track_lengths.append(len(chain))
                for idx in range(len(chain) - 2):
                    k0, i0 = chain[idx]
                    k1, i1 = chain[idx + 1]
                    k2, i2 = chain[idx + 2]
                    p0 = frames[k0][2][i0]
                    p1 = frames[k1][2][i1]
                    p2 = frames[k2][2][i2]
                    two_step_transitions += 1
                    if _is_velocity_jump(
                        p0,
                        p1,
                        p2,
                        k1 - k0,
                        k2 - k1,
                        jump_angle_deg=jump_angle_deg,
                        jump_ratio=jump_ratio,
                    ):
                        jump_transitions += 1

    if track_lengths:
        arr = np.asarray(track_lengths, dtype=np.float64)
        track_length_stats: dict[str, float | int] = {
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "max": int(np.max(arr)),
            "count": int(len(track_lengths)),
        }
    else:
        track_length_stats = {
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "max": 0,
            "count": 0,
        }

    jump_step_rate = (
        float(jump_transitions) / float(two_step_transitions)
        if two_step_transitions > 0
        else 0.0
    )

    gap_distribution = {
        "step_1": int(step_1),
        "gap_1": int(gap_1),
        "gap_2": int(gap_2),
        "gap_gt_2": int(gap_gt_2),
        "unlinked_starts": int(unlinked_starts),
        "unlinked_ends": int(unlinked_ends),
        "broken_forward": int(broken_forward),
        "broken_backward": int(broken_backward),
    }

    anomaly_flags = _evaluate_anomalies(
        reciprocity_rate=reciprocity_rate,
        track_length_stats=track_length_stats,
        jump_step_rate=jump_step_rate,
        estimated_ghost_rate=estimated_ghost_rate,
        total_particles=total_particles,
        total_forward_links=total_forward_links,
    )

    return {
        "first": int(lo),
        "last": int(hi),
        "n_frames": int(len(frames)),
        "total_particles": int(total_particles),
        "reciprocity_rate": float(reciprocity_rate),
        "track_length_stats": track_length_stats,
        "jump_step_rate": float(jump_step_rate),
        "gap_distribution": gap_distribution,
        "estimated_ghost_rate": float(estimated_ghost_rate),
        "anomaly_flags": anomaly_flags,
    }


def format_report(metrics: Mapping[str, Any]) -> str:
    """Format self-consistency metrics as a human-readable report string."""
    tls = metrics["track_length_stats"]
    gaps = metrics["gap_distribution"]
    lines = [
        "=== Post-Run Tracking Self-Consistency Report ===",
        f"Frames: {metrics.get('first')}..{metrics.get('last')} "
        f"({metrics.get('n_frames', 0)} frames, {metrics.get('total_particles', 0)} particles)",
        f"Reciprocity rate:     {metrics['reciprocity_rate']:.3f}",
        f"Track length (n={tls['count']}): "
        f"median={tls['median']:.1f}, p10={tls['p10']:.1f}, "
        f"p90={tls['p90']:.1f}, max={tls['max']}",
        f"Jump-step rate:       {metrics['jump_step_rate']:.3f}",
        f"Estimated ghost rate: {metrics['estimated_ghost_rate']:.3f}",
        f"Gaps / ends:          step_1={gaps['step_1']}, gap_1={gaps['gap_1']}, "
        f"gap_2={gaps['gap_2']}, unlinked_starts={gaps['unlinked_starts']}, "
        f"unlinked_ends={gaps['unlinked_ends']}, broken_fwd={gaps['broken_forward']}, "
        f"broken_bwd={gaps['broken_backward']}",
    ]
    flags = metrics.get("anomaly_flags") or []
    if flags:
        lines.append("Anomaly warnings:")
        for flag in flags:
            lines.append(f"  - {flag}")
    else:
        lines.append("Anomaly warnings:     none")
    return "\n".join(lines)


def generate_report(
    store: "RunStore",
    first: int | None = None,
    last: int | None = None,
    *,
    linkage_name: str = "ptv_is",
    max_step: int = MAX_LINK_STEP,
    jump_angle_deg: float = 90.0,
    jump_ratio: float = 1.0,
    print_report: bool = True,
    decision: "TrackingDecision | None" = None,
    save_decision: bool = True,
    stream: TextIO | None = None,
) -> dict[str, Any]:
    """Compute, optionally print, and optionally persist the post-run report.

    When ``decision`` is provided, sets ``decision.report_metrics`` to the
    computed metrics dict and persists it via ``decision.save(store)`` if
    ``save_decision`` is True.
    """
    metrics = compute_self_consistency_metrics(
        store,
        first,
        last,
        linkage_name=linkage_name,
        max_step=max_step,
        jump_angle_deg=jump_angle_deg,
        jump_ratio=jump_ratio,
    )
    if print_report:
        print(format_report(metrics), file=stream or sys.stdout)

    if decision is not None:
        decision.report_metrics = dict(metrics)
        if save_decision:
            decision.save(store)

    return metrics


def compute_trajectory_quality_scores(
    store: "RunStore",
    first: int | None = None,
    last: int | None = None,
    *,
    linkage_name: str = "ptv_is",
    max_step: int = MAX_LINK_STEP,
    jump_angle_deg: float = 90.0,
    jump_ratio: float = 1.0,
    write_prio: bool = False,
) -> dict[tuple[int, int], float]:
    """Score each particle's trajectory in ``[0.0, 1.0]`` and optionally write quantized ``prio``.

    Scores combine trajectory length, bidirectional reciprocity, kinematic
    smoothness (absence of velocity-jump kinks), and gap count. Isolated
    particles receive ``0.0``. When ``write_prio=True``, quantized ``int32``
    priorities (``round(score * 1000)`` in ``0..1000``) are persisted to
    ``store`` via ``write_linkage(..., prio=...)``.
    """
    frames, lo, hi = _load_linkage_frames(store, first, last, linkage_name)
    if not frames or lo is None or hi is None:
        return {}

    def prev_of(frame: int) -> np.ndarray | None:
        entry = frames.get(frame)
        return entry[0] if entry is not None else None

    fwd_next: dict[tuple[int, int], tuple[int, int, int, bool]] = {}
    has_prev: set[tuple[int, int]] = set()
    all_nodes: list[tuple[int, int]] = []

    for k in sorted(frames):
        _prv, nxt, xyz = frames[k]
        for i in range(len(xyz)):
            node = (k, i)
            all_nodes.append(node)
            j = int(nxt[i])
            if j < 0 or k >= hi:
                continue
            s = link_step(prev_of, k, i, j, max_step=max_step)
            if s > 0:
                target = (k + s, j)
                fwd_next[node] = (k + s, j, s, True)
                has_prev.add(target)
            elif k + 1 in frames and 0 <= j < len(frames[k + 1][0]):
                target = (k + 1, j)
                fwd_next[node] = (k + 1, j, 1, False)
                has_prev.add(target)

    scores: dict[tuple[int, int], float] = {}
    visited: set[tuple[int, int]] = set()

    def _score_chain(chain: list[tuple[int, int]], edges: list[tuple[int, bool]]) -> float:
        if len(chain) < 2:
            return 0.0
        n_edges = len(edges)
        length_score = min(1.0, float(n_edges) / 5.0)
        recip_frac = float(sum(1 for _, is_rec in edges if is_rec)) / float(n_edges)
        n_gaps = sum(1 for step, _ in edges if step > 1)
        gap_penalty = min(0.4, 0.1 * float(n_gaps) / float(n_edges))

        if len(chain) < 3:
            smoothness = 0.5
        else:
            kinks = 0
            two_steps = len(chain) - 2
            for idx in range(two_steps):
                k0, i0 = chain[idx]
                k1, i1 = chain[idx + 1]
                k2, i2 = chain[idx + 2]
                if _is_velocity_jump(
                    frames[k0][2][i0],
                    frames[k1][2][i1],
                    frames[k2][2][i2],
                    k1 - k0,
                    k2 - k1,
                    jump_angle_deg=jump_angle_deg,
                    jump_ratio=jump_ratio,
                ):
                    kinks += 1
            smoothness = 1.0 - float(kinks) / float(two_steps)

        raw = 0.40 * length_score + 0.30 * recip_frac + 0.30 * smoothness - gap_penalty
        return float(np.clip(raw, 0.0, 1.0))

    # Walk chains from starts first, then any remaining nodes (e.g. cycles/merges).
    ordered_starts = [n for n in all_nodes if n not in has_prev] + all_nodes
    for start in ordered_starts:
        if start in visited:
            continue
        chain = [start]
        edges: list[tuple[int, bool]] = []
        visited.add(start)
        cur = start
        while cur in fwd_next:
            nk, nj, step, is_rec = fwd_next[cur]
            nxt_node = (nk, nj)
            edges.append((step, is_rec))
            if nxt_node in visited:
                break
            visited.add(nxt_node)
            chain.append(nxt_node)
            cur = nxt_node
        chain_score = _score_chain(chain, edges)
        for node in chain:
            scores[node] = chain_score

    if write_prio:
        for k in sorted(frames):
            prv, nxt, xyz = frames[k]
            prio = np.array(
                [int(round(scores.get((k, i), 0.0) * 1000.0)) for i in range(len(xyz))],
                dtype=np.int32,
            )
            store.write_linkage(k, prv, nxt, xyz, name=linkage_name, prio=prio)

    return scores

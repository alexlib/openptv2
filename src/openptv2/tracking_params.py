"""Recommend trackcorr parameters from measured track statistics.

The recipe (validated in openptv-analysis, logs 023-026, against synthetic
ground truth matched to a 4-camera experiment):

1. Track a short stretch (e.g. the first 20 frames) with LOOSE limits --
   angle off (270 gon), a generous ``dacc`` -- so the statistics are not
   shaped by the limits being chosen.
2. From links inside tracks of at least ``min_len`` points measure:

   * the position noise per axis, from the lag-1 covariance of the second
     difference ``a = x[i+1] - 2 x[i] + x[i-1]``: for white noise it is
     ``-4 sigma^2`` (the physical acceleration of tracers at kHz frame rates is
     negligible next to it);
   * the step per axis (99.9th percentile) and the median step length.

3. Derive:

   * ``dv`` per axis = p99.9 of |step| + 3 * sqrt(2) * sigma, rounded up;
   * ``dacc`` = ``dacc_factor`` * the ``dacc_quantile`` percentile of the
     magnitude of pure-noise acceleration (per-axis std sqrt(6) * sigma);
   * ``angle`` = off (270 gon) when the median step is less than
     ``angle_ratio`` times the step noise sqrt(2) * |sigma| -- the turning
     angle between two steps is then set by noise, and an angle limit only
     rejects correct links (on the wp1 data the 100 gon limit cost 6 points of
     link recall on synthetic truth and shortened trajectories 2-3x).

2-camera correspondences are a separate decision: see ``ptv.pair_flag``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "noise_sigma",
    "track_statistics",
    "recommend_trackcorr_params",
    "tracks_from_store",
    "recommend_from_store",
]


def _segment_lengths(tid):
    start = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    lens = np.diff(np.r_[start, len(tid)])
    return np.repeat(lens, lens)


def _sorted(tid, frame, pos):
    tid, frame = np.asarray(tid), np.asarray(frame)
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    order = np.lexsort((frame, tid))
    return tid[order], frame[order], pos[order]


def noise_sigma(tid, frame, pos, min_len: int = 5) -> np.ndarray:
    """Per-axis position noise from the lag-1 covariance of second differences.

    Only consecutive-frame triplets inside tracks of ``min_len`` points or more
    count; second differences beyond 6 robust standard deviations (wrong links)
    are dropped before the covariance is taken."""
    tid, frame, pos = _sorted(tid, frame, pos)
    same = (tid[1:] == tid[:-1]) & (frame[1:] == frame[:-1] + 1)
    long_ = _segment_lengths(tid) >= min_len
    a = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]
    ok = same[1:] & same[:-1] & long_[1:-1]
    pair = ok[1:] & ok[:-1]
    sig = np.zeros(3)
    for ax in range(3):
        x = a[:, ax]
        if ok.sum() < 10:
            continue
        s = 1.4826 * np.median(np.abs(x[ok] - np.median(x[ok])))
        good = ok & (np.abs(x) <= 6.0 * s)
        p = pair & good[1:] & good[:-1]
        if p.sum() < 10:
            continue
        c1 = float((x[1:] * x[:-1])[p].mean())
        sig[ax] = np.sqrt(max(-c1 / 4.0, 0.0))
    return sig


def track_statistics(tid, frame, pos, min_len: int = 5) -> dict:
    """Noise and step statistics of linked tracks (positions in any length unit)."""
    tid, frame, pos = _sorted(tid, frame, pos)
    same = (tid[1:] == tid[:-1]) & (frame[1:] == frame[:-1] + 1)
    same &= _segment_lengths(tid)[:-1] >= min_len
    step = (pos[1:] - pos[:-1])[same]
    if len(step) == 0:
        raise ValueError(f"no links inside tracks of >= {min_len} points to measure")
    sig = noise_sigma(tid, frame, pos, min_len=min_len)
    return {
        "n_links": int(len(step)),
        "noise_sigma": sig,
        "step_abs_p999": np.percentile(np.abs(step), 99.9, axis=0),
        "step_p50": float(np.median(np.linalg.norm(step, axis=1))),
        "step_noise": float(np.sqrt(2.0) * np.linalg.norm(sig)),
    }


def recommend_trackcorr_params(
    stats: dict,
    dacc_factor: float = 1.2,
    dacc_quantile: float = 99.5,
    angle_ratio: float = 10.0,
    dv_round: float = 0.1,
    seed: int = 0,
) -> tuple[dict, list[str]]:
    """``track`` parameters and a list of plain-language reasons.

    Returns ``({"dvxmin", "dvxmax", ..., "dacc", "angle"}, reasons)``."""
    sig = np.asarray(stats["noise_sigma"], dtype=float)
    params: dict = {}
    reasons = []
    for ax, name in enumerate("xyz"):
        lim = float(stats["step_abs_p999"][ax] + 3.0 * np.sqrt(2.0) * sig[ax])
        lim = float(np.ceil(lim / dv_round) * dv_round)
        params[f"dv{name}min"], params[f"dv{name}max"] = -lim, lim
    reasons.append(
        "dv per axis = p99.9 |step| + 3*sqrt(2)*sigma: "
        + ", ".join(f"{n} +-{params[f'dv{n}max']:g}" for n in "xyz")
    )
    rng = np.random.default_rng(seed)
    acc_noise = np.linalg.norm(rng.normal(0.0, 1.0, (200_000, 3)) * np.sqrt(6.0) * sig, axis=1)
    q = float(np.percentile(acc_noise, dacc_quantile))
    params["dacc"] = float(np.ceil(dacc_factor * q / (dv_round / 2)) * (dv_round / 2))
    reasons.append(
        f"dacc = {dacc_factor} x p{dacc_quantile:g} of noise acceleration ({q:.4g}) -> {params['dacc']:g}"
    )
    ratio = stats["step_p50"] / stats["step_noise"] if stats["step_noise"] > 0 else np.inf
    if ratio < angle_ratio:
        params["angle"] = 270.0
        reasons.append(
            f"median step / step noise = {ratio:.2g} < {angle_ratio:g}: the turning angle is "
            "set by noise -> angle limit off (270 gon)"
        )
    else:
        params["angle"] = 200.0
        reasons.append(
            f"median step / step noise = {ratio:.2g} >= {angle_ratio:g}: steps dominate noise; "
            "angle limited to 200 gon (tighten only with evidence of wrong links)"
        )
    return params, reasons


def tracks_from_store(store, first: int, last: int, name: str = "ptv_is"):
    """(tid, frame, pos) of every particle in frames first..last, with track ids
    following the ``prev`` pointers (a pointer claimed twice keeps its first
    claimant)."""
    tids, frames, positions = [], [], []
    prev_tid, next_id = None, 0
    for f in range(first, last + 1):
        if not store.has_linkage(f, name):
            prev_tid = None
            continue
        prev, _nxt, xyz = store.read_linkage(f, name)
        prev = np.asarray(prev, dtype=np.int64)
        n = len(prev)
        tid = np.empty(n, dtype=np.int64)
        linked = np.zeros(n, dtype=bool)
        if prev_tid is not None:
            linked = (prev >= 0) & (prev < len(prev_tid))
            first_claim = np.zeros(n, dtype=bool)
            first_claim[np.unique(np.where(linked, prev, -1 - np.arange(n)), return_index=True)[1]] = True
            linked &= first_claim
            tid[linked] = prev_tid[prev[linked]]
        tid[~linked] = np.arange(next_id, next_id + int((~linked).sum()))
        next_id += int((~linked).sum())
        prev_tid = tid
        tids.append(tid)
        frames.append(np.full(n, f))
        positions.append(np.asarray(xyz, dtype=float).reshape(-1, 3))
    if not tids:
        raise ValueError(f"no linkage '{name}' in frames {first}-{last}")
    return np.concatenate(tids), np.concatenate(frames), np.concatenate(positions)


def recommend_from_store(store, first: int, last: int, name: str = "ptv_is", min_len: int = 5, **kwargs):
    """Statistics and recommended ``track`` parameters from a (loosely) tracked store.

    Returns ``(params, stats, reasons)``."""
    stats = track_statistics(*tracks_from_store(store, first, last, name), min_len=min_len)
    params, reasons = recommend_trackcorr_params(stats, **kwargs)
    return params, stats, reasons

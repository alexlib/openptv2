"""Frame-quality awareness (backbone improvement B3).

Skip or interpolate frames with far fewer correspondences than the sequence
median. Low-correspondence frames are usually detection failures or
occlusions; tracking through them produces garbage links.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def frame_quality_scores(store: Any, first: int, last: int) -> dict[int, float]:
    """Return per-frame quality in [0,1] based on correspondence count.

    Quality = min(1, count / median_count). Frames with 0 correspondences = 0.
    """
    counts = {}
    for f in range(first, last + 1):
        try:
            pos, _ = store.read_correspondences(f)
            counts[f] = len(pos)
        except Exception:
            counts[f] = 0
    if not counts:
        return {}
    med = float(np.median(list(counts.values())))
    if med < 1:
        med = 1.0
    return {f: min(1.0, c / med) for f, c in counts.items()}


def low_quality_frames(store: Any, first: int, last: int, threshold: float = 0.5) -> list[int]:
    """Return frames whose quality is below threshold."""
    scores = frame_quality_scores(store, first, last)
    return [f for f, q in scores.items() if q < threshold]


def interpolate_frame_correspondences(
    store: Any, frame: int, first: int, last: int
) -> int:
    """If a frame has no correspondences, interpolate from neighbors. Returns count written."""
    try:
        pos, cam = store.read_correspondences(frame)
        if len(pos) > 0:
            return 0
    except Exception:
        pos, cam = None, None
    # find nearest non-empty neighbor
    for delta in range(1, 5):
        for cand in (frame - delta, frame + delta):
            if cand < first or cand > last:
                continue
            try:
                p, c = store.read_correspondences(cand)
                if len(p) > 0:
                    store.write_correspondences(frame, np.asarray(p).copy(), np.asarray(c).copy())
                    return len(p)
            except Exception:
                continue
    return 0

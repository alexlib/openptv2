"""Temporal background estimation for bubble/glare-point detection.

Large-scale HFSB frames (e.g. Ilmenau barrel) carry uneven, STATIC
background: wall shading, seams, stuck sensor pattern. The bubbles move, so
the background is whatever does NOT change over time. For each pixel:

- background level = temporal MEDIAN over sampled frames (a moving bubble
  occupies a given pixel in a minority of frames, so the median is blind
  to it);
- constant-pixel test = temporal STD below ``median(std) * std_factor``.
  Pixels that fail (bubble traffic, flicker) fall back to the temporal
  MINIMUM for bright-on-dark polarity (bubbles only ever brighten the
  pixel) or MAXIMUM for dark-on-bright.

Subtracting this background before the local z-score in
``detect_bubbles_fast`` removes the static shading that otherwise inflates
the local mean/std and buries faint far bubbles.
"""

from __future__ import annotations

import numpy as np


def estimate_background(
    stack,
    method: str = "median",
    std_factor: float = 3.0,
    polarity: str = "bright",
) -> tuple[np.ndarray, dict]:
    """Estimate a static background frame from a time stack.

    Args:
        stack: (K, H, W) array-like of frames (any integer/float dtype).
            K should span enough motion that every pixel is bubble-free in
            a majority of frames (K >= ~11 recommended).
        method: ``"median"`` (default, robust to bubble traffic) or
            ``"mean"`` (only for clean sequences).
        std_factor: constant-pixel cutoff as a multiple of the frame-wide
            median temporal std. Pixels above it use the min/max fallback.
        polarity: ``"bright"`` (bubbles brighter than background, fallback
            = temporal minimum) or ``"dark"`` (fallback = temporal maximum).

    Returns:
        (background float32 (H, W), info dict with ``frac_unstable``,
        ``std_cutoff`` and ``K``).
    """
    arr = np.asarray(stack)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"Need a (K>=2, H, W) stack, got {arr.shape}")
    if method not in ("median", "mean"):
        raise ValueError(f"Unknown bg method: {method!r}")
    if polarity not in ("bright", "dark"):
        raise ValueError(f"Unknown polarity: {polarity!r}")

    work = arr.astype(np.float64, copy=False)
    med = (
        np.median(work, axis=0)
        if method == "median"
        else np.mean(work, axis=0)
    )
    tstd = np.std(work, axis=0)
    cutoff = float(np.median(tstd)) * float(std_factor)
    unstable = tstd > cutoff

    bg = med.copy()
    if bool(unstable.any()):
        fallback = (
            np.min(work, axis=0)
            if polarity == "bright"
            else np.max(work, axis=0)
        )
        bg[unstable] = fallback[unstable]

    info = {
        "K": int(arr.shape[0]),
        "method": method,
        "std_cutoff": cutoff,
        "frac_unstable": float(unstable.mean()),
    }
    return bg.astype(np.float32), info


def subtract_background(img, bg, polarity: str = "bright") -> np.ndarray:
    """Subtract a background frame, clipped to the physical range.

    Bright polarity: ``clip(img - bg, 0, None)`` (bubbles stay positive).
    Dark polarity: ``clip(bg - img, 0, None)`` so bubbles stay positive too
    and downstream code keeps its bright-peak assumption.
    """
    img_f = np.asarray(img, dtype=np.float64)
    bg_f = np.asarray(bg, dtype=np.float64)
    if img_f.shape != bg_f.shape:
        raise ValueError(f"Shape mismatch: {img_f.shape} vs {bg_f.shape}")
    if polarity == "bright":
        return np.clip(img_f - bg_f, 0.0, None)
    if polarity == "dark":
        return np.clip(bg_f - img_f, 0.0, None)
    raise ValueError(f"Unknown polarity: {polarity!r}")

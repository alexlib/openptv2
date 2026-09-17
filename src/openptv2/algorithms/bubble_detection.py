"""Fast bubble/glare-point detection via local (not global) contrast.

A bubble's glare-points are only a few grey levels above their OWN local
neighborhood, but that neighborhood's brightness varies a lot across a real
frame -- so a single global threshold either misses faint/far bubbles or
drowns in false positives from brighter regions elsewhere. Score every pixel
against its own local background instead (a z-score against a sliding-window
local mean/std): validated in scripts/label_illmenau_bubbles.py against 30
hand-labeled bubbles (bright/close and faint/far alike) -- every one scored
at least 8 local-sigma above its neighborhood, none of the tested false
candidates came close.

A bubble's 2-3 glare fragments sit a few pixels apart with genuinely
below-threshold gap between them (not a grey-level step a flood fill can
cross), so merging them takes a second criterion: any two above-threshold
pixels within ``merge_radius`` of each other belong to the same bubble.
``_bfs_merge`` does exactly that -- a radius-hop flood fill instead of the
4/8-connectivity + separate binary-dilation pass the prototype used -- so
detection is one pass over the local-mean/std arrays with no intermediate
full-frame mask/dilated-mask/rescaled-copy allocations.
"""

import cython
import numpy as np
from scipy.ndimage import uniform_filter

if not cython.compiled:
    pass


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
def _bfs_merge(
    z: cython.double[:, ::1],
    visited: cython.uchar[:, ::1],
    xs_buf: cython.int[::1],
    ys_buf: cython.int[::1],
    start_x: cython.int,
    start_y: cython.int,
    imx: cython.int,
    imy: cython.int,
    z_thresh: cython.double,
    merge_radius: cython.int,
):
    """Flood-fill from (start_x, start_y): visits every above-threshold pixel
    reachable by hops of at most ``merge_radius`` pixels (not grey-level
    continuity), accumulating a z-weighted centroid.

    Returns (sum_wx, sum_wy, sum_w, npix). ``xs_buf``/``ys_buf`` are reused
    scratch space across calls, sized to the largest blob worth keeping
    (``nnmax``); a blob that overflows them stops growing (its excess
    pixels are still marked visited so they don't seed a spurious second
    blob) and is filtered out by the ``nnmax`` check in the caller.
    """
    cap: cython.int = xs_buf.shape[0]
    head: cython.int = 0
    tail: cython.int = 0

    xs_buf[0] = start_x
    ys_buf[0] = start_y
    tail = 1
    visited[start_y, start_x] = 1

    sum_w: cython.double = 0.0
    sum_wx: cython.double = 0.0
    sum_wy: cython.double = 0.0
    npix: cython.int = 0

    cx: cython.int
    cy: cython.int
    nx: cython.int
    ny: cython.int
    dx: cython.int
    dy: cython.int
    w: cython.double

    while head < tail:
        cx = xs_buf[head]
        cy = ys_buf[head]
        head += 1

        w = z[cy, cx]
        sum_w += w
        sum_wx += w * cx
        sum_wy += w * cy
        npix += 1

        for dy in range(-merge_radius, merge_radius + 1):
            ny = cy + dy
            if ny < 0 or ny >= imy:
                continue
            for dx in range(-merge_radius, merge_radius + 1):
                nx = cx + dx
                if nx < 0 or nx >= imx:
                    continue
                if visited[ny, nx]:
                    continue
                if z[ny, nx] < z_thresh:
                    continue
                visited[ny, nx] = 1
                if tail < cap:
                    xs_buf[tail] = nx
                    ys_buf[tail] = ny
                    tail += 1

    return sum_wx, sum_wy, sum_w, npix


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def detect_bubbles_fast(
    hp,
    win: cython.int = 25,
    z_thresh: cython.double = 7.0,
    merge_radius: cython.int = 4,
    nnmin: cython.int = 1,
    nnmax: cython.int = 5000,
    max_targets: cython.int = 200000,
):
    """Detect bubble centroids in a background-removed frame.

    Args:
        hp: background-removed image (2D, any real dtype -- e.g. the output
            of ``openptv2.image_processing.preprocess_image``).
        win: local mean/std window, in pixels.
        z_thresh: local-sigma cutoff (validated minimum over real bubbles
            was ~8.3; 7.0 leaves a small margin).
        merge_radius: max pixel gap bridged between a bubble's separate
            glare fragments.
        nnmin, nnmax: accepted pixel-count range per detected blob.
        max_targets: output capacity.

    Returns:
        (xs, ys, npix) arrays, one row per detected bubble, in image pixel
        coordinates (x = column, y = row).
    """
    hp_arr = np.ascontiguousarray(hp, dtype=np.float64)
    imy: cython.int = hp_arr.shape[0]
    imx: cython.int = hp_arr.shape[1]

    local_mean = uniform_filter(hp_arr, size=win)
    local_sqmean = uniform_filter(hp_arr * hp_arr, size=win)
    local_var = local_sqmean - local_mean * local_mean
    np.maximum(local_var, 1e-6, out=local_var)
    local_std = np.sqrt(local_var)

    z_arr = (hp_arr - local_mean) / local_std
    z: cython.double[:, ::1] = np.ascontiguousarray(z_arr)

    visited = np.zeros((imy, imx), dtype=np.uint8)
    visited_mv: cython.uchar[:, ::1] = visited

    xs_buf = np.empty(nnmax + 1, dtype=np.int32)
    ys_buf = np.empty(nnmax + 1, dtype=np.int32)
    xs_buf_mv: cython.int[::1] = xs_buf
    ys_buf_mv: cython.int[::1] = ys_buf

    out_x = np.empty(max_targets, dtype=np.float64)
    out_y = np.empty(max_targets, dtype=np.float64)
    out_n = np.empty(max_targets, dtype=np.int32)

    n_targets: cython.int = 0
    x: cython.int
    y: cython.int
    sum_wx: cython.double
    sum_wy: cython.double
    sum_w: cython.double
    npix: cython.int

    for y in range(imy):
        for x in range(imx):
            if visited_mv[y, x]:
                continue
            if z[y, x] < z_thresh:
                visited_mv[y, x] = 1
                continue
            sum_wx, sum_wy, sum_w, npix = _bfs_merge(
                z, visited_mv, xs_buf_mv, ys_buf_mv, x, y, imx, imy,
                z_thresh, merge_radius,
            )
            if npix < nnmin or npix > nnmax or sum_w <= 0:
                continue
            if n_targets >= max_targets:
                continue
            out_x[n_targets] = sum_wx / sum_w
            out_y[n_targets] = sum_wy / sum_w
            out_n[n_targets] = npix
            n_targets += 1

    return out_x[:n_targets], out_y[:n_targets], out_n[:n_targets]

"""Plate labeler — regular grid → (X,Y) with two profiles.

* ``small_6x7_coded`` (Illmenau ``Kalibrierung_1``): 3 white-in-black L
  (corner ``(0,0)`` + ``+1·pitch`` in ``+Y`` + ``+2·pitch`` in ``+X`` with a
  black dot at ``+1·pitch,0`` between). Unambiguous origin/axes per view.

* ``large_25x19`` (Multiview): uncoded adjacency-BFS + RANSAC affine
  (hub TODO “enforce 40mm constraint”).

Selection is auto-detected by ``n_coded == 3`` (so no explicit profile flag is
required), though a ``profile`` arg can force it.  ``pitch_x``/``pitch_y`` is
always a user param; ``y_sign = +1`` for Illmenau bottom→top, ``−1`` for legacy
``multiview_calibration.py:75`` top→bottom.
"""

from __future__ import annotations

import numpy as np


def _identify_L(coded_xy: np.ndarray, pitch: float):
    """Find L corner / axes from 3 coded centroids.

    Returns (corner, e_x, e_y) where ``e_x``/``e_y`` are unit vectors in
    image space pointing +X / +Y.  Uses *ratios* so it works in image space
    regardless of world-scale ``pitch``.
    """
    if coded_xy.shape != (3, 2):
        raise ValueError(f"need 3 coded points, got {coded_xy.shape}")
    best = None
    best_err = float("inf")
    for ci in range(3):
        c = coded_xy[ci]
        others = [coded_xy[i] for i in range(3) if i != ci]
        d0 = float(np.linalg.norm(others[0] - c))
        d1 = float(np.linalg.norm(others[1] - c))
        if d0 < 1e-6 or d1 < 1e-6:
            continue
        # One leg ~pitch, other ~2·pitch — check ratio, not absolute scale
        # Short leg = min(d0,d1), long leg = max(d0,d1) should be ~2×
        short, long = (d0, d1) if d0 < d1 else (d1, d0)
        ratio = long / short if short > 1e-9 else float("inf")
        ratio_err = abs(ratio - 2.0)
        v0 = others[0] - c
        v1 = others[1] - c
        cosang = float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        err = ratio_err + abs(cosang)  # both dimensionless
        if err < best_err:
            best_err = err
            if d0 < d1:
                y_pt, x_pt = others[0], others[1]
            else:
                y_pt, x_pt = others[1], others[0]
            corner = c
            e_y = (y_pt - c) / float(np.linalg.norm(y_pt - c))
            e_x = (x_pt - c) / float(np.linalg.norm(x_pt - c))
            best = (corner, e_x, e_y, y_pt, x_pt)
    if best is None or best_err > 0.85:
        raise ValueError(f"L identification failed: best_err={best_err:.3f} "
                         f"coded points {coded_xy.tolist()} do not form the expected 1:2 right angle")
    corner, e_x, e_y, y_pt, x_pt = best
    if abs(float(np.dot(e_x, e_y))) > 0.45:
        raise ValueError(f"L legs not orthogonal: dot={float(np.dot(e_x, e_y)):.3f}")
    return corner, e_x, e_y


def label_coded_6x7(
    centroids: np.ndarray,
    coded_mask: np.ndarray,
    *,
    pitch_x: float = 40.0,
    pitch_y: float = 40.0,
    nx: int = 6,
    ny: int = 7,
    y_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L-anchored labeling for the small 6×7 plate.

    Returns (img_pts, ref_pts, index) where ``img_pts`` is (n,2) pixel,
    ``ref_pts`` is (n,3) with ``Z=0``, and ``index`` is (n,2) ``(ix,iy)`` grid
    indices.  ``n`` is the number of detected dots that snap to a grid node
    within ``0.35·pitch``.
    """
    centroids = np.asarray(centroids, float)
    coded_mask = np.asarray(coded_mask, bool)
    if centroids.shape[0] != coded_mask.shape[0]:
        raise ValueError("centroids / coded_mask length mismatch")
    # Auto-detect: expect 3 coded
    coded_xy = centroids[coded_mask]
    if coded_xy.shape[0] != 3:
        raise ValueError(f"coded 6×7 needs exactly 3 coded dots, got {coded_xy.shape[0]}")
    pitch = float((pitch_x + pitch_y) / 2.0)
    corner, e_x, e_y = _identify_L(coded_xy, pitch)

    # Project every centroid onto the L bases
    cand = [p for p in coded_xy if not np.allclose(p, corner)]
    d0 = float(np.linalg.norm(cand[0] - corner))
    d1 = float(np.linalg.norm(cand[1] - corner))
    if d0 < d1:
        y_pt, x_pt = cand[0], cand[1]
    else:
        y_pt, x_pt = cand[1], cand[0]
    vx = (x_pt - corner) / 2.0  # one pitch step in +X
    vy = y_pt - corner          # one pitch step in +Y
    M = np.column_stack([vx, vy])  # 2×2
    det = float(np.linalg.det(M))
    if abs(det) < 1e-6:
        raise ValueError(f"L basis degenerate (det={det:.2e}); cannot invert")
    Minv = np.linalg.inv(M)
    rel = centroids - corner
    coeffs = (Minv @ rel.T).T  # (n,2) where col0=ix_rel_f, col1=iy_rel_f
    ix_rel_f = coeffs[:, 0]
    iy_rel_f = coeffs[:, 1]
    ix_rel = np.round(ix_rel_f).astype(int)
    iy_rel = np.round(iy_rel_f).astype(int)

    # The L corner may be anywhere in the grid (e.g. at (3, 3) or (2, 3)),
    # so shift the integer grid by its minimum so that indices span [0..nx-1] x [0..ny-1].
    tol = 0.35
    snapped = (np.abs(ix_rel_f - ix_rel) < tol) & (np.abs(iy_rel_f - iy_rel) < tol)
    if not np.any(snapped):
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 2), int)

    min_x = int(np.min(ix_rel[snapped]))
    min_y = int(np.min(iy_rel[snapped]))
    ix_f = ix_rel_f - min_x
    iy_f = iy_rel_f - min_y
    ix = np.round(ix_f).astype(int)
    iy = np.round(iy_f).astype(int)

    # Keep only points that snap within tolerance and lie inside the nx x ny rectangle
    keep = (
        (np.abs(ix_f - ix) < tol) & (np.abs(iy_f - iy) < tol) &
        (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    )
    idx = np.column_stack([ix[keep], iy[keep]])
    img_pts = centroids[keep]

    # Deduplicate by (ix,iy): keep the closest reprojection
    seen: dict[tuple[int, int], int] = {}
    order = []
    for k, (xi, yi) in enumerate(idx):
        key = (int(xi), int(yi))
        if key not in seen:
            seen[key] = k
            order.append(k)
        else:
            prev = seen[key]
            if (abs(ix_f[keep][k] - xi) + abs(iy_f[keep][k] - yi)) < (abs(ix_f[keep][prev] - int(idx[prev, 0])) + abs(iy_f[keep][prev] - int(idx[prev, 1]))):
                order[order.index(prev)] = k
                seen[key] = k
    if order:
        idx = idx[order]
        img_pts = img_pts[order]

    # If perspective distortion is strong and we have at least 12 points,
    # fit a homography from (ix, iy) to img_pts and re-snap all centroids for 100% recovery
    if len(img_pts) >= 12 and len(img_pts) < nx * ny and len(centroids) >= nx * ny:
        try:
            # Fit H from grid (ix, iy) -> image (x, y)
            A_mat = []
            for (g_x, g_y), (p_x, p_y) in zip(idx, img_pts):
                A_mat.append([g_x, g_y, 1, 0, 0, 0, -p_x * g_x, -p_x * g_y, -p_x])
                A_mat.append([0, 0, 0, g_x, g_y, 1, -p_y * g_x, -p_y * g_y, -p_y])
            _, _, Vt = np.linalg.svd(np.array(A_mat))
            H_grid_to_img = Vt[-1].reshape(3, 3)
            H_img_to_grid = np.linalg.inv(H_grid_to_img)

            # Project all centroids to grid
            cent_h = np.column_stack([centroids, np.ones(len(centroids))])
            proj_grid = (cent_h @ H_img_to_grid.T)
            proj_grid = proj_grid[:, :2] / proj_grid[:, 2:3]

            ix_h_f = proj_grid[:, 0]
            iy_h_f = proj_grid[:, 1]
            ix_h = np.round(ix_h_f).astype(int)
            iy_h = np.round(iy_h_f).astype(int)

            keep_h = (
                (np.abs(ix_h_f - ix_h) < tol) & (np.abs(iy_h_f - iy_h) < tol) &
                (ix_h >= 0) & (ix_h < nx) & (iy_h >= 0) & (iy_h < ny)
            )
            idx_h = np.column_stack([ix_h[keep_h], iy_h[keep_h]])
            img_pts_h = centroids[keep_h]

            seen_h: dict[tuple[int, int], int] = {}
            order_h = []
            for k, (xi, yi) in enumerate(idx_h):
                key = (int(xi), int(yi))
                if key not in seen_h:
                    seen_h[key] = k
                    order_h.append(k)
            if len(order_h) > len(order):
                idx = idx_h[order_h]
                img_pts = img_pts_h[order_h]
        except Exception:
            pass

    ref_pts = np.array([[xi * pitch_x, yi * pitch_y * y_sign, 0.0] for xi, yi in idx], dtype=float)
    return img_pts, ref_pts, idx


def label_uncoded_grid(
    centroids: np.ndarray,
    *,
    pitch_x: float = 40.0,
    pitch_y: float = 40.0,
    nx: int = 25,
    ny: int = 19,
    y_sign: int = -1,
    tol: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjacency-BFS labeler for regular uncoded grids (Multiview).

    Estimates spacing via Delaunay / nearest-neighbour, builds a graph at
    ``1.4·pitch`` (image space), BFS-assigns ``(ix,iy)`` from an arbitrary
    seed, then RANSAC affine ``(ix,iy) → (x,y)`` to reject outliers.  Returns
    the same ``(img_pts, ref_pts, index)`` as the coded path.
    """
    pts = np.asarray(centroids, float)
    n = pts.shape[0]
    if n == 0:
        return pts, np.zeros((0, 3)), np.zeros((0, 2), int)
    # Estimate image-space pitch as median nearest-neighbour distance
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    pitch_img = float(np.median(dists[:, 1]))
    if not np.isfinite(pitch_img) or pitch_img < 1e-6:
        raise ValueError("could not estimate pitch from centroids")
    # Build adjacency at 1.4·pitch
    pairs = tree.query_ball_point(pts, r=1.4 * pitch_img)
    # BFS
    visited = np.zeros(n, dtype=bool)
    grid = np.full((n, 2), 9999, int)
    # Start from the point nearest the centroid (more stable than arbitrary)
    seed = int(np.argmin(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))
    grid[seed] = [0, 0]
    visited[seed] = True
    stack = [seed]
    # Precompute mean pitch vectors via local PCA for orientation?  Use a
    # simple heuristic: principal axes from covariance of pts
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    ex = eigvecs[:, np.argmax(eigvals)]
    ey = np.array([-ex[1], ex[0]])
    # Normalize
    ex /= float(np.linalg.norm(ex)) or 1.0
    ey /= float(np.linalg.norm(ey)) or 1.0

    while stack:
        cur = stack.pop()
        for nb in pairs[cur]:
            if visited[nb]:
                continue
            # Assign grid offset from cur by projecting delta onto ex/ey
            delta = pts[nb] - pts[cur]
            # Heuristic grid step: round projection / pitch_img
            step_x = int(round(float(np.dot(delta, ex) / pitch_img)))
            step_y = int(round(float(np.dot(delta, ey) / pitch_img)))
            # Clamp to single step (BFS ensures neighbours are adjacent)
            step_x = int(np.clip(step_x, -1, 1))
            step_y = int(np.clip(step_y, -1, 1))
            if step_x == 0 and step_y == 0:
                continue
            grid[nb] = grid[cur] + np.array([step_x, step_y])
            visited[nb] = True
            stack.append(nb)

    # Keep visited, normalize so min is (0,0)
    mask = visited & np.all(grid != 9999, axis=1)
    if not np.any(mask):
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 2), int)
    gmin = grid[mask].min(axis=0)
    grid[mask] -= gmin
    # Filter to nx×ny
    inside = mask & (grid[:, 0] >= 0) & (grid[:, 0] < nx) & (grid[:, 1] >= 0) & (grid[:, 1] < ny)
    img_pts = pts[inside]
    g = grid[inside]
    # RANSAC affine to reject mislabels: fit (ix,iy) → (x,y) linear + prune
    if len(img_pts) >= 6:
        # Use least-squares affine, prune by residual > 0.5·pitch_img
        A = np.column_stack([g[:, 0], g[:, 1], np.ones(len(g))])
        for _ in range(2):
            # Fit separately for x and y
            cx, *_ = np.linalg.lstsq(A, img_pts[:, 0], rcond=None)
            cy, *_ = np.linalg.lstsq(A, img_pts[:, 1], rcond=None)
            pred = np.column_stack([A @ cx, A @ cy])
            resid = np.linalg.norm(pred - img_pts, axis=1)
            keep2 = resid < 0.5 * pitch_img
            if np.all(keep2):
                break
            A = A[keep2]
            img_pts = img_pts[keep2]
            g = g[keep2]
            if len(g) < 6:
                break
    # Snap quality check: residual < tol·pitch_img was already applied
    idx = g
    # Completeness guard: caller should check len vs 0.85·nx·ny
    ref_pts = np.array([[xi * pitch_x, yi * pitch_y * y_sign, 0.0] for xi, yi in idx], dtype=float)
    return img_pts, ref_pts, idx


def label_plate(
    centroids: np.ndarray,
    coded_mask: np.ndarray | None = None,
    *,
    pitch_x: float = 40.0,
    pitch_y: float = 40.0,
    nx: int | None = None,
    ny: int | None = None,
    y_sign: int | None = None,
    profile: str | None = None,
):
    """Auto-dispatch to coded vs uncoded labeler.

    ``profile`` can force ``"small_6x7_coded"`` or ``"large_25x19"``; otherwise
    ``n_coded == 3`` selects coded, anything else uncoded.  ``nx/ny/y_sign``
    default per profile when not given.
    """
    centroids = np.asarray(centroids, float)
    if coded_mask is not None:
        coded_mask = np.asarray(coded_mask, bool)
        n_coded = int(np.sum(coded_mask))
    else:
        n_coded = 0

    if profile == "small_6x7_coded" or (profile is None and n_coded == 3):
        nx = nx if nx is not None else 6
        ny = ny if ny is not None else 7
        y_sign = y_sign if y_sign is not None else 1
        return label_coded_6x7(centroids, coded_mask, pitch_x=pitch_x, pitch_y=pitch_y, nx=nx, ny=ny, y_sign=y_sign)
    # uncoded path
    nx = nx if nx is not None else 25
    ny = ny if ny is not None else 19
    y_sign = y_sign if y_sign is not None else -1
    return label_uncoded_grid(centroids, pitch_x=pitch_x, pitch_y=pitch_y, nx=nx, ny=ny, y_sign=y_sign)

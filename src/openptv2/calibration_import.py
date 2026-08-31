"""Foreign-calibration import — OpenCV / points-file doors.

No ``cv2`` here.  ``scipy.spatial.transform.Rotation`` is allowed (SciPy is
already a core dependency).  Primitives ``angles_from_dm`` /
``exterior_from_rotation`` are imported from ``calibration_seed`` — that module
is the canonical owner (hub plan Part 2 S1.1).

Verified facts live in ``docs/plans/2026-08-30-calibration-hub-multi-source.md``
and ``…-illmenau-dots-plate-pipeline.md`` — do not re-derive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Glass,
    Interior,
    MmLut,
)
from openptv2.calibration_seed import exterior_from_rotation

# ---------------------------------------------------------------------------
# B door: OpenCV → openPTV
# ---------------------------------------------------------------------------


def calibration_from_opencv(
    K,
    dist,
    rvec,
    tvec,
    *,
    imx: int,
    imy: int,
    pix_x: float,
    pix_y: float | None = None,
    glass_vec: tuple[float, float, float] = (0.0, 0.0, 1.0),
    pixel_origin: str = "corner",
    n1: float = 1.0,
    n2: float = 1.0,
    n3: float = 1.0,
    d: float = 1.0,
) -> tuple[Calibration, float]:
    """Convert an OpenCV calibration to openPTV's ``.ori``/``.addpar`` model.

    ``K`` is 3×3, ``dist`` is OpenCV order ``(k1,k2,p1,p2[,k3[,k4..]])`` of
    length 4, 5, 8, 12 or 14 (first five used; raise naming ``k4`` etc. when
    unsupported terms are non-zero).  ``rvec`` / ``tvec`` are 3-vectors (world
    → camera as OpenCV ``[R|t]``).  Returns ``(Calibration, pix_y_used)``.

    ``pixel_origin``: ``"corner"`` (no shift, COLMAP/Metashape/openPTV) or
    ``"centre"`` (+0.5 px on ``cx, cy`` — OpenCV/Kalibr integer-at-centre
    convention).  The 0.5 px question appears as a constant bias in the
    residual field (hub Step 7).

    Glass defaults to air ``1/1/1/1`` with a unit vector along ``+Z``; inert
    while ``n1==n2==n3`` but must be non-zero unless you want the pinhole
    short-circuit at ``imgcoord.py:196``.
    """
    K_a = np.asarray(K, float)
    if K_a.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K_a.shape}")
    fx = float(K_a[0, 0])
    fy = float(K_a[1, 1])
    cx = float(K_a[0, 2])
    cy = float(K_a[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"fx/fy must be >0, got fx={fx} fy={fy}")
    if pixel_origin == "centre":
        cx = cx + 0.5
        cy = cy + 0.5
    elif pixel_origin != "corner":
        raise ValueError("pixel_origin must be 'corner' or 'centre'")

    dist_a = np.asarray(dist, float).ravel() if dist is not None else np.zeros(5)
    if dist_a.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"dist length must be 4/5/8/12/14, got {dist_a.size}")
    # Pad to 5 for uniform indexing, but check trailing unsupported terms first
    names_extra = ["k4", "k5", "k6", "s1", "s2", "s3", "s4", "taux", "tauy"]
    # dist layout for OpenCV extended: k1,k2,p1,p2,k3,k4,k5,k6,s1..s4,taux,tauy
    # For len 8: k1,k2,p1,p2,k3,k4,k5,k6 ; len12 adds s1..s4 ; len14 adds taux/tauy
    full_extra = dist_a[5:] if dist_a.size > 5 else np.array([])
    # Check in order, naming the first non-zero
    for idx, val in enumerate(full_extra):
        if abs(float(val)) > 1e-12:
            # Map index to name
            if dist_a.size == 8:
                nm = ["k4", "k5", "k6"][idx] if idx < 3 else f"extra[{idx}]"
            elif dist_a.size == 12:
                if idx < 3:
                    nm = ["k4", "k5", "k6"][idx]
                else:
                    nm = ["s1", "s2", "s3", "s4"][idx - 3]
            elif dist_a.size == 14:
                if idx < 3:
                    nm = ["k4", "k5", "k6"][idx]
                elif idx < 7:
                    nm = ["s1", "s2", "s3", "s4"][idx - 3]
                else:
                    nm = ["taux", "tauy"][idx - 7]
            else:
                nm = names_extra[idx] if idx < len(names_extra) else f"extra[{idx}]"
            raise ValueError(
                f"dist {nm}={val} non-zero — openPTV has no representation; "
                "use door C (resample the foreign model on a grid)"
            )
    # First five (pad missing k3 as 0)
    k1_cv = float(dist_a[0]) if dist_a.size >= 1 else 0.0
    k2_cv = float(dist_a[1]) if dist_a.size >= 2 else 0.0
    p1_cv = float(dist_a[2]) if dist_a.size >= 3 else 0.0
    p2_cv = float(dist_a[3]) if dist_a.size >= 4 else 0.0
    k3_cv = float(dist_a[4]) if dist_a.size >= 5 else 0.0

    rvec_a = np.asarray(rvec, float).ravel()
    tvec_a = np.asarray(tvec, float).ravel()
    if rvec_a.size != 3 or tvec_a.size != 3:
        raise ValueError(f"rvec/tvec must be length-3, got {rvec_a.shape} / {tvec_a.shape}")

    # Verified block (hub 108)
    S = np.diag([1.0, -1.0, -1.0])
    R_cv = Rotation.from_rotvec(rvec_a).as_matrix()  # world → camera
    C = -R_cv.T @ tvec_a  # camera centre → ext.x0,y0,z0 [mm]
    dm = R_cv.T @ S  # S on RIGHT

    if pix_x is None or float(pix_x) <= 0:
        raise ValueError(f"pix_x must be >0, got {pix_x}")
    pix_x_f = float(pix_x)
    cc = float(fx * pix_x_f)
    if pix_y is not None:
        pix_y_f = float(pix_y)
        # Consistency check: fx*pix_x ~ fy*pix_y up to cc tolerance
        if abs(cc - float(fy * pix_y_f)) > 1e-6 * max(abs(cc), 1.0):
            # Not an error — just use provided pix_y and keep cc from fx/pix_x;
            # fy will be slightly inconsistent (anisotropy not representable in scx)
            pass
    else:
        pix_y_f = cc / float(fy)

    xh = (cx - imx / 2.0) * pix_x_f
    yh = (imy / 2.0 - cy) * pix_y_f  # sign flip: openPTV y UP

    # Distortion scaling: OpenCV r is normalised, openPTV r is mm
    k1 = k1_cv / (cc ** 2) if cc != 0 else 0.0
    k2 = k2_cv / (cc ** 4) if cc != 0 else 0.0
    k3 = k3_cv / (cc ** 6) if cc != 0 else 0.0
    # p1/p2 swap + sign (hub 124)
    p1 = p2_cv / cc if cc != 0 else 0.0
    p2 = -p1_cv / cc if cc != 0 else 0.0
    scx, she = 1.0, 0.0

    ext = exterior_from_rotation(C, dm)
    int_par = Interior(xh=float(xh), yh=float(yh), cc=float(cc))
    # Glass
    gv = np.asarray(glass_vec, float).ravel()
    if gv.shape != (3,):
        raise ValueError(f"glass_vec must be length-3, got {gv.shape}")
    n = float(np.linalg.norm(gv))
    if n < 1e-12:
        raise ValueError("glass_vec must be non-zero")
    gv = gv / n
    glass = Glass(vec_x=float(gv[0]), vec_y=float(gv[1]), vec_z=float(gv[2]),
                  n1=float(n1), n2=float(n2), n3=float(n3), d=float(d))
    ap = AddedPar(k1=float(k1), k2=float(k2), k3=float(k3),
                  p1=float(p1), p2=float(p2), scx=float(scx), she=float(she))
    cal = Calibration(ext_par=ext, int_par=int_par, glass_par=glass,
                      added_par=ap, mmlut=MmLut())
    return cal, float(pix_y_f)


def opencv_from_calibration(
    cal: Calibration,
    *,
    imx: int,
    imy: int,
    pix_x: float,
    pix_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`calibration_from_opencv`. Returns (K, dist, rvec, tvec).

    Uses the same verified relations; round-trip must reproduce projections
    within 1e-9 px when xh=yh=0 and distortion is modest.
    """
    pix_x_f = float(pix_x)
    pix_y_f = float(pix_y)
    cc = float(cal.int_par.cc)
    xh = float(cal.int_par.xh)
    yh = float(cal.int_par.yh)
    # K
    fx = cc / pix_x_f
    fy = cc / pix_y_f
    cx = xh / pix_x_f + imx / 2.0
    cy = imy / 2.0 - yh / pix_y_f
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=float)
    # dist
    k1_cv = float(cal.added_par.k1) * (cc ** 2)
    k2_cv = float(cal.added_par.k2) * (cc ** 4)
    k3_cv = float(cal.added_par.k3) * (cc ** 6)
    # inverse of p swap: p1 = p2_cv/cc, p2 = -p1_cv/cc  →  p2_cv = p1*cc, p1_cv = -p2*cc
    p1_cv = -float(cal.added_par.p2) * cc
    p2_cv = float(cal.added_par.p1) * cc
    dist = np.array([k1_cv, k2_cv, p1_cv, p2_cv, k3_cv], dtype=float)
    # R/t from ext
    C = np.array([float(cal.ext_par.x0), float(cal.ext_par.y0), float(cal.ext_par.z0)])
    dm = np.array(cal.ext_par.dm, float)  # camera→world
    S = np.diag([1.0, -1.0, -1.0])
    # dm = R_cv.T @ S  →  R_cv.T = dm @ S  (S involutory)  →  R_cv = (dm @ S).T
    R_cv = (dm @ S).T
    tvec = -R_cv @ C
    rvec = Rotation.from_matrix(R_cv).as_rotvec()
    return K, dist, rvec, tvec


# ---------------------------------------------------------------------------
# A door: universal 5-column points file  →  (img_pts, ref_pts)
# ---------------------------------------------------------------------------


def read_xyXYZ(path: str | Path, *, delimiter: str | None = None):
    """Read the universal 5-column point file → (img_pts (n,2), ref_pts (n,3)).

    Covers proPTV ``markers_cN.txt``, MyPTV ``camN_cal_points``,
    ``Multiview-Calibration`` ``cN_xyXYZ.txt`` and DaVis plate exports — all
    the same ``x y X Y Z``, differing only in separator and header.

    Accepts whitespace **or** comma separation (retries with ``delimiter=","``),
    skips ``#`` comment lines, tolerates an optional 6th column (MyPTV PR#67
    view index).  Raises naming the file/line on ``<5`` columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"points file not found: {p}")
    # Try whitespace first, then comma
    for delim, label in [(None, "whitespace"), (",", "comma")]:
        try:
            arr = np.genfromtxt(str(p), comments="#", delimiter=delim, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 5:
            # Retry with other delimiter before giving up
            continue
        if arr.shape[1] >= 5:
            # valid — return 5-col slice (ignore 6th if present)
            img = arr[:, :2]
            ref = arr[:, 2:5]
            return np.asarray(img, float), np.asarray(ref, float)
    # Fallback: manual line parse to provide a nicer error naming the line
    with p.open() as fh:
        for lineno, line in enumerate(fh, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 5:
                raise ValueError(f"{p}:{lineno}: need ≥5 columns x y X Y Z, got {len(parts)}: {line.strip()!r}")
            break
    # If we got here with no 5-col success, raise with context
    raise ValueError(f"{p}: could not parse as 5-column x y X Y Z (tried whitespace and comma)")


def read_opencv_flat15(path: str | Path) -> dict:
    """Read the Ilmenau ``calib_cN.txt``: 15 floats, one per line.

    Layout (from Multiview-Calibration's own extract_calibration.py):
        [0:3]  rvec  [3:6]  tvec  [6] fx [7] fy [8] cx [9] cy [10:15] dist=k1,k2,p1,p2,k3
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"opencv flat15 file not found: {p}")
    vals = []
    with p.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # allow lines with multiple numbers as well
            for tok in s.replace(",", " ").split():
                vals.append(float(tok))
    if len(vals) != 15:
        raise ValueError(f"{p}: expected 15 floats, got {len(vals)}")
    arr = np.asarray(vals, float)
    return {
        "rvec": arr[0:3],
        "tvec": arr[3:6],
        "fx": float(arr[6]),
        "fy": float(arr[7]),
        "cx": float(arr[8]),
        "cy": float(arr[9]),
        "dist": arr[10:15],  # k1,k2,p1,p2,k3
    }


# ---------------------------------------------------------------------------
# Frame change (Kabsch) — only if needed; otherwise YAGNI.
# ---------------------------------------------------------------------------


def similarity_from_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    with_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch/Procrustes: returns (A, b, s) with dst ≈ s·A·src + b.

    ``A`` is 3×3 orthonormal ``det=+1``, ``b`` is 3-vector, ``s`` is scalar.
    Apply via ``dm_new = A @ dm_old``, ``C_new = A @ C_old + b`` (hub
    verified 3e-12 px).  If ``with_scale`` is False, ``s`` is 1.0.
    """
    src_a = np.asarray(src, float)
    dst_a = np.asarray(dst, float)
    if src_a.shape != dst_a.shape or src_a.shape[1] != 3:
        raise ValueError(f"src/dst must be (n,3) same shape, got {src_a.shape} / {dst_a.shape}")
    if src_a.shape[0] < 3:
        raise ValueError(f"need ≥3 correspondences, got {src_a.shape[0]}")
    cs = src_a.mean(axis=0)
    cd = dst_a.mean(axis=0)
    S = (src_a - cs).T @ (dst_a - cd)
    U, sig, Vt = np.linalg.svd(S)
    A = Vt.T @ U.T
    if float(np.linalg.det(A)) < 0:
        Vt[-1, :] *= -1
        A = Vt.T @ U.T
    if with_scale:
        var = float(np.sum((src_a - cs) ** 2))
        s = float(np.sum(sig) / var) if var > 1e-12 else 1.0
    else:
        s = 1.0
    b = cd - s * (A @ cs)
    return A, b, s

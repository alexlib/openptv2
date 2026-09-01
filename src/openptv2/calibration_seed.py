"""Seeding helpers for the first `.ori` — when there is nothing to import.

This module owns the primitives that both the look-at rig description
(``rig.yaml`` → ``.ori``) and the DLT resection path use.  ``calibration_import``
imports from here, not the other way round.

No ``cv2`` dependency — only ``numpy`` / ``scipy`` (``Rotation``) which are
already core deps.  No Cython rebuild needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Exterior,
    Glass,
    Interior,
    MmLut,
)

# ---------------------------------------------------------------------------
# S1.1 primitives — canonical copies (scripts/calibrate_proptv_dlt.py leaves
# a one-line shim after this move).
# ---------------------------------------------------------------------------


def angles_from_dm(dm: np.ndarray) -> tuple[float, float, float]:
    """Invert ``Exterior.compute_rotation_matrix``.

    ``dm = Rx(omega) @ Ry(phi) @ Rz(kappa)`` (``calibration.py:44``).  Uses the
    closed form from the hub plan; raises on gimbal lock instead of returning
    a silently degenerate pair.
    """
    dm = np.asarray(dm, float)
    if dm.shape != (3, 3):
        raise ValueError(f"dm must be 3x3, got shape {dm.shape}")
    # phi = asin(dm[0,2])
    s_phi = float(np.clip(dm[0, 2], -1.0, 1.0))
    if abs(s_phi) > 1 - 1e-9:
        raise ValueError(
            f"gimbal lock: |dm[0,2]|={abs(s_phi):.3e} ≈ 1 (cos(phi)→0); "
            "omega/kappa degenerate"
        )
    phi = float(np.arcsin(s_phi))
    omega = float(np.arctan2(-dm[1, 2], dm[2, 2]))
    kappa = float(np.arctan2(-dm[0, 1], dm[0, 0]))
    return omega, phi, kappa


def exterior_from_rotation(C: np.ndarray, dm: np.ndarray) -> Exterior:
    """Build an :class:`Exterior` from camera centre and camera→world matrix.

    Goes through ``omega/phi/kappa`` + ``compute_rotation_matrix()`` — writing
    ``.dm`` directly is silently discarded (hub trap 4, projects 8e7 px off).
    Asserts the round-trip before returning.
    """
    C = np.asarray(C, float).ravel()
    dm = np.asarray(dm, float)
    if C.shape != (3,):
        raise ValueError(f"C must be length-3, got {C.shape}")
    omega, phi, kappa = angles_from_dm(dm)
    ext = Exterior(
        x0=float(C[0]),
        y0=float(C[1]),
        z0=float(C[2]),
        omega=omega,
        phi=phi,
        kappa=kappa,
    )
    ext.compute_rotation_matrix()
    if float(np.abs(ext.dm - dm).max()) > 1e-12:
        raise AssertionError(
            f"exterior_from_rotation round-trip failed: max|dm−dm_recomp|="
            f"{float(np.abs(ext.dm - dm).max()):.2e}"
        )
    return ext


def dm_from_lookat(
    C: np.ndarray,
    target: np.ndarray,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    roll: float = 0.0,
) -> np.ndarray:
    """Camera→world rotation for “camera at C looking at target”.

    Mirrors ``calibration_diagnostics.viewing_dir`` convention (``-rot[:,2]``
    points toward the scene) and the independently verified MATLAB
    ``cameraLookAtToExtrinsic`` logic.

    Parameters
    ----------
    C, target : (3,)
        World-space positions.
    up : (3,)
        World up direction (defaults to +Z).
    roll : float
        In-plane roll [rad] about the viewing axis.
    """
    C_a = np.asarray(C, float).ravel()
    T_a = np.asarray(target, float).ravel()
    if C_a.shape != (3,) or T_a.shape != (3,):
        raise ValueError("C and target must be length-3")
    if np.allclose(C_a, T_a):
        raise ValueError(f"C == target ({C_a}); viewing direction undefined")
    back = C_a - T_a
    n = float(np.linalg.norm(back))
    if n < 1e-12:
        raise ValueError("C and target coincide")
    back /= n
    up_a = np.asarray(up, float).ravel()
    right = np.cross(up_a, back)
    rn = float(np.linalg.norm(right))
    if rn < 1e-8:
        # up parallel to view axis — fall back to world +X
        right = np.cross(np.array([1.0, 0.0, 0.0]), back)
        rn = float(np.linalg.norm(right))
        if rn < 1e-12:
            right = np.cross(np.array([0.0, 1.0, 0.0]), back)
            rn = float(np.linalg.norm(right))
    right /= rn
    up_prime = np.cross(back, right)
    dm = np.column_stack([right, up_prime, back])
    if roll:
        c, s = float(np.cos(roll)), float(np.sin(roll))
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        dm = dm @ Rz
    return dm


# ---------------------------------------------------------------------------
# S1.2 Tier-3 seed: look-at + focal length  →  Calibration
# ---------------------------------------------------------------------------


def seed_from_lookat(
    *,
    position: np.ndarray,
    target: np.ndarray,
    focal_mm: float,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    roll: float = 0.0,
    glass_vec: tuple[float, float, float] | None = None,
    n1: float = 1.0,
    n2: float = 1.0,
    n3: float = 1.0,
    d: float = 1.0,
) -> Calibration:
    """Human-friendly rig seed: “camera at P, looking at T, with an f mm lens”.

    ``cc = focal_mm``, ``xh = yh = 0``, ``AddedPar`` at defaults
    ``(0,0,0,0,0,1,0)``.  When ``glass_vec`` is ``None`` the normal is oriented
    from the viewing direction as ``benchmarking/camera_rig.py:221`` does
    (non-zero, along the view axis).

    Note per M2: position error will be about ``|position−target| ×
    (relative focal error)`` — a 2 % barrel marking error at 800 mm standoff
    puts the camera ~16 mm off while RMS can still look “excellent” (0.10 px).
    """
    if focal_mm is None or float(focal_mm) <= 0:
        raise ValueError(f"focal_mm must be >0, got {focal_mm!r}")
    pos = np.asarray(position, float).ravel()
    tgt = np.asarray(target, float).ravel()
    if pos.shape != (3,) or tgt.shape != (3,):
        raise ValueError("position and target must be length-3")
    dm = dm_from_lookat(pos, tgt, up=up, roll=float(roll))
    ext = exterior_from_rotation(pos, dm)
    int_par = Interior(xh=0.0, yh=0.0, cc=float(focal_mm))
    # Glass normal
    if glass_vec is None:
        # Use viewing direction: -dm[:,2] points toward the scene, so the
        # interface normal should be roughly along the view axis and unit.
        # Keep the same sign convention as benchmarking/camera_rig.
        vd = -dm[:, 2]
        gv = vd / float(np.linalg.norm(vd))
    else:
        gv = np.asarray(glass_vec, float).ravel()
        if gv.shape != (3,):
            raise ValueError(f"glass_vec must be length-3, got {gv.shape}")
        n = float(np.linalg.norm(gv))
        if n < 1e-12:
            raise ValueError("glass_vec must be non-zero")
        gv = gv / n
    glass = Glass(
        vec_x=float(gv[0]),
        vec_y=float(gv[1]),
        vec_z=float(gv[2]),
        n1=float(n1),
        n2=float(n2),
        n3=float(n3),
        d=float(d),
    )
    cal = Calibration(
        ext_par=ext,
        int_par=int_par,
        glass_par=glass,
        added_par=AddedPar(),
        mmlut=MmLut(),
    )
    return cal


# ---------------------------------------------------------------------------
# S1.3 Tier-1 seed: DLT resection  →  Calibration (pose AND cc)
# ---------------------------------------------------------------------------


def _dlt_resection(ref_pts: np.ndarray, metric_pts: np.ndarray):
    """Inline DLT (copied from scripts/calibrate_proptv_dlt.py:36) to avoid a
    hard import cycle with scripts/.  Returns (R, C, cc) with R world→camera.

    Kept private — external code should call ``seed_from_dlt``.
    """
    from scipy.linalg import (
        rq as _rq,  # local import so SciPy stays optional at import time
    )

    ref = np.asarray(ref_pts, float)
    met = np.asarray(metric_pts, float)
    n = ref.shape[0]
    if n < 6:
        raise ValueError(f"DLT needs ≥6 points, got {n}")
    if ref.shape[1] != 3 or met.shape[1] != 2:
        raise ValueError(
            f"shapes: ref {ref.shape} need (n,3), metric {met.shape} need (n,2)"
        )
    # Coplanarity guard — DLT degenerate on a plane without a plausible wrong answer.
    centred = ref - ref.mean(axis=0)
    s = np.linalg.svd(centred, compute_uv=False)
    if s[-1] < 1e-6 * max(s[0], 1.0):
        raise ValueError(
            f"ref_pts near-coplanar (smallest singular value {s[-1]:.2e}); "
            "DLT degenerate on a single plane"
        )
    A = np.zeros((2 * n, 12))
    u = met[:, 0]
    v = met[:, 1]
    for i in range(n):
        x, y, z = ref[i]
        A[2 * i] = [x, y, z, 1, 0, 0, 0, 0, -u[i] * x, -u[i] * y, -u[i] * z, -u[i]]
        A[2 * i + 1] = [0, 0, 0, 0, x, y, z, 1, -v[i] * x, -v[i] * y, -v[i] * z, -v[i]]
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    depth_sign = P[2, :3] @ ref.T + P[2, 3]
    if np.mean(depth_sign > 0) < 0.5:
        P = -P
    M = P[:, :3]
    K, R = _rq(M)
    signs = np.sign(np.diag(K))
    target = np.array([-1.0, -1.0, 1.0])
    fix = signs * target
    S = np.diag(fix)
    K = K @ S
    R = S @ R
    scale = K[2, 2]
    K = K / scale
    C = -np.linalg.solve(P[:, :3], P[:, 3])
    cc = K[0, 0]
    if np.linalg.det(R) < 0:
        R = -R
        cc = -cc
    cam = (R @ (ref - C).T).T
    proj = -cc * cam[:, :2] / cam[:, 2:3]
    if np.mean(np.sum((proj - met) ** 2, axis=1)) > np.mean(
        np.sum((-proj - met) ** 2, axis=1)
    ):
        cc = -cc
    return R, C, cc


def seed_from_dlt(
    ref_pts: np.ndarray,
    img_pts: np.ndarray,
    cpar,
) -> Calibration:
    """DLT resection: pose **and** ``cc`` from ≥6 non-coplanar correspondences.

    ``img_pts`` are pixel coordinates; they are converted to metric via
    ``cpar`` (``pixel_to_metric``) before DLT.  Raises with a named message
    when ``<6`` points or near-coplanar.
    """
    ref = np.asarray(ref_pts, float)
    img = np.asarray(img_pts, float)
    if ref.shape[0] < 6:
        raise ValueError(f"seed_from_dlt needs ≥6 points, got {ref.shape[0]}")
    # pixel → metric
    from openptv2.algorithms.trafo import pixel_to_metric as _p2m

    met = np.array([_p2m(float(x), float(y), cpar) for x, y in img], dtype=float)
    R, C, cc = _dlt_resection(ref, met)
    # Calibration stores dm as camera→world, i.e. Rᵀ
    dm_c2w = R.T
    ext = exterior_from_rotation(C, dm_c2w)
    cal = Calibration(
        ext_par=ext,
        int_par=Interior(xh=0.0, yh=0.0, cc=float(cc)),
        glass_par=Glass(n1=1.0, n2=1.0, n3=1.0, d=1.0),
        added_par=AddedPar(),
        mmlut=MmLut(),
    )
    return cal


# ---------------------------------------------------------------------------
# S1.4 rig.yaml → .ori  (the headline first-time-user piece)
# ---------------------------------------------------------------------------


def read_rig(path: str | Path) -> dict[str, Any]:
    """Read and validate a ``rig.yaml`` file.

    Returns the raw dict.  Validation mirrors the plan: ``cameras`` is
    required, each entry requires ``position`` (3) and ``focal_mm`` (>0);
    ``target`` defaults to ``volume_centre``, ``up`` to ``(0,0,1)``, ``roll``
    to ``0``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rig.yaml not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if "cameras" not in raw:
        raise ValueError(f"{p}: missing 'cameras' key")
    cams = raw["cameras"]
    if not isinstance(cams, list) or len(cams) == 0:
        raise ValueError(f"{p}: 'cameras' must be a non-empty list")
    vol = raw.get("volume_centre", [0.0, 0.0, 0.0])
    try:
        vol_a = np.asarray(vol, float).ravel()
        if vol_a.shape != (3,):
            raise ValueError
    except Exception:
        raise ValueError(f"{p}: volume_centre must be length-3, got {vol!r}")
    for i, cam in enumerate(cams):
        if "position" not in cam:
            raise ValueError(f"{p}: cameras[{i}] missing 'position'")
        if "focal_mm" not in cam:
            raise ValueError(
                f"{p}: cameras[{i}] missing 'focal_mm' (M5: there is no safe default)"
            )
        try:
            pos = np.asarray(cam["position"], float).ravel()
            if pos.shape != (3,):
                raise ValueError
        except Exception:
            raise ValueError(
                f"{p}: cameras[{i}].position must be length-3, got {cam['position']!r}"
            )
        try:
            f = float(cam["focal_mm"])
        except Exception:
            raise ValueError(
                f"{p}: cameras[{i}].focal_mm must be a number, got {cam['focal_mm']!r}"
            )
        if f <= 0:
            raise ValueError(f"{p}: cameras[{i}].focal_mm must be >0, got {f}")
        if f < 1 or f > 2000:
            import warnings

            warnings.warn(
                f"{p}: cameras[{i}].focal_mm={f} outside 1-2000 mm — did you mean mm?"
            )
        # optional fields validation
        if "target" in cam:
            try:
                t = np.asarray(cam["target"], float).ravel()
                if t.shape != (3,):
                    raise ValueError
            except Exception:
                raise ValueError(
                    f"{p}: cameras[{i}].target must be length-3, got {cam['target']!r}"
                )
        if "up" in cam:
            try:
                u = np.asarray(cam["up"], float).ravel()
                if u.shape != (3,):
                    raise ValueError
            except Exception:
                raise ValueError(
                    f"{p}: cameras[{i}].up must be length-3, got {cam['up']!r}"
                )
        if "roll" in cam:
            try:
                float(cam["roll"])
            except Exception:
                raise ValueError(
                    f"{p}: cameras[{i}].roll must be a number, got {cam['roll']!r}"
                )
    return raw


def seed_rig(
    path_or_spec: str | Path | dict,
    cpar=None,
) -> dict[int, Calibration]:
    """Build ``{cam_index: Calibration}`` from a ``rig.yaml`` path or dict.

    ``cpar`` is accepted for API symmetry but not currently used (glass
    defaults to air ``1/1/1/1`` unless overridden per-camera).
    """
    if isinstance(path_or_spec, dict):
        raw = path_or_spec
    else:
        raw = read_rig(path_or_spec)
    vol = np.asarray(raw.get("volume_centre", [0.0, 0.0, 0.0]), float).ravel()
    cams = raw["cameras"]
    if (
        cpar is not None
        and hasattr(cpar, "num_cams")
        and len(cams) != int(cpar.num_cams)
    ):
        raise ValueError(
            f"rig.yaml has {len(cams)} cameras but ptv.par/YAML num_cams={cpar.num_cams}"
        )
    out: dict[int, Calibration] = {}
    for i, cam in enumerate(cams):
        pos = np.asarray(cam["position"], float).ravel()
        tgt = np.asarray(cam.get("target", vol), float).ravel()
        up = tuple(cam.get("up", [0.0, 0.0, 1.0]))
        roll = float(cam.get("roll", 0.0))
        f = float(cam["focal_mm"])
        gv = cam.get("glass_vec")
        if gv is not None:
            gv = tuple(gv)
        cal = seed_from_lookat(
            position=pos, target=tgt, focal_mm=f, up=up, roll=roll, glass_vec=gv
        )
        out[i] = cal
    return out


def write_rig_ori(
    rig_path: str | Path,
    dataset_dir: str | Path,
    *,
    overwrite: bool = False,
    cpar=None,
) -> list[Path]:
    """Write ``rig.yaml``-derived ``.ori``/``.addpar`` files into a dataset.

    Resolves output paths with ``autocalibration.cam_files`` so files land
    where every other code path looks.  Refuses to clobber an existing
    ``.ori`` unless ``overwrite=True`` (and then backs up to ``.autobck``).
    Returns the written ``.ori`` paths.
    """
    from openptv2.autocalibration import cam_files

    ds = Path(dataset_dir)
    rp = Path(rig_path)
    # resolve cpar if not given
    if cpar is None:
        from openptv2.autocalibration import _load_dataset_params, resolve_calblock

        try:
            cb = resolve_calblock(ds)
            dp = _load_dataset_params(ds, cb)
            cpar = dp.cpar
        except Exception:
            pass
    rig_cals = seed_rig(rp, cpar=cpar)
    written: list[Path] = []
    for cam_idx, cal in rig_cals.items():
        _, ori, addpar = cam_files(ds, cam_idx)
        if ori.exists() and not overwrite:
            raise FileExistsError(
                f"{ori} already exists — pass overwrite=True to replace "
                f"(backup → {ori}.autobck)"
            )
        ori.parent.mkdir(parents=True, exist_ok=True)
        if ori.exists() and overwrite:
            import shutil

            bck = Path(str(ori) + ".autobck")
            shutil.copy2(ori, bck)
            if addpar.exists():
                shutil.copy2(addpar, Path(str(addpar) + ".autobck"))
        cal.to_file(str(ori), str(addpar))
        written.append(ori)
    return written


# ---------------------------------------------------------------------------
# S1.5 zero-cc guard (imported by the three fallback call sites)
# ---------------------------------------------------------------------------


def guard_cc(cal: Calibration, context: str = "") -> None:
    """Raise if ``cal.int_par.cc <= 0`` — the imaging model divides by ``cc``.

    Call before ``ray_tracing`` / ``imgcoord`` with a user-facing hint.
    """
    cc = float(getattr(cal.int_par, "cc", 0.0) or 0.0)
    if cc <= 0:
        where = f" ({context})" if context else ""
        raise ValueError(
            f"Calibration cc={cc} is not positive{where} — the .ori is missing "
            "or unset (Interior.cc defaults to 0).  Create it with "
            "seed_from_lookat / seed_from_dlt / rig.yaml → write_rig_ori, "
            "or import an existing calibration via calibration_import."
        )

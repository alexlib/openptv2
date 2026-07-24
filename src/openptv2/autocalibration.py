"""Headless, turnkey multi-camera calibration for OpenPTV datasets.

Drives the full calibration pipeline from the standard on-disk inputs — no GUI
point-picking required:

    external_calibration (4 manual seed points from man_ori.par + man_ori.dat)
      -> sortgrid (match the whole 3D calibration body to detected targets)
      -> refine loop (re-sortgrid with the improved orientation, refit)
      -> full_calibration (bundle adjustment; best distortion flag-set by RMS)

Camera image/.ori/.addpar paths and the calblock path are resolved from the
dataset YAML's cal_ori: block (img_cal_name, img_ori, fixp_name) when
present via cam_files()/resolve_calblock() -- that's the same YAML the GUI
and the dataset's own parameters/cal_ori.par read, so there is exactly one
naming convention per dataset, not a separate "camN.tif" convention this
module assumes regardless of what the dataset actually calls its files.
Falls back to the classic cal/camN.tif convention when no YAML exists (e.g.
test_data/test_cavity):

    <dataset>/
      parameters/ptv.par         # control params: cams, image size, pixel, mm
      parameters/sortgrid.par     # matching radius (px)
      parameters/man_ori.par      # 4 calibration-point IDs per camera
      parameters/man_ori.dat      # 4 clicked pixel coords per camera
      cal/target_on_a_side.txt    # 3D calibration body (id x y z)  [calblock]
      cal/camN.tif                # calibration image per camera
      cal/camN.tif.ori/.addpar    # existing calibration (used as initial guess)
      cal/camN.tif_targets        # detected targets per camera

This module is pure-Python orchestration over the compiled `algorithms`
modules; it is imported by `scripts/autocalibrate.py` and the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.orientation import (
    external_calibration,
    full_calibration,
    read_man_ori_fix,
)
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.sortgrid import read_calblock, sortgrid
from openptv2.algorithms.tracking_frame_buf import read_targets
from openptv2.algorithms.trafo import metric_to_pixel

# Distortion flag-sets tried per camera; lowest reprojection RMS wins.
CANDIDATE_FLAGS: list[list[str]] = [
    ["cc", "xh", "yh"],
    ["cc", "xh", "yh", "k1", "k2"],
    ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2"],
    # + the glass-interface tilt (interf). A tilted refractive wall bends rays
    # into a keystone the radial+decentering Brown model can't represent; on a
    # real splitter rig this recovered a camera stuck at 2.4px down to ~1.1px
    # once the glass vector was allowed to tilt off the optical axis. Kept as a
    # separate candidate so the best-RMS selection only adopts it when it
    # actually helps (an untilted interface leaves the glass vector put).
    ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2", "interf"],
]

REFINE_ITERS = 3  # sortgrid<->refit passes to grow the matched set


@dataclass
class CamResult:
    """Per-camera calibration outcome."""

    cam: int  # 0-based
    matched: int
    nfix: int
    rms: float
    flags: list[str]
    cal: Calibration | None
    ref: np.ndarray = field(repr=False)  # (n,3) matched 3D points
    det: np.ndarray = field(repr=False)  # (n,2) detected pixels
    rep: np.ndarray = field(repr=False)  # (n,2) reprojected pixels
    error: str | None = None  # set instead of raising when this camera failed


def _reproject_px(cal, mm, fix_xyz, cpar):
    xp, yp = img_coord(fix_xyz, cal, mm)
    return metric_to_pixel(
        xp, yp, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield
    )


def _matched_pairs(cal, cpar, fix, sorted_pix):
    ref, det, rep = [], [], []
    for i, t in enumerate(sorted_pix):
        if t.pnr < 0:
            continue
        ref.append(fix[i])
        det.append((t.x, t.y))
        rep.append(_reproject_px(cal, cpar.mm, fix[i], cpar))
    ref_arr = np.asarray(ref) if len(ref) else np.empty((0, 3))
    det_arr = np.asarray(det) if len(det) else np.empty((0, 2))
    rep_arr = np.asarray(rep) if len(rep) else np.empty((0, 2))
    return ref_arr, det_arr, rep_arr


def rms_px(det, rep) -> float:
    """Root-mean-square reprojection error in pixels."""
    if len(det) == 0:
        return float("inf")
    d = np.asarray(det) - np.asarray(rep)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _find_yaml(base: Path) -> Path | None:
    """Preferred dataset YAML: parameters_Run1.yaml, else first parameters_*.yaml."""
    pref = base / "parameters_Run1.yaml"
    if pref.exists():
        return pref
    cands = sorted(base.glob("parameters_*.yaml"))
    return cands[0] if cands else None


def _cal_ori_yaml(base: Path) -> dict:
    """The dataset YAML's cal_ori: block, or {} if there is no YAML / no block."""
    yaml_path = _find_yaml(base)
    if yaml_path is None:
        return {}
    y = yaml.safe_load(yaml_path.read_text()) or {}
    return y.get("cal_ori") or {}


def cam_files(base: Path, cam: int) -> tuple[Path, Path, Path]:
    """Resolve (image, .ori, .addpar) paths for one camera.

    The dataset YAML's cal_ori.img_cal_name / img_ori are the source of
    truth when present (that's what the GUI and the dataset's own
    parameters/cal_ori.par actually reference) -- reading them here instead
    of assuming a fixed cal/camN.tif naming means the same files the GUI
    calibrates are the ones this module reads and writes, with no separate
    adapter-copy naming convention to keep in sync (and no risk of the two
    silently drifting apart, as happened when the adapter copies for a real
    dataset were cleaned up as apparent clutter and the corrupted GUI result
    then had no local backup).

    Falls back to cal/cam{cam+1}.tif(.ori/.addpar) when no YAML exists or it
    doesn't have cal_ori.img_cal_name/img_ori for this camera (e.g. the
    classic test_data/test_cavity fixture, which has no YAML at all).

    Splitter datasets (cal_ori.cal_splitter: true) put only camera 0's real
    combined-frame path in img_cal_name; cameras 1+ use the '---' placeholder
    (the same convention test_data/test_splitter's own committed YAML uses)
    since there's no separate physical image per camera -- they all read the
    same raw frame and split it in memory. img_ori is still fully specified
    per camera (each gets its own .ori/.addpar/_targets), so a '---' image
    entry falls back to img_cal_name[0] rather than being treated as a
    literal (nonexistent) filename.
    """
    cal_ori = _cal_ori_yaml(base)
    img_cal_name = cal_ori.get("img_cal_name") or []
    img_ori = cal_ori.get("img_ori") or []
    if cam < len(img_cal_name) and cam < len(img_ori) and img_cal_name[cam] and img_ori[cam]:
        img_name = img_cal_name[cam] if img_cal_name[cam] != "---" else img_cal_name[0]
        img = base / img_name
        ori = base / img_ori[cam]
        addpar = ori.with_suffix(ori.suffix + ".addpar") if ori.suffix != ".ori" else ori.with_suffix(".addpar")
        # ori path is typically "....tif.ori"; addpar is the same stem with
        # ".ori" replaced by ".addpar", not simply swapping the last suffix.
        addpar = Path(str(ori)[: -len(".ori")] + ".addpar") if str(ori).endswith(".ori") else addpar
        return img, ori, addpar
    stem = base / "cal" / f"cam{cam + 1}.tif"
    return stem, stem.with_suffix(".tif.ori"), stem.with_suffix(".tif.addpar")


def target_base(base: Path, cam: int) -> Path:
    """Resolve the per-camera base name for that camera's `_targets` file.

    Always distinct per camera, even in splitter mode where `cam_files()`'s
    image path is intentionally the SAME shared raw frame for every camera
    (there's no separate physical image to read). Deriving the targets name
    from `img` there would silently collide across all 4 cameras -- every
    camera's detection would overwrite the same file, and calibration would
    read cam N's detections back for every other camera too. Derive from
    `.ori` instead (already guaranteed per-camera by cam_files()) by
    stripping its suffix, e.g. cal/cam_1.tif.ori -> cal/cam_1.tif (then
    read_targets/write_targets append the _targets suffix).
    """
    _, ori, _ = cam_files(base, cam)
    ori_str = str(ori)
    return Path(ori_str[: -len(".ori")]) if ori_str.endswith(".ori") else ori


def resolve_calblock(base: Path) -> Path:
    """Resolve the 3D calibration-body (calblock) path.

    cal_ori.fixp_name in the dataset YAML is the source of truth when
    present; falls back to the legacy cal/target_on_a_side.txt convention.
    """
    fixp_name = _cal_ori_yaml(base).get("fixp_name")
    if fixp_name:
        return base / fixp_name
    return base / "cal" / "target_on_a_side.txt"


@dataclass
class DatasetParams:
    cpar: object
    num_cams: int
    eps: int
    ids_per_cam: list[list[int]]      # 4 calibration-point IDs per camera
    clicks_per_cam: list[np.ndarray]  # (4,2) pixel seed clicks per camera
    source: str                        # "yaml" or "par"


def _cpar_from_ptv(ptv: dict, num_cams: int):
    """Build a ControlPar from a YAML 'ptv' block (same fields as the .par reader)."""
    mm = MmNp(
        nlay=1,
        n1=float(ptv["mmp_n1"]),
        n2=[float(ptv["mmp_n2"]), 1.0, 1.0],
        d=[float(ptv["mmp_d"]), 0.0, 0.0],
        n3=float(ptv["mmp_n3"]),
    )
    return ControlPar(
        num_cams=num_cams,
        hp_flag=int(ptv.get("hp_flag", 0)),
        allCam_flag=int(ptv.get("allcam_flag", 0)),
        tiff_flag=int(ptv.get("tiff_flag", 1)),
        imx=int(ptv["imx"]),
        imy=int(ptv["imy"]),
        pix_x=float(ptv["pix_x"]),
        pix_y=float(ptv["pix_y"]),
        chfield=int(ptv.get("chfield", 0)),
        mm=mm,
    )


def _seed_from_par(base: Path, num_cams: int, calblock: Path):
    """Fallback seed source: man_ori.par (IDs) + man_ori.dat (clicks)."""
    par = base / "parameters"
    dat_file = (par / "man_ori.dat") if (par / "man_ori.dat").exists() else (base / "man_ori.dat")
    par_file = (par / "man_ori.par") if (par / "man_ori.par").exists() else (base / "man_ori.par")
    clicks = np.loadtxt(dat_file).reshape(-1, 2)
    ids_per_cam, clicks_per_cam = [], []
    for cam in range(num_cams):
        fix4 = read_man_ori_fix(str(calblock), str(par_file), cam)
        if fix4 is None:
            raise RuntimeError(f"cam{cam + 1}: could not read man_ori.par IDs")
        # read_man_ori_fix returns 3D coords; recover the IDs from man_ori.par
        toks = par_file.read_text().split()
        ids_per_cam.append([int(toks[cam * 4 + i]) for i in range(4)])
        clicks_per_cam.append(clicks[cam * 4:(cam + 1) * 4])
    return ids_per_cam, clicks_per_cam


def _load_dataset_params(base: Path, calblock: Path) -> DatasetParams:
    """Resolve control params + calibration seed, YAML-first with .par fallback.

    The processing parameters (ptv, sortgrid) and the manual-orientation seed
    (man_ori ids + pixel clicks) are read from the dataset YAML when present;
    legacy .par/.dat files are used only as a fallback (backward compatibility).
    """
    yaml_path = _find_yaml(base)
    if yaml_path is not None:
        y = yaml.safe_load(yaml_path.read_text())
        num_cams = int(y.get("num_cams") or y["ptv"].get("num_cams"))
        cpar = _cpar_from_ptv(y["ptv"], num_cams)
        eps = int(y.get("sortgrid", {}).get("radius", 0))
        mo = y.get("man_ori") or {}
        coords = y.get("man_ori_coordinates") or {}
        if eps and mo.get("nr") and coords:
            nr = [int(v) for v in mo["nr"]]
            ids_per_cam, clicks_per_cam = [], []
            for cam in range(num_cams):
                ids_per_cam.append(nr[cam * 4:(cam + 1) * 4])
                pts = coords[f"camera_{cam}"]
                clicks_per_cam.append(
                    np.array([[pts[f"point_{k}"]["x"], pts[f"point_{k}"]["y"]]
                              for k in range(1, 5)], dtype=float)
                )
            return DatasetParams(cpar, num_cams, eps, ids_per_cam,
                                 clicks_per_cam, "yaml")
        # YAML present but lacks a usable seed/sortgrid -> fall back for those
        if not eps:
            eps = int((base / "parameters" / "sortgrid.par").read_text().strip())
        ids_per_cam, clicks_per_cam = _seed_from_par(base, num_cams, calblock)
        return DatasetParams(cpar, num_cams, eps, ids_per_cam, clicks_per_cam,
                             "yaml+par-seed")

    # No YAML at all: pure legacy .par path.
    par = base / "parameters"
    cpar = ControlPar.from_file(str(par / "ptv.par"))
    num_cams = cpar.num_cams
    eps = int((par / "sortgrid.par").read_text().strip())
    ids_per_cam, clicks_per_cam = _seed_from_par(base, num_cams, calblock)
    return DatasetParams(cpar, num_cams, eps, ids_per_cam, clicks_per_cam, "par")


def calibrate_camera(
    cam: int,
    base: Path,
    cpar,
    fix: np.ndarray,
    nfix: int,
    eps: int,
    fix4: np.ndarray,
    pix4: np.ndarray,
) -> CamResult:
    """Run the full calibration pipeline for a single camera.

    fix4: (4,3) 3D coords of the manual-orientation seed points.
    pix4: (4,2) pixel clicks for those seed points.
    """
    img, ori, addpar = cam_files(base, cam)

    fix4 = np.asarray(fix4, float)
    pix4 = np.asarray(pix4, float)
    if pix4.shape[0] < 4 or fix4.shape[0] < 4:
        raise RuntimeError(f"cam{cam + 1}: need 4 seed points, got {pix4.shape[0]}")

    pix = read_targets(str(target_base(base, cam)), 0)
    if not pix:
        raise RuntimeError(f"cam{cam + 1}: no detected targets found")

    def _seeded():
        c = Calibration.from_file(str(ori), str(addpar))
        if not external_calibration(c, fix4, pix4, cpar):
            raise RuntimeError(f"cam{cam + 1}: external_calibration did not converge")
        return c

    # Refine exterior: sortgrid -> fit -> re-sortgrid until matches stabilize.
    cal = _seeded()
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
    n_matched = sum(1 for t in sorted_pix if t.pnr >= 0)
    for _ in range(REFINE_ITERS):
        try:
            full_calibration(cal, fix, sorted_pix, cpar, ["cc", "xh", "yh"])
        except (ValueError, RuntimeError):
            break
        sp = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
        n = sum(1 for t in sp if t.pnr >= 0)
        sorted_pix = sp
        if n <= n_matched:
            n_matched = n
            break
        n_matched = n

    # Final bundle adjustment: pick the flag-set with lowest reprojection RMS.
    best = None
    for flags in CANDIDATE_FLAGS:
        trial = _seeded()
        try:
            full_calibration(trial, fix, sorted_pix, cpar, flags)
        except (ValueError, RuntimeError):
            continue
        ref, det, rep = _matched_pairs(trial, cpar, fix, sorted_pix)
        r = rms_px(det, rep)
        if best is None or r < best[0]:
            best = (r, trial, flags, ref, det, rep)

    if best is None:
        raise RuntimeError(f"cam{cam + 1}: no flag-set converged")
    r, cal_best, flags, ref, det, rep = best
    return CamResult(cam, n_matched, nfix, r, flags, cal_best, ref, det, rep)


def calibrate_dataset(
    dataset_dir: str | Path,
    *,
    write: bool = False,
    overlays: bool = False,
    outdir: str | Path | None = None,
) -> list[CamResult]:
    """Calibrate every camera in a dataset.

    Args:
        dataset_dir: dataset root (contains cal/ and parameters/).
        write: overwrite cal/camN.tif.ori/.addpar (backups saved as *.autobck).
        overlays: save detected-vs-reprojected PNGs into outdir.
        outdir: where overlays go (default: <dataset>/cal/auto_calib).

    Returns:
        One CamResult per camera.
    """
    base = Path(dataset_dir).resolve()

    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))

    dp = _load_dataset_params(base, calblock)
    cpar, num_cams, eps = dp.cpar, dp.num_cams, dp.eps

    out = Path(outdir) if outdir else base / "cal" / "auto_calib"
    if overlays:
        out.mkdir(parents=True, exist_ok=True)

    results: list[CamResult] = []
    for cam in range(num_cams):
        ids = dp.ids_per_cam[cam]
        fix4 = np.asarray([fix[i - 1] for i in ids], dtype=float)
        try:
            res = calibrate_camera(
                cam, base, cpar, fix, nfix, eps, fix4, dp.clicks_per_cam[cam]
            )
        except RuntimeError as exc:
            # One camera's seed/initial-guess failing to converge (common on a
            # freshly bootstrapped naive guess) shouldn't lose every other
            # camera's result -- report it and keep going.
            results.append(CamResult(
                cam=cam, matched=0, nfix=nfix, rms=float("inf"), flags=[],
                cal=None, ref=np.empty((0, 3)), det=np.empty((0, 2)),
                rep=np.empty((0, 2)), error=str(exc),
            ))
            continue
        results.append(res)

        if overlays:
            save_overlay(res, base, out)
        if write:
            _, ori, addpar = cam_files(base, cam)
            import shutil

            shutil.copy2(ori, Path(str(ori) + ".autobck"))
            shutil.copy2(addpar, Path(str(addpar) + ".autobck"))
            res.cal.write(str(ori).encode(), str(addpar).encode())

    return results


def cross_camera_rcm(results: list[CamResult], cpar) -> dict | None:
    """Cross-camera ray-convergence miss distance (mm) over calblock points
    seen by >= 2 cameras. None when < 2 cameras have a valid result or < 3
    common points. Per-camera reprojection RMS cannot see cross-camera
    inconsistency; this can."""
    from openptv2.algorithms.orientation import COORD_UNUSED
    from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
    from openptv2.orientation import multi_cam_point_positions

    valid = [r for r in results
             if r.cal is not None and r.error is None and len(r.ref) > 0]
    if len(valid) < 2:
        return None

    n_cams = cpar.num_cams
    cal_by_cam = {r.cam: r.cal for r in valid}

    # 3D calblock point -> {cam: (px, py)}
    point_pixels: dict[tuple, dict[int, tuple]] = {}
    for r in valid:
        for ref_row, det_row in zip(r.ref, r.det):
            key = tuple(np.round(ref_row, 3))
            point_pixels.setdefault(key, {})[r.cam] = (det_row[0], det_row[1])

    seen = [pix for pix in point_pixels.values() if len(pix) >= 2]
    if len(seen) < 3:
        return None
    n_common = sum(1 for pix in point_pixels.values() if len(pix) == n_cams)

    targets = np.full((len(seen), n_cams, 2), COORD_UNUSED, dtype=np.float64)
    for i, pix in enumerate(seen):
        for cam, (px, py) in pix.items():
            mx, my = pixel_to_metric(px, py, cpar)
            cal = cal_by_cam[cam]
            fx, fy = dist_to_flat(
                mx, my, cal.int_par.xh, cal.int_par.yh,
                cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
                cal.added_par.p1, cal.added_par.p2,
                cal.added_par.scx, cal.added_par.she,
            )
            targets[i, cam] = (fx, fy)

    cals = [cal_by_cam.get(c) or next(iter(cal_by_cam.values()))
            for c in range(n_cams)]
    _pos, rcm = multi_cam_point_positions(targets, cpar, cals)
    return {
        "n_points": int(len(rcm)),
        "n_common": int(n_common),
        "median": float(np.median(rcm)),
        "p90": float(np.percentile(rcm, 90)),
        "p95": float(np.percentile(rcm, 95)),
        "max": float(np.max(rcm)),
    }


def save_overlay(res: CamResult, base: Path, outdir: Path) -> Path:
    """Save a detected-vs-reprojected overlay PNG for one camera."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6.4))
    img_path, _, _ = cam_files(base, res.cam)
    try:
        import imageio.v3 as iio

        ax.imshow(iio.imread(img_path), cmap="gray")
    except Exception:
        ax.invert_yaxis()
    ax.scatter(res.det[:, 0], res.det[:, 1], s=40, facecolors="none",
               edgecolors="lime", linewidths=1.2, label="detected")
    ax.scatter(res.rep[:, 0], res.rep[:, 1], s=8, c="red", label="reprojected")
    ax.set_title(
        f"cam{res.cam + 1}  RMS={res.rms:.3f}px  n={res.matched}/{res.nfix}  "
        f"flags={'+'.join(res.flags)}"
    )
    ax.legend(loc="upper right", framealpha=0.7)
    fig.tight_layout()
    dest = outdir / f"cam{res.cam + 1}_overlay.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest

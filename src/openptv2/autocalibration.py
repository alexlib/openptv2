"""Headless, turnkey multi-camera calibration for OpenPTV datasets.

Drives the full calibration pipeline from the standard on-disk inputs — no GUI
point-picking required:

    external_calibration (4 manual seed points from man_ori.par + man_ori.dat)
      -> sortgrid (match the whole 3D calibration body to detected targets)
      -> refine loop (re-sortgrid with the improved orientation, refit)
      -> full_calibration (bundle adjustment; best distortion flag-set by RMS)

Expected dataset layout (classic OpenPTV, e.g. test_data/test_cavity):

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

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.orientation import (
    external_calibration,
    full_calibration,
    read_man_ori_fix,
)
from openptv2.algorithms.parameters import ControlPar
from openptv2.algorithms.sortgrid import read_calblock, sortgrid
from openptv2.algorithms.tracking_frame_buf import read_targets
from openptv2.algorithms.trafo import metric_to_pixel

# Distortion flag-sets tried per camera; lowest reprojection RMS wins.
CANDIDATE_FLAGS: list[list[str]] = [
    ["cc", "xh", "yh"],
    ["cc", "xh", "yh", "k1", "k2"],
    ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2"],
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
    cal: Calibration
    ref: np.ndarray = field(repr=False)  # (n,3) matched 3D points
    det: np.ndarray = field(repr=False)  # (n,2) detected pixels
    rep: np.ndarray = field(repr=False)  # (n,2) reprojected pixels


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
    return np.asarray(ref), np.asarray(det), np.asarray(rep)


def rms_px(det, rep) -> float:
    """Root-mean-square reprojection error in pixels."""
    if len(det) == 0:
        return float("inf")
    d = np.asarray(det) - np.asarray(rep)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def calibrate_camera(
    cam: int,
    base: Path,
    cpar,
    fix: np.ndarray,
    nfix: int,
    eps: int,
    calblock: Path,
    man_par: Path,
    man_dat: Path,
) -> CamResult:
    """Run the full calibration pipeline for a single camera."""
    ori = base / "cal" / f"cam{cam + 1}.tif.ori"
    addpar = base / "cal" / f"cam{cam + 1}.tif.addpar"

    fix4 = read_man_ori_fix(str(calblock), str(man_par), cam)
    if fix4 is None:
        raise RuntimeError(f"cam{cam + 1}: could not read man_ori point IDs")
    fix4 = np.asarray(fix4, float)
    clicks = np.loadtxt(man_dat).reshape(-1, 2)
    pix4 = clicks[cam * 4:(cam + 1) * 4]
    if pix4.shape[0] < 4:
        raise RuntimeError(f"cam{cam + 1}: man_ori.dat has <4 clicks")

    pix = read_targets(str(base / "cal" / f"cam{cam + 1}.tif"), 0)
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
    par = base / "parameters"
    cpar = ControlPar.from_file(str(par / "ptv.par"))
    num_cams = cpar.num_cams
    eps = int((par / "sortgrid.par").read_text().strip())

    calblock = base / "cal" / "target_on_a_side.txt"
    fix, nfix = read_calblock(str(calblock))
    man_par = par / "man_ori.par"
    man_dat = par / "man_ori.dat"

    out = Path(outdir) if outdir else base / "cal" / "auto_calib"
    if overlays:
        out.mkdir(parents=True, exist_ok=True)

    results: list[CamResult] = []
    for cam in range(num_cams):
        res = calibrate_camera(
            cam, base, cpar, fix, nfix, eps, calblock, man_par, man_dat
        )
        results.append(res)

        if overlays:
            save_overlay(res, base, out)
        if write:
            ori = base / "cal" / f"cam{cam + 1}.tif.ori"
            addpar = base / "cal" / f"cam{cam + 1}.tif.addpar"
            import shutil

            shutil.copy2(ori, ori.with_suffix(".ori.autobck"))
            shutil.copy2(addpar, addpar.with_suffix(".addpar.autobck"))
            res.cal.write(str(ori).encode(), str(addpar).encode())

    return results


def save_overlay(res: CamResult, base: Path, outdir: Path) -> Path:
    """Save a detected-vs-reprojected overlay PNG for one camera."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6.4))
    img_path = base / "cal" / f"cam{res.cam + 1}.tif"
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

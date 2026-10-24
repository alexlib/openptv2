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

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def rms_px(det: np.ndarray, rep: np.ndarray) -> float:
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
    if (
        cam < len(img_cal_name)
        and cam < len(img_ori)
        and img_cal_name[cam]
        and img_ori[cam]
    ):
        img_name = img_cal_name[cam] if img_cal_name[cam] != "---" else img_cal_name[0]
        img = base / img_name
        ori = base / img_ori[cam]
        addpar = (
            ori.with_suffix(ori.suffix + ".addpar")
            if ori.suffix != ".ori"
            else ori.with_suffix(".addpar")
        )
        # ori path is typically "....tif.ori"; addpar is the same stem with
        # ".ori" replaced by ".addpar", not simply swapping the last suffix.
        addpar = (
            Path(str(ori)[: -len(".ori")] + ".addpar")
            if str(ori).endswith(".ori")
            else addpar
        )
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
        return base / str(fixp_name)
    return base / "cal" / "target_on_a_side.txt"


@dataclass
class DatasetParams:
    cpar: object
    num_cams: int
    eps: int
    ids_per_cam: list[list[int]]  # 4 calibration-point IDs per camera
    clicks_per_cam: list[np.ndarray]  # (4,2) pixel seed clicks per camera
    source: str  # "yaml" or "par"


def _cpar_from_ptv(ptv: dict, num_cams: int) -> ControlPar:
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
        pair_flag=int(bool(ptv.get("pair_flag", ptv.get("use_pairs", False)))),
        tiff_flag=int(ptv.get("tiff_flag", 1)),
        imx=int(ptv["imx"]),
        imy=int(ptv["imy"]),
        pix_x=float(ptv["pix_x"]),
        pix_y=float(ptv["pix_y"]),
        chfield=int(ptv.get("chfield", 0)),
        mm=mm,
    )


def _seed_from_par(
    base: Path, num_cams: int, calblock: Path
) -> tuple[list[list[int]], list[np.ndarray]]:
    """Fallback seed source: man_ori.par (IDs) + man_ori.dat (clicks)."""
    par = base / "parameters"
    dat_file = (
        (par / "man_ori.dat")
        if (par / "man_ori.dat").exists()
        else (base / "man_ori.dat")
    )
    par_file = (
        (par / "man_ori.par")
        if (par / "man_ori.par").exists()
        else (base / "man_ori.par")
    )
    clicks = np.loadtxt(dat_file).reshape(-1, 2)
    ids_per_cam, clicks_per_cam = [], []
    for cam in range(num_cams):
        fix4 = read_man_ori_fix(str(calblock), str(par_file), cam)
        if fix4 is None:
            raise RuntimeError(f"cam{cam + 1}: could not read man_ori.par IDs")
        # read_man_ori_fix returns 3D coords; recover the IDs from man_ori.par
        toks = par_file.read_text().split()
        ids_per_cam.append([int(toks[cam * 4 + i]) for i in range(4)])
        clicks_per_cam.append(clicks[cam * 4 : (cam + 1) * 4])
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
                ids_per_cam.append(nr[cam * 4 : (cam + 1) * 4])
                pts = coords[f"camera_{cam}"]
                clicks_per_cam.append(
                    np.array(
                        [
                            [pts[f"point_{k}"]["x"], pts[f"point_{k}"]["y"]]
                            for k in range(1, 5)
                        ],
                        dtype=float,
                    )
                )
            return DatasetParams(
                cpar, num_cams, eps, ids_per_cam, clicks_per_cam, "yaml"
            )
        # YAML present but lacks a usable seed/sortgrid -> fall back for those
        if not eps:
            eps = int((base / "parameters" / "sortgrid.par").read_text().strip())
        ids_per_cam, clicks_per_cam = _seed_from_par(base, num_cams, calblock)
        return DatasetParams(
            cpar, num_cams, eps, ids_per_cam, clicks_per_cam, "yaml+par-seed"
        )

    # No YAML at all: pure legacy .par path.
    par = base / "parameters"
    cpar = ControlPar.from_file(str(par / "ptv.par"))
    num_cams = cpar.num_cams
    eps = int((par / "sortgrid.par").read_text().strip())
    ids_per_cam, clicks_per_cam = _seed_from_par(base, num_cams, calblock)
    return DatasetParams(cpar, num_cams, eps, ids_per_cam, clicks_per_cam, "par")


def _tpar_from_dataset(base: Path) -> Any:
    """Load TargetPar from dataset YAML or legacy targ_rec.par."""
    from openptv2.algorithms.parameters import TargetPar

    yaml_path = _find_yaml(base)
    if yaml_path is not None:
        y = yaml.safe_load(yaml_path.read_text()) or {}
        tr = y.get("targ_rec") or y.get("detect_plate") or {}
        if tr:
            gvthres = tr.get("gvthres") or [
                tr.get("gvth_1", 10),
                tr.get("gvth_2", 10),
                tr.get("gvth_3", 10),
                tr.get("gvth_4", 10),
            ]
            return TargetPar(
                discont=int(tr.get("disco", tr.get("tol_dis", 100))),
                nnmin=int(tr.get("nnmin", tr.get("min_npix", 4))),
                nnmax=int(tr.get("nnmax", tr.get("max_npix", 100))),
                nxmin=int(tr.get("nxmin", tr.get("min_npix_x", 2))),
                nxmax=int(tr.get("nxmax", tr.get("max_npix_x", 100))),
                nymin=int(tr.get("nymin", tr.get("min_npix_y", 2))),
                nymax=int(tr.get("nymax", tr.get("max_npix_y", 100))),
                sumg_min=int(tr.get("sumg_min", tr.get("sum_grey", 100))),
                cr_sz=int(tr.get("cr_sz", tr.get("size_cross", 2))),
                gvthres=list(map(int, gvthres)),
            )
    par = base / "parameters" / "targ_rec.par"
    if par.exists():
        return TargetPar.from_file(str(par))
    return None


def _refine_and_select(
    cam: int,
    cal: Calibration,
    cpar: Any,
    fix: np.ndarray,
    nfix: int,
    eps: int,
    pix: list,
    *,
    presorted: bool = False,
) -> CamResult:
    """Refine an already-seeded calibration and pick the best distortion flag-set.

    `cal` must already hold a reasonably-close pose -- sortgrid needs to
    match most of `pix` against `fix` within `eps` pixels to make progress.
    Shared by every calibration source (see
    `openptv2.calibration_registry.CALIBRATION_SOURCE_REGISTRY`): only how
    `cal`/`fix`/`pix` get built differs. `calibrate_camera` below builds
    them from a calblock + manual/existing seed; `calibrate_from_source`
    builds them from any other registered source.

    When ``presorted=True`` the point sets are already matched
    (``pix[i].pnr == i`` and aligned with ``fix[i]``) — skip all
    ``sortgrid`` calls and the coarse pre-pass, then go straight to the
    ``CANDIDATE_FLAGS`` loop.
    """
    if presorted:
        # Already matched: pix is index-aligned with fix via _target_from_xy
        sorted_pix = pix
        n_matched = len(pix)
    else:
        # Coarse sortgrid pass to align overall plate orientation
        sp_coarse = sortgrid(cal, cpar, nfix, fix, len(pix), max(15, eps), pix)
        try:
            full_calibration(cal, fix, sp_coarse, cpar, ["cc", "xh", "yh"])
        except (ValueError, RuntimeError):
            pass

        # Refine exterior at target radius: sortgrid -> fit -> re-sortgrid until matches stabilize.
        sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
        n_matched = sum(1 for t in sorted_pix if t.pnr >= 0)
        for _ in range(REFINE_ITERS + 2):
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

    # Final bundle adjustment: pick the flag-set with lowest reprojection RMS starting from refined cal.
    best = None
    for flags in CANDIDATE_FLAGS:
        trial = copy.deepcopy(cal)
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


def calibrate_camera(
    cam: int,
    base: Path,
    cpar: Any,
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
        # Auto-detect targets from the calibration image if _targets file is missing
        tpar = _tpar_from_dataset(base)
        if tpar is not None and img.exists():
            from imageio.v3 import imread
            from skimage.color import rgb2gray
            from skimage.util import img_as_ubyte

            from openptv2.algorithms.tracking_frame_buf import write_targets
            from openptv2.segmentation import target_recognition

            raw_img = _cam_view(base, cam, imread(img))
            if raw_img.ndim == 3:
                raw_img = rgb2gray(raw_img)
            raw_img = img_as_ubyte(raw_img)
            from openptv2.image_processing import preprocess_image

            hp_img = preprocess_image(raw_img, cpar.hp_flag or 1, cpar, 25)
            detected = target_recognition(hp_img, tpar, cam, cpar)
            if detected:
                write_targets(detected, len(detected), str(target_base(base, cam)), 0)
                pix = read_targets(str(target_base(base, cam)), 0)

    if not pix:
        raise RuntimeError(f"cam{cam + 1}: no detected targets found")

    def _seeded():
        c = Calibration.from_file(str(ori), str(addpar))
        if not external_calibration(c, fix4, pix4, cpar):
            raise RuntimeError(f"cam{cam + 1}: external_calibration did not converge")
        return c

    # Try existing .ori first if it yields reasonable matches, else fallback to external_calibration seed
    cal = Calibration.from_file(str(ori), str(addpar))
    sp_test = sortgrid(cal, cpar, nfix, fix, len(pix), max(15, eps), pix)
    n_test = sum(1 for t in sp_test if t.pnr >= 0)
    if n_test < 10:
        cal = _seeded()

    return _refine_and_select(cam, cal, cpar, fix, nfix, eps, pix)


def calibrate_from_source(
    source_name: str,
    cam: int,
    cpar: Any,
    point_set: Any,
    eps: int = 15,
    *,
    initial_cal: Calibration | None = None,
    fix4: np.ndarray | None = None,
    pix4: np.ndarray | None = None,
    presorted: bool = False,
) -> CamResult:
    """Calibrate one camera from any registered calibration source.

    `point_set` is a `openptv2.calibration_registry.CalibrationPointSet`
    (ref_pts/img_pts/optional seed) produced by the named source. Three ways
    to get a starting pose, tried in order:

    1. `point_set.seed` -- the source already recovered a usable pose (e.g.
        a homography-derived seed) and no further bootstrapping is needed.
        Use this for doors that carry a converted distortion model (OpenCV
        ``k1..p2``): ``external_calibration``/``raw_orient`` zeroes all
        distortion before solving (``orientation.py:540-546``), so path 2
        would **destroy** the imported model — door B must use path 1.
    2. `initial_cal` + `fix4`/`pix4` -- an approximate interior guess
        (camera constant/principal point; `external_calibration` only
        adjusts the 6 exterior parameters, so `cc`/`xh`/`yh` must already be
        plausible on `initial_cal`, e.g. from an existing .ori or a rough
        guess from the sensor size) plus a 4-point manual/known seed,
        matching today's calblock path (`calibrate_camera`'s `_seeded()`).
    3. Neither -- raises, rather than bootstrapping from a Calibration()
        with cc=0.0 (Interior's default), which external_calibration cannot
        usefully refine (division-by-camera-constant is undefined there).

    ``presorted`` skips ``sortgrid`` (point sets already matched, e.g. door A
    ``read_xyXYZ`` output) and is threaded to ``_refine_and_select``.

    This is the multi-source counterpart to `calibrate_camera`: same
    `_refine_and_select` core, different (and pluggable) point acquisition.
    """
    from openptv2.calibration_registry import get_source_info

    get_source_info(source_name)  # KeyError if unregistered -- fail fast

    ref_pts = np.asarray(point_set.ref_pts, float)
    img_pts = np.asarray(point_set.img_pts, float)
    nfix = ref_pts.shape[0]
    if nfix == 0:
        raise RuntimeError(f"cam{cam + 1}: {source_name} produced no correspondences")

    pix = [_target_from_xy(i, xy[0], xy[1]) for i, xy in enumerate(img_pts)]

    if point_set.seed is not None:
        cal = copy.deepcopy(point_set.seed)
    elif initial_cal is not None and fix4 is not None and pix4 is not None:
        cal = copy.deepcopy(initial_cal)
        if not external_calibration(
            cal, np.asarray(fix4, float), np.asarray(pix4, float), cpar
        ):
            raise RuntimeError(f"cam{cam + 1}: external_calibration did not converge")
    else:
        raise RuntimeError(
            f"cam{cam + 1}: {source_name} has no seed pose; pass either a "
            "point_set with .seed set, or initial_cal + fix4 + pix4"
        )

    return _refine_and_select(
        cam, cal, cpar, ref_pts, nfix, eps, pix, presorted=presorted
    )


def _target_from_xy(pnr: int, x: float, y: float) -> Any:
    """Build a Target with the given pixel coordinates (helper for
    calibrate_from_source, which starts from plain (n,2) arrays rather than
    a `_targets` file)."""
    from openptv2.algorithms.tracking_frame_buf import Target

    t = Target()
    t.pnr = pnr
    t.x = x
    t.y = y
    return t


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
            results.append(
                CamResult(
                    cam=cam,
                    matched=0,
                    nfix=nfix,
                    rms=float("inf"),
                    flags=[],
                    cal=None,
                    ref=np.empty((0, 3)),
                    det=np.empty((0, 2)),
                    rep=np.empty((0, 2)),
                    error=str(exc),
                )
            )
            continue
        results.append(res)

        if overlays:
            save_overlay(res, base, out)
        if write:
            _, ori, addpar = cam_files(base, cam)
            import shutil

            shutil.copy2(ori, Path(str(ori) + ".autobck"))
            shutil.copy2(addpar, Path(str(addpar) + ".autobck"))
            if res.cal is not None:
                res.cal.write(str(ori).encode(), str(addpar).encode())

    return results


def cross_camera_rcm(results: list[CamResult], cpar: Any) -> dict | None:
    """Cross-camera ray-convergence miss distance (mm) over calblock points
    seen by >= 2 cameras. None when < 2 cameras have a valid result or < 3
    common points. Per-camera reprojection RMS cannot see cross-camera
    inconsistency; this can."""
    from openptv2.algorithms.orientation import COORD_UNUSED
    from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
    from openptv2.orientation import multi_cam_point_positions

    valid = [
        r for r in results if r.cal is not None and r.error is None and len(r.ref) > 0
    ]
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
            if cal is None:
                continue
            fx, fy = dist_to_flat(
                mx,
                my,
                cal.int_par.xh,
                cal.int_par.yh,
                cal.added_par.k1,
                cal.added_par.k2,
                cal.added_par.k3,
                cal.added_par.p1,
                cal.added_par.p2,
                cal.added_par.scx,
                cal.added_par.she,
            )
            targets[i, cam] = (fx, fy)

    cals = [cal_by_cam.get(c) or next(iter(cal_by_cam.values())) for c in range(n_cams)]
    _pos, rcm = multi_cam_point_positions(targets, cpar, cals)
    return {
        "n_points": int(len(rcm)),
        "n_common": int(n_common),
        "median": float(np.median(rcm)),
        "p90": float(np.percentile(rcm, 90)),
        "p95": float(np.percentile(rcm, 95)),
        "max": float(np.max(rcm)),
    }


# Distortion groups introduced one at a time by the greedy shaker, in this
# fixed order (least to most entangled with exterior/depth).
DIST_GROUPS = [
    ("k1k2k3", "get_radial_distortion", "set_radial_distortion", 3),
    ("p1p2", "get_decentering", "set_decentering", 2),
    ("scaleshear", "get_affine", "set_affine_trans", 2),
    ("glass", "get_glass_vec", "set_glass_vec", 3),
]


def joint_plate_bundle_adjust(
    results,
    cpar,
    *,
    reg_weight=1.0,
    max_nfev=100,
    shake_distortion=False,
    rcm_margin=0.0,
    verbose=False,
):
    """Couple all cameras by jointly refining camera exteriors (pos+angles) and
    the shared 3D plate points, minimizing total reprojection with the plate
    points softly anchored to their nominal calblock coords (reg_weight). Unlike
    per-camera resection this has a cross-camera term, so it can lower the
    cross-camera RCM that reprojection RMS alone can't see. Distortion (k/p/scale/
    shear/interf) is held fixed at the per-camera-fit values.

    Mutates copies -- returns (new_results, info) where new_results is a list of
    CamResult with updated .cal (and .rep/.rms recomputed), and info is a dict
    {rcm_before, rcm_after, n_points, n_cams, cost_before, cost_after, success}.
    Returns (results, {...'skipped': reason}) unchanged when < 2 valid cams or
    < 4 common points (too few to constrain the joint fit)."""
    from scipy.optimize import least_squares

    from openptv2.algorithms.trafo import pixel_to_metric
    from openptv2.imgcoord import image_coordinates

    if reg_weight <= 0:
        # The nominal-anchor term is the gauge fix: without it the joint fit has
        # a 7-DOF similarity freedom (translate/rotate/scale cloud+cameras with
        # zero reprojection change) and least_squares wanders silently.
        raise ValueError("reg_weight must be > 0 (it fixes the 7-DOF gauge)")

    new_results = [dataclasses.replace(r, cal=copy.deepcopy(r.cal)) for r in results]
    valid = [
        r
        for r in new_results
        if r.cal is not None and r.error is None and len(r.ref) > 0
    ]
    if len(valid) < 2:
        return results, {"skipped": "need >=2 valid cameras"}

    # Shared-point structure keyed by rounded 3D coord (mirrors cross_camera_rcm).
    obs: dict[tuple, dict[int, tuple]] = {}
    nominal: dict[tuple, np.ndarray] = {}
    for r in valid:
        for ref_row, det_row in zip(r.ref, r.det):
            key = tuple(np.round(ref_row, 3))
            mx, my = pixel_to_metric(det_row[0], det_row[1], cpar)
            obs.setdefault(key, {})[r.cam] = (mx, my)
            nominal[key] = np.asarray(ref_row, float)

    keys = [k for k in obs if len(obs[k]) >= 2]
    if len(keys) < 4:
        return results, {"skipped": "need >=4 common points"}
    key_index = {k: i for i, k in enumerate(keys)}
    nominal_arr = np.array([nominal[k] for k in keys], float)  # (n_pts, 3)
    n_pts = len(keys)

    mm = cpar.mm
    valid_cams = [r.cam for r in valid]
    cal_by_cam = {r.cam: r.cal for r in valid}
    # Per-camera: (point-row indices, observed metric coords) for its points.
    cam_obs: dict[Any, tuple[Any, Any]] = {}
    for cam in valid_cams:
        rows, mets = [], []
        for k in keys:
            if cam in obs[k]:
                rows.append(key_index[k])
                mets.append(obs[k][cam])
        cam_obs[cam] = (np.asarray(rows, int), np.asarray(mets, float))

    n_cam_params = 6 * len(valid_cams)
    sqrt_w = np.sqrt(reg_weight)

    def _solve(active_groups, base_cals):
        """Solve exterior+points (+ active distortion groups) starting from a
        deepcopy of base_cals. Returns (trial_results, rcm_median, cost, success)
        or None on solver error. Never mutates base_cals (fresh deepcopy), so a
        rejected trial is simply discarded."""
        cals = {c: copy.deepcopy(base_cals[c]) for c in valid_cams}
        # Layout: exterior block, distortion block (group-major), point block.
        x0: Any = []
        for cam in valid_cams:
            x0.extend(cals[cam].get_pos())
            x0.extend(cals[cam].get_angles())
        for _name, getter, _setter, _n in active_groups:
            for cam in valid_cams:
                x0.extend(getattr(cals[cam], getter)())
        x0.extend(nominal_arr.ravel())
        x0 = np.asarray(x0, float)
        n_dist = sum(n for _n0, _g, _s, n in active_groups) * len(valid_cams)
        pts_off = n_cam_params + n_dist

        def _resid(x):
            for i, cam in enumerate(valid_cams):
                cals[cam].set_pos(x[i * 6 : i * 6 + 3])
                cals[cam].set_angles(x[i * 6 + 3 : i * 6 + 6])
            off = n_cam_params
            for _name, _getter, setter, n in active_groups:
                for cam in valid_cams:
                    getattr(cals[cam], setter)(x[off : off + n])
                    off += n
            pts = x[pts_off:].reshape(n_pts, 3)
            res = []
            for cam in valid_cams:
                rows, mets = cam_obs[cam]
                proj = image_coordinates(pts[rows], cals[cam], mm)
                res.append((proj - mets).ravel())
            res.append((sqrt_w * (pts - nominal_arr)).ravel())
            return np.nan_to_num(np.concatenate(res), nan=1e6, posinf=1e6, neginf=-1e6)

        try:
            sol = least_squares(
                _resid, x0, max_nfev=max_nfev, method="trf", verbose=2 if verbose else 0
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
        _resid(sol.x)  # leave cals set to the solution

        trial = []
        for r in new_results:
            if r.cam in cals and r.error is None and len(r.ref) > 0:
                cal = cals[r.cam]
                rep = np.array(
                    [_reproject_px(cal, cpar.mm, row, cpar) for row in r.ref]
                )
                trial.append(
                    dataclasses.replace(r, cal=cal, rep=rep, rms=rms_px(r.det, rep))
                )
            else:
                trial.append(r)
        rcm = cross_camera_rcm(trial, cpar)
        rcm_med = rcm["median"] if rcm else float("inf")
        return trial, rcm_med, float(sol.cost * 2), bool(sol.success)

    # cost_before: residual at the seed (points=nominal, exterior unmoved).
    seed_res: list[Any] = []
    for cam in valid_cams:
        rows, mets = cam_obs[cam]
        proj = image_coordinates(nominal_arr[rows], cal_by_cam[cam], mm)
        seed_res.append((proj - mets).ravel())
    x0_cost = float(np.sum(np.concatenate(seed_res) ** 2))

    # Baseline: exterior + points only (the original behavior).
    base = _solve([], cal_by_cam)
    if base is None:
        return results, {"skipped": "solver error"}
    best_results, best_rcm, cost_after, success = base
    rcm_exterior_only = best_rcm

    rcm_initial = cross_camera_rcm(results, cpar)
    rcm_before = rcm_initial["median"] if rcm_initial else None

    shaken_groups: list[str] = []
    rcm_trace: list[tuple] = []
    accepted: list[Any] = []
    if shake_distortion:
        for group in DIST_GROUPS:
            best_cals = {r.cam: r.cal for r in best_results if r.cam in cal_by_cam}
            trial = _solve(accepted + [group], best_cals)
            if trial is None:
                rcm_trace.append((group[0], None, False))
                continue
            t_results, t_rcm, t_cost, t_success = trial
            better = t_rcm < best_rcm - rcm_margin
            rcm_trace.append((group[0], t_rcm, better))
            if better:
                best_results, best_rcm, cost_after, success = (
                    t_results,
                    t_rcm,
                    t_cost,
                    t_success,
                )
                accepted.append(group)
                shaken_groups.append(group[0])

    info = {
        "rcm_before": rcm_before,
        "rcm_after": best_rcm,
        "rcm_exterior_only": rcm_exterior_only,
        "shaken_groups": shaken_groups,
        "rcm_trace": rcm_trace,
        "n_points": n_pts,
        "n_cams": len(valid_cams),
        "cost_before": x0_cost,
        "cost_after": cost_after,
        "success": success,
    }
    return best_results, info


def _cam_view(base: Path, cam: int, raw: Any) -> Any:
    """The image ONE camera actually sees, given the raw calibration frame.

    On a splitter rig every camera shares the same multiplexed frame but its
    targets, .ori and reprojections all live in that camera's own 512x512
    quadrant. Drawing them over the full 1024x1024 raw frame puts every
    camera's points in the top-left quadrant -- the overlay then looks wrong
    even when the calibration is right. Detection already splits (see
    detect_targets.py); this makes the overlay agree with it.
    """
    yaml_path = _find_yaml(base)
    y = yaml.safe_load(yaml_path.read_text()) if yaml_path else {}
    ptv = y.get("ptv") or {}
    if not ptv.get("splitter"):
        return raw
    from openptv2.gui.ptv import image_split

    return image_split(raw, order=ptv.get("splitter_order") or [0, 1, 3, 2])[cam]


def save_overlay(res: CamResult, base: Path, outdir: Path) -> Path:
    """Save a detected-vs-reprojected overlay PNG for one camera."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6.4))
    img_path, _, _ = cam_files(base, res.cam)
    try:
        import imageio.v3 as iio

        ax.imshow(_cam_view(base, res.cam, iio.imread(img_path)), cmap="gray")
    except Exception:
        ax.invert_yaxis()
    ax.scatter(
        res.det[:, 0],
        res.det[:, 1],
        s=40,
        facecolors="none",
        edgecolors="lime",
        linewidths=1.2,
        label="detected",
    )
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


def _pick_eps0(rows):
    """Choose the recommended eps0 from sweep rows [{eps0, correct, wrong}, ...].

    The knee: the LARGEST eps0 that is still spurious-free (wrong == 0) -- it
    admits the most correct quadruplets without a single false one; widening
    further only adds wrong matches. If no row is clean (noisy rig), fall back
    to the eps0 maximizing (correct - 2*wrong). Returns (eps0, correct, wrong).
    """
    if not rows:
        return None
    clean = [r for r in rows if r["wrong"] == 0 and r["correct"] > 0]
    if clean:
        best = max(clean, key=lambda r: (r["correct"], r["eps0"]))
        # largest eps0 that still achieves that max-correct while clean
        tied = [r for r in clean if r["correct"] == best["correct"]]
        pick = max(tied, key=lambda r: r["eps0"])
    else:
        pick = max(rows, key=lambda r: (r["correct"] - 2 * r["wrong"], -r["eps0"]))
    return pick["eps0"], pick["correct"], pick["wrong"]


def suggest_eps0(base, cpar, cals, *, sweep=None, gt_radius=3.0):
    """Sweep the epipolar band (VolumePar.eps0) and recommend the value giving
    the most CORRECT quadruplets with no spurious ones.

    Ground truth is free right after calibration: each detected target's
    calblock ID is known (reproject the calblock + nearest-neighbour within
    gt_radius px), so a 4-camera correspondence is CORRECT iff all four linked
    dots share one ID. Reuses the real correspondence engine, so the answer
    reflects the actual matching, not a proxy.

    Needs num_cams == 4 and a criteria: block in the dataset YAML
    (X_lay/Zmin_lay/Zmax_lay/cn/cnx/cny/csumg/corrmin). Returns None otherwise.

    Returns {"recommended", "max_correct", "current", "sweep": [{eps0, quads,
    correct, wrong}, ...]}.
    """
    from openptv2.algorithms.correspondences import correct_frame, correspondences
    from openptv2.algorithms.parameters import VolumePar
    from openptv2.algorithms.tracking_frame_buf import Frame, read_targets

    if cpar.num_cams != 4:
        return None
    yaml_path = _find_yaml(base)
    y = yaml.safe_load(yaml_path.read_text()) if yaml_path else {}
    crit = y.get("criteria")
    if not crit:
        return None

    fix, _ = read_calblock(str(resolve_calblock(base)))
    fix = np.asarray(fix, float)

    # Frame declares targets/num_targets as object (Cython buffer); Any here.
    frm: Any = Frame(4, 1000)
    gt = []  # per-camera: detected-target-index -> calblock id (1-based) or -1
    for c in range(4):
        proj = np.array([_reproject_px(cals[c], cpar.mm, p, cpar) for p in fix])
        pix = read_targets(str(target_base(base, c)), 0)
        ids = np.full(len(pix), -1, dtype=int)
        for k, t in enumerate(pix):
            d = np.hypot(proj[:, 0] - t.x, proj[:, 1] - t.y)
            j = int(d.argmin())
            if d[j] <= gt_radius:
                ids[k] = j + 1
        frm.targets[c] = list(pix)
        frm.num_targets[c] = len(pix)
        gt.append(ids)

    corrected = correct_frame(frm, cals, cpar, 0.0001)
    base_vpar = dict(
        X_lay=crit.get("X_lay", [-100, 100]),
        Zmin_lay=crit.get("Zmin_lay", [-100, -100]),
        Zmax_lay=crit.get("Zmax_lay", [100, 100]),
        cn=crit.get("cn", 0.0),
        cnx=crit.get("cnx", 0.0),
        cny=crit.get("cny", 0.0),
        csumg=crit.get("csumg", 0.0),
        corrmin=crit.get("corrmin", 0.0),
    )
    current = float(crit.get("eps0", 0.05)) or 0.05
    if sweep is None:
        sweep = list(np.geomspace(current * 0.2, current * 8.0, 15))

    rows = []
    for eps0 in sweep:
        vpar = VolumePar(eps0=float(eps0), **base_vpar)
        con, mc = correspondences(frm, corrected, vpar, cpar, cals)
        correct = wrong = 0
        for nt in con:
            p = nt.p
            if all(p[c] >= 0 for c in range(4)):
                oids = [gt[c][corrected[c][p[c]].pnr] for c in range(4)]
                if all(o != -1 for o in oids) and len(set(oids)) == 1:
                    correct += 1
                else:
                    wrong += 1
        rows.append(
            {
                "eps0": round(float(eps0), 4),
                "quads": int(mc[0]),
                "correct": correct,
                "wrong": wrong,
            }
        )

    pick = _pick_eps0(rows)
    return {
        "recommended": pick[0] if pick else None,
        "max_correct": pick[1] if pick else 0,
        "current": current,
        "sweep": rows,
    }


def _flat_targets_from_obs(obs_list, cals, cpar):
    """Build the (n_pts, n_cams, 2) FLAT-metric array multi_cam_point_positions
    wants, from a list of {cam: (px, py)} observation dicts. Cameras that didn't
    see a point get the COORD_UNUSED sentinel."""
    from openptv2.algorithms.orientation import COORD_UNUSED
    from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric

    n_cams = cpar.num_cams
    targets = np.full((len(obs_list), n_cams, 2), COORD_UNUSED, dtype=np.float64)
    for i, pix in enumerate(obs_list):
        for cam, (px, py) in pix.items():
            mx, my = pixel_to_metric(px, py, cpar)
            cal = cals[cam]
            fx, fy = dist_to_flat(
                mx,
                my,
                cal.int_par.xh,
                cal.int_par.yh,
                cal.added_par.k1,
                cal.added_par.k2,
                cal.added_par.k3,
                cal.added_par.p1,
                cal.added_par.p2,
                cal.added_par.scx,
                cal.added_par.she,
            )
            targets[i, cam] = (fx, fy)
    return targets


def _tracer_rcm_array(obs_list, cals, cpar):
    """Per-particle ray-convergence miss distance (mm) over tracer observations."""
    from openptv2.orientation import multi_cam_point_positions

    if len(obs_list) == 0:
        return np.empty(0, dtype=np.float64)
    targets = _flat_targets_from_obs(obs_list, cals, cpar)
    _pos, rcm = multi_cam_point_positions(targets, cpar, cals)
    return np.asarray(rcm, dtype=np.float64)


def _tracer_rcm_median(obs_list, cals, cpar):
    """Median ray-convergence miss distance (mm) over tracer observations."""
    if len(obs_list) < 3:
        return None
    rcm = _tracer_rcm_array(obs_list, cals, cpar)
    return float(np.median(rcm))


def _tracer_ghost_probabilities(obs_list, seed_xyz, cals, cpar):
    """Compute per-particle (rcm, n_seen, p_ghost) for candidate observations."""
    from openptv2.point_quality import fit_scale, ghost_probability

    n = len(obs_list)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty
    rcm = _tracer_rcm_array(obs_list, cals, cpar)
    n_seen = np.array([len(pix) for pix in obs_list], dtype=np.int32)
    pos = np.asarray(seed_xyz, dtype=np.float64).reshape(n, 3)
    scale = fit_scale(pos, rcm, n_seen)
    p_ghost = ghost_probability(pos, rcm, n_seen, scale)
    return rcm, n_seen, np.asarray(p_ghost, dtype=np.float64)


def camera_ghost_diagnostics(obs_list, seed_xyz, cals, cpar, *, max_ghost_prob=0.20):
    """Per-camera ghost and reprojection diagnostics on tracer observations.

    Identifies which camera contributes most to 3-camera partial correspondences
    (the dominant source of ghosts) and which camera has the largest reprojection
    mismatch on high-confidence 4-camera real particles.
    """
    n_cams = cpar.num_cams
    rcm, n_seen, p_ghost = _tracer_ghost_probabilities(obs_list, seed_xyz, cals, cpar)
    seed = np.asarray(seed_xyz, dtype=np.float64).reshape(-1, 3)

    missing_3cam = np.zeros(n_cams, dtype=int)
    n_3cam = int((n_seen == max(2, n_cams - 1)).sum())
    for i, pix in enumerate(obs_list):
        if len(pix) == max(2, n_cams - 1):
            for c in range(n_cams):
                if c not in pix:
                    missing_3cam[c] += 1

    real_4cam = (n_seen == n_cams) & (p_ghost <= max_ghost_prob)
    if real_4cam.sum() < 5:
        real_4cam = n_seen >= min(n_cams, 3)
    if real_4cam.sum() < 5:
        real_4cam = np.ones(len(obs_list), dtype=bool)

    per_cam = []
    rms_arr = np.zeros(n_cams, dtype=np.float64)
    med_arr = np.zeros(n_cams, dtype=np.float64)
    miss_share_arr = np.zeros(n_cams, dtype=np.float64)
    for cam in range(n_cams):
        res_px = []
        for i in np.flatnonzero(real_4cam):
            if cam not in obs_list[i]:
                continue
            proj = _reproject_px(cals[cam], cpar.mm, seed[i], cpar)
            dx = obs_list[i][cam][0] - proj[0]
            dy = obs_list[i][cam][1] - proj[1]
            res_px.append(float(np.hypot(dx, dy)))
        arr = np.asarray(res_px, dtype=np.float64)
        rms = float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0
        med = float(np.median(arr)) if len(arr) else 0.0
        m_share = float(missing_3cam[cam]) / max(n_3cam, 1)
        rms_arr[cam] = rms
        med_arr[cam] = med
        miss_share_arr[cam] = m_share
        per_cam.append(
            {
                "cam": cam,
                "missing_3cam_count": int(missing_3cam[cam]),
                "missing_3cam_share": round(m_share, 4),
                "reproj_rms_px": round(rms, 4),
                "reproj_median_px": round(med, 4),
            }
        )

    med_norm = med_arr / max(float(np.median(med_arr)), 1e-6)
    miss_norm = miss_share_arr / max(float(np.mean(miss_share_arr)), 1e-6)
    badness = 0.7 * med_norm + 0.3 * miss_norm
    for c in range(n_cams):
        per_cam[c]["badness_score"] = round(float(badness[c]), 4)

    best_cam = int(np.argmin(badness))
    worst_cam = int(np.argmax(badness))
    real_mask = p_ghost <= max_ghost_prob
    ghost_mask = p_ghost > max_ghost_prob
    real_rcm = (
        float(np.median(rcm[real_mask])) if real_mask.any() else float(np.median(rcm))
    )
    ghost_rcm = float(np.median(rcm[ghost_mask])) if ghost_mask.any() else None
    sep_ratio = (
        float(ghost_rcm / max(real_rcm, 1e-9))
        if ghost_rcm is not None and real_rcm > 0
        else None
    )
    return {
        "per_camera": per_cam,
        "best_cam": best_cam,
        "worst_cam": worst_cam,
        "real_rcm_median": real_rcm,
        "ghost_rcm_median": ghost_rcm,
        "separation_ratio": sep_ratio,
        "n_real": int(real_mask.sum()),
        "n_ghost": int(ghost_mask.sum()),
    }


def _load_tracer_frame_data(base: Path, cpar: Any, frames: Any) -> Any:
    """Read tracked 3D points, correspondences, and per-camera 2D targets from
    the dataset's Zarr ``RunStore`` (`res/run.zarr` or `run.zarr`).

    Returns (frame_data, skip_reason): frame_data is a list of dicts
    ``{"pts": (N,3), "det": list[(M_c,2)], "stored_obs": list[dict] | None,
    "linked": (N,) bool}`` on success; skip_reason is None on success, else
    the reason ``tracer_self_calibrate`` should skip.
    """
    from openptv2.storage import RunStore, find_existing_store

    n_cams = cpar.num_cams
    store_path = find_existing_store(base)
    if store_path is None or not store_path.exists():
        return None, "no Zarr RunStore found (run sequence + tracking first)"

    store = RunStore(store_path, mode="r")
    all_store_frames = store.frames()
    has_lnk = [f for f in all_store_frames if store.has_linkage(f, "ptv_is")]
    has_corr = [f for f in all_store_frames if store.has_correspondences(f)]
    frame_nums = has_lnk if has_lnk else has_corr
    if frames is not None:
        wanted = set(frames)
        frame_nums = [f for f in frame_nums if f in wanted]
    if not frame_nums:
        return None, "no tracked-linkage or correspondence frames in RunStore"

    frame_data = []
    for frame in frame_nums:
        det = []
        rows_per_cam = []
        for cam in range(n_cams):
            key = f"targets/cam_{cam}/frame_{frame:06d}"
            if key in store.root:
                r = np.asarray(store.root[key], dtype=np.float64)
                rows_per_cam.append(r)
                det.append(r[:, 1:3] if len(r) else np.empty((0, 2)))
            else:
                rows_per_cam.append(None)
                det.append(np.empty((0, 2)))

        stored_obs = None
        linked = None
        if store.has_correspondences(frame):
            pos, cam_ids = store.read_correspondences(frame)
            pts = np.asarray(pos, dtype=np.float64)
            if len(pts) == 0:
                continue
            stored_obs = []
            for i in range(len(pts)):
                pix = {}
                for cam in range(n_cams):
                    r = rows_per_cam[cam]
                    cid = int(cam_ids[i, cam])
                    if r is not None and 0 <= cid < len(r):
                        pix[cam] = (float(r[cid, 1]), float(r[cid, 2]))
                stored_obs.append(pix)
            if store.has_linkage(frame, "ptv_is"):
                prev_ids, next_ids, _pos_lnk = store.read_linkage(frame, "ptv_is")
                if len(prev_ids) == len(pts):
                    linked = (prev_ids >= 0) | (next_ids >= 0)
        elif store.has_linkage(frame, "ptv_is"):
            prev_ids, next_ids, pos = store.read_linkage(frame, "ptv_is")
            pts = np.asarray(pos, dtype=np.float64)
            if len(pts) == 0:
                continue
            linked = (prev_ids >= 0) | (next_ids >= 0)
        else:
            continue

        if linked is None:
            linked = np.ones(len(pts), dtype=bool)
        frame_data.append(
            {
                "pts": pts,
                "det": det,
                "stored_obs": stored_obs,
                "linked": np.asarray(linked, dtype=bool),
            }
        )
    if not frame_data:
        return None, "no tracked points in the selected frames"

    return frame_data, None


def retriangulate_store(
    store_or_base: Any,
    cals: list[Calibration],
    cpar: Any,
    *,
    frames: list[int] | None = None,
) -> dict:
    """Re-triangulate 3D correspondences in a Zarr ``RunStore`` using ``cals``.

    After ``tracer_self_calibrate`` refines the camera exteriors, updating the
    stored 3D correspondence positions (and point-quality marks if present)
    sharpens real-vs-ghost separation immediately for subsequent tracking runs
    without needing to re-run 2D target detection.
    """
    from openptv2.algorithms.orientation import COORD_UNUSED, multi_cam_point_positions
    from openptv2.algorithms.trafo import (
        correct_brown_affine_batch,
        pixel_to_metric_batch,
    )
    from openptv2.point_quality import read_frame_arrays
    from openptv2.storage import RunStore, find_existing_store

    if isinstance(store_or_base, (str, Path)):
        base = Path(store_or_base)
        store_path = base if base.suffix == ".zarr" else find_existing_store(base)
        if store_path is None or not store_path.exists():
            raise FileNotFoundError(f"No RunStore found at {base}")
        store = RunStore(store_path, mode="a")
    else:
        store = store_or_base

    n_cams = len(cals)
    target_frames = [
        f
        for f in store.frames()
        if store.has_correspondences(f) and (frames is None or f in set(frames))
    ]
    rcm_after_list = []
    for f in target_frames:
        pos, cam_ids, rows = read_frame_arrays(store, f, n_cams)
        if len(pos) == 0:
            continue
        targets = np.full((len(pos), n_cams, 2), COORD_UNUSED, dtype=np.float64)
        for cam in range(n_cams):
            if rows[cam] is None:
                continue
            sel = np.flatnonzero(cam_ids[:, cam] >= 0)
            if not len(sel):
                continue
            cal = cals[cam]
            a = cal.added_par
            xy = np.ascontiguousarray(
                rows[cam][cam_ids[sel, cam]][:, 1:3], dtype=np.float64
            )
            met = np.ascontiguousarray(
                pixel_to_metric_batch(xy, cpar), dtype=np.float64
            )
            flat = np.asarray(
                correct_brown_affine_batch(
                    met, a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she
                )
            )
            targets[sel, cam, 0] = flat[:, 0] - cal.int_par.xh
            targets[sel, cam, 1] = flat[:, 1] - cal.int_par.yh
        new_pos, rcm_after = multi_cam_point_positions(targets, cpar, cals)
        rcm_after_list.append(np.asarray(rcm_after, dtype=np.float64))
        store.write_correspondences(f, np.asarray(new_pos, dtype=np.float64), cam_ids)

    rcm_all = (
        np.concatenate(rcm_after_list)
        if rcm_after_list
        else np.empty(0, dtype=np.float64)
    )
    return {
        "n_frames": len(target_frames),
        "n_points": int(len(rcm_all)),
        "rcm_median": float(np.median(rcm_all)) if len(rcm_all) else None,
    }


def tracer_self_calibrate(
    base,
    cpar,
    cals,
    *,
    frames=None,
    tol_px=2.0,
    hold_cam: int | str = 0,
    min_cams=2,
    max_particles=400,
    iters=1,
    max_nfev=100,
    filter_ghosts: bool = True,
    ghost_prob_schedule: tuple[float, ...] = (0.25, 0.15, 0.08),
    focus_cameras: bool = True,
    reg_weight: float = 0.05,
    cam_reg_weight: float = 0.01,
    verbose=False,
):
    """Refine camera exteriors on REAL tracked tracer particles from the Zarr RunStore.

    Unlike naive shaking that feeds all quadruplets/pairs (including ghosts)
    into an unweighted fit, this routine:
      1. Diagnoses per-camera ghost involvement and reprojection residuals
         (:func:`camera_ghost_diagnostics`), auto-selecting the cleanest camera
         as ``hold_cam`` when ``hold_cam="auto"``.
      2. Filters out ghost candidates using ray-convergence ghost probability
         (``p_ghost``) and trajectory linkage, preferring linked all-camera
         real particles and weighting residuals by ``1 - p_ghost``.
      3. Includes the held reference camera's 2D observations in the bundle
         adjustment residual (fixing its 6 exterior DOFs while coupling all
         free cameras and 3D particle positions to it), with scaled angle
         increments (mrad) and robust ``soft_l1`` loss.
      4. Progressively tightens ``p_ghost`` across iterations
         (``ghost_prob_schedule``), increasing the RCM separation between real
         (shaken) particles and ghosts on each pass.
    """
    import copy

    from scipy import sparse
    from scipy.optimize import least_squares

    from openptv2.algorithms.trafo import pixel_to_metric
    from openptv2.imgcoord import image_coordinates

    base = Path(base)
    frame_data, skip_reason = _load_tracer_frame_data(base, cpar, frames)
    if skip_reason is not None:
        print(f"[tracer self-cal] skipped: {skip_reason}", flush=True)
        return cals, {"skipped": skip_reason}

    n_cams = cpar.num_cams

    def _gather_candidates(cur_cals):
        """Build candidate (obs, seed, linked) lists under ``cur_cals``."""
        obs, seed, lnk = [], [], []
        for fd in frame_data:
            pts = fd["pts"]
            det = fd["det"]
            stored_obs = fd["stored_obs"]
            linked = fd["linked"]
            for i, p in enumerate(pts):
                pix = {}
                if stored_obs is not None and len(stored_obs[i]) >= min_cams:
                    for cam, (px, py) in stored_obs[i].items():
                        proj = _reproject_px(cur_cals[cam], cpar.mm, p, cpar)
                        if np.hypot(px - proj[0], py - proj[1]) <= max(
                            tol_px * 2.0, 4.0
                        ):
                            pix[cam] = (px, py)
                if len(pix) < min_cams:
                    pix = {}
                    for cam in range(n_cams):
                        d = det[cam]
                        if len(d) == 0:
                            continue
                        proj = _reproject_px(cur_cals[cam], cpar.mm, p, cpar)
                        dist = np.hypot(d[:, 0] - proj[0], d[:, 1] - proj[1])
                        j = int(np.argmin(dist))
                        if dist[j] <= tol_px:
                            pix[cam] = (float(d[j, 0]), float(d[j, 1]))
                if len(pix) >= min_cams:
                    obs.append(pix)
                    seed.append(p)
                    lnk.append(bool(linked[i]))
        return (
            obs,
            np.asarray(seed, dtype=np.float64).reshape(-1, 3),
            np.asarray(lnk, dtype=bool),
        )

    def _select_real_particles(obs_all, seed_all, lnk_all, cur_cals, q_thresh):
        """Select up to ``max_particles`` high-confidence real particles."""
        if len(obs_all) == 0:
            return [], np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
        rcm, n_seen, p_ghost = _tracer_ghost_probabilities(
            obs_all, seed_all, cur_cals, cpar
        )
        if not filter_ghosts:
            idx = (
                np.linspace(0, len(obs_all) - 1, max_particles).astype(int)
                if len(obs_all) > max_particles
                else np.arange(len(obs_all))
            )
            return (
                [obs_all[i] for i in idx],
                seed_all[idx],
                np.ones(len(idx), dtype=np.float64),
            )

        # Tier 1: linked all-camera real particles with p_ghost <= q_thresh
        tier1 = np.flatnonzero((n_seen == n_cams) & lnk_all & (p_ghost <= q_thresh))
        # Tier 2: all-camera particles with p_ghost <= q_thresh
        tier2 = np.flatnonzero((n_seen == n_cams) & (p_ghost <= q_thresh))
        # Tier 3: linked >=min_cams particles with p_ghost <= max(q_thresh, 0.35)
        tier3 = np.flatnonzero(
            (n_seen >= min_cams) & lnk_all & (p_ghost <= max(q_thresh, 0.35))
        )
        # Tier 4: fallback to all >=min_cams particles (for tiny synthetic test fixtures)
        tier4 = np.arange(len(obs_all))

        cand = tier1
        for fallback in (tier2, tier3, tier4):
            if len(cand) >= min(25, len(obs_all)):
                break
            cand = fallback

        if len(cand) > max_particles:
            idx = cand[np.linspace(0, len(cand) - 1, max_particles).astype(int)]
        else:
            idx = cand
        weights = np.clip(1.0 - p_ghost[idx], 0.1, 1.0)
        return [obs_all[i] for i in idx], seed_all[idx], weights

    best_cals = [copy.deepcopy(c) for c in cals]
    obs_all0, seed_all0, lnk_all0 = _gather_candidates(best_cals)
    if len(obs_all0) < 10:
        print(
            f"[tracer self-cal] skipped: only {len(obs_all0)} multi-cam tracer "
            "particles (need >= 10)",
            flush=True,
        )
        return cals, {"skipped": f"only {len(obs_all0)} multi-cam tracer particles"}

    diag_before = camera_ghost_diagnostics(obs_all0, seed_all0, best_cals, cpar)
    if hold_cam == "auto":
        hold_cam_idx = int(diag_before["best_cam"])
    else:
        hold_cam_idx = int(hold_cam)

    free_cams = [c for c in range(n_cams) if c != hold_cam_idx]
    n_cam_params = 6 * len(free_cams)

    print(
        f"[tracer self-cal] starting: base={base}, "
        f"frames={len(frame_data)}, tol_px={tol_px}, hold_cam={hold_cam_idx + 1}, "
        f"worst_cam={diag_before['worst_cam'] + 1}, max_particles={max_particles}, "
        f"iters={iters}",
        flush=True,
    )

    def _fit(obs_list, seed_xyz, pt_weights, cur_cals, cam_badness):
        """One coupled joint fit over free-camera exteriors and real 3D points."""
        cur = [copy.deepcopy(c) for c in cur_cals]
        seed = np.asarray(seed_xyz, dtype=np.float64).reshape(-1, 3)
        n_pts = len(obs_list)

        # Camera focus weights: emphasize cameras with higher badness so the shared
        # 3D points and camera adjustments pull the worst cameras into alignment.
        cam_focus = np.ones(n_cams, dtype=np.float64)
        if focus_cameras and cam_badness is not None:
            b = np.asarray(cam_badness, dtype=np.float64)
            cam_focus = np.clip(b / max(float(np.mean(b)), 1e-6), 0.75, 1.5)

        cam_rows, cam_mets, cam_w = {}, {}, {}
        for cam in range(n_cams):
            rows, mets, ws = [], [], []
            cf = float(cam_focus[cam])
            for i, pix in enumerate(obs_list):
                if cam in pix:
                    rows.append(i)
                    mets.append(pixel_to_metric(pix[cam][0], pix[cam][1], cpar))
                    pw = 1.0 if pt_weights is None else float(pt_weights[i])
                    ws.append(cf * pw)
            cam_rows[cam] = np.asarray(rows, dtype=int)
            cam_mets[cam] = np.asarray(mets, dtype=np.float64).reshape(-1, 2)
            cam_w[cam] = np.asarray(ws, dtype=np.float64)[:, None]

        base_pos = {
            c: np.asarray(cur[c].get_pos(), dtype=np.float64) for c in free_cams
        }
        base_ang = {
            c: np.asarray(cur[c].get_angles(), dtype=np.float64) for c in free_cams
        }
        x0 = np.zeros(n_cam_params + 3 * n_pts, dtype=np.float64)

        def _resid(x):
            pts = seed + x[n_cam_params:].reshape(n_pts, 3)
            for ci, cam in enumerate(free_cams):
                cur[cam].set_pos(base_pos[cam] + x[ci * 6 : ci * 6 + 3])
                cur[cam].set_angles(base_ang[cam] + x[ci * 6 + 3 : ci * 6 + 6] * 1e-3)
            res = []
            # Include ALL cameras (including hold_cam_idx) in the residual so the
            # free cameras and 3D points are anchored to hold_cam_idx!
            for cam in range(n_cams):
                rows = cam_rows[cam]
                if len(rows) == 0:
                    continue
                proj = image_coordinates(pts[rows], cur[cam], cpar.mm)
                res.append(((proj - cam_mets[cam]) * cam_w[cam]).ravel())
            if reg_weight > 0:
                res.append((reg_weight * x[n_cam_params:]).ravel())
            if cam_reg_weight > 0 and n_cam_params > 0:
                res.append((cam_reg_weight * x[:n_cam_params]).ravel())
            return np.nan_to_num(
                np.concatenate(res) if res else np.zeros(1),
                nan=1e6,
                posinf=1e6,
                neginf=-1e6,
            )

        n_obs_res = 2 * sum(len(cam_rows[c]) for c in range(n_cams))
        n_reg_res = (3 * n_pts if reg_weight > 0 else 0) + (
            n_cam_params if cam_reg_weight > 0 else 0
        )
        jac = sparse.lil_matrix((n_obs_res + n_reg_res, x0.size), dtype=np.int8)
        r = 0
        for cam in range(n_cams):
            ci = free_cams.index(cam) if cam in free_cams else -1
            for row in cam_rows[cam]:
                for k in range(2):
                    if ci >= 0:
                        jac[r + k, ci * 6 : ci * 6 + 6] = 1
                    jac[r + k, n_cam_params + 3 * row : n_cam_params + 3 * row + 3] = 1
                r += 2
        off_r = n_obs_res
        if reg_weight > 0:
            for k in range(3 * n_pts):
                jac[off_r + k, n_cam_params + k] = 1
            off_r += 3 * n_pts
        if cam_reg_weight > 0:
            for k in range(n_cam_params):
                jac[off_r + k, k] = 1
        try:
            sol = least_squares(
                _resid,
                x0,
                max_nfev=max_nfev,
                method="trf",
                loss="soft_l1",
                f_scale=0.002,
                jac_sparsity=jac.tocsr(),
                verbose=2 if verbose else 0,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
        _resid(sol.x)
        return cur, bool(sol.success)

    q0 = ghost_prob_schedule[0] if ghost_prob_schedule else 0.25
    obs0, _seed0, _w0 = _select_real_particles(
        obs_all0, seed_all0, lnk_all0, best_cals, q0
    )
    if len(obs0) < 10:
        print(
            f"[tracer self-cal] skipped: only {len(obs0)} real multi-cam tracer "
            "particles (need >= 10)",
            flush=True,
        )
        return cals, {"skipped": f"only {len(obs0)} multi-cam tracer particles"}

    rcm_before = _tracer_rcm_median(obs0, best_cals, cpar)
    best_rcm = rcm_before
    trace = []
    success = False
    rcm_before_str = "n/a" if rcm_before is None else f"{rcm_before * 1000:.1f} um"
    sep_before_str = (
        "n/a"
        if diag_before["separation_ratio"] is None
        else f"{diag_before['separation_ratio']:.2f}x"
    )
    print(
        f"[tracer self-cal] initial match: {len(obs0)} real particles "
        f"(>={min_cams} cams), RCM before = {rcm_before_str}, "
        f"ghost/real sep = {sep_before_str}",
        flush=True,
    )

    n_iters = max(1, iters)
    for it in range(n_iters):
        q_thresh = (
            ghost_prob_schedule[min(it, len(ghost_prob_schedule) - 1)]
            if ghost_prob_schedule
            else 0.20
        )
        obs_all_it, seed_all_it, lnk_all_it = (
            (obs_all0, seed_all0, lnk_all0)
            if it == 0
            else _gather_candidates(best_cals)
        )
        obs, seed, pt_w = _select_real_particles(
            obs_all_it, seed_all_it, lnk_all_it, best_cals, q_thresh
        )
        if len(obs) < 10:
            print(
                f"[tracer self-cal] iter {it + 1}/{n_iters}: only {len(obs)} "
                "particles matched, stopping",
                flush=True,
            )
            break
        cur_diag = camera_ghost_diagnostics(obs_all_it, seed_all_it, best_cals, cpar)
        cam_bad = [c["badness_score"] for c in cur_diag["per_camera"]]
        print(
            f"[tracer self-cal] iter {it + 1}/{n_iters}: fitting {len(obs)} "
            f"real particles (p_ghost <= {q_thresh:.2f})...",
            flush=True,
        )
        fit = _fit(obs, seed, pt_w, best_cals, cam_bad)
        if fit is None:
            print(
                f"[tracer self-cal] iter {it + 1}/{n_iters}: solver failed, stopping",
                flush=True,
            )
            break
        cand_cals, cand_success = fit
        # Evaluate candidate cals on the initial clean reference set `obs0` (and
        # on the current iteration's `obs`) so tightening `q_thresh` across passes
        # is compared apples-to-apples.
        cand_rcm_ref = _tracer_rcm_median(obs0, cand_cals, cpar)
        cand_rcm = _tracer_rcm_median(obs, cand_cals, cpar)
        improved = cand_rcm_ref is not None and (
            best_rcm is None or cand_rcm_ref < best_rcm - 1e-9
        )
        cand_diag = camera_ghost_diagnostics(obs_all_it, seed_all_it, cand_cals, cpar)
        trace.append(
            {
                "rcm": cand_rcm_ref,
                "rcm_subset": cand_rcm,
                "n_particles": len(obs),
                "q_thresh": q_thresh,
                "real_rcm": cand_diag["real_rcm_median"],
                "ghost_rcm": cand_diag["ghost_rcm_median"],
                "separation_ratio": cand_diag["separation_ratio"],
                "accepted": improved,
            }
        )
        cand_rcm_str = (
            "n/a" if cand_rcm_ref is None else f"{cand_rcm_ref * 1000:.1f} um"
        )
        sep_str = (
            "n/a"
            if cand_diag["separation_ratio"] is None
            else f"{cand_diag['separation_ratio']:.2f}x"
        )
        print(
            f"[tracer self-cal] iter {it + 1}/{n_iters}: RCM = {cand_rcm_str}, "
            f"ghost/real sep = {sep_str} "
            f"({'accepted, improved' if improved else 'not improved, stopping'})",
            flush=True,
        )
        if not improved:
            break
        best_cals, best_rcm, success = cand_cals, cand_rcm_ref, cand_success

    obs_all_final, seed_all_final, lnk_all_final = _gather_candidates(best_cals)
    obs_final, _, _ = _select_real_particles(
        obs_all_final,
        seed_all_final,
        lnk_all_final,
        best_cals,
        ghost_prob_schedule[-1] if ghost_prob_schedule else 0.15,
    )
    diag_after = camera_ghost_diagnostics(
        obs_all_final, seed_all_final, best_cals, cpar
    )
    print("[tracer self-cal] done", flush=True)
    return best_cals, {
        "rcm_before": rcm_before,
        "rcm_after": best_rcm,
        "real_rcm_before": diag_before["real_rcm_median"],
        "real_rcm_after": diag_after["real_rcm_median"],
        "ghost_rcm_before": diag_before["ghost_rcm_median"],
        "ghost_rcm_after": diag_after["ghost_rcm_median"],
        "separation_ratio_before": diag_before["separation_ratio"],
        "separation_ratio_after": diag_after["separation_ratio"],
        "camera_diagnostics_before": diag_before["per_camera"],
        "camera_diagnostics_after": diag_after["per_camera"],
        "best_cam": diag_before["best_cam"],
        "worst_cam": diag_before["worst_cam"],
        "n_particles": len(obs_final),
        "n_obs": sum(len(p) for p in obs_final),
        "hold_cam": hold_cam_idx,
        "iterations": len([t for t in trace if t["accepted"]]),
        "rcm_trace": trace,
        "success": success,
    }

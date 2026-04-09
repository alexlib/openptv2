"""Pure-Python batch processing for 3D-PTV.

This module mirrors ``gui.pyptv.pyptv_batch`` but uses **only** the
``algorithms`` package (Python / Numba).  No Cython ``optv`` bindings are
required, so it can run anywhere that Python + NumPy are available.

Typical usage
-------------
Command line::

    python -m algorithms.batch tests/test_cavity/parameters_Run1.yaml 10001 10004

Python API::

    from algorithms.batch import main
    main("tests/test_cavity/parameters_Run1.yaml", 10001, 10004)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from .calibration import Calibration, read_calibration
from .correspondences import MatchedCoords, correspondences
from .image_processing import preprocess_image
from .orientation import point_positions
from .parameters import (
    ControlPar,
    MultimediaPar,
    SequencePar,
    TargetPar,
    TrackPar,
    VolumePar,
)
from .segmentation import target_recognition
from .track import Tracker, TrackingObserver, default_naming
from .tracking_frame_buf import Frame, read_targets


# ---------------------------------------------------------------------------
# Parameter helpers — build pure-Python objects straight from YAML dicts
# ---------------------------------------------------------------------------

def _build_control_par(ptv: dict, num_cams: int) -> ControlPar:
    mm = MultimediaPar(
        nlay=1,
        n1=ptv["mmp_n1"],
        n2=[ptv["mmp_n2"]],
        d=[ptv["mmp_d"]],
        n3=ptv["mmp_n3"],
    )
    img_base = [ptv["img_name"][i] for i in range(num_cams)]
    cal_img = [ptv["img_cal"][i] for i in range(num_cams)]
    return ControlPar(
        num_cams=num_cams,
        img_base_name=img_base,
        cal_img_base_name=cal_img,
        hp_flag=int(ptv.get("hp_flag", True)),
        all_cam_flag=int(ptv.get("allcam_flag", False)),
        tiff_flag=int(ptv.get("tiff_flag", True)),
        chfield=ptv.get("chfield", 0),
        imx=ptv["imx"],
        imy=ptv["imy"],
        pix_x=ptv["pix_x"],
        pix_y=ptv["pix_y"],
        mm=mm,
    )


def _build_sequence_par(seq: dict, num_cams: int) -> SequencePar:
    base_names = seq["base_name"]
    return SequencePar(
        img_base_name=[base_names[i] for i in range(num_cams)],
        first=seq["first"],
        last=seq["last"],
    )


def _build_volume_par(crit: dict) -> VolumePar:
    return VolumePar(
        x_lay=crit["X_lay"],
        z_min_lay=crit["Zmin_lay"],
        z_max_lay=crit["Zmax_lay"],
        cn=crit["cn"],
        cnx=crit["cnx"],
        cny=crit["cny"],
        csumg=crit["csumg"],
        eps0=crit["eps0"],
        corrmin=crit["corrmin"],
    )


def _build_track_par(tp: dict) -> TrackPar:
    return TrackPar(
        dvxmin=tp["dvxmin"],
        dvxmax=tp["dvxmax"],
        dvymin=tp["dvymin"],
        dvymax=tp["dvymax"],
        dvzmin=tp["dvzmin"],
        dvzmax=tp["dvzmax"],
        dangle=tp["angle"],
        dacc=tp["dacc"],
        add=int(tp.get("flagNewParticles", True)),
    )


def _build_target_par(targ_rec: dict, num_cams: int) -> TargetPar:
    return TargetPar(
        gvthresh=targ_rec["gvthres"][:num_cams],
        discont=targ_rec.get("disco", 100),
        nnmin=targ_rec.get("nnmin", 1),
        nnmax=targ_rec.get("nnmax", 500),
        nxmin=targ_rec.get("nxmin", 1),
        nxmax=targ_rec.get("nxmax", 100),
        nymin=targ_rec.get("nymin", 1),
        nymax=targ_rec.get("nymax", 100),
        sumg_min=targ_rec.get("sumg_min", 10),
        cr_sz=targ_rec.get("cr_sz", 1),
    )


def _read_calibrations_py(cal_ori: dict, num_cams: int) -> list[Calibration]:
    """Load calibration files (*.ori + *.addpar) for every camera."""
    cals: list[Calibration] = []
    ori_files = cal_ori.get("img_ori", [])
    for i in range(num_cams):
        ori_file = Path(ori_files[i]) if i < len(ori_files) else None
        if ori_file is not None and ori_file.exists():
            addpar = ori_file.with_suffix(".addpar")
            if not addpar.exists():
                # Try conventional naming: basename.addpar
                base = ori_file.with_suffix("")
                addpar = Path(str(base) + ".addpar")
            cal = read_calibration(
                ori_file, addpar if addpar.exists() else None
            )
        else:
            print(f"  Camera {i + 1}: calibration not found ({ori_file}), using defaults")
            cal = Calibration()
        cals.append(cal)
    return cals


def _target_file_bases(seq_base_names: list[str], num_cams: int) -> list[str]:
    """Derive per-camera short target-file bases from sequence base names."""
    bases = []
    for i, bn in enumerate(seq_base_names):
        parent = Path(bn).parent
        bases.append(str(parent / f"cam{i + 1}"))
    return bases


# ---------------------------------------------------------------------------
# Sequence loop (detection + correspondence + 3-D determination)
# ---------------------------------------------------------------------------

def _sequence_loop(
    params: dict,
    cpar: ControlPar,
    spar: SequencePar,
    vpar: VolumePar,
    tpar: TargetPar,
    cals: list[Calibration],
    num_cams: int,
) -> None:
    """Run detection → correspondence → 3-D determination for every frame."""
    first_frame = spar.first
    last_frame = spar.last
    short_file_bases = _target_file_bases(spar.img_base_name, num_cams)
    existing_target = params.get("pft_version", {}).get("Existing_Target", False)

    for frame in range(first_frame, last_frame + 1):
        t_frame = time.perf_counter()
        detections = []
        corrected = []

        print(f"Frame {frame}: loading images …", flush=True)
        for i_cam in range(num_cams):
            if existing_target:
                targs = read_targets(short_file_bases[i_cam], frame)
            else:
                img_name = spar.img_base_name[i_cam] % frame
                img_path = Path(img_name)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                from skimage.io import imread
                from skimage.color import rgb2gray
                from skimage.util import img_as_ubyte

                img = imread(str(img_path))
                if img.ndim > 2:
                    img = rgb2gray(img)
                if img.dtype != np.uint8:
                    img = img_as_ubyte(img)

                if params.get("ptv", {}).get("hp_flag", True):
                    img = preprocess_image(img, 0, cpar, 3)

                targs = target_recognition(img, tpar, i_cam, cpar)

            # Sort targets by y-coordinate
            if hasattr(targs, "sort_y"):
                targs.sort_y()
            elif isinstance(targs, list):
                targs.sort(key=lambda t: t.y)

            detections.append(targs)
            mc = MatchedCoords(targs, cpar, cals[i_cam])
            corrected.append(mc)

        tgt_counts = [len(d) for d in detections]
        print(f"  detection done – targets per cam: {tgt_counts}  "
              f"({time.perf_counter()-t_frame:.1f}s)", flush=True)

        # Build a Frame for the correspondence routine
        frm = Frame(num_cams=num_cams)
        for i_cam in range(num_cams):
            n = len(detections[i_cam])
            frm.num_targets[i_cam] = n
            for tnum in range(n):
                t = detections[i_cam][tnum]
                frm.targets[i_cam][tnum].pnr = getattr(t, "pnr", tnum)
                frm.targets[i_cam][tnum].tnr = -1
                frm.targets[i_cam][tnum].x = getattr(t, "x", 0)
                frm.targets[i_cam][tnum].y = getattr(t, "y", 0)
                frm.targets[i_cam][tnum].n = getattr(t, "n", 0)
                frm.targets[i_cam][tnum].nx = getattr(t, "nx", 0)
                frm.targets[i_cam][tnum].ny = getattr(t, "ny", 0)
                frm.targets[i_cam][tnum].sumg = getattr(t, "sumg", 0)

        print(f"  stereo-matching …", flush=True)
        t_corr = time.perf_counter()
        match_counts = [0] * 4
        con = correspondences(frm, corrected, vpar, cpar, cals, match_counts)
        print(f"  stereo-matching done  ({time.perf_counter()-t_corr:.1f}s)", flush=True)

        total = match_counts[3] if len(match_counts) > 3 else 0
        if total > 0:
            valid = con[:total]
            order = np.argsort(-valid.corr)
            valid = valid[order]
            # con.p values are sorted-array indices into corrected[cam].
            # Convert to particle numbers (pnr) for get_by_pnrs lookups.
            corresp_idx = np.array([list(row.p) for row in valid]).T  # (num_cams, N)
            corresp = np.empty_like(corresp_idx)
            for i_cam in range(num_cams):
                mask = corresp_idx[i_cam] >= 0
                corresp[i_cam] = corresp_idx[i_cam]
                corresp[i_cam, mask] = corrected[i_cam][corresp_idx[i_cam, mask]].pnr
        else:
            corresp = np.zeros((num_cams, 0), dtype=int)

        # Write target files
        for i_cam in range(num_cams):
            targs = detections[i_cam]
            out = Path(f"{short_file_bases[i_cam]}.{frame:04d}_targets")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf8") as f:
                f.write(f"{len(targs)}\n")
                for t in targs:
                    f.write(
                        f"{getattr(t, 'pnr', 0):4d} "
                        f"{getattr(t, 'x', 0.0):9.4f} "
                        f"{getattr(t, 'y', 0.0):9.4f} "
                        f"{getattr(t, 'n', 0):5d} "
                        f"{getattr(t, 'nx', 0):5d} "
                        f"{getattr(t, 'ny', 0):5d} "
                        f"{getattr(t, 'sumg', 0):5d} "
                        f"{getattr(t, 'tnr', -1):5d}\n"
                    )

        # 3-D determination via point_positions
        print(f"  3-D determination …", flush=True)
        t_3d = time.perf_counter()
        if corresp.shape[1] > 0:
            flat = np.array(
                [
                    corr.get_by_pnrs(c_row)
                    for corr, c_row in zip(corrected, corresp)
                ]
            )
            pos, _ = point_positions(
                flat.transpose(1, 0, 2), cpar.mm, cals, vpar
            )
        else:
            pos = np.zeros((0, 3))
        print(f"  3-D done – {pos.shape[0]} points  ({time.perf_counter()-t_3d:.1f}s)", flush=True)

        # Pad to 4 cameras if fewer
        if num_cams < 4:
            print_corresp = -1 * np.ones((4, corresp.shape[1]), dtype=int)
            print_corresp[:num_cams, :] = corresp
        else:
            print_corresp = corresp

        # Write rt_is (correspondence) file
        corres_base = default_naming["corres"]
        out_rt = Path(f"{corres_base}.{frame}")
        out_rt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_rt, "w", encoding="utf8") as f:
            f.write(f"{pos.shape[0]}\n")
            for pix, pt in enumerate(pos):
                pt_args = (pix + 1,) + tuple(pt) + tuple(print_corresp[:, pix])
                f.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)

        print(
            f"Frame {frame} done: {corresp.shape[1]} correspondences"
            f"  [{time.perf_counter()-t_frame:.1f}s]",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ProcessingError(Exception):
    """Raised when batch processing fails."""


def run_batch(
    yaml_file: Path,
    seq_first: int,
    seq_last: int,
    mode: str = "both",
    observer: TrackingObserver | None = None,
) -> Tracker | None:
    """Run batch processing using pure-Python algorithms.

    Args:
        yaml_file: Resolved path to the YAML parameter file.
        seq_first: First frame number.
        seq_last: Last frame number.
        mode: ``"both"`` (default), ``"sequence"``, or ``"tracking"``.
        observer: Optional :class:`TrackingObserver` attached to the tracker.

    Returns:
        The :class:`Tracker` instance (only when *mode* includes tracking).
    """
    if not yaml_file.exists():
        raise ProcessingError(f"YAML file not found: {yaml_file}")

    exp_path = yaml_file.parent
    original_cwd = Path.cwd()

    try:
        os.chdir(exp_path)

        with open(yaml_file) as f:
            params = yaml.safe_load(f)

        t_setup = time.perf_counter()
        num_cams = params["num_cams"]
        cpar = _build_control_par(params["ptv"], num_cams)
        spar = _build_sequence_par(params["sequence"], num_cams)
        vpar = _build_volume_par(params["criteria"])
        tpar_track = _build_track_par(params["track"])
        tpar_detect = _build_target_par(params["targ_rec"], num_cams)
        cals = _read_calibrations_py(params["cal_ori"], num_cams)
        print(f"Parameter setup: {time.perf_counter()-t_setup:.3f}s")

        # Override frame range
        spar.first = seq_first
        spar.last = seq_last

        tracker = None

        if mode in ("both", "sequence"):
            print("Running sequence loop (python)…")
            _sequence_loop(params, cpar, spar, vpar, tpar_detect, cals, num_cams)

        if mode in ("both", "tracking"):
            print("Running tracking (python)…")
            tracker = Tracker(
                cpar, vpar, tpar_track, spar, cals, default_naming,
            )
            tracker.full_forward(observer=observer)

        print("Batch processing complete (python engine).")
        return tracker

    except Exception as e:
        raise ProcessingError(f"Batch processing failed: {e}") from e
    finally:
        os.chdir(original_cwd)


def main(
    yaml_file: Union[str, Path],
    first: Union[str, int],
    last: Union[str, int],
    repetitions: int = 1,
    mode: str = "both",
    observer: TrackingObserver | None = None,
) -> Tracker | None:
    """Entry point for pure-Python batch processing.

    Args:
        yaml_file: Path to the YAML parameter file.
        first: First frame number.
        last: Last frame number.
        repetitions: How many times to repeat (default 1).
        mode: ``"both"``, ``"sequence"``, or ``"tracking"``.
        observer: Optional :class:`TrackingObserver` to attach.

    Returns:
        The last :class:`Tracker` instance when *mode* includes tracking.
    """
    start_time = time.time()

    yaml_file = Path(yaml_file).resolve()
    seq_first = int(first)
    seq_last = int(last)

    if seq_first > seq_last:
        raise ValueError(
            f"First frame ({seq_first}) must be <= last frame ({seq_last})"
        )

    res_path = yaml_file.parent / "res"
    res_path.mkdir(parents=True, exist_ok=True)

    print(f"Python-only batch: {yaml_file}")
    print(f"Frames {seq_first}–{seq_last}, repetitions={repetitions}")

    tracker = None
    for i in range(repetitions):
        if repetitions > 1:
            print(f"--- repetition {i + 1}/{repetitions} ---")
        tracker = run_batch(
            yaml_file, seq_first, seq_last, mode=mode, observer=observer,
        )

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s")
    return tracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pure-Python batch processing for OpenPTV"
    )
    parser.add_argument("yaml_file", help="YAML parameter file")
    parser.add_argument("first_frame", nargs="?", type=int, default=None)
    parser.add_argument("last_frame", nargs="?", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["both", "sequence", "tracking"],
        default="both",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml_file).resolve()
    if not yaml_path.exists():
        print(f"File not found: {yaml_path}")
        sys.exit(1)

    if args.first_frame is None or args.last_frame is None:
        with open(yaml_path) as _f:
            _p = yaml.safe_load(_f)
        first = args.first_frame or _p["sequence"]["first"]
        last = args.last_frame or _p["sequence"]["last"]
    else:
        first = args.first_frame
        last = args.last_frame

    main(yaml_path, first, last, mode=args.mode)

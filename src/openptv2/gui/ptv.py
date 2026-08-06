"""PyPTV core functionality module.

This module provides the core functionality for the PyPTV package, including
image processing, calibration, tracking, and other utilities.
"""

# Standard library imports
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

# Third-party imports
import numpy as np
from imageio.v3 import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

# Backend imports from openptv2 (dual-engine: optv or algorithms)
from openptv2.calibration import Calibration
from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.image_processing import preprocess_image
from openptv2.orientation import point_positions
from openptv2.parameters import (
    ControlParams,
    SequenceParams,
    TargetParams,
    TrackingParams,
    VolumeParams,
)
from openptv2.segmentation import target_recognition
from openptv2.tracker import Tracker, default_naming
from openptv2.tracking_framebuf import TargetArray

# PyPTV imports
from . import ptv_calibration
from .parameter_manager import ParameterManager

# Re-exported for callers that import them from this module (GUI + tests).
from .ptv_calibration import (  # noqa: F401
    _read_calibrations,
    clone_calibration,
    full_scipy_calibration,
)

# example from Tracker documentation:
#         dict naming - a dictionary with naming rules for the frame buffer
#             files. Keys: 'corres', 'linkage', 'prio'. Values can be either
#             strings or bytes. Strings are automatically encoded to UTF-8.
#             If None, uses default_naming.
#
#     default_naming = {
#         'corres': 'res/rt_is',
#         'linkage': 'res/ptv_is',
#         'prio': 'res/added'
#     }

# Constants
DEFAULT_FRAME_NUM = 123456789
DEFAULT_HIGHPASS_FILTER_SIZE = 25
DEFAULT_NO_FILTER = 0
SHORT_BASE = "cam"  # Use this as the short base for camera file naming


def _safe_decode(val) -> str:
    """Safely decode bytes to string or return the string directly."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _extract_frame_num(img_name: str) -> int:
    """Extract frame number from image filename if possible, else return DEFAULT_FRAME_NUM."""
    if not img_name:
        return DEFAULT_FRAME_NUM

    # Try parsing suffix as an integer, e.g. "img/cam1.10002" -> 10002
    suffix = Path(img_name).suffix
    if suffix and suffix.startswith("."):
        try:
            return int(suffix[1:])
        except ValueError:
            pass

    # Try finding digits in the stem, e.g. "00000001" or "cam1_10002"
    stem = Path(img_name).stem
    digits = re.findall(r"\d+", stem)
    if digits:
        try:
            return int(digits[-1])
        except ValueError:
            pass

    return DEFAULT_FRAME_NUM


def _prepare_output_path(filename: str) -> Path:
    """Return a writable output path, creating parent directories when needed."""
    output_path = Path(filename)
    parent = output_path.parent

    if parent != Path("."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                "Unable to prepare output directory "
                f"'{parent}' for '{output_path.name}'. "
                "Please choose a writable experiment folder or change the folder permissions."
            ) from exc

    return output_path


def _raise_output_write_error(output_path: Path, exc: OSError) -> None:
    """Raise an actionable write error for generated output files."""
    if isinstance(exc, PermissionError):
        raise PermissionError(
            f"Cannot write output file '{output_path}'. "
            f"PyPTV does not have permission to write to '{output_path.parent}'. "
            "Please change the folder permissions or move the experiment to a user-writable directory."
        ) from exc

    raise OSError(f"Failed to write output file '{output_path}': {exc}") from exc


def _ensure_directory_writable(directory: Union[str, Path], label: str) -> Path:
    """Create and probe an output directory before writing generated files."""
    directory = Path(directory)
    print(f"Checking {label} directory {directory}")
    probe_path = _prepare_output_path(
        str(directory / f".pyptv_write_probe_{os.getpid()}")
    )

    try:
        with open(probe_path, "w", encoding="utf-8") as probe_file:
            probe_file.write("pyptv write probe\n")
    except OSError as exc:
        _raise_output_write_error(probe_path, exc)
    finally:
        try:
            probe_path.unlink(missing_ok=True)
        except OSError:
            pass

    print(f"{label} directory {directory} is writable.")
    return directory


def _ensure_target_output_writable(short_file_bases: List[str]) -> None:
    """Check target output directories before the first target file write."""
    checked_dirs = set()

    for short_file_base in short_file_bases:
        directory = Path(short_file_base).parent
        directory_key = (
            str(directory.resolve()) if directory.exists() else str(directory)
        )
        if directory_key in checked_dirs:
            continue

        _ensure_directory_writable(directory, "Target output")
        checked_dirs.add(directory_key)


def _process_frame_worker(args: Tuple) -> int:
    """Worker function to process a single frame for target recognition.

    This function runs in a separate worker process to avoid GIL bottlenecks.
    It takes a tuple of arguments to support easy pickling.
    """
    (
        frame,
        num_cams,
        img_base_names,
        short_file_bases,
        ptv_params,
        targ_params,
        negative_flag,
        masking_params,
        zarr_store_path,
    ) = args

    # Recreate the ControlParams and TargetParams objects inside the worker
    cpar = _populate_cpar(ptv_params, num_cams)
    tpar = _populate_tpar(targ_params, num_cams)

    store = None
    if zarr_store_path:
        from openptv2.storage import ZarrFrameStore

        store = ZarrFrameStore(zarr_store_path, mode="a")

    for i_cam in range(num_cams):
        imname = Path(img_base_names[i_cam] % frame)
        if not imname.exists():
            raise FileNotFoundError(f"{imname} does not exist")
        else:
            img = imread(imname)
            if img.ndim > 2:
                img = rgb2gray(img[:, :, :3])
            if img.dtype != np.uint8:
                img = img_as_ubyte(img)
        if negative_flag:
            img = negative(img)
        if masking_params and masking_params.get("mask_flag", False):
            try:
                background_name = masking_params["mask_base_name"] % (i_cam + 1)
                background = imread(background_name)
                img = np.clip(img - background, 0, 255).astype(np.uint8)
            except (ValueError, FileNotFoundError):
                pass
        high_pass = simple_highpass(
            img, cpar, ptv_params.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE)
        )
        targs = target_recognition(high_pass, tpar, i_cam, cpar)

        if len(targs) > 0:
            if hasattr(targs, "sort_y"):
                targs.sort_y()
            else:
                targs.sort(key=lambda t: t.y)

        if store is not None:
            store.write_targets(i_cam, frame, targs)
        else:
            write_targets(targs, short_file_bases[i_cam], frame)

    return frame


def preprocess_and_detect_all_parallel(
    exp, num_workers: int = None, zarr_store_path: str = None
) -> None:
    """Preprocess and detect targets in parallel across all frames.

    Args:
        exp: Either an Experiment object with pm attribute,
             or a MainGUI object with exp1.pm and cached parameter objects
        num_workers: Optional number of worker processes. Defaults to all available cores.
        zarr_store_path: Optional path to Zarr store for writing targets directly.
    """
    from concurrent.futures import ProcessPoolExecutor

    # Handle both Experiment objects and MainGUI objects
    if hasattr(exp, "pm"):
        pm = exp.pm
        num_cams = pm.num_cams
        spar = exp.spar
    elif hasattr(exp, "exp1") and hasattr(exp.exp1, "pm"):
        pm = exp.exp1.pm
        num_cams = exp.num_cams
        spar = exp.spar
    else:
        raise ValueError("Object must have either pm or exp1.pm attribute")

    first_frame = spar.get_first()
    last_frame = spar.get_last()
    img_base_names = [spar.get_img_base_name(i) for i in range(num_cams)]
    short_file_bases = exp.target_filenames

    _ensure_target_output_writable(short_file_bases)

    if zarr_store_path:
        from openptv2.storage.zarr_store import ZarrFrameStore, _get_or_create_group

        pre_store = ZarrFrameStore(zarr_store_path, mode="a")
        targets_grp = _get_or_create_group(pre_store.root, "targets")
        for icam in range(num_cams):
            _get_or_create_group(targets_grp, f"cam_{icam}")

    # Extract clean python dicts for parameters
    if hasattr(pm, "parameters") and isinstance(pm.parameters, dict):
        ptv_params_dict = pm.parameters.get("ptv", {})
        masking_params_dict = pm.parameters.get("masking", {})
        targ_params_dict = {}
        if "targ_rec" in pm.parameters:
            targ_params_dict["targ_rec"] = pm.parameters["targ_rec"]
        elif "detect_plate" in pm.parameters:
            targ_params_dict["detect_plate"] = pm.parameters["detect_plate"]
    else:
        try:
            ptv_params_dict = pm.get_parameter("ptv")
        except ValueError:
            ptv_params_dict = {}
        try:
            masking_params_dict = pm.get_parameter("masking")
        except ValueError:
            masking_params_dict = {}
        targ_params_dict = {}
        try:
            targ_params_dict["targ_rec"] = pm.get_parameter("targ_rec")
        except ValueError:
            try:
                targ_params_dict["detect_plate"] = pm.get_parameter("detect_plate")
            except ValueError:
                pass

    negative_flag = ptv_params_dict.get("negative", False)

    # Determine num_workers
    if num_workers is None:
        try:
            num_workers = int(os.environ.get("OPENPTV_NUM_WORKERS", 0))
            if num_workers <= 0:
                num_workers = None
        except ValueError:
            num_workers = None

    tasks = [
        (
            frame,
            num_cams,
            img_base_names,
            short_file_bases,
            ptv_params_dict,
            targ_params_dict,
            negative_flag,
            masking_params_dict,
            zarr_store_path,
        )
        for frame in range(first_frame, last_frame + 1)
    ]

    print(
        f"Starting parallel target detection for {len(tasks)} frames using "
        f"{num_workers or 'all'} processes..."
    )

    if num_workers == 1:
        print("Running target detection sequentially on 1 worker...")
        for task in tasks:
            _process_frame_worker(task)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_process_frame_worker, tasks))

    print("Parallel target detection completed successfully.")


DEFAULT_SPLITTER_ORDER = (0, 1, 3, 2)


def image_split(img: np.ndarray, order=None) -> List[np.ndarray]:
    """Split image into four quadrants, reordered by ``order``.

    ``order`` maps output camera index to quadrant (TL, TR, BL, BR) and
    comes from the ``ptv.splitter_order`` YAML parameter; the default is
    the historical hardware order.
    """
    if order is None:
        order = DEFAULT_SPLITTER_ORDER
    list_of_images = [
        img[: img.shape[0] // 2, : img.shape[1] // 2],
        img[: img.shape[0] // 2, img.shape[1] // 2 :],
        img[img.shape[0] // 2 :, : img.shape[1] // 2],
        img[img.shape[0] // 2 :, img.shape[1] // 2 :],
    ]
    list_of_images = [list_of_images[i] for i in order]
    return list_of_images


def negative(img: np.ndarray) -> np.ndarray:
    """Convert an 8-bit image to its negative."""
    return 255 - img


def simple_highpass(
    img: np.ndarray, cpar: ControlParams, filter_size: int = None
) -> np.ndarray:
    """Apply a simple highpass filter to an image using liboptv preprocess_image."""
    if filter_size is None:
        filter_size = DEFAULT_HIGHPASS_FILTER_SIZE
    return preprocess_image(img, 0, cpar, filter_size)


def _populate_cpar(ptv_params: dict, num_cams: int) -> ControlParams:
    """Populate a ControlParams object from a dictionary containing full parameters.

    Args:
        params: Full parameter dictionary with global num_cams and ptv section
    """
    # ptv_params = params.get('ptv', {})

    img_cal_list = ptv_params.get("img_cal", [])
    if len([x for x in img_cal_list if x is not None]) < num_cams:
        raise ValueError("img_cal_list is too short")

    cpar = ControlParams(num_cams)
    # Set required parameters directly from the dictionary, no defaults
    cpar.set_image_size((ptv_params["imx"], ptv_params["imy"]))
    cpar.set_pixel_size((ptv_params["pix_x"], ptv_params["pix_y"]))
    cpar.set_hp_flag(ptv_params["hp_flag"])
    cpar.set_allCam_flag(ptv_params["allcam_flag"])
    cpar.set_tiff_flag(ptv_params["tiff_flag"])
    cpar.set_chfield(ptv_params["chfield"])

    mm_params = cpar.get_multimedia_params()
    mm_params.set_n1(ptv_params["mmp_n1"])
    mm_params.set_layers([ptv_params["mmp_n2"]], [ptv_params["mmp_d"]])
    mm_params.set_n3(ptv_params["mmp_n3"])

    img_cal_list = ptv_params["img_cal"]

    for i in range(num_cams):  # Use global num_cams
        cpar.set_cal_img_base_name(i, img_cal_list[i])
    return cpar


def _populate_spar(seq_params: dict, num_cams: int) -> SequenceParams:
    """Populate a SequenceParams object from a dictionary.

    Raises ValueError if required sequence parameters are missing.
    No default values are provided to avoid silent failures.
    """
    required_params = ["first", "last", "base_name"]
    missing_params = [param for param in required_params if param not in seq_params]

    if missing_params:
        raise ValueError(
            f"Missing required sequence parameters: {missing_params}. "
            f"Available parameters: {list(seq_params.keys())}"
        )

    base_name_list = seq_params["base_name"]

    if len([x for x in base_name_list if x is not None]) < num_cams:
        raise ValueError(
            f"base_name_list length ({len(base_name_list)}) does not match num_cams ({num_cams})"
        )

    spar = SequenceParams(num_cams=num_cams)
    spar.set_first(seq_params["first"])
    spar.set_last(seq_params["last"])

    # Set base names for each camera
    for i in range(num_cams):
        spar.set_img_base_name(i, base_name_list[i])

    return spar


def _populate_vpar(crit_params: dict) -> VolumeParams:
    """Populate a VolumeParams object from a dictionary."""
    vpar = VolumeParams()
    vpar.set_X_lay(crit_params["X_lay"])
    vpar.set_Zmin_lay(crit_params["Zmin_lay"])
    vpar.set_Zmax_lay(crit_params["Zmax_lay"])

    # Set correspondence parameters
    vpar.set_eps0(crit_params["eps0"])
    vpar.set_cn(crit_params["cn"])
    vpar.set_cnx(crit_params["cnx"])
    vpar.set_cny(crit_params["cny"])
    vpar.set_csumg(crit_params["csumg"])
    vpar.set_corrmin(crit_params["corrmin"])

    return vpar


def _populate_track_par(track_params: dict) -> TrackingParams:
    """Populate a TrackingParams object from a dictionary.

    Raises ValueError if required tracking parameters are missing.
    No default values are provided to avoid silent tracking failures.
    """
    required_params = [
        "dvxmin",
        "dvxmax",
        "dvymin",
        "dvymax",
        "dvzmin",
        "dvzmax",
        "angle",
        "dacc",
        "flagNewParticles",
    ]
    missing_params = [param for param in required_params if param not in track_params]

    if missing_params:
        raise ValueError(
            f"Missing required tracking parameters: {missing_params}. "
            f"Available parameters: {list(track_params.keys())}"
        )

    track_par = TrackingParams()
    track_par.set_dvxmin(track_params["dvxmin"])
    track_par.set_dvxmax(track_params["dvxmax"])
    track_par.set_dvymin(track_params["dvymin"])
    track_par.set_dvymax(track_params["dvymax"])
    track_par.set_dvzmin(track_params["dvzmin"])
    track_par.set_dvzmax(track_params["dvzmax"])
    track_par.set_dangle(track_params["angle"])
    track_par.set_dacc(track_params["dacc"])
    track_par.set_add(track_params["flagNewParticles"])
    return track_par


def _populate_tpar(targ_params: dict, num_cams: int) -> TargetParams:
    """Populate a TargetParams object from a dictionary."""
    # targ_params = params.get('targ_rec', {})

    # Get global num_cams - the single source of truth
    # num_cams = params.get('num_cams', 0)

    tpar = TargetParams(num_cams)
    # Handle both 'targ_rec' and 'detect_plate' parameter variants
    if "targ_rec" in targ_params:
        params = targ_params["targ_rec"]
        tpar.set_grey_thresholds(params["gvthres"])
        tpar.set_pixel_count_bounds((params["nnmin"], params["nnmax"]))
        tpar.set_xsize_bounds((params["nxmin"], params["nxmax"]))
        tpar.set_ysize_bounds((params["nymin"], params["nymax"]))
        tpar.set_min_sum_grey(params["sumg_min"])
        tpar.set_max_discontinuity(params["disco"])
        tpar.set_cross_size(params.get("cr_sz", 2))
    elif "detect_plate" in targ_params:
        params = targ_params["detect_plate"]
        # Convert detect_plate keys to TargetParams fields
        # Ensure all required grey thresholds are present
        required_gvth_keys = ["gvth_1", "gvth_2", "gvth_3", "gvth_4"]
        missing_keys = [k for k in required_gvth_keys if k not in params]
        if missing_keys:
            raise ValueError(
                f"Missing required grey threshold keys in detect_plate: {missing_keys}"
            )
        tpar.set_grey_thresholds(
            [
                params["gvth_1"],
                params["gvth_2"],
                params["gvth_3"],
                params["gvth_4"],
            ]
        )
        # Remove default values - all parameters must be explicitly provided
        required_detect_keys = [
            "min_npix",
            "max_npix",
            "min_npix_x",
            "max_npix_x",
            "min_npix_y",
            "max_npix_y",
            "sum_grey",
            "tol_dis",
        ]
        missing_detect_keys = [k for k in required_detect_keys if k not in params]
        if missing_detect_keys:
            raise ValueError(
                f"Missing required detect_plate keys: {missing_detect_keys}"
            )

        tpar.set_pixel_count_bounds((params["min_npix"], params["max_npix"]))
        tpar.set_xsize_bounds((params["min_npix_x"], params["max_npix_x"]))
        tpar.set_ysize_bounds((params["min_npix_y"], params["max_npix_y"]))
        tpar.set_min_sum_grey(params["sum_grey"])
        tpar.set_max_discontinuity(params["tol_dis"])
        tpar.set_cross_size(params.get("size_cross", 3))
    else:
        raise ValueError(
            "Target parameters must contain either 'targ_rec' or 'detect_plate' section."
        )
    return tpar


def py_start_proc_c(
    pm: ParameterManager,
) -> Tuple[
    ControlParams,
    SequenceParams,
    VolumeParams,
    TrackingParams,
    TargetParams,
    List[Calibration],
    dict,
]:
    """Read all parameters needed for processing using ParameterManager."""
    try:
        params = pm.parameters
        num_cams = pm.num_cams

        cpar = _populate_cpar(params["ptv"], num_cams)
        spar = _populate_spar(params["sequence"], num_cams)
        vpar = _populate_vpar(params["criteria"])
        track_par = _populate_track_par(params["track"])

        # Create a dict that contains targ_rec for _populate_tpar
        # Use targ_rec instead of detect_plate to match manual GUI operations
        target_params_dict = {"targ_rec": params["targ_rec"]}
        tpar = _populate_tpar(target_params_dict, num_cams)

        epar = params.get("examine")

        cals = ptv_calibration._read_calibrations(cpar, num_cams)

        # NOTE: the multimedia LUT is deliberately NOT pre-built here. The
        # benchmark (tests/perf/test_mmlut_benchmark.py) shows that in the
        # compiled runtime the iterative Snell solve is already fast, and the
        # main correspondence path (epi.epi_mm -> flat_image_coord) does not
        # even pass a LUT, so building one adds cost with no measured speedup.
        # openptv2.algorithms.multimed.prepare_mmluts(vpar, cpar, cals) is
        # available if a future efficient LUT call path (see the mmlut plan's
        # Phase 4) makes it worthwhile. The tracker still builds its own LUT.

        return cpar, spar, vpar, track_par, tpar, cals, epar

    except IOError as e:
        raise IOError(f"Failed to read parameter files: {e}")


def py_pre_processing_c(
    num_cams: int,
    list_of_images: List[np.ndarray],
    ptv_params: dict,
) -> List[np.ndarray]:
    """Apply pre-processing to a list of images."""
    # num_cams = len(list_of_images)
    cpar = _populate_cpar(ptv_params, num_cams)
    processed_images = []
    for i, img in enumerate(list_of_images):
        img_lp = img.copy()
        processed_images.append(
            simple_highpass(
                img_lp,
                cpar,
                ptv_params.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE),
            )
        )

    return processed_images


def py_detection_proc_c(
    num_cams: int,
    list_of_images: List[np.ndarray],
    ptv_params: dict,
    target_params: dict,
    existing_target: bool = False,
) -> Tuple[List[TargetArray], List[MatchedCoords]]:
    """Detect targets in a list of images."""
    # num_cams = len(ptv_params.get('img_cal', []))

    if len(list_of_images) != num_cams:
        raise ValueError(
            f"Number of images ({len(list_of_images)}) must match number of cameras ({num_cams})"
        )

    cpar = _populate_cpar(ptv_params, num_cams)

    # Create a dict that contains targ_rec for _populate_tpar
    # target_params_dict = {'targ_rec': target_params}
    tpar = _populate_tpar(target_params, num_cams)

    cals = ptv_calibration._read_calibrations(cpar, num_cams)

    if existing_target:
        raise NotImplementedError("Existing targets are not implemented")

    from concurrent.futures import ThreadPoolExecutor

    def _detect(args):
        i_cam, img = args
        targs = target_recognition(img.copy(), tpar, i_cam, cpar)
        targs.sort_y()
        return targs

    with ThreadPoolExecutor(max_workers=num_cams) as pool:
        detections = list(pool.map(_detect, enumerate(list_of_images)))

    corrected = [
        MatchedCoords(targs, cpar, cals[i]) for i, targs in enumerate(detections)
    ]

    return detections, corrected


def py_correspondences_proc_c(exp):
    """Provides correspondences"""
    frame = DEFAULT_FRAME_NUM
    if hasattr(exp, "exp1") and hasattr(exp.exp1, "pm"):
        pm = exp.exp1.pm
    elif hasattr(exp, "pm"):
        pm = exp.pm
    else:
        pm = None

    if pm is not None:
        ptv_params = pm.get_parameter("ptv")
        if (
            isinstance(ptv_params, dict)
            and "img_name" in ptv_params
            and ptv_params["img_name"]
        ):
            frame = _extract_frame_num(ptv_params["img_name"][0])

    sorted_pos, sorted_corresp, num_targs = correspondences(
        exp.detections, exp.corrected, exp.cals, exp.vpar, exp.cpar
    )

    # img_base_names = [exp.spar.get_img_base_name(i) for i in range(exp.num_cams)]
    short_file_bases = exp.target_filenames
    print(f"short_file_bases: {short_file_bases}")
    _ensure_target_output_writable(short_file_bases)

    for i_cam in range(exp.num_cams):
        write_targets(exp.detections[i_cam], short_file_bases[i_cam], frame)

    print(f"Frame {frame} had {[s.shape[1] for s in sorted_pos]!r} correspondences.")

    return sorted_pos, sorted_corresp, num_targs


def py_determination_proc_c(
    num_cams: int,
    sorted_pos: List[np.ndarray],
    sorted_corresp: List[np.ndarray],
    corrected: List[MatchedCoords],
    cpar: ControlParams,
    vpar: VolumeParams,
    cals: List[Calibration],
    frame: int = DEFAULT_FRAME_NUM,
) -> None:
    """Calculate 3D positions from 2D correspondences and save to file."""
    np.concatenate(sorted_pos, axis=1)
    concatenated_corresp = np.concatenate(sorted_corresp, axis=1)

    flat = np.array(
        [
            corr.get_by_pnrs(corresp)
            for corr, corresp in zip(corrected, concatenated_corresp)
        ]
    )

    pos, _ = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)

    if num_cams < 4:
        print_corresp = -1 * np.ones((4, concatenated_corresp.shape[1]))
        print_corresp[: len(cals), :] = concatenated_corresp
    else:
        print_corresp = concatenated_corresp

    storage_mode = os.environ.get("OPENPTV_STORAGE", "zarr").lower()
    if storage_mode in ("zarr", "zarr_only"):
        from openptv2.storage import ZarrFrameStore

        zarr_path = Path("res/run.zarr")
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        store = ZarrFrameStore(zarr_path, mode="a")
        store.write_correspondences(
            frame=frame, pos_3d=pos, cam_target_ids=print_corresp.T
        )
        print(f"Saved 3D correspondences for frame {frame} to Zarr store {zarr_path}")

    if storage_mode == "zarr_only":
        return

    output_path = _prepare_output_path(
        f"{_safe_decode(default_naming['corres'])}.{frame}"
    )

    print(f"Prepared {output_path} to write positions")

    try:
        with open(output_path, "w", encoding="utf-8") as rt_is:
            print(f"Opened {output_path}")
            rt_is.write(f"{pos.shape[0]}\n")
            for pix, pt in enumerate(pos):
                pt_args = (pix + 1,) + tuple(pt) + tuple(print_corresp[:, pix])
                rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
    except OSError as exc:
        _raise_output_write_error(output_path, exc)


def run_sequence_plugin(exp) -> None:
    """Load and run the sequence plugin selected in ``exp.plugins.sequence_alg``."""
    from openptv2.plugins import run_sequence_plugin as _run_sequence_plugin

    _run_sequence_plugin(exp.plugins.sequence_alg, exp)


def run_tracking_plugin(exp) -> None:
    """Load and run the tracking plugin selected in ``exp.plugins.track_alg``."""
    from openptv2.plugins import run_tracking_plugin as _run_tracking_plugin

    _run_tracking_plugin(exp.plugins.track_alg, exp)


def _frame_image_name(base_name, frame: int) -> Path:
    """Format a sequence base name into the image path for one frame."""
    base_name = _safe_decode(base_name)
    try:
        return Path(base_name % frame)
    except (TypeError, ValueError):
        # No usable % placeholder: append the frame number (legacy naming).
        base_path = Path(base_name)
        return base_path.parent / f"{base_path.stem}_{frame:04d}{base_path.suffix}"


def _read_gray_uint8(imname: Path) -> np.ndarray:
    """Read an image as 2D uint8 grayscale."""
    img = imread(imname)
    if img.ndim > 2:
        img = rgb2gray(img[:, :, :3])
    if img.dtype != np.uint8:
        img = img_as_ubyte(img)
    return img


def read_frame_images(pm, img_base_names, num_cams, frame) -> List[np.ndarray]:
    """Return the per-camera images for one frame, entirely in memory.

    This is the single image-acquisition point of the sequence pipeline:

    - splitter mode (``ptv.splitter``): read ONE multiplexed image (the
      camera-0 base name), optionally negate the full frame, and split it
      into ``num_cams`` views using ``ptv.splitter_order``. The split views
      are never written to disk — detection and stereo matching consume
      them directly.
    In both modes the per-camera background mask (``masking`` section) is
    subtracted per view afterwards.
    """
    ptv_params = pm.get_parameter("ptv") or {}
    masking_params = pm.get_parameter("masking") or {}
    apply_negative = ptv_params.get("negative", False)

    # 0. Check for native res/images.zarr store
    zarr_img_path = Path("res/images.zarr")
    if zarr_img_path.exists():
        try:
            import zarr

            zstore = zarr.open_group(str(zarr_img_path), mode="r")
            if "raw_images" in zstore:
                raw_arr = zstore["raw_images"]
                first_frame = pm.spar.get_first() if hasattr(pm, "spar") else 1
                frame_idx = (
                    frame - first_frame
                    if (frame - first_frame) < raw_arr.shape[0]
                    else (frame - 1)
                )
                if 0 <= frame_idx < raw_arr.shape[0]:
                    img = np.asarray(raw_arr[frame_idx])
                    if apply_negative:
                        img = negative(img)
                    if ptv_params.get("splitter", False):
                        order = ptv_params.get("splitter_order") or list(
                            DEFAULT_SPLITTER_ORDER
                        )
                        images = [
                            view.copy()
                            for view in image_split(img, order=order)[:num_cams]
                        ]
                        return images
                    elif raw_arr.ndim == 4:  # (N, cams, h, w)
                        images = [
                            np.asarray(raw_arr[frame_idx, c]) for c in range(num_cams)
                        ]
                        return images
        except Exception as e:
            if os.environ.get("OPENPTV_STORAGE") == "zarr_only":
                raise RuntimeError(
                    f"Failed to read frame {frame} from res/images.zarr: {e}"
                ) from e
            print(
                f"Warning: Failed to read from res/images.zarr: {e}, falling back to disk files."
            )

    if ptv_params.get("splitter", False):
        imname = _frame_image_name(img_base_names[0], frame)
        if not imname.exists():
            raise FileNotFoundError(f"{imname} does not exist")
        img = _read_gray_uint8(imname)
        if apply_negative:
            img = negative(img)
        order = ptv_params.get("splitter_order") or list(DEFAULT_SPLITTER_ORDER)
        images = [view.copy() for view in image_split(img, order=order)[:num_cams]]
    else:
        images = []
        for i_cam in range(num_cams):
            imname = _frame_image_name(img_base_names[i_cam], frame)
            if not imname.exists():
                raise FileNotFoundError(f"{imname} does not exist")
            img = _read_gray_uint8(imname)
            if apply_negative:
                img = negative(img)
            images.append(img)

    if masking_params.get("mask_flag", False):
        for i_cam in range(num_cams):
            try:
                background_name = masking_params["mask_base_name"] % (i_cam + 1)
                background = imread(background_name)
                images[i_cam] = np.clip(images[i_cam] - background, 0, 255).astype(
                    np.uint8
                )
            except (ValueError, FileNotFoundError, TypeError):
                print("failed to read the mask")

    return images


def py_sequence_loop(exp) -> None:
    """Run a sequence of detection, stereo-correspondence, and determination.

    Splitter mode is handled transparently: when ``ptv.splitter`` is set,
    each frame is one multiplexed image that is split in memory (see
    read_frame_images) before detection.

    Args:
        exp: Either an Experiment object with pm attribute,
             or a MainGUI object with exp1.pm and cached parameter objects
    """

    # Handle both Experiment objects and MainGUI objects
    if hasattr(exp, "pm"):
        # Traditional experiment object
        pm = exp.pm
        num_cams = pm.num_cams
        cpar = exp.cpar
        spar = exp.spar
        vpar = exp.vpar
        tpar = exp.tpar
        cals = exp.cals
    elif hasattr(exp, "exp1") and hasattr(exp.exp1, "pm"):
        # MainGUI object - ensure parameter objects are initialized
        pm = exp.exp1.pm
        num_cams = exp.num_cams
        cpar = exp.cpar
        spar = exp.spar
        vpar = exp.vpar
        tpar = exp.tpar
        cals = exp.cals
    else:
        raise ValueError("Object must have either pm or exp1.pm attribute")

    existing_target = pm.get_parameter("pft_version").get("Existing_Target", False)

    # Check if we should run parallel preprocessing (Approach C)
    ptv_params_dict = (
        pm.parameters.get("ptv", {})
        if hasattr(pm, "parameters") and isinstance(pm.parameters, dict)
        else {}
    )
    if not isinstance(ptv_params_dict, dict):
        try:
            ptv_params_dict = pm.get_parameter("ptv")
        except ValueError:
            ptv_params_dict = {}

    parallel_preprocess = os.environ.get("OPENPTV_PARALLEL_PREPROCESS", "").lower() in (
        "true",
        "1",
    )
    if not parallel_preprocess and isinstance(ptv_params_dict, dict):
        parallel_preprocess = ptv_params_dict.get("parallel_preprocess", False)

    storage_mode = os.environ.get("OPENPTV_STORAGE", "zarr").lower()
    zarr_store_path = None
    if storage_mode in ("zarr", "zarr_only"):
        exp_path = getattr(exp, "exp_path", None)
        if not isinstance(exp_path, (str, Path)) or hasattr(
            exp_path, "_mock_return_value"
        ):
            exp_path = getattr(exp, "exp_dir", ".")
        if not isinstance(exp_path, (str, Path)) or hasattr(
            exp_path, "_mock_return_value"
        ):
            exp_path = "."
        zarr_store_path = str(Path(exp_path) / "res" / "run.zarr")

    if parallel_preprocess and not existing_target:
        preprocess_and_detect_all_parallel(exp, zarr_store_path=zarr_store_path)
        existing_target = True

    first_frame = spar.get_first()
    last_frame = spar.get_last()
    # Generate short_file_bases once per experiment
    img_base_names = [spar.get_img_base_name(i) for i in range(num_cams)]
    short_file_bases = exp.target_filenames
    _ensure_target_output_writable(short_file_bases)

    for frame in range(first_frame, last_frame + 1):
        detections = []
        corrected = []
        if not existing_target:
            frame_images = read_frame_images(pm, img_base_names, num_cams, frame)
        for i_cam in range(num_cams):
            if existing_target:
                if storage_mode in ("zarr", "zarr_only") and zarr_store_path:
                    from openptv2.storage import ZarrFrameStore

                    store = ZarrFrameStore(zarr_store_path, mode="r")
                    targs = store.read_targets(i_cam, frame)
                else:
                    targs = read_targets(short_file_bases[i_cam], frame)
            else:
                high_pass = simple_highpass(
                    frame_images[i_cam],
                    cpar,
                    ptv_params_dict.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE),
                )
                targs = target_recognition(high_pass, tpar, i_cam, cpar)

            if len(targs) > 0:
                targs.sort_y()

            detections.append(targs)
            matched_coords = MatchedCoords(targs, cpar, cals[i_cam])
            pos, _ = matched_coords.as_arrays()
            corrected.append(matched_coords)

        # After we finished all targs, we can move to correspondences
        sorted_pos, sorted_corresp, _ = correspondences(
            detections, corrected, cals, vpar, cpar
        )
        if storage_mode in ("zarr", "zarr_only") and zarr_store_path:
            from openptv2.storage import ZarrFrameStore

            store = ZarrFrameStore(zarr_store_path, mode="a")
            for i_cam in range(num_cams):
                store.write_targets(i_cam, frame, detections[i_cam])

        if storage_mode != "zarr_only":
            for i_cam in range(num_cams):
                write_targets(detections[i_cam], short_file_bases[i_cam], frame)

        print(
            "Frame "
            + str(frame)
            + " had "
            + repr([s.shape[1] for s in sorted_pos])
            + " correspondences."
        )
        sorted_pos = np.concatenate(sorted_pos, axis=1)
        sorted_corresp = np.concatenate(sorted_corresp, axis=1)
        flat = np.array(
            [
                corr.get_by_pnrs(corresp)
                for corr, corresp in zip(corrected, sorted_corresp)
            ]
        )
        pos, _ = point_positions(flat.transpose(1, 0, 2), exp.cpar, exp.cals, exp.vpar)
        if len(exp.cals) < 4:
            print_corresp = -1 * np.ones((4, sorted_corresp.shape[1]))
            print_corresp[: len(exp.cals), :] = sorted_corresp
        else:
            print_corresp = sorted_corresp

        if storage_mode in ("zarr", "zarr_only") and zarr_store_path:
            from openptv2.storage import ZarrFrameStore

            store = ZarrFrameStore(zarr_store_path, mode="a")
            store.write_correspondences(
                frame=frame, pos_3d=pos, cam_target_ids=print_corresp.T
            )

        if storage_mode != "zarr_only":
            output_path = _prepare_output_path(
                f"{_safe_decode(default_naming['corres'])}.{frame}"
            )
            try:
                with open(output_path, "w", encoding="utf8") as rt_is:
                    rt_is.write(f"{pos.shape[0]}\n")
                    for pix, pt in enumerate(pos):
                        pt_args = (pix + 1,) + tuple(pt) + tuple(print_corresp[:, pix])
                        rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
            except OSError as exc:
                _raise_output_write_error(output_path, exc)


def py_sequence_loop_python(exp) -> None:
    """Run detection, stereo-correspondence, and determination using the Python algorithms engine.

    Produces the same output files as py_sequence_loop (optv engine) so that
    results can be compared byte-for-byte.

    Args:
        exp: Same ProcessingExperiment object as py_sequence_loop expects.
    """
    from openptv2.algorithms.correspondences import (
        correspondences as alg_correspondences,
    )
    from openptv2.algorithms.orientation import point_positions as alg_point_positions
    from openptv2.algorithms.parameters import (
        ControlPar,
        TargetPar,
        VolumePar,
    )
    from openptv2.calibration import Calibration as AlgCalibration
    from openptv2.correspondences import MatchedCoords as AlgMatchedCoords
    from openptv2.segmentation import target_recognition as alg_target_recognition
    from openptv2.tracker import default_naming as alg_default_naming
    from openptv2.tracking_framebuf import Frame, read_targets

    # Handle both Experiment objects and MainGUI objects
    if hasattr(exp, "pm"):
        pm = exp.pm
        num_cams = pm.num_cams
        cpar = exp.cpar
        spar = exp.spar
        vpar = exp.vpar
        tpar = exp.tpar
        cals = exp.cals
    elif hasattr(exp, "exp1") and hasattr(exp.exp1, "pm"):
        pm = exp.exp1.pm
        num_cams = exp.num_cams
        cpar = exp.cpar
        spar = exp.spar
        vpar = exp.vpar
        tpar = exp.tpar
        cals = exp.cals
    else:
        raise ValueError("Object must have either pm or exp1.pm attribute")

    existing_target = pm.get_parameter("pft_version").get("Existing_Target", False)

    # Check if we should run parallel preprocessing (Approach C)
    ptv_params_dict = (
        pm.parameters.get("ptv", {})
        if hasattr(pm, "parameters") and isinstance(pm.parameters, dict)
        else {}
    )
    if not isinstance(ptv_params_dict, dict):
        try:
            ptv_params_dict = pm.get_parameter("ptv")
        except ValueError:
            ptv_params_dict = {}

    parallel_preprocess = os.environ.get("OPENPTV_PARALLEL_PREPROCESS", "").lower() in (
        "true",
        "1",
    )
    if not parallel_preprocess and isinstance(ptv_params_dict, dict):
        parallel_preprocess = ptv_params_dict.get("parallel_preprocess", False)

    if parallel_preprocess and not existing_target:
        preprocess_and_detect_all_parallel(exp)
        existing_target = True

    first_frame = spar.get_first()
    last_frame = spar.get_last()
    img_base_names = [spar.get_img_base_name(i) for i in range(num_cams)]
    short_file_bases = exp.target_filenames
    _ensure_target_output_writable(short_file_bases)

    # Convert optv ControlParams to algorithms ControlPar
    cpar_py = ControlPar(num_cams=num_cams)
    imx, imy = cpar.get_image_size()
    cpar_py.imx = imx
    cpar_py.imy = imy
    pix_x, pix_y = cpar.get_pixel_size()
    cpar_py.pix_x = pix_x
    cpar_py.pix_y = pix_y
    if hasattr(cpar, "get_hp_flag"):
        cpar_py.hp_flag = cpar.get_hp_flag()
    if hasattr(cpar, "get_allCam_flag"):
        cpar_py.all_cam_flag = cpar.get_allCam_flag()
        cpar_py.allCam_flag = cpar.get_allCam_flag()
    if hasattr(cpar, "get_tiff_flag"):
        cpar_py.tiff_flag = cpar.get_tiff_flag()
    if hasattr(cpar, "get_chfield"):
        cpar_py.chfield = cpar.get_chfield()
    # Copy multimedia params
    if hasattr(cpar, "get_multimedia_params"):
        optv_mm = cpar.get_multimedia_params()
        if hasattr(optv_mm, "get_n1"):
            cpar_py.mm.n1 = optv_mm.get_n1()
        if hasattr(optv_mm, "get_n3"):
            cpar_py.mm.n3 = optv_mm.get_n3()
        if hasattr(optv_mm, "get_nlay"):
            nlay = optv_mm.get_nlay()
            cpar_py.mm.nlay = nlay
        if hasattr(optv_mm, "get_d"):
            cpar_py.mm.d = list(optv_mm.get_d())
        if hasattr(optv_mm, "get_n2"):
            cpar_py.mm.n2 = list(optv_mm.get_n2())

    # Convert optv VolumeParams to algorithms VolumePar
    vpar_py = VolumePar()
    if hasattr(vpar, "get_X_lay"):
        try:
            val = list(vpar.get_X_lay())
            vpar_py.x_lay = val
            vpar_py.X_lay = np.array(val, dtype=np.float64)
        except Exception:
            pass
    if hasattr(vpar, "get_Zmin_lay"):
        try:
            val = list(vpar.get_Zmin_lay())
            vpar_py.z_min_lay = val
            vpar_py.Zmin_lay = np.array(val, dtype=np.float64)
        except Exception:
            pass
    if hasattr(vpar, "get_Zmax_lay"):
        try:
            val = list(vpar.get_Zmax_lay())
            vpar_py.z_max_lay = val
            vpar_py.Zmax_lay = np.array(val, dtype=np.float64)
        except Exception:
            pass
    for attr in ("cn", "cnx", "cny", "csumg", "eps0", "corrmin"):
        getter = f"get_{attr}"
        if hasattr(vpar, getter):
            try:
                setattr(vpar_py, attr, getattr(vpar, getter)())
            except Exception:
                pass

    # Convert optv TargetParams to algorithms TargetPar
    tpar_py = TargetPar()
    if hasattr(tpar, "get_grey_thresholds"):
        try:
            tpar_py.gvthresh = list(tpar.get_grey_thresholds())
        except Exception:
            pass
    if hasattr(tpar, "get_max_discontinuity"):
        try:
            tpar_py.discont = tpar.get_max_discontinuity()
        except Exception:
            pass
    if hasattr(tpar, "get_pixel_count_bounds"):
        try:
            lo, hi = tpar.get_pixel_count_bounds()
            tpar_py.nnmin = lo
            tpar_py.nnmax = hi
        except Exception:
            pass
    if hasattr(tpar, "get_xsize_bounds"):
        try:
            lo, hi = tpar.get_xsize_bounds()
            tpar_py.nxmin = lo
            tpar_py.nxmax = hi
        except Exception:
            pass
    if hasattr(tpar, "get_ysize_bounds"):
        try:
            lo, hi = tpar.get_ysize_bounds()
            tpar_py.nymin = lo
            tpar_py.nymax = hi
        except Exception:
            pass
    if hasattr(tpar, "get_min_sum_grey"):
        try:
            tpar_py.sumg_min = tpar.get_min_sum_grey()
        except Exception:
            pass
    if hasattr(tpar, "get_cross_size"):
        try:
            tpar_py.cr_sz = tpar.get_cross_size()
        except Exception:
            pass
    if hasattr(tpar, "get_grey"):
        try:
            tpar_py.set_grey(tpar.get_grey())
        except Exception:
            pass

    # Convert optv Calibrations to algorithms Calibrations
    cals_py = []
    for cal in cals:
        py_cal = AlgCalibration()
        if hasattr(cal, "get_pos"):
            py_cal.set_pos(cal.get_pos())
        if hasattr(cal, "get_angles"):
            py_cal.set_angles(cal.get_angles())
        if hasattr(cal, "get_primary_point"):
            pp = cal.get_primary_point()
            py_cal.set_primary_point(pp)
        if hasattr(cal, "get_radial_distortion"):
            rd = cal.get_radial_distortion()
            py_cal.set_radial_distortion(rd)
        if hasattr(cal, "get_decentering"):
            dc = cal.get_decentering()
            py_cal.set_decentering(dc)
        if hasattr(cal, "get_affine"):
            at = cal.get_affine()
            py_cal.set_affine_trans(at)
        if hasattr(cal, "get_glass_vec"):
            gv = cal.get_glass_vec()
            py_cal.set_glass_vec(gv)
        cals_py.append(py_cal)

    for frame in range(first_frame, last_frame + 1):
        detections = []
        corrected = []
        for i_cam in range(num_cams):
            if existing_target:
                targs = read_targets(short_file_bases[i_cam], frame)
            else:
                imname = Path(img_base_names[i_cam] % frame)
                if not imname.exists():
                    raise FileNotFoundError(f"{imname} does not exist")
                else:
                    img = imread(imname)
                    if img.ndim > 2:
                        img = rgb2gray(img[:, :, :3])
                    if img.dtype != np.uint8:
                        img = img_as_ubyte(img)
                if pm.get_parameter("ptv").get("negative", False):
                    print("Negative image")
                    img = negative(img)
                masking_params = pm.get_parameter("masking")
                if masking_params and masking_params.get("mask_flag", False):
                    try:
                        background_name = masking_params["mask_base_name"] % (i_cam + 1)
                        background = imread(background_name)
                        img = np.clip(img - background, 0, 255).astype(np.uint8)
                    except (ValueError, FileNotFoundError):
                        print("failed to read the mask")
                high_pass = simple_highpass(
                    img,
                    cpar,
                    ptv_params_dict.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE),
                )
                targs = alg_target_recognition(high_pass, tpar, i_cam, cpar)

            if len(targs) > 0:
                if hasattr(targs, "sort_y"):
                    targs.sort_y()
                else:
                    targs.sort(key=lambda t: t.y)

            detections.append(targs)
            matched_coords = AlgMatchedCoords(targs, cpar, cals_py[i_cam])
            pos, _ = matched_coords.as_arrays()
            corrected.append(matched_coords)

        # Build a Frame for algorithms correspondences
        frm = Frame(num_cams=num_cams, max_targets=10000)
        for i_cam in range(num_cams):
            n = len(detections[i_cam])
            frm.num_targets[i_cam] = n
            for tnum in range(n):
                t = detections[i_cam][tnum]
                t_native = t._target if hasattr(t, "_target") else t
                frm.targets[i_cam][tnum].pnr = t_native.pnr
                frm.targets[i_cam][tnum].tnr = -1
                frm.targets[i_cam][tnum].x = t_native.x
                frm.targets[i_cam][tnum].y = t_native.y
                frm.targets[i_cam][tnum].n = t_native.n
                frm.targets[i_cam][tnum].nx = t_native.nx
                frm.targets[i_cam][tnum].ny = t_native.ny
                frm.targets[i_cam][tnum].sumg = t_native.sumg

                # Update SoA
                frm.targ_x[i_cam][tnum] = t_native.x
                frm.targ_y[i_cam][tnum] = t_native.y
                frm.targ_tnr[i_cam][tnum] = -1

        match_counts = [0] * 4  # [4-cam, 3-cam, 2-cam, total]
        con, counts = alg_correspondences(
            frm,
            [mc._corrected for mc in corrected],
            vpar_py,
            cpar_py,
            [c._cal if hasattr(c, "_cal") else c for c in cals_py],
        )
        match_counts[0] = counts[0]
        match_counts[1] = counts[1]
        match_counts[2] = counts[2]
        match_counts[3] = counts[3]
        total = match_counts[3] if len(match_counts) > 3 else 0
        if total > 0:
            valid = con[:total]
            # Sort by correlation descending (highest quality first)
            valid.sort(key=lambda x: x.corr, reverse=True)

            # Map the x-sorted indices row.p to original target indices (pnrs)
            corresp_list = []
            for row in valid:
                mapped_p = []
                for cam in range(num_cams):
                    idx = row.p[cam]
                    if idx >= 0:
                        mapped_p.append(corrected[cam]._corrected[idx].pnr)
                    else:
                        mapped_p.append(-1)
                corresp_list.append(mapped_p)

            corresp = np.array(corresp_list).T  # (num_cams, N)
            sorted_corresp = [corresp]
            sorted_pos = [np.zeros((3, corresp.shape[1]))]
        else:
            sorted_corresp = [np.zeros((num_cams, 0), dtype=int)]
            sorted_pos = [np.zeros((3, 0))]

        # Write targets
        for i_cam in range(num_cams):
            targs = detections[i_cam]
            output_path = _prepare_output_path(
                f"{short_file_bases[i_cam]}.{frame:04d}_targets"
            )
            try:
                with open(output_path, "w", encoding="utf8") as f:
                    f.write(f"{len(targs)}\n")
                    for t in targs:
                        t_native = t._target if hasattr(t, "_target") else t
                        pnr = t_native.pnr
                        x = t_native.x
                        y = t_native.y
                        n = t_native.n
                        nx = t_native.nx
                        ny = t_native.ny
                        sumg = t_native.sumg
                        tnr = t_native.tnr
                        f.write(
                            f"{pnr:4d} {x:9.4f} {y:9.4f} {n:5d} {nx:5d} "
                            f"{ny:5d} {sumg:5d} {tnr:5d}\n"
                        )
            except OSError as exc:
                _raise_output_write_error(output_path, exc)

        concatenated_corresp = np.concatenate(sorted_corresp, axis=1)
        np.concatenate(sorted_pos, axis=1)

        # Look up corrected coords for correspondence matches and compute 3D positions
        if concatenated_corresp.shape[1] > 0:
            flat = np.array(
                [
                    corr.get_by_pnrs(corresp)
                    for corr, corresp in zip(corrected, concatenated_corresp)
                ]
            )
            pos, _ = alg_point_positions(
                flat.transpose(1, 0, 2), cpar_py, [c._cal for c in cals_py], vpar_py
            )
        else:
            pos = np.zeros((0, 3))

        if num_cams < 4:
            print_corresp = -1 * np.ones((4, concatenated_corresp.shape[1]), dtype=int)
            print_corresp[:num_cams, :] = concatenated_corresp
        else:
            print_corresp = concatenated_corresp

        corres_path = alg_default_naming["corres"]
        output_path = _prepare_output_path(f"{corres_path}.{frame}")
        try:
            with open(output_path, "w", encoding="utf8") as rt_is:
                rt_is.write(f"{pos.shape[0]}\n")
                for pix, pt in enumerate(pos):
                    pt_args = (pix + 1,) + tuple(pt) + tuple(print_corresp[:, pix])
                    rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
        except OSError as exc:
            _raise_output_write_error(output_path, exc)

        print(
            "Frame "
            + str(frame)
            + " had "
            + repr([s.shape[1] for s in sorted_pos])
            + " correspondences (python engine)."
        )


def py_trackcorr_init(exp):
    """Reads all the necessary stuff into Tracker"""

    # Generate short_file_bases once per experiment
    # img_base_names = [exp.spar.get_img_base_name(i) for i in range(exp.cpar.get_num_cams())]
    # exp.short_file_bases = exp.target_filenames
    target_filenames = getattr(exp, "target_filenames", None)
    if target_filenames is None:
        target_filenames = []

    try:
        target_filenames = list(target_filenames)
    except TypeError:
        target_filenames = []

    if not target_filenames:
        img_base_names = [
            exp.spar.get_img_base_name(i) for i in range(exp.cpar.get_num_cams())
        ]
        target_filenames = generate_short_file_bases(img_base_names)
        exp.target_filenames = target_filenames

    for cam_id, short_name in enumerate(target_filenames):
        # print(f"Setting tracker image base name for cam {cam_id+1}: {Path(short_name).resolve()}")
        exp.spar.set_img_base_name(cam_id, str(Path(short_name).resolve()) + ".")

    # print("exp.spar.img_base_names:", [exp.spar.get_img_base_name(i) for i in range(exp.cpar.get_num_cams())])

    # print(
    #     exp.track_par.get_dvxmin(), exp.track_par.get_dvxmax(),
    #     exp.track_par.get_dvymin(), exp.track_par.get_dvymax(),
    #     exp.track_par.get_dvzmin(), exp.track_par.get_dvzmax(),
    #     exp.track_par.get_dangle(), exp.track_par.get_dacc(),
    #     exp.track_par.get_add()
    # )

    print("Initializing Tracker with parameters:")

    # Get parameters from experiment object
    cpar = exp.cpar
    vpar = exp.vpar
    track_par = exp.track_par
    spar = exp.spar
    cals = exp.cals

    print("[ENGINE] Using single Cython 3 tracker runtime")

    tracker = Tracker(cpar, vpar, track_par, spar, cals, default_naming)
    return tracker

    return tracker


# ------- Utilities ----------#


def py_get_pix(
    x: List[List[int]], y: List[List[int]]
) -> Tuple[List[List[int]], List[List[int]]]:
    """Get target positions (stub function)."""
    return x, y


def py_calibration(selection, exp):
    """Calibration

    Args:
        selection: Calibration selection type
        exp: Either an Experiment object with pm attribute,
             or a MainGUI object with exp1.pm and cached parameter objects
    """
    if selection == 1:
        pass

    if selection == 2:
        pass

    if selection == 9:
        pass

    if selection == 12:
        """ Calibration with dumbbell ."""
        return ptv_calibration.calib_dumbbell(exp)

    if selection == 10:
        """ Calibration with particles ."""

        return ptv_calibration.calib_particles(exp)


def write_targets(targets: TargetArray, short_file_base: str, frame: int) -> bool:
    """Write targets to a file."""
    output_path = _prepare_output_path(f"{short_file_base}.{frame:04d}_targets")
    num_targets = len(targets)
    success = False
    if num_targets == 0:
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("0\n")
        except OSError as exc:
            _raise_output_write_error(output_path, exc)
        return True  # No targets to write, but file created successfully

    try:
        target_arr = np.array(
            [
                ([t.pnr(), *t.pos(), *t.count_pixels(), t.sum_grey_value(), t.tnr()])
                for t in targets
            ]
        )
        np.savetxt(
            output_path,
            target_arr,
            fmt="%4d %9.4f %9.4f %5d %5d %5d %5d %5d",
            header=f"{num_targets}",
            comments="",
        )
        success = True
    except OSError as exc:
        _raise_output_write_error(output_path, exc)
    return success


def read_targets(short_file_base: str, frame: int) -> TargetArray:
    """Read targets from a file."""
    filename = f"{short_file_base}.{frame:04d}_targets"
    print(f" Reading targets from: filename: {filename}")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Targets file does not exist: {filename}")

    try:
        with open(filename, "r", encoding="utf-8") as file:
            num_targets = int(file.readline().strip())
            targs = TargetArray(num_targets)

            for tix in range(num_targets):
                line = file.readline().strip().split()

                if len(line) != 8:
                    raise ValueError(f"Bad format for file: {filename}")

                targ = targs[tix]
                targ.set_pnr(int(line[0]))
                targ.set_pos([float(line[1]), float(line[2])])
                targ.set_pixel_counts(int(line[3]), int(line[4]), int(line[5]))
                targ.set_sum_grey_value(int(line[6]))
                targ.set_tnr(int(line[7]))

    except IOError as err:
        print(f"Can't open targets file: {filename}")
        raise err

    return targs


def extract_cam_ids(file_bases: list[str]) -> list[int]:
    """
    Given a list of file base strings, extract the camera identification number from each.
    The camera id is the digit or number that is the main difference between the names,
    typically close to 'cam', 'c', 'img', etc.
    Returns a list of integers, one for each file base.
    """
    # Try to find all numbers in each string, and their context
    if not file_bases:
        raise ValueError("file_bases list is empty")

    # If input is a string, convert to a list
    if isinstance(file_bases, str):
        file_bases = [file_bases]

    # Remove frame number patterns like %d, %04d, etc.
    clean_bases = [re.sub(r"%0?\d*d", "", s) for s in file_bases]
    file_bases = clean_bases

    # Helper to extract all (number, context) pairs from a string
    def extract_number_context(s):
        # Find all numbers with up to 4 chars before and after
        matches = []
        for m in re.finditer(r"([a-zA-Z]{0,4})?(\d+)", s):
            prefix = m.group(1) or ""
            number = m.group(2)
            start = m.start(2)
            matches.append((number, prefix.lower(), start))
        return matches

    # Build a list of all numbers and their context for each string
    all_matches = [extract_number_context(s) for s in file_bases]

    # Transpose to group by position in the list
    # Find which number position varies the most across the list
    # (i.e., the one that is different between the names)
    candidate_indices = []
    maxlen = max(len(m) for m in all_matches) if all_matches else 0
    for idx in range(maxlen):
        nums = []
        for m in all_matches:
            if len(m) > idx:
                nums.append(m[idx][0])
            else:
                nums.append(None)
        # Count unique numbers (ignoring None)
        unique = set(n for n in nums if n is not None)
        candidate_indices.append((idx, len(unique)))

    # Pick the index with the most unique numbers (should be the cam id)
    candidate_indices.sort(key=lambda x: -x[1])
    if not candidate_indices or candidate_indices[0][1] <= 1:
        # fallback: just use the last number in each string
        fallback_ids = []
        for idx, s in enumerate(file_bases):
            found = re.findall(r"(\d+)", s)
            if found:
                fallback_ids.append(int(found[-1]))
            else:
                # fallback to default SHORT_BASE+idx+1
                fallback_ids.append(None)
        # If any fallback_ids are None, use default SHORT_BASE+idx+1
        if any(x is None for x in fallback_ids):
            fallback_ids = list(range(1, len(file_bases) + 1))
            print("fall back to default list", fallback_ids)

        return fallback_ids

    cam_idx = candidate_indices[0][0]

    # Now, for each string, get the number at cam_idx
    cam_ids = []
    for idx, m in enumerate(all_matches):
        if len(m) > cam_idx:
            cam_ids.append(int(m[cam_idx][0]))
        else:
            # fallback: last number or default SHORT_BASE+idx+1
            nums = re.findall(r"(\d+)", "".join([x[0] for x in m]))
            if nums:
                cam_ids.append(int(nums[-1]))
            else:
                cam_ids.append(f"{SHORT_BASE}{idx + 1}")
    # If any cam_ids are not int, fallback to default SHORT_BASE+idx+1
    if any(not isinstance(x, int) for x in cam_ids):
        cam_ids = list(range(1, len(file_bases) + 1))
        print("Fallback to default list {cam_ids}")

    return cam_ids


def generate_short_file_bases(img_base_names: List[str]) -> List[str]:
    """
    Given a list of image base names (full paths) for all cameras, generate a list of short_file_base strings for targets.
    The short file base will be in the same directory as the original, but with the filename replaced by SHORT_BASE + index.
    """
    ids = extract_cam_ids(img_base_names)
    short_bases = []
    for idx, full_path in enumerate(img_base_names):
        parent = Path(full_path).parent
        short_name = f"{SHORT_BASE}{ids[idx]}"
        short_bases.append(str(parent / short_name))
    return short_bases


def read_rt_is_file(filename) -> List[List[float]]:
    """Read data from an rt_is file or Zarr store and return the parsed values."""
    if not Path(filename).exists():
        p = Path(filename)
        frame_match = re.search(r"\.(\d+)$", p.name)
        if frame_match:
            frame = int(frame_match.group(1))
            zarr_candidates = [
                p.parent / "run.zarr",
                p.parent / "targets.zarr",
                p.parent.parent / "res" / "run.zarr",
            ]
            for zpath in zarr_candidates:
                if zpath.exists():
                    from openptv2.storage import ZarrFrameStore

                    try:
                        store = ZarrFrameStore(zpath, mode="r")
                        pos_3d, cam_ids = store.read_correspondences(frame)
                        data = []
                        for pt, c in zip(pos_3d, cam_ids):
                            data.append(
                                [
                                    float(pt[0]),
                                    float(pt[1]),
                                    float(pt[2]),
                                    int(c[0]),
                                    int(c[1]),
                                    int(c[2]),
                                    int(c[3]),
                                ]
                            )
                        return data
                    except Exception:
                        pass

    try:
        with open(filename, "r", encoding="utf-8") as file:
            num_rows = int(file.readline().strip())
            if num_rows == 0:
                raise ValueError("Failed to read the number of rows")

            data = []
            for _ in range(num_rows):
                line = file.readline().strip()
                if not line:
                    break

                values = line.split()
                if len(values) != 8:
                    raise ValueError("Incorrect number of values in line")

                x = float(values[1])
                y = float(values[2])
                z = float(values[3])
                p1 = int(values[4])
                p2 = int(values[5])
                p3 = int(values[6])
                p4 = int(values[7])

                data.append([x, y, z, p1, p2, p3, p4])

            return data

    except IOError as e:
        print(f"Can't open ascii file: {filename}")
        raise e

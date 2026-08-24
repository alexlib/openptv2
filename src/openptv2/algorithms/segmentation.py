# ruff: noqa: E741
"""Particle detection via thresholding and peak fitting.

Translation of lib/src/segmentation.c and lib/include/segmentation.h.

Provides:
- targ_rec: thresholding and center-of-gravity with peak fitting (delegates to
  track_kernels._targ_rec_fast for the hot BFS — that function compiles to
  near-pure C with typed memoryviews).
- peak_fit: two-pass component labeling with reunification (alternative
  implementation, used only in tests).
"""

from dataclasses import dataclass

import cython
import numpy as np

from .track_kernels import targ_rec_fast as _targ_rec_fast
from .tracking_frame_buf import Target

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt

# Constant for no correspondence assigned
CORRES_NONE = -1


@cython.cclass
@dataclass
class Peak:
    """Detected peak for connectivity analysis."""

    pos: cython.int = 0
    status: cython.int = 0
    xmin: cython.int = 0
    xmax: cython.int = 0
    ymin: cython.int = 0
    ymax: cython.int = 0
    n: cython.int = 0
    sumg: cython.int = 0
    x: cython.double = 0.0
    y: cython.double = 0.0
    unr: cython.int = 0
    touch: list = cython.declare(list, visibility="public")
    n_touch: cython.int = 0

    def __post_init__(self):
        if self.touch is None:
            self.touch = [0, 0, 0, 0]


@cython.ccall
def _is_local_maximum(
    img: cython.uchar[:, ::1], i: cython.int, j: cython.int
) -> cython.bint:
    """Check if pixel at (i, j) is an 8-neighbor local maximum.

    Pure C pointer arithmetic when compiled — no Python overhead.

    peak_fit's default xmin/ymin=1 keeps callers away from row/col 0, but an
    explicit xmin=0/ymin=0 (or xmax/ymax at the far edge) reaches this with
    i or j at the image boundary, where an out-of-range neighbour used to be
    read unchecked (wraparound=False, so img[i, -1] is a literal invalid
    index, not "last column"). A neighbour that falls off the image can't
    exceed gv, so it's simply skipped -- the standard convention for
    boundary pixels in local-maximum detection.
    """
    imy: cython.int = img.shape[0]
    imx: cython.int = img.shape[1]
    gv = img[i, j]
    if j > 0 and gv < img[i, j - 1]:
        return False
    if j + 1 < imx and gv < img[i, j + 1]:
        return False
    if i > 0 and gv < img[i - 1, j]:
        return False
    if i + 1 < imy and gv < img[i + 1, j]:
        return False
    if i > 0 and j > 0 and gv < img[i - 1, j - 1]:
        return False
    if i + 1 < imy and j > 0 and gv < img[i + 1, j - 1]:
        return False
    if i > 0 and j + 1 < imx and gv < img[i - 1, j + 1]:
        return False
    if i + 1 < imy and j + 1 < imx and gv < img[i + 1, j + 1]:
        return False
    return True


@cython.ccall
def check_touch(tpeak: Peak, p1: cython.int, p2: cython.int):
    """Check whether p1, p2 are already marked as touching and mark them otherwise."""
    if p2 == 0 or p2 == p1:
        return

    m: cython.int
    for m in range(tpeak.n_touch):
        if tpeak.touch[m] == p2:
            return

    tpeak.touch[tpeak.n_touch] = p2
    tpeak.n_touch += 1
    if tpeak.n_touch > 3:
        tpeak.n_touch = 3


@cython.ccall
def targ_rec(
    img: cython.uchar[:, ::1],
    gvthres: cython.int,
    discont: cython.int,
    nnmin: cython.int,
    nnmax: cython.int,
    nxmin: cython.int,
    nxmax: cython.int,
    nymin: cython.int,
    nymax: cython.int,
    sumg_min: cython.int,
    xmin: cython.int = 1,
    xmax: cython.int = -1,
    ymin: cython.int = 1,
    ymax: cython.int = -1,
) -> list:
    """Thresholding and center-of-gravity with peak fitting (C targ_rec translation).

    Delegates to the compiled BFS in track_kernels._targ_rec_fast.

    Args:
        img: input image (2D uint8, shape (imy, imx)).
        gvthres: grey value threshold for binarization.
        discont: maximum discontinuity for peak growth.
        nnmin, nnmax: min/max number of pixels per target.
        nxmin, nxmax: min/max extent in x.
        nymin, nymax: min/max extent in y.
        sumg_min: minimum sum of grey values.
        xmin, xmax, ymin, ymax: search area (defaults to image bounds).

    Returns:
        List of detected targets.
    """
    imy: cython.int = img.shape[0]
    imx: cython.int = img.shape[1]
    if xmax < 0:
        xmax = imx - 1
    if ymax < 0:
        ymax = imy - 1

    xmin = max(xmin, 1)
    ymin = max(ymin, 1)
    xmax = min(xmax, imx - 1)
    ymax = min(ymax, imy - 1)

    img_u8 = np.ascontiguousarray(img, dtype=np.uint8)
    img0 = img_u8.copy()
    max_targets = (xmax - xmin) * (ymax - ymin)

    n_found, ox, oy, on, onx, ony, osumg = _targ_rec_fast(
        img_u8,
        img0,
        gvthres,
        discont,
        nnmin,
        nnmax,
        nxmin,
        nxmax,
        nymin,
        nymax,
        sumg_min,
        xmin,
        ymin,
        xmax,
        ymax,
        max_targets,
    )
    if n_found == 0:
        return [Target(pnr=1, x=1, y=1, n=1, nx=1, ny=1, sumg=1, tnr=CORRES_NONE)]
    return [
        Target(
            pnr=k,
            x=float(ox[k]),
            y=float(oy[k]),
            n=int(on[k]),
            nx=int(onx[k]),
            ny=int(ony[k]),
            sumg=int(osumg[k]),
            tnr=CORRES_NONE,
        )
        for k in range(n_found)
    ]


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def peak_fit(
    img: cython.uchar[:, ::1],
    gvthres: cython.int,
    discont: cython.int,
    nnmin: cython.int,
    nnmax: cython.int,
    nxmin: cython.int,
    nxmax: cython.int,
    nymin: cython.int,
    nymax: cython.int,
    sumg_min: cython.int,
    xmin: cython.int = 1,
    xmax: cython.int = -1,
    ymin: cython.int = 1,
    ymax: cython.int = -1,
) -> list:
    """Two-pass component labeling with peak fitting and reunification."""
    imy: cython.int = img.shape[0]
    imx: cython.int = img.shape[1]
    if xmax < 0:
        xmax = imx
    if ymax < 0:
        ymax = imy

    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    gv: cython.int

    # Pre-allocated typed arrays for BFS queue (maximally sized)
    _qx = np.empty(imy * imx, dtype=np.int32)
    _qy = np.empty(imy * imx, dtype=np.int32)
    qx: cython.int[:] = _qx
    qy: cython.int[:] = _qy

    # Static direction lookup arrays — compile to C constant arrays
    dx4 = [-1, 1, 0, 0]
    dy4 = [0, 0, -1, 1]
    di8 = [-1, -1, -1, 0, 0, 1, 1, 1]
    dj8 = [-1, 0, 1, -1, 1, -1, 0, 1]

    qhead: cython.int
    qtail: cython.int
    wx: cython.int
    wy: cython.int
    nx_pos: cython.int
    ny_pos: cython.int
    gvref: cython.int
    neighbor_gv: cython.int
    d: cython.int

    # Label image
    label_img: cython.int[:, :] = np.zeros((imy, imx), dtype=np.int32)
    peaks: list = []

    # ---- Pass 1: Connectivity analysis with peak search ----
    for i in range(ymin, ymax - 1):
        for j in range(xmin, xmax):
            n = i * imx + j
            gv = int(img[i, j])

            if gv <= gvthres:
                continue
            if label_img[i, j] != 0:
                continue

            # Check local maximum
            if not _is_local_maximum(img, i, j):
                continue

            # New peak
            n_peaks = len(peaks) + 1
            label_img[i, j] = n_peaks

            peak = Peak(pos=n, status=1, xmin=j, xmax=j, ymin=i, ymax=i)
            peaks.append(peak)

            # BFS region growing — typed array queue
            qhead = 0
            qtail = 0
            qx[qtail] = j
            qy[qtail] = i
            qtail += 1
            label_img[i, j] = n_peaks

            while qhead < qtail:
                wx = qx[qhead]
                wy = qy[qhead]
                qhead += 1
                gvref = int(img[wy, wx])

                for d in range(4):
                    nx_pos = wx + dx4[d]
                    ny_pos = wy + dy4[d]
                    if not (0 <= nx_pos < imx and 0 <= ny_pos < imy):
                        continue
                    if label_img[ny_pos, nx_pos] != 0:
                        continue

                    neighbor_gv = int(img[ny_pos, nx_pos])

                    if (
                        neighbor_gv > gvthres
                        and xmin <= nx_pos < xmax
                        and ymin <= ny_pos < ymax - 1
                        and neighbor_gv <= gvref + discont
                        and gvref + discont >= img[ny_pos - 1, nx_pos]
                        and gvref + discont >= img[ny_pos + 1, nx_pos]
                        and gvref + discont >= img[ny_pos, nx_pos - 1]
                        and gvref + discont >= img[ny_pos, nx_pos + 1]
                    ):
                        label_img[ny_pos, nx_pos] = n_peaks
                        qx[qtail] = nx_pos
                        qy[qtail] = ny_pos
                        qtail += 1

    # ---- Pass 2: Collect data and detect touches ----
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            n = i * imx + j
            label = label_img[i, j]

            if label <= 0:
                continue

            pnr = label - 1
            peak = peaks[pnr]
            gv = img[i, j]

            peak.n += 1
            peak.sumg += int(gv)
            peak.x += float(j) * gv
            peak.y += float(i) * gv

            if j < peak.xmin:
                peak.xmin = j
            if j > peak.xmax:
                peak.xmax = j
            if i < peak.ymin:
                peak.ymin = i
            if i > peak.ymax:
                peak.ymax = i

            # Check 8-neighbors for touches
            for d in range(8):
                ni: cython.int = i + di8[d]
                nj: cython.int = j + dj8[d]
                if 0 <= ni < imy and 0 <= nj < imx:
                    neighbor_label = label_img[ni, nj]
                    check_touch(peak, label, neighbor_label)

    # ---- Pass 3: Reunification test ----
    for peak_i in peaks:
        if peak_i.n_touch == 0 or peak_i.unr != 0:
            continue

        x1 = peak_i.x / peak_i.sumg
        y1 = peak_i.y / peak_i.sumg
        pi: cython.int = peak_i.pos // imx
        pj: cython.int = peak_i.pos % imx
        gv1 = img[pi, pj]

        for j_idx in peak_i.touch:
            p2 = j_idx - 1
            if p2 < 0 or p2 >= len(peaks) or peaks[p2].unr != 0:
                continue

            peak_j = peaks[p2]
            x2 = peak_j.x / peak_j.sumg
            y2 = peak_j.y / peak_j.sumg
            pj2: cython.int = peak_j.pos // imx
            pj3: cython.int = peak_j.pos % imx
            gv2 = img[pj2, pj3]

            s12 = c_sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Profile criterion
            unify: cython.bint = s12 < 2.0
            if not unify:
                unify = True
                l: cython.int
                for l in range(1, int(s12)):
                    intx1: cython.int = int(x1 + l * (x2 - x1) / s12)
                    inty1: cython.int = int(y1 + l * (y2 - y1) / s12)

                    if 0 <= inty1 < imy and 0 <= intx1 < imx:
                        gv = img[inty1, intx1] + discont
                        if gv < gv1 + l * (gv2 - gv1) / s12 or gv < gv1 or gv < gv2:
                            unify = False
                            break
                    else:
                        unify = False
                        break

            if not unify:
                continue

            # Unify targets
            peak_i.unr = p2 + 1  # 1-indexed
            peak_j.x += peak_i.x
            peak_j.y += peak_i.y
            peak_j.sumg += peak_i.sumg
            peak_j.n += peak_i.n
            if peak_i.xmin < peak_j.xmin:
                peak_j.xmin = peak_i.xmin
            if peak_i.ymin < peak_j.ymin:
                peak_j.ymin = peak_i.ymin
            if peak_i.xmax > peak_j.xmax:
                peak_j.xmax = peak_i.xmax
            if peak_i.ymax > peak_j.ymax:
                peak_j.ymax = peak_i.ymax

    # ---- Pass 4: Output targets ----
    targets = []
    for peak in peaks:
        if peak.unr != 0:
            continue

        width = xmax - xmin
        if width > 32:
            if peak.xmin == xmin or peak.ymin == ymin:
                continue
            if peak.xmax == xmax - 1 or peak.ymax == ymax - 1:
                continue

        nx = peak.xmax - peak.xmin + 1
        ny = peak.ymax - peak.ymin + 1

        if (
            peak.sumg > sumg_min
            and nxmin <= nx <= nxmax
            and nymin <= ny <= nymax
            and nnmin <= peak.n <= nnmax
        ):
            x_final = 0.5 + peak.x / peak.sumg
            y_final = 0.5 + peak.y / peak.sumg

            targets.append(
                Target(
                    pnr=len(targets),
                    x=x_final,
                    y=y_final,
                    n=peak.n,
                    nx=nx,
                    ny=ny,
                    sumg=peak.sumg,
                    tnr=CORRES_NONE,
                )
            )

    return targets


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled


def _load_image_array(img_source) -> np.ndarray:
    """Load and normalize an image source into a C-contiguous uint8 2D array."""
    from pathlib import Path

    if isinstance(img_source, (str, Path)):
        p = Path(img_source)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        from skimage.io import imread
        from skimage.color import rgb2gray
        from skimage.util import img_as_ubyte

        arr = imread(p)
        if arr.ndim > 2:
            arr = rgb2gray(arr[:, :, :3])
        if arr.dtype != np.uint8:
            arr = img_as_ubyte(arr)
        return np.ascontiguousarray(arr, dtype=np.uint8)

    elif isinstance(img_source, np.ndarray):
        arr = img_source
        if arr.ndim > 2:
            from skimage.color import rgb2gray
            arr = rgb2gray(arr[:, :, :3])
        if arr.dtype != np.uint8:
            from skimage.util import img_as_ubyte
            arr = img_as_ubyte(arr)
        return np.ascontiguousarray(arr, dtype=np.uint8)

    elif isinstance(img_source, dict) and "shm_name" in img_source:
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=img_source["shm_name"])
        try:
            shape = img_source["shape"]
            dtype = np.dtype(img_source.get("dtype", "uint8"))
            offset = img_source.get("offset", 0)
            raw = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=offset)
            return raw.copy()
        finally:
            shm.close()

    else:
        raise TypeError(f"Unsupported image source type: {type(img_source)}")


def _detect_single_worker(task: tuple) -> dict:
    """Worker task executed in separate processes for batch target detection."""
    from pathlib import Path

    img_source, params_dict, write_path, cam_idx, frame_idx = task
    img = _load_image_array(img_source)
    imy, imx = img.shape

    xmin = max(int(params_dict.get("xmin", 1)), 1)
    ymin = max(int(params_dict.get("ymin", 1)), 1)
    xmax = int(params_dict.get("xmax", -1))
    ymax = int(params_dict.get("ymax", -1))
    if xmax < 0:
        xmax = imx - 1
    if ymax < 0:
        ymax = imy - 1
    xmax = min(xmax, imx - 1)
    ymax = min(ymax, imy - 1)

    gvthres = int(params_dict.get("gvthres", 10))
    discont = int(params_dict.get("discont", 100))
    nnmin = int(params_dict.get("nnmin", 1))
    nnmax = int(params_dict.get("nnmax", 100))
    nxmin = int(params_dict.get("nxmin", 1))
    nxmax = int(params_dict.get("nxmax", 100))
    nymin = int(params_dict.get("nymin", 1))
    nymax = int(params_dict.get("nymax", 100))
    sumg_min = int(params_dict.get("sumg_min", 10))
    max_targets = int(params_dict.get("max_targets", (xmax - xmin) * (ymax - ymin)))

    img0 = img.copy()
    n_found, ox, oy, on, onx, ony, osumg = _targ_rec_fast(
        img,
        img0,
        gvthres,
        discont,
        nnmin,
        nnmax,
        nxmin,
        nxmax,
        nymin,
        nymax,
        sumg_min,
        xmin,
        ymin,
        xmax,
        ymax,
        max_targets,
    )

    if write_path is not None:
        p = Path(write_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(f"{n_found}\n")
            for k in range(n_found):
                f.write(
                    f"{k:4d} {ox[k]:9.4f} {oy[k]:9.4f} "
                    f"{on[k]:5d} {onx[k]:5d} {ony[k]:5d} {osumg[k]:5d} {-1:5d}\n"
                )

    return {
        "cam_idx": cam_idx,
        "frame_idx": frame_idx,
        "n_found": int(n_found),
        "x": ox[:n_found].copy(),
        "y": oy[:n_found].copy(),
        "n": on[:n_found].copy(),
        "nx": onx[:n_found].copy(),
        "ny": ony[:n_found].copy(),
        "sumg": osumg[:n_found].copy(),
        "write_path": str(write_path) if write_path is not None else None,
    }


def _normalize_targ_params(targ_rec_params, num_items: int, cam_indices=None) -> list:
    """Normalize user-provided target parameters into a list of dicts."""
    if isinstance(targ_rec_params, list):
        if len(targ_rec_params) == num_items:
            return [dict(p) if isinstance(p, dict) else p for p in targ_rec_params]
        raise ValueError(
            f"Length of targ_rec_params list ({len(targ_rec_params)}) does not match num_items ({num_items})"
        )

    # Object with getter methods (e.g. TargetPar)
    if hasattr(targ_rec_params, "get_grey_thresholds"):
        thresholds = targ_rec_params.get_grey_thresholds()
        nn_bounds = targ_rec_params.get_pixel_count_bounds()
        nx_bounds = targ_rec_params.get_xsize_bounds()
        ny_bounds = targ_rec_params.get_ysize_bounds()
        discont = targ_rec_params.get_max_discontinuity()
        sumg_min = targ_rec_params.get_min_sum_grey()

        param_list = []
        for i in range(num_items):
            cam = cam_indices[i] if cam_indices is not None else 0
            gv = int(thresholds[cam]) if cam < len(thresholds) else int(thresholds[0])
            param_list.append(
                {
                    "gvthres": gv,
                    "discont": int(discont),
                    "nnmin": int(nn_bounds[0]),
                    "nnmax": int(nn_bounds[1]),
                    "nxmin": int(nx_bounds[0]),
                    "nxmax": int(nx_bounds[1]),
                    "nymin": int(ny_bounds[0]),
                    "nymax": int(ny_bounds[1]),
                    "sumg_min": int(sumg_min),
                }
            )
        return param_list

    if isinstance(targ_rec_params, dict):
        return [dict(targ_rec_params) for _ in range(num_items)]

    raise TypeError(f"Unsupported targ_rec_params type: {type(targ_rec_params)}")


def detect_targets_batch_parallel(
    images,
    targ_rec_params,
    n_workers=None,
    chunksize: int = None,
    return_type: str = "targets",
    write_paths=None,
    zarr_store_path: str = None,
    cam_indices=None,
    frame_indices=None,
    use_shared_memory: bool = True,
):
    """Detect targets across a batch of images concurrently.

    Supports image file paths, 2D arrays, or a 3D numpy array. Employs ProcessPoolExecutor
    and optional SharedMemory to achieve near-linear multi-core scaling with zero IPC overhead.

    Args:
        images: List of file paths (str/Path), list of 2D uint8 numpy arrays, or 3D numpy array (N, H, W).
        targ_rec_params: Parameter dict, TargetPar object, or list of dicts.
        n_workers: Number of worker processes (default: os.cpu_count()).
        chunksize: Number of images per worker batch chunk.
        return_type: Format of return values ('targets', 'arrays', or 'counts').
        write_paths: Optional list of output target file paths.
        zarr_store_path: Optional path to Zarr store to write targets directly.
        cam_indices: Optional list of camera indices corresponding to each image.
        frame_indices: Optional list of frame numbers corresponding to each image.
        use_shared_memory: If True, uses multiprocessing.shared_memory for 3D numpy arrays.

    Returns:
        List of detected targets, arrays, or counts matching return_type.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    if isinstance(images, np.ndarray) and images.ndim == 3:
        num_items = images.shape[0]
        is_3d_array = True
    else:
        images = list(images)
        num_items = len(images)
        is_3d_array = False

    if num_items == 0:
        return []

    if cam_indices is None:
        cam_indices = [0] * num_items
    if frame_indices is None:
        frame_indices = list(range(num_items))
    if write_paths is None:
        write_paths = [None] * num_items

    norm_params = _normalize_targ_params(targ_rec_params, num_items, cam_indices)

    shm = None
    tasks = []

    try:
        if is_3d_array and use_shared_memory and num_items > 1 and n_workers != 1:
            from multiprocessing import shared_memory

            images_u8 = np.ascontiguousarray(images, dtype=np.uint8)
            shm = shared_memory.SharedMemory(create=True, size=images_u8.nbytes)
            shm_arr = np.ndarray(images_u8.shape, dtype=np.uint8, buffer=shm.buf)
            shm_arr[:] = images_u8[:]

            item_bytes = images_u8[0].nbytes
            for idx in range(num_items):
                img_source = {
                    "shm_name": shm.name,
                    "shape": images_u8[idx].shape,
                    "offset": idx * item_bytes,
                    "dtype": "uint8",
                }
                tasks.append(
                    (
                        img_source,
                        norm_params[idx],
                        write_paths[idx],
                        cam_indices[idx],
                        frame_indices[idx],
                    )
                )
        else:
            for idx in range(num_items):
                tasks.append(
                    (
                        images[idx],
                        norm_params[idx],
                        write_paths[idx],
                        cam_indices[idx],
                        frame_indices[idx],
                    )
                )

        if n_workers == 1 or num_items == 1:
            raw_results = [_detect_single_worker(t) for t in tasks]
        else:
            max_workers = n_workers if n_workers is not None else min(os.cpu_count() or 1, num_items)
            csize = chunksize if chunksize is not None else max(1, num_items // (max_workers * 4))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                raw_results = list(executor.map(_detect_single_worker, tasks, chunksize=csize))

    finally:
        if shm is not None:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    if zarr_store_path is not None:
        from openptv2.storage import ZarrFrameStore
        store = ZarrFrameStore(zarr_store_path, mode="a")
        for res in raw_results:
            c_idx = res["cam_idx"]
            f_idx = res["frame_idx"]
            n_found = res["n_found"]
            targs = [
                Target(
                    pnr=k,
                    x=float(res["x"][k]),
                    y=float(res["y"][k]),
                    n=int(res["n"][k]),
                    nx=int(res["nx"][k]),
                    ny=int(res["ny"][k]),
                    sumg=int(res["sumg"][k]),
                    tnr=CORRES_NONE,
                )
                for k in range(n_found)
            ]
            store.write_targets(c_idx, f_idx, targs)

    if return_type == "arrays":
        return raw_results

    elif return_type == "counts":
        return [res["n_found"] for res in raw_results]

    else:  # 'targets'
        output_targets = []
        for res in raw_results:
            n_found = res["n_found"]
            if n_found == 0:
                output_targets.append(
                    [Target(pnr=1, x=1.0, y=1.0, n=1, nx=1, ny=1, sumg=1, tnr=CORRES_NONE)]
                )
            else:
                output_targets.append(
                    [
                        Target(
                            pnr=k,
                            x=float(res["x"][k]),
                            y=float(res["y"][k]),
                            n=int(res["n"][k]),
                            nx=int(res["nx"][k]),
                            ny=int(res["ny"][k]),
                            sumg=int(res["sumg"][k]),
                            tnr=CORRES_NONE,
                        )
                        for k in range(n_found)
                    ]
                )
        return output_targets


"""Conversion helpers between openptv_python objects and optv py_bind objects."""

from __future__ import annotations

from types import ModuleType
from typing import Iterable, List

import numpy as np

from ._native_compat import (
    HAS_NATIVE_CALIBRATION,
    HAS_NATIVE_TARGETS,
    HAS_OPTV,
    optv_calibration,
    optv_parameters,
    optv_tracking_framebuf,
)
from .calibration import Calibration
from .parameters import ControlPar, TargetPar
from .tracking_frame_buf import Target


def _require_optv_parameters() -> None:
    if not HAS_OPTV or optv_parameters is None:
        raise RuntimeError("optv py_bind parameters are not available")


def _optv_parameters_module() -> ModuleType:
    _require_optv_parameters()
    assert optv_parameters is not None
    return optv_parameters


def to_native_calibration(cal: Calibration):
    if not HAS_NATIVE_CALIBRATION or optv_calibration is None:
        raise RuntimeError("optv Calibration is not available")

    native = optv_calibration.Calibration()
    native.set_pos(np.asarray(cal.get_pos(), dtype=np.float64))
    native.set_angles(np.asarray(cal.get_angles(), dtype=np.float64))
    native.set_primary_point(np.asarray(cal.get_primary_point(), dtype=np.float64))
    native.set_radial_distortion(
        np.asarray(cal.get_radial_distortion(), dtype=np.float64)
    )
    native.set_decentering(np.asarray(cal.get_decentering(), dtype=np.float64))
    native.set_affine_trans(np.asarray(cal.get_affine(), dtype=np.float64))
    native.set_glass_vec(np.asarray(cal.get_glass_vec(), dtype=np.float64))
    return native


def to_native_control_par(cpar: ControlPar):
    parameters_module = _optv_parameters_module()

    flags = [
        flag_name
        for enabled, flag_name in (
            (bool(cpar.hp_flag), "hp"),
            (bool(cpar.all_cam_flag), "allcam"),
            (bool(cpar.tiff_flag), "headers"),
        )
        if enabled
    ]

    native = parameters_module.ControlParams(
        cpar.num_cams,
        flags=flags,
        image_size=(cpar.imx, cpar.imy),
        pixel_size=(cpar.pix_x, cpar.pix_y),
        cam_side_n=cpar.mm.n1,
        wall_ns=list(cpar.mm.n2),
        wall_thicks=list(cpar.mm.d),
        object_side_n=cpar.mm.n3,
    )
    native.set_chfield(cpar.chfield)

    for cam_index, img_base_name in enumerate(cpar.img_base_name):
        native.set_img_base_name(cam_index, img_base_name)

    for cam_index, cal_img_base_name in enumerate(cpar.cal_img_base_name):
        native.set_cal_img_base_name(cam_index, cal_img_base_name)

    return native


def to_native_target_par(tpar: TargetPar):
    parameters_module = _optv_parameters_module()

    thresholds = list(tpar.gvthresh)
    if len(thresholds) < 4:
        thresholds.extend([0] * (4 - len(thresholds)))

    return parameters_module.TargetParams(
        discont=tpar.discont,
        gvthresh=thresholds[:4],
        pixel_count_bounds=(tpar.nnmin, tpar.nnmax),
        xsize_bounds=(tpar.nxmin, tpar.nxmax),
        ysize_bounds=(tpar.nymin, tpar.nymax),
        min_sum_grey=tpar.sumg_min,
        cross_size=tpar.cr_sz,
    )


def to_native_target(target: Target):
    if not HAS_NATIVE_TARGETS or optv_tracking_framebuf is None:
        raise RuntimeError("optv Target is not available")

    return optv_tracking_framebuf.Target(
        pnr=target.pnr,
        x=target.x,
        y=target.y,
        n=target.n,
        nx=target.nx,
        ny=target.ny,
        sumg=target.sumg,
        tnr=target.tnr,
    )


def from_native_target(native_target) -> Target:
    x, y = native_target.pos()
    n, nx, ny = native_target.count_pixels()
    return Target(
        pnr=int(native_target.pnr()),
        x=float(x),
        y=float(y),
        n=int(n),
        nx=int(nx),
        ny=int(ny),
        sumg=int(native_target.sum_grey_value()),
        tnr=int(native_target.tnr()),
    )


def from_native_target_array(native_targets: Iterable[object]) -> List[Target]:
    return [from_native_target(target) for target in native_targets]

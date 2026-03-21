"""Optional compatibility layer for reusing optv py_bind as a native provider."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except Exception:
        return None


optv_calibration = _optional_import("optv.calibration")
optv_image_processing = _optional_import("optv.image_processing")
optv_parameters = _optional_import("optv.parameters")
optv_segmentation = _optional_import("optv.segmentation")
optv_tracking_framebuf = _optional_import("optv.tracking_framebuf")

HAS_OPTV = any(
    module is not None
    for module in (
        optv_calibration,
        optv_image_processing,
        optv_parameters,
        optv_segmentation,
        optv_tracking_framebuf,
    )
)

HAS_NATIVE_PREPROCESS = (
    optv_image_processing is not None
    and hasattr(optv_image_processing, "preprocess_image")
    and optv_parameters is not None
    and hasattr(optv_parameters, "ControlParams")
)

HAS_NATIVE_SEGMENTATION = (
    optv_segmentation is not None
    and hasattr(optv_segmentation, "target_recognition")
    and optv_parameters is not None
    and hasattr(optv_parameters, "TargetParams")
    and hasattr(optv_parameters, "ControlParams")
    and optv_tracking_framebuf is not None
    and hasattr(optv_tracking_framebuf, "Target")
)

HAS_NATIVE_CALIBRATION = optv_calibration is not None and hasattr(
    optv_calibration, "Calibration"
)

HAS_NATIVE_TARGETS = optv_tracking_framebuf is not None and hasattr(
    optv_tracking_framebuf, "Target"
)


def native_preprocess_image(*args: Any, **kwargs: Any) -> Any:
    if not HAS_NATIVE_PREPROCESS or optv_image_processing is None:
        raise RuntimeError("optv native preprocess_image is not available")
    return optv_image_processing.preprocess_image(*args, **kwargs)


def native_target_recognition(*args: Any, **kwargs: Any) -> Any:
    if not HAS_NATIVE_SEGMENTATION or optv_segmentation is None:
        raise RuntimeError("optv native target_recognition is not available")
    return optv_segmentation.target_recognition(*args, **kwargs)

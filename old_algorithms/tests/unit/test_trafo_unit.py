"""Unit tests for trafo module numerical invariants."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.parameters import ControlPar
from algorithms.trafo import (
    arr_metric_to_pixel,
    arr_pixel_to_metric,
    fast_pixel_to_metric,
    metric_to_pixel,
    pixel_to_metric,
)


@pytest.mark.unit
def test_pixel_metric_round_trip_scalar() -> None:
    cpar = ControlPar(num_cams=1, imx=1280, imy=1024, pix_x=0.01, pix_y=0.01)

    x_pix, y_pix = 777.25, 123.75
    x_m, y_m = pixel_to_metric(x_pix, y_pix, cpar)
    x_back, y_back = metric_to_pixel(x_m, y_m, cpar)

    assert x_back == pytest.approx(x_pix)
    assert y_back == pytest.approx(y_pix)


@pytest.mark.unit
def test_pixel_metric_round_trip_array() -> None:
    cpar = ControlPar(num_cams=1, imx=1920, imy=1080, pix_x=0.008, pix_y=0.008)
    pixels = np.array([[100, 200], [350, 900], [1400, 500]], dtype=np.int32)

    metric = arr_pixel_to_metric(pixels, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y)
    back = arr_metric_to_pixel(metric, cpar)

    np.testing.assert_allclose(back, pixels.astype(np.float64), rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_pixel_to_metric_zero_pixel_size_raises() -> None:
    with pytest.raises(ValueError, match="Pixel size cannot be zero"):
        fast_pixel_to_metric(1.0, 2.0, 100, 100, 0.0, 0.01)

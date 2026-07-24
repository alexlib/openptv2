"""save_residual_field_figure writes a PNG (smoke test, no display)."""
from pathlib import Path

import numpy as np
import pytest

from openptv2.calibration_diagnostics import save_residual_field_figure


@pytest.mark.unit
def test_residual_field_png_written(tmp_path):
    rng = np.random.default_rng(0)
    det = rng.uniform(0, 512, (20, 2))
    rep = det + rng.normal(0, 1.0, (20, 2))
    err = np.hypot(det[:, 0] - rep[:, 0], det[:, 1] - rep[:, 1])
    dest = tmp_path / "field.png"
    out = save_residual_field_figure(det, rep, err, None, dest, scale=15.0)
    assert Path(out).exists() and Path(out).stat().st_size > 0


@pytest.mark.unit
def test_residual_field_with_image_background(tmp_path):
    img = np.zeros((512, 512), dtype=np.uint8)
    det = np.array([[100.0, 100.0], [300.0, 300.0]])
    rep = det + 0.5
    err = np.hypot(det[:, 0] - rep[:, 0], det[:, 1] - rep[:, 1])
    dest = tmp_path / "field_img.png"
    save_residual_field_figure(det, rep, err, img, dest)
    assert dest.exists()

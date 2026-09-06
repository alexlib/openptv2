"""Windows-authored parameter YAMLs must resolve on Linux too.

A YAML using '\\' as a path separator, or mixing '.TIF'/'.tif' case for the
same physical file, works by accident on Windows (backslash is a valid
separator there; the filesystem is case-insensitive) and used to break
silently -- FileNotFoundError / "calibration files missing" -- the moment it
ran inside a Linux container. See ptv._frame_image_name and
ptv_calibration._resolve_ci / _read_calibrations.
"""

import sys
from pathlib import Path

import pytest

from openptv2.gui.ptv import _frame_image_name
from openptv2.gui.ptv_calibration import _read_calibrations, _resolve_ci
from openptv2.parameters import ControlParams

#: A file named "cam_1.TIF" and one named "cam_1.tif" are the same file on
#: Windows (case-insensitive filesystem) -- the mismatch these tests exist
#: for cannot even be constructed there, so the case-insensitive-fallback
#: path never triggers (the exact-name check already succeeds). Linux CI is
#: what actually exercises it.
_CASE_SENSITIVE_FS = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows filesystems are case-insensitive; cam_1.TIF and cam_1.tif "
    "are the same file there, so this scenario can't be constructed",
)


def test_frame_image_name_normalizes_backslash_separator(tmp_path, monkeypatch):
    (tmp_path / "img").mkdir()
    target = tmp_path / "img" / "Exp1_000001.tif"
    target.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    resolved = _frame_image_name(r"img\Exp1_%06d.tif", 1)

    assert resolved == Path("img/Exp1_000001.tif")
    assert resolved.exists()


@_CASE_SENSITIVE_FS
def test_frame_image_name_falls_back_case_insensitively(tmp_path, monkeypatch):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "cam_1.tif").write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    # Parameter file spells it uppercase; actual file on disk is lowercase.
    resolved = _frame_image_name("img/CAM_1.TIF", 1)

    assert resolved.exists()
    assert resolved.name == "cam_1.tif"


@_CASE_SENSITIVE_FS
def test_resolve_ci_matches_different_case(tmp_path):
    (tmp_path / "cam_1.tif.ori").write_bytes(b"")

    assert _resolve_ci(tmp_path / "cam_1.TIF.ori") == str(tmp_path / "cam_1.tif.ori")
    assert _resolve_ci(tmp_path / "cam_1.tif.ori") == str(tmp_path / "cam_1.tif.ori")
    assert _resolve_ci(tmp_path / "does_not_exist.ori") is None


@_CASE_SENSITIVE_FS
def test_read_calibrations_tolerates_case_mismatch(tmp_path):
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    # A real (minimal, valid) .ori/.addpar pair -- Calibration.from_file
    # parses fixed fields positionally and errors on a malformed stub.
    (cal_dir / "cam_1.tif.ori").write_text(
        "  -133.4327    -93.3135    528.3781\n"
        "     0.3307792  -0.4467329   1.6382096\n"
        "\n"
        "    -0.0607515 -0.8998148 -0.4320214\n"
        "     0.9530930  0.0762827 -0.2929072\n"
        "     0.2965180 -0.4295511  0.8529730\n"
        "\n"
        "      5.5284 -10.1467\n"
        "     59.8710\n"
        "\n"
        "       0.000000000000000    0.000000000000000    50.00000000000000\n"
    )
    (cal_dir / "cam_1.tif.addpar").write_text(
        "0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000\n"
    )

    cpar = ControlParams(1)
    # ptv.img_cal spelled uppercase, like a Windows-authored YAML; the actual
    # files on disk (above) are lowercase.
    cpar.set_cal_img_base_name(0, "cal/cam_1.TIF")

    cals = _read_calibrations(cpar, 1, base_dir=tmp_path)

    assert len(cals) == 1

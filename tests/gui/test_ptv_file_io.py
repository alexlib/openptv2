"""Unit tests for file I/O functions in ptv.py"""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from openptv2.gui.ptv import (
    _ensure_directory_writable,
    _ensure_target_output_writable,
    _prepare_output_path,
    _raise_output_write_error,
    extract_cam_ids,
    generate_short_file_bases,
    py_correspondences_proc_c,
    read_rt_is_file,
    read_targets,
    write_targets,
)


class TestReadTargets:
    """Test read_targets function (RunStore-backed)"""

    def test_read_targets_from_store(self, tmp_path):
        from openptv2.storage import RunStore
        from openptv2.tracking_framebuf import TargetArray

        store = RunStore(tmp_path / "run.zarr", mode="w")
        tarr = TargetArray(2)
        tarr[0].set_pnr(0)
        tarr[0].set_pos((100.5, 200.5))
        tarr[1].set_pnr(1)
        tarr[1].set_pos((110.5, 210.5))
        store.write_targets(0, 10000, tarr)

        result = read_targets("img/cam1", 10000, store=store, cam_idx=0)
        assert len(result) == 2

    def test_read_targets_without_store_raises(self):
        with pytest.raises(FileNotFoundError):
            read_targets("img/cam1", 10000)


class TestWriteTargets:
    """Test write_targets function (RunStore-backed)"""

    def test_write_targets_to_store(self, tmp_path):
        from openptv2.storage import RunStore
        from openptv2.tracking_framebuf import TargetArray

        store = RunStore(tmp_path / "run.zarr", mode="w")
        tarr = TargetArray(1)
        tarr[0].set_pnr(0)
        tarr[0].set_pos((100.5, 200.5))

        result = write_targets(tarr, "cam1", 123456789, store=store, cam_idx=0)
        assert result is True
        assert store.has_targets(0, 123456789)

    def test_write_targets_without_store_raises(self):
        with pytest.raises(ValueError, match="RunStore"):
            write_targets([], "cam1", 123456789)


class TestOutputHelpers:
    """Test low-level output path helper functions."""

    def test_prepare_output_path_creates_parent_directory(self, tmp_path):
        output_path = tmp_path / "nested" / "res" / "out.txt"

        result = _prepare_output_path(str(output_path))

        assert result == output_path
        assert output_path.parent.exists()

    def test_prepare_output_path_raises_for_invalid_parent(self, tmp_path):
        blocking_file = tmp_path / "blocked"
        blocking_file.write_text("x", encoding="utf-8")

        with pytest.raises(OSError, match="Unable to prepare output directory"):
            _prepare_output_path(str(blocking_file / "out.txt"))

    def test_raise_output_write_error_for_permission_error(self):
        output_path = Path("res/out.txt")

        with pytest.raises(PermissionError, match="Cannot write output file"):
            _raise_output_write_error(output_path, PermissionError("denied"))

    def test_raise_output_write_error_for_generic_oserror(self):
        output_path = Path("res/out.txt")

        with pytest.raises(OSError, match="Failed to write output file"):
            _raise_output_write_error(output_path, OSError("disk full"))

    def test_ensure_directory_writable_creates_and_probes_directory(self, tmp_path):
        output_dir = tmp_path / "targets"

        result = _ensure_directory_writable(output_dir, "Target output")

        assert result == output_dir
        assert output_dir.exists()
        assert not list(output_dir.glob(".pyptv_write_probe*"))

    def test_ensure_target_output_writable_checks_unique_directories(self, tmp_path):
        base_paths = [
            str(tmp_path / "cam1"),
            str(tmp_path / "cam2"),
            str((tmp_path / "nested") / "cam3"),
        ]

        with patch("openptv2.gui.ptv._ensure_directory_writable") as ensure_dir:
            _ensure_target_output_writable(base_paths)

        checked_dirs = [call.args[0] for call in ensure_dir.call_args_list]
        assert checked_dirs == [Path(tmp_path), Path(tmp_path / "nested")]


class TestCorrespondenceWritePreflight:
    def test_correspondences_checks_target_directory_before_writing(self):
        exp = Mock()
        exp.detections = [Mock()]
        exp.corrected = Mock()
        exp.cals = Mock()
        exp.vpar = Mock()
        exp.cpar = Mock()
        exp.num_cams = 1
        exp.target_filenames = ["cam1"]

        sorted_pos = [np.zeros((2, 1))]
        sorted_corresp = [np.zeros((1, 1), dtype=int)]

        with patch(
            "openptv2.gui.ptv.correspondences",
            return_value=(sorted_pos, sorted_corresp, 1),
        ):
            with patch(
                "openptv2.gui.ptv._ensure_target_output_writable"
            ) as ensure_writable:
                with patch("openptv2.gui.ptv.write_targets") as write_targets_mock:
                    py_correspondences_proc_c(exp)

        ensure_writable.assert_called_once_with(exp.target_filenames)
        write_targets_mock.assert_called_once()


def clean_bases(file_bases):
    import re

    """Remove frame number patterns like %d, %04d, etc. from file bases"""
    return [re.sub(r"%0?\d*d", "", s) for s in file_bases]


class TestExtractCamIds:
    """Test extract_cam_ids function"""

    def test_extract_cam_ids_basic(self):
        """Test extraction of camera ids from typical file base names"""
        file_bases = [
            "cam1_%04d.tif",
            "img_cam2_%03d.tif",
            "exp_test_cam_01_frame_%04d.tif",
            "c5_%d",
            "Cam12_extra",
            "c13",
            "C001H001S0001000001.tif",
        ]
        expected = [1, 2, 1, 5, 12, 13, 1]
        result = extract_cam_ids(file_bases)
        assert result == expected

    def test_extract_cam_ids_multiple_numbers(self):
        """Test extraction when multiple numbers are present in base names"""
        file_bases = [
            "prefix_cam1_img2_%04d.tif",
            "prefix_cam2_img3_%04d.tif",
            "prefix_cam3_img4_%04d.tif",
        ]
        # The cam id should be the one that varies (cam1, cam2, cam3 -> 1,2,3)
        expected = [1, 2, 3]
        result = extract_cam_ids(file_bases)
        assert result == expected

    def test_extract_cam_ids_no_numbers(self):
        """Test extraction when no numbers are present"""
        file_bases = ["camera0_%d.tif", "camera1_%d.tif"]
        expected = [0, 1]
        result = extract_cam_ids(file_bases)
        assert result == expected

    def test_extract_cam_ids_single_entry(self):
        """Test extraction with a single file base"""
        file_bases = ["cam7_%04d.tif"]
        expected = [7]
        result = extract_cam_ids(file_bases)
        assert result == expected

    def test_extract_cam_ids_empty_list(self):
        """Test extraction with empty list should raise ValueError"""
        file_bases = []
        with pytest.raises(ValueError):
            extract_cam_ids(file_bases)

    def test_extract_cam_ids_trailing_number(self):
        """Test extraction when only trailing number is present"""
        file_bases = ["foo_bar_99"]
        expected = [99]
        result = extract_cam_ids(file_bases)
        assert result == expected

    def test_extract_cam_ids_varied_patterns(self):
        """Test extraction with varied patterns and leading zeros"""
        file_bases = ["cam01_%04d.tif", "cam02_%04d.tif", "cam03_%04d.tif"]
        expected = [1, 2, 3]
        result = extract_cam_ids(file_bases)
        assert result == expected

    def test_extract_cam_ids_with_percent_d(self):
        """Test extraction with percent-d patterns"""
        file_bases = ["img_c1_%d", "img_c2_%d", "img_c3_%d"]
        expected = [1, 2, 3]
        result = extract_cam_ids(file_bases)
        assert result == expected

    def test_extract_cam_ids_fallback(self):
        """Test fallback to last number if no varying position"""
        file_bases = ["foo_1_bar_2", "foo_1_bar_2"]
        expected = [2, 2]
        result = extract_cam_ids(file_bases)
        assert result == expected


class TestCleanBases:
    """Test clean_bases utility function"""

    def test_clean_bases_removes_percent_d(self):
        file_bases = [
            "cam1_%04d.tif",
            "img_cam2_%03d.tif",
            "exp_test_cam_01_frame_%04d.tif",
            "c5_%d",
        ]
        expected = ["cam1_.tif", "img_cam2_.tif", "exp_test_cam_01_frame_.tif", "c5_"]
        result = clean_bases(file_bases)
        assert result == expected

    def test_clean_bases_no_pattern(self):
        file_bases = ["cam1.tif", "img_cam2.tif"]
        expected = ["cam1.tif", "img_cam2.tif"]
        result = clean_bases(file_bases)
        assert result == expected

    def test_clean_bases_empty(self):
        file_bases = []
        expected = []
        result = clean_bases(file_bases)
        assert result == expected


class TestFileBaseToFilename:
    """Test file_base_to_short_file_base function"""

    def test_extract_cam_id(self):
        """Test extraction of cam_id from various base names"""
        test_cases = [
            ("cam1_%04d.tif", [1]),
            ("img_cam2_%03d.tif", [2]),
            ("exp_test_cam_01_frame_%04d.tif", [1]),
            ("c5_%%d", [5]),
            ("Cam12_extra", [12]),
            ("c13", [13]),
            ("C001H001S0001%05d.tif", [1]),
        ]

        for base_name, expected_id in test_cases:
            cam_id = extract_cam_ids(base_name)
            assert cam_id == expected_id, (
                f"{base_name} -> {cam_id}, expected {expected_id}"
            )

    # def test_generate_short_file_bases(self):
    #     """Test generation of short file bases from a list of base names"""
    #     base_names = [s
    #         "cam1_%04d.tif",
    #         "img_cam2_%03d.tif",
    #         "exp_test_cam_01_frame_%04d.tif",
    #         "c5_%%d",
    #         "Cam12_extra",
    #         "c13",
    #     ]
    #     short_bases = generate_short_file_bases(base_names)
    #     assert len(short_bases) == len(base_names)
    #     for base, short in zip(base_names, short_bases):
    #         cam_id = extract_cam_id(base)
    #         assert short.startswith(f"cam{cam_id}"), f"Short base {short} does not start with cam{cam_id}"


class TestReadRtIsFile:
    """Test read_rt_is_file function (Zarr-backed)"""

    def test_read_rt_is_file_from_store(self, tmp_path):
        from openptv2.storage import RunStore

        res_dir = tmp_path / "res"
        res_dir.mkdir()
        store = RunStore(res_dir / "run.zarr", mode="w")
        store.write_correspondences(
            10001,
            np.array([[100.5, 200.5, 50.0], [110.5, 210.5, 60.0]]),
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
        )
        result = read_rt_is_file(str(res_dir / "rt_is.10001"))
        assert len(result) == 2
        assert result[0] == [100.5, 200.5, 50.0, 1, 2, 3, 4]
        assert result[1] == [110.5, 210.5, 60.0, 5, 6, 7, 8]

    def test_read_rt_is_file_nonexistent_file(self):
        """Test reading nonexistent store raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            read_rt_is_file("nonexistent_file.10001")


class TestExtractFrameNum:
    def test_extract_frame_num_patterns(self):
        from openptv2.gui.ptv import _extract_frame_num

        assert _extract_frame_num("img/cam1.10002") == 10002
        assert _extract_frame_num("img/cam1_10002.png") == 10002
        assert _extract_frame_num("img/00000025.tif") == 25
        assert _extract_frame_num("img/cam_no_numbers.png") == 123456789
        assert _extract_frame_num("") == 123456789


class TestFrameImageName:
    def test_frame_image_name_sorts_candidates_deterministically(self, tmp_path):
        """Glob fallback must be sorted — filesystem order is nondeterministic."""
        from openptv2.gui.ptv import _frame_image_name

        # No exact cam1.0001.tif, but two prefix matches: _A sorts before _B
        (tmp_path / "cam1.0001_B.tif").write_bytes(b"fake")
        (tmp_path / "cam1.0001_A.tif").write_bytes(b"fake")
        base = str(tmp_path / "cam1.%04d.tif")
        # Intentionally create files in reverse lex order on disk; result must still be _A
        result = _frame_image_name(base, 1)
        assert result.name == "cam1.0001_A.tif"

    def test_frame_image_name_returns_exact_when_exists(self, tmp_path):
        from openptv2.gui.ptv import _frame_image_name

        exact = tmp_path / "cam1.0001.tif"
        exact.write_bytes(b"fake")
        base = str(tmp_path / "cam1.%04d.tif")
        assert _frame_image_name(base, 1) == exact


if __name__ == "__main__":
    pytest.main([__file__])

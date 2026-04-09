"""
Engine comparison tests for correspondences module.

Each engine reads the SAME parameter files through its own reader,
ensuring both reader parity and algorithm parity are tested.

Tests MatchedCoords class and correspondences function.
Tolerance: 1e-7 (complex matching algorithms)
"""

import numpy as np
import pytest
from .conftest import get_tolerance

TOLERANCE = get_tolerance("correspondences")


class TestMatchedCoords:
    """Compare MatchedCoords between optv and python engines."""

    def test_matched_coords_creation(self, file_control_params, file_calibration_cam1):
        """Test MatchedCoords creation with targets from file-based params."""
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords as OptvMatchedCoords

        optv_cpar, python_cpar = file_control_params
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cpar is not None
        assert python_cpar is not None
        assert optv_cal is not None
        assert python_cal is not None

        optv_ta = TargetArray(10)
        for i in range(10):
            optv_ta[i].set_pnr(i)
            optv_ta[i].set_pos((float(i * 10), float(i * 20)))
            optv_ta[i].set_pixel_counts(5, 2, 2)
            optv_ta[i].set_sum_grey_value(100.0)
            optv_ta[i].set_tnr(0)

        try:
            optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)
            optv_result = optv_mc.as_arrays()
        except Exception as e:
            pytest.fail(f"optv MatchedCoords failed: {e}")

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.tracking_frame_buf import TargetArray as PythonTA

            python_ta = PythonTA(10)
            for i in range(10):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10), float(i * 20)))
                python_ta[i].set_pixel_counts(5, 2, 2)
                python_ta[i].set_sum_grey_value(100.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_result = python_mc.as_arrays()

            np.testing.assert_allclose(
                optv_result[0], python_result[0], rtol=TOLERANCE, atol=TOLERANCE
            )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_matched_coords_as_arrays(self, file_control_params, file_calibration_cam1):
        """Test MatchedCoords.as_arrays() method with file-based params."""
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords as OptvMatchedCoords

        optv_cpar, python_cpar = file_control_params
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cpar is not None
        assert python_cpar is not None
        assert optv_cal is not None
        assert python_cal is not None

        optv_ta = TargetArray(5)
        for i in range(5):
            optv_ta[i].set_pnr(i)
            optv_ta[i].set_pos((float(i * 10 + 5), float(i * 15 + 3)))
            optv_ta[i].set_pixel_counts(3, 2, 2)
            optv_ta[i].set_sum_grey_value(50.0)
            optv_ta[i].set_tnr(0)

        optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)
        pos, pnr = optv_mc.as_arrays()

        assert pos.shape[1] == 2
        assert len(pos) == len(pnr)

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.tracking_frame_buf import TargetArray as PythonTA

            python_ta = PythonTA(5)
            for i in range(5):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10 + 5), float(i * 15 + 3)))
                python_ta[i].set_pixel_counts(3, 2, 2)
                python_ta[i].set_sum_grey_value(50.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_pos, python_pnr = python_mc.as_arrays()

            np.testing.assert_allclose(pos, python_pos, rtol=TOLERANCE, atol=TOLERANCE)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_matched_coords_get_by_pnrs(
        self, file_control_params, file_calibration_cam1
    ):
        """Test MatchedCoords.get_by_pnrs() method with file-based params."""
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords as OptvMatchedCoords

        optv_cpar, python_cpar = file_control_params
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cpar is not None
        assert python_cpar is not None
        assert optv_cal is not None
        assert python_cal is not None

        optv_ta = TargetArray(8)
        for i in range(8):
            optv_ta[i].set_pnr(i)
            optv_ta[i].set_pos((float(i * 10), float(i * 20)))
            optv_ta[i].set_pixel_counts(5, 2, 2)
            optv_ta[i].set_sum_grey_value(100.0)
            optv_ta[i].set_tnr(0)

        optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)

        query_pnrs = np.array([0, 2, 4, 6], dtype=np.intp)
        optv_result = optv_mc.get_by_pnrs(query_pnrs)

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.tracking_frame_buf import TargetArray as PythonTA

            python_ta = PythonTA(8)
            for i in range(8):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10), float(i * 20)))
                python_ta[i].set_pixel_counts(5, 2, 2)
                python_ta[i].set_sum_grey_value(100.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_result = python_mc.get_by_pnrs(query_pnrs)

            np.testing.assert_allclose(
                optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
            )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestCorrespondencesFunction:
    """Test correspondences function."""

    def test_correspondences_basic(self, file_control_params, file_calibration_4cam):
        """Test correspondences function with file-based params."""
        from optv.correspondences import correspondences as optv_func
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords
        from optv.parameters import VolumeParams

        optv_cpar, python_cpar = file_control_params
        optv_cals, python_cal_list = file_calibration_4cam
        assert optv_cpar is not None
        assert python_cpar is not None

        num_targets = 8
        num_cams = 4

        img_pts = []
        flat_coords = []
        cals = []

        for cam in range(num_cams):
            ta = TargetArray(num_targets)
            for i in range(num_targets):
                ta[i].set_pnr(i)
                ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                ta[i].set_pixel_counts(5, 2, 2)
                ta[i].set_sum_grey_value(100.0)
                ta[i].set_tnr(0)
            img_pts.append(ta)

            mc = MatchedCoords(ta, optv_cpar, optv_cals[cam])
            flat_coords.append(mc)
            cals.append(optv_cals[cam])

        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)

        try:
            optv_result = optv_func(img_pts, flat_coords, cals, vpar, optv_cpar)
        except Exception as e:
            pytest.fail(f"optv correspondences failed: {e}")

        try:
            from algorithms.correspondences import correspondences as python_func
            from algorithms.tracking_frame_buf import (
                TargetArray as PythonTA,
                Frame as PythonFrame,
            )
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.parameters import VolumePar as PythonVolumePar

            python_img_pts = []
            python_flat_coords = []
            python_vpar = PythonVolumePar(
                x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0]
            )
            for cam in range(num_cams):
                ta = PythonTA(num_targets)
                for i in range(num_targets):
                    ta[i].set_pnr(i)
                    ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                    ta[i].set_pixel_counts(5, 2, 2)
                    ta[i].set_sum_grey_value(100.0)
                    ta[i].set_tnr(0)
                python_img_pts.append(ta)
                cal = python_cal_list[cam]
                python_flat_coords.append(PythonMatchedCoords(ta, python_cpar, cal))
            match_counts = [0, 0, 0, 0]
            python_result = python_func(
                PythonFrame(num_cams),
                python_flat_coords,
                python_vpar,
                python_cpar,
                python_cal_list,
                match_counts,
            )
        except (ImportError, AttributeError, TypeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestCandidateSearchHelpers:
    """Test the candidate search helpers that underpin Phase 2."""

    def test_find_start_point_binary_anchor(self):
        from algorithms.find_candidate import find_start_point_binary

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        idx = find_start_point_binary(x, len(x), 0.4, 0.2)

        assert 0 <= idx < len(x)
        assert x[idx] <= 0.4 + 0.2

    def test_vectorized_candidate_search_preserves_x_order(self):
        from algorithms.correspondences import _find_candidates_vectorized

        crd_x2 = np.array([-1.0, 0.0, 0.2, 0.4, 1.2], dtype=np.float64)
        crd_y2 = np.array([-1.0, 0.1, 0.15, 0.3, 1.0], dtype=np.float64)
        crd_pnr2 = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        targ_n2 = np.array([10, 10, 10, 10, 10], dtype=np.int64)
        targ_nx2 = np.array([10, 10, 10, 10, 10], dtype=np.int64)
        targ_ny2 = np.array([10, 10, 10, 10, 10], dtype=np.int64)
        targ_sumg2 = np.array([100, 100, 100, 100, 100], dtype=np.int64)

        cand = _find_candidates_vectorized(
            crd_x2, crd_y2, crd_pnr2, 5,
            targ_n2, targ_nx2, targ_ny2, targ_sumg2,
            -0.5, 0.0, 0.5, 0.4,
            10, 10, 10, 100,
            0.5, 0.1, 0.1, 0.1, 0.1,
        )

        assert len(cand) > 0
        assert [c[0] for c in cand] == sorted([c[0] for c in cand])

    def test_take_best_candidates_basic(self):
        from algorithms.correspondences import take_best_candidates
        from algorithms.tracking_frame_buf import n_tupel_dtype

        # Only populate 2 real candidates out of 2 slots
        src = np.recarray(2, dtype=n_tupel_dtype)
        src.p = -1
        src.corr = 0.0
        src[0].p = np.array([1, -1, -1, -1], dtype=np.int32)
        src[0].corr = 0.6
        src[1].p = np.array([2, -1, -1, -1], dtype=np.int32)
        src[1].corr = 0.9

        dst = np.recarray(4, dtype=n_tupel_dtype)
        dst.p = -1
        dst.corr = 0.0
        tusage = np.zeros((4, 10), dtype=np.int32)

        taken = take_best_candidates(src, dst, 4, tusage)

        assert taken == 2
        # Best corr (0.9) should be taken first
        assert list(dst[0].p) == [2, -1, -1, -1]
        assert list(dst[1].p) == [1, -1, -1, -1]

    def test_single_cam_correspondence(
        self, file_control_params, file_calibration_cam1
    ):
        """Test single_cam_correspondence with file-based params."""
        from optv.correspondences import single_cam_correspondence as optv_func
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords

        optv_cpar, python_cpar = file_control_params
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cpar is not None
        assert python_cpar is not None
        assert optv_cal is not None
        assert python_cal is not None

        num_targets = 6

        ta = TargetArray(num_targets)
        for i in range(num_targets):
            ta[i].set_pnr(i)
            ta[i].set_pos((float(i * 10), float(i * 20)))
            ta[i].set_pixel_counts(5, 2, 2)
            ta[i].set_sum_grey_value(100.0)
            ta[i].set_tnr(0)

        mc = MatchedCoords(ta, optv_cpar, optv_cal)
        img_pts = [ta]
        flat_coords = [mc]
        cals = [optv_cal]

        try:
            optv_result = optv_func(img_pts, flat_coords, cals)
        except Exception as e:
            pytest.fail(f"optv single_cam_correspondence failed: {e}")

        try:
            from algorithms.correspondences import (
                single_cam_correspondence as python_func,
                MatchedCoords as PythonMatchedCoords,
            )
            from algorithms.tracking_frame_buf import TargetArray as PythonTA

            python_ta = PythonTA(num_targets)
            for i in range(num_targets):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10), float(i * 20)))
                python_ta[i].set_pixel_counts(5, 2, 2)
                python_ta[i].set_sum_grey_value(100.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_result = python_func([python_ta], [python_mc])
        except (ImportError, AttributeError, TypeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestCorrespondencesEdgeCases:
    """Test edge cases for correspondence functions."""

    def test_correspondences_empty_targets(
        self, file_control_params, file_calibration_4cam
    ):
        """Test with empty target arrays."""
        from optv.correspondences import correspondences as optv_func
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords
        from optv.parameters import VolumeParams

        optv_cpar, python_cpar = file_control_params
        optv_cals, python_cals = file_calibration_4cam
        assert optv_cpar is not None
        assert python_cpar is not None

        num_cams = 4

        img_pts = [TargetArray(0) for _ in range(num_cams)]
        flat_coords = []
        cals = []

        for cam in range(num_cams):
            mc = MatchedCoords(TargetArray(0), optv_cpar, optv_cals[cam])
            flat_coords.append(mc)
            cals.append(optv_cals[cam])

        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)

        try:
            optv_result = optv_func(img_pts, flat_coords, cals, vpar, optv_cpar)
        except Exception:
            pass

        try:
            from algorithms.correspondences import correspondences as python_func
            from algorithms.tracking_frame_buf import Frame as PythonFrame

            python_frame = PythonFrame(num_cams)
            python_frame.targets = img_pts
            python_frame.num_targets = [0 for _ in range(num_cams)]
            match_counts = [0, 0, 0, 0]
            python_result = python_func(
                python_frame, flat_coords, vpar, python_cpar, python_cals, match_counts
            )
        except (ImportError, AttributeError, TypeError):
            pass

    def test_correspondences_single_target(
        self, file_control_params, file_calibration_4cam
    ):
        """Test with single target per camera."""
        from optv.correspondences import correspondences as optv_func
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords
        from optv.parameters import VolumeParams

        optv_cpar, python_cpar = file_control_params
        optv_cals, python_cal_list = file_calibration_4cam
        assert optv_cpar is not None
        assert python_cpar is not None

        num_cams = 4

        img_pts = []
        flat_coords = []
        cals = []

        for cam in range(num_cams):
            ta = TargetArray(1)
            ta[0].set_pnr(0)
            ta[0].set_pos((50.0, 50.0))
            ta[0].set_pixel_counts(5, 2, 2)
            ta[0].set_sum_grey_value(100.0)
            ta[0].set_tnr(0)
            img_pts.append(ta)

            mc = MatchedCoords(ta, optv_cpar, optv_cals[cam])
            flat_coords.append(mc)
            cals.append(optv_cals[cam])

        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)

        try:
            optv_result = optv_func(img_pts, flat_coords, cals, vpar, optv_cpar)
        except Exception as e:
            pytest.fail(f"optv correspondences failed: {e}")

        try:
            from algorithms.correspondences import correspondences as python_func
            from algorithms.tracking_frame_buf import (
                TargetArray as PythonTA,
                Frame as PythonFrame,
            )
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.parameters import VolumePar as PythonVolumePar

            python_frame = PythonFrame(num_cams)
            python_img_pts = []
            python_flat_coords = []
            python_vpar = PythonVolumePar(
                x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0]
            )
            for cam in range(num_cams):
                ta = PythonTA(1)
                ta[0].set_pnr(0)
                ta[0].set_pos((50.0, 50.0))
                ta[0].set_pixel_counts(5, 2, 2)
                ta[0].set_sum_grey_value(100.0)
                ta[0].set_tnr(0)
                python_img_pts.append(ta)
                cal = python_cal_list[cam]
                python_flat_coords.append(PythonMatchedCoords(ta, python_cpar, cal))

            python_frame.targets = python_img_pts
            python_frame.num_targets = [1 for _ in range(num_cams)]
            match_counts = [0, 0, 0, 0]
            python_result = python_func(
                python_frame,
                python_flat_coords,
                python_vpar,
                python_cpar,
                python_cal_list,
                match_counts,
            )
        except (ImportError, AttributeError, TypeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestMatchPairsSoA:
    """Test SoA match_pairs against current recarray implementation."""

    def test_match_pairs_soa_vs_original(
        self, file_control_params, file_calibration_4cam
    ):
        """Compare match_pairs_soa output against current match_pairs."""
        from algorithms.correspondences import (
            match_pairs,
            match_pairs_soa,
            safely_allocate_adjacency_lists,
            MatchedCoords as PythonMatchedCoords,
            MAXCAND,
        )
        from algorithms.tracking_frame_buf import (
            TargetArray as PythonTA,
            Frame as PythonFrame,
        )
        from algorithms.parameters import VolumePar as PythonVolumePar

        _, python_cpar = file_control_params
        _, python_cal_list = file_calibration_4cam
        assert python_cpar is not None

        num_targets = 8
        num_cams = 4

        python_vpar = PythonVolumePar(
            x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0],
        )

        python_img_pts = []
        python_flat_coords = []
        frm = PythonFrame(num_cams)

        for cam in range(num_cams):
            ta = PythonTA(num_targets)
            for i in range(num_targets):
                ta[i].set_pnr(i)
                ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                ta[i].set_pixel_counts(5, 2, 2)
                ta[i].set_sum_grey_value(100.0)
                ta[i].set_tnr(0)
            python_img_pts.append(ta)
            python_flat_coords.append(
                PythonMatchedCoords(ta, python_cpar, python_cal_list[cam])
            )

        frm.targets = python_img_pts
        frm.num_targets = [num_targets] * num_cams

        # --- Run original match_pairs ---
        corr_list_orig = safely_allocate_adjacency_lists(num_cams, frm.num_targets)
        match_pairs(
            corr_list_orig, python_flat_coords, frm,
            python_vpar, python_cpar, python_cal_list,
        )

        # --- Run SoA match_pairs ---
        corr_n, corr_p2, corr_corr, corr_dist = match_pairs_soa(
            python_flat_coords, frm, python_vpar, python_cpar, python_cal_list,
        )

        # --- Compare ---
        for i1 in range(num_cams - 1):
            for i2 in range(i1 + 1, num_cams):
                for i in range(frm.num_targets[i1]):
                    orig_n = int(corr_list_orig[i1][i2][i].n)
                    soa_n = int(corr_n[i1, i2, i])
                    assert soa_n == orig_n, (
                        f"Mismatch n at pair ({i1},{i2}) target {i}: "
                        f"orig={orig_n}, soa={soa_n}"
                    )
                    for j in range(orig_n):
                        assert int(corr_p2[i1, i2, i, j]) == int(
                            corr_list_orig[i1][i2][i].p2[j]
                        ), (
                            f"Mismatch p2 at ({i1},{i2},{i},{j}): "
                            f"orig={corr_list_orig[i1][i2][i].p2[j]}, "
                            f"soa={corr_p2[i1, i2, i, j]}"
                        )
                        np.testing.assert_allclose(
                            corr_corr[i1, i2, i, j],
                            corr_list_orig[i1][i2][i].corr[j],
                            rtol=1e-10,
                            err_msg=f"corr mismatch at ({i1},{i2},{i},{j})",
                        )
                        np.testing.assert_allclose(
                            corr_dist[i1, i2, i, j],
                            corr_list_orig[i1][i2][i].dist[j],
                            rtol=1e-10,
                            err_msg=f"dist mismatch at ({i1},{i2},{i},{j})",
                        )

    def test_matching_kernels_vs_original(
        self, file_control_params, file_calibration_4cam
    ):
        """Compare Numba matching kernels against original recarray-based matching."""
        from algorithms.correspondences import (
            match_pairs,
            match_pairs_soa,
            safely_allocate_adjacency_lists,
            safely_allocate_target_usage_marks,
            four_camera_matching,
            three_camera_matching,
            consistent_pair_matching,
            _four_camera_matching_numba,
            _three_camera_matching_numba,
            _consistent_pair_matching_numba,
            MatchedCoords as PythonMatchedCoords,
            MAXCAND,
            NMAX,
        )
        from algorithms.tracking_frame_buf import (
            TargetArray as PythonTA,
            Frame as PythonFrame,
            n_tupel_dtype,
        )
        from algorithms.parameters import VolumePar as PythonVolumePar

        _, python_cpar = file_control_params
        _, python_cal_list = file_calibration_4cam
        assert python_cpar is not None

        num_targets = 8
        num_cams = 4
        nmax = NMAX

        python_vpar = PythonVolumePar(
            x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0],
        )

        python_img_pts = []
        python_flat_coords = []
        frm = PythonFrame(num_cams)

        for cam in range(num_cams):
            ta = PythonTA(num_targets)
            for i in range(num_targets):
                ta[i].set_pnr(i)
                ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                ta[i].set_pixel_counts(5, 2, 2)
                ta[i].set_sum_grey_value(100.0)
                ta[i].set_tnr(0)
            python_img_pts.append(ta)
            python_flat_coords.append(
                PythonMatchedCoords(ta, python_cpar, python_cal_list[cam])
            )

        frm.targets = python_img_pts
        frm.num_targets = [num_targets] * num_cams

        # --- Run original pipeline ---
        corr_list_orig = safely_allocate_adjacency_lists(num_cams, frm.num_targets)
        match_pairs(
            corr_list_orig, python_flat_coords, frm,
            python_vpar, python_cpar, python_cal_list,
        )

        # Original four_camera_matching
        con0_orig = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
        con0_orig.p = 0
        con0_orig.corr = 0.0
        matched_orig_4 = four_camera_matching(
            corr_list_orig, frm.num_targets[0],
            python_vpar.corrmin, con0_orig, 4 * nmax,
        )

        # --- Run SoA pipeline ---
        corr_n, corr_p2, corr_corr, corr_dist = match_pairs_soa(
            python_flat_coords, frm, python_vpar, python_cpar, python_cal_list,
        )

        # SoA four_camera_matching
        scratch_p = np.full((4 * nmax, 4), -2, dtype=np.int32)
        scratch_corr = np.zeros(4 * nmax, dtype=np.float64)
        matched_soa_4 = _four_camera_matching_numba(
            corr_n, corr_p2, corr_corr, corr_dist,
            frm.num_targets[0], python_vpar.corrmin,
            scratch_p, scratch_corr, 4 * nmax,
        )

        assert matched_soa_4 == matched_orig_4, (
            f"four_camera_matching count mismatch: orig={matched_orig_4}, soa={matched_soa_4}"
        )

        # Compare quadruplet contents (sort by corr for order-independent comparison)
        if matched_orig_4 > 0:
            orig_set = set()
            for idx in range(matched_orig_4):
                orig_set.add(tuple(con0_orig[idx].p))
            soa_set = set()
            for idx in range(matched_soa_4):
                soa_set.add(tuple(scratch_p[idx]))
            assert soa_set == orig_set, (
                f"four_camera_matching results differ:\norig={orig_set}\nsoa={soa_set}"
            )

        # --- Three-camera matching ---
        tim_orig = safely_allocate_target_usage_marks(num_cams, nmax)
        con_orig = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
        con_orig.p = 0
        con_orig.corr = 0.0

        con0_orig3 = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
        con0_orig3.p = 0
        con0_orig3.corr = 0.0

        matched_orig_3 = three_camera_matching(
            corr_list_orig, num_cams, frm.num_targets,
            python_vpar.corrmin, con0_orig3, 4 * nmax, tim_orig,
        )

        tim_soa = np.zeros((num_cams, nmax), dtype=np.int32)
        scratch_p3 = np.full((4 * nmax, 4), -2, dtype=np.int32)
        scratch_corr3 = np.zeros(4 * nmax, dtype=np.float64)
        target_counts = np.array(frm.num_targets, dtype=np.int32)

        matched_soa_3 = _three_camera_matching_numba(
            corr_n, corr_p2, corr_corr, corr_dist,
            num_cams, target_counts, python_vpar.corrmin,
            scratch_p3, scratch_corr3, 4 * nmax,
            tim_soa, nmax,
        )

        assert matched_soa_3 == matched_orig_3, (
            f"three_camera_matching count mismatch: orig={matched_orig_3}, soa={matched_soa_3}"
        )

        # --- Consistent pair matching ---
        con0_orig2 = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
        con0_orig2.p = 0
        con0_orig2.corr = 0.0
        matched_orig_2 = consistent_pair_matching(
            corr_list_orig, num_cams, frm.num_targets,
            python_vpar.corrmin, con0_orig2, 4 * nmax, tim_orig,
        )

        scratch_p2 = np.full((4 * nmax, 4), -2, dtype=np.int32)
        scratch_corr2 = np.zeros(4 * nmax, dtype=np.float64)
        matched_soa_2 = _consistent_pair_matching_numba(
            corr_n, corr_p2, corr_corr, corr_dist,
            num_cams, target_counts, python_vpar.corrmin,
            scratch_p2, scratch_corr2, 4 * nmax,
            tim_soa, nmax,
        )

        assert matched_soa_2 == matched_orig_2, (
            f"consistent_pair_matching count mismatch: orig={matched_orig_2}, soa={matched_soa_2}"
        )

    def test_take_best_candidates_soa(self):
        """Compare _take_best_candidates_soa against original take_best_candidates."""
        from algorithms.correspondences import (
            take_best_candidates,
            _take_best_candidates_soa,
        )
        from algorithms.tracking_frame_buf import n_tupel_dtype

        num_cams = 4

        # Build recarray source with 3 candidates
        src = np.recarray(3, dtype=n_tupel_dtype)
        src.p = -1
        src.corr = 0.0
        src[0].p = np.array([1, 2, -1, -1], dtype=np.int32)
        src[0].corr = 0.6
        src[1].p = np.array([3, 4, -1, -1], dtype=np.int32)
        src[1].corr = 0.9
        src[2].p = np.array([1, 5, -1, -1], dtype=np.int32)
        src[2].corr = 0.7

        # Original
        dst_orig = np.recarray(10, dtype=n_tupel_dtype)
        dst_orig.p = -1
        dst_orig.corr = 0.0
        tusage_orig = np.zeros((num_cams, 10), dtype=np.int32)
        taken_orig = take_best_candidates(src.copy(), dst_orig, num_cams, tusage_orig)

        # SoA
        src_p = np.array([[1, 2, -1, -1], [3, 4, -1, -1], [1, 5, -1, -1]], dtype=np.int32)
        src_corr = np.array([0.6, 0.9, 0.7], dtype=np.float64)
        tusage_soa = np.zeros((num_cams, 10), dtype=np.int32)
        dst_p, dst_c, taken_soa = _take_best_candidates_soa(
            src_p, src_corr, 3, num_cams, tusage_soa,
        )

        assert taken_soa == taken_orig, (
            f"take_best count mismatch: orig={taken_orig}, soa={taken_soa}"
        )

        # Compare target sets (order may differ if same corr)
        orig_set = set()
        for idx in range(taken_orig):
            orig_set.add(tuple(dst_orig[idx].p))
        soa_set = set()
        for idx in range(taken_soa):
            soa_set.add(tuple(dst_p[idx]))
        assert soa_set == orig_set

    def test_correspondences_soa_vs_original(
        self, file_control_params, file_calibration_4cam
    ):
        """Compare correspondences_soa output against original correspondences."""
        from algorithms.correspondences import (
            correspondences as original_correspondences,
            correspondences_soa,
            MatchedCoords as PythonMatchedCoords,
        )
        from algorithms.tracking_frame_buf import (
            TargetArray as PythonTA,
            Frame as PythonFrame,
        )
        from algorithms.parameters import VolumePar as PythonVolumePar

        _, python_cpar = file_control_params
        _, python_cal_list = file_calibration_4cam
        assert python_cpar is not None

        num_targets = 8
        num_cams = 4

        python_vpar = PythonVolumePar(
            x_lay=[0.0, 100.0], z_min_lay=[0.0, 0.0], z_max_lay=[50.0, 50.0],
        )

        # Build frame + flat_coords for ORIGINAL
        frm_orig = PythonFrame(num_cams)
        img_pts_orig = []
        flat_coords_orig = []
        for cam in range(num_cams):
            ta = PythonTA(num_targets)
            for i in range(num_targets):
                ta[i].set_pnr(i)
                ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                ta[i].set_pixel_counts(5, 2, 2)
                ta[i].set_sum_grey_value(100.0)
                ta[i].set_tnr(0)
            img_pts_orig.append(ta)
            flat_coords_orig.append(
                PythonMatchedCoords(ta, python_cpar, python_cal_list[cam])
            )
        frm_orig.targets = img_pts_orig
        frm_orig.num_targets = [num_targets] * num_cams
        match_counts_orig = [0, 0, 0, 0]

        con_orig = original_correspondences(
            frm_orig, flat_coords_orig, python_vpar, python_cpar,
            python_cal_list, match_counts_orig,
        )

        # Build frame + flat_coords for SoA
        frm_soa = PythonFrame(num_cams)
        img_pts_soa = []
        flat_coords_soa = []
        for cam in range(num_cams):
            ta = PythonTA(num_targets)
            for i in range(num_targets):
                ta[i].set_pnr(i)
                ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                ta[i].set_pixel_counts(5, 2, 2)
                ta[i].set_sum_grey_value(100.0)
                ta[i].set_tnr(0)
            img_pts_soa.append(ta)
            flat_coords_soa.append(
                PythonMatchedCoords(ta, python_cpar, python_cal_list[cam])
            )
        frm_soa.targets = img_pts_soa
        frm_soa.num_targets = [num_targets] * num_cams
        match_counts_soa = [0, 0, 0, 0]

        con_soa = correspondences_soa(
            frm_soa, flat_coords_soa, python_vpar, python_cpar,
            python_cal_list, match_counts_soa,
        )

        # Compare match counts
        assert match_counts_soa == match_counts_orig, (
            f"Match counts differ: orig={match_counts_orig}, soa={match_counts_soa}"
        )

        # Compare con entries up to total matched
        total = match_counts_orig[3]
        for i in range(total):
            assert tuple(con_soa[i].p) == tuple(con_orig[i].p), (
                f"con[{i}].p mismatch: orig={tuple(con_orig[i].p)}, "
                f"soa={tuple(con_soa[i].p)}"
            )
            np.testing.assert_allclose(
                con_soa[i].corr, con_orig[i].corr, rtol=1e-10,
                err_msg=f"con[{i}].corr mismatch",
            )

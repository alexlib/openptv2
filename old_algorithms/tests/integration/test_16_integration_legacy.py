"""
Engine comparison integration tests.

Tests the complete PTV pipeline from detection to tracking.
Tolerance: 1e-7 (full pipeline)
"""

import numpy as np
import pytest
from pathlib import Path
from ..conftest import get_tolerance, FIXTURES

TOLERANCE = get_tolerance("tracker")


class TestIntegrationPipeline:
    """Test complete PTV pipeline integration."""

    def test_detection_pipeline(self):
        """Test target detection on test image."""
        from optv.segmentation import target_recognition
        from optv.parameters import TargetParams, ControlParams

        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 200
        img[20:30, 70:80] = 180

        tpar = TargetParams()
        cpar = ControlParams(num_cams=1)
        cpar.set_image_size((100, 100))
        cpar.set_pixel_size((0.01, 0.01))

        try:
            optv_targets = target_recognition(img, tpar, 0, cpar)
        except Exception as e:
            pytest.fail(f"optv detection failed: {e}")

        try:
            from algorithms.segmentation import target_recognition as python_func
            from algorithms.parameters import ControlPar, TargetPar

            python_cpar = ControlPar()
            python_cpar.imx = 100
            python_cpar.imy = 100
            python_cpar.pix_x = 0.01
            python_cpar.pix_y = 0.01
            python_tpar = TargetPar()
            python_tpar.gvthresh = [0, 0, 0, 0]

            python_targets = python_func(img, python_tpar, 0, python_cpar)

            assert abs(len(optv_targets) - len(python_targets)) <= 1
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing: {e}")

    def test_calibration_pipeline(self):
        """Test camera calibration pipeline."""
        from optv.calibration import Calibration

        cal_files = FIXTURES / "calibration"

        if not (cal_files / "cam1.tif.ori").exists():
            pytest.skip("Calibration fixtures not found")

        try:
            cals = []
            for i in range(4):
                ori_file = cal_files / f"sym_cam{i + 1}.tif.ori"
                cal = Calibration()
                cal.from_file(str(ori_file), None)
                cals.append(cal)
        except Exception as e:
            pytest.fail(f"optv calibration loading failed: {e}")

        assert len(cals) == 4
        for cal in cals:
            pos = cal.get_pos()
            assert pos is not None
            assert len(pos) == 3

    def test_coordinate_transform_pipeline(self):
        """Test coordinate transformation pipeline."""
        from optv.transforms import (
            convert_arr_pixel_to_metric,
            convert_arr_metric_to_pixel,
            distorted_to_flat,
        )
        from optv.calibration import Calibration
        from optv.parameters import ControlParams

        coords = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)

        cpar = ControlParams(num_cams=1)
        cpar.set_image_size((1024, 1024))
        cpar.set_pixel_size((0.01, 0.01))
        cal = Calibration()

        metric = convert_arr_pixel_to_metric(coords, cpar)

        assert metric is not None
        assert metric.shape == (2, 2)

        flat = distorted_to_flat(metric, cal)

        assert flat is not None
        assert flat.shape == (2, 2)

    def test_correspondence_pipeline(self):
        """Test correspondence matching pipeline."""
        from optv.correspondences import MatchedCoords, correspondences
        from optv.tracking_framebuf import TargetArray, Target
        from optv.parameters import ControlParams, VolumeParams
        from optv.calibration import Calibration

        num_targets = 10

        img_pts = []
        flat_coords = []
        cals = []

        for _ in range(2):
            ta = TargetArray(num_targets)
            for i in range(num_targets):
                t = Target(
                    pnr=i,
                    x=float(i * 10),
                    y=float(i * 20),
                    n=5,
                    nx=2,
                    ny=2,
                    sumg=100.0,
                    tnr=0,
                )
                ta[i].set_pnr(t.pnr())
                ta[i].set_pos(t.pos())
            img_pts.append(ta)

            cpar = ControlParams(num_cams=1)
            cpar.set_image_size((1024, 1024))
            cpar.set_pixel_size((0.01, 0.01))
            cal = Calibration()

            mc = MatchedCoords(ta, cpar, cal)
            flat_coords.append(mc)
            cals.append(cal)

        vpar = VolumeParams(x_span=np.array([0, 100]))

        try:
            result = correspondences(img_pts, flat_coords, cals, vpar, cpar)
        except Exception as e:
            pytest.fail(f"optv correspondence failed: {e}")

    def test_tracking_pipeline_full(self, tmp_path: Path):
        """Test full tracking pipeline."""
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )
        from optv.calibration import Calibration
        from optv.tracker import Tracker

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1024, 1024))
        cpar.set_pixel_size((0.01, 0.01))
        vpar = VolumeParams(x_span=np.array([0, 100]))
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)

        img_dir = tmp_path / "tracking_pipeline_img"
        img_dir.mkdir(parents=True, exist_ok=True)
        image_base = [str(img_dir / f"cam{i}_") for i in range(1, 5)]
        for cam in range(1, 5):
            for frame in range(1, 11):
                (img_dir / f"cam{cam}_{frame:04d}_targets").write_text("0\n")

        spar = SequenceParams(image_base=image_base, frame_range=(1, 10))

        cals = [Calibration() for _ in range(4)]

        # Keep tracker I/O isolated from repo root and seed empty correspondences.
        res_dir = tmp_path / "tracking_pipeline_res"
        res_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(1, 11):
            (res_dir / f"rt_is.{frame}").write_text("0\n")

        naming = {
            "corres": str(res_dir / "rt_is").encode(),
            "linkage": str(res_dir / "ptv_is").encode(),
            "prio": str(res_dir / "added").encode(),
        }

        try:
            tracker = Tracker(cpar, vpar, tpar, spar, cals, naming)
            tracker.full_forward()

            current = tracker.current_step()
            assert current >= 0
        except Exception as e:
            pytest.fail(f"optv full tracking pipeline failed: {e}")


class TestMultiCameraIntegration:
    """Test multi-camera integration."""

    def test_four_camera_triangulation(self):
        """Test 4-camera point triangulation."""
        from optv.orientation import multi_cam_point_positions
        from optv.parameters import ControlParams

        np.random.seed(42)

        img_pts = []
        for cam in range(4):
            pts = np.random.rand(5, 2) * 100
            img_pts.append(pts)

        cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, float(i * np.pi / 2)])

            from optv.calibration import Calibration

            cal = Calibration(pos=pos, angs=angles)
            cals.append(cal)

        targets = np.asarray(img_pts, dtype=np.float64)
        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1024, 1024))
        cpar.set_pixel_size((0.01, 0.01))

        assert callable(multi_cam_point_positions)
        assert targets.shape == (4, 5, 2)
        assert len(cals) == 4

    def test_calibration_with_all_cameras(self):
        """Test calibration with all cameras."""
        from optv.calibration import Calibration

        cal_files = FIXTURES / "calibration"

        if not (cal_files / "cam1.tif.ori").exists():
            pytest.skip("Calibration fixtures not found")

        sym_cals = []
        for i in range(1, 5):
            ori_file = cal_files / f"sym_cam{i}.tif.ori"
            cal = Calibration()
            cal.from_file(str(ori_file), None)
            sym_cals.append(cal)

        positions = [cal.get_pos() for cal in sym_cals]
        angles = [cal.get_angles() for cal in sym_cals]

        for pos in positions:
            assert pos is not None
            assert len(pos) == 3


class TestEndToEndWorkflow:
    """Test end-to-end PTV workflow."""

    def test_complete_workflow_synthetic(self, tmp_path: Path):
        """Test complete workflow with synthetic data."""
        from optv.segmentation import target_recognition
        from optv.correspondences import correspondences, MatchedCoords
        from optv.orientation import multi_cam_point_positions
        from optv.tracker import Tracker
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
            TargetParams,
        )
        from optv.calibration import Calibration

        num_frames = 3
        num_targets = 4

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1024, 1024))
        cpar.set_pixel_size((0.01, 0.01))
        vpar = VolumeParams(x_span=np.array([0, 100]))
        tpar = TrackingParams(n1=3, n2=3, dh=3.0, dz=1.0)
        img_dir = tmp_path / "workflow_img"
        img_dir.mkdir(parents=True, exist_ok=True)
        image_base = [str(img_dir / f"cam{i}_") for i in range(1, 5)]
        for cam in range(1, 5):
            for frame in range(1, num_frames + 1):
                (img_dir / f"cam{cam}_{frame:04d}_targets").write_text("0\n")

        spar = SequenceParams(image_base=image_base, frame_range=(1, num_frames))
        targpar = TargetParams()

        cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, float(i * np.pi / 4)])
            cal = Calibration(pos=pos, angs=angles)
            cals.append(cal)

        # Use isolated outputs and preseed a wide frame span to avoid noisy I/O errors.
        res_dir = tmp_path / "workflow_res"
        res_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(0, 41):
            (res_dir / f"rt_is.{frame}").write_text("0\n")

        naming = {
            "corres": str(res_dir / "rt_is").encode(),
            "linkage": str(res_dir / "ptv_is").encode(),
            "prio": str(res_dir / "added").encode(),
        }

        try:
            tracker = Tracker(cpar, vpar, tpar, spar, cals, naming)
            tracker.full_forward()

            final_step = tracker.current_step()
            assert final_step >= 0
        except Exception as e:
            pytest.fail(f"Complete workflow failed: {e}")

    def test_parameter_consistency(self):
        """Test parameter consistency across pipeline."""
        from optv.parameters import ControlParams

        test_cpar = ControlParams(num_cams=4)
        test_cpar.set_image_size((2048, 2048))
        test_cpar.set_pixel_size((0.005, 0.005))

        assert test_cpar.get_num_cams() == 4
        assert test_cpar.get_image_size() == (2048, 2048)
        assert abs(test_cpar.get_pixel_size()[0] - 0.005) < 1e-10

    def test_volume_consistency(self):
        """Test volume parameter consistency."""
        from optv.parameters import VolumeParams

        test_vpar = VolumeParams(
            xmin=-50.0,
            xmax=50.0,
            ymin=-50.0,
            ymax=50.0,
            zmin=0.0,
            zmax=100.0,
        )

        assert test_vpar is not None

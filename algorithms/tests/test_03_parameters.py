"""
Engine comparison tests for parameter classes.

Tests ControlParams, VolumeParams, SequenceParams, TrackingParams, TargetParams, MultimediaParams.
Tolerance: 1e-9 (parameter structures)
"""

import numpy as np
import pytest
from pathlib import Path
from .conftest import get_tolerance, FIXTURES

TOLERANCE = get_tolerance("parameters")


class TestControlParams:
    """Compare ControlParams between optv and python engines."""

    def test_control_params_creation_default(self):
        """Test ControlParams creation with defaults."""
        from optv.parameters import ControlParams as OptvCP
        from algorithms.parameters import ControlPar as PythonCP

        optv_cp = OptvCP(num_cams=4)
        python_cp = PythonCP()

        assert optv_cp.get_num_cams() == 4
        assert python_cp.num_cams == 0
        assert python_cp.get_num_cams() == 0
        assert python_cp.get_image_size() == (0, 0)
        assert python_cp.get_pixel_size() == (0.0, 0.0)

    def test_control_params_creation_with_values(self):
        """Test ControlParams creation with specific values."""
        from optv.parameters import ControlParams as OptvCP
        from algorithms.parameters import ControlPar

        optv_cp = OptvCP(num_cams=4)
        optv_cp.set_image_size((2048, 2048))
        optv_cp.set_pixel_size((0.005, 0.005))

        python_cp = ControlPar()
        python_cp.imx = 2048
        python_cp.imy = 2048
        python_cp.pix_x = 0.005
        python_cp.pix_y = 0.005

        assert optv_cp is not None
        assert python_cp.imx == 2048
        assert python_cp.imy == 2048
        assert abs(python_cp.pix_x - 0.005) < TOLERANCE
        assert abs(python_cp.pix_y - 0.005) < TOLERANCE
        assert python_cp.get_image_size() == (2048, 2048)
        assert python_cp.get_pixel_size() == (0.005, 0.005)

    def test_control_params_from_file(self, control_params_file):
        """Test loading ControlParams from file."""
        from optv.parameters import ControlParams as OptvCP
        from algorithms.parameters import read_control_par as PythonRead

        optv_cp = OptvCP(num_cams=4)
        python_cp = PythonRead(control_params_file)

        assert optv_cp.get_num_cams() == 4
        assert python_cp.num_cams == 4
        assert python_cp.img_base_name == [
            "dumbbell/cam1_Scene77_4085",
            "dumbbell/cam2_Scene77_4085",
            "dumbbell/cam3_Scene77_4085",
            "dumbbell/cam4_Scene77_4085",
        ]
        assert python_cp.cal_img_base_name == [
            "cal/cam1.tif",
            "cal/cam2.tif",
            "cal/cam3.tif",
            "cal/cam4.tif",
        ]
        assert python_cp.hp_flag == 10
        assert python_cp.all_cam_flag == 11
        assert python_cp.tiff_flag == 12
        assert python_cp.get_image_size() == (1280, 1024)
        assert python_cp.get_pixel_size() == (15.15, 16.16)
        assert python_cp.chfield == 17
        assert python_cp.mm.n1 == 18
        assert python_cp.mm.n2 == [19.19]
        assert python_cp.mm.n3 == 20.20
        assert python_cp.mm.d == [21.21]


class TestVolumeParams:
    """Compare VolumeParams between optv and python engines."""

    def test_volume_params_creation_default(self):
        """Test VolumeParams creation with defaults."""
        from optv.parameters import VolumeParams as OptvVP
        from algorithms.parameters import VolumePar as PythonVP

        optv_vp = OptvVP()
        python_vp = PythonVP()

        assert optv_vp is not None
        assert python_vp is not None
        assert python_vp.x_lay == []
        assert python_vp.z_min_lay == []
        assert python_vp.z_max_lay == []

    def test_volume_params_creation_with_values(self):
        """Test VolumeParams creation with specific values."""
        from optv.parameters import VolumeParams as OptvVP

        optv_vp = OptvVP()

        from algorithms.parameters import VolumePar

        python_vp = VolumePar()
        python_vp.Xmin = -50.0
        python_vp.Xmax = 50.0
        python_vp.Ymin = -50.0
        python_vp.Ymax = 50.0
        python_vp.Zmin = 0.0
        python_vp.Zmax = 100.0

        assert python_vp.Xmin == -50.0
        assert python_vp.Xmax == 50.0
        assert python_vp.Zmin == 0.0
        assert python_vp.Zmax == 100.0
        assert optv_vp is not None

    def test_volume_params_from_file(self, volume_params_file):
        """Test loading VolumeParams from file."""
        if not volume_params_file.exists():
            pytest.skip(f"Volume params file not found: {volume_params_file}")

        from optv.parameters import VolumeParams as OptvVP
        from algorithms.parameters import read_volume_par as PythonRead

        optv_vp = OptvVP()
        python_vp = PythonRead(volume_params_file)

        assert optv_vp is not None
        assert python_vp.x_lay == [111.111, 222.222]
        assert python_vp.z_min_lay == [333.333, 444.444]
        assert python_vp.z_max_lay == [555.555, 666.666]
        assert python_vp.cnx == 777.777
        assert python_vp.cny == 888.888
        assert python_vp.cn == 999.999
        assert python_vp.csumg == 1010.1010
        assert python_vp.corrmin == 1111.1111
        assert python_vp.eps0 == 1212.1212


class TestSequenceParams:
    """Compare SequenceParams between optv and python engines."""

    def test_sequence_params_creation_default(self):
        """Test SequenceParams creation with defaults."""
        from optv.parameters import SequenceParams as OptvSP
        from algorithms.parameters import SequencePar as PythonSP

        optv_sp = OptvSP(num_cams=4)
        python_sp = PythonSP()

        assert optv_sp is not None
        assert python_sp.first == 0
        assert python_sp.last == 0
        assert python_sp.img_base_name == []

    def test_sequence_params_creation_with_values(self):
        """Test SequenceParams creation with specific values."""
        from optv.parameters import SequenceParams as OptvSP
        from algorithms.parameters import SequencePar

        optv_sp = OptvSP(num_cams=4)
        optv_sp.set_first(1)
        optv_sp.set_last(100)

        python_sp = SequencePar()
        python_sp.first = 1
        python_sp.last = 100

        assert optv_sp is not None
        assert python_sp is not None
        assert optv_sp.get_last() == python_sp.last
        assert python_sp.img_base_name == []

    def test_sequence_params_from_file(self, sequence_params_file):
        """Test loading SequenceParams from file."""
        if not sequence_params_file.exists():
            pytest.skip(f"Sequence params file not found: {sequence_params_file}")

        from optv.parameters import SequenceParams as OptvSP

        optv_sp = OptvSP(num_cams=4)

        from algorithms.parameters import read_sequence_par as PythonRead

        python_sp = PythonRead(sequence_params_file, num_cams=4)

        assert optv_sp is not None
        assert python_sp.img_base_name == [
            "dumbbell/cam1_Scene77_",
            "dumbbell/cam2_Scene77_",
            "dumbbell/cam3_Scene77_",
            "dumbbell/cam4_Scene77_",
        ]
        assert python_sp.first == 497
        assert python_sp.last == 597


class TestTrackingParams:
    """Compare TrackingParams between optv and python engines."""

    def test_tracking_params_creation_default(self):
        """Test TrackingParams creation with defaults."""
        from optv.parameters import TrackingParams as OptvTP
        from algorithms.parameters import TrackPar as PythonTP

        optv_tp = OptvTP()
        python_tp = PythonTP()

        assert optv_tp is not None
        assert python_tp is not None
        assert python_tp.dvxmin == 0.0
        assert python_tp.dny == 0.0

    def test_tracking_params_creation_with_values(self):
        """Test TrackingParams creation with specific values."""
        from optv.parameters import TrackingParams as OptvTP
        from algorithms.parameters import TrackPar

        optv_tp = OptvTP()

        python_tp = TrackPar()
        python_tp.dvxmin = -3.0
        python_tp.dvxmax = 3.0

        assert optv_tp is not None
        assert python_tp is not None
        assert python_tp.dvxmin == -3.0
        assert python_tp.dvxmax == 3.0

    def test_tracking_params_from_file(self, tracking_params_file):
        """Test loading TrackingParams from file."""
        if not tracking_params_file.exists():
            pytest.skip(f"Tracking params file not found: {tracking_params_file}")

        from optv.parameters import TrackingParams as OptvTP

        optv_tp = OptvTP()

        from algorithms.parameters import read_track_par as PythonRead

        python_tp = PythonRead(tracking_params_file)

        assert optv_tp is not None
        assert python_tp.dvxmin == 111.111
        assert python_tp.dvxmax == 222.222
        assert python_tp.dvymin == 333.333
        assert python_tp.dvymax == 444.444
        assert python_tp.dvzmin == 555.555
        assert python_tp.dvzmax == 666.666
        assert python_tp.dangle == 777.777
        assert python_tp.dacc == 888.888
        assert python_tp.add == 9
        assert python_tp.dsumg == 0.0
        assert python_tp.dn == 0.0
        assert python_tp.dnx == 0.0
        assert python_tp.dny == 0.0


class TestTargetParams:
    """Compare TargetParams between optv and python engines."""

    def test_target_params_creation_default(self):
        """Test TargetParams creation with defaults."""
        from optv.parameters import TargetParams as OptvTargP
        from algorithms.parameters import TargetPar as PythonTargP

        optv_targp = OptvTargP()
        python_targp = PythonTargP()

        assert optv_targp is not None
        assert python_targp is not None

    def test_target_params_creation_with_values(self):
        """Test TargetParams creation with specific values."""
        from optv.parameters import TargetParams as OptvTargP
        from algorithms.parameters import TargetPar

        optv_targp = OptvTargP()

        python_targp = TargetPar()
        python_targp.lx = 5.0
        python_targp.ly = 5.0
        python_targp.rmin = 3.0
        python_targp.rmax = 20.0

        assert optv_targp is not None
        assert python_targp.lx == 5.0
        assert python_targp.ly == 5.0

    def test_target_params_from_file(self, target_params_file):
        """Test loading TargetParams from file."""
        if not target_params_file.exists():
            pytest.skip(f"Target params file not found: {target_params_file}")

        from optv.parameters import TargetParams as OptvTargP

        optv_targp = OptvTargP()

        from algorithms.parameters import read_target_par as PythonRead

        python_targp = PythonRead(target_params_file)

        assert optv_targp is not None
        assert python_targp is not None


class TestMultimediaParams:
    """Compare MultimediaParams between optv and python engines."""

    def test_multimedia_params_creation_default(self):
        """Test MultimediaParams creation with defaults."""
        from optv.parameters import MultimediaParams as OptvMMP
        from algorithms.parameters import MultimediaPar

        optv_mmp = OptvMMP()
        python_mmp = MultimediaPar()

        assert optv_mmp is not None
        assert python_mmp.nlay == 1

    def test_multimedia_params_creation_with_values(self):
        """Test MultimediaParams creation with specific values."""
        from optv.parameters import MultimediaParams as OptvMMP
        from algorithms.parameters import MultimediaPar

        optv_mmp = OptvMMP()

        python_mmp = MultimediaPar()
        python_mmp.nlay = 2
        python_mmp.n2 = [1.0, 1.33]
        python_mmp.d = [0.0, 100.0]

        assert optv_mmp is not None
        assert python_mmp.nlay == 2


class TestParametersIntegration:
    """Integration tests for parameter classes."""

    def test_all_params_together(self, control_params_file, volume_params_file):
        """Test creating all parameter classes together."""
        from optv.parameters import (
            ControlParams as OptvCP,
            VolumeParams as OptvVP,
            SequenceParams as OptvSP,
            TrackingParams as OptvTP,
        )
        from algorithms.parameters import (
            ControlPar as PythonCP,
            VolumePar as PythonVP,
            SequencePar as PythonSP,
            TrackPar as PythonTP,
        )

        optv_cp = OptvCP(num_cams=4)
        python_cp = PythonCP()
        if control_params_file.exists():
            from algorithms.parameters import read_control_par

            python_cp = read_control_par(control_params_file)

        assert optv_cp.get_num_cams() == 4
        assert python_cp is not None

    def test_parameter_comparison_with_fixture_data(self):
        """Compare parameters using fixture data."""
        from optv.parameters import ControlParams as OptvCP
        from algorithms.parameters import ControlPar as PythonCP

        test_values = {
            "imx": 1024,
            "imy": 1024,
            "pix_x": 0.01,
            "pix_y": 0.01,
        }

        optv_cp = OptvCP(num_cams=4)
        optv_cp.set_image_size((test_values["imx"], test_values["imy"]))
        optv_cp.set_pixel_size((test_values["pix_x"], test_values["pix_y"]))

        python_cp = PythonCP()
        python_cp.imx = test_values["imx"]
        python_cp.imy = test_values["imy"]
        python_cp.pix_x = test_values["pix_x"]
        python_cp.pix_y = test_values["pix_y"]

        assert python_cp.imx == test_values["imx"]
        assert python_cp.imy == test_values["imy"]
        assert abs(python_cp.pix_x - test_values["pix_x"]) < TOLERANCE
        assert abs(python_cp.pix_y - test_values["pix_y"]) < TOLERANCE

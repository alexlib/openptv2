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

        optv_cp = OptvCP()
        python_cp = PythonCP()

        assert optv_cp._control_par[0].imx == python_cp.imx
        assert optv_cp._control_par[0].imy == python_cp.imy

    def test_control_params_creation_with_values(self):
        """Test ControlParams creation with specific values."""
        from optv.parameters import ControlParams as OptvCP

        optv_cp = OptvCP(
            imx=2048,
            imy=2048,
            pix_x=0.005,
            pix_y=0.005,
            sw_x1=0.0,
            sw_y1=0.0,
            sw_x2=2048.0,
            sw_y2=2048.0,
            mm_nlay=2,
            eta1=1.33,
        )

        from algorithms.parameters import ControlPar as PythonCP

        python_cp = ControlPar()
        python_cp.imx = 2048
        python_cp.imy = 2048
        python_cp.pix_x = 0.005
        python_cp.pix_y = 0.005

        assert optv_cp._control_par[0].imx == python_cp.imx
        assert optv_cp._control_par[0].imy == python_cp.imy

    def test_control_params_from_file(self, control_params_file):
        """Test loading ControlParams from file."""
        from optv.parameters import ControlParams as OptvCP

        optv_cp = OptvCP(str(control_params_file))

        from algorithms.parameters import read_control_par as PythonRead

        python_cp = PythonRead(control_params_file)

        assert optv_cp._control_par[0].imx == python_cp.imx
        assert optv_cp._control_par[0].imy == python_cp.imy


class TestVolumeParams:
    """Compare VolumeParams between optv and python engines."""

    def test_volume_params_creation_default(self):
        """Test VolumeParams creation with defaults."""
        from optv.parameters import VolumeParams as OptvVP
        from algorithms.parameters import VolumePar as PythonVP

        optv_vp = OptvVP()
        python_vp = PythonVP()

        assert optv_vp._volume_par[0].Xmin == python_vp.Xmin
        assert optv_vp._volume_par[0].Xmax == python_vp.Xmax

    def test_volume_params_creation_with_values(self):
        """Test VolumeParams creation with specific values."""
        from optv.parameters import VolumeParams as OptvVP

        optv_vp = OptvVP(
            xmin=-50.0,
            xmax=50.0,
            ymin=-50.0,
            ymax=50.0,
            zmin=0.0,
            zmax=100.0,
        )

        from algorithms.parameters import VolumePar as PythonVP

        python_vp = VolumePar()
        python_vp.Xmin = -50.0
        python_vp.Xmax = 50.0
        python_vp.Ymin = -50.0
        python_vp.Ymax = 50.0
        python_vp.Zmin = 0.0
        python_vp.Zmax = 100.0

        assert optv_vp._volume_par[0].Xmin == python_vp.Xmin
        assert optv_vp._volume_par[0].Xmax == python_vp.Xmax
        assert optv_vp._volume_par[0].Zmin == python_vp.Zmin
        assert optv_vp._volume_par[0].Zmax == python_vp.Zmax

    def test_volume_params_from_file(self, volume_params_file):
        """Test loading VolumeParams from file."""
        if not volume_params_file.exists():
            pytest.skip(f"Volume params file not found: {volume_params_file}")

        from optv.parameters import VolumeParams as OptvVP

        optv_vp = OptvVP(str(volume_params_file))

        from algorithms.parameters import read_volume_par as PythonRead

        python_vp = PythonRead(volume_params_file)

        assert abs(optv_vp._volume_par[0].Xmin - python_vp.Xmin) < TOLERANCE
        assert abs(optv_vp._volume_par[0].Xmax - python_vp.Xmax) < TOLERANCE


class TestSequenceParams:
    """Compare SequenceParams between optv and python engines."""

    def test_sequence_params_creation_default(self):
        """Test SequenceParams creation with defaults."""
        from optv.parameters import SequenceParams as OptvSP
        from algorithms.parameters import SequencePar as PythonSP

        optv_sp = OptvSP()
        python_sp = PythonSP()

        assert optv_sp._sequence_par[0].first == python_sp.first
        assert optv_sp._sequence_par[0].last == python_sp.last

    def test_sequence_params_creation_with_values(self):
        """Test SequenceParams creation with specific values."""
        from optv.parameters import SequenceParams as OptvSP

        optv_sp = OptvSP(
            first=1,
            last=100,
            dStep=1,
            name=b"test",
        )

        from algorithms.parameters import SequencePar as PythonSP

        python_sp = SequencePar()
        python_sp.first = 1
        python_sp.last = 100
        python_sp.dStep = 1
        python_sp.name = "test"

        assert optv_sp._sequence_par[0].first == python_sp.first
        assert optv_sp._sequence_par[0].last == python_sp.last

    def test_sequence_params_from_file(self, sequence_params_file):
        """Test loading SequenceParams from file."""
        if not sequence_params_file.exists():
            pytest.skip(f"Sequence params file not found: {sequence_params_file}")

        from optv.parameters import SequenceParams as OptvSP

        optv_sp = OptvSP(str(sequence_params_file))

        from algorithms.parameters import read_sequence_par as PythonRead

        python_sp = PythonRead(sequence_params_file, num_cams=4)

        assert optv_sp._sequence_par[0].first == python_sp.first


class TestTrackingParams:
    """Compare TrackingParams between optv and python engines."""

    def test_tracking_params_creation_default(self):
        """Test TrackingParams creation with defaults."""
        from optv.parameters import TrackingParams as OptvTP
        from algorithms.parameters import TrackPar as PythonTP

        optv_tp = OptvTP()
        python_tp = PythonTP()

        assert optv_tp._track_par[0].n1 == python_tp.n1
        assert optv_tp._track_par[0].n2 == python_tp.n2

    def test_tracking_params_creation_with_values(self):
        """Test TrackingParams creation with specific values."""
        from optv.parameters import TrackingParams as OptvTP

        optv_tp = OptvTP(
            n1=3,
            n2=3,
            nxmax=16,
            dz=1.0,
            dh=3.0,
            k=2.0,
            rho=0.95,
            dt=3,
            dy=2.0,
        )

        from algorithms.parameters import TrackPar as PythonTP

        python_tp = TrackPar()
        python_tp.n1 = 3
        python_tp.n2 = 3
        python_tp.nxmax = 16
        python_tp.dz = 1.0
        python_tp.dh = 3.0
        python_tp.k = 2.0
        python_tp.rho = 0.95
        python_tp.dt = 3
        python_tp.dy = 2.0

        assert optv_tp._track_par[0].n1 == python_tp.n1
        assert optv_tp._track_par[0].n2 == python_tp.n2
        assert abs(optv_tp._track_par[0].dz - python_tp.dz) < TOLERANCE

    def test_tracking_params_from_file(self, tracking_params_file):
        """Test loading TrackingParams from file."""
        if not tracking_params_file.exists():
            pytest.skip(f"Tracking params file not found: {tracking_params_file}")

        from optv.parameters import TrackingParams as OptvTP

        optv_tp = OptvTP(str(tracking_params_file))

        from algorithms.parameters import read_track_par as PythonRead

        python_tp = PythonRead(tracking_params_file)

        assert optv_tp._track_par[0].n1 == python_tp.n1


class TestTargetParams:
    """Compare TargetParams between optv and python engines."""

    def test_target_params_creation_default(self):
        """Test TargetParams creation with defaults."""
        from optv.parameters import TargetParams as OptvTargP
        from algorithms.parameters import TargetPar as PythonTargP

        optv_targp = OptvTargP()
        python_targp = PythonTargP()

        assert optv_targp._targ_par[0].lx == python_targp.lx
        assert optv_targp._targ_par[0].ly == python_targp.ly

    def test_target_params_creation_with_values(self):
        """Test TargetParams creation with specific values."""
        from optv.parameters import TargetParams as OptvTargP

        optv_targp = OptvTargP(
            lx=5.0,
            ly=5.0,
            rmin=3.0,
            rmax=20.0,
            cnt=50,
            cal=[50.0, 200.0],
        )

        from algorithms.parameters import TargetPar as PythonTargP

        python_targp = TargetPar()
        python_targp.lx = 5.0
        python_targp.ly = 5.0
        python_targp.rmin = 3.0
        python_targp.rmax = 20.0

        assert optv_targp._targ_par[0].lx == python_targp.lx
        assert optv_targp._targ_par[0].ly == python_targp.ly

    def test_target_params_from_file(self, target_params_file):
        """Test loading TargetParams from file."""
        if not target_params_file.exists():
            pytest.skip(f"Target params file not found: {target_params_file}")

        from optv.parameters import TargetParams as OptvTargP

        optv_targp = OptvTargP(str(target_params_file))

        from algorithms.parameters import read_target_par as PythonRead

        python_targp = PythonRead(target_params_file)

        assert abs(optv_targp._targ_par[0].lx - python_targp.lx) < TOLERANCE


class TestMultimediaParams:
    """Compare MultimediaParams between optv and python engines."""

    def test_multimedia_params_creation_default(self):
        """Test MultimediaParams creation with defaults."""
        from optv.parameters import MultimediaParams as OptvMMP
        from algorithms.parameters import MultimediaPar as PythonMMP

        optv_mmp = OptvMMP()
        python_mmp = MultimediaPar()

        assert optv_mmp._mm_np[0].nlay == python_mmp.nlay
        assert optv_mmp._mm_np[0].eta[0] == python_mmp.eta[0]

    def test_multimedia_params_creation_with_values(self):
        """Test MultimediaParams creation with specific values."""
        from optv.parameters import MultimediaParams as OptvMMP

        optv_mmp = OptvMMP(
            nlay=2,
            eta=[1.0, 1.33, 1.0],
            z=[0.0, 10.0, 100.0],
        )

        from algorithms.parameters import MultimediaPar as PythonMMP

        python_mmp = MultimediaPar()
        python_mmp.nlay = 2
        python_mmp.eta[0] = 1.0
        python_mmp.eta[1] = 1.33
        python_mmp.eta[2] = 1.0

        assert optv_mmp._mm_np[0].nlay == python_mmp.nlay
        assert abs(optv_mmp._mm_np[0].eta[1] - python_mmp.eta[1]) < TOLERANCE


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

        optv_cp = (
            OptvCP(str(control_params_file))
            if control_params_file.exists()
            else OptvCP()
        )
        python_cp = PythonCP()
        if control_params_file.exists():
            from algorithms.parameters import read_control_par

            python_cp = read_control_par(control_params_file)

        assert optv_cp._control_par[0].imx == python_cp.imx
        assert optv_cp._control_par[0].imy == python_cp.imy

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

        optv_cp = OptvCP(
            imx=test_values["imx"],
            imy=test_values["imy"],
            pix_x=test_values["pix_x"],
            pix_y=test_values["pix_y"],
        )

        python_cp = PythonCP()
        python_cp.imx = test_values["imx"]
        python_cp.imy = test_values["imy"]
        python_cp.pix_x = test_values["pix_x"]
        python_cp.pix_y = test_values["pix_y"]

        assert optv_cp._control_par[0].imx == python_cp.imx
        assert optv_cp._control_par[0].imy == python_cp.imy
        assert abs(optv_cp._control_par[0].pix_x - python_cp.pix_x) < TOLERANCE
        assert abs(optv_cp._control_par[0].pix_y - python_cp.pix_y) < TOLERANCE

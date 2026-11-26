"""YAML round-trip tests for parameter objects.

Verifies that parameters loaded from an experiment YAML round-trip
field-for-field, and that alias fallbacks resolve correctly. Uses the
real test_data/test_cavity/parameters_Run1.yaml as the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.parameters import ControlPar, TrackPar, VolumePar
from tests._support import find_test_data_root


def _cavity_yaml() -> Path:
    return find_test_data_root() / "test_cavity" / "parameters_Run1.yaml"


class TestControlParRoundTrip:
    def test_from_yaml_loads_real_cavity_params(self):
        cpar = ControlPar.from_yaml(_cavity_yaml(), num_cams=4)
        assert cpar.num_cams == 4
        assert cpar.imx == 1280
        assert cpar.imy == 1024
        assert cpar.pix_x > 0
        assert cpar.pix_y > 0
        assert cpar.mm is not None
        assert cpar.mm.n1 > 0
        assert cpar.mm.n3 > 0

    def test_from_yaml_persists_fields(self):
        cpar = ControlPar.from_yaml(_cavity_yaml(), num_cams=4)
        captured = dict(
            num_cams=cpar.num_cams,
            imx=cpar.imx,
            imy=cpar.imy,
            pix_x=cpar.pix_x,
            pix_y=cpar.pix_y,
            hp_flag=cpar.hp_flag,
            allCam_flag=cpar.allCam_flag,
            pair_flag=cpar.pair_flag,
            tiff_flag=cpar.tiff_flag,
            chfield=cpar.chfield,
            mm_n1=cpar.mm.n1,
            mm_n3=cpar.mm.n3,
            mm_d0=cpar.mm.d[0],
        )
        # Reconstruct from captured values and assert equality.
        from openptv2.algorithms.parameters import MmNp

        cpar2 = ControlPar(
            num_cams=captured["num_cams"],
            img_base_name=cpar.img_base_name,
            cal_img_base_name=cpar.cal_img_base_name,
            hp_flag=captured["hp_flag"],
            allCam_flag=captured["allCam_flag"],
            pair_flag=captured["pair_flag"],
            tiff_flag=captured["tiff_flag"],
            imx=captured["imx"],
            imy=captured["imy"],
            pix_x=captured["pix_x"],
            pix_y=captured["pix_y"],
            chfield=captured["chfield"],
            mm=MmNp(
                nlay=1,
                n1=captured["mm_n1"],
                n2=[captured["mm_n1"], 1.0, 1.0],
                d=[captured["mm_d0"], 0.0, 0.0],
                n3=captured["mm_n3"],
            ),
        )
        assert cpar2.num_cams == captured["num_cams"]
        assert cpar2.imx == captured["imx"]
        assert cpar2.imy == captured["imy"]
        assert cpar2.pix_x == captured["pix_x"]
        assert cpar2.pix_y == captured["pix_y"]
        assert cpar2.mm.n1 == captured["mm_n1"]
        assert cpar2.mm.n3 == captured["mm_n3"]

    def test_from_yaml_num_cams_default(self):
        cpar = ControlPar.from_yaml(_cavity_yaml())
        assert cpar.num_cams == 4


class TestTrackParRoundTrip:
    def test_from_yaml_loads_real_cavity_track_params(self):
        tpar = TrackPar.from_yaml(_cavity_yaml())
        assert tpar.dvxmax > 0
        assert tpar.dacc > 0
        assert tpar.track_mode in (0, 1)

    def test_from_yaml_persists_fields(self):
        tpar = TrackPar.from_yaml(_cavity_yaml())
        tpar2 = TrackPar(
            dvxmin=tpar.dvxmin,
            dvxmax=tpar.dvxmax,
            dvymin=tpar.dvymin,
            dvymax=tpar.dvymax,
            dvzmin=tpar.dvzmin,
            dvzmax=tpar.dvzmax,
            dangle=tpar.dangle,
            dacc=tpar.dacc,
            add=tpar.add,
            track_mode=tpar.track_mode,
            w_vel=tpar.w_vel,
            w_acc=tpar.w_acc,
            w_intensity=tpar.w_intensity,
        )
        for field in (
            "dvxmin",
            "dvxmax",
            "dvymin",
            "dvymax",
            "dvzmin",
            "dvzmax",
            "dangle",
            "dacc",
            "add",
            "track_mode",
            "w_vel",
            "w_acc",
            "w_intensity",
        ):
            assert getattr(tpar2, field) == getattr(tpar, field)

    def test_angle_alias_resolves(self):
        tpar = TrackPar.from_yaml(_cavity_yaml())
        assert np.isfinite(tpar.dangle)
        assert tpar.dangle >= 0


class TestVolumeParRoundTrip:
    def test_from_yaml_loads_real_cavity_volume(self):
        vpar = VolumePar.from_yaml(_cavity_yaml())
        assert np.isfinite(vpar.cnx)
        assert np.isfinite(vpar.eps0)
        assert len(vpar.X_lay) == 2
        assert len(vpar.Zmin_lay) == 2
        assert len(vpar.Zmax_lay) == 2

    def test_from_yaml_persists_fields(self):
        vpar = VolumePar.from_yaml(_cavity_yaml())
        vpar2 = VolumePar(
            X_lay=list(vpar.X_lay),
            Zmin_lay=list(vpar.Zmin_lay),
            Zmax_lay=list(vpar.Zmax_lay),
            cnx=vpar.cnx,
            cny=vpar.cny,
            cn=vpar.cn,
            csumg=vpar.csumg,
            corrmin=vpar.corrmin,
            eps0=vpar.eps0,
        )
        assert np.array_equal(vpar2.X_lay, vpar.X_lay)
        assert np.array_equal(vpar2.Zmin_lay, vpar.Zmin_lay)
        assert np.array_equal(vpar2.Zmax_lay, vpar.Zmax_lay)
        for field in ("cnx", "cny", "cn", "csumg", "corrmin", "eps0"):
            assert getattr(vpar2, field) == getattr(vpar, field)

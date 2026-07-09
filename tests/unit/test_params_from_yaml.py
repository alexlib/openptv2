"""Parity: <Param>.from_yaml must equal <Param>.from_file.

openptv2 is YAML-only at runtime; parameter objects are built from the
experiment YAML via ``from_yaml``. These guard that the YAML constructors
produce the same values as the legacy ``.par`` readers (which are kept only
for the converter), so migrating tests from .par to YAML changes nothing.

Uses test_data/parameters (full .par set) + test_data/parameters.yaml
(its converted YAML).
"""

import numpy as np

from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
)

PAR = "test_data/parameters"
YAML = "test_data/parameters.yaml"


def test_control_par_from_yaml_matches_from_file():
    f = ControlPar.from_file(f"{PAR}/ptv.par")
    y = ControlPar.from_yaml(YAML)
    assert (y.num_cams, y.imx, y.imy, y.chfield) == (f.num_cams, f.imx, f.imy, f.chfield)
    assert (y.pix_x, y.pix_y, y.hp_flag) == (f.pix_x, f.pix_y, f.hp_flag)
    assert (y.mm.n1, y.mm.n2[0], y.mm.n3, y.mm.d[0]) == (
        f.mm.n1, f.mm.n2[0], f.mm.n3, f.mm.d[0],
    )


def test_volume_par_from_yaml_matches_from_file():
    f = VolumePar.from_file(f"{PAR}/criteria.par")
    y = VolumePar.from_yaml(YAML)
    assert np.allclose(y.X_lay, f.X_lay)
    assert np.allclose(y.Zmin_lay, f.Zmin_lay)
    assert np.allclose(y.Zmax_lay, f.Zmax_lay)
    assert (y.cn, y.cnx, y.cny, y.csumg, y.corrmin, y.eps0) == (
        f.cn, f.cnx, f.cny, f.csumg, f.corrmin, f.eps0,
    )


def test_sequence_par_from_yaml_matches_from_file():
    f = SequencePar.from_file(f"{PAR}/sequence.par", 4)
    y = SequencePar.from_yaml(YAML)
    assert (y.first, y.last) == (f.first, f.last)


def test_track_par_from_yaml_matches_from_file():
    f = TrackPar.from_file(f"{PAR}/track.par")
    y = TrackPar.from_yaml(YAML)
    assert (y.dvxmin, y.dvxmax, y.dvymin, y.dvymax, y.dvzmin, y.dvzmax) == (
        f.dvxmin, f.dvxmax, f.dvymin, f.dvymax, f.dvzmin, f.dvzmax,
    )
    assert (y.dacc, y.dangle, y.add) == (f.dacc, f.dangle, f.add)

import os
import numpy as np
import pytest
from algorithms.parameters import (
    SequencePar, TrackPar, VolumePar, ControlPar, MmNp, TargetPar
)

TEST_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'test_data')


def test_read_write_compare_targ_rec_par(tmp_path):
    filename_read = os.path.join(TEST_DATA, "parameters", "targ_rec_all_different_fields.par")
    filename_write = os.path.join(str(tmp_path), "targ_out_read.par")

    targ_correct = TargetPar(
        gvthres=[1, 2, 3, 4],
        discont=5,
        nnmin=6, nnmax=7,
        nxmin=8, nxmax=9,
        nymin=10, nymax=11,
        sumg_min=12,
        cr_sz=13
    )
    targ_read = TargetPar.from_file(filename_read)
    assert compare_target_par(targ_read, targ_correct)

    targ_read.to_file(filename_write)
    targ_written = TargetPar.from_file(filename_write)
    assert compare_target_par(targ_written, targ_correct)


def test_read_compare_mm_np_par():
    mm1 = MmNp(nlay=2, n1=3.1, n2=[3.8, 3.8, 3.8], d=[3.7, 0.0, 0.0], n3=3.6)
    mm2 = MmNp(nlay=3, n1=3.2, n2=[3.3, 3.3, 3.3], d=[3.4, 0.0, 0.0], n3=3.5)
    assert compare_mmnp(mm1, mm2) is False


def test_read_compare_sequence_par():
    test_file_path = os.path.join(TEST_DATA, "parameters", "sequence.par")
    num_cams = 4
    seqp = SequencePar.from_file(test_file_path, num_cams)
    seqp2 = SequencePar(
        num_cams=num_cams,
        img_base_name=[f"dumbbell/cam{i+1}_Scene77_4085" for i in range(num_cams)],
        first=1000, last=2000,
    )
    for cam in range(num_cams):
        fname = f"dumbbell/cam{cam + 1}_Scene77_"
        assert seqp.img_base_name[cam].startswith(fname)
    assert compare_sequence_par(seqp, seqp2)
    seqp2.first = -999
    assert not compare_sequence_par(seqp, seqp2)


def test_read_track_par():
    tpar = TrackPar.from_file(os.path.join(TEST_DATA, "parameters", "track.par"))
    assert np.isclose(tpar.dvxmin, 0.4)
    assert np.isclose(tpar.dvxmax, 120.0)
    assert np.isclose(tpar.dvymin, 2.0)
    assert np.isclose(tpar.dvymax, -2.0)
    assert np.isclose(tpar.dvzmin, 2.0)
    assert np.isclose(tpar.dvzmax, -2.0)
    assert np.isclose(tpar.dangle, 2.0)
    assert np.isclose(tpar.dacc, -2.0)
    assert tpar.add == 1
    assert tpar.dsumg == 0
    assert tpar.dn == 0
    assert tpar.dnx == 0
    assert tpar.dny == 0


def test_read_volume_par():
    vpar = VolumePar.from_file(os.path.join(TEST_DATA, "parameters", "criteria.par"))
    vpar_correct = VolumePar(
        X_lay=(-250.0, 250.0),
        Zmin_lay=(-100.0, -100.0),
        Zmax_lay=(100.0, 100.0),
        cnx=0.01, cny=0.3, cn=0.3, csumg=0.01, corrmin=1.0, eps0=33.0
    )
    assert compare_volume_par(vpar, vpar_correct)


def test_read_control_par():
    cpar = ControlPar.from_file(os.path.join(TEST_DATA, "parameters", "ptv.par"))
    cpar_correct = ControlPar(
        num_cams=4,
        img_base_name=[f"dumbbell/cam{i+1}_Scene77_4085" for i in range(4)],
        cal_img_base_name=[f"cal/cam{i+1}.tif" for i in range(4)],
        hp_flag=1,
        allCam_flag=0,
        tiff_flag=1,
        imx=1280,
        imy=1024,
        pix_x=0.017,
        pix_y=0.017,
        chfield=0,
        mm=MmNp(nlay=1, n1=1.0, n2=[1.49, 1.0, 1.0], d=[5.0, 0.0, 0.0], n3=1.33)
    )
    assert compare_control_par(cpar, cpar_correct)


def compare_target_par(a, b):
    return (
        np.all(a.gvthres == b.gvthres)
        and a.discont == b.discont
        and a.nnmin == b.nnmin and a.nnmax == b.nnmax
        and a.nxmin == b.nxmin and a.nxmax == b.nxmax
        and a.nymin == b.nymin and a.nymax == b.nymax
        and a.sumg_min == b.sumg_min
        and a.cr_sz == b.cr_sz
    )


def compare_mmnp(a, b):
    return (
        a.nlay == b.nlay and np.allclose(a.n2, b.n2) and np.allclose(a.d, b.d)
        and np.isclose(a.n1, b.n1) and np.isclose(a.n3, b.n3)
    )


def compare_sequence_par(a, b):
    return (
        a.num_cams == b.num_cams
        and a.img_base_name == b.img_base_name
        and a.first == b.first and a.last == b.last
    )


def compare_volume_par(a, b):
    return (
        np.allclose(a.X_lay, b.X_lay)
        and np.allclose(a.Zmin_lay, b.Zmin_lay)
        and np.allclose(a.Zmax_lay, b.Zmax_lay)
        and np.isclose(a.cnx, b.cnx)
        and np.isclose(a.cny, b.cny)
        and np.isclose(a.cn, b.cn)
        and np.isclose(a.csumg, b.csumg)
        and np.isclose(a.corrmin, b.corrmin)
        and np.isclose(a.eps0, b.eps0)
    )


def compare_control_par(a, b):
    return (
        a.num_cams == b.num_cams
        and a.img_base_name == b.img_base_name
        and a.cal_img_base_name == b.cal_img_base_name
        and a.hp_flag == b.hp_flag
        and a.allCam_flag == b.allCam_flag
        and a.tiff_flag == b.tiff_flag
        and a.imx == b.imx
        and a.imy == b.imy
        and np.isclose(a.pix_x, b.pix_x)
        and np.isclose(a.pix_y, b.pix_y)
        and a.chfield == b.chfield
        and compare_mmnp(a.mm, b.mm)
    )
